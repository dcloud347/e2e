# ttt-text-eval 运行指南

本文说明如何用 `ttt-text-eval` 对不同尺寸的 TTT-E2E checkpoint 跑 raw text inner-loop eval。

`ttt-text-eval` 和仓库原来的 `train.py` 不同：

- 输入是自然语言文本或文本文件；
- 文本会先用 Llama-3 tokenizer 转成 token ids；
- token ids 会转成 `Batch(input_ids, target_tokens, loss_masks)`；
- 模型以 `training.train_mode=meta` 运行已有的 inner-loop TTT；
- 输出是 loss / perplexity，不输出生成文本；
- 不需要 Zarr 数据集，不初始化 W&B，不走训练 dataloader。

## 1. 环境准备

安装锁定依赖：

```bash
uv sync --exact
```

如果 tokenizer 从 Hugging Face 下载，先设置访问 token：

```bash
export HF_TOKEN='<YOUR_HF_TOKEN>'
```

Llama-3 tokenizer 可能需要在 Hugging Face 上确认模型访问权限。也可以把 tokenizer 下载到本地目录，然后把 `--tokenizer` 指向本地路径。

准备一个输入文件：

```bash
mkdir -p examples
cat > examples/input.txt <<'EOF'
Test-time training adapts a language model on the context it sees at inference time.
This file is used to run a small raw text evaluation.
EOF
```

## 2. 输出含义

典型输出：

```text
input_tokens: 42
windows: 1
seq_length: 8192
inner_chunks_per_window: 8
window[0]: tokens=42 valid_targets=41 loss=... ppl=...
valid_targets: 41
loss: ...
perplexity: ...
```

字段说明：

| 字段 | 含义 |
|---|---|
| `input_tokens` | tokenizer 后的 token 总数，默认会额外加 BOS |
| `windows` | 被评估的固定长度窗口数 |
| `seq_length` | 每个窗口的模型输入长度 |
| `inner_chunks_per_window` | 每个窗口被切成多少个 inner-loop chunk |
| `valid_targets` | 参与 loss 计算的 target token 数 |
| `loss` | token-weighted average next-token NLL |
| `perplexity` | `exp(loss)` |

短文本默认会 pad 到 `seq_length + 1`，但 padding target 不计入 loss。

## 3. Checkpoint 和实验配置

预训练 DCLM checkpoint：

| 模型 | `--experiment` | `--checkpoint` |
|---|---|---|
| 125M | `125m/pretrain/pretrain-125m-e2e` | `gs://ttt-e2e-checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc` |
| 1B | `1b/pretrain/pretrain-1b-e2e` | `gs://ttt-e2e-checkpoints/1b_ttt_e2e_pretrain_dclm_8k_1x_cc` |
| 3B | `3b/pretrain/pretrain-3b-e2e` | `gs://ttt-e2e-checkpoints/3b_ttt_e2e_pretrain_dclm_8k_3x_cc` |

Books fine-tuned checkpoint：

| 模型 | `--experiment` | `--checkpoint` |
|---|---|---|
| 125M Books 8K | `125m/pretrain/pretrain-125m-e2e` | `gs://ttt-e2e-checkpoints/125m_ttt_e2e_finetune_books_8k_1x_cc` |
| 1B Books 8K | `1b/pretrain/pretrain-1b-e2e` | `gs://ttt-e2e-checkpoints/1b_ttt_e2e_finetune_books_8k_1x_cc` |
| 3B Books 8K | `3b/pretrain/pretrain-3b-e2e` | `gs://ttt-e2e-checkpoints/3b_ttt_e2e_finetune_books_8k_3x_cc` |

注意：

- 发布 checkpoint 不包含 optimizer state；`ttt-text-eval` 内部固定按 params-only 路径加载。
- GCS checkpoint bucket 开启 Requester Pays 时，直接从 `gs://` 恢复可能需要正确的 GCP 凭据和计费项目。
- 如果已经把 checkpoint 下载到本地，可以把 `--checkpoint` 换成本地 Orbax checkpoint 目录。

## 4. 125M eval

125M 是最适合先验证的模型。

单卡 A100/H100：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --exact ttt-text-eval \
  --checkpoint gs://ttt-e2e-checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc \
  --tokenizer meta-llama/Meta-Llama-3-8B \
  --text-file examples/input.txt \
  --experiment 125m/pretrain/pretrain-125m-e2e \
  --max-windows 1 \
  --override backend.num_devices=1 \
  --override training.n_data_parallel=1 \
  --override training.n_state_parallel=1 \
  --print-config
```

资源建议：

| GPU | 判断 |
|---|---|
| 1×A100 40GB | 推荐 |
| 1×H100 80GB | 很稳 |
| 1×24GB GPU | 可尝试，只建议先 `--max-windows 1` |
| 8GB/12GB GPU | 不建议 |

## 5. 1B eval

1B 可以用单张 H100 尝试，但更建议 2 张 H100 做 state parallel。

2×H100：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --exact ttt-text-eval \
  --checkpoint gs://ttt-e2e-checkpoints/1b_ttt_e2e_pretrain_dclm_8k_1x_cc \
  --tokenizer meta-llama/Meta-Llama-3-8B \
  --text-file examples/input.txt \
  --experiment 1b/pretrain/pretrain-1b-e2e \
  --max-windows 1 \
  --override backend.num_devices=2 \
  --override training.n_data_parallel=1 \
  --override training.n_state_parallel=2 \
  --print-config
```

为什么要写 `training.n_state_parallel=2`：

```text
默认 2 张卡会更容易被当成 data parallel，模型复制到每张卡；
state parallel 会把模型参数沿 state 轴切到两张卡上，降低单卡显存压力。
```

单张 H100 冒烟：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --exact ttt-text-eval \
  --checkpoint gs://ttt-e2e-checkpoints/1b_ttt_e2e_pretrain_dclm_8k_1x_cc \
  --tokenizer meta-llama/Meta-Llama-3-8B \
  --text-file examples/input.txt \
  --experiment 1b/pretrain/pretrain-1b-e2e \
  --max-windows 1 \
  --override backend.num_devices=1 \
  --override training.n_data_parallel=1 \
  --override training.n_state_parallel=1 \
  --print-config
```

资源建议：

| GPU | 判断 |
|---|---|
| 2×H100 80GB，state parallel 2 | 推荐 |
| 1×H100 80GB | 可尝试 |
| 1×A100 80GB | 可尝试，但不如 H100 稳 |
| 1×A100 40GB | 不建议作为首选 |

## 6. 3B eval

3B 的 checkpoint 是 8K DCLM，模型显存和 inner-loop gradient 压力都明显更高。建议从 4 张或 8 张 H100 做 state parallel 开始。

4×H100：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --exact ttt-text-eval \
  --checkpoint gs://ttt-e2e-checkpoints/3b_ttt_e2e_pretrain_dclm_8k_3x_cc \
  --tokenizer meta-llama/Meta-Llama-3-8B \
  --text-file examples/input.txt \
  --experiment 3b/pretrain/pretrain-3b-e2e \
  --max-windows 1 \
  --override backend.num_devices=4 \
  --override training.n_data_parallel=1 \
  --override training.n_state_parallel=4 \
  --print-config
```

8×H100：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --exact ttt-text-eval \
  --checkpoint gs://ttt-e2e-checkpoints/3b_ttt_e2e_pretrain_dclm_8k_3x_cc \
  --tokenizer meta-llama/Meta-Llama-3-8B \
  --text-file examples/input.txt \
  --experiment 3b/pretrain/pretrain-3b-e2e \
  --max-windows 1 \
  --override backend.num_devices=8 \
  --override training.n_data_parallel=1 \
  --override training.n_state_parallel=8 \
  --print-config
```

资源建议：

| GPU | 判断 |
|---|---|
| 8×H100 80GB，state parallel 8 | 最稳 |
| 4×H100 80GB，state parallel 4 | 推荐先试 |
| 2×H100 80GB | 不建议作为首选，可能 OOM |
| 单卡 | 不建议 |

## 7. 评估长文本

默认 `seq_length=8192`。如果文本超过一个窗口，可以去掉 `--max-windows 1`，或者指定更多窗口：

```bash
--max-windows 4
```

长文本会按 `seq_length` 前进切窗，每个窗口需要 `seq_length + 1` 个 token 来构造 next-token target。窗口之间会重叠 1 个 token，避免漏掉跨窗口的 next-token transition。

最后一个不足窗口的处理方式由 `--final-window` 控制：

| 参数 | 行为 |
|---|---|
| `--final-window pad` | 默认，padding 但不把 padding target 计入 loss |
| `--final-window drop` | 丢掉不足一个窗口的尾部 |
| `--final-window error` | 尾部不足时直接报错 |

## 8. 短 context 冒烟

如果只想确认环境、tokenizer、checkpoint restore 能跑，可以临时缩短上下文：

```bash
--seq-length 4096
```

这会覆盖 `training.seq_length`，并让每个窗口只有 4096 tokens。它适合诊断 OOM 或编译问题，但不是完整 8K eval。

注意 `seq_length` 必须能被模型的 `mini_batch_size` 整除。E2E preset 默认：

```text
model.mini_batch_size = 1024
```

所以常用值是 `1024`、`2048`、`4096`、`8192`。

## 9. 常见问题

### Tokenizer 权限错误

如果看到 Hugging Face 权限或 401/403 错误：

```bash
export HF_TOKEN='<YOUR_HF_TOKEN>'
```

并确认账号对 `meta-llama/Meta-Llama-3-8B` 有访问权限。也可以使用本地 tokenizer：

```bash
--tokenizer /workspace/tokenizers/Meta-Llama-3-8B
```

### Checkpoint 访问错误

GCS bucket 可能开启 Requester Pays。可以先把 checkpoint 复制到自己的 bucket 或本地磁盘，再传本地路径：

```bash
--checkpoint /workspace/checkpoints/1b_ttt_e2e_pretrain_dclm_8k_1x_cc
```

### OOM

按这个顺序排查：

1. 确认 `CUDA_VISIBLE_DEVICES` 和 `backend.num_devices` 一致；
2. 对 1B/3B 优先使用 `training.n_state_parallel`，不要只做 data parallel；
3. 先加 `--max-windows 1`；
4. 临时用 `--seq-length 4096` 冒烟；
5. 换更多或更大显存 GPU。

### 编译时间长

第一次运行 JAX 会编译，尤其是 inner-loop TTT 路径。GPU 利用率低不一定表示卡死。建议开另一个窗口观察：

```bash
nvidia-smi
```

### 输出不是自然语言

`ttt-text-eval` 只做评估，输出 loss/perplexity。它不会生成文本。生成需要后续单独实现 `ttt-generate`。
