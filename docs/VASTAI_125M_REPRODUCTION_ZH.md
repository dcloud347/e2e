# 在 Vast.ai 的 H100 上复刻 125M TTT-E2E 预训练实验

本文给出一套从租用 Vast.ai 实例、配置环境、下载数据、冒烟测试，到正式训练、断点恢复和回收实例的完整流程。目标实验是仓库中的：

```text
configs/experiment/125m/pretrain/pretrain-125m-e2e.yaml
```

也就是论文的 **125M TTT-E2E、DCLM、8K context、1× Chinchilla** 预训练实验。

> 本文的命令以 Vast.ai 的 Ubuntu/Docker SSH 实例为例。Vast.ai 的报价、界面和可用机器会变化；租用前请以控制台实际显示为准。

## 1. 先确认实验规模

仓库中的原始配置为：

| 项目 | 原始值 |
|---|---:|
| 模型层数 / hidden size | 12 / 768 |
| 上下文长度 | 8,192 tokens |
| 全局 batch size | 64 |
| 训练步数 | 4,800 |
| 训练 token 总数 | 2,516,582,400（约 2.52B） |
| 外循环优化器 | AdamW |
| 外循环峰值学习率 | 3e-3 |
| warmup | 480 steps |
| TTT 内循环优化器 | SGD |
| 内循环初始学习率 | 0.1 |
| 数据集 | `dclm_filter_8k` |
| 精度 | BF16 计算、FP32 参数/状态 |

计算方式：

```text
4800 × 64 × 8192 = 2,516,582,400 tokens
```

仓库的 `configs/deploy/interactive.yaml` 默认声明 **单机 8 GPU**。因此，最接近原始配置的 Vast.ai 方案是租一台 **8×H100 80GB** 的整机实例。不要把“租 H100”默认理解成只租一张卡：单卡可以用于安装验证和小规模冒烟测试，但不能在未验证显存、吞吐和数值等价性的情况下称为严格复刻。

### 推荐的两阶段做法

1. 先租一张 H100，使用 dummy dataset 跑 1 step，验证 CUDA、JAX、依赖和代码。
2. 再租同机 8×H100 80GB，下载真实数据并按原配置完整训练。

这样可以避免在昂贵的 8 卡实例上排查基础安装问题。

## 2. Vast.ai 选机清单

在 Vast.ai 控制台选择模板后搜索 offer。正式复刻建议筛选：

- GPU：H100 SXM 80GB 优先；选择 **8 GPU/instance**。
- 可靠性：优先较高 reliability、已验证机器和稳定网络。
- GPU 互联：优先同机 SXM/NVLink/NVSwitch；不要用 8 台互不相连的单卡实例代替单机 8 卡。
- 磁盘：容器磁盘创建后不能扩容。数据集实际大小应先核实；若无法预先核实，建议从 **400–500GB** 起，并为 uv 缓存、日志和 checkpoint 留余量。
- 网络：下载 GCS 数据需要足够的带宽；留意 Vast.ai offer 显示的 internet download/upload 指标和流量价格。
- 租用类型：第一次完整复刻优先按需（on-demand）。可中断实例只有在 checkpoint 已验证、且你接受重编译和恢复成本时再用。
- 镜像：选择带 SSH 的 NVIDIA CUDA development 镜像，系统 CUDA 与仓库推荐的 CUDA 12.8、cuDNN 9.8 尽量接近。不要只看驱动版本，要进入实例后实际验证。

建议在实例创建时：

- Launch mode 选择 SSH，能用 direct SSH 就优先 direct SSH；
- 设置有辨识度的 label，例如 `ttt-e2e-125m-8xh100`；
- 不把 W&B key、GCP 凭据写入公开模板或 on-start script；
- 重要 checkpoint 使用外部对象存储备份。Vast.ai 的普通 container storage 会随实例 Destroy 永久删除；Volume 虽能在销毁实例后保留，但仍绑定创建它的物理主机。

Vast.ai 官方说明：[创建实例](https://docs.vast.ai/api-reference/creating-instances-with-api)、[存储类型](https://docs.vast.ai/guides/instances/storage/types)、[管理实例](https://docs.vast.ai/guides/instances/manage-instances)。

## 3. 登录并检查机器

从 Vast.ai Instances 页面复制 SSH 命令，例如：

```bash
ssh -p <SSH_PORT> root@<SSH_HOST>
```

进入后先执行：

```bash
nvidia-smi
nvidia-smi topo -m
df -h
free -h
python3 --version
```

正式 8 卡机器应能看到 8 张 H100。拓扑命令用于检查 GPU 间连接；若卡数不符、GPU 有异常进程、磁盘明显不足或网络与 offer 描述严重不符，应在开始大规模下载和训练前处理。

为避免 SSH 断线终止训练，安装并启动 `tmux`：

```bash
apt-get update
apt-get install -y git curl tmux ca-certificates build-essential
tmux new -s ttt125m
```

以后重连使用：

```bash
tmux attach -t ttt125m
```

## 4. 获取代码和安装锁定环境

以下路径只是建议；本文后续统一使用 `/workspace`：

```bash
mkdir -p /workspace
cd /workspace
git clone https://github.com/dcloud347/e2e.git
cd e2e
git rev-parse HEAD
```

把输出的 commit SHA 记录到实验日志中。真正可复现的实验必须固定 commit，而不是长期依赖会变化的默认分支。

安装 uv，然后严格按锁文件安装 Python 3.12 环境：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
uv sync --exact
```

验证 JAX 能发现全部 GPU：

```bash
uv run --exact python -c 'import jax; print(jax.__version__); print(jax.devices())'
```

正式 8 卡实例应列出 8 个 CUDA device。再做一个简单计算：

```bash
uv run --exact python -c 'import jax.numpy as jnp; print((jnp.ones((1024, 1024)) @ jnp.ones((1024, 1))).block_until_ready().shape)'
```

### 常见环境问题

- `jax.devices()` 只有 CPU：镜像没有正确暴露 NVIDIA GPU、驱动不兼容，或安装到了 CPU-only JAX。
- 报 CUDA/cuDNN 动态库错误：优先换成与 CUDA 12 系 JAX wheel 相容的 CUDA development 模板，不要在正式实例上盲目混装多个 CUDA 版本。
- 只发现部分 GPU：检查 `CUDA_VISIBLE_DEVICES`、Vast.ai 实际租用卡数，以及容器 GPU 映射。
- `uv sync --exact` 下载失败：先检查 DNS、磁盘空间和实例出网，不要直接删除 `uv.lock` 或改成未锁定安装。

## 5. 下载 DCLM 8K 数据集

实验读取已经用 Llama-3 tokenizer 处理好的 Zarr 数据，不能拿普通文本文件直接替代。官方 bucket 是：

```text
gs://llama3-dclm-filter-8k/
```

### 5.1 安装 Google Cloud CLI

优先按照 Google 官方当前的 Ubuntu 安装说明安装 `gcloud`。如果所选镜像已包含它，直接验证：

```bash
gcloud version
```

### 5.2 Requester Pays 配置

该 bucket 可能启用了 Requester Pays。你需要一个已启用 Billing 的 GCP project，并完成登录：

```bash
gcloud auth login --no-launch-browser
gcloud config set project <YOUR_GCP_PROJECT_ID>
gcloud auth list
```

下载到容器磁盘：

```bash
mkdir -p /workspace/data
gcloud storage cp --recursive \
  --billing-project=<YOUR_GCP_PROJECT_ID> \
  gs://llama3-dclm-filter-8k/ \
  /workspace/data/llama3-dclm-filter-8k
```

若当前 `gcloud storage cp` 版本不接受上述 flag，请以 `gcloud storage cp --help` 和 Google Cloud Requester Pays 文档为准；关键是让请求携带一个启用了 Billing 的 requester project。

下载后检查：

```bash
du -sh /workspace/data/llama3-dclm-filter-8k
find /workspace/data/llama3-dclm-filter-8k -maxdepth 2 -type f | head
df -h /workspace
```

如果 bucket 复制后多套了一层同名目录，用 `find` 确认 Zarr 根目录，再把那个实际目录传给 `deploy_paths.data.dclm_filter_8k`。

## 6. 准备 W&B 和输出目录

创建 checkpoint、日志目录：

```bash
mkdir -p /workspace/checkpoints /workspace/experiments /workspace/logs
cd /workspace/e2e
```

通过当前 shell 注入 W&B 信息，不要写入仓库：

```bash
export WANDB_ENTITY='<YOUR_WANDB_ENTITY>'
export WANDB_PROJECT='ttt-e2e-125m'
read -rsp 'W&B API key: ' WANDB_API_KEY
export WANDB_API_KEY
echo
```

本文的启动命令仍显式将这些环境变量传给 Hydra，因为本仓库的三个 `training.wandb_*` 字段是必填项。不要把实际 API key 贴进 shell history、Markdown、Git commit 或公开日志。

## 7. 先做配置展开和冒烟测试

### 7.1 检查最终 Hydra 配置

先只打印配置，不启动训练：

```bash
uv run --exact train --cfg job --resolve \
  +deploy=interactive \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  deploy_paths.data.dclm_filter_8k=/workspace/data/llama3-dclm-filter-8k \
  deploy_paths.checkpoint=/workspace/checkpoints \
  training.exp_dir=/workspace/experiments \
  training.wandb_entity="$WANDB_ENTITY" \
  training.wandb_project="$WANDB_PROJECT" \
  training.wandb_key="$WANDB_API_KEY"
```

核对至少以下字段：

```text
training.total_steps: 4800
training.seq_length: 8192
training.global_batch_size: 64
training.train_mode: meta
model.prime: true
model.suffix_len: 3
backend.num_devices: 8
```

### 7.2 单卡安装冒烟测试

如果此时租的是单卡 H100，必须覆盖部署配置中默认的设备数，并缩小 batch。dummy dataset 不验证真实 Zarr 数据，但能较早发现编译、显存和训练循环问题：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --exact train \
  +deploy=interactive \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  backend.num_devices=1 \
  training.n_data_parallel=1 \
  training.n_state_parallel=1 \
  training.dummy_dataset=true \
  training.global_batch_size=1 \
  training.total_steps=1 \
  training.save_milestone_freq=0 \
  deploy_paths.data.dclm_filter_8k=/workspace/data/llama3-dclm-filter-8k \
  deploy_paths.checkpoint=/workspace/checkpoints \
  training.exp_dir=/workspace/experiments \
  training.exp_name=smoke-125m-e2e-1gpu \
  training.wandb_entity="$WANDB_ENTITY" \
  training.wandb_project="$WANDB_PROJECT" \
  training.wandb_key="$WANDB_API_KEY" \
  2>&1 | tee /workspace/logs/smoke-1gpu.log
```

E2E meta-training 的单步编译可能需要较长时间；JAX 编译期间 GPU 利用率低并不自动表示卡死。观察系统内存、GPU 显存和日志是否仍有进展。

### 7.3 8 卡真实数据短测

在正式 8 卡实例上，用真实数据跑 2 step。给它单独的实验名，避免与正式 W&B run/checkpoint 混在一起：

```bash
uv run --exact train \
  +deploy=interactive \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  training.total_steps=2 \
  training.save_milestone_freq=0 \
  deploy_paths.data.dclm_filter_8k=/workspace/data/llama3-dclm-filter-8k \
  deploy_paths.checkpoint=/workspace/checkpoints \
  training.exp_dir=/workspace/experiments \
  training.exp_name=smoke-125m-e2e-8gpu-realdata \
  training.wandb_entity="$WANDB_ENTITY" \
  training.wandb_project="$WANDB_PROJECT" \
  training.wandb_key="$WANDB_API_KEY" \
  2>&1 | tee /workspace/logs/smoke-8gpu-realdata.log
```

确认能读取真实 batch、完成编译、loss 为有限值、两步均结束，且 W&B 页面能看到对应 run。

## 8. 正式启动 8×H100 实验

先记录环境快照：

```bash
cd /workspace/e2e
git rev-parse HEAD | tee /workspace/logs/git-commit.txt
nvidia-smi | tee /workspace/logs/nvidia-smi-before-train.txt
uv pip freeze | tee /workspace/logs/python-packages.txt
```

然后在 `tmux` 中运行原始配置。以下命令只覆盖机器路径和凭据，不改变论文实验的模型、batch、学习率、训练步数或随机种子：

```bash
cd /workspace/e2e

uv run --exact train \
  +deploy=interactive \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  deploy_paths.data.dclm_filter_8k=/workspace/data/llama3-dclm-filter-8k \
  deploy_paths.checkpoint=/workspace/checkpoints \
  training.exp_dir=/workspace/experiments \
  training.wandb_entity="$WANDB_ENTITY" \
  training.wandb_project="$WANDB_PROJECT" \
  training.wandb_key="$WANDB_API_KEY" \
  2>&1 | tee /workspace/logs/pretrain-125m-e2e.log
```

正式运行时另开一个 SSH/tmux window 监控：

```bash
watch -n 2 nvidia-smi
```

以及：

```bash
tail -f /workspace/logs/pretrain-125m-e2e.log
```

不要根据别人机器的速度硬填预算。完成短测后，用稳定阶段的平均 step time 估算：

```text
预计剩余小时 = 剩余 steps × 稳态平均秒/step ÷ 3600
预计 GPU 费用 = 实例每小时总价 × 预计剩余小时
```

另外加入首次编译、评估、checkpoint、数据下载、可能的中断和 Vast.ai 存储/流量费用余量。

## 9. Checkpoint 与断点恢复

默认 checkpoint 目录由以下字段拼接：

```text
<deploy_paths.checkpoint>/<training.exp_folder>/<training.exp_name>
```

本指南的正式实验通常会写入：

```text
/workspace/checkpoints/demo/pretrain-125m-e2e
```

仓库默认 `save_milestone_freq=2500`，并会在最后一步保存。长任务若希望降低 Vast.ai 中断造成的损失，可以把保存频率改小，例如在正式命令中增加：

```text
training.save_milestone_freq=500
```

但这会偏离原始运行配置的 I/O 行为并增加磁盘占用；应先确认 checkpoint 大小和保存耗时。无论频率如何，都应定期把 checkpoint 同步到独立于 Vast.ai 实例的对象存储。

同名 W&B run 与同一 checkpoint 目录都存在时，训练入口会自动找到最新 checkpoint，并恢复模型、优化器和数据迭代器。恢复时使用与正式训练相同的命令和 `training.exp_name`。在重新启动前先确认：

```bash
find /workspace/checkpoints/demo/pretrain-125m-e2e -maxdepth 2 -type d | sort | tail
```

注意：仅仅 Stop Vast.ai 实例不会保证稍后还能重新拿回原 GPU；而 Destroy 普通实例会删除 container storage。不要把 Vast.ai 本地 checkpoint 当作唯一副本。

## 10. 常见故障排查

### CUDA OOM

先保存完整报错和 `nvidia-smi` 输出。不要直接把正式实验的 batch 改小后仍宣称复刻成功。

- 检查是否有其他进程占用显存。
- 确认确实是 8 张卡且 JAX 全部可见。
- 确认没有误把 `n_state_parallel` 或设备可见性改掉。
- 冒烟测试可减小 `training.global_batch_size`；这只用于诊断。
- 若正式原配置在 8×H100 80GB 上仍 OOM，记录 commit、JAX/CUDA 版本和峰值显存，再评估是否需要状态并行或梯度累积。任何这类更改都属于新实验，应单独命名并记录。

### NCCL / 多卡初始化错误

- 用 `nvidia-smi topo -m` 检查拓扑。
- 确认一次进程能看到同机全部 8 卡。
- 检查容器共享内存、主机驱动和 NCCL 版本。
- 先跑本文的 1 step dummy test，再跑真实数据 2 step test。
- 不要把面向 Slurm 多节点的 `+deploy=submitit` 用在普通 Vast.ai 单机实例；此处应使用 `+deploy=interactive`。

### 数据集打不开或找不到 split

- 确认传入的是 Zarr 根目录，而不是外层空目录或普通 token 文件。
- 用 `du` 和 `find` 检查下载是否完整。
- 检查 `training.dataset_name` 最终解析为 `dclm_filter_8k`。
- 检查数据目录权限和剩余磁盘空间。

### W&B 恢复行为与预期不符

本仓库会用 experiment name 查找已有 W&B run。冒烟测试必须使用独立 `training.exp_name`。正式恢复则必须保持原正式名称、checkpoint 路径和 W&B entity/project 一致。

### SSH 断开

训练应在 `tmux` 中运行。普通 SSH 断线后重新登录并执行：

```bash
tmux attach -t ttt125m
```

## 11. 如何判断“复刻完成”

至少保存以下材料：

- Git commit SHA；
- `uv.lock` 与 `uv pip freeze` 输出；
- GPU 型号、数量、驱动版本和拓扑；
- Hydra 展开的完整最终配置；
- W&B run URL、完整 loss/learning-rate 曲线；
- stdout/stderr 日志；
- 最终 checkpoint 及其外部备份位置；
- 总 wall-clock 时间、稳定 step time、Vast.ai 实例单价和总成本；
- 任何偏离原配置的 override。

“程序跑完 4,800 step”只是必要条件。复刻报告还应把曲线和论文/发布 checkpoint 的指标进行比较，并明确硬件与软件版本差异。仓库发布的 125M checkpoint 位于：

```text
gs://ttt-e2e-checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc
```

该 checkpoint bucket 同样可能启用 Requester Pays。仓库说明发布 checkpoint 不包含 optimizer state，因此它适合参数比较或用 `training.load_part=params` 初始化其他实验，不适合冒充你这次训练的完整断点续训状态。

## 12. 结束租用前的清单

1. 等待最后一个 checkpoint 异步写入完成，并确认训练进程正常退出。
2. 将 checkpoint、日志、Hydra 配置和环境快照复制到外部持久存储。
3. 从外部位置抽查文件能列出，最好验证大小或 checksum。
4. 清除 shell 中的临时密钥：

   ```bash
   unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT
   ```

5. 确认不再需要实例本地数据后，在 Vast.ai 控制台 **Destroy** 实例。

Vast.ai 官方说明：Stop 会保留实例数据但仍收取存储费，而且重启时原 GPU 可能已被别人租走；Destroy 会永久删除实例及其普通容器数据。不要在外部备份验证完成前 Destroy。计费细节见 [Vast.ai Billing](https://docs.vast.ai/guides/reference/billing)。

## 13. 最短执行路径

如果环境、数据和 W&B 都已经准备好，正式复刻的核心命令只有：

```bash
cd /workspace/e2e

uv sync --exact

uv run --exact train \
  +deploy=interactive \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  deploy_paths.data.dclm_filter_8k=/workspace/data/llama3-dclm-filter-8k \
  deploy_paths.checkpoint=/workspace/checkpoints \
  training.exp_dir=/workspace/experiments \
  training.wandb_entity="$WANDB_ENTITY" \
  training.wandb_project="$WANDB_PROJECT" \
  training.wandb_key="$WANDB_API_KEY" \
  2>&1 | tee /workspace/logs/pretrain-125m-e2e.log
```

但第一次运行时不要跳过前面的设备检查、Hydra 配置展开、dummy 冒烟测试、真实数据短测和备份设计。
