# 在 RunPod 的 8×A100 上复刻 125M TTT-E2E 实验

本文是一份可直接执行的 RunPod 部署指南，目标是复刻仓库中的：

```text
configs/experiment/125m/pretrain/pretrain-125m-e2e.yaml
```

即 **125M TTT-E2E、DCLM、8K context、1× Chinchilla** 预训练实验。命令默认在 RunPod GPU Pod 中执行，工作目录统一为 `/workspace`。

> RunPod 的价格、GPU 库存、镜像标签和控制台入口会变化。本文的平台操作依据 2026-08-18 的官方文档；实际部署时以 RunPod 控制台为准。

## 1. 原始实验规模与硬件选择

仓库配置中的关键参数如下：

| 项目 | 原始值 |
|---|---:|
| 模型 | 12 层，hidden size 768 |
| 上下文长度 | 8,192 tokens |
| 全局 batch size | 64 |
| 训练步数 | 4,800 |
| 训练 token 数 | 2,516,582,400（约 2.52B） |
| 外循环优化器 | AdamW，峰值学习率 3e-3 |
| warmup | 480 steps |
| 内循环优化器 | SGD，初始学习率 0.1 |
| 计算 / 参数精度 | BF16 / FP32 |
| 数据 | Llama-3 tokenized DCLM 8K Zarr |

token 数计算：

```text
4800 × 64 × 8192 = 2,516,582,400
```

`configs/deploy/interactive.yaml` 默认声明单机 8 张 GPU，因此当前租用的 **8×A100 单 Pod** 在设备数量上与仓库默认配置一致：

- **8×A100 80GB SXM** 最稳妥；
- 8×A100 40GB 也可以先验证，但不能在实测前保证原始配置不会 OOM；
- SXM、NVLink/NVSwitch 互联优先，PCIe 版本通常会更慢；
- 不用 8 个独立单卡 Pod 拼成“8 卡”；该仓库的 interactive 配置不是跨 Pod 分布式方案；
- 正式复刻选择 On-Demand；Spot 更适合已验证可靠恢复流程后的尝试。

A100 支持本实验使用的 BF16。8 卡数据并行下，全局 batch 64 对应每张卡 8 条 8K 序列；E2E meta-training 还会保留内外循环相关状态，不能只按 125M 参数量估算显存。应先用当前 8 卡 Pod 跑本文的 dummy test 和真实数据 2-step test，再决定是否执行完整 4,800 step。

## 2. 先规划 RunPod 存储

RunPod Pod 有三类相关存储：

| 类型 | 默认位置 | 停止 Pod | 终止 Pod | 适合内容 |
|---|---|---|---|---|
| Container disk | 系统目录 | 会清空/重置 | 删除 | 系统包、临时缓存 |
| Volume disk | `/workspace` | 保留 | 删除 | 与单个 Pod 同生命周期的数据 |
| Network volume | `/workspace` | 独立保留 | 独立保留 | 数据集、checkpoint、跨 Pod 迁移 |

正式长训练推荐 **Network Volume**，因为 Pod 被终止后数据仍然存在，也可以将其挂到新 Pod。对于约 1.5TB 的完整数据集，建议使用 2TB 卷，剩余空间用于代码、uv 缓存、日志和 checkpoint。需要注意：

- Network volume 只在 Secure Cloud Pods 可用；
- 必须在部署 Pod 时选择，创建后不能给现有 Pod 临时挂载；
- 它会替换默认挂在 `/workspace` 的 volume disk；
- 它所在的数据中心会限制可选 GPU；
- 容量可以增加但不能减小；
- 它仍持续计费，也不是长期备份的替代品。

官方说明：[Pod 存储类型](https://docs.runpod.io/pods/storage/types)、[Network Volumes](https://docs.runpod.io/storage/network-volumes)。

### 2TB 存储的成本比较

按 RunPod 官方当前标价估算（实际账单以控制台为准）：

| 2TB 存储 | 月成本估算 | 日成本估算 | 风险 |
|---|---:|---:|---|
| Network volume | 约 $120 | 约 $4.00 | 独立于 Pod，仍需外部备份 |
| Volume disk（Pod 运行） | 约 $200 | 约 $6.67 | 终止 Pod 即删除 |
| Volume disk（Pod 停止） | 约 $400 | 约 $13.33 | 更贵，且原 GPU 可能被租走 |
| Container disk（Pod 运行） | 约 $200 | 约 $6.67 | stop/restart 会清空 |

以上按控制台配置 2,000GB、首个 1,000GB × $0.07、其余 1,000GB × $0.05 粗略计算；RunPod 对“TB”的具体计费边界和最终小数以账单为准。RunPod 按小时计网络卷，因此只使用数天不需要支付整月费用。以 1.5TB 数据集计算，2TB 只剩约 500GB 余量，不建议再缩小，除非已经准确测量数据、checkpoint 和缓存占用。

官方价格页：[RunPod Pods Pricing](https://docs.runpod.io/pods/pricing)。下载后持续检查：

```bash
df -h /workspace
du -sh /workspace/data /workspace/checkpoints 2>/dev/null
```

不要把重要文件只放在 `/root`、`/tmp` 或其他 container disk 路径。

### 为什么不能只下载约 2.52B tokens

虽然本实验实际训练约 2.52B tokens，但当前 `lm_dataset` 会基于完整 Zarr 数据集进行 shuffle/random access。直接截取数据集开头或只复制任意一部分会改变训练样本及其顺序，不再是严格复刻。要在保持相同样本的前提下制作瘦身数据集，需要先精确复现 Grain 在给定 seed 下访问的索引并物化对应序列；仓库目前没有提供这种导出工具。

如果目标只是验证代码而不是复刻论文，可以自建小 Zarr 或使用 `training.dummy_dataset=true`。正式复刻则建议保留完整数据集。

## 3. 创建 Network Volume 和 Pod

### 3.1 选择数据中心

先在 RunPod Pods 的 Deploy 页面查看哪些区域有可用的 8×A100。然后到 Storage 页面：

1. 点击 **New Network Volume**；
2. 选择同一个数据中心；
3. 名称设为 `ttt-e2e-125m`；
4. 设置所需容量；
5. 创建卷。

不要反过来在没有 8×A100 库存的区域先创建卷，否则该卷会限制 Pod 的可部署位置。

### 3.2 部署 Pod

在 Pods → Deploy 中：

1. 选择刚创建的 Network Volume；
2. 选择 A100，并确认 Pod 的 GPU count 是 8；
3. 选择官方 RunPod PyTorch 模板或自建 CUDA development 模板；
4. 镜像应尽量接近仓库推荐的 CUDA 12.8.1、cuDNN 9.8、NCCL 2.26.2；
5. 开启 SSH Terminal Access / 暴露 TCP 22；
6. Container disk 给系统和构建缓存保留足够空间；
7. 选择 Deploy On-Demand。

RunPod 官方自定义模板示例当前提供含 CUDA 12.8.1 的 `runpod/pytorch` 镜像，但标签可能更新，部署前应查看模板详情：[创建自定义 Pod 模板](https://docs.runpod.io/pods/templates/create-custom-template)。

不要把 W&B key 或 GCP 凭据写进公开模板环境变量。

## 4. SSH 登录与机器验收

先在本机把 SSH 公钥添加到 RunPod Account Settings。Pod 启动后打开 Connect，优先复制 **SSH over exposed TCP** 命令，例如：

```bash
ssh root@<PUBLIC_IP> -p <SSH_PORT> -i ~/.ssh/id_ed25519
```

连接方式见 [RunPod Connect to a Pod](https://docs.runpod.io/pods/connect-to-a-pod)。进入后检查：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi topo -m
df -h
df -h /workspace
free -h
python3 --version
```

正式 Pod 应看到 8 张 A100，并明确每张是约 40GB 还是 80GB。若卡数、显存、互联、磁盘或 CPU/RAM 明显不符，应在下载数据前解决。

为避免 SSH 断线影响训练：

```bash
apt-get update
apt-get install -y git curl tmux ca-certificates build-essential
tmux new -s ttt125m
```

重新连接后恢复会话：

```bash
tmux attach -t ttt125m
```

## 5. 获取代码并安装锁定环境

代码、环境缓存和输出都放在网络卷：

```bash
cd /workspace
git clone https://github.com/dcloud347/e2e.git
cd e2e
git rev-parse HEAD
```

记录 commit SHA。安装 uv 并严格使用锁文件：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
export UV_CACHE_DIR=/workspace/.cache/uv
uv --version
uv sync --exact
```

验证 JAX：

```bash
uv run --exact python -c 'import jax; print(jax.__version__); print(jax.devices())'
uv run --exact python -c 'import jax.numpy as jnp; print((jnp.ones((1024,1024)) @ jnp.ones((1024,1))).block_until_ready().shape)'
```

8 卡 Pod 的第一条命令应列出 8 个 CUDA device。常见错误：

- 只有 CPU：模板未暴露 GPU、驱动不兼容或装成 CPU-only JAX；
- 只有部分 GPU：检查 `CUDA_VISIBLE_DEVICES` 和 Pod GPU count；
- CUDA/cuDNN 动态库错误：换兼容的 CUDA 12 development 模板，避免混装多套 CUDA；
- `uv sync --exact` 失败：检查出网、DNS、`/workspace` 空间，不要删除 `uv.lock`。

## 6. 下载 DCLM 8K 数据

实验要求已经用 Llama-3 tokenizer 处理的 Zarr 数据，不能用普通文本替代：

```text
gs://llama3-dclm-filter-8k/
```

确认或安装 Google Cloud CLI，然后登录：

```bash
gcloud version
gcloud auth login --no-launch-browser
gcloud config set project <YOUR_GCP_PROJECT_ID>
gcloud auth list
```

bucket 可能启用 Requester Pays，需要一个已启用 Billing 的 GCP project：

```bash
mkdir -p /workspace/data
gcloud storage cp --recursive \
  --billing-project=<YOUR_GCP_PROJECT_ID> \
  gs://llama3-dclm-filter-8k/ \
  /workspace/data/llama3-dclm-filter-8k
```

若当前 `gcloud` 的 flag 位置或名称不同，查看 `gcloud storage cp --help`；关键是请求携带启用了 Billing 的 requester project。

检查下载结果：

```bash
du -sh /workspace/data/llama3-dclm-filter-8k
find /workspace/data/llama3-dclm-filter-8k -maxdepth 2 -type f | head
df -h /workspace
```

若复制后多了一层同名目录，应将真正的 Zarr 根目录传给 Hydra。

## 7. 准备 W&B 和输出目录

```bash
mkdir -p /workspace/checkpoints /workspace/experiments /workspace/logs
cd /workspace/e2e

export WANDB_ENTITY='<YOUR_WANDB_ENTITY>'
export WANDB_PROJECT='ttt-e2e-125m'
read -rsp 'W&B API key: ' WANDB_API_KEY
export WANDB_API_KEY
echo
```

不要把实际 key 写进仓库、模板或 shell history。

## 8. 展开并核对 Hydra 配置

此命令只打印配置，不启动训练：

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

至少确认：

```text
training.total_steps: 4800
training.seq_length: 8192
training.global_batch_size: 64
training.train_mode: meta
model.prime: true
model.suffix_len: 3
backend.num_devices: 8
```

## 9. 冒烟测试

### 9.1 单卡 A100 安装测试

若先租单卡，覆盖默认 8 卡配置并将 batch 缩为 1：

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

JAX 首次编译可能较久，编译期间 GPU 利用率低不一定是卡死。

### 9.2 8 卡真实数据短测

正式 Pod 上先跑 2 step，并使用独立实验名：

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

确认真实 batch 可读、loss 有限、两步完成且 W&B 能看到 run。

## 10. 正式启动 8×A100 训练

记录环境：

```bash
cd /workspace/e2e
git rev-parse HEAD | tee /workspace/logs/git-commit.txt
nvidia-smi | tee /workspace/logs/nvidia-smi-before-train.txt
uv pip freeze | tee /workspace/logs/python-packages.txt
```

在 `tmux` 中运行。下列 override 只改变机器路径和凭据，不改变论文实验超参数：

```bash
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

另开一个 SSH window：

```bash
watch -n 2 nvidia-smi
```

查看日志：

```bash
tail -f /workspace/logs/pretrain-125m-e2e.log
```

根据短测后的稳态速度估算费用：

```text
剩余小时 = 剩余 steps × 平均秒/step ÷ 3600
预计计算费 = Pod 每小时总价 × 剩余小时
```

另外预留首次编译、评估、checkpoint、下载、Network Volume 和可能中断的成本。

## 11. Checkpoint 和恢复

正式 checkpoint 通常位于：

```text
/workspace/checkpoints/demo/pretrain-125m-e2e
```

默认里程碑保存频率为 2,500 step，并在最后一步保存。若希望减少中断损失，可在正式命令加入：

```text
training.save_milestone_freq=500
```

这会改变原始 I/O 行为并增加占用，应先测量 checkpoint 大小和写入时间。

同名 W&B run 和同一 checkpoint 目录都存在时，训练入口会自动恢复最新 checkpoint，包括模型、优化器和数据迭代器。恢复时重新运行同一正式命令，并保持 `training.exp_name`、W&B entity/project 和 checkpoint 路径一致。

恢复前检查：

```bash
find /workspace/checkpoints/demo/pretrain-125m-e2e -maxdepth 2 -type d | sort | tail
```

Network Volume 提高了 Pod 迁移能力，但仍应定期同步 checkpoint 到 GCS/S3 等独立对象存储。

## 12. 常见问题

### CUDA OOM

- 检查其他进程和 8 张卡是否全部可见；
- 检查设备和 sharding override 是否被误改；
- 缩小 batch 只用于诊断，改变正式 batch 后不能宣称严格复刻；
- 若原配置仍 OOM，记录 commit、CUDA/JAX 版本、拓扑和峰值显存，再单独设计状态并行或梯度累积实验。

### NCCL / 多卡错误

- 检查 `nvidia-smi topo -m`；
- 确认这是单个 8 卡 Pod；
- 检查模板共享内存、驱动和 NCCL；
- RunPod 普通单机 Pod 使用 `+deploy=interactive`，不要使用 Slurm 的 `+deploy=submitit`。

### 数据集错误

- 确认传入真正的 Zarr 根目录；
- 用 `du`、`find` 验证下载完整性；
- 确认最终 `training.dataset_name=dclm_filter_8k`；
- 检查 `/workspace` 空间。

### Pod Stop 后没有 GPU

RunPod 官方说明，Pod 重启时原 GPU 可能已无库存，甚至只能零 GPU 启动以取回数据。Network Volume 可让你终止旧 Pod 后把同一数据挂载到另一台可用机器。见 [Zero GPU Pods on restart](https://docs.runpod.io/pods/troubleshooting/zero-gpus)。

## 13. 复刻验收材料

至少保存：

- Git commit SHA；
- `uv.lock` 和 `uv pip freeze`；
- GPU 型号、数量、驱动和拓扑；
- Hydra 展开的最终配置；
- W&B run URL 和完整曲线；
- stdout/stderr 日志；
- 最终 checkpoint 及外部备份；
- wall-clock 时间、step time、Pod/存储价格和总成本；
- 所有偏离原配置的 override。

发布的参考参数在：

```text
gs://ttt-e2e-checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc
```

该 bucket 也可能启用 Requester Pays。发布包不含 optimizer state，因此适合参数/指标比较或以 `training.load_part=params` 初始化实验，不等于你的完整断点。

## 14. 结束 Pod 前

1. 确认最后一个异步 checkpoint 已写完、训练正常退出；
2. 将 checkpoint、日志和配置同步到独立对象存储；
3. 从外部位置抽查文件大小或 checksum；
4. 清除当前 shell 密钥：

   ```bash
   unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT
   ```

5. 确认 Network Volume 中的数据完整；
6. 在 RunPod 控制台 **Terminate Pod**，停止 GPU 计算费；
7. 不再需要数据时才删除 Network Volume。

普通 volume disk 在 Pod 终止时删除；Network Volume 在 Pod 终止后继续存在并继续计费。挂载 Network Volume 的 Pod 不能 Stop，只能 Terminate，数据保留在卷中。官方说明见 [Manage Pods](https://docs.runpod.io/pods/manage-pods)。

## 15. 正式训练核心命令

```bash
cd /workspace/e2e
export UV_CACHE_DIR=/workspace/.cache/uv
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

第一次运行不要跳过设备验收、配置展开、单卡 dummy test、8 卡真实数据短测和外部备份设计。
