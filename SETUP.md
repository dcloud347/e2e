# 环境配置指南

## 系统依赖

在安装 Python 环境前，确保以下 GPU 库已就位：

| 组件 | 版本 |
|---|---|
| CUDA Toolkit | 12.8.1 |
| cuDNN | 9.8.0 |
| NCCL | 2.26.2（for CUDA 12.8）|

---

## 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. 安装 Python 依赖

```bash
cd /path/to/e2e
uv sync --exact
```

按 `uv.lock` 安装锁定版本的 Python 3.12 环境。

---

## 3. 下载数据集

数据集托管在 Google Cloud Storage，需要 `gcloud` CLI：

```bash
gcloud storage cp -r gs://llama3-dclm-filter-8k/ llama3-dclm-filter-8k
gcloud storage cp -r gs://llama3-books3/ llama3-books3
```

> **注意（Requester Pays）**：这些 bucket 可能开启了费用由请求方承担。如果遇到权限或计费报错，参考 [Google Cloud 文档](https://cloud.google.com/storage/docs/requester-pays)。

---

## 4. 配置本地路径

编辑 `configs/deploy/interactive.yaml`，填入数据集和 checkpoint 的本地路径：

```yaml
deploy_paths:
  data:
    books3: /your/path/to/llama3-books3
    dclm_filter_8k: /your/path/to/llama3-dclm-filter-8k
  checkpoint: /your/path/to/checkpoints
```

多节点 Slurm 作业则编辑 `configs/deploy/submitit.yaml`。

---

## 5. 配置 Weights & Biases

每次启动训练时通过命令行传入（不要写入配置文件）：

```
training.wandb_entity=<你的 entity>
training.wandb_project=<你的 project>
training.wandb_key=<你的 API key>
```

---

## 6. 验证安装

用 `dummy_dataset` 跑 5 步冒烟测试，不需要真实数据：

```bash
uv run --exact train \
  +deploy=interactive \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  training.dummy_dataset=true \
  training.total_steps=5 \
  training.wandb_entity=MY_ENTITY \
  training.wandb_project=MY_PROJECT \
  training.wandb_key=MY_KEY
```

无报错即表示环境配置正常。

---

## 7. 正式启动训练

**单机交互节点：**

```bash
uv run --exact train \
  +deploy=interactive \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  training.wandb_entity=MY_ENTITY \
  training.wandb_project=MY_PROJECT \
  training.wandb_key=MY_KEY
```

**多节点 Slurm（4 节点示例）：**

```bash
uv run --exact train \
  +deploy=submitit \
  hydra.launcher.nodes=4 \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  training.wandb_entity=MY_ENTITY \
  training.wandb_project=MY_PROJECT \
  training.wandb_key=MY_KEY
```

Slurm 的 partition、account、GPUs per node 等参数在 `configs/deploy/submitit.yaml` 中配置。