# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A JAX implementation of **End-to-End Test-Time Training (TTT)** for long-context language modeling. The core idea: treat long-context LM as continual learning. At test time, the model keeps learning via next-token prediction, compressing the context into its weights. At training time, meta-learning improves the initialization for this inner adaptation. See the [paper](https://test-time-training.github.io/e2e.pdf).

## Commands

```bash
# Install (locked Python 3.12 environment)
uv sync --exact

# Launch training on an interactive node
uv run --exact train \
  +deploy=interactive \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  training.wandb_entity=MY_ENTITY \
  training.wandb_project=MY_PROJECT \
  training.wandb_key=MY_KEY

# Launch multi-node training via Slurm/Submitit
uv run --exact train \
  +deploy=submitit \
  hydra.launcher.nodes=4 \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  training.wandb_entity=MY_ENTITY \
  training.wandb_project=MY_PROJECT \
  training.wandb_key=MY_KEY

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Tests (none exist yet; add under tests/)
uv run pytest
```

**Ruff settings:** 160-char line length, rules E4/E7/E9/F/UP/I/ARG.

## Configuration System

[Hydra](https://hydra.cc/) manages all config. The hierarchy is:

```
configs/
  config.yaml                 # root defaults list
  backend/gpu.yaml            # JAX device settings
  model/{125m,350m,...}.yaml  # architecture sizes
  training/{size}/{task}.yaml # dataset, steps, optimizer hyperparams
  deploy/interactive.yaml     # local paths (fill in dataset/checkpoint dirs here)
  deploy/submitit.yaml        # Slurm settings
  experiment/                 # full experiment presets (combine model + training)
```

`configs/config.yaml` wires together `backend`, `model`, `training`, `checkpoint`, and `deploy_paths` via Hydra defaults. Experiment configs under `configs/experiment/` are added with `+experiment=<path>`. Dataset paths and checkpoint dirs are injected through `deploy_paths` in the deploy config — **do not hardcode paths in code**.

Config classes in `ttt/config.py` are typed dataclasses registered with `ConfigStore`. The top-level `Config` object has `.training`, `.model`, `.backend`, `.checkpoint`, `.deploy_paths`.

## Code Architecture

### Training entry (`ttt/train.py`)

Orchestrates everything: parses config → initializes JAX distributed → creates mesh → builds sharded model + optimizer → loops over batches → checkpoints. The per-step math lives in `ttt/model/loop.py`. The key call is:

```python
model, opt_state, loss, metrics = train_on_sequence(state, model, opt_state, batch, cfg)
```

### Two training modes (`training.train_mode`)

- **`pretrain`**: Standard next-token prediction, no inner loop. Uses `Attention` or `SWAFull`.
- **`meta`**: E2E TTT. The model splits into `prefix_blocks` (reads full context once) and `suffix_blocks` (runs per-chunk with an inner SGD update). Only `suffix_blocks`'s prime FFN parameters are updated by the inner loop.

### Model hierarchy (`ttt/model/transformer.py`)

```
MetaModel
  └── CausalLM
        └── TransformerModel
              ├── wte (token embedding)
              ├── BlockCollectionSplit   ← E2E TTT: splits into prefix/suffix
              │     ├── prefix_blocks: BlockCollection
              │     └── suffix_blocks: BlockCollection (with PrimeStorage if prime=True)
              └── ln_f (final RMSNorm)
```

`MetaModel.loss_for_sequence()` is the main forward entry. In `meta` mode it:
1. Runs `prefix_call()` once over the full sequence
2. Slices the sequence into `mini_batch_size` chunks
3. Runs `inner_loop_step()` per chunk (forward → backward → SGD update of prime FFN only)

### Attention variants (`ttt/model/attention.py`)

| Class | Use case |
|---|---|
| `Attention` | Standard causal full-context attention (`pretrain` baseline) |
| `SWAFull` | Local-window attention over full chunk (cuDNN path) |
| `SWA` | Manual KV-cache sliding-window attention — the correct choice for long-context chunk-by-chunk TTT |

`SWA` maintains `kv_cache_index` and `chunk_index` in `eqx.nn.State`. The prefix path calls `full_sw_attention()` without updating cache.

### Outer vs. inner optimizer

- **Outer loop** (AdamW by default): updates `trainable_parameters()` — parameters matching `training.spec_outer` (default `**` = all).
- **Inner loop** (SGD by default): updates `inner_parameters()` — parameters matching `training.spec_inner`. In E2E TTT configs this is `language_model.**.suffix_blocks.feed_forward_prime.**`.

`spec_outer`/`spec_inner` use glob syntax: `.` for hierarchy, `*` for one level, `**` for any depth. Logic is in `ttt/utils/filter_utils.py`.

### Sharding (`ttt/model/sharding.py`)

Device mesh has two axes: `data` (data parallelism) and `state` (model/tensor parallelism). Controlled by `training.n_data_parallel` and `training.n_state_parallel`. `train_on_sequence` is `filter_vmap`'d over the `data_parallel` axis (acts like pmap); gradients are reduced via `pmean`.

### Checkpointing (`ttt/infra/checkpoint.py`)

Uses [Orbax](https://orbax.readthedocs.io/). Saves model weights, optimizer state, and data iterator state so training can resume exactly. Set `training.load_part=params` to load only weights (e.g. from released checkpoints). Checkpoint directory resolves to `${deploy_paths.checkpoint}/${training.exp_folder}/${training.exp_name}`.

### Typical E2E TTT config pattern

```yaml
seq_modeling_block: SWA
prime: True
suffix_len: 3
spec_inner: ["language_model.**.suffix_blocks.feed_forward_prime.**"]
train_mode: meta
```