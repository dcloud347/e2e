# Repository Guidelines

## Project Structure & Module Organization
This repository contains a JAX implementation of End-to-End Test-Time Training for long-context language modeling. Core Python source lives in `ttt/`: `ttt/train.py` is the training entrypoint, `ttt/model/` contains model and training-loop code, `ttt/dataloader/` contains language-model dataset loading, `ttt/infra/` handles checkpointing and Weights & Biases logging, and `ttt/utils/` holds shared JAX helpers. Hydra configuration is under `configs/`, grouped by `model/`, `training/`, `deploy/`, `backend/`, and paper experiment presets in `configs/experiment/`. The custom Submitit launcher plugin lives in `hydra_plugins/submitit_ttt/`.

## Build, Test, and Development Commands
- `uv sync --exact`: install the locked Python 3.12 environment from `uv.lock`.
- `uv run --exact train +deploy=interactive +experiment=125m/pretrain/pretrain-125m-e2e ...`: launch a local or interactive-node experiment with Hydra overrides.
- `uv run --exact train +deploy=submitit hydra.launcher.nodes=4 +experiment=125m/pretrain/pretrain-125m-e2e ...`: launch a multi-node Slurm job through Submitit.
- `uv run ruff check .`: run lint checks configured in `pyproject.toml`.
- `uv run ruff format .`: format Python files with Ruff.
- `uv run pytest`: run tests when test files are added.

## Coding Style & Naming Conventions
Use Python 3.12 syntax and keep code compatible with JAX transformations. Ruff is the source of truth for linting, with a 160-character line length and import sorting enabled. Prefer typed dataclasses for configuration, as in `ttt/config.py`. Use `snake_case` for functions, variables, and module names; `PascalCase` for classes; and descriptive Hydra config names that match the existing size/task pattern, such as `pretrain-125m-e2e.yaml`.

## Testing Guidelines
There is currently no committed test suite, but `pytest` is a project dependency. Add focused tests under a future `tests/` directory using names like `test_checkpoint.py` or `test_lm_dataset.py`. For JAX-heavy changes, include small CPU-friendly shape, dtype, and sharding checks where possible instead of relying only on full training runs.

## Commit & Pull Request Guidelines
Recent history uses short, imperative or descriptive subjects such as `Fix typo in README.md` and `Release model checkpoints`. Keep commit subjects concise and scoped. Pull requests should describe the behavioral change, list commands run, mention affected experiment/config paths, and include logs or screenshots for training, checkpointing, or WandB-facing changes.

## Security & Configuration Tips
Do not commit W&B keys, dataset paths, checkpoints, or local cluster account details. Pass secrets through Hydra overrides or environment-specific config files, and keep machine-specific paths in `configs/deploy/interactive.yaml` or `configs/deploy/submitit.yaml`.
