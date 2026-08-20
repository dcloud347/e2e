"""Runtime helpers for raw text inner-loop evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from hydra import compose, initialize_config_dir
from jax.sharding import PartitionSpec as P
from omegaconf import DictConfig, OmegaConf

from ttt.config import Config, register_configs
from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
from ttt.model.data import Batch
from ttt.model.sharding import ModelSharding
from ttt.model.transformer import MetaModel
from ttt.utils.jax_utils import eval_shape_and_sharding, initialize_distibuted, set_random_seed


@dataclass(frozen=True)
class TextEvalRuntime:
    """Loaded model runtime for text evaluation."""

    cfg: Config | DictConfig
    mesh: jax.sharding.Mesh
    model: MetaModel
    state: eqx.nn.State


def compose_text_eval_config(
    *,
    checkpoint: str,
    experiment: str,
    deploy: str,
    overrides: list[str] | None = None,
) -> Config | DictConfig:
    """Compose the Hydra config needed for raw text eval."""

    register_configs()
    config_dir = str(Path("configs").absolute().resolve())
    hydra_overrides = [
        f"+deploy={deploy}",
        f"+experiment={experiment}",
        "training.eval_mode=true",
        "training.load_part=params",
        f"checkpoint.resume_checkpoint_dir={checkpoint}",
        "deploy_paths.data.dclm_filter_8k=/tmp/unused-dclm-filter-8k",
        "deploy_paths.data.books3=/tmp/unused-books3",
        "deploy_paths.checkpoint=/tmp/ttt-e2e-checkpoints",
        "training.wandb_entity=disabled",
        "training.wandb_project=disabled",
        "training.wandb_key=disabled",
    ]
    if overrides:
        hydra_overrides.extend(overrides)

    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=hydra_overrides)

    cfg.model.seq_len = cfg.training.seq_length
    return cfg


def _prepare_data_parallelism(cfg: Config | DictConfig, global_dev_num: int) -> int:
    if cfg.training.n_data_parallel is None:
        if global_dev_num % cfg.training.n_state_parallel != 0:
            raise ValueError("Number of devices must be divisible by training.n_state_parallel.")
        cfg.training.n_data_parallel = global_dev_num // cfg.training.n_state_parallel

    if cfg.training.n_data_parallel * cfg.training.n_state_parallel != global_dev_num:
        raise ValueError(
            f"Data parallelism ({cfg.training.n_data_parallel}) and state parallelism ({cfg.training.n_state_parallel}) "
            f"must match the number of devices ({global_dev_num})."
        )
    return cfg.training.n_data_parallel


def _create_sharded_model_and_state(cfg: Config | DictConfig, mesh: jax.sharding.Mesh, key: jax.Array) -> tuple[MetaModel, eqx.nn.State]:
    model_sharding = ModelSharding(cfg, mesh)
    model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
    state = jax.device_put(state, jax.NamedSharding(mesh, P()))
    model = model_sharding.shard_params(model)
    return model, state


def load_text_eval_runtime(cfg: Config | DictConfig) -> TextEvalRuntime:
    """Initialize JAX, create the model, and restore pretrained parameters."""

    initialize_distibuted(cfg.backend)
    key = set_random_seed(cfg.training.model_seed)

    global_dev_num = jax.device_count()
    n_data_parallel = _prepare_data_parallelism(cfg, global_dev_num)
    mesh = jax.make_mesh(axis_shapes=(n_data_parallel, cfg.training.n_state_parallel), axis_names=("data", "state"))

    with mesh:
        abstract_model_weights = eval_shape_and_sharding(lambda: _create_sharded_model_and_state(cfg, mesh, key)[0].weights())
        checkpointer = Checkpointer(config=cfg, for_saving=False)
        out_state = checkpointer.load_checkpoint(
            step=cfg.training.resume_step,
            targets={"model_weights": abstract_model_weights},
            restore=cfg.training.load_part,
        )

        model, state = _create_sharded_model_and_state(cfg, mesh, key)
        model = unify_dict_with_eqx_module(out_state["model_weights"], model)[0]
        state = state.set(model.step_index, jnp.array(jnp.iinfo(jnp.int32).max - 100, dtype=jnp.int32))

    return TextEvalRuntime(cfg=cfg, mesh=mesh, model=model, state=state)


def summarize_config(cfg: Config | DictConfig) -> str:
    """Return a compact resolved config summary for logs."""

    keys = {
        "experiment": cfg.training.exp_name,
        "train_mode": cfg.training.train_mode,
        "seq_length": cfg.training.seq_length,
        "mini_batch_size": cfg.model.mini_batch_size,
        "suffix_len": cfg.model.suffix_len,
        "prime": cfg.model.prime,
        "n_data_parallel": cfg.training.n_data_parallel,
        "n_state_parallel": cfg.training.n_state_parallel,
        "checkpoint": cfg.checkpoint.resume_checkpoint_dir,
    }
    return OmegaConf.to_yaml(OmegaConf.create(keys), resolve=True)


@eqx.filter_jit
def eval_text_batch(model: MetaModel, state: eqx.nn.State, batch: Batch):
    """Run the existing inner-loop loss path for one token window."""

    return model.loss_for_sequence(batch, state)

