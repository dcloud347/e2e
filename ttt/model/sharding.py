"""模型参数 sharding 规则。

本项目把设备 mesh 拆成两个轴：
- data：数据并行轴，不切参数，只复制不同 batch。
- state：模型/状态并行轴，用来切大矩阵和归一化参数。
"""

from collections.abc import Callable
from typing import Any, TypeVar

import equinox as eqx
import jax
from jax.sharding import PartitionSpec as P
from jaxtyping import PyTree

from ttt.config import Config
from ttt.model.transformer import MetaModel

T = TypeVar("T", bound=PyTree)


def shard_fn[T: PyTree](tree: T, mesh: jax.sharding.Mesh, where_spec_pairs: list[tuple[Callable[[MetaModel], tuple[Any, ...]], P]]) -> T:
    """对指定子树添加 JAX sharding constraint。"""

    for where, spec in where_spec_pairs:
        # `where` 找到模型里的某些参数，`spec` 描述这些参数如何映射到 mesh 轴。
        tree = eqx.tree_at(where, tree, replace_fn=lambda x: jax.lax.with_sharding_constraint(x, jax.NamedSharding(mesh, spec)), is_leaf=lambda x: x is None)
    return tree


class ModelSharding:
    """集中管理模型参数的切分方式。"""

    def __init__(self, cfg: Config, mesh: jax.sharding.Mesh | None = None):
        self.config = cfg
        self.mesh = mesh

        if self.mesh is None:
            global_dev_num = jax.device_count()
            if cfg.training.n_data_parallel is None:
                # 默认尽量把剩余设备都用于 data parallel。
                assert global_dev_num % cfg.training.n_state_parallel == 0, "Number of devices must be divisible by state parallelism"
                n_data_parallel = global_dev_num // cfg.training.n_state_parallel
            else:
                n_data_parallel = cfg.training.n_data_parallel

            assert n_data_parallel * cfg.training.n_state_parallel == global_dev_num, (
                f"Data parallelism ({cfg.training.n_data_parallel}) and state parallelism ({cfg.training.n_state_parallel}) must match the number of devices ({global_dev_num})"
            )

            self.mesh = jax.make_mesh(axis_shapes=(n_data_parallel, cfg.training.n_state_parallel), axis_names=("data", "state"))

    def shard_params(self, model_params: MetaModel) -> MetaModel:
        """给模型中不同形状的参数指定 sharding。

        线性层权重按输入或输出维度切到 `state` 轴；embedding、norm 和 lm_head 按各自维度切。
        """

        shard_cfg = [
            # 末尾 RMSNorm 是一维参数，直接沿 state 轴切分。
            (lambda m: (m.language_model.model.ln_f,), P("state")),
            (
                # embedding、block norm、lm_head 等二维或一维参数。
                lambda m: (
                    m.language_model.model.wte,
                    m.language_model.model.h.blocks.seq_norm,
                    m.language_model.model.h.blocks.ffn_norm,
                    m.language_model.lm_head,
                ),
                P(None, "state"),
            ),
            (
                # wq/wk/wv 和 MLP 的扩张矩阵按输出维切分。
                lambda m: (
                    m.language_model.model.h.blocks.seq_modeling_block.wq,
                    m.language_model.model.h.blocks.seq_modeling_block.wk,
                    m.language_model.model.h.blocks.seq_modeling_block.wv,
                    m.language_model.model.h.blocks.feed_forward.w1,
                    m.language_model.model.h.blocks.feed_forward.w3,
                ),
                P(None, "state", None),
            ),
            (
                # wo 和 MLP 回投影矩阵按输入维切分。
                lambda m: (
                    m.language_model.model.h.blocks.feed_forward.w2,
                    m.language_model.model.h.blocks.seq_modeling_block.wo,
                ),
                P(None, None, "state"),
            ),
        ]

        if self.config.model.prime:
            # prime 参数只在 E2E TTT 配置中存在，需要和普通 suffix FFN 使用一致的切分方式。
            shard_cfg.extend(
                [
                    (
                        lambda m: (m.language_model.model.h.prime_storage.ffn_prime_norm,),
                        P(None, "state"),
                    ),
                    (
                        lambda m: (
                            m.language_model.model.h.prime_storage.feed_forward_prime.w1,
                            m.language_model.model.h.prime_storage.feed_forward_prime.w3,
                        ),
                        P(None, "state", None),
                    ),
                    (
                        lambda m: (m.language_model.model.h.prime_storage.feed_forward_prime.w2,),
                        P(None, None, "state"),
                    ),
                ]
            )

        return shard_fn(model_params, self.mesh, shard_cfg)
