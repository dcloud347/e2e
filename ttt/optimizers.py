"""优化器构造函数。

外循环通常使用 AdamW 更新模型初始化；E2E TTT 的内循环通常使用 SGD 更新少量 suffix/prime 参数。
这里统一返回 Optax optimizer 和可用于日志的学习率调度信息。
"""

import re

import jax.numpy as jnp
import optax

from ttt.config import AdamWOptimizerConfig, OptimizerConfig, SGDOptimizerConfig
from ttt.utils.filter_utils import get_mask_fn


def make_adamw_optimizer(config: AdamWOptimizerConfig, weight_decay_mask=None):
    """创建 AdamW 优化器。

    学习率采用 warmup + cosine decay。`emb_wd=False` 时会把词嵌入参数排除在 weight decay 之外。
    """

    if config.lr == 0.0:
        # lr 为 0 时直接固定为 0，常用于冻结或调试。
        learning_rate_schedule = optax.constant_schedule(0.0)
    else:
        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.init_lr,
            peak_value=config.lr,
            warmup_steps=config.lr_warmup_steps,
            decay_steps=config.lr_decay_steps,
            end_value=config.end_lr,
        )

    optimizer_info = dict(learning_rate_schedule=learning_rate_schedule)

    if not config.emb_wd:
        # 参数路径里包含 wte 的权重是 token embedding，不做 weight decay。
        exclude_emb = lambda name: False if re.search("wte", name) else True  # no wd on word embedding
        weight_decay_mask = lambda params: get_mask_fn(exclude_emb, params)
    else:
        weight_decay_mask = None

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.clip_gradient),
        optax.adamw(
            learning_rate=learning_rate_schedule,
            weight_decay=config.weight_decay,
            b1=config.b1,
            b2=config.b2,
            mask=weight_decay_mask,
            mu_dtype=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
        ),
    )

    return optimizer, optimizer_info


def make_sgd_optimizer(config: SGDOptimizerConfig, ilr_multiplier: jnp.ndarray = None):
    """创建 SGD 优化器。

    `ilr_multiplier` 用于内循环学习率 warmup，把配置里的 lr 乘上一个动态倍率。
    """

    learning_rate_schedule = optax.constant_schedule(config.lr * ilr_multiplier)
    optimizer_info = dict(learning_rate_schedule=learning_rate_schedule)
    if config.clip_gradient > 0.0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optax.sgd(learning_rate=learning_rate_schedule, momentum=None),
        )
    else:
        optimizer = optax.sgd(learning_rate=learning_rate_schedule, momentum=None)
    return optimizer, optimizer_info


def make_optimizer(optimizer_config: OptimizerConfig, ilr_multiplier: jnp.ndarray = None) -> tuple[optax.GradientTransformation, dict]:
    """根据配置分发到具体优化器构造函数。"""

    if optimizer_config.optimizer_type == "adamw":
        del ilr_multiplier
        optimizer, optimizer_info = make_adamw_optimizer(optimizer_config)
    elif optimizer_config.optimizer_type == "sgd":
        optimizer, optimizer_info = make_sgd_optimizer(optimizer_config, ilr_multiplier)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_config.optimizer_type}")

    return optimizer, optimizer_info
