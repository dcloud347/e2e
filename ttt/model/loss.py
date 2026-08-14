"""语言模型 loss 计算。"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Scalar


@jax.jit
def cross_entropy_loss_and_accuracy(
    logits: Float[Array, " seq_length vocab_size"],
    tokens: Int[Array, " seq_length"],
    valid=None,
) -> tuple[Scalar, Scalar]:
    """计算 masked next-token cross entropy。

    这个函数名里保留了 accuracy，但当前返回的两个值都是 loss：
    第一个用于反传，第二个用于日志记录。
    """

    if valid is None:
        # 没传 mask 时默认所有 token 都参与 loss。
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    # 避免整条序列都被 mask 掉时除以 0。
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32)

    log_prob = jax.nn.log_softmax(logits, axis=-1)
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(log_prob, jnp.expand_dims(tokens, -1), axis=-1),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))

    # 每条序列先按有效 token 求平均，再对 batch 求平均。
    token_wise_loss = -token_log_prob
    loss_pure_ce = jnp.mean(jnp.sum(token_wise_loss, axis=-1) / valid_text_length)
    loss = jnp.mean(jnp.sum(token_wise_loss, axis=-1) / valid_text_length)

    return loss, loss_pure_ce


def token_log_probs(logits, targets) -> jnp.ndarray:
    """返回每个目标 token 的 log probability，用于记录 token-level NLL。"""

    token_log_probs = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(targets, -1),
            axis=-1,
        ),
        -1,
    )
    return token_log_probs
