"""模型输入输出的数据结构。

这些结构都是 Equinox Module，可以作为 JAX pytree 在 jit/vmap/scan 中传递。
"""

from __future__ import annotations

import typing as tp
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree

_T = tp.TypeVar("_T", bound=PyTree)


def tree_slice[T: PyTree](tree: _T, i: int) -> _T:
    """对 pytree 的每个 leaf 取同一个下标。"""

    return jax.tree.map(lambda x: x[i], tree)


class BaseModelOutput(eqx.Module):
    """Transformer 层和语言模型前向传播的通用输出。"""

    state: nn.State
    last_hidden_state: jnp.ndarray | None = None
    logits: jnp.ndarray | None = None


class Batch(eqx.Module):
    """一个语言模型 batch。

    `input_ids` 是输入 token，`target_tokens` 是向右错一位的预测目标。
    `loss_masks` 用来屏蔽不需要参与 loss 的位置。
    """

    input_ids: jnp.ndarray
    target_tokens: jnp.ndarray
    loss_masks: jnp.ndarray
    attention_mask: jnp.ndarray | None = None
    position_ids: jnp.ndarray | None = None
    index: int | slice | None = eqx.field(static=True, default=None)

    @property
    def shape(self):
        """返回 input_ids 的 shape，方便调用处把 Batch 当作数组批次看待。"""

        return self.input_ids.shape

    def slice_index(self, index: int | slice) -> Batch:
        """按下标切分 Batch，同时记录这个切片的 index。"""

        return replace(tree_slice(self, index), index=index)
