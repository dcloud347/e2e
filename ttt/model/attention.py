"""注意力层实现。

包含普通 causal self-attention、基于 cuDNN local window 的 SWAFull，
以及手动维护 KV cache 的分块 sliding-window attention（SWA）。
"""

from typing import override

import equinox as eqx
import jax
import jax.ad_checkpoint
import jax.numpy as jnp
import jax.random as jrandom
from equinox import nn
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray, PyTree

from ttt.config import ModelConfig
from ttt.model.data import Batch
from ttt.utils.jax_utils import get_float_dtype_by_name, maybe_double_remat, promote_dtype, tree_rearrange


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """预计算 RoPE 需要的复数形式 sin/cos 频率表。"""

    # 每两个 hidden 维度组成一个复数对，因此只取 dim/2 个频率。
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs).astype(dtype)
    sin, cos = jnp.sin(freqs), jnp.cos(freqs)
    freqs_cis = jnp.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(x, freqs_cis: jnp.ndarray):
    """把 RoPE 旋转位置编码应用到 query 或 key 上。"""

    input_dtype = x.dtype
    # freqs_cis 扩展到能和 `[T, heads, head_dim/2]` 这类形状广播。
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:-1], 1, *freqs_cis.shape[-1:]))
    # 把最后一维按两个数一组看成复数，再乘以复数旋转因子。
    reshape_x = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
    x_ = jax.lax.complex(reshape_x[..., 0], reshape_x[..., 1])
    x_out = x_ * freqs_cis
    x_out = jnp.stack((jnp.real(x_out), jnp.imag(x_out)), axis=-1).reshape(*x_out.shape[:-1], -1)
    return x_out.astype(input_dtype)


class NormalLinear(eqx.Module):
    """无 bias 线性层，权重使用正态分布初始化。"""

    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    weight: jax.Array
    name: str = eqx.field(static=True)

    def __init__(self, config: ModelConfig, in_features: int, out_features: int, *, name: str = "", std: float, key: PRNGKeyArray):
        """初始化线性层权重和 dtype 信息。"""

        self.compute_dtype = get_float_dtype_by_name(config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(config.param_dtype)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = jrandom.normal(key, shape=(in_features, out_features), dtype=self.param_dtype) * std
        self.name = name

    @jax.named_scope("ttt.transformer.NormalLinear")
    def __call__(self, x: Array) -> Array:
        """执行 `x @ weight`，并在需要时给 checkpoint 加可读名称。"""

        if self.name:
            x = jax.ad_checkpoint.checkpoint_name(x, f"pre_promote_{self.name}")
        # 参数可能用 fp32 保存，但计算用 bf16/fp16。
        x, weight = promote_dtype(x, self.weight, dtype=self.compute_dtype)
        if self.name:
            x = jax.ad_checkpoint.checkpoint_name(x, f"pre_{self.name}")
        x = x @ weight
        if self.name:
            x = jax.ad_checkpoint.checkpoint_name(x, f"post_{self.name}")
        return x


class AttentionBase(eqx.Module):
    """普通 attention 和 SWA 共享的 QKV 投影、RoPE、输出投影逻辑。"""

    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    wq: NormalLinear
    wk: NormalLinear
    wv: NormalLinear
    wo: NormalLinear
    q_norm: nn.RMSNorm
    k_norm: nn.RMSNorm

    resid_dropout: nn.Dropout = eqx.field(static=True)

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        """初始化 Q/K/V/O 投影和 QK RMSNorm。"""

        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)

        embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = embed_dim // self.num_heads

        self.q_norm = nn.RMSNorm(self.head_dim, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)

        keys = jax.random.split(key, 4)

        self.wq, self.wk, self.wv, self.wo = (
            NormalLinear(
                self.config,
                in_features=embed_dim,
                out_features=embed_dim,
                std=config.initializer_range,
                key=w_key,
                name=name,
            )
            for w_key, name in zip(keys, ("wq", "wk", "wv", "wo"))
        )

        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

    @property
    def causal_mask(self):
        """子类可按需要实现自己的 mask。"""

        raise NotImplementedError

    @property
    def freqs_cis(self):
        """返回编译期生成的 RoPE 频率表。"""

        with jax.ensure_compile_time_eval():
            # 预留 2 * seq_len，SWA 中窗口缓存和当前 chunk 的位置都可能需要索引。
            freqs_cis = precompute_freqs_cis(
                self.head_dim,
                2 * self.config.seq_len,
                theta=self.config.rope_theta,
                dtype=jnp.float32,
            )

        return freqs_cis

    def _split_heads(self, x):
        """把最后一维 hidden 拆成 `[num_heads, head_dim]`。"""

        return tree_rearrange(x, "... (head head_dim) -> ... head head_dim", head=self.num_heads, head_dim=self.head_dim)

    def _merge_heads(self, x):
        """把 `[num_heads, head_dim]` 合回 hidden 维。"""

        return tree_rearrange(x, "... head head_dim -> ... (head head_dim)", head=self.num_heads, head_dim=self.head_dim)

    def project_qkv(self, hidden_states):
        """从 hidden states 计算 query、key、value。"""

        xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

        return xq, xk, xv

    def get_attention_input(self, hidden_states, position_ids):
        """生成 attention 的 Q/K/V，并对 Q/K 做可选 norm 和 RoPE。"""

        xq, xk, xv = self.project_qkv(hidden_states)  # [T,D]

        xq, xk, xv = self._split_heads((xq, xk, xv))  # [T,nh,d]

        if self.config.qk_norm:
            # QK norm 可以稳定训练，尤其是长上下文和大 rope_theta 设置下。
            rms_forward_fn = maybe_double_remat(
                nn.RMSNorm.__call__, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd
            )
            xq = jax.vmap(jax.vmap(lambda x: rms_forward_fn(self.q_norm, x)))(xq)
            xk = jax.vmap(jax.vmap(lambda x: rms_forward_fn(self.k_norm, x)))(xk)

        xq, xk = self.apply_rope((xq, xk), position_ids=position_ids)
        return xq, xk, xv

    def apply_rope(self, xis: PyTree[jnp.ndarray], position_ids: jnp.ndarray) -> PyTree[jnp.ndarray]:
        """按 position_ids 给一个或多个张量应用 RoPE。"""

        freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)
        apply_rotary_emb_fn = maybe_double_remat(
            apply_rotary_emb, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd
        )
        out_xis = jax.tree.map(lambda x: apply_rotary_emb_fn(x, freqs_cis), xis)
        return out_xis

    def get_attention_output(self, attn_output):
        """attention 输出经过 O 投影和 residual dropout。"""

        o_output = self.wo(attn_output)
        attn_output = self.resid_dropout(o_output)
        return attn_output

    def core_attention_op(self, xq, xk, xv, attention_mask):
        """执行带显式 mask 的 dot-product attention。"""

        if self.config.attn_pdrop > 0.0:
            raise ValueError("Not implemented")

        if self.config.force_flash:
            # cuDNN attention 对 sharding 更敏感，这里约束 head/state 维度布局。
            xq = jax.lax.with_sharding_constraint(xq, P(None, "state", None))
            xk = jax.lax.with_sharding_constraint(xk, P(None, "state", None))
            xv = jax.lax.with_sharding_constraint(xv, P(None, "state", None))

        attn_output = jax.nn.dot_product_attention(xq, xk, xv, mask=attention_mask, implementation="cudnn" if self.config.force_flash else None)

        if self.config.force_flash:
            attn_output = jax.lax.with_sharding_constraint(attn_output, P(None, "state", None))

        attn_output = self._merge_heads(attn_output)

        return attn_output

    def __call__(self, *_args, **_kwargs) -> tuple[Array, nn.State]:
        """抽象接口，由具体 attention 类型实现。"""

        raise NotImplementedError


class Attention(AttentionBase):
    """普通 causal self-attention。"""

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(config, key=key)

    @override
    def __call__(self, hidden_states, seq: Batch, state: nn.State, is_prefix: bool = False):
        """对完整 chunk 做 causal attention。"""

        xq, xk, xv = self.get_attention_input(hidden_states, position_ids=jnp.arange(seq.shape[0]) if seq.position_ids is None else seq.position_ids)

        if self.config.force_flash or is_prefix:
            # prefix 计算通常是整段上下文，强制使用适合 cuDNN 的 sharding。
            xq = jax.lax.with_sharding_constraint(xq, P(None, "state", None))
            xk = jax.lax.with_sharding_constraint(xk, P(None, "state", None))
            xv = jax.lax.with_sharding_constraint(xv, P(None, "state", None))

        attn_output = jax.nn.dot_product_attention(xq, xk, xv, is_causal=True, implementation="cudnn" if (self.config.force_flash or is_prefix) else None)

        if self.config.force_flash or is_prefix:
            attn_output = jax.lax.with_sharding_constraint(attn_output, P(None, "state", None))

        attn_output = self._merge_heads(attn_output)

        attn_output = self.get_attention_output(attn_output)

        return (attn_output, state)


class SWAFull(Attention):
    """直接调用 JAX/cudnn local_window_size 的滑动窗口注意力。"""

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(config, key=key)

    @override
    def __call__(self, hidden_states, seq: Batch, state: nn.State, is_prefix: bool = False):
        """用内置 local-window attention 处理整段 hidden states。"""

        xq, xk, xv = self.get_attention_input(hidden_states, position_ids=jnp.arange(seq.shape[0]) if seq.position_ids is None else seq.position_ids)

        if self.config.force_flash or is_prefix:
            xq = jax.lax.with_sharding_constraint(xq, P(None, "state", None))
            xk = jax.lax.with_sharding_constraint(xk, P(None, "state", None))
            xv = jax.lax.with_sharding_constraint(xv, P(None, "state", None))

        attn_output = jax.nn.dot_product_attention(
            xq,
            xk,
            xv,
            local_window_size=(self.config.sliding_window_size - 1, 0),
            is_causal=True,
            implementation="cudnn" if (self.config.force_flash or is_prefix) else None,
        )

        if self.config.force_flash or is_prefix:
            attn_output = jax.lax.with_sharding_constraint(attn_output, P(None, "state", None))

        attn_output = self._merge_heads(attn_output)

        attn_output = self.get_attention_output(attn_output)

        return (attn_output, state)


class SWA(AttentionBase):
    """手动维护 KV cache 的分块 sliding-window attention。

    每次只处理 `mini_batch_size` 个 token，并把最近 `sliding_window_size` 个 K/V 存进 state。
    """

    kv_cache_index: nn.StateIndex
    chunk_index: nn.StateIndex
    mini_batch_size: int = eqx.field(static=True)
    window_size: int = eqx.field(static=True)

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        """初始化 attention 参数和 KV cache state index。"""

        super().__init__(config, key=key)
        self.mini_batch_size = self.config.mini_batch_size
        self.window_size = self.config.sliding_window_size
        self.kv_cache_index = nn.StateIndex(self.init_kv_cache())
        self.chunk_index = nn.StateIndex(jnp.array(0, dtype=jnp.int32))

    def init_kv_cache(self):
        """创建空 KV cache，shape 为 `[window_size, hidden_size]`。"""

        return (
            jnp.zeros((self.window_size, self.config.hidden_size), dtype=self.compute_dtype),
            jnp.zeros((self.window_size, self.config.hidden_size), dtype=self.compute_dtype),
        )

    def sw_causal_mask(self, chunk_id):
        """为当前 chunk 构造滑动窗口 causal mask。"""

        nk = self.window_size + self.mini_batch_size
        nq = self.mini_batch_size

        # 当前 query 的绝对位置范围。
        starting_query_idx = chunk_id * nq
        ending_query_idx = starting_query_idx + self.mini_batch_size
        # key 包含历史窗口和当前 chunk，结束位置和当前 query chunk 对齐。
        ending_key_idx = ending_query_idx
        qi = (jnp.arange(0, nq, dtype=jnp.int32) + starting_query_idx)[:, None]
        ki = (jnp.arange(-nk, 0, dtype=jnp.int32) + ending_key_idx)[None, :]

        # 同时满足 causal、窗口范围、key 位置非负。
        mask = (qi >= ki) & (qi < ki + self.window_size) & (ki >= 0)
        return mask

    def full_sw_attention(
        self,
        hidden_states,
        seq: Batch,
        state: nn.State,
    ):
        """prefix 阶段使用完整 local-window attention，不更新 KV cache。"""

        xq, xk, xv = self.get_attention_input(hidden_states, position_ids=jnp.arange(seq.shape[0]) if seq.position_ids is None else seq.position_ids)

        xq = jax.lax.with_sharding_constraint(xq, P(None, "state", None))
        xk = jax.lax.with_sharding_constraint(xk, P(None, "state", None))
        xv = jax.lax.with_sharding_constraint(xv, P(None, "state", None))

        attn_output = jax.nn.dot_product_attention(xq, xk, xv, is_causal=True, local_window_size=(self.window_size - 1, 0), implementation="cudnn")

        attn_output = jax.lax.with_sharding_constraint(attn_output, P(None, "state", None))

        attn_output = self._merge_heads(attn_output)

        attn_output = self.get_attention_output(attn_output)

        return (attn_output, state)

    @override
    def __call__(self, hidden_states, seq: Batch, state: nn.State, is_prefix: bool = False):
        """处理一个 suffix chunk，并把新的 K/V 写回 state。"""

        if is_prefix:
            return self.full_sw_attention(hidden_states, seq, state)

        # suffix chunk 路径需要手动拼接历史 KV cache。
        xq, xk, xv = self.project_qkv(hidden_states)

        xq, xk, xv = self._split_heads((xq, xk, xv))  # [CS,nh,d]

        if self.config.qk_norm:
            rms_forward_fn = maybe_double_remat(
                nn.RMSNorm.__call__, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd
            )
            xq = jax.vmap(jax.vmap(lambda x: rms_forward_fn(self.q_norm, x)))(xq)
            xk = jax.vmap(jax.vmap(lambda x: rms_forward_fn(self.k_norm, x)))(xk)

        # 取出上一个 chunk 留下来的窗口 K/V。
        prev_kv_cache = state.get(self.kv_cache_index)  # [WS,D]
        prev_k, prev_v = prev_kv_cache
        prev_k, prev_v = self._split_heads((prev_k, prev_v))

        assert self.mini_batch_size == xq.shape[0]
        assert self.window_size == prev_k.shape[0]

        # 把历史窗口和当前 chunk 拼起来，作为当前 attention 的 key/value。
        xk = jnp.concatenate([prev_k, xk], axis=0)  # [CS+WS,nh,d]
        xv = jnp.concatenate([prev_v, xv], axis=0)

        # 只保留最后 window_size 个 K/V 给下一个 chunk 使用。
        new_kv_cache = self._merge_heads((xk[-self.window_size :], xv[-self.window_size :]))  # [WS,D]

        # query 只对应当前 chunk 的位置；key 覆盖历史窗口 + 当前 chunk 的位置。
        xq = self.apply_rope(xq, position_ids=jnp.arange((self.window_size + self.mini_batch_size), dtype=jnp.int32)[-self.mini_batch_size :])
        xk = self.apply_rope(xk, position_ids=jnp.arange((self.window_size + self.mini_batch_size), dtype=jnp.int32))

        chunk_id = state.get(self.chunk_index)
        causal_mask = self.sw_causal_mask(chunk_id)

        attn_output = self.core_attention_op(xq, xk, xv, causal_mask)

        attn_output = self.get_attention_output(attn_output)

        state = state.set(self.kv_cache_index, new_kv_cache)
        state = state.set(self.chunk_index, (chunk_id + 1))

        outputs = (attn_output, state)

        return outputs
