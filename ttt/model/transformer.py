"""Transformer 和 E2E TTT MetaModel 实现。

这个文件包含三层结构：
1. 基础 Transformer 组件：MLP、Block、BlockCollection。
2. E2E TTT 组件：prefix/suffix 拆分、prime FFN、内循环更新。
3. CausalLM/MetaModel 包装：计算语言模型 logits、loss 和训练参数筛选。
"""

from __future__ import annotations

from enum import StrEnum, auto
from functools import partial

import equinox as eqx
import jax
import jax.ad_checkpoint
import jax.nn
import jax.numpy as jnp
import jax.random as jrandom
from einops import rearrange
from equinox import nn
from jaxtyping import PRNGKeyArray
from optax import OptState

from ttt.config import Config, ModelConfig
from ttt.model.attention import SWA, Attention, AttentionBase, NormalLinear, SWAFull
from ttt.model.data import BaseModelOutput, Batch
from ttt.model.loss import cross_entropy_loss_and_accuracy, token_log_probs
from ttt.optimizers import make_optimizer
from ttt.utils.filter_utils import filter_apply_updates, filter_parameters, get_filter_spec
from ttt.utils.jax_utils import (
    clone_pytree,
    get_float_dtype_by_name,
    maybe_double_remat,
    promote_dtype,
    scan_or_loop,
    scan_remat_chunk,
    tree_rearrange,
)


class SwiGLUMLP(eqx.Module):
    """单个 SwiGLU MLP block。

    结构是 w1、w3 做门控，再通过 w2 投回 hidden size。
    """

    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)
    w1: NormalLinear
    w2: NormalLinear
    w3: NormalLinear
    dropout: nn.Dropout = eqx.field(static=True)

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        """按模型配置初始化三组无 bias 线性层。"""

        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)

        w1_key, w2_key, w3_key = jrandom.split(key, 3)

        self.w1 = NormalLinear(
            self.config, in_features=config.hidden_size, out_features=config.intermediate_size, std=config.initializer_range, key=w1_key, name="w1"
        )

        self.w2 = NormalLinear(
            self.config, in_features=config.intermediate_size, out_features=config.hidden_size, std=config.initializer_range, key=w2_key, name="w2"
        )

        self.w3 = NormalLinear(
            self.config, in_features=config.hidden_size, out_features=config.intermediate_size, std=config.initializer_range, key=w3_key, name="w3"
        )

        self.dropout = nn.Dropout(p=self.config.resid_pdrop)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """执行 SwiGLU 前向传播。"""

        z1 = self.w1(x)
        z1_act = jax.nn.silu(z1)
        z3 = self.w3(x)
        x2 = z1_act * z3
        z2 = self.w2(x2)
        output = self.dropout(z2)
        return output


class PrimeStorage(eqx.Module):
    """保存 E2E TTT 使用的 prime FFN 参数。

    prime 参数会被复制到 suffix blocks 中，用于内循环快速适配上下文。
    """

    ffn_prime_norm: nn.RMSNorm
    ffn_prime_post_norm: nn.RMSNorm
    feed_forward_prime: SwiGLUMLP

    def __init__(
        self,
        config: ModelConfig,
        *,
        key,
    ) -> None:
        """为每个 suffix block 初始化一套 prime FFN 和 norm。"""

        param_dtype = get_float_dtype_by_name(config.param_dtype)
        suffix_len = config.suffix_len

        suffix_keys = jrandom.split(key, suffix_len)
        if config.feed_forward_prime != "swiglu":
            raise NotImplementedError("Only feed_forward_prime='swiglu' is supported.")

        self.feed_forward_prime = jax.vmap(lambda k: SwiGLUMLP(config, key=k))(suffix_keys)
        self.ffn_prime_norm = jax.vmap(lambda _: nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_bias=False, dtype=param_dtype))(suffix_keys)
        self.ffn_prime_post_norm = jax.vmap(lambda _: nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_bias=False, dtype=param_dtype))(suffix_keys)

    def __call__(self):
        """PrimeStorage 只作为参数容器，不直接参与前向调用。"""

        pass


class Block(eqx.Module):
    """一个 Transformer block。

    每个 block 包含 sequence modeling layer（attention/SWA）、普通 FFN，
    以及可选的 prime FFN。prime FFN 只在 E2E TTT 的 suffix blocks 中启用。
    """

    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    seq_modeling_block: AttentionBase
    feed_forward: SwiGLUMLP
    seq_norm: nn.RMSNorm
    ffn_norm: nn.RMSNorm
    seq_post_norm: nn.RMSNorm
    ffn_post_norm: nn.RMSNorm
    ffn_prime_norm: nn.RMSNorm | None
    ffn_prime_post_norm: nn.RMSNorm | None
    feed_forward_prime: SwiGLUMLP | None

    def __init__(
        self,
        config: ModelConfig,
        *,
        key,
        feed_forward_prime: SwiGLUMLP | None = None,
        ffn_prime_norm: nn.RMSNorm = None,
        ffn_prime_post_norm: nn.RMSNorm | None = None,
    ) -> None:
        """初始化 block，并根据配置选择 attention 类型。"""

        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)

        seq_modeling_block_type = self.config.seq_modeling_block

        match seq_modeling_block_type:
            case "self_attention":
                # 普通 causal self-attention。
                seq_modeling_block = Attention
            case "SWA":
                # 手动 KV cache 的 sliding-window attention。
                seq_modeling_block = SWA
            case "SWAFull":
                # 直接调用 cuDNN local-window attention。
                seq_modeling_block = SWAFull
            case _:
                raise NotImplementedError(f"Sequence Modeling Layer {self.config.seq_modeling_block} Not Implemented.")

        key_seq_modeling_block, key_ffn = jrandom.split(key, 2)
        self.seq_modeling_block = seq_modeling_block(self.config, key=key_seq_modeling_block)
        self.feed_forward = SwiGLUMLP(self.config, key=key_ffn)
        self.seq_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.ffn_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.seq_post_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.ffn_post_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)

        self.ffn_prime_norm = ffn_prime_norm
        self.ffn_prime_post_norm = ffn_prime_post_norm
        self.feed_forward_prime = feed_forward_prime

    def seq_modeling_forward(
        self, seq_modeling_block_fn, rms_forward_fn, seq_norm, seq_modeling_block, seq_post_norm, hidden_states, state: nn.State, seq: Batch
    ):
        """执行 attention/SWA 子层，包含可选 pre-norm 和 post-norm。"""

        if self.config.pre_norm:
            # 对序列维逐 token 应用 RMSNorm。
            seq_modeling_input = jax.vmap(lambda x: rms_forward_fn(seq_norm, x))(hidden_states)
        else:
            seq_modeling_input = hidden_states

        seq_modeling_hidden_states, state = seq_modeling_block_fn(seq_modeling_block, seq_modeling_input, seq, state)

        if self.config.post_norm:
            seq_modeling_hidden_states = jax.vmap(lambda x: rms_forward_fn(seq_post_norm, x))(seq_modeling_hidden_states)

        return seq_modeling_hidden_states, state

    def ffn_forward(self, feed_forward_fn, rms_forward_fn, ffn_norm, feed_forward, ffn_post_norm, hidden_states):
        """执行一个 FFN 子层，包含可选 pre-norm 和 post-norm。"""

        if self.config.pre_norm:
            feed_forward_input = jax.vmap(lambda x: rms_forward_fn(ffn_norm, x))(hidden_states)
        else:
            feed_forward_input = hidden_states

        feed_forward_hidden_states = feed_forward_fn(feed_forward, feed_forward_input)

        if self.config.post_norm:
            feed_forward_hidden_states = jax.vmap(lambda x: rms_forward_fn(ffn_post_norm, x))(feed_forward_hidden_states)

        return feed_forward_hidden_states

    def __call__(self, hidden_states, state: nn.State, seq: Batch, is_prefix: bool = False):
        """执行完整 Transformer block 前向传播。"""

        config = self.config

        # 按配置给 attention、MLP、RMSNorm 包 remat，控制显存和重算开销。
        seq_modeling_block_fn = maybe_double_remat(
            partial(self.seq_modeling_block.__class__.__call__, is_prefix=is_prefix),
            prevent_cse=True,
            policy_remat=config.remat_attention,
            policy_remat_bwd=config.remat_attention_bwd,
        )
        feed_forward_fn = maybe_double_remat(
            self.feed_forward.__class__.__call__, prevent_cse=True, policy_remat=config.remat_mlp, policy_remat_bwd=config.remat_mlp_bwd
        )
        rms_forward_fn = maybe_double_remat(nn.RMSNorm.__call__, prevent_cse=True, policy_remat=config.remat_rms, policy_remat_bwd=config.remat_rms_bwd)
        if self.feed_forward_prime is not None:
            feed_forward_prime_fn = maybe_double_remat(
                self.feed_forward_prime.__class__.__call__, prevent_cse=True, policy_remat=config.remat_mlp, policy_remat_bwd=config.remat_mlp_bwd
            )

        seq_modeling_output, state = self.seq_modeling_forward(
            seq_modeling_block_fn, rms_forward_fn, self.seq_norm, self.seq_modeling_block, self.seq_post_norm, hidden_states, state, seq
        )

        hidden_states = hidden_states + seq_modeling_output

        feed_forward_prime_hidden_states = None
        if self.feed_forward_prime is not None:
            # E2E TTT 的 suffix block 先经过可内循环更新的 prime FFN。
            feed_forward_prime_hidden_states = self.ffn_forward(
                feed_forward_prime_fn,
                rms_forward_fn,
                self.ffn_prime_norm,
                self.feed_forward_prime,
                self.ffn_prime_post_norm,
                hidden_states,
            )

            feed_forward_prime_hidden_states = hidden_states + feed_forward_prime_hidden_states
            hidden_states = feed_forward_prime_hidden_states

        feed_forward_hidden_states = self.ffn_forward(feed_forward_fn, rms_forward_fn, self.ffn_norm, self.feed_forward, self.ffn_post_norm, hidden_states)

        # 标准 FFN 残差连接。
        hidden_states = hidden_states + feed_forward_hidden_states

        return hidden_states, state

    def weights(self):
        """返回 block 内所有浮点参数。"""

        return eqx.filter(self, eqx.is_inexact_array)

    def inner_parameters(self, config: Config):
        """返回这个 block 中被内循环选中的参数。"""

        inner_specs = config.training.spec_inner
        inner_specs_rebased = []
        for spec in inner_specs:
            assert "suffix_blocks" in spec, "Inner params must lie in suffix blocks"
            # e.g., **.suffix_blocks.feed_forward_prime.** --> feed_forward_prime.**
            spec_rebased = spec.split("suffix_blocks")[1][1:]
            inner_specs_rebased.append(spec_rebased)
        return filter_parameters(self.weights(), inner_specs_rebased, "inner parameters")


class BlockCollectionSplit(eqx.Module):
    """把完整 block stack 拆成 prefix blocks 和 suffix blocks。

    E2E TTT 中 prefix 只负责读取上下文并产生 hidden states；
    suffix 会按 chunk 运行，并在内循环中更新指定参数。
    """

    config: ModelConfig = eqx.field(static=True, repr=False)
    prefix_blocks: Block  # vmap-ed init and application
    suffix_blocks: Block  # vmap-ed init and application

    def __init__(
        self,
        config: ModelConfig,
        block_collection: Block,
        prime_storage: PrimeStorage,
        *,
        key: PRNGKeyArray,
    ):
        """根据 suffix_len 拆分 block，并把 prime 参数插入 suffix blocks。"""

        self.config = config
        suffix_len = self.config.suffix_len
        # 前 num_layers - suffix_len 层是 prefix；suffix_len 为 0 时全部作为 prefix。
        self.prefix_blocks = jax.tree.map(lambda m: m[:-suffix_len], block_collection) if suffix_len > 0 else block_collection
        self.suffix_blocks = None

        if suffix_len > 0:
            # 后 suffix_len 层会参与 suffix 计算。
            self.suffix_blocks = jax.tree.map(lambda m: m[-suffix_len:], block_collection)
            if prime_storage is not None:
                suffix_keys = jrandom.split(key, suffix_len)

                # 先用 prime 参数构造一套 suffix block 模板。
                argdict = {"key": suffix_keys}
                argdict["ffn_prime_norm"] = prime_storage.ffn_prime_norm
                argdict["ffn_prime_post_norm"] = prime_storage.ffn_prime_post_norm
                argdict["feed_forward_prime"] = prime_storage.feed_forward_prime

                # copy in prime params
                suffix_blocks_template = jax.vmap(lambda **kwargs: Block(config=self.config, **kwargs))(**argdict)

                # copy in non-prime params
                # 再把原始 suffix block 的 attention、普通 FFN 和 norm 参数拷回模板。
                self.suffix_blocks = eqx.tree_at(
                    lambda m: (m.seq_norm, m.seq_modeling_block, m.seq_post_norm, m.ffn_norm, m.feed_forward, m.ffn_post_norm),
                    suffix_blocks_template,
                    (
                        self.suffix_blocks.seq_norm,
                        self.suffix_blocks.seq_modeling_block,
                        self.suffix_blocks.seq_post_norm,
                        self.suffix_blocks.ffn_norm,
                        self.suffix_blocks.feed_forward,
                        self.suffix_blocks.ffn_post_norm,
                    ),
                )

    @staticmethod
    def split_state(state: nn.State, suffix_len: int):
        """把 block state 按 prefix/suffix 维度拆开。"""

        if suffix_len > 0:
            return (
                jax.tree.map(lambda s: s[:-suffix_len], state),
                # 有些 state 可能没有足够层数，补零到 suffix_len 方便 scan 对齐。
                jax.tree.map(lambda s: s[-suffix_len:] if len(s) >= suffix_len else jnp.zeros((suffix_len, *s.shape[1:]), dtype=s.dtype), state),
            )
        else:
            return (state, None)

    def prefix_call(self, prefix_blocks, hidden_states: jnp.ndarray, state: nn.State, seq: Batch):
        """只运行 prefix blocks，返回 suffix 的输入 hidden states。"""

        if prefix_blocks is not None:
            prefix_fn = partial(prefix_blocks.__class__.__call__, is_prefix=True)
            block_fn = maybe_double_remat(prefix_fn, prevent_cse=True, policy_remat=self.config.remat_prefix_block, policy_remat_bwd="")

            # Note: Prefix has no state
            def apply_block_prefix(x, block):
                x, _ = block_fn(block, x, None, seq)
                return x, None

            hidden_states, _ = jax.lax.scan(
                apply_block_prefix,
                hidden_states,
                prefix_blocks,
                unroll=self.config.unroll_block_scan,
            )

        outputs = BaseModelOutput(last_hidden_state=hidden_states, state=state)
        return outputs

    def suffix_call(self, hidden_states: jnp.ndarray, state: nn.State, seq: Batch):
        """运行 suffix blocks，并更新 suffix 对应的 state。"""

        if self.suffix_blocks is not None:
            suffix_fn = partial(self.suffix_blocks.__class__.__call__, is_prefix=False)
            block_fn = maybe_double_remat(
                suffix_fn,
                prevent_cse=True,
                policy_remat=self.config.remat_block,
                policy_remat_bwd=self.config.remat_block_bwd,
            )

            def apply_block_suffix(x, block__substate):
                block, substate = block__substate
                x, substate = block_fn(block, x, substate, seq)
                return x, substate

            hidden_states, state = jax.lax.scan(
                apply_block_suffix,
                hidden_states,
                (self.suffix_blocks, state),
                unroll=self.config.unroll_block_scan,
            )

        outputs = BaseModelOutput(last_hidden_state=hidden_states, state=state)
        return outputs

    def __call__(
        self,
        hidden_states,
        state: tuple[nn.State, nn.State],
        seq: Batch,
    ):
        """按 prefix 再 suffix 的顺序运行完整拆分后的 block stack。"""

        block_fn = maybe_double_remat(
            self.prefix_blocks.__class__.__call__,
            prevent_cse=True,
            policy_remat=self.config.remat_block,
            policy_remat_bwd=self.config.remat_block_bwd,
        )

        def apply_block_prefix(x, block__substate):
            block, substate = block__substate
            x, substate = block_fn(block, x, substate, seq)
            return x, substate

        substate_prefix, substate_suffix = state

        hidden_states, substate_prefix = jax.lax.scan(
            apply_block_prefix,
            hidden_states,
            (self.prefix_blocks, substate_prefix),
            unroll=self.config.unroll_block_scan,
        )

        if self.suffix_blocks is not None:

            def apply_block_suffix(x, block__substate):
                block, substate = block__substate
                x, substate = block_fn(block, x, substate, seq)
                return x, substate

            hidden_states, substate_suffix = jax.lax.scan(
                apply_block_suffix,
                hidden_states,
                (self.suffix_blocks, substate_suffix),
                unroll=self.config.unroll_block_scan,
            )

        outputs = BaseModelOutput(last_hidden_state=hidden_states, state=(substate_prefix, substate_suffix))
        return outputs


class BlockCollection(eqx.Module):
    """完整 Transformer block stack。

    `blocks` 是通过 vmap 初始化的一组 Block，scan 时逐层执行。
    """

    config: ModelConfig = eqx.field(static=True, repr=False)
    blocks: Block  # vmap-ed init and application
    prime_storage: PrimeStorage

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        """初始化所有 Transformer blocks，以及可选 prime 参数容器。"""

        self.config = config

        key, prime_key = jrandom.split(key, 2)

        # 每一层用独立随机 key 初始化，然后 vmap 成一个 block pytree。
        keys = jrandom.split(key, config.num_hidden_layers)
        self.blocks = jax.vmap(lambda k: Block(config, key=k))(keys)

        self.prime_storage = None

        if config.prime:
            # prime 参数需要存在于永久模型结构中，外循环才能对它求梯度和保存 checkpoint。
            self.prime_storage = PrimeStorage(config, key=prime_key)

    def __call__(
        self,
        hidden_states,
        state: nn.State,
        seq: Batch,
    ):
        """逐层执行完整 block stack。"""

        # 从 Equinox state 里取出只属于 blocks 的子状态，例如 SWA 的 KV cache。
        substate = state.substate(self.blocks)
        block_fn = maybe_double_remat(
            self.blocks.__class__.__call__,
            prevent_cse=True,
            policy_remat=self.config.remat_block,
            policy_remat_bwd=self.config.remat_block_bwd,
        )

        def apply_block(x, block__substate):
            """scan 的单层执行函数。"""

            block, substate = block__substate

            x, substate = block_fn(block, x, substate, seq)
            return x, substate

        hidden_states, substate = scan_or_loop(apply_block, hidden_states, (self.blocks, substate), use_loop=self.config.unroll_block_scan)

        # 把 blocks 子状态写回全局 state。
        state = state.update(substate)

        outputs = BaseModelOutput(last_hidden_state=hidden_states, state=state)
        return outputs


class TransformerModel(eqx.Module):
    """不含语言模型 loss 的 Transformer 主体。"""

    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    wte: nn.Embedding
    dropout: nn.Dropout = eqx.field(static=True)
    ln_f: nn.RMSNorm
    h: BlockCollection | BlockCollectionSplit

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        """初始化 token embedding、block stack 和最终 RMSNorm。"""

        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)

        key_embed, key_block = jrandom.split(key, 2)

        vocab_size, embed_dim = config.vocab_size, config.hidden_size

        # 词嵌入矩阵。如果 tie_word_embeddings=True，输出层会复用它。
        self.wte = nn.Embedding(
            weight=jax.nn.initializers.normal(stddev=self.config.initializer_range, dtype=self.param_dtype)(key_embed, (vocab_size, embed_dim)),
        )

        self.dropout = nn.Dropout(p=self.config.embd_pdrop)
        self.h = BlockCollection(
            self.config,
            key=key_block,
        )
        self.ln_f = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)

    def wte_call(self, input_ids: jnp.ndarray):
        """只执行 token embedding 和 embedding dropout。"""

        input_embeds = jax.vmap(self.wte)(input_ids.astype(jnp.int32))
        input_embeds = input_embeds.astype(self.compute_dtype)
        hidden_states = self.dropout(input_embeds)
        return hidden_states

    def prefix_call(
        self,
        prefix: Block,
        hidden_states: jnp.ndarray,
        state: nn.State,
        seq: Batch,
    ):
        """调用拆分后的 prefix blocks。"""

        outputs: BaseModelOutput = self.h.prefix_call(prefix, hidden_states, state=state, seq=seq)
        return outputs

    def suffix_call(
        self,
        prefix_outputs: jnp.ndarray,
        state: nn.State,
        seq: Batch,
    ):
        """调用拆分后的 suffix blocks，并接最终 RMSNorm。"""

        outputs: BaseModelOutput = self.h.suffix_call(prefix_outputs, state=state, seq=seq)
        hidden_states = outputs.last_hidden_state
        hidden_states = jax.vmap(self.ln_f)(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states, state=outputs.state)

    def __call__(
        self,
        state: nn.State,
        seq: Batch,
    ):
        """普通完整 Transformer 前向传播。"""

        rms_forward_fn = maybe_double_remat(
            nn.RMSNorm.__call__, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd
        )

        # token id -> hidden states。
        input_embeds = jax.vmap(self.wte)(seq.input_ids.astype(jnp.int32))
        input_embeds = input_embeds.astype(self.compute_dtype)
        hidden_states = self.dropout(input_embeds)
        outputs: BaseModelOutput = self.h(
            hidden_states,
            state=state,
            seq=seq,
        )
        hidden_states = outputs.last_hidden_state
        # 最后一层 RMSNorm 后交给 lm_head。
        hidden_states = jax.vmap(lambda x: rms_forward_fn(self.ln_f, x))(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states, state=outputs.state)


class MetaModel(eqx.Module):
    """训练用的高层模型封装。

    它包含 CausalLM、本轮 step 状态、内循环优化器构造，以及计算 meta/pretrain loss 的逻辑。
    """

    class Output(eqx.Module):
        """MetaModel 前向输出。当前代码主要通过 loss 函数使用模型。"""

        lm_output: CausalLM.Output
        state: nn.State

    class MetricType(StrEnum):
        """训练和评估过程中会记录的指标名称。"""

        loss = auto()
        token_nll_loss = auto()
        outer_grad_norm = auto()

    config: Config = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)
    state_dtype: jnp.dtype = eqx.field(static=True)

    step_index: nn.StateIndex
    language_model: CausalLM

    def __init__(
        self,
        config: Config,
        *,
        key: PRNGKeyArray,
    ):
        """初始化语言模型和 step state。"""

        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.model.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.model.param_dtype)
        self.state_dtype = get_float_dtype_by_name(self.config.model.state_dtype)

        self.step_index = nn.StateIndex(jnp.array(0, dtype=jnp.int32))

        self.language_model = CausalLM(config.model, key=key)

    def get_ilr_multiplier(self, step: jnp.ndarray):
        """计算 inner learning rate 的 warmup 倍率。"""

        if self.config.training.ilr_warmup_steps == 0:
            ilr_multiplier = 1.0
        else:
            assert self.config.training.ilr_warmup_steps > 0
            # 从 ilr_init 线性 warmup 到 optimizer_inner.lr。
            progress = jnp.minimum(1.0, 1.0 * (step + 1) / self.config.training.ilr_warmup_steps)
            ilr = self.config.training.ilr_init + (self.config.training.optimizer_inner.lr - self.config.training.ilr_init) * progress
            ilr_multiplier = (ilr / self.config.training.optimizer_inner.lr).astype(self.state_dtype)

        return ilr_multiplier

    def inner_optimizer(self, state: nn.State):
        """基于当前 step 创建内循环优化器。"""

        step = state.get(self.step_index)
        ilr_multiplier = self.get_ilr_multiplier(step)
        optimizer, _optimizer_info = make_optimizer(self.config.training.optimizer_inner, ilr_multiplier)
        return optimizer

    def __call__(self, seq: Batch, state: nn.State) -> MetaModel.Output:
        """预留的直接前向接口；当前训练路径不调用它。"""

        pass

    class AdaptResult(eqx.Module):
        """Prompt/context adaptation result for inference paths."""

        adapted_model: MetaModel
        adapted_state: nn.State | tuple[nn.State, nn.State]
        loss: jnp.ndarray
        metrics: dict[MetaModel.MetricType, jnp.ndarray]

        def __iter__(self):
            """Allow `model, state, loss, metrics = result` unpacking."""

            return iter((self.adapted_model, self.adapted_state, self.loss, self.metrics))

    class InnerLoopStepResult(eqx.Module):
        """一次内循环更新后的返回结构。"""

        new_model: MetaModel
        new_optimizer_state: OptState
        new_state: nn.State
        metrics: dict[MetaModel.MetricType, jnp.ndarray]

        def __iter__(self):
            """允许 `new_model, opt_state, state, metrics = result` 解包。"""

            return iter((self.new_model, self.new_optimizer_state, self.new_state, self.metrics))

    def inner_loop_step(
        self,
        opt_state: OptState,
        state_tuple: tuple[nn.State, nn.State, nn.State],
        seq: Batch,
        prefix_outputs: jnp.ndarray,
    ) -> InnerLoopStepResult:
        """执行一次 E2E TTT 内循环更新。

        内循环只更新 `training.spec_inner` 选中的参数，通常是 suffix blocks 的 prime FFN。

        Args:
            opt_state: 内循环优化器状态。
            state_tuple: 全局 state 和 suffix state。
            seq: 当前 chunk 的 token batch。
            prefix_outputs: prefix blocks 对当前 chunk 的 hidden states。

        Returns:
            内循环更新后的模型、优化器状态、state 和指标。
        """

        M = MetaModel.MetricType
        md: dict[MetaModel.MetricType, jnp.ndarray] = {}

        state_all, suffix_state = state_tuple

        # 对当前 chunk 的语言模型 loss 求梯度，并把 loss/token NLL 作为 aux 返回。
        value_and_grad_fn = eqx.filter_value_and_grad(MetaModel.lm_loss, has_aux=True)

        (_loss_with_aux, (md[M.loss], md[M.token_nll_loss], new_suffix_state)), grads = value_and_grad_fn(
            self, seq, suffix_state, prefix_outputs=prefix_outputs
        )

        # 只保留内循环允许更新的参数梯度。
        inner_grads = grads.inner_parameters()
        updates, new_optimizer_state = self.inner_optimizer(state_all).update(inner_grads, opt_state, self.inner_parameters())
        new_model = filter_apply_updates(self, updates)

        # state_all 保存 step 等全局状态；suffix_state 保存 SWA KV cache 等 suffix 状态。
        new_state_tuple = (state_all, new_suffix_state)

        return MetaModel.InnerLoopStepResult(new_model=new_model, new_optimizer_state=new_optimizer_state, new_state=new_state_tuple, metrics=md)

    def lm_loss(
        self, seq: Batch, state: nn.State, *, prefix_outputs: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, nn.State]]:
        """计算一段 token 的语言模型 loss。

        如果传入 prefix_outputs，则只运行 suffix；否则运行完整 CausalLM。
        """

        if prefix_outputs is None:
            lm_outputs = self.language_model(
                seq=seq,
                state=state,
            )
        else:
            # meta 模式中 prefix 已经预先算好，suffix 只接收 prefix hidden states。
            lm_outputs = self.language_model.suffix_call(prefix_outputs=prefix_outputs, state=state, seq=seq)

        loss, loss_pure_ce = cross_entropy_loss_and_accuracy(lm_outputs.logits, seq.target_tokens, seq.loss_masks)
        token_nll_loss = -token_log_probs(lm_outputs.logits, seq.target_tokens)

        return loss, (loss_pure_ce, token_nll_loss, lm_outputs.new_state)

    def next_token_logits_for_sequence(self, seq: Batch, state: nn.State, last_index: jnp.ndarray) -> jnp.ndarray:
        """Return next-token logits at `last_index` without materializing all sequence logits.

        This is intentionally a conservative full-prefix inference path: it recomputes the visible
        context with the adapted model, then projects only the selected hidden state to vocabulary logits.
        """

        tokens_per_chunk = self.config.model.mini_batch_size
        last_index = jnp.asarray(last_index, dtype=jnp.int32)

        if isinstance(self.language_model.model.h, BlockCollectionSplit):
            h = self.language_model.model.h
            state_prefix = state.substate(h.prefix_blocks)

            xt_embed = self.language_model.wte_call(seq.input_ids)
            prefix_output = eqx.filter_checkpoint(self.language_model.prefix_call)(
                h.prefix_blocks,
                xt_embed,
                state_prefix,
                seq,
            ).last_hidden_state

            if h.suffix_blocks is None:
                hidden_states = jax.vmap(self.language_model.model.ln_f)(prefix_output)
                return self.language_model.wte_disembed_call(hidden_states[last_index])

            state_suffix = state.substate(h.suffix_blocks)
            seq_chunks = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
            prefix_chunks = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
            chunk_indices = jnp.arange(seq.input_ids.shape[0] // tokens_per_chunk, dtype=jnp.int32)
            last_chunk_index = last_index // tokens_per_chunk
            last_token_index = last_index % tokens_per_chunk
            init_logits = jnp.zeros((self.config.model.vocab_size,), dtype=self.compute_dtype)

            def process_suffix_chunk(carry, inputs):
                suffix_state, selected_logits = carry
                chunk_index, suffix_chunk, prefix_chunk = inputs

                def run_chunk(run_carry):
                    run_state, run_logits = run_carry
                    outputs: BaseModelOutput = self.language_model.model.suffix_call(prefix_chunk, state=run_state, seq=suffix_chunk)

                    def select_logits(_):
                        return self.language_model.wte_disembed_call(outputs.last_hidden_state[last_token_index])

                    run_logits = jax.lax.cond(chunk_index == last_chunk_index, select_logits, lambda _: run_logits, operand=None)
                    return outputs.state, run_logits

                return jax.lax.cond(chunk_index <= last_chunk_index, run_chunk, lambda skip_carry: skip_carry, (suffix_state, selected_logits)), None

            (_state_suffix, selected_logits), _ = scan_or_loop(
                process_suffix_chunk,
                (state_suffix, init_logits),
                (chunk_indices, seq_chunks, prefix_chunks),
                use_loop=self.config.model.unroll_inner_scan,
            )
            return selected_logits

        outputs = self.language_model.model(
            state,
            seq,
        )
        return self.language_model.wte_disembed_call(outputs.last_hidden_state[last_index])

    @staticmethod
    def _flatten_sequence_metrics(metrics: dict[MetricType, jnp.ndarray]) -> dict[MetricType, jnp.ndarray]:
        """Flatten chunk/token metrics to match the historical loss_for_sequence output."""

        return jax.tree.map(lambda x: x if x.ndim == 1 else rearrange(x, "window data ... -> (window data) ..."), metrics)

    def adapt_on_sequence(self, seq: Batch, state: nn.State) -> AdaptResult:
        """Run the existing inner-loop sequence adaptation and return the final adapted model/state.

        meta 模式会复制模型权重做内循环更新，避免修改输入模型本身。

        Args:
            seq: 单条序列，不能带 batch 维，shape 约为 `[T]`。
            state: Equinox 模型状态。

        Returns:
            适配后的模型/state、序列 loss，以及包含 token-level NLL 等信息的指标字典。
        """
        cfg = self.config

        # 先把完整 block stack 拆成 prefix/suffix，必要时把 prime 参数插入 suffix。
        block_collection = self.language_model.model.h.blocks
        prime_storage = self.language_model.model.h.prime_storage if cfg.model.prime else None
        new_collection = BlockCollectionSplit(
            cfg.model,
            block_collection=block_collection,
            prime_storage=prime_storage,
            key=jrandom.PRNGKey(0),
        )

        # state 也要按 prefix/suffix 拆开，因为 SWA 的 KV cache 是逐 block 保存的。
        state_prefix_suffix = state.substate(self.language_model.model.h.blocks)

        state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
        state_all = clone_pytree(state)

        # 临时把模型中的完整 BlockCollection 替换成拆分后的 BlockCollectionSplit。
        split_model: MetaModel = eqx.tree_at(lambda m: m.language_model.model.h, self, new_collection)

        seqlen = cfg.training.seq_length
        tokens_per_chunk = cfg.model.mini_batch_size

        assert seqlen % tokens_per_chunk == 0, f"For now, seqlen {seqlen} must be divisible by chunk {tokens_per_chunk}"

        M = MetaModel.MetricType

        if cfg.training.train_mode == "meta":
            # meta 模式中会在一条序列内部做 inner-loop TTT。
            model: MetaModel = jax.tree.map(lambda p: p.astype(split_model.state_dtype), split_model)
            inner_opt_state = model.inner_optimizer(state_all).init(model.inner_parameters())

            # prefix 对整条序列只算一次，得到每个 token 的 prefix hidden state。
            xt_embed = split_model.language_model.wte_call(seq.input_ids)
            prefix_output = eqx.filter_checkpoint(split_model.language_model.prefix_call)(
                split_model.language_model.model.h.prefix_blocks, xt_embed, state_prefix, seq
            ).last_hidden_state

            def process_suffix_chunk(model__opt_state__state, inputs: tuple[Batch, jnp.ndarray]):
                """处理一个 suffix chunk：计算 loss，更新内循环参数，再把模型传给下个 chunk。"""

                model_inner, inner_opt_state, state_tuple = model__opt_state__state
                suffix_chunk, prefix_chunk = inputs

                # 内循环参数来自已经更新过的 model_inner，外循环参数保持原始 model。
                spec_inner = get_filter_spec(model_inner, split_model.config.training.spec_inner, "inner parameters")
                inner_params, _ = eqx.partition(model_inner, spec_inner)
                _, outer_params = eqx.partition(model, spec_inner)
                model_inner: MetaModel = eqx.combine(inner_params, outer_params)

                new_model, inner_opt_state, state_tuple, metrics = MetaModel.inner_loop_step(
                    model_inner, inner_opt_state, state_tuple, suffix_chunk, prefix_chunk
                )

                return (new_model, inner_opt_state, state_tuple), metrics

            # 把长序列按 mini_batch_size 切成多个 chunk，逐块做内循环。
            seq = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
            prefix_output = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)

            carry, metrics = scan_remat_chunk(
                eqx.filter_checkpoint(process_suffix_chunk, prevent_cse=False),
                (model, inner_opt_state, (state_all, state_suffix)),
                (seq, prefix_output),
                remat_n_loops=cfg.training.inner_remat_freq,
                unroll=cfg.model.unroll_inner_scan,
            )

            loss = metrics[M.loss].mean()
            adapted_model, _inner_opt_state, adapted_state = carry

        elif cfg.training.train_mode == "pretrain":
            # pretrain 模式不做内循环，只按 chunk 顺序计算普通 LM loss。
            metrics: dict[MetaModel.MetricType, jnp.ndarray] = {}

            seq = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)

            def process_one_window(state, seq_chunk):
                """普通预训练模式下处理一个 chunk。"""

                loss, (loss_pure_ce, token_nll_loss, state) = split_model.lm_loss(seq_chunk, state)
                return state, (loss, loss_pure_ce, token_nll_loss)

            adapted_state, (loss, metrics[M.loss], metrics[M.token_nll_loss]) = scan_remat_chunk(
                process_one_window,
                (state_prefix, state_suffix),
                seq,
                remat_n_loops=cfg.training.inner_remat_freq,
                unroll=cfg.model.unroll_inner_scan,
            )
            loss = loss.mean()
            adapted_model = split_model

        else:
            raise NotImplementedError(f"Training mode {cfg.training.train_mode} not implemented")

        # Flatten window into data dimension
        # 把 `[window, data, ...]` 合成 `[window * data, ...]`，方便后续日志和平均。
        metrics = MetaModel._flatten_sequence_metrics(metrics)
        return MetaModel.AdaptResult(adapted_model=adapted_model, adapted_state=adapted_state, loss=loss, metrics=metrics)

    def loss_for_sequence(self, seq: Batch, state: nn.State) -> tuple[jnp.ndarray, dict[MetricType, jnp.ndarray]]:
        """处理单条序列并返回 loss 和指标。"""

        result = self.adapt_on_sequence(seq, state)
        loss, metrics = result.loss, result.metrics
        return loss, metrics

    def weights(self):
        """返回所有浮点权重，包括冻结参数。"""

        return eqx.filter(self, eqx.is_inexact_array)

    def trainable_parameters(self):
        """返回外循环可训练参数，由 `training.spec_outer` 控制。"""

        return filter_parameters(self.weights(), self.config.training.spec_outer, "outer parameters")

    def inner_parameters(self):
        """返回内循环可训练参数，由 `training.spec_inner` 控制。"""

        return filter_parameters(self.weights(), self.config.training.spec_inner, "inner parameters")


class CausalLM(eqx.Module):
    """带语言模型输出头的 Transformer。"""

    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    model: TransformerModel
    lm_head: NormalLinear | None

    class Output(eqx.Module):
        """CausalLM 前向输出。"""

        last_hidden_states: jnp.ndarray
        logits: jnp.ndarray
        new_state: nn.State

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        """初始化 Transformer 主体和可选 lm_head。"""

        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)
        key_model, key_word_embeddings = jrandom.split(key, 2)

        self.model = TransformerModel(self.config, key=key_model)

        if not self.config.tie_word_embeddings:
            # 不共享 embedding 时，单独初始化输出投影。
            self.lm_head = NormalLinear(
                self.config,
                in_features=config.hidden_size,
                out_features=config.output_size,
                std=config.initializer_range,
                key=key_word_embeddings,
                name="lm_head",
            )
        else:
            # 共享 embedding 时，输出 logits 使用 wte.weight.T。
            self.lm_head = None

    def wte_call(self, input_ids: jnp.ndarray):
        """调用底层 token embedding。"""

        hidden_states = self.model.wte_call(input_ids)
        return hidden_states

    def prefix_call(self, prefix: Block, hidden_states: jnp.ndarray, state: nn.State, seq: Batch):
        """运行 prefix blocks，返回 suffix 所需 hidden states。"""

        outputs = self.model.prefix_call(prefix, hidden_states, state, seq)
        hidden_states = outputs.last_hidden_state
        assert hidden_states.dtype == self.compute_dtype, "The hidden_states before lm_head should be in compute_dtype"
        return outputs

    def suffix_call(self, prefix_outputs: jnp.ndarray, state: nn.State, seq: Batch):
        """运行 suffix blocks 并计算 logits。"""

        outputs = self.model.suffix_call(
            prefix_outputs,
            state,
            seq,
        )
        hidden_states = outputs.last_hidden_state
        assert hidden_states.dtype == self.compute_dtype, "The hidden_states before lm_head should be in compute_dtype"

        if self.config.tie_word_embeddings:
            # tied embedding：复用输入 embedding 权重的转置作为输出层。
            shared_kernel = self.model.wte.weight.T
            hidden_states, shared_kernel = promote_dtype(hidden_states, shared_kernel, dtype=self.compute_dtype)
            lm_logits = hidden_states @ shared_kernel
        else:
            lm_logits = self.lm_head(hidden_states)

        return CausalLM.Output(last_hidden_states=hidden_states, logits=lm_logits, new_state=outputs.state)

    def wte_disembed_call(self, hidden_states: jnp.ndarray):
        """只执行 hidden states 到 vocab logits 的投影。"""

        if self.config.tie_word_embeddings:
            shared_kernel = self.model.wte.weight.T
            hidden_states, shared_kernel = promote_dtype(hidden_states, shared_kernel, dtype=self.compute_dtype)
            lm_logits = hidden_states @ shared_kernel
        else:
            lm_logits = self.lm_head(hidden_states)

        return lm_logits

    def __call__(
        self,
        state: nn.State,
        seq: Batch,
    ) -> CausalLM.Output:
        """完整 CausalLM 前向传播：Transformer + lm_head。"""

        outputs = self.model(
            state,
            seq,
        )
        hidden_states = outputs.last_hidden_state
        assert hidden_states.dtype == self.compute_dtype, "The hidden_states before lm_head should be in compute_dtype"

        if self.config.tie_word_embeddings:
            # tied embedding 路径。
            shared_kernel = self.model.wte.weight.T
            hidden_states, shared_kernel = promote_dtype(hidden_states, shared_kernel, dtype=self.compute_dtype)
            lm_logits = hidden_states @ shared_kernel
        else:
            # 独立 lm_head 路径。
            lm_logits = self.lm_head(hidden_states)

        return CausalLM.Output(last_hidden_states=hidden_states, logits=lm_logits, new_state=outputs.state)
