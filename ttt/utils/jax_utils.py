"""JAX 训练工具函数。

这里放的是跨模块复用的底层工具：分布式初始化、随机种子、pytree 变换、scan/remat、
dtype 转换和梯度范数计算。
"""

import logging
import os
import random
import typing as tp
from collections.abc import Callable, Hashable
from functools import partial, wraps
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from einops import rearrange
from jax import lax
from jaxtyping import PRNGKeyArray, PyTree
from optax._src import numerics
from tqdm import tqdm as _tqdm

from ttt.config import JaxDistributedConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Dtype = jax.typing.DTypeLike | Any


def master_log(logger, *args, level=logging.INFO, **kwargs):
    """只在主进程打印日志，避免多机训练时重复输出。"""

    if jax.process_index() == 0:
        logger.log(level, *args, **kwargs)


def initialize_distibuted(distributed_config: JaxDistributedConfig):
    """根据配置初始化 JAX 后端和多进程分布式环境。"""

    if distributed_config.backend:
        os.environ["JAX_PLATFORM_NAME"] = distributed_config.backend

        if distributed_config.backend == "cpu":
            # CPU 调试时可以伪造多个 JAX device，方便测试并行逻辑。
            cpu_count = os.cpu_count()
            if distributed_config.num_devices and cpu_count is not None and cpu_count < distributed_config.num_devices:
                raise ValueError(f"Requested {distributed_config.num_devices} CPU devices, but only {os.cpu_count()} are available.")

            core_count = distributed_config.num_devices or os.cpu_count()
            os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={core_count}"

    if distributed_config.distributed:
        try:
            local_device_ids = None
            if distributed_config.local_device_ids:
                # 例如 "0,1,2,3,4,5,6,7" 会限制当前进程只看这些本地 GPU。
                local_device_ids = [int(x) for x in distributed_config.local_device_ids.split(",")]

            jax.distributed.initialize(
                coordinator_address=distributed_config.coordinator_address,
                num_processes=distributed_config.num_processes,
                process_id=distributed_config.process_id,
                local_device_ids=local_device_ids,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize JAX distributed: {e}")


def get_float_dtype_by_name(dtype):
    """把配置里的 dtype 字符串转换成 JAX dtype。"""

    match dtype:
        case "bf16" | "bfloat16":
            return jnp.bfloat16
        case "fp16" | "float16":
            return jnp.float16
        case "fp32" | "float32":
            return jnp.float32
        case "fp64" | "float64":
            return jnp.float64
        case _:
            raise ValueError(f"Unknown dtype: {dtype}")


def get_gradient_checkpoint_policy(
    name: Literal["everything_saveable", "nothing_saveable", "checkpoint_dots", "checkpoint_dots_with_no_batch_dims"] | Callable[..., bool],
):
    """把 remat 策略名称转换成 JAX checkpoint policy。"""

    if not isinstance(name, str):
        # 允许调用者直接传入自定义 policy 函数。
        return name
    match name:
        case "everything_saveable":
            return jax.checkpoint_policies.everything_saveable
        case "nothing_saveable":
            return jax.checkpoint_policies.nothing_saveable
        case "checkpoint_dots":
            return jax.checkpoint_policies.checkpoint_dots
        case "checkpoint_dots_with_no_batch_dims":
            return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
        case _:
            raise ValueError(f"Unknown policy: {name}")


def set_random_seed(seed: int) -> PRNGKeyArray:
    """同时设置 NumPy、Python random 和 JAX PRNG seed。"""

    np.random.seed(seed)
    random.seed(seed)

    return jrandom.PRNGKey(seed)


def get_custom_tqdm():
    """返回带训练速度日志的 tqdm 类。

    训练前 50 步通常包含编译和 warmup，之后再估算平均 step 时间会更接近真实吞吐。
    """

    logger = logging.getLogger("Custom TQDM Timing")
    logger.setLevel(logging.INFO)  # Set the logging level to INFO

    class tqdm(_tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.warmup_time_elapsed = 0

        def update(self, n=1):
            super().update(n)
            warmup_steps = 50
            step_passed = self.n - self.initial
            if step_passed == warmup_steps:
                # 记录 warmup 结束时的耗时，后续 ETA 会扣掉这部分。
                self.warmup_time_elapsed = self.format_dict["elapsed"]
                logger.info(f"Warmup {warmup_steps} Iteration Time: {self.format_interval(self.warmup_time_elapsed)}")
            if (step_passed > warmup_steps and step_passed % 100 == 0) or self.n == self.total:
                # NOTE: the starting up time is also included in the elapsed time
                elapsed = self.format_dict["elapsed"] - self.warmup_time_elapsed
                inv_rate = elapsed / (step_passed - warmup_steps)
                eta = (self.total - self.n) * inv_rate
                logger.info(
                    f"{self.n}/{self.total}: Average Speed: {inv_rate:.2f} s/it, elapsed: {self.format_interval(elapsed)} | remaining: {self.format_interval(eta)} "
                )

    return tqdm


def vmap_mean(fun, batch, *, axis_name: Hashable):
    """对 batch 的第一维 vmap 执行函数，并返回跨该维度平均后的结果。"""

    vmap_dim_size = jax.tree.flatten(batch)[0][0].shape[0]
    if vmap_dim_size == 1:
        # batch size 为 1 时跳过 vmap，减少一层无意义的并行轴。
        single_microbatch = tree_rearrange(batch, "1 ... -> ...")
        return fun(single_microbatch)

    @partial(jax.vmap, in_axes=(0,), out_axes=None, axis_name=axis_name)
    def vmapped_fn(x):
        return jax.lax.pmean(fun(x), axis_name=axis_name)

    return vmapped_fn(batch)


def welfords_online_mean(fun, batch):
    """
    逐个处理 batch 切片并在线求均值，避免把所有中间结果都存下来。
    数学上等价于 `mean([fun(x) for x in batch])`，但内存压力更小。

    Args:
        fun: 对单个切片执行的函数。
        batch: 第一维会被逐个 scan 的 pytree。

    Returns:
        fun 输出的在线平均结果。
    """
    num_loops = jax.tree.flatten(batch)[0][0].shape[0]
    if num_loops == 1:  # Skip if trivial
        single_microbatch = tree_rearrange(batch, "1 ... -> ...")
        return fun(single_microbatch)

    def update_online_grad_mean(carry, batch_slice):
        """Welford's online mean algorithm for stable numerics"""
        (acc_carry, count) = carry

        acc_delta = fun(batch_slice)

        acc_carry = jax.tree.map(lambda delta, acc: acc + (delta - acc) / count, acc_delta, acc_carry)

        return (acc_carry, count + 1), None

    first_batch_slice = jax.tree.map(lambda x: x[0], batch)

    acc_init = jax.tree.map(lambda x: jnp.zeros_like(x), jax.eval_shape(fun, first_batch_slice))
    count_init = 1

    (acc_result, _count), _ = lax.scan(update_online_grad_mean, (acc_init, count_init), batch)

    return acc_result


def scan_or_loop(
    f,
    init,
    xs,
    use_loop=False,
):
    """在 `jax.lax.scan` 和 Python for-loop 之间切换。

    scan 性能更好；Python loop 更适合调试 NaN 或查看逐步结果。
    """
    if not use_loop:
        return jax.lax.scan(f, init, xs)

    carry = init
    xs_size = jax.tree.leaves(xs)[0].shape[0]
    ys = []
    for i in range(xs_size):
        x = tree_slice(xs, i)
        carry, y = f(carry, x)
        ys.append(y)

    stack_args = lambda *args: jnp.stack(args) if not all(arg is None for arg in args) else None  # Nones should stack to None

    return carry, jax.tree.map(stack_args, *ys)


def scan_remat_chunk(f, carry, x, *, remat_n_loops: int, unroll: bool):
    """分块 scan，并可对每块使用 remat 节省反向传播显存。

    Args:
        f: 每个 chunk 上执行的函数。
        carry: scan 的初始状态。
        x: 输入 pytree，第一维是循环维度。
        remat_n_loops: 每多少步包一层 remat；0 表示不 remat。
        unroll: 是否改用 Python loop，主要用于调试。
    """

    num_loops = jax.tree.leaves(x)[0].shape[0]

    if remat_n_loops == 0:
        carry, y = scan_or_loop(f, carry, x, use_loop=unroll)
        return carry, y

    n_remat_chunks = num_loops // remat_n_loops

    # 先把长循环拆成若干 remat chunk，每个 chunk 内再 scan。
    x_grouped = tree_rearrange(x, "(remat_chunk remat_loops) ... -> remat_chunk remat_loops ...", remat_chunk=n_remat_chunks, remat_loops=remat_n_loops)

    @partial(jax.remat, prevent_cse=False, policy=get_gradient_checkpoint_policy("nothing_saveable"))
    def chunk_f(carry, x_chunk):
        return scan_or_loop(f, carry, x_chunk, use_loop=unroll)

    carry, result = scan_or_loop(chunk_f, carry, x_grouped, use_loop=unroll)

    result = tree_rearrange(result, "remat_chunk remat_loops ... -> (remat_chunk remat_loops) ...")
    return carry, result


def tree_slice[T: PyTree](tree: T, i: int) -> T:
    """对 pytree 中每个 leaf 取第 i 个元素。"""

    return jax.tree.map(lambda x: x[i], tree)


def tree_rearrange[T: PyTree](tree: T, pattern: str, **axes_lengths) -> T:
    """对 pytree 中每个 leaf 应用 einops.rearrange。"""

    def rearrange_fn(x):
        return rearrange(x, pattern, **axes_lengths)

    return jax.tree.map(rearrange_fn, tree)


def canonicalize_dtype(*args, dtype: Dtype | None = None, inexact: bool = True) -> Dtype:
    """根据输入和目标 dtype 推断最终计算 dtype。"""
    if dtype is None:
        args_filtered = [jnp.asarray(x) for x in args if x is not None]
        dtype = jnp.result_type(*args_filtered)
        if inexact and not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.promote_types(jnp.float32, dtype)
    if inexact and not jnp.issubdtype(dtype, jnp.inexact):
        raise ValueError(f"Dtype must be inexact: {dtype}")
    return dtype


def promote_dtype(*args, dtype=None, inexact=True) -> list[Any]:
    """把多个输入统一转换到同一个 dtype。"""
    dtype = canonicalize_dtype(*args, dtype=dtype, inexact=inexact)
    return [jnp.asarray(x, dtype) if x is not None else None for x in args]


def eval_shape_and_sharding(f, *args, **kwargs):
    """类似 `jax.eval_shape`，但额外保留编译后的输出 sharding 信息。"""

    f_jit = jax.jit(f)
    shapes = f_jit.eval_shape(*args, **kwargs)
    sharding = f_jit.lower(*args, **kwargs).compile().output_shardings

    def add_sharding(shapes, sharding):
        # Orbax restore 需要 shape 和 sharding 都可见，才能恢复到正确的分片布局。
        shapes.sharding = sharding
        return shapes

    return jax.tree.map(add_sharding, shapes, sharding)


_StaticArgs = tp.TypeVar("_StaticArgs")
_SavedArgs = tp.TypeVar("_SavedArgs", bound=PyTree)


def remat_bwd(
    fun: tp.Callable[..., tp.Any],
    *,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: Callable[..., bool] | None = None,
) -> Callable[..., tp.Any]:
    """只在反向传播路径上应用 remat。

    普通 `jax.remat` 会影响前向和反向；这个包装让调用者更细地控制反向里保存哪些中间值。

    Args:
        fun: 要包装的函数。
        prevent_cse: 是否阻止公共子表达式消除；scan 内部 remat 通常设为 False。
        static_argnums: 哪些参数是静态参数。
        policy: checkpoint 策略。
    """

    @wraps(fun)
    @jax.custom_vjp
    def standard_fn(*args):
        # 前向计算保持原函数语义。
        return fun(*args)

    @partial(jax.remat, prevent_cse=prevent_cse, policy=policy, static_argnums=static_argnums)
    def fwd_fn(*args):
        # 保存 VJP 函数作为 residual，反向时再计算梯度。
        output, vjp_compute_fn = jax.vjp(fun, *args)
        residuals = vjp_compute_fn
        return output, residuals

    @partial(jax.remat, prevent_cse=prevent_cse, policy=policy, static_argnums=static_argnums)
    def bwd_fn(residuals: _SavedArgs, g):
        dl_d_output = g
        vjp_compute_fn = residuals
        d_saved_args = vjp_compute_fn(dl_d_output)
        return d_saved_args

    standard_fn.defvjp(fwd_fn, bwd_fn)

    standard_fn = jax.remat(standard_fn, prevent_cse=prevent_cse, policy=policy, static_argnums=static_argnums)

    return standard_fn


def clone_pytree(tree: PyTree):
    """复制 pytree 结构，但保留 leaf 值。

    主要用于 Equinox State。我们想复用 state 时，重新创建结构可以避免 get/set 过程中的状态失效问题。
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    tree_clone = jax.tree_util.tree_unflatten(treedef, leaves)
    return tree_clone


def maybe_remat(
    fun: Callable,
    *,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: str,
) -> Callable[..., bool]:
    """当 policy 非空时包一层 `jax.remat`，否则原样返回函数。"""
    if policy:
        return jax.remat(fun, prevent_cse=prevent_cse, static_argnums=static_argnums, policy=get_gradient_checkpoint_policy(policy))
    else:
        return fun


@wraps(remat_bwd)
def maybe_remat_bwd(
    fun: tp.Callable[..., tp.Any],
    *,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: str,
) -> Callable:
    """当 policy 非空时只给反向传播包 remat。"""

    if policy:
        return remat_bwd(fun, prevent_cse=prevent_cse, static_argnums=static_argnums, policy=get_gradient_checkpoint_policy(policy))
    else:
        return fun


def maybe_double_remat(
    fun: Callable,
    *,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy_remat: str,
    policy_remat_bwd: str,
) -> Callable:
    """按配置同时应用普通 remat 和 backward-only remat。"""

    return maybe_remat_bwd(
        fun=maybe_remat(
            fun,
            prevent_cse=prevent_cse,
            static_argnums=static_argnums,
            policy=policy_remat,
        ),
        prevent_cse=prevent_cse,
        static_argnums=static_argnums,
        policy=policy_remat_bwd,
    )


def safe_sqrt(x, eps=1e-5):
    """带 epsilon 的 sqrt，避免输入接近 0 时数值不稳定。"""
    return jnp.sqrt(x + eps)


def global_norm_safe(updates):
    """Compute the global norm across a nested structure of tensors."""
    return safe_sqrt(sum(jnp.sum(numerics.abs_sq(x)) for x in jax.tree.leaves(updates)))
