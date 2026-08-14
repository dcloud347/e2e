"""checkpoint 保存与恢复。

Orbax 负责保存模型权重和优化器状态；这里额外实现了 Grain 数据迭代器的保存，
这样恢复训练时可以继续从相同的数据进度往后跑。
"""

import dataclasses
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

import equinox as eqx
import grain.python as grain
import jax
import orbax.checkpoint as ocp
from etils import epath
from grain._src.python import data_loader
from grain._src.python.dataset import dataset
from omegaconf import OmegaConf
from optax import OptState
from orbax.checkpoint import options as ocp_opt

from ttt.config import Config, TrainingConfig
from ttt.model.transformer import MetaModel

IteratorType = TypeVar("IteratorType", data_loader.DataLoaderIterator, dataset.DatasetIterator)


class CustomPyGrainCheckpointHandler(grain.PyGrainCheckpointHandler):
    """让 Orbax 能保存和恢复 PyGrain iterator 的自定义 handler。"""

    def save(
        self,
        directory: epath.Path,
        args: Any = None,
    ):
        """把 iterator 的进度状态写成 JSON 文件。"""

        item = args.item
        if isinstance(item, dataset.DatasetIterator):
            # MapDataset iterator 的状态本身就是 JSON 可序列化对象。
            state = json.dumps(item.get_state(), indent=4)
        else:
            # DataLoaderIterator 返回 bytes，需要转成字符串存储。
            state = item.get_state().decode()
        filename = directory / "global_batch_progress.json"

        if jax.process_index() == 0:
            # 多进程训练中只让主进程写这个小文件。
            filename.write_text(state)

    def restore(
        self,
        directory: epath.Path,
        args: Any = None,
    ) -> IteratorType:
        """从 JSON 文件恢复 iterator 进度。"""

        item = args.item
        filename = directory / "global_batch_progress.json"
        if not filename.exists():
            raise ValueError(f"File {filename} does not exist.")
        state = filename.read_text()
        if isinstance(item, dataset.DatasetIterator):
            state = json.loads(state)
        else:
            state = state.encode()
        item.set_state(state)
        return item


@ocp.args.register_with_handler(CustomPyGrainCheckpointHandler, for_save=True)
@dataclasses.dataclass
class CustomPyGrainCheckpointSave(ocp.args.CheckpointArgs):
    """保存 Grain iterator 时传给 Orbax 的参数容器。"""

    item: Any


@ocp.args.register_with_handler(CustomPyGrainCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class CustomPyGrainCheckpointRestore(ocp.args.CheckpointArgs):
    """恢复 Grain iterator 时传给 Orbax 的参数容器。"""

    item: Any


class Checkpointer:
    """训练 checkpoint 管理器。

    根据 `for_saving` 决定使用当前实验目录，还是使用 resume 实验目录。
    """

    def __init__(self, config: Config, for_saving: bool = True):
        self.config = config

        if for_saving:
            checkpoint_path = config.checkpoint.checkpoint_dir
        else:
            checkpoint_path = config.checkpoint.resume_checkpoint_dir

        if not checkpoint_path.startswith("gs://"):
            # 本地路径转成绝对路径；GCS 路径保持 gs:// 形式。
            checkpoint_path = Path(checkpoint_path).resolve()

        # 显式注册每个 item 的保存/恢复 handler。
        handler_registry = ocp.DefaultCheckpointHandlerRegistry()
        handler_registry.add("train_ds_iter", CustomPyGrainCheckpointRestore, CustomPyGrainCheckpointHandler)
        handler_registry.add("train_ds_iter", CustomPyGrainCheckpointSave, CustomPyGrainCheckpointHandler)
        handler_registry.add("opt_state", ocp.args.StandardRestore, ocp.StandardCheckpointHandler)
        handler_registry.add("opt_state", ocp.args.StandardSave, ocp.StandardCheckpointHandler)
        handler_registry.add("model_weights", ocp.args.StandardRestore, ocp.StandardCheckpointHandler)
        handler_registry.add("model_weights", ocp.args.StandardSave, ocp.StandardCheckpointHandler)

        mp_opts = ocp_opt.MultiprocessingOptions(primary_host=0)
        ckpt_opts = ocp.CheckpointManagerOptions(multiprocessing_options=mp_opts)

        self.manager = ocp.CheckpointManager(
            checkpoint_path,
            options=ckpt_opts,
            handler_registry=handler_registry,
        )

    def save_checkpoint(self, step: int, model: MetaModel, opt_state: OptState, train_ds_iter, is_milestone: bool = False):
        """保存模型权重、优化器状态和训练数据迭代器进度。"""

        model_weights = model.weights()

        self.manager.save(
            step=step,
            args=ocp.args.Composite(
                opt_state=ocp.args.StandardSave(opt_state),
                model_weights=ocp.args.StandardSave(model_weights),
                train_ds_iter=CustomPyGrainCheckpointSave(train_ds_iter),
            ),
            force=is_milestone,
        )

    def checkpoint_exists(self) -> bool:
        """当前 checkpoint 目录下是否已经有可恢复的 step。"""

        return self.manager.latest_step() is not None

    def load_checkpoint(self, targets, restore: TrainingConfig.LoadPart, step=None):
        """按 restore 策略恢复 checkpoint。

        `params` 只恢复模型参数；`all` 会同时恢复优化器和数据迭代器。
        """

        if step is None:
            step = self.manager.latest_step()

        if step is None:
            raise FileNotFoundError(f"No checkpoints found at {self.manager.directory}")

        model_weights_metadata = self.manager.item_metadata(step)["model_weights"]
        # 根据当前模型结构补齐 Orbax restore target，并保持 shard/shape 信息。
        model_weights_target = fetch_from_eqx_module(model_weights_metadata, targets["model_weights"])[0]

        if restore == TrainingConfig.LoadPart.all:
            opt_state_metadata = self.manager.item_metadata(step)["opt_state"]
            opt_state_target = fetch_from_eqx_module(opt_state_metadata, targets["opt_state"])[0]

            restored = self.manager.restore(
                step=step,
                args=ocp.args.Composite(
                    opt_state=ocp.args.StandardRestore(opt_state_target),
                    model_weights=ocp.args.StandardRestore(model_weights_target),
                    train_ds_iter=CustomPyGrainCheckpointRestore(targets["train_ds_iter"]),
                ),
            )
            return {
                "opt_state": restored["opt_state"],
                "model_weights": restored["model_weights"],
                "train_ds_iter": restored["train_ds_iter"],
            }
        elif restore == TrainingConfig.LoadPart.params:
            restored = self.manager.restore(
                step=step,
                args=ocp.args.Composite(
                    model_weights=ocp.args.StandardRestore(model_weights_target),
                ),
            )
            return {"model_weights": restored["model_weights"]}
        else:
            raise ValueError(f"Invalid restore option: {restore:r}")

    def wait_until_finished(self):
        """等待异步 checkpoint 写入完成。"""

        self.manager.wait_until_finished()

    def close(self):
        """关闭 manager，并确保后台保存任务结束。"""

        self.manager.close()


def make_save_checkpoint(
    checkpointer,
    gather_fns,
    model_config,
):
    """旧版保存函数包装器。

    当前主训练循环直接使用 `Checkpointer.save_checkpoint`，这个函数保留给兼容旧调用方式。
    """

    def save_checkpoint(train_state, train_loader, milestone=False, train_state_name=None):
        """把旧版 train_state 格式保存成 checkpoint。"""

        step = int(jax.device_get(train_state["step"]))
        metadata = dict(step=step, model_config=OmegaConf.to_container(model_config))
        sampler_state_dict = {
            "random_state": train_loader.sampler.state_dict()["random_state"],
            "counter": train_loader.sampler.state_dict()["counter"],
            "shuffle_log": train_loader.sampler.state_dict()["shuffle_log"],
        }
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=deepcopy(sampler_state_dict),
            milestone=milestone,
            train_state_name=train_state_name,
        )

    return save_checkpoint


M = TypeVar("M", bound=eqx.Module)


def unify_dict_with_eqx_module[M: eqx.Module](d: dict, module: M) -> tuple[M, list[str]]:
    """把 checkpoint 字典里的值填回 Equinox module。

    checkpoint 和 module 的树结构应基本一致；找不到的值会保留 module 原值。

    Args:
        d: Orbax 恢复出来的权重字典。
        module: 当前代码创建出的 Equinox module。

    Returns:
        填入 checkpoint 值后的 module，以及没有匹配到的路径。
    """
    from jax._src.lib import pytree

    weights_map = {p: v for p, v in jax.tree.flatten_with_path(d)[0]}  # list -> dict: {keypath: array}

    not_found_paths = []

    def find_weight(path, value):
        # Orbax 字典路径使用 DictKey，Equinox module 路径使用 GetAttrKey，需要互相转换。
        dict_path = tuple(pytree.DictKey(p.name) if isinstance(p, pytree.GetAttrKey) else p for p in path)
        if dict_path in weights_map:
            return weights_map[dict_path]
        else:
            not_found_paths.append(jax.tree_util.keystr(path))
            return value

    new_module = jax.tree.map_with_path(find_weight, module)

    if not_found_paths:
        import warnings

        warnings.warn(f"Could not find the following paths in the dictionary: {not_found_paths}")

    return new_module, not_found_paths


def fetch_from_eqx_module[M: eqx.Module](d: dict, module: M) -> tuple[M, list[str]]:
    """按 checkpoint 元数据结构，从当前 module 中取出 restore target。"""

    from jax._src.lib import pytree

    eqx_map = {p: v for p, v in jax.tree.flatten_with_path(module)[0]}  # list -> dict: {keypath: array}

    not_found_paths = []

    def find_weight(path, value):
        # 这里方向和 unify 相反：把 DictKey 转成 GetAttrKey 后去当前 module 里查。
        dict_path = tuple(pytree.GetAttrKey(p.key) if isinstance(p, pytree.DictKey) else p for p in path)
        if dict_path in eqx_map:
            new_value = eqx_map[dict_path]
            assert new_value.shape == value.shape, f"Shape mismatch for {jax.tree_util.keystr(path)}: {new_value.shape} != {value.shape}"
            return new_value
        else:
            not_found_paths.append(jax.tree_util.keystr(path))
            return value

    new_dict = jax.tree.map_with_path(find_weight, d)
    if not_found_paths:
        import warnings

        warnings.warn(f"Could not find the following paths in the checkpoint module: {not_found_paths}")
    return new_dict, not_found_paths
