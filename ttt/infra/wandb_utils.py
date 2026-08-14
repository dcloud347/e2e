"""Weights & Biases 日志封装。

训练代码只通过 `WandbLogger` 写日志；这个类负责处理多进程只在主进程登录、创建或恢复 run。
"""

import logging
from pathlib import Path

import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import numpy as np
from hydra.core.hydra_config import HydraConfig

from ttt.config import TrainingConfig
from ttt.utils.jax_utils import master_log

LoadPart = TrainingConfig.LoadPart
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WandbLogger:
    """管理一个 W&B run 的初始化、恢复和日志写入。"""

    def __init__(
        self,
        entity: str,
        project: str,
        exp_name: str,
        load_part: LoadPart,
        log_dir: Path,
        wandb_key: str,
        logging_process: int,
        config: dict = None,
        enabled: bool = True,
    ):
        """初始化 W&B。

        非主进程不会真正写日志，但会通过 broadcast 得知是否已有同名 run。
        """

        import wandb
        from wandb.sdk.wandb_settings import Settings

        self.wandb = wandb
        self.is_master = jax.process_index() == logging_process
        self.entity = entity
        self.project = project
        self.exp_name = exp_name
        self.load_part = load_part
        self.enabled = enabled
        self.run = None
        self.log_dir = log_dir

        # Settings for wandb.init()
        self.wandb_settings = Settings(
            api_key=wandb_key,
            entity=self.entity,
            project=self.project,
        )

        if self.is_master:
            # 只在主进程访问 W&B API，避免多机训练时重复创建 run。
            # Pass API key directly to Api()
            wandb.login(key=wandb_key)
            api = wandb.Api(api_key=wandb_key)
            runs = api.runs(f"{self.entity}/{self.project}", filters={"display_name": self.exp_name})
            num_existing = len(runs)
        else:
            num_existing = -1

        num_existing = jax.experimental.multihost_utils.broadcast_one_to_all(jnp.asarray(num_existing), self.is_master).item()
        self.preexisting = num_existing > 0

        if self.is_master and self.enabled:
            if num_existing == 0:
                # 新实验：把 Hydra overrides 也写进 W&B config，方便复现。
                config["overrides"] = list(HydraConfig.get().overrides.task) + list(HydraConfig.get().overrides.hydra)
                self.run = wandb.init(project=self.project, entity=self.entity, name=self.exp_name, config=config, settings=self.wandb_settings)
                master_log(logger, f"Initialized new run: {self.run.name} (ID: {self.run.id})")
            else:
                # 同名 run 已存在时认为是在恢复训练。
                if num_existing > 1:
                    runs = sorted(runs, key=lambda r: r.created_at, reverse=True)
                    master_log(logger, f"Warning: Multiple runs found with name '{self.exp_name}'. Using the latest run: {runs[0].id}")
                self.run = runs[0]
                resumed_run = wandb.init(project=self.project, entity=self.entity, id=self.run.id, resume="allow", settings=self.wandb_settings)
                master_log(logger, f"Resumed existing run: {resumed_run.name} (ID: {resumed_run.id})")

    def log(self, metrics: dict, step: int):
        """按 step 写入标量或 W&B 对象。"""

        if self.is_master and self.enabled:
            self.wandb.log(metrics, step=step)

    def save(self, path: str | Path, base_path: str | Path = "./"):
        """把本地文件登记到 W&B run。"""

        if self.is_master and self.enabled:
            self.wandb.save(path, base_path=base_path)

    def log_token_nll_loss(self, token_nll_loss: np.ndarray, step: int, k: str):
        """把 token 级别 NLL 画成折线图写到 W&B。"""

        if not self.is_master or not self.enabled:
            return
        import wandb

        if token_nll_loss.ndim == 1:
            # 统一成二维，方便逐行画图。
            token_nll_loss = token_nll_loss[np.newaxis, ...]

        for r, row in enumerate(token_nll_loss):
            table = wandb.Table(data=[(step, i, tloss) for i, tloss in enumerate(row)], columns=["step", "token", "token_nll_loss"])
            self.log(
                {
                    f"{k}/token_nll_loss(before gs {r})": wandb.plot.line(
                        table, "token", "token_nll_loss", "step", title=f"Token NLL loss at step {step} (before gs {r})", split_table=True
                    )
                },
                step,
            )
