import os
import tempfile
from argparse import Namespace
from typing import (
    Any,
    AnyStr,
    Dict,
    Literal,
    Optional,
    Union,
    Sequence,
    TYPE_CHECKING,
    cast,
)

import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.utilities.model_summary import summarize
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch import Tensor
from torch.nn import Module
from .base import BaseLogger

if TYPE_CHECKING:
    from clearml import Task, TaskTypes
    from lightning import LightningModule, pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning import Trainer


class ClearMLLogger(BaseLogger):
    """
    References:
        https://github.com/bmartinn/pytorch-lightning/blob/6517ca874baf31ea1ff2249b47695a9f4aee080e/pytorch_lightning/loggers/trains.py
        https://github.com/Aleksandar1932/lightning/blob/49cad2e88a6b4787f1dd1d5d5363b8b328b828d5/src/pytorch_lightning/loggers/clearml.py

    """

    def __init__(
        self,
        project_name: str | None = None,
        task_name: str | None = None,
        task_type: TaskTypes = TaskTypes.training,
        task_id: str | None = None,
        tags: Sequence[str] | None = None,
        *,
        init_new: bool = True,
        **comet_task_kwargs
    ):
        """

        Args:
            project_name:
            task_name:
            task_type:
            task_id: The existed task id. Only be used when `init_new` = False
            tags:
            init_new: If False, use Task.get_task to get the old task.
        """
        self._project_name = project_name
        self._task_name = task_name
        self._task_type = task_type
        self._task_id = task_id
        self._tags = tags
        self._init_new = init_new
        self._comet_task_kwargs = comet_task_kwargs

        # This pattern adapts to some lightning loggers to ensure experiment created for all cases,
        # such as when strategy=ddp_spawn.
        self._task: Task | None = None
        self._init_task()

    def _init_task(self):
        from clearml import Task

        if self._init_new:
            self._task = Task.init(
                project_name=self._project_name,
                task_name=self._task_name,
                task_type=self._task_type,
                tags=self._tags,
                auto_connect_frameworks={"pytorch": False},
                **self._comet_task_kwargs
            )
        else:
            self._task = Task.get_task(
                self._task_id,
                self._project_name,
                self._task_name,
                self._tags,
                **self._comet_task_kwargs
            )

    @property
    def experiment(self) -> Task:
        if self._task is None:
            self._init_task()
        return self._task

    @property
    def name(self) -> Optional[str]:
        return self.experiment.name

    @property
    def version(self) -> str:
        return self.experiment.id

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, Union[float, Tensor]], step: Optional[int] = None
    ) -> None:
        # clearml logger.report_scalar requires value to be float or int.
        for metric, value in metrics.items():
            self.experiment.logger.report_scalar(
                title=metric,
                series=metric,
                value=value.item() if isinstance(value, Tensor) else value,
                iteration=step,
            )

    @rank_zero_only
    def log_hyperparams(
        self, params: Union[Dict[str, AnyStr], Namespace], *args: Any, **kwargs: Any
    ) -> None:
        self.experiment.connect(
            params if isinstance(params, dict) else vars(params), *args, *kwargs
        )

    @rank_zero_only
    def log_graph(self, model: Module, input_array: Optional[Tensor] = None) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
            tmp.write(str(summarize(cast(LightningModule, model))))
            tmp_path = tmp.name

        self.experiment.upload_artifact("model_summary.txt", tmp_path)
        os.remove(tmp_path)

    @rank_zero_only
    def finalize(
        self, status: Literal["success", "failed", "aborted"] = "success"
    ) -> None:
        """Finalize the experiment. Mark the task completed or otherwise given the status.
        Args:
            status: Status that the experiment finished with (e.g. success, failed, aborted)
        """

        self.experiment.flush(True)
        if status == "success":
            self.experiment.close()  # Not use mark_complete, which will terminate the process.
        elif status == "failed":
            self.experiment.mark_failed()
        elif status == "aborted":
            self.experiment.mark_stopped()
        self._task = None

    def after_save_checkpoint(
        self, checkpoint_callback: ModelCheckpoint, trainer: pl.Trainer, filepath: str
    ) -> None:
        pass

    def after_remove_checkpoint(
        self, checkpoint_callback: ModelCheckpoint, trainer: pl.Trainer, filepath: str
    ) -> None:
        pass

    def __getstate__(self) -> Union[str, None]:
        if self.experiment:
            return self.experiment.id

    def __setstate__(self, state: str) -> None:
        if state:
            self._task_id = state
            self._init_new = False
            self._init_task()
