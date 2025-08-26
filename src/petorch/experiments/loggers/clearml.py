import pprint
from argparse import Namespace
from typing import (
    Any,
    AnyStr,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    Sequence,
    cast,
    TYPE_CHECKING,
)
from uuid import uuid4

import pandas as pd
import torch
from clearml import Task
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch import Tensor

from petorch import logger
from petorch.utilities import b64encode

if TYPE_CHECKING:
    from comet_ml import (
        ExistingExperiment,
        Experiment,
        OfflineExperiment,
        APIExperiment,
    )


# https://github.com/Lightning-AI/pytorch-lightning/pull/14139
class ClearMLLogger(pl_loggers.Logger):
    """Log using `ClearML <https://clear.ml>`_.
    Install it with pip:
    .. code-block:: bash
        pip install clearml
    .. code-block:: python
        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import ClearMLLogger
        cml_logger = ClearMLLogger(project_name="lightning-project", task_name="task-name")
        trainer = Trainer(logger=cml_logger)
    Use the logger anywhere in your :class:`~pytorch_lightning.core.module.LightningModule` as follows:
    .. code-block:: python
        from pytorch_lightning import LightningModule
        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # example
                self.logger.experiment.whatever_clear_ml_supports(...)
            def any_lightning_module_function_or_hook(self):
                self.logger.experiment.whatever_clear_ml_supports(...)
    Args:
        project_name: Name of the ClearML project
        task_name: Name of the ClearML task
        task_id: Optional ID of an existing ClearML task to be reused.
    Raises:
        ModuleNotFoundError:
            If required ClearML package is not installed on the device.
    """
    #todo: save adapter only one, and support load state dict with adapter name.in format {adapter_name}.lora_A
    def __init__(self,
                 task_id: Optional[str] = None,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        allow_archived: bool = True,
        task_filter: Optional[dict] = None,
):
        super().__init__()
        self.project_name = project_name
        self.task_name = task_name
        self.id = task_id
        self._step = 0

        if not self.id:
            self._initialized = True
            self.task = Task.init(
                project_name=self.project_name, task_name=self.task_name
            )
        else:
            self._initialized = True
            self.task = Task.get_task(task_id=self.id)

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, Union[float, Tensor]], step: Optional[int] = None
    ) -> None:
        if step is None:
            step = self._step

        def _handle_value(value: Union[float, Tensor]):
            if isinstance(value, Tensor):
                return value.item()
            return value

        for metric, value in metrics.items():
            self.task.logger.report_scalar(
                title=metric, series=metric, value=_handle_value(value), iteration=step
            )

        self._step += 1

    @rank_zero_only
    def log_hyperparams(
        self, params: Union[Dict[str, AnyStr], Namespace], *args: Any, **kwargs: Any
    ) -> None:
        self.task.connect(params, *args, *kwargs)

    @rank_zero_only
    def log_table(
        self,
        key: str,
        columns: List[str] = None,
        data: List[List[Any]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
    ) -> None:

        table: Optional[Union[pd.DataFrame, List[List[Any]]]] = None

        if dataframe is not None:
            table = dataframe
            if columns is not None:
                table.columns = columns

        if data is not None:
            table = data
            assert len(columns) == len(
                table[0]
            ), "number of column names should match the total number of columns"
            table.insert(0, columns)

        if table is not None:
            self.task.logger.report_table(
                title=key, series=key, iteration=step, table_plot=table
            )

    @rank_zero_only
    def finalize(
        self, status: Literal["success", "failed", "aborted"] = "sucess"
    ) -> None:
        """Finalize the experiment. Mark the task completed or otherwise given the status.
        Args:
            status: Status that the experiment finished with (e.g. success, failed, aborted)
        """

        if status == "success":
            self.task.mark_completed()
        elif status == "failed":
            self.task.mark_failed()
        elif status == "aborted":
            self.task.mark_stopped()

    @property
    def name(self) -> Optional[str]:
        """Gets the name of the experiment, being the name of the ClearML task
        Returns:
            The name of the ClearML task
        """
        return self.task.name

    @property
    def version(self) -> str:
        """Gets the version of the experiment, being the ID of the ClearML task.
        Returns:
            The id of the ClearML task
        """
        return self.task.id
