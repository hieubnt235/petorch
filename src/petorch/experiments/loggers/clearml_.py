import os
import tempfile
from argparse import Namespace
from pathlib import Path
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

from clearml.backend_interface.task.models import TaskModels
from clearml.model import Framework
from lightning.pytorch.utilities.model_summary import summarize
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch import Tensor
from torch.nn import Module
from .base import BaseLogger

import clearml as cml
from lightning import LightningModule, pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer

from ... import logger

Task = cml.Task
TaskTypes = cml.TaskTypes


class ClearMLLogger(BaseLogger):
    """
    References:
        https://github.com/bmartinn/pytorch-lightning/blob/6517ca874baf31ea1ff2249b47695a9f4aee080e/pytorch_lightning/loggers/trains.py
        https://github.com/Aleksandar1932/lightning/blob/49cad2e88a6b4787f1dd1d5d5363b8b328b828d5/src/pytorch_lightning/loggers/clearml.py

    """

    def __init__(
        self,
        log_model_checkpoint: bool = True,
        init_new: bool = True,
        *,
        project_name: str | None = None,
        task_name: str | None = None,
        task_type: TaskTypes = TaskTypes.training,
        task_id: str | None = None,
        tags: Sequence[str] | None = None,
        **comet_task_kwargs,
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

        self._log_model_checkpoint = log_model_checkpoint
        self._output_models: dict[str, cml.OutputModel] = {}
        """filepath: OutputModel"""

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
                **self._comet_task_kwargs,
            )
        else:
            self._task = Task.get_task(
                self._task_id,
                self._project_name,
                self._task_name,
                self._tags,
                **self._comet_task_kwargs,
            )

    @property
    def experiment(self) -> Task:
        """For clearml convention, should use this instead of Task.current_task() for init exactly with the same expected configs.
        Exactly the same object with `self.task`.

        See Also:
            https://clear.ml/docs/latest/docs/references/sdk/task (The first INFO)
        """

        if self._task is None:
            self._init_task()
        return self._task

    @property
    def task(self) -> Task:
        """For clearml convention, should use this instead of Task.current_task() for init exactly with the same expected configs.
        Exactly the same object with `self.experiment`

        See Also:
            https://clear.ml/docs/latest/docs/references/sdk/task (The first INFO)
        """
        return self.experiment

    @property
    def task_models(self) -> TaskModels:
        return cast(TaskModels, self.task.models)

    @property
    def task_output_models(self) -> dict[str, cml.Model]:
        """
        The readonly task model outputs.
        Returns:
            dict-like with keys are model.name, values are cml.Model
        """
        return cast(dict[str, cml.Model], self.task_models.output)

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

    def _filepath_to_model_name(self, path: str) -> str:
        pass

    def _filepath_to_repo_path(self, local_path: str) -> str:
        pass

    @rank_zero_only
    def after_save_checkpoint(
        self,
        checkpoint_callback: ModelCheckpoint,
    ) -> None:
        if self._log_model_checkpoint:

            saved_filepath = checkpoint_callback._last_checkpoint_saved
            best_k_models = checkpoint_callback.best_k_models
            last_model_path = checkpoint_callback.last_model_path

            assert Path(saved_filepath).exists()

            feat: dict[str, Any] = {
                "filepath": saved_filepath,
            }
            if saved_filepath in best_k_models:
                assert checkpoint_callback.monitor is not None
                feat[checkpoint_callback.monitor] = best_k_models[saved_filepath].item()
                tags = [
                    "best_k",
                    checkpoint_callback.monitor,
                    best_k_models[saved_filepath].item(),
                ]
            elif saved_filepath == last_model_path:
                feat["last"] = True
                tags = ["last"]
            else:
                logger.warning(
                    f"The saved file path `{saved_filepath}' is not the best k or the last. "
                    f"This is unexpected behavior, does not log any model."
                )
                return

            # If it does not have the saved model, add it.
            if saved_filepath not in self._output_models:
                self._output_models[saved_filepath] = cml.OutputModel(
                    self.task,
                    name=self._filepath_to_model_name(saved_filepath),
                    framework=Framework.pytorch,
                )

            # Update the weights
            try:
                self._output_models[saved_filepath].update_weights(
                    saved_filepath,
                    target_filename=self._filepath_to_repo_path(saved_filepath),
                    iteration=checkpoint_callback._last_global_step_saved or None,
                    async_enable=False,  # Block.
                )
                self._output_models[saved_filepath].comment = feat
                self._output_models[saved_filepath].tags = tags
                logger.info(f"Loaded new checkpoint successfully:\n {saved_filepath}")
                
            except Exception as e:
                logger.error(e)

            hold_filepaths = []
            hold_filepaths.extend(checkpoint_callback.best_k_models.keys())
            if checkpoint_callback.last_model_path:
                hold_filepaths.append(checkpoint_callback.last_model_path)
            assert (
                saved_filepath in hold_filepaths
            )  # The saved_filepath must the last or the part of best_k.

            logger.debug(f"Hold filepaths:\n { "\n".join(hold_filepaths)}")

            # Remove the olds
            filepath_keys = self._output_models.keys()
            for filepath in filepath_keys:
                if filepath not in hold_filepaths:
                    try:
                        model = self._output_models[filepath]
                        flag = cml.Model.remove(
                            model.id, force=True, raise_on_errors=True
                        )
                        if not flag:
                            raise ValueError(
                                f"Remove model fails (partial remove): `name={model.name}`, `id={model}`."
                            )
                    except ValueError as e:
                        logger.error(e)
                    self._output_models.pop(filepath)

    def after_remove_checkpoint(self, filepath: str) -> None:
        pass

    def __getstate__(self) -> Union[str, None]:
        if self.experiment:
            return self.experiment.id

    def __setstate__(self, state: str) -> None:
        if state:
            self._task_id = state
            self._init_new = False
            self._init_task()
