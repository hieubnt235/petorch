from abc import ABC, abstractmethod
from typing import (
    Sequence,
    TYPE_CHECKING,
    cast,
)

import torch
from lightning import Callback
from lightning.pytorch.loggers import (
    WandbLogger,
    NeptuneLogger,
)
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from petorch import logger
from petorch.experiments.loggers import CometLogger, ClearmlLogger

if TYPE_CHECKING:
    import lightning as pl

    cast(type, pl.Trainer)


class OutputEvaluationCallback(Callback, ABC):

    def __init__(self, *args, every_n_epochs: int, **kwargs) -> None:
        self.every_n_epochs = every_n_epochs
        self.gen_args = args
        self.gen_kwargs = kwargs

    @abstractmethod
    def gen_and_store_on_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *args, **kwargs
    ) -> None:
        pass

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.is_global_zero and trainer.current_epoch % self.every_n_epochs == 0:
            logger.info(
                f"{self.__class__.__name__}: Generating and storing output at step={trainer.global_step}..."
            )

            with torch.no_grad():
                self.gen_and_store_on_epoch_end(
                    trainer, pl_module, *self.gen_args, **self.gen_kwargs
                )


class ImageOutputWandbCallback(OutputEvaluationCallback):

    def __init__(
        self, *args, every_n_epoch: int = 5, wandb_logger: WandbLogger = None, **kwargs
    ):
        """
        Args:
            wandb_logger: If None, try to find logger from Trainer.

        """
        super().__init__(every_n_epoch=every_n_epoch, *args, **kwargs)
        self.wdb_logger = wandb_logger

    def gen_and_store_on_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *args, **kwargs
    ) -> None:
        wdb_logger = self.wdb_logger
        if wdb_logger is None:
            for pl_logger in trainer.loggers:
                if isinstance(pl_logger, WandbLogger):
                    wdb_logger = pl_logger
                    break

        if wdb_logger is not None:
            assert isinstance(wdb_logger, WandbLogger)
            images = pl_module.forward(*args, **kwargs)
            wdb_logger.log_image(f"images_step_{trainer.global_step}", images)


class ImageOutputNeptuneCallback(OutputEvaluationCallback):

    def __init__(
        self,
        *args,
        every_n_epoch: int = 5,
        neptune_logger: NeptuneLogger = None,
        **kwargs,
    ):
        """
        Args:
            neptune_logger: If None, try to find logger from Trainer.

        """
        super().__init__(every_n_epoch=every_n_epoch, *args, **kwargs)
        self.nt_logger = neptune_logger

    def gen_and_store_on_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *args, **kwargs
    ) -> None:
        nt_logger = self.nt_logger
        if nt_logger is None:
            for pl_logger in trainer.loggers:
                if isinstance(pl_logger, NeptuneLogger):
                    nt_logger = pl_logger
                    break

        if nt_logger is not None:
            assert isinstance(nt_logger, NeptuneLogger)
            images = pl_module.forward(*args, **kwargs)
            nt_logger.experiment[f"output_images/step_{trainer.global_step}"] = images


class ImageOutputCometCallback(OutputEvaluationCallback):

    def __init__(
        self, *args, every_n_epochs: int = 5, comet_logger: CometLogger = None, **kwargs
    ):
        """
        Args:
            comet_logger: If None, try to find logger from Trainer.

        """
        super().__init__(every_n_epochs=every_n_epochs, *args, **kwargs)
        self.cm_logger = comet_logger

    def gen_and_store_on_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *args, **kwargs
    ) -> None:
        cm_logger = self.cm_logger
        if cm_logger is None:
            for pl_logger in trainer.loggers:
                if isinstance(pl_logger, CometLogger):
                    cm_logger = pl_logger
                    break

        if cm_logger is not None:
            assert isinstance(cm_logger, CometLogger)

            images = pl_module.forward(*args, **kwargs)
            images = images if isinstance(images, Sequence) else [images]
            for i, img in enumerate(images):
                cm_logger.experiment.log_image(
                    img, f"output_images_step_{trainer.global_step}_{i}"
                )


class ImageOutputClearmlCallback(OutputEvaluationCallback):
    def __init__(
        self,
        *args,
        every_n_epochs: int = 5,
        clearml_logger: ClearmlLogger = None,
        **kwargs,
    ):
        """
        Args:
            comet_logger: If None, try to find logger from Trainer.

        """
        super().__init__(every_n_epochs=every_n_epochs, *args, **kwargs)
        self.cml_logger = clearml_logger

    def gen_and_store_on_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *args, **kwargs
    ) -> None:
        cml_logger = self.cml_logger
        if cml_logger is None:
            for pl_logger in trainer.loggers:
                if isinstance(pl_logger, ClearmlLogger):
                    cml_logger = pl_logger
                    break

        if cml_logger is not None:
            assert isinstance(cml_logger, ClearmlLogger)

            images = pl_module.forward(*args, **kwargs)
            images = images if isinstance(images, Sequence) else [images]
            for i, img in enumerate(images):
                cml_logger.experiment.logger.report_image(
                    f"output_images_step_{trainer.global_step}_{i}",
                    f"image_output_evaluation",
                    iteration=trainer.global_step,
                    image=img,
                )
