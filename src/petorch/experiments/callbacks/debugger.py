from abc import ABC, abstractmethod
from typing import (
    Sequence,
    TYPE_CHECKING,
    cast, Any, Union,
)

import numpy as np
import torch
from PIL import Image
from lightning import Callback
from lightning.pytorch.loggers import (
    WandbLogger,
    NeptuneLogger,
)
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.optim import Optimizer

from petorch import logger
from petorch.experiments.loggers import CometLogger, ClearmlLogger

if TYPE_CHECKING:
    import lightning as pl

    cast(type, pl.Trainer)


class SampleDebugger(Callback, ABC):

    def __init__(self, *args, every_n_epochs: int|None=1, every_n_train_steps:int|None=None, gen_method:str|None=None, **kwargs) -> None:
        assert every_n_epochs is None or every_n_epochs>=1
        assert every_n_train_steps is None or every_n_train_steps>=1
        assert every_n_epochs or every_n_train_steps

        self.every_n_epochs = int(every_n_epochs) if every_n_epochs is not None else None
        self.every_n_train_steps = int(every_n_train_steps) if every_n_train_steps is not None else None

        self.gen_method = gen_method or "__call__"
        self.gen_args = args
        self.gen_kwargs = kwargs

    @abstractmethod
    def store_sample(
        self, sample: Any, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """

        Args:
            sample: Output of getattr(lightning_module, self.gen_method)(*self.gen_args, self.**gen_kwargs)
            trainer:
            pl_module:
        """
        pass

    def _gen_and_store(self,trainer: "pl.Trainer", pl_module: "pl.LightningModule" ):
        try:
            with torch.no_grad():
                sample = getattr(pl_module, self.gen_method)(*self.gen_args, **self.gen_kwargs)
                self.store_sample(sample, trainer, pl_module)
        except Exception as e:
            logger.error(f"Error during generating and storing debug sample: {e}")

    @rank_zero_only
    def on_before_zero_grad(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer) -> None:
        if trainer.is_global_zero and self.every_n_train_steps is not None and trainer.global_step % self.every_n_train_steps == 0:
            self._gen_and_store(trainer, pl_module)
            # Call log here to get the method name also.
            logger.info(f"{self.__class__.__name__}: Generating and storing output at "
                        f"step={trainer.global_step},"
                        f" epoch={trainer.current_epoch}..."
            )

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.is_global_zero and self.every_n_epochs is not None and trainer.current_epoch % self.every_n_epochs == 0:
            self._gen_and_store(trainer, pl_module)
            # Call log here to get the method name also.
            logger.info(f"{self.__class__.__name__}: Generating and storing output at "
                        f"step={trainer.global_step}, "
                        f"epoch={trainer.current_epoch}..."
            )

ImageType = Union[np.ndarray, Image.Image]
class ClearmlImageDebugger(SampleDebugger):
    def __init__(
        self,
        *args,
        every_n_epochs: int = 5,
        every_n_train_steps: int|None=None,
        clearml_logger: ClearmlLogger = None,
        **kwargs,
    ):
        """
        Args:
            clearml_logger: If None, try to find logger from Trainer.
            every_n_epochs:
            every_n_train_steps: Save every n optimization steps, should use every_n_epoch to save memory, because
            this can be called during holding gradients, leading to Out of Memory.


        """
        super().__init__(every_n_epochs=every_n_epochs, every_n_train_steps = every_n_train_steps,*args, **kwargs)
        self.cml_logger = clearml_logger

    def store_sample(
        self, images: ImageType| Sequence[ImageType], trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        cml_logger = self.cml_logger
        if cml_logger is None:
            for pl_logger in trainer.loggers:
                if isinstance(pl_logger, ClearmlLogger):
                    cml_logger = pl_logger
                    break

        if cml_logger is not None:
            assert isinstance(cml_logger, ClearmlLogger)

            if isinstance(images, (np.ndarray, Image.Image)):
                images = [images]
            elif isinstance(images, Sequence):
                images = list(images)
            else:
                raise TypeError(f"images must be ImageType or Sequence[ImageType], got {type(images)}")

            for i, img in enumerate(images):
                if isinstance(img, ImageType):
                    if isinstance(img, Image.Image):
                        info = f"PIL.Image info: mode={img.mode},size={img.size} (width x height)"
                    else:
                        assert isinstance(img, np.ndarray)
                        info = f"Numpy image: dtype={img.dtype}, shape={img.shape}"

                    cml_logger.experiment.logger.report_image(
                        f"Image samples",
                        f"step_{trainer.global_step}_epoch_{trainer.current_epoch}_idx_{i}",
                        image=img
                    )
                    logger.info(f"Reporting image...{info}")
                    cml_logger.experiment.logger.flush(True)
                else:
                    logger.error(f"Unsupported image type: {type(img)}")


class ImageOutputCometCallback(SampleDebugger):

    def __init__(
        self, *args, every_n_epochs: int = 5, comet_logger: CometLogger = None, **kwargs
    ):
        """
        Args:
            comet_logger: If None, try to find logger from Trainer.

        """
        super().__init__(every_n_epochs=every_n_epochs, *args, **kwargs)
        self.cm_logger = comet_logger

    def store_sample(
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


class ImageOutputWandbCallback(SampleDebugger):

    def __init__(
        self, *args, every_n_epoch: int = 5, wandb_logger: WandbLogger = None, **kwargs
    ):
        """
        Args:
            wandb_logger: If None, try to find logger from Trainer.

        """
        super().__init__(every_n_epoch=every_n_epoch, *args, **kwargs)
        self.wdb_logger = wandb_logger

    def store_sample(
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


class ImageOutputNeptuneCallback(SampleDebugger):

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

    def store_sample(
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


