import contextlib
import io
import os
from datetime import timedelta
from pathlib import Path
from types import MethodType
from typing import Literal, Callable, Optional, Any
from weakref import proxy

import fsspec
import lightning.pytorch as pl
import lightning.pytorch.callbacks as pl_callbacks
import safetensors.torch as st
import torch
from lightning import LightningModule
from lightning.fabric.utilities.cloud_io import _load
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.plugins.io import CheckpointIO as PLCheckpointIO
from torch import Tensor, nn

from petorch import logger

StateDictExtractorType: type[
    Callable[[LightningModule | nn.Module, ...], dict[str, Tensor]]
] = Callable[[LightningModule, ...], dict[str, Tensor]]


class ModelCheckpoint(pl_callbacks.ModelCheckpoint):
    """
    Leverage the logic of ModelCheckpoint
    """

    def __init__(
        self,
        # --- Default ModelCheckpoint args ---
        dirpath: str | Path | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: bool | Literal["link"] | None = None,
        save_top_k: int = 1,
        save_on_exception: bool = False,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        enable_version_counter: bool = True,
        *,
        # --- Addition args ---
        module_state_dict_extractor: StateDictExtractorType | None = None,
        storage_options: bool | Any = False,
        file_extension: Literal[".safetensors"] | str | None = None,
    ):
        """

        Args:
            dirpath:
            filename:
            monitor:
            verbose:
            save_last:
            save_top_k:
            save_on_exception:
            save_weights_only:
            mode:
            auto_insert_metric_name:
            every_n_train_steps:
            train_time_interval:
            every_n_epochs:
            save_on_train_epoch_end:
            enable_version_counter:

            module_state_dict_extractor: This is for customize the state dict of the lightning module to save.
             Note that be careful when training with sparse files, because all parts of the model do not in one place.
             So the customization logic leads to unintentional error. Should check through `Strategy.save_checkpoint`
             for all Strategy in `lightning.pytorch.strategies`.

            storage_options: This will be passed to CheckpointIO, and you can customize the logic of saving depends on this.
             Useful for the case you save multiple checkpoints, and they have different expected storage.
             If `True` or given any object, pass the `locals()` of the `_save_checkpoint` method.
             
            Notes:
                `storage_options` must be False or None if the training strategy is not DDP or SingleDevice, or will raise an error.

            file_extension: Override the FILE_EXTENSION. The DefaultCheckpointIO will check if it is '.safetensors'. Then using save load.
        """
        super().__init__(
            dirpath,
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_on_exception,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            train_time_interval,
            every_n_epochs,
            save_on_train_epoch_end,
            enable_version_counter,
        )
        self._state_dict_extractor = module_state_dict_extractor
        self._storage_options = storage_options
        if file_extension:
            self.FILE_EXTENSION = file_extension

    @contextlib.contextmanager
    def _patch_pl_state_dict(self, module: LightningModule | nn.Module):
        org_state_dict = None
        try:
            if self._state_dict_extractor is not None:
                assert callable(self._state_dict_extractor)
                org_state_dict = module.state_dict
                module.state_dict = MethodType(self._state_dict_extractor, module)
            yield
        finally:
            if org_state_dict is not None:
                module.state_dict = org_state_dict

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        # Does not check for rank 0, because maybe multi-rank save checkpoints. Depend on the Strategy.

        # Trainer.save_checkpoint check this, But we need to access model now, so just check this first.
        if trainer.model is None:
            raise AttributeError(
                "Saving a checkpoint is only possible if a model is attached to the Trainer. Did you call"
                " `Trainer.save_checkpoint()` before calling `Trainer.{fit,validate,test,predict}`?"
            )
        with self._patch_pl_state_dict(trainer.lightning_module):
            trainer.save_checkpoint(
                filepath,
                self.save_weights_only,
                storage_options=locals() if self._storage_options else None,
            )
            self._last_global_step_saved = trainer.global_step
            self._last_checkpoint_saved = filepath

            # notify loggers
            if trainer.is_global_zero:
                for logger in trainer.loggers:
                    logger.after_save_checkpoint(proxy(self))


# Helper functions for custom CheckpointIO


def torch_save_checkpoint(checkpoint: dict[str, Any], path: str):
    logger.info(f"Saving checkpoint: {path}")

    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)

    fs, urlpath = fsspec.core.url_to_fs(path)
    with fs.transaction, fs.open(urlpath, "wb") as f:
        f.write(buffer.getvalue())


def torch_load_checkpoint(path: _PATH, map_location: Optional[Any] = None
) -> dict[str, Any]:
    return _load(path, map_location)


def safe_load_checkpoint(
    path_or_file: str | Path, map_location: Optional[Any] = None
) -> dict[str, Any]:
    # Check
    path_or_file = str(path_or_file)
    if path_or_file.startswith("http"):
        raise ValueError("Remote URLs not supported yet for safetensors")
    fs, _ = fsspec.core.url_to_fs(path_or_file)
    if not fs.exists(path_or_file):
        raise FileNotFoundError(f"Checkpoint file not found: {path_or_file}")

    # Handle fsspec-style filesystems
    with fs.open(path_or_file, "rb") as f:
        return st.load(f.read())  # load from bytes


def safe_save_checkpoint(checkpoint: dict[str, Tensor], path: str):
    assert path.endswith(".safetensors")
    logger.info(f"Safe saving checkpoint: {path}")
    byte_tensor = st.save(checkpoint)

    fs, urlpath = fsspec.core.url_to_fs(path)
    with fs.transaction, fs.open(urlpath, "wb") as f:
        f.write(byte_tensor)


def remove_checkpoint(path: _PATH) -> None:
    fs, _ = fsspec.core.url_to_fs(path)
    if fs.exists(path):
        fs.rm(path, recursive=True)


class DefaultCheckpointIO(PLCheckpointIO):
    """
    Notes:
        CheckpointIO only be use if train strategy is DDP or SingleDevice.
    """
    
    def save_checkpoint(
        self,
        checkpoint: dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        """
        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: Optional parameters when saving the model/training states.

        Notes:
            Trainer.save_checkpoint return the structure (copy from Trainer._checkpoint_connector.dump_checkpoint):
                structured dictionary: {
                    'epoch':                     training epoch
                    'global_step':               training global step
                    'pytorch-lightning_version': The version of PyTorch Lightning that produced this checkpoint
                    'callbacks':                 "callback specific state"[] # if not weights_only
                    'optimizer_states':          "PT optim's state_dict"[]   # if not weights_only
                    'lr_schedulers':             "PT sched's state_dict"[]   # if not weights_only
                    'state_dict':                Model's state_dict (e.g. network weights)
                    precision_plugin.__class__.__qualname__:  precision plugin state_dict # if not weights_only
                    CHECKPOINT_HYPER_PARAMS_NAME:
                    CHECKPOINT_HYPER_PARAMS_KEY:
                    CHECKPOINT_HYPER_PARAMS_TYPE:
                    something_cool_i_want_to_save: anything you define through model.on_save_checkpoint
                    LightningDataModule.__class__.__qualname__: pl DataModule's state
                }
        """
        path = str(path)
        fs, _ = fsspec.core.url_to_fs(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)

        if path.endswith(".safetensors"):
            safe_save_checkpoint(checkpoint["state_dict"], path)
        else:
            torch_save_checkpoint(checkpoint, path)

    def load_checkpoint(
        self, path: _PATH, map_location: Optional[Any] = None
    ) -> dict[str, Any]:
        fs, _ = fsspec.core.url_to_fs(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        if str(path).endswith(".safetensors"):
            return safe_load_checkpoint(path, map_location=map_location)
        else:
            return torch_load_checkpoint(path, map_location=map_location)

    def remove_checkpoint(self, path: _PATH) -> None:
        return remove_checkpoint(path)


def create_metric_checkpoint_callback(
    checkpoints_path: str,
    metric: str,
    save_best_every_n_epochs: int,
    save_best_every_n_train_steps: int,
    mode="min",
):
    # This call back will default as every_n_train_steps = 1.
    # And `save_on_train_epoch_end=True`. So that it check when `on_train_epoch_end` hook.
    # save_last is None, so only save for top k of monitor value.
    # This note save last because we not set save last, and provide monitor.
    return ModelCheckpoint(
        dirpath=checkpoints_path,
        filename=f"{{epoch}}-{{step}}-min_{{{metric}:.4f}}",
        monitor=metric,
        mode=mode,
        save_top_k=1,
        # I save best depend on the epoch metric, so i set every_n_epochs here.
        every_n_epochs=save_best_every_n_epochs,
        every_n_train_steps=save_best_every_n_train_steps,
        save_on_train_epoch_end=True,
        save_weights_only=True,
    )

