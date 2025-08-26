from abc import ABC, abstractmethod
from contextlib import ExitStack
from pathlib import Path
from types import MethodType
from typing import (Callable, TypeVar, cast, Type, Generic, Iterator, Sequence, Any, Optional, )

# Must be import first to log Pytorch output
from comet_ml import CometExperiment
from lightning.fabric.utilities.types import _PATH

# Dummy usage for not reorder during reformat file.
cast(type, CometExperiment)

import fsspec
import lightning as pl
import safetensors.torch as st
import torch
import torchvision.transforms.v2 as transforms
from datasets import load_dataset, Dataset as ArrDataset
from lightning import LightningDataModule, Trainer, Callback
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
    ModelSummary,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import (
    WandbLogger,
    Logger,
    NeptuneLogger, )
from lightning.pytorch.plugins.io import CheckpointIO

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.optim import Optimizer
from torch.utils.data import DataLoader as TorchDataLoader, default_collate
from transformers import CLIPTokenizerFast, PreTrainedTokenizerBase

from petorch import AdapterAPI, logger
from petorch.integrations.diffusers.stable_diffusion import (
    StableDiffusionModule,
    SDBatch,
    MetricKey,
)
from petorch.prebuilt.configs import LoraConfig
from petorch.experiments.loggers import CometLogger


model_id = "stabilityai/stable-diffusion-2-1"


def get_image_transform(
    size: int | Sequence[int] | None = None,
) -> transforms.Transform:
    return transforms.Compose(
        [
            transforms.Resize(size or (256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),  # PIL to tensor
            transforms.ToDtype(
                dtype=torch.get_default_dtype(), scale=True
            ),  # convert PIL → tensor [0,1]
            transforms.Normalize([0.5], [0.5]),  # map [0,1] → [-1,1]
        ]
    )


def get_clip_tokenizer(
    model_path: str = "stabilityai/stable-diffusion-2-1", subfolder: str = "tokenizer"
) -> CLIPTokenizerFast:
    return CLIPTokenizerFast.from_pretrained(model_path, subfolder=subfolder)


def get_text_transform(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[str | list[str]], torch.Tensor]:
    def text_transform(text: str | list[str]) -> torch.Tensor:
        batch_encoding = tokenizer.__call__(text, padding=True, return_tensors="pt")
        return batch_encoding.input_ids

    return text_transform


_T_Co = TypeVar("_T_Co", covariant=True)


class DataLoader(TorchDataLoader, Generic[_T_Co]):
    def __iter__(self) -> Iterator[_T_Co]:
        return cast(Iterator[_T_Co], super().__iter__())


_Batch_T = TypeVar("_Batch_T", bound=SDBatch, default=SDBatch, covariant=True)


class NarutoBlipDataModule(LightningDataModule, Generic[_Batch_T]):
    model_id = "lambdalabs/naruto-blip-captions"
    batch_type: Type[_Batch_T] = SDBatch

    def __init__(
        self,
        *,
        image_transform_fn: Callable = None,
        text_transform_fn: Callable = None,
        train_ratio: float = 0.8,
        num_workers: int = 8,
        batch_size=8,
    ):
        """

        Args:
            image_transform_fn: If not provide, they will be init after trainer is assigned, see `setup_transforms`.
            text_transform_fn:  See `image_transform_fn`
            train_ratio:
            num_workers:
            batch_size:
        """

        super().__init__()
        # If None, assign when trainer is assigned (in `setup`).
        self.image_transform = image_transform_fn
        self.text_transform = text_transform_fn

        assert train_ratio > 0
        self.train_ratio = min(1.0, train_ratio)
        if self.train_ratio < 1.0:
            self.val_dataset: ArrDataset | None = None

            def _f(obj: NarutoBlipDataModule):
                return obj._return_dataloader(obj.val_dataset)

            self.val_dataloader = MethodType(_f, self)

        self.dataset: ArrDataset | None = None
        self.train_dataset: ArrDataset | None = None

        self.num_workers = num_workers
        self.batch_size = batch_size

    def _collate_fn(self, batch_l: list[dict]) -> _Batch_T:
        return self.batch_type.model_validate(default_collate(batch_l))

    def prepare_data(self) -> None:
        load_dataset(self.model_id)

    @property
    def lightning_module(self):
        return self.trainer.lightning_module

    def setup_transforms(self):
        if self.text_transform is None:
            tokenizer = getattr(self.lightning_module, "tokenizer", None)
            if tokenizer is None:
                logger.info(
                    f"`text_transform` was not given, and `lightning_module` does not have `tokenizer` attribute."
                    f"Use the default from `{get_clip_tokenizer}` constructor for create transform."
                )
                tokenizer = get_clip_tokenizer()
            else:
                logger.info(
                    f"`text_transform` was not given, use `lightning_module.tokenizer` for transform."
                )
            self.text_transform = get_text_transform(tokenizer)

        if self.image_transform is None:
            sample_size = getattr(self.lightning_module, "sample_size", None)
            if sample_size is None:
                logger.info(
                    f"`image_transform` was not given, and `lightning_module` does not have `sample_size` attribute."
                    f"Use the default from `{get_image_transform}` constructor for create transform."
                )
            else:
                logger.info(
                    f"`image_transform` was not given, use `lightning_module.sample_size`={sample_size} for constructing transform."
                )
            self.image_transform = get_image_transform(sample_size)

    def setup(self, stage: str = None) -> None:
        dataset = load_dataset(self.model_id, split="train")
        self.setup_transforms()

        def _transform(sample: dict):
            return dict(
                images=self.image_transform(sample["image"]),
                input_ids=self.text_transform(sample["text"]),
            )

        dataset.set_transform(_transform)
        self.dataset = dataset

        if self.train_ratio >= 1:
            self.train_dataset = self.dataset
            logger.info(f"Length train dataset: {len(self.train_dataset)}\n"
                    f"Length val dataset: None")
        else:
            ds_dict = self.dataset.train_test_split(train_size=self.train_ratio)
            self.train_dataset = ds_dict["train"]
            self.val_dataset = ds_dict["test"]
            logger.info(f"Length train dataset: {len(self.train_dataset)}\n"
                        f"Length val dataset: {len(self.val_dataset)}")

    def _return_dataloader(self, dataset) -> DataLoader[_Batch_T]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last = True
        )

    def train_dataloader(self) -> DataLoader[_Batch_T]:
        return self._return_dataloader(self.train_dataset)

    # def val_dataloader(self) -> DataLoader[_Batch_T]:
    #     if self.val_dataset is None:
    #         return super().val_dataloader()
    #     else:

    def transfer_batch_to_device(
        self, batch: SDBatch, device: torch.device, dataloader_idx: int
    ) -> SDBatch:
        return batch.to(device=device, dtype=self.lightning_module.dtype)


class SafetensorsCheckpointIO(CheckpointIO):

    def save_checkpoint(self, checkpoint: dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

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
            But `safetensors` only flatten Tensor. We can extract this, but for now save only state_dict.

        """

        # Create folder
        path = str(path)
        if not path.endswith(".safetensors"):
            path+= ".safetensors"
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save state dict only
        byte_tensor = st.save(checkpoint["state_dict"])
        n_params = 0
        for param in checkpoint["state_dict"].values():
            n_params+=param.numel()

        fs, urlpath = fsspec.core.url_to_fs(path)
        logger.info(f"Saving checkpoint (Num params={n_params:,} - File size={len(byte_tensor):,} bytes): {urlpath}")
        with fs.transaction, fs.open(urlpath, "wb") as f:
            f.write(byte_tensor)

    def load_checkpoint(self, path_or_file: str|Path, map_location: Optional[Any] = None) -> dict[str, Any]:
        # Check
        path_or_file = str(path_or_file)
        if path_or_file.startswith("http"):
            raise ValueError("Remote URLs not supported yet for safetensors")
        fs,_ = fsspec.core.url_to_fs(path_or_file)
        if not fs.exists(path_or_file):
            raise FileNotFoundError(f"Checkpoint file not found: {path_or_file}")

        # Handle fsspec-style filesystems
        with fs.open(path_or_file, "rb") as f:
            return st.load(f.read())  # load from bytes

    def remove_checkpoint(self, path: _PATH) -> None:
        fs,_ = fsspec.core.url_to_fs(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)


class AdapterOnlyCheckpoint(ModelCheckpoint):
    FILE_EXTENSION = ".safetensors"

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        # Adapt to `trainer.save_checkpoint(filepath, self.save_weights_only)`, which is called by the ModelCheckpoint._save_checkpoint.
        # Does not check for rank 0, because maybe multi-rank save checkpoints. Depend on the Strategy.

        # Trainer.save_checkpoint check this, But we need to access model now, so just check this first.
        if trainer.model is None:
            raise AttributeError(
                "Saving a checkpoint is only possible if a model is attached to the Trainer. Did you call"
                " `Trainer.save_checkpoint()` before calling `Trainer.{fit,validate,test,predict}`?"
            )

        # Call super to save. If model does not have adapter, save raw state dict.
        try:
            with ExitStack() as stack:
                # Note we check `lightning_module`, not `model`.
                # Model is a wrapper of module with class `torch.nn.parallel.distributed.DistributedDataParallel`
                if custom_cm:=getattr(trainer.lightning_module, "enable_custom_state_dict", None):
                    stack.enter_context(custom_cm(AdapterAPI.get_adapter_state_dict))
                    super()._save_checkpoint(trainer, filepath)
                else:
                    logger.warning(f"Model {trainer.model.__class__} does not have `enable_custom_state_dict` method,"
                                   f"so cannot save adapter only. So just skip and don't save anything.")
        except ValueError as e:
            if "Model does not have any adapter" in str(e):
                logger.warning(
                    f"Model {trainer.model.__class__} have `enable_custom_state_dict` method,"
                    f"but does not have any adapter. So just skip and don't save anything."
                )
            else:
                raise e



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


class DebugCallback(pl.Callback):
    def __init__(
        self,
        every_n_batch_steps: int = 10,
        verbose: bool = False,
        rank_zero_only: bool = True,
    ):
        self.every_n_batch_steps = every_n_batch_steps
        self.total_training_batches = -999
        self.verbose = verbose
        self.rank_zero_only = rank_zero_only

    def _should_skip(self, trainer: "pl.Trainer") -> bool:
        if self.rank_zero_only and (not trainer.is_global_zero):
            return True
        return False

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._should_skip(trainer):
            return

        self.total_training_batches = (
            trainer.estimated_stepping_batches * trainer.accumulate_grad_batches
        )
        logger.info(
            f"""
------- Training Info------------------------
Total training batches: {self.total_training_batches}
Training batches per epoch: {trainer.num_training_batches}

Batch size: {trainer.train_dataloader.batch_size}
Train dataset length: {trainer.train_dataloader.dataset.__len__()}
Validation batches during training: {trainer.num_val_batches}

Gradient accumulation batch steps: {trainer.accumulate_grad_batches}
Total optimization steps: {trainer.estimated_stepping_batches}
Current optimization steps: {trainer.global_step - 1}
--------------------------------------------------
            """
        )

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._should_skip(trainer):
            return

        self.total_training_batches = -999

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._should_skip(trainer):
            return

        if trainer.fit_loop.total_batch_idx % self.every_n_batch_steps == 0:
            if self.verbose:
                if isinstance(batch, SDBatch):
                    batch = cast(SDBatch, batch)
                    images, input_ids = batch.images, batch.input_ids
                    logger.debug(
                        f"`images`: {images.shape}-{images.device}-{images.dtype}\n"
                        f"`input_ids`: {input_ids.shape}-{input_ids.device}-{input_ids.dtype}\n"
                    )

    def on_before_zero_grad(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: Optimizer,
    ) -> None:
        if self._should_skip(trainer):
            return

        if self.verbose:
            logger.debug(
                f"on_before_zero_grad on batch_idx {trainer.fit_loop.total_batch_idx}/{self.total_training_batches}"
            )

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: Optimizer,
    ) -> None:
        if self._should_skip(trainer):
            return

        if self.verbose:
            logger.debug(
                f"on_before_optimizer_step on batch_idx {trainer.fit_loop.total_batch_idx}/{self.total_training_batches}"
            )


def create_metric_checkpoint_callback(
    checkpoints_path: str,
    metric: MetricKey,
    save_best_every_n_epochs: int,
    save_best_every_n_train_steps: int,
    mode="min",
):
    # This call back will default as every_n_train_steps = 1.
    # And `save_on_train_epoch_end=True`. So that it check when `on_train_epoch_end` hook.
    # save_last is None, so only save for top k of monitor value.
    # This note save last because we not set save last, and provide monitor.
    return AdapterOnlyCheckpoint(
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


def get_trainer(
    dirpath: str,
    checkpoints_path: str = None,
    version: int | None = None,
    *,
    max_steps=3000,
    max_epochs=None,
    save_every_n_epochs=None,
    save_every_n_train_steps=None,
    save_top_k=1,
    accumulate_grad_batches=4,
    precision="bf16-true",
    addition_callbacks: list[Callback] = None,
    addition_loggers: list[Logger] = None,
    debug: bool = False,
    debug_downscale: float = 0.2,
    **kwargs,
):
    """

    Args:
        addition_loggers:
        version:
        dirpath:
        checkpoints_path:
        max_steps: Number of doing optimizer.step().
        max_epochs:
        save_every_n_epochs:
        save_every_n_train_steps: Number of doing optimizer.step().
        save_top_k:
        precision:
        accumulate_grad_batches:
        addition_callbacks:
        debug:
        debug_downscale:
        **kwargs:

    Returns:

    """
    checkpoints_path = checkpoints_path or f"{dirpath}/checkpoints"
    if version is None:
        version = 0
        while True:
            if not Path(checkpoints_path).joinpath(f"version_{version}").exists():
                break
            version += 1
    assert isinstance(version, int) and version >= 0
    checkpoints_path = Path(checkpoints_path).joinpath(f"version_{version}").as_posix()

    # Default callbacks
    callbacks: list[pl.Callback] = []
    callbacks.extend(
        [
            # Save last checkpoint
            # see `ModelCheckpoint._save_none_monitor_checkpoint`, it will call `_save_none_monitor_checkpoint` if monitor
            # is not provide, so although we not use save_last, it still save the last each epoch.
            # Not set save_weights_only, save for everything.
            AdapterOnlyCheckpoint(
                dirpath=checkpoints_path,
                filename="{epoch}-{step}",
                monitor="step",  # For saving top k after every epoch. Compare by `step`.
                mode="max",
                save_top_k=save_top_k,
                every_n_epochs=save_every_n_epochs,
                every_n_train_steps=save_every_n_train_steps,
                save_on_train_epoch_end=True,
                verbose= debug
            ),
            TQDMProgressBar(leave=True),
            ModelSummary(max_depth=1),
        ]
    )
    if addition_callbacks:
        callbacks.extend(addition_callbacks)
    loggers: list[Logger] = []
    if addition_loggers:
        loggers.extend(addition_loggers)

    if debug:
        assert 0 <= debug_downscale <= 1
        kwargs["limit_train_batches"] = (
            kwargs.get("limit_train_batches") or debug_downscale
        )
        kwargs["limit_val_batches"] = kwargs.get("limit_val_batches") or debug_downscale
        max_steps = debug_downscale * max_steps
        accumulate_grad_batches = 1
        callbacks.append(DebugCallback(1, verbose=True))
    else:
        callbacks.append(DebugCallback(10, verbose=False))

    trainer = Trainer(
        default_root_dir=dirpath,
        callbacks=callbacks,
        logger=loggers,
        precision=precision,
        max_steps=max_steps,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
        plugins= SafetensorsCheckpointIO(),
        # accelerator="cpu",
        # fast_dev_run=10,
        # limit_train_batches=0.2,
        # limit_val_batches=0.2,
        **kwargs,
    )
    return trainer


def get_sd_module(
    adt_config: LoraConfig = None, adt_ckpt: str = None, **module_kwargs
) -> StableDiffusionModule:
    logger.debug(f"Load model from: {model_id}")
    module = StableDiffusionModule(model_id=model_id, **module_kwargs)
    if adt_config:
        logger.debug(f"Add adapter from config: {adt_config}")
        module.requires_grad_(False)
        AdapterAPI.add_adapter(module, config, activate=True)
        if adt_ckpt:
            logger.debug(f"Load Adapter checkpoint from: {adt_ckpt}")
            adt_state_dict = st.load_file(adt_ckpt)
            AdapterAPI.load_adapter_state_dict(module, adt_state_dict, strict_load=True)
    n, t = 0, 0
    for name, param in module.named_parameters():
        t += param.numel()
        if param.requires_grad:
            assert "lora" in name
            n += param.numel()
    logger.debug(f"Trainable params={n:,} | Total params={t:,} | Ratio={n / t:.4f}")
    return module


if __name__ == "__main__":
    PROJECT_NAME = "sd_2_1"
    storage_path = "/home/a3ilab01/petorch/storage/"
    adt_ckpt = None

    # 1. Prepare model
    config = LoraConfig(
        adapter_name="default", rank=16, alpha=16, fqname_filter=lambda _n: "unet" in _n
    )
    module = get_sd_module(config, adt_ckpt)

    # 2. Prepare data
    data_module = NarutoBlipDataModule(batch_size=4, num_workers=32, train_ratio=0.95)

    # 3. Prepare trainer
    pl_trainer = get_trainer(
        storage_path + PROJECT_NAME,
        max_steps=5000,
        save_every_n_train_steps=120, # change for debug
        save_top_k=3,
        accumulate_grad_batches=4,
        addition_loggers=[
            CometLogger(
                log_model_checkpoint=True,
                checkpoint_path="checkpoints",
                workspace="hieubnt235",
                project=PROJECT_NAME,
                name="naruto_blip_lora_r16",
            )
        ],
        addition_callbacks=[
            LearningRateMonitor(),
            ImageOutputCometCallback("A beautiful girl with glasses", every_n_epochs=3),
        ],
        log_every_n_steps=10,  # batch steps (training_step), not optimization step.
        # debug=True,
    )
    pl_trainer.fit(module, datamodule=data_module)
