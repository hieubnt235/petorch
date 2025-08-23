import pprint
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from types import MethodType
from typing import (
    Callable,
    TypeVar,
    cast,
    Type,
    Generic,
    Iterator,
    Sequence,
    Any,
    Optional,
    Literal,
)
from uuid import uuid4
from weakref import proxy

# Must be import first to log Pytorch output
from comet_ml import CometExperiment, API

from petorch.utilities import b64encode

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
    NeptuneLogger,
    CometLogger as PLCometLogger,
)
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
        else:
            ds_dict = self.dataset.train_test_split(train_size=self.train_ratio)
            self.train_dataset = ds_dict["train"]
            self.val_dataset = ds_dict["test"]

    def _return_dataloader(self, dataset) -> DataLoader[_Batch_T]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
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


class LoraCheckpoint(ModelCheckpoint):
    FILE_EXTENSION = ".safetensors"

    def save_state_dict(
        self, state_dict: dict[str, torch.Tensor], trainer: pl.Trainer, filepath: str
    ):
        cast(pl.Trainer, trainer)
        # Create folder
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving Lora checkpoint: {filepath}")

        # Save
        byte_tensor = st.save(state_dict)
        fs, urlpath = fsspec.core.url_to_fs(str(filepath))
        with fs.transaction, fs.open(urlpath, "wb") as f:
            f.write(byte_tensor)
        # Above saving steps can be simple like this if don't care about atomic of saving.
        # st.save_file(state_dict, filepath)

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        # Adapt to CheckpointIO of DDPStrategy.
        if trainer.is_global_zero:
            try:
                state_dict = AdapterAPI.get_adapter_state_dict(trainer.lightning_module)
                if state_dict:
                    self.save_state_dict(state_dict, trainer, filepath)
                    # Copy from parent
                    self._last_global_step_saved = trainer.global_step
                    self._last_checkpoint_saved = filepath
                    for tl_logger in trainer.loggers:
                        tl_logger.after_save_checkpoint(
                            proxy(self)
                        )  # See detail in this method for each Logger.
                else:
                    logger.error("Got blank lora state dict.")

            except ValueError as e:
                if "Model does not have any adapter" not in str(e):
                    raise e
                else:
                    warnings.warn(str(e))


class CometLogger(PLCometLogger):

    def __init__(
        self,
        log_model_checkpoint: bool = True,
        checkpoint_path: str = "checkpoints", # Will be concat with dirpath_id of checkpoint model.
        encode_dirpath:bool=True,
        *,
        # Lightning CometLogger kwargs.
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        experiment_key: Optional[str] = None,
        mode: Optional[Literal["get_or_create", "get", "create"]] = None,
        online: Optional[bool] = None,
        prefix: Optional[str] = None,
        name: str | None = None,  # experiment name
        **kwargs: Any,
    ):
        super().__init__(
            api_key=api_key,
            workspace=workspace,
            project=project,
            experiment_key=experiment_key,
            mode=mode,
            online=online,
            prefix=prefix,
            name=name,
            **kwargs,
        )
        self._log_model_checkpoints = log_model_checkpoint
        assert checkpoint_path
        self._model_name = checkpoint_path
        self._encode_dirpath = encode_dirpath

    def _get_model_path(self, dirpath: str)->str:
        if self._encode_dirpath:
            dirpath = b64encode(dirpath)
        return f"{self._model_name}/{dirpath}"

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        if not self._log_model_checkpoints:
            return
        comet_path = self._get_model_path(checkpoint_callback.dirpath)
        comet_expr = self.experiment
        def _log_model(local_model_path: str)->dict:
            return comet_expr.log_model(comet_path, local_model_path, overwrite=True)

        uid = uuid4()
        cn = f"{self.__class__.__name__}-ID={uid}"
        # flushed =[]
        logger.info(
            f"{cn}: Uploading checkpoints...\n"
            f"workspace=`{comet_expr.workspace}` - project_name=`{comet_expr.project_name}` - model_name=`{self._model_name}`"
        )
        uploaded_files = dict()
        """Id of files that just upload in this method. Keys are assetId, values are filename (excluding dir)."""

        # log best model path
        if paths := getattr(checkpoint_callback, "best_k_models", None):
            paths = cast(dict[str, torch.Tensor], paths)
            for path, score in paths.items():
                file = path.rsplit("/", maxsplit=1)[-1]

                assert file not in uploaded_files.values()
                logger.info(
                    f"Uploading `best_model_path` checkpoint with score={score.item()}: {path}"
                )
                aid = _log_model(path)["assetId"]
                uploaded_files[aid] = file

        if path := getattr(checkpoint_callback, "last_model_path", None):
            file = path.rsplit("/", maxsplit=1)[-1]
            if file in uploaded_files.values():
                assert path not in checkpoint_callback.best_k_models
                logger.info(
                    f"`last_model_path` checkpoint name `{file}` already in hold_files (be one of the best k files)."
                    f" Add postfix `_last` in filename."
                )
                f, extension = file.rsplit(".", 1)
                f += "_last"
                file = f"{f}.{extension}"
                assert file not in uploaded_files.values()
            logger.info(f"Uploading `last_model_path` checkpoint: {path}")
            aid = _log_model(path)["assetId"]
            uploaded_files[aid] = file

        # None when no running experiment, not error.
        if not (f := comet_expr.flush()) and f is not None:
            logger.error(f"{cn}: Upload checkpoint fails...")
        else:
            logger.info(
                f"{cn}: Uploaded checkpoints: \n{pprint.pformat(uploaded_files)}"
            )

        # remove old models logged to experiment if they are not part of best k models at this point
        # Note that there's a case that model files are the same in Comet, but ID are different, so I use ID as key here.
        api_expr = API().get(
            comet_expr.workspace,
            comet_expr.project_name,
            experiment=comet_expr.get_key(),
        )
        all_model_files = {
            model["assetId"]: model["fileName"]
            for model in api_expr.get_model_asset_list(comet_path)
        }
        logger.info(  # todo: change to debug
            f"{cn}: ALl model checkpoints after uploading: \n"
            f"{pprint.pformat(all_model_files)}\n"
            f"Should hold files: \n"
            f"{pprint.pformat(uploaded_files)}")
        assert len(all_model_files) >= len(uploaded_files)

        # Delete all keys not in hold_files. Or if duplicate, delete the one not have ID in uploaded_files.
        del_ids = {}  # assetID: filename
        remain_files = {}  # filename: assetID
        for aid, filename in all_model_files.items():
            if filename not in uploaded_files.values():
                del_ids[aid] = filename
                continue

            # Not allow upload files have the name but differnent IDs, Except the ID just upload.
            if filename in remain_files: # Duplicate
                if (rid:=remain_files[filename]) not in uploaded_files.keys():
                    assert aid in uploaded_files
                    del_ids[rid] = filename
                    remain_files[filename] = aid
                else:
                    assert aid not in del_ids
                    assert aid not in uploaded_files
                    del_ids[aid] = filename
            else:
                remain_files[filename] = aid
        logger.info(
            f"Deleting checkpoints: \n"
            f"{pprint.pformat(del_ids)}"
        )
        for del_id in del_ids.keys():
            api_expr.delete_asset(del_id)
        if not (f := comet_expr.flush()) and f is not None:
            logger.error(f"{cn}: Delete checkpoints fails...")
        else:
            logger.info(f"{cn}: Deleted checkpoints successfully. ")
        logger.info(f"{cn}: End uploading.")


class OutputEvaluationCallback(Callback, ABC):

    def __init__(self, *args, every_n_epoch: int, **kwargs) -> None:
        self.every_n_epoch = every_n_epoch
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
        if trainer.is_global_zero and trainer.current_epoch % self.every_n_epoch == 0:
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
        self, *args, every_n_epoch: int = 5, comet_logger: CometLogger = None, **kwargs
    ):
        """
        Args:
            comet_logger: If None, try to find logger from Trainer.

        """
        super().__init__(every_n_epoch=every_n_epoch, *args, **kwargs)
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
                    img, f"output_images_step_{trainer.global_step}_i"
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
------- Training starting ------------------------
Training batches per epoch: {trainer.num_training_batches}
Total training batches: {self.total_training_batches}
Gradient accumulation batch steps: {trainer.accumulate_grad_batches}
Total optimization steps: {trainer.estimated_stepping_batches}
Current optimization steps: {trainer.global_step - 1}
Validation batches during training: {trainer.num_val_batches}
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
    return LoraCheckpoint(
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
            LoraCheckpoint(
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
    data_module = NarutoBlipDataModule(batch_size=4, num_workers=32, train_ratio=0.8)

    # 3. Prepare trainer
    pl_trainer = get_trainer(
        storage_path + PROJECT_NAME,
        max_steps=6000,
        save_every_n_train_steps=300,
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
            ImageOutputCometCallback("A beautiful girl with glasses", every_n_epochs=5),
        ],
        log_every_n_steps=10,  # batch steps (training_step), not optimization step.
        # debug=False,
    )
    pl_trainer.fit(module, datamodule=data_module)
