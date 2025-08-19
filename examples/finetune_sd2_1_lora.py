import warnings
from types import MethodType
from typing import Callable, TypeVar, cast, Type, Generic, Iterator
from weakref import proxy
import lightning as pl
import safetensors.torch as st
import torch
import torchvision.transforms.v2 as transforms
from datasets import load_dataset, Dataset as ArrDataset
from lightning import LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, ModelSummary
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.model_summary import summarize
from torch.utils.data import DataLoader as TorchDataLoader, default_collate
from transformers import CLIPTokenizerFast
from pathlib import Path
from petorch import AdapterAPI
from petorch.integrations.diffusers.stable_diffusion import (
    StableDiffusionModule,
    SDBatch,
    MetricKey,
)
from petorch.prebuilt.configs import LoraConfig
from loguru import logger

model_id = "stabilityai/stable-diffusion-2-1"


image_train_size = (256, 256)
image_transform = transforms.Compose(
    [
        transforms.Resize(image_train_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToImage(),  # PIL to tensor
        transforms.ToDtype(
            dtype=torch.get_default_dtype(), scale=True
        ),  # convert PIL → tensor [0,1]
        transforms.Normalize([0.5], [0.5]),  # map [0,1] → [-1,1]
    ]
)

tokenizer: CLIPTokenizerFast = CLIPTokenizerFast.from_pretrained(
    model_id, subfolder="tokenizer"
)


def text_transform(text: str | list[str]) -> torch.Tensor:
    batch_encoding = tokenizer.__call__(text, padding=True, return_tensors="pt")
    return batch_encoding.input_ids


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
        image_transform_fn: Callable = None,
        text_transform_fn: Callable = None,
        train_ratio: float = 0.9,
        num_workers: int = 8,
        batch_size=8,
    ):
        super().__init__()
        self.image_transform = image_transform_fn or image_transform
        self.text_transform = text_transform_fn or text_transform

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

    def setup(self, stage: str = None) -> None:
        dataset = load_dataset(self.model_id, split="train")

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


class LoraCheckpoint(ModelCheckpoint):
    FILE_EXTENSION = ".safetensors"

    def _save_checkpoint(self, pl_trainer: pl.Trainer, filepath: str) -> None:
        try:
            state_dict = AdapterAPI.get_adapter_state_dict(pl_trainer.lightning_module)

            if state_dict:
                # Create folder
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Save checkpoint: {filepath}")
                st.save_file(state_dict, filepath)

                self._last_global_step_saved = trainer.global_step
                self._last_checkpoint_saved = filepath

                # notify loggers
                if trainer.is_global_zero:
                    for tl_logger in trainer.loggers:
                        tl_logger.after_save_checkpoint(proxy(self))
            else:
                logger.error("Got blank lora state dict.")

        except ValueError as e:
            if "Model does not have any adapter" not in str(e):
                raise e
            else:
                warnings.warn(str(e))


dirpath = "sd_2_1_naruto"
checkpoints_path = f"{dirpath}/checkpoints"
save_every_n_epochs = 3
max_steps = 3000


def create_metric_checkpoint_callback(metric: MetricKey, mode="min"):

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
        # every_n_epochs=save_every_n_epochs,
        save_on_train_epoch_end=True,
        save_weights_only=True,
    )


callbacks = [
    create_metric_checkpoint_callback(metric)
    for metric in [MetricKey.TRAIN_LOSS, MetricKey.VAL_LOSS]
]
# noinspection PyTypeChecker
callbacks.extend(
    [
        # see `ModelCheckpoint._save_none_monitor_checkpoint`, it will call `_save_none_monitor_checkpoint` if monitor
        # is not provide, so although we not use save_last, it still save the last each epoch.
        # Not set save_weights_only, save for everything.
        ModelCheckpoint(
            dirpath=checkpoints_path,
            filename="{epoch}-{step}-last",
            save_top_k=1,
            # every_n_train_steps=int(max_steps / 10),
            save_on_train_epoch_end=True,
        ),
        TQDMProgressBar(leave=True),
        ModelSummary(max_depth=-1),
    ]
)
loggers = [CSVLogger(dirpath, version=0),]

trainer = Trainer(
    default_root_dir=dirpath,
    callbacks=callbacks,
    logger=loggers,
    precision="bf16-true",
    max_steps=max_steps,
    num_sanity_val_steps=0,
    # accelerator="cpu",
    # fast_dev_run=10,
    limit_train_batches=0.2,
    limit_val_batches=0.2,
)

if __name__ == "__main__":
    data_module = NarutoBlipDataModule(batch_size=8)
    module = StableDiffusionModule(model_id=model_id)

    # Freeze and add adapter to unet
    module.requires_grad_(False)
    config = LoraConfig(adapter_name="default", rank=8, alpha=16,
                        fqname_filter=lambda n: "unet" in n)
    AdapterAPI.add_adapter(module, config, activate=True)

    n,t  = 0,0
    for name, param in module.named_parameters():
        t+=param.numel()
        if param.requires_grad:
            assert "lora" in name
            n+=param.numel()
    print(f"Trainable params/Total params/Ratio: {n:,}/{t:,}/{n/t:.4f}")

    trainer.fit(module, datamodule=data_module)
