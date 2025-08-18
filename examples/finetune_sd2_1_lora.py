from types import MethodType
from typing import Callable, TypeVar, cast, Type, Generic, Iterator

import torch
import torchvision.transforms.v2 as transforms
from datasets import load_dataset, Dataset as ArrDataset
from lightning import LightningDataModule, Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader as TorchDataLoader, default_collate
from transformers import CLIPTokenizerFast

from petorch.integrations.diffusers.stable_diffusion import StableDiffusionModule, SDBatch, MonitorKey

model_id = "stabilityai/stable-diffusion-2-1"


image_train_size = (256,256)
image_transform = transforms.Compose([
    transforms.Resize(image_train_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToImage(),  # PIL to tensor
    transforms.ToDtype(dtype=torch.get_default_dtype(), scale=True),  # convert PIL → tensor [0,1]
    transforms.Normalize([0.5], [0.5])  # map [0,1] → [-1,1]
])

tokenizer: CLIPTokenizerFast = CLIPTokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
def text_transform(text: str | list[str]) -> torch.Tensor:
    batch_encoding = tokenizer.__call__(
        text,
        padding=True,
        return_tensors="pt"
    )
    return batch_encoding.input_ids

_T_Co = TypeVar("_T_Co", covariant=True)
class DataLoader(TorchDataLoader, Generic[_T_Co]):
    def __iter__(self)->Iterator[_T_Co]:
        return cast(Iterator[_T_Co],super().__iter__())


_Batch_T = TypeVar("_Batch_T",bound=SDBatch, default = SDBatch, covariant=True)
class NarutoBlipDataModule(LightningDataModule, Generic[_Batch_T]):
    model_id = "lambdalabs/naruto-blip-captions"
    batch_type: Type[_Batch_T] = SDBatch

    def __init__(self,
                 image_transform_fn: Callable = None,
                 text_transform_fn: Callable=None,
                 train_ratio:float = 0.9,
                 num_workers: int=8,
                 batch_size = 8
                 ):
        super().__init__()
        self.image_transform = image_transform_fn or image_transform
        self.text_transform = text_transform_fn or text_transform

        assert train_ratio > 0
        self.train_ratio = min(1.0, train_ratio)
        if self.train_ratio<1.0:
            self.val_dataset: ArrDataset | None = None
            def _f(obj:NarutoBlipDataModule):
                return obj._return_dataloader(obj.val_dataset)
            self.val_dataloader = MethodType(_f, self)

        self.dataset: ArrDataset|None = None
        self.train_dataset:ArrDataset|None = None

        self.num_workers = num_workers
        self.batch_size = batch_size


    def _collate_fn(self, batch_l: list[dict])->_Batch_T:
        return self.batch_type.model_validate(default_collate(batch_l))

    def prepare_data(self) -> None:
        load_dataset(self.model_id)

    def setup(self, stage: str=None) -> None:
        dataset = load_dataset(self.model_id, split="train")
        def _transform(sample: dict):
            return dict(
                images=self.image_transform(sample["image"]),
                input_ids=self.text_transform(sample["text"])
            )
        dataset.set_transform(_transform)
        self.dataset = dataset

        if self.train_ratio>=1:
            self.train_dataset = self.dataset
        else:
            ds_dict = self.dataset.train_test_split(train_size=self.train_ratio)
            self.train_dataset = ds_dict["train"]
            self.val_dataset = ds_dict["test"]

    def _return_dataloader(self, dataset)->DataLoader[_Batch_T]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )

    def train_dataloader(self) -> DataLoader[_Batch_T]:
        return self._return_dataloader(self.train_dataset)

    # def val_dataloader(self) -> DataLoader[_Batch_T]:
    #     if self.val_dataset is None:
    #         return super().val_dataloader()
    #     else:


data_module = NarutoBlipDataModule(batch_size=8)
module = StableDiffusionModule(model_id=model_id)

dirpath = "sd_2_1_naruto"
checkpoints_path = f"{dirpath}/checkpoints"


def create_metric_checkpoint_callback(metric: MonitorKey, mode="min"):

    return ModelCheckpoint(
        dirpath=checkpoints_path,
        filename=f"{{epoch}}-{{step}}-min_{{{metric}:.4f}}",
        monitor=metric,
        mode=mode,
        save_top_k=1,
        save_on_train_epoch_end=True,
    )


callbacks = [create_metric_checkpoint_callback(metric) for metric in [MonitorKey.TRAIN_LOSS, MonitorKey.VAL_LOSS] ]
callbacks.extend(
    [
        ModelCheckpoint(
            dirpath=checkpoints_path,
            filename="{epoch}-{step}-last",
            save_top_k=1,
            save_on_train_epoch_end=True,
        ),
        TQDMProgressBar(leave=True),
        LearningRateMonitor("epoch")
    ]
)

loggers = [
    CSVLogger(dirpath, version=0),
]
trainer = Trainer(
    default_root_dir=dirpath,
    callbacks=callbacks,
    logger=loggers,
    # accelerator="cpu",
    # fast_dev_run=10,
    precision="bf16-true",
    max_steps=3000,
    num_sanity_val_steps=0
)

trainer.fit(module, datamodule=data_module)
