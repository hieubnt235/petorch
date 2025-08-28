from abc import ABC, abstractmethod
from types import MethodType
from typing import Callable, Protocol, TypeVar, Generic, Iterator, cast, Self, TypeAlias, \
    Sequence, Any

import torch
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset, random_split

from petorch import logger


class DataBatch(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, validate_default=True)

    @abstractmethod
    def to(self, device: torch.device|str|None=None, dtype: torch.dtype|str|None=None, **kwargs:dict[str, Any]) -> Self:
        """
        Convert data to model device and dtype.

        Args:
            device: model device
            dtype: model dtype
            **kwargs:

        Returns:
            Instance of DataBatch subclass

        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    def __getitem__(self, item:str)->Any:
        return getattr(self,item)


Sample_T = TypeVar("Sample_T")
Batch_T = TypeVar("Batch_T", bound=DataBatch)
TFSample_T= TypeVar("TFSample_T")
Module_T = TypeVar("Module_T", bound=LightningModule, covariant=True)
SampleTransform: TypeAlias = Callable[[Sample_T], TFSample_T]

class DatasetType(Protocol[Sample_T]):
    def __getitem__(self, index: int) -> Sample_T:
        ...
    def __len__(self) -> int:
        ...

class TransformWrapperDataset(TorchDataset[Sample_T]):

    def __init__(self, dataset: DatasetType[Sample_T], sample_transform: SampleTransform[Sample_T, TFSample_T]):
        self.dataset = dataset
        self.transform = sample_transform

    def __getitem__(self, item: int)->TFSample_T:
        return self.transform(self.dataset[item])

    def __len__(self)->int:
        return len(self.dataset)

class DataLoader(TorchDataLoader[Sample_T], Generic[Sample_T, Batch_T]):
    def __iter__(self) -> Iterator[Batch_T]: # type: ignore[override]
        return cast(Iterator[Batch_T], super().__iter__())

class BaseDataModule(LightningDataModule, Generic[Sample_T,Batch_T, Module_T, TFSample_T], ABC):
    sample_type: type[Sample_T]
    module_type: type[Module_T]
    batch_type: type[Batch_T]

    val_dataloader: Callable[[], Any]
    """This is a hack to disable the check for overriding of lightning, also pass the method assignment checking of mypy.
    This have no affect at runtime, just ignore it.
    """

    def __init_subclass__(cls, **kwargs)->None:
        assert hasattr(cls, "sample_type"),"`sample_type` must be specified."
        # assert issubclass(bt:=getattr(cls,"batch_type"), DataBatch), f"`batch_type` must be subclass of `DataBatch`. Got {type(bt)}."

        assert hasattr(cls, "batch_type"),"`batch_type` must be specified."
        assert issubclass(bt:=getattr(cls,"batch_type"), DataBatch), f"`batch_type` must be subclass of `DataBatch`. Got {type(bt)}."

        assert hasattr(cls, "module_type"),"`module_type` must be specified."
        assert issubclass(m:=getattr(cls,"module_type"), LightningModule), f"`module_type` must be subclass of `LightningModule`. Got {type(m)}."


    def __init__(
        self,
        dataset_factory: Callable[[],DatasetType[Sample_T]] | None  = None,
        *,
        dataset: DatasetType[Sample_T] | None = None,
        train_ratio: float = 0.8,
        num_workers: int = 8,
        batch_size: int =8,
    )->None:

        if dataset_factory is not None and dataset is not None:
            raise ValueError(
                "`dataset_factory` and `dataset` can not be given at the same time, choose one only. ")
        if dataset is not None:
            assert dataset_factory is None
            dataset_factory = lambda : dataset # fake factory

        assert callable(dataset_factory)
        self._dataset_factory:Callable[...,DatasetType[Sample_T]] = dataset_factory
        self._dataset: DatasetType[Sample_T] | None = None

        super().__init__()
        assert train_ratio > 0
        self.train_ratio = min(1.0, train_ratio)
        if self.train_ratio < 1.0:
            self.val_dataset: DatasetType[Sample_T] | None = None

            # Patch it
            def _f(obj: BaseDataModule[Sample_T,Batch_T, Module_T, TFSample_T])->DataLoader[Sample_T,Batch_T]:
                assert obj.val_dataset is not None
                return obj._return_dataloader(obj.val_dataset)
            self.val_dataloader = MethodType(_f, self)

        self.train_dataset: DatasetType[Sample_T] | None = None

        self.num_workers = num_workers
        self.batch_size = batch_size

    @property
    def module(self)-> Module_T:
        if self.trainer is None:
            raise ValueError("Trainer have not yet been set. You must pass this class in Trainer.fit instead of call manually. ")
        module =  self.trainer.lightning_module
        assert isinstance(module, self.module_type), (f"The `module_type` of {self.__class__.__name__} is {self.module_type}. "
                                                      f"But {module} is {module.__class__.__name__}. ")
        return module

    @property
    def dataset(self)->DatasetType[Sample_T]:
        assert self._dataset is not None, "`dataset` has not yet been set."
        return self._dataset

    # ---- Abstract class ----

    def _setup_with_module_and_dataset(self)->None:
        """Sometime setup need module or dataset information. So it should be setup at running,
        when you can access model through `self.module` and dataset through `self.dataset` properties.
        """
        pass

    def _sample_transform(self, sample: Sample_T)->TFSample_T:
        """Override this method to provide sample transform function, also known as transform."""
        assert self
        return sample

    @abstractmethod
    def _collate_fn(self, samples: Sequence[TFSample_T])->Batch_T:
        """Override this method to provide collation function, also known as batch transform."""
        ...

    # ---- Lightning hooks ----


    def prepare_data(self) -> None:
        self._dataset_factory()

    def setup(self, stage: str = TrainerFn.FITTING) -> None:
        """This is call by Trainer, if you call manually, make sure it does not have any process access to self.Trainer,
         such as auto find for resize sample (see `_setup_transforms`), means that if you pass your custom resize, it will run well.
        """
        self._dataset = self._dataset_factory()
        if self.train_ratio >= 1:
            self.train_dataset = self._dataset
            logger.info(
                f"Dataset lengths: train={len(self.train_dataset)}, val=None")
        else:
            self.train_dataset, self.val_dataset  = random_split(
                cast(TorchDataset[Sample_T], self._dataset),
                [
                    self.train_ratio,
                    1- self.train_ratio
                ]
            )
            logger.info(
                f"Dataset lengths: train={len(self.train_dataset)}, val={len(self.val_dataset)}")
        self._setup_with_module_and_dataset()


    def _return_dataloader(self, dataset:DatasetType[Sample_T]) -> DataLoader[Sample_T,Batch_T]:
        return DataLoader[Sample_T, Batch_T](
            TransformWrapperDataset(dataset,self._sample_transform),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader[Sample_T,Batch_T]:
        assert self.train_dataset is not None
        return self._return_dataloader(self.train_dataset)

    def transfer_batch_to_device(
        self, batch: Batch_T, device: torch.device|str|None, dataloader_idx: int
    ) -> Batch_T:
        return batch.to(device=device, dtype=cast(torch.dtype,self.module.dtype))

