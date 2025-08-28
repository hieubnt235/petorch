from abc import ABC, abstractmethod
from types import MethodType
from typing import Callable, Protocol, TypeVar, Generic, Iterator, cast, TYPE_CHECKING, Self, TypeAlias, \
    Sequence, Any

from lightning import LightningDataModule, LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset, random_split

from petorch import logger

if TYPE_CHECKING:
    from pydantic import BaseModel, ConfigDict
    import lightning as pl
    import torch

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


Sample_T_Co = TypeVar("Sample_T_Co", covariant=True)
Batch_T = TypeVar("Batch_T", bound=DataBatch)
Module_T = TypeVar("Module_T", bound=LightningModule, covariant=True)
SampleTransform: TypeAlias = Callable[[Sample_T_Co], Sample_T_Co]
BatchTransform: TypeAlias = Callable[[Sequence[Sample_T_Co]], Batch_T]

class DatasetType(Protocol[Sample_T_Co]):
    def __getitem__(self, index: int) -> Sample_T_Co:
        ...
    def __len__(self) -> int:
        ...

class TransformWrapperDataset(TorchDataset[Sample_T_Co]):

    def __init__(self, dataset: DatasetType[Sample_T_Co], transform: SampleTransform[Sample_T_Co]):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item: int)->Sample_T_Co:
        return self.transform(self.dataset[item])

    def __getattr__(self, item: str)->Any:
        return getattr(self.dataset, item)

class DataLoader(TorchDataLoader[Sample_T_Co], Generic[Sample_T_Co, Batch_T]):
    def __iter__(self) -> Iterator[Batch_T]: # type: ignore[override]
        return cast(Iterator[Batch_T], super().__iter__())

class BaseDataModule(LightningDataModule, Generic[Sample_T_Co,Batch_T, Module_T], ABC):
    module_type: type[Module_T]
    batch_type: type[Batch_T]

    val_dataloader: Callable[[], Any]
    """This is a hack to disable the check for overriding of lightning, also pass the method assignment checking of mypy.
    This have no affect at runtime, just ignore it.
    """

    def __init_subclass__(cls, **kwargs)->None:
        assert hasattr(cls, "module_type"),"`module_type` must be specified."
        assert isinstance(getattr(cls,"module_type"), LightningModule), "`module_type` must be subclass of `LightningModule`.`"

        assert hasattr(cls, "batch_type"),"`batch_type` must be specified."
        assert isinstance(getattr(cls,"batch_type"), DataBatch), "`batch_type` must be subclass of `DataBatch`."

    def __init__(
        self,
        dataset_factory: Callable[[],DatasetType[Sample_T_Co]] |None  = None,
        *,
        dataset:DatasetType[Sample_T_Co] |None = None,
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
        self._dataset_factory:Callable[...,DatasetType[Sample_T_Co]] = dataset_factory
        self._dataset: DatasetType[Sample_T_Co] |None = None

        super().__init__()
        assert train_ratio > 0
        self.train_ratio = min(1.0, train_ratio)
        if self.train_ratio < 1.0:
            self.val_dataset: DatasetType[Sample_T_Co] | None = None

            # Patch it
            def _f(obj: BaseDataModule[Sample_T_Co,Batch_T, Module_T])->DataLoader[Sample_T_Co,Batch_T]:
                assert obj.val_dataset is not None
                return obj._return_dataloader(obj.val_dataset)
            self.val_dataloader = MethodType(_f, self)

        self.train_dataset: DatasetType[Sample_T_Co] | None = None

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.transform_fn: SampleTransform[Sample_T_Co] | None = None

    @property
    def module(self)-> Module_T:
        if self.trainer is None:
            raise ValueError("Trainer have not yet been set. You must pass this class in Trainer.fit instead of call manually. ")
        module =  self.trainer.lightning_module
        assert isinstance(module, self.module_type), (f"The `module_type` of {self.__class__.__name__} is {self.module_type}. "
                                                      f"But {module} is {module.__class__.__name__}. ")
        return module

    @property
    def dataset(self)->DatasetType[Sample_T_Co]:
        assert self._dataset is not None, "`dataset` has not yet been set."
        return self._dataset

    def _setup_with_module_and_dataset(self)->None:
        """Sometime setup need module or dataset information. So it should be setup at running,
        when you can access model through `self.module` and dataset through `self.dataset` properties.
        """
        pass

    @abstractmethod
    def _collate_fn(self,samples: Sequence[Sample_T_Co])->Batch_T:
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
                cast(TorchDataset[Sample_T_Co], self._dataset),
                [
                    self.train_ratio,
                    1- self.train_ratio
                ]
            )
            logger.info(
                f"Dataset lengths: train={len(self.train_dataset)}, val={len(self.val_dataset)}")
        self._setup_with_module_and_dataset()

    def _return_dataloader(self, dataset:DatasetType[Sample_T_Co]) -> DataLoader[Sample_T_Co,Batch_T]:
        return DataLoader[Sample_T_Co, Batch_T](
            cast(TorchDataset[Sample_T_Co],dataset),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader[Sample_T_Co,Batch_T]:
        assert self.train_dataset is not None
        return self._return_dataloader(self.train_dataset)

    def transfer_batch_to_device(
        self, batch: Batch_T, device: torch.device|str|None, dataloader_idx: int
    ) -> Batch_T:
        return batch.to(device=device, dtype=cast(torch.dtype,self.module.dtype))

