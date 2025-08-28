from abc import ABC, abstractmethod
from types import MethodType
from typing import Callable, Protocol, TypeVar, Generic, TypedDict, Iterator, cast, TYPE_CHECKING, Self

from lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader, default_collate

if TYPE_CHECKING:
    from pydantic import BaseModel

class DataBatch(BaseModel, ABC):

    @abstractmethod
    def to(self, device=None, dtype=None, **kwargs) -> Self:
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


Sample_T_Co = TypeVar("Sample_T_Co", covariant=True)
Batch_T_Co = TypeVar("Batch_T_Co", bound=DataBatch, covariant=True)


class DatasetType(Protocol[Sample_T_Co]):
    def __getitem__(self, index: int) -> Sample_T_Co:
        ...
    def __len__(self) -> int:
        ...

class TransformWrapperDataset(Generic[Sample_T_Co]):

    def __init__(self, dataset: DatasetType[Sample_T_Co], transform: Callable[[Sample_T_Co], Sample_T_Co]):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item)->Sample_T_Co:
        return self.transform(self.dataset[item])

    def __getattr__(self, item):
        return getattr(self.dataset, item)


class DataLoader(TorchDataLoader[Sample_T_Co], Generic[Sample_T_Co, Batch_T_Co]):
    def __iter__(self) -> Iterator[Batch_T_Co]:
        return cast(Iterator[Batch_T_Co], super().__iter__())


class DataModule(LightningDataModule, Generic[Sample_T_Co,Batch_T_Co]):
    batch_type: type[Batch_T_Co]

    def __init__(
        self,
        dataset_factory: Callable[...,DatasetType[Sample_T_Co]] |None  = None,
        *,
        dataset:DatasetType[Sample_T_Co] |None = None,
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
        if dataset_factory is not None and dataset is not None:
            raise ValueError(
                "`dataset_factory` and `dataset` can not be given at the same time, choose one only. ")
        if dataset is not None:
            assert dataset_factory is None
            dataset_factory = lambda : dataset # fake factory

        self._dataset_factory = dataset_factory

        super().__init__()
        # If None, assign when trainer is assigned (in `setup`).
        self.image_transform = image_transform_fn
        self.text_transform = text_transform_fn

        assert train_ratio > 0
        self.train_ratio = min(1.0, train_ratio)
        if self.train_ratio < 1.0:
            self.val_dataset: DatasetType[Sample_T_Co] | None = None

            def _f(obj: DataModule):
                return obj._return_dataloader(obj.val_dataset)

            self.val_dataloader = MethodType(_f, self)

        self.dataset: DatasetType[Sample_T_Co] |None = None
        self.train_dataset: DatasetType[Sample_T_Co] | None = None

        self.num_workers = num_workers
        self.batch_size = batch_size

    def _collate_fn(self, batch_l: list[dict]) -> Batch_T_Co:
        return self.batch_type.model_validate(default_collate(batch_l))

    def prepare_data(self) -> None:
        self._dataset_factory()

    @property
    def lightning_module(self):
        if self.trainer is None:
            raise ValueError("Trainer have not yet been set. You must pass this class in Trainer.fit instead of call manually. ")
        return self.trainer.lightning_module

    def _setup_transforms(self):
        """This method must only be called after trainer was set."""
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

    def _transform(self, sample: ):
        return dict(
            images=self.image_transform(sample["image"]),
            input_ids=self.text_transform(sample["text"]),
        )

    def setup(self, stage: str = None) -> None:
        """This is call by Trainer, if you call manually, make sure it does not have any process access to self.Trainer,
         such as auto find for resize sample (see `_setup_transforms`), means that if you pass your custom resize, it will run well.
        """
        dataset = self._dataset_factory()
        self._setup_transforms()
        self.dataset = dataset

        if self.train_ratio >= 1:
            self.train_dataset = self.dataset
            logger.info(
                f"Dataset lengths: train={len(self.train_dataset)}, val=None")
        else:
            self.train_dataset, self.val_dataset  = random_split(
                cast(TorchDataset, dataset),
                [
                    self.train_ratio,
                    1- self.train_ratio
                ]
            )
            logger.info(
                f"Dataset lengths: train={len(self.train_dataset)}, val={len(self.val_dataset)}")

    def _return_dataloader(self, dataset) -> DataLoader[Batch_T_Co]:
        return DataLoader(
            TransformWrapperDataset(dataset=dataset,transform= self._transform),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader[Batch_T_Co]:
        return self._return_dataloader(self.train_dataset)

    def transfer_batch_to_device(
        self, batch: SDBatch, device: torch.device, dataloader_idx: int
    ) -> SDBatch:
        return batch.to(device=device, dtype=self.lightning_module.dtype)

