from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn, Tensor, IntTensor, FloatTensor
from enum import StrEnum

from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutputWithPooling
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from lightning import LightningModule, LightningDataModule, Trainer
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    DDPMScheduler,
)
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTokenizerFast,
    CLIPImageProcessor,
)
from typing import Any, Self, cast, Optional
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    OptimizerLRScheduler,
    OptimizerLRSchedulerConfig,
    LRSchedulerConfigType,
    LRSchedulerTypeUnion,
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
)
from pydantic import BaseModel, ConfigDict, model_validator
import torch
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from loguru import logger


class SDBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_ids: torch.Tensor
    images: torch.Tensor

    @model_validator(mode="after")
    def check_size(self) -> Self:
        assert self.input_ids.size(0) == self.images.size(0)
        assert self.images.size(1) == 3
        return self


class PredictionType(StrEnum):
    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"


class MonitorKeys(StrEnum):
    TRAIN_LOSS = "train_loss"
    VAL_LOSS = "val_loss"


class StableDiffusionModule(LightningModule):
    """
    Attributes:
        unet:
        vae:
        text_encoder:
        tokenizer:
         scheduler:

    You can override all methods defined on this class to customize for your case.
    """

    def __init__(
        self,
        pipeline: StableDiffusionPipeline | None = None,
        *,
        model_id: str | None = None,
        vae: AutoencoderKL | None = None,
        unet: UNet2DConditionModel | None = None,
        text_encoder: CLIPTextModel | None = None,
        tokenizer: CLIPTokenizer | CLIPTokenizerFast | None = None,
        scheduler: (
            DDIMScheduler | DDPMScheduler | None
        ) = None,  # TODO: Make enum for typehint, hf so stupid
        feature_extractor: CLIPImageProcessor | None = None,
        **addition_kwargs,
    ):
        super().__init__()
        kwargs = dict(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            model_id=model_id,
            feature_extractor=feature_extractor,
        )
        # Delete all None values.
        del_k = []
        for k, v in kwargs.items():
            if v is None:
                del_k.append(k)
        for k in del_k:
            del kwargs[k]

        if pipeline is not None:
            assert isinstance(pipeline, StableDiffusionPipeline)
            if kwargs != {}:
                raise ValueError(
                    f"All arguments must be None when `pipeline` is given. Got `{kwargs}.`"
                )
        else:
            if kwargs.pop("model_id", None) is not None:
                pipeline = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
            else:
                # Does not require feature extractor.
                if "feature_extractor" not in kwargs:
                    kwargs["feature_extractor"] = None
                pipeline = StableDiffusionPipeline(
                    **kwargs,
                    requires_safety_checker=False,
                    safety_checker=None,
                    image_encoder=None,
                )

        self.unet: UNet2DConditionModel = pipeline.unet
        self.vae: AutoencoderKL = pipeline.vae
        self.text_encoder: CLIPTextModel = pipeline.text_encoder
        self.tokenizer: CLIPTokenizer | CLIPTokenizerFast = pipeline.tokenizer
        self.scheduler: DDIMScheduler | DDPMScheduler = pipeline.scheduler
        self.pipeline = pipeline

        self.addition_kwargs = addition_kwargs

        self._hook_storage: dict[Any, Any] = {}
        """
        Simple storage to store output of any hook methods. The logic of store, compose, clear,... depend on hook.
        Hook must take care about the collision.
        
        Example usecase: Store step loss for each batch, then calculate and log mean loss, then clear the storage.
        """

    @property
    def latents_values_scaling(self) -> float:
        """Default to 0.18215 and latents values should be multiplied after `vae.encode` and `divided` before `vae.decode`."""
        return self.vae.config.scaling_factor

    @property
    def latents_spatial_reduced_ratio(self) -> int:
        """The reduced ratio"""
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    def vae_encode(self, images: torch.Tensor) -> torch.Tensor:
        latent_dist: DiagonalGaussianDistribution = self.vae.encode(images).latent_dist
        latents = latent_dist.sample() * self.latents_values_scaling
        return latents

    def vae_decode(self, latents: Tensor) -> torch.Tensor:
        return self.vae.decode(
            cast(FloatTensor, latents / self.latents_values_scaling)
        ).sample

    def loss_fn(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reduction = self.addition_kwargs.get("reduction", "mean")
        return F.mse_loss(prediction, target, reduction=reduction)

    def forward_batch_loss(
        self, batch: SDBatch, batch_index: int, **kwargs
    ) -> torch.Tensor:
        latents = self.vae_encode(images := batch.images)
        noises = torch.rand_like(latents)
        timesteps = cast(
            IntTensor,
            torch.randint(
                0, self.scheduler.config.num_train_timesteps, (latents.size(0),)
            ),
        )
        noisy_latents = self.scheduler.add_noise(latents, noises, timesteps)

        text_encoder_output: BaseModelOutputWithPooling = self.text_encoder(
            input_ids := batch.input_ids
        )
        text_embeddings = text_encoder_output.last_hidden_state

        unet_output: UNet2DConditionOutput = self.unet(
            noisy_latents, timesteps, text_embeddings
        )
        model_pred = unet_output.sample

        if (
            pred_type := self.scheduler.config.prediction_type
        ) == PredictionType.EPSILON:
            target = noises
        elif pred_type == PredictionType.SAMPLE:
            target = images
        elif pred_type == PredictionType.V_PREDICTION:
            target = self.scheduler.get_velocity(latents, noises, timesteps)
        else:
            raise RuntimeError(f"The `prediction_type`=`{pred_type}` is not supported.")

        loss = self.loss_fn(model_pred, target)
        return loss

    def training_step(
        self, batch: SDBatch, batch_index: int, **kwargs: Any
    ) -> STEP_OUTPUT:
        loss = self.forward_batch_loss(batch, batch_index, **kwargs)
        return loss

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        assert isinstance(outputs, Tensor) and len(outputs.shape) == 0
        loss_k = MonitorKeys.TRAIN_LOSS
        outputs = outputs[None]

        if (losses := self._hook_storage.get(loss_k)) is None:
            self._hook_storage[loss_k] = outputs
        else:
            assert isinstance(losses, Tensor) and len(cast(Tensor, losses).shape) == 1
            self._hook_storage[loss_k] = torch.cat([losses, outputs], dim=0)

    def on_train_epoch_end(self) -> None:
        """
        Note that this hook will be called after `on_validation_epoch_end` or `on_validation_end`.
        So that can be called to process both train and val at once.
        """
        loss_key = MonitorKeys.TRAIN_LOSS
        mean_loss = cast(Tensor, self._hook_storage[loss_key]).mean().item()
        self.log(loss_key, mean_loss, logger=True, prog_bar=True)

    def validation_step(
        self, batch: SDBatch, batch_index: int, **kwargs: Any
    ) -> STEP_OUTPUT:
        loss = self.forward_batch_loss(batch, batch_index, **kwargs)
        return loss

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """In the hook loop, this method call directly after `validation_step`, so you can
        just implement everything in the ` validation_step ` method.
        See Also: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks
        """
        # Copy of `on_train_batch_end`. Don't make it abstract because maybe two methods have different process.

        assert isinstance(outputs, Tensor) and len(outputs.shape) == 0
        loss_k = MonitorKeys.VAL_LOSS
        outputs = outputs[None]

        if (losses := self._hook_storage.get(loss_k)) is None:
            self._hook_storage[loss_k] = outputs
        else:
            assert isinstance(losses, Tensor) and len(cast(Tensor, losses).shape) == 1
            self._hook_storage[loss_k] = torch.cat([losses, outputs], dim=0)

    def on_validation_epoch_end(self) -> None:
        loss_key = MonitorKeys.VAL_LOSS
        mean_loss = cast(Tensor, self._hook_storage[loss_key]).mean().item()
        self.log(loss_key, mean_loss, logger=True, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        This method only be called by Trainer, and Trainer will assign itself into this.
        So that you can extract training parameters through `self.trainer`.

        References:
            https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        num_train_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(0.02 * num_train_steps)
        current_step = self.trainer.global_step
        step_freq = 1

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=0.5,
            last_epoch=current_step,
        )

        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler=LRSchedulerConfigType(
                scheduler=lr_scheduler,
                name="default_sd_optimizer",
                # Updating the frequency of the scheduler ( call scheduler.step())
                interval="step",
                frequency=step_freq,
                # The value for scheduler to depend on to update
                # monitor=Metric.TRAIN_LOSS,
                # strict=True # Must have monitor value
            ),
        )


class SDTrainer(Trainer):
    pass


class StableDiffusionDataModule(LightningDataModule):
    def prepare_data(self) -> None:
        logger.debug("Preparing data...")

    def setup(self, stage: TrainerFn = TrainerFn.FITTING) -> None:
        logger.debug("Setting up data...")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dl = DataLoader(
            self.train_ds,
            batch_size=self.config.train_dl_cfg.batchsize,
            num_workers=self.config.train_dl_cfg.num_workers,
            collate_fn=self.collate_fn,
        )
        return dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.val_ds,
            batch_size=self.config.val_dl_cfg.batchsize,
            num_workers=self.config.val_dl_cfg.num_workers,
            collate_fn=self.collate_fn,
        )
        return dl

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.val_ds,
            batch_size=self.config.test_dl_cfg.batchsize,
            num_workers=self.config.test_dl_cfg.num_workers,
            collate_fn=self.collate_fn,
        )
        return dl

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.val_ds,
            batch_size=self.config.predict_dl_cfg.batchsize,
            num_workers=self.config.predict_dl_cfg.num_workers,
            collate_fn=self.collate_fn,
        )
        return dl


def sd_train(trainer: SDTrainer = None):
    pass
