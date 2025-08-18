from enum import StrEnum
from typing import Any, Self, cast, Optional, Sequence

import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    DDPMScheduler
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.optimization import get_cosine_schedule_with_warmup
from lightning import LightningModule
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    OptimizerLRScheduler,
    OptimizerLRSchedulerConfig,
    LRSchedulerConfigType,
    LRSchedulerTypeUnion,
)
from loguru import logger
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor, IntTensor, FloatTensor
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTokenizerFast,
    CLIPImageProcessor,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling


class SDBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    input_ids: torch.Tensor
    images: torch.Tensor

    @model_validator(mode="after")
    def check_size(self) -> Self:
        input_ids = self.input_ids
        images = self.images
        assert input_ids.size(0) == images.size(0)

        assert not torch.is_floating_point(input_ids)


        assert torch.is_floating_point(images)
        assert images.size(1) == 3
        assert len(images.shape) == 4
        assert (self.images.max()<=1.0).all() and (self.images.min()>=-1.0).all()


        return self

    def to(self, device = None, dtype=None, **kwargs)->Self:
        return self.__class__(
            input_ids = self.input_ids.to(device = device, **kwargs),
            images = self.images.to(device=device, dtype=dtype, **kwargs),
        )

    def __len__(self)->int:
        return self.images.size(0)

class PredictionType(StrEnum):
    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"


class MonitorKey(StrEnum):
    TRAIN_LOSS = "train_loss"
    VAL_LOSS = "val_loss"
    LRS = "lrs"

# noinspection PyUnresolvedReferences
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

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.addition_kwargs = addition_kwargs

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

    def create_timesteps(self, size: Sequence[int]|int) -> IntTensor:
        size = size if isinstance(size, Sequence) else [size]
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            size,
            device=self.device,
            dtype=torch.int32,
        )
        return cast(IntTensor,timesteps)

    def create_noises(self, size: Sequence[int]):
        return torch.randn(size, dtype=self.dtype, device=self.device)


    def forward_batch_loss(
        self, batch: SDBatch, batch_index: int, **kwargs
    ) -> torch.Tensor:

        latents = self.vae_encode(batch.images)
        noises = self.create_noises(latents.shape)
        timesteps = self.create_timesteps(latents.shape[0])
        noisy_latents = self.scheduler.add_noise(latents, noises, timesteps)

        text_encoder_output: BaseModelOutputWithPooling = self.text_encoder(batch.input_ids)
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
            target = latents
        elif pred_type == PredictionType.V_PREDICTION:
            target = self.scheduler.get_velocity(latents, noises, timesteps)
        else:
            raise RuntimeError(f"The `prediction_type`=`{pred_type}` is not supported.")

        loss = self.loss_fn(model_pred, target)
        return loss

    def get_lrs(self)->dict:
        optims = self.optimizers()
        if not isinstance(optims, Sequence):
            optims = [optims]
        lrs = {}
        for i, optim in enumerate(optims):
            for j, pg in enumerate(optim.optimizer.param_groups):
                lrs[f"optim_{i}-pg_{j}_lr"] = pg["lr"]
        return lrs

    # Lightning Hooks
    def transfer_batch_to_device(self, batch: SDBatch, device: torch.device, dataloader_idx: int) -> SDBatch:
        return batch.to(device=device, dtype = self.dtype)

    def _log_step(self, key: MonitorKey, value: Any) -> None:
        self.log(
            key, value,
            prog_bar=True,
            logger=False,
            on_step = True ,
            on_epoch=False,
        )

    def _log_epoch(self, key: MonitorKey, value: Any, **kwargs) -> None:
        self.log(
            key, value,
            prog_bar=False,
            logger=True,
            on_step = False ,
            on_epoch=True,
            **kwargs
        )


    def training_step(
        self, batch: SDBatch, batch_index: int, **kwargs: Any
    ) -> STEP_OUTPUT:
        loss = self.forward_batch_loss(batch, batch_index, **kwargs)
        self._log_step(MonitorKey.TRAIN_LOSS+f"_step", loss)
        self._log_epoch(MonitorKey.TRAIN_LOSS, loss, batch_size=len(batch))
        return loss

    # def on_train_batch_end(
    #     self, outputs: STEP_OUTPUT, batch: SDBatch, batch_index: int
    # ) -> None:
    #     # outputs = outputs["loss"].cpu() # dont know why this is a dict instead of a tensor, the `on_validation_batch_end` still receive tensor.
    #     # self._store_loss(MonitorKeys.TRAIN_LOSS, outputs)
    #     # self.log("train_loss", outputs, logger=False,prog_bar=True)
    #     self.log_dict(
    #         self.get_lrs(),
    #         on_step=True,
    #         on_epoch=False,
    #         prog_bar=True,
    #         logger= False
    #     ) # Stream lr on progress bar only.

    # def on_train_epoch_end(self) -> None:
    #     """
    #     Note that this hook will be called after `on_validation_epoch_end` or `on_validation_end`.
    #     So that can be called to process both train and val at once.
    #     """
    #     # loss_dict = dict(
    #     #     train_loss = self._hook_storage.pop(MonitorKeys.TRAIN_LOSS),
    #     #     val_loss = self._hook_storage.pop(MonitorKeys.VAL_LOSS),
    #     #     **self.get_lrs()
    #     # )
    #     self.log_dict(
    #         self.get_lrs(),
    #         logger=False, prog_bar=False, sync_dist=True)
    #     # logger.debug(f"Logged_metric epoch end: {self.trainer.logged_metrics}")
    #     # logger.debug(f"callback_metrics epoch end: {self.trainer.callback_metrics}")


    def validation_step(
        self, batch: SDBatch, batch_index: int, **kwargs: Any
    ) -> STEP_OUTPUT:
        loss = self.forward_batch_loss(batch, batch_index, **kwargs)
        self._log_step(MonitorKey.VAL_LOSS+f"_step", loss)
        self._log_epoch(MonitorKey.VAL_LOSS, loss, batch_size=len(batch), sync_dist=True)
        return loss

    # def on_validation_batch_end(
    #     self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    # ) -> None:
    #     """In the hook loop, this method call directly after `validation_step`, so you can
    #     just implement everything in the ` validation_step ` method. Separate now just by personal.
    #     See Also: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks
    #     """
    #     outputs = outputs.cpu()
    #     self.log("val_loss", outputs)
    #     # self._store_loss(MonitorKeys.VAL_LOSS, outputs)

    # def on_validation_epoch_end(self) -> None:
    #     """
    #     This step is performed in `on_train_epoch_end`.
    #     """
    #     # loss_key = MonitorKeys.VAL_LOSS
    #     # mean_loss = cast(Tensor, self._hook_storage[loss_key]).mean().item()
    #     # self.log(loss_key, mean_loss, logger=True, prog_bar=True, sync_dist=True)
    #     pass
    #
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
        current_step = self.global_step-1
        step_freq = 1

        logger.debug(f"Total train steps={num_train_steps},current step={current_step}.")

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


