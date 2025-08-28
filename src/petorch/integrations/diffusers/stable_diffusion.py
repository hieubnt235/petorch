from enum import StrEnum
from typing import Any, Self, cast, Sequence, Callable, Iterator, Literal

import PIL
import numpy as np
import torch
from diffusers import (StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler, )
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.optimization import get_cosine_schedule_with_warmup
from lightning import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import (STEP_OUTPUT, OptimizerLRScheduler, OptimizerLRSchedulerConfig,
                                               LRSchedulerConfigType, )
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor, IntTensor, FloatTensor
from torch import nn
from torch.optim import Optimizer
from torchmetrics import MeanMetric, Metric
from transformers import (CLIPTokenizer, CLIPTextModel, CLIPTokenizerFast, CLIPImageProcessor, )
from transformers.modeling_outputs import BaseModelOutputWithPooling

from petorch import logger


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
        assert (self.images.max() <= 1.0).all() and (self.images.min() >= -1.0).all()
        return self

    def to(self, device=None, dtype=None, **kwargs) -> Self:
        """

        Args:
            device:
            dtype:
            **kwargs:

        Returns:

        """
        # For not make a mistake in assignment checking (ex: batch sizes must be the same). I recreate new instance instead.
        return self.__class__(
            input_ids=self.input_ids.to(
                device=device, **kwargs
            ),  # This must always int64.
            images=self.images.to(device=device, dtype=dtype, **kwargs),
        )

    def __len__(self) -> int:
        return self.images.size(0)


class PredictionType(StrEnum):
    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"


class MetricKey(StrEnum):
    TRAIN_LOSS = "train_loss"
    VAL_LOSS = "val_loss"


OptimizerFactoryType = Callable[[Iterator[nn.Parameter]], Optimizer]
LRSchedulerFactoryType = Callable[[Optimizer, int, int], LRSchedulerConfigType]
"""Input is Optimizer instance, total number of optimizing steps during training and current step index."""


def default_optimizer_factory(
    params: Iterator[nn.Parameter], *, lr=1e-4, **kwargs
) -> Optimizer:
    return torch.optim.AdamW(params, lr=lr, **kwargs)


def default_lr_scheduler_factory(
    optimizer: Optimizer,
    max_optim_steps: int,
    current_step: int,
    *,
    num_warmup_steps: float | int = 0.05,
    interval: Literal["epoch", "step"] = "step",
    lr_freq: int = 1,
) -> LRSchedulerConfigType:
    """

    Args:
        optimizer:
        max_optim_steps:
        current_step:
        num_warmup_steps: should be from 0.02-0.2, lower for the model that already be pretrained well.
        interval:
        lr_freq:

    Returns:

    """
    max_optim_steps = int(max_optim_steps)
    if isinstance(num_warmup_steps, float):
        num_warmup_steps = int(num_warmup_steps * max_optim_steps)
    assert num_warmup_steps < max_optim_steps

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_optim_steps,
        num_cycles=0.5,
        last_epoch=current_step,
    )
    return LRSchedulerConfigType(
        scheduler=lr_scheduler,
        name="default_sd_optimizer",
        # Updating the frequency of the scheduler ( call scheduler.step())
        interval=interval,
        frequency=lr_freq,
        # The value for scheduler to depend on to update
        # monitor=Metric.TRAIN_LOSS,
        # strict=True # Must have monitor value
    )


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
        scheduler: DDIMScheduler | DDPMScheduler | None = None,
        feature_extractor: CLIPImageProcessor | None = None,
        optimizer_factory: None | OptimizerFactoryType = None,
        lr_scheduler_factory: None | LRSchedulerFactoryType = None,
        **addition_kwargs,
    ):
        """

        Args:
            pipeline:
            model_id:
            vae:
            unet:
            text_encoder:
            tokenizer:
            scheduler:
            feature_extractor:
            **addition_kwargs: Such as `metric_compute_on_cpu` or `train_loss_compute_on_cpu`.

        """
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

        # 1. Preprocess args

        ## Delete all None values.
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

        # 2. Components
        self.unet: UNet2DConditionModel = pipeline.unet
        self.vae: AutoencoderKL = pipeline.vae
        self.text_encoder: CLIPTextModel = pipeline.text_encoder
        self.tokenizer: CLIPTokenizer | CLIPTokenizerFast = pipeline.tokenizer
        self.scheduler: DDIMScheduler | DDPMScheduler = pipeline.scheduler
        self.pipeline = pipeline

        self.vae.requires_grad_(False)
        # self.text_encoder.requires_grad_(False)

        # 3. Loss fn
        self.loss_fn = nn.MSELoss(reduction="mean")

        # 4. Metrics
        metric_kwargs: dict[MetricKey, dict] = {
            MetricKey.TRAIN_LOSS: {},
            MetricKey.VAL_LOSS: {},
        }
        for k, v in addition_kwargs.items():
            if (prefix := k.split("_")[0]) in MetricKey:
                metric_kwargs[mkey].update({k.removeprefix(prefix + "_"): v})
            elif k.startswith("metric_"):
                for kw in metric_kwargs.values():
                    kw.update({k.removeprefix("metric_"): v})

        self.metrics = nn.ModuleDict(
            {
                MetricKey.TRAIN_LOSS: MeanMetric(**metric_kwargs[MetricKey.TRAIN_LOSS]),
                MetricKey.VAL_LOSS: MeanMetric(**metric_kwargs[MetricKey.VAL_LOSS]),
            }
        )

        # 5. Additions
        self.addition_kwargs = addition_kwargs
        self.optimizer_factory = optimizer_factory or default_optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory or default_lr_scheduler_factory

    @property
    def sample_size(self) -> int | tuple[int, int] | None:
        """The images size that use to train base diffusion model.
        Use this sample size for resizing the finetuning data images for more compatible with the pretrained one (But not required).
        """
        return self.unet.config.sample_size * self.latents_spatial_reduced_ratio

    @property
    def num_train_timesteps(self) -> int:
        return self.scheduler.config.num_train_timesteps

    @property
    def latents_values_scaling(self) -> float:
        """Default to 0.18215 and latents values should be multiplied after `vae.encode` and `divided` before `vae.decode`."""
        return self.vae.config.scaling_factor

    @property
    def latents_spatial_reduced_ratio(self) -> int:
        """The reduced ratio of vae latents"""
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    def vae_encode(self, images: torch.Tensor) -> torch.Tensor:
        latent_dist: DiagonalGaussianDistribution = self.vae.encode(images).latent_dist
        latents = latent_dist.sample() * self.latents_values_scaling
        return latents.to(device=self.device, dtype=self.dtype)

    def vae_decode(self, latents: Tensor) -> torch.Tensor:
        return self.vae.decode(
            cast(FloatTensor, latents / self.latents_values_scaling)
        ).sample

    def create_timesteps(self, size: Sequence[int] | int) -> IntTensor:
        size = size if isinstance(size, Sequence) else [size]
        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            size,
            device=self.device,
            dtype=torch.long,
        )
        return cast(IntTensor, timesteps)

    def create_noises(self, size: Sequence[int]):
        return torch.randn(size, dtype=self.dtype, device=self.device)

    def forward(self, *args: Any, **kwargs: Any) -> list[PIL.Image.Image] | np.ndarray:
        # todo: typehint input
        return self.pipeline.__call__(*args, **kwargs).images

    def forward_batch_loss(
        self, batch: SDBatch, batch_index: int, **kwargs
    ) -> torch.Tensor:

        latents = self.vae_encode(batch.images)
        noises = self.create_noises(latents.shape)
        timesteps = self.create_timesteps(latents.shape[0])
        noisy_latents = self.scheduler.add_noise(latents, noises, timesteps)

        text_encoder_output: BaseModelOutputWithPooling = self.text_encoder(
            batch.input_ids
        )
        text_embeddings = text_encoder_output.last_hidden_state

        unet_output: UNet2DConditionOutput = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings,
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

        assert model_pred.shape == target.shape
        loss = self.loss_fn(model_pred, target)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected at step {self.global_step}: {loss.item()}"
            )

        return loss

    def get_metric(self, key: MetricKey) -> Metric:
        return cast(Metric, self.metrics[key])

    # Lightning Hooks

    def training_step(
        self, batch: SDBatch, batch_index: int, **kwargs: Any
    ) -> STEP_OUTPUT:
        loss = self.forward_batch_loss(batch, batch_index, **kwargs)

        mkey = MetricKey.TRAIN_LOSS
        # IMPORTANCE: WHEN USE Metric instance, the batch_size in self.log is not use.
        # If you pass only value, make sure batch_size in self.log =1.
        self.get_metric(mkey).update(loss, len(batch))

        # Note that when the value is logged in step hook, logger only received after `Trainer.log_in_every_n_steps`
        self.log(
            f"{mkey}_step",
            loss,
            logger=True,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        """
        Note that this hook will be called after `on_validation_epoch_end` or `on_validation_end`.
        So that can be called to process both train and val at once.
        """
        train_metric = self.get_metric(MetricKey.TRAIN_LOSS)
        val_metric = self.get_metric(MetricKey.VAL_LOSS)
        self.log_dict(
            {
                MetricKey.TRAIN_LOSS: train_metric.compute(),
                MetricKey.VAL_LOSS: val_metric.compute(),
            },
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        train_metric.reset()
        val_metric.reset()

    def validation_step(
        self, batch: SDBatch, batch_index: int, **kwargs: Any
    ) -> STEP_OUTPUT:
        loss = self.forward_batch_loss(batch, batch_index, **kwargs)

        mkey = MetricKey.VAL_LOSS
        self.get_metric(mkey).update(loss, len(batch))
        # Note, this log is log on train progress bar. See `TQDMProgressBar` callback on this hook.
        # self.log(f"{mkey}_step",loss, logger=False, prog_bar=True, on_step=True, on_epoch=False)

        # For Trainer.validate return only.
        if self.trainer.state.fn == TrainerFn.VALIDATING:
            self.log(mkey, loss, on_epoch=True, on_step=False, batch_size=len(batch))

        return loss

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """In the hook loop, this method call directly after `validation_step`, so you can
        just implement everything in the ` validation_step ` method. Redefine here just to notices.
        See Also: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks
        """
        pass

    def on_validation_epoch_end(self) -> None:
        """This step is performed in `on_train_epoch_end`.Redefine here just to notices."""
        pass

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        This method only be called by Trainer, and Trainer will assign itself into this.
        So that you can extract training parameters through `self.trainer`.

        References:
            https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        """
        num_optim_steps = self.trainer.estimated_stepping_batches
        current_step = self.global_step - 1
        logger.debug(
            f"Total train steps={num_optim_steps},current step={current_step}."
        )

        optimizer = self.optimizer_factory(self.parameters())
        lr_scheduler = self.lr_scheduler_factory(
            optimizer, num_optim_steps, current_step
        )

        return OptimizerLRSchedulerConfig(
            optimizer=optimizer, lr_scheduler=lr_scheduler
        )
