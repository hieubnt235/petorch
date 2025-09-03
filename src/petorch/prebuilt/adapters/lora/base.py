import math
import warnings
from abc import ABC, abstractmethod
from typing import (Unpack, cast, override, )

import torch
from pydantic import PositiveInt, NonNegativeFloat, BaseModel, PositiveFloat
from torch import nn

from petorch.adapter import (BaseAdapter, ValidateConfigKwargs, AdapterConfig, BaseModule_T, )


class LoraAdapterConfig(AdapterConfig):
    rank: PositiveInt = 8
    alpha: PositiveFloat = 16
    dropout: NonNegativeFloat = 0.1
    bias: bool = False
    scale: PositiveFloat = 1.0


class BaseLoraAdapter(BaseAdapter[BaseModule_T], ABC):

    config_class = LoraAdapterConfig

    @override
    def __init__(
        self,
        base_layer: BaseModule_T,
        config: dict | BaseModel,
        **kwargs: Unpack[ValidateConfigKwargs],
    ):

        super().__init__(base_layer, config, **kwargs)

        self.lora_dropout = (
            nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity()
        )

        self.lora_A: nn.Module | None = None
        self.lora_B: nn.Module | None = None
        self._init_lora_layers()
        if not (
            isinstance(self.lora_A, nn.Module) and isinstance(self.lora_B, nn.Module)
        ):
            raise ValueError(
                f"The derived method `_init_lora_layers` must be init the `lora_A` and `lora_B` attributes to `torch.nn.Module`."
                f"Got `{self.lora_A}` and `{self.lora_B}`."
            )
        if (b := getattr(self.lora_A, "bias", None)) is not None:
            raise ValueError(f"Not allow bias in `lora_A`. Got bias=`{b}`.")

        if self.is_lora_B_bias and not self.is_bias:
            warnings.warn(
                f"Unexpected behavior: `lora_B` has bias while base or config does not have (by checking `is_bias` property)."
                f"The bias of `lora_B` should depend on the `is_bias` property."
            )

        self.reset_parameters()

    @override
    def get_delta_bias(self) -> torch.Tensor | None:
        if self.is_bias:
            return self.lora_B.bias * self.scaling
        return None
    
    @override
    def get_delta(self, batch_input: torch.Tensor) -> torch.tensor:
        return self.lora_B(self.lora_A(self.lora_dropout(batch_input))) * self.scaling

    
    # ---Abstract methods---

    @abstractmethod
    def _init_lora_layers(self) -> None:
        """Override this method to change `self.lora_A` and `self.lora_B`.
        The `bias` of lora B should depend on the `is_bias` attribute.
        """


    # ---Optional override---

    def reset_parameters(self):
        """Override this to init weight for modules. Default is standard for Conv and Linear Lora."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        if self.is_bias:
            nn.init.zeros_(self.lora_B.bias)


    # ---Properties---

    @property
    def is_lora_B_bias(self) -> bool:
        if not isinstance(self.lora_B, nn.Module):
            raise ValueError(
                "This property does not expected to access before `lora_B` is initialized. Use `is_bias` instead."
            )
        return (
            True
            if isinstance(getattr(self.lora_B, "bias", None), nn.Parameter)
            else False
        )

    @property
    def is_bias(self) -> bool:
        """
        Returns:
            True if all `base_layer`,`lora_B`, `config` has bias.
        """
        if self.lora_B is not None:
            assert isinstance(self.lora_B, nn.Module)
            is_lora_B_bias = self.is_lora_B_bias
        else:
            # For checking bias during init lora_B.
            is_lora_B_bias = True

        is_base_bias = (
            True
            if isinstance(getattr(self.base_layer, "bias", None), nn.Parameter)
            else False
        )
        return is_lora_B_bias and is_base_bias and self.config.bias

    @property
    def rank(self) -> int:
        return self.config.rank

    @property
    def alpha(self) -> float:
        return self.config.alpha

    @property
    def dropout(self) -> float:
        return self.config.dropout

    @property
    def scale(self) -> float:
        return self.config.scale

    @property
    def scaling(self) -> float:
        return self.scale * self.alpha / self.rank
