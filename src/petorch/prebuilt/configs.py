from collections.abc import Callable
from typing import cast, Unpack

from pydantic import PositiveInt, NonNegativeFloat, BaseModel
from torch import nn

from petorch.adapter import (
    BaseAdaptedModelConfig,
    BaseAdapter,
    BaseAdaptedLayer,
    ValidateConfigKwargs,
)
from petorch.utilities import TorchInitMethod
from .lora.linear import LoraLinearAdapter, LoraLinearAdaptedLayer


class LoraLinearModelConfig(BaseAdaptedModelConfig):
    rank: PositiveInt = 8
    alpha: PositiveInt = 16
    dropout: NonNegativeFloat = 0.1
    bias: bool = False
    scale: PositiveInt = 1
    
    def dispatch_adapter(
        self,
        fqname: str,
        base_layer: nn.Module,
        *args,
        lora_init_method: TorchInitMethod | Callable | None = None,
        lora_a_init_method: TorchInitMethod | Callable | None = None,
        lora_b_init_method: TorchInitMethod | Callable | None = None,
        **kwargs: Unpack[ValidateConfigKwargs]
    ) -> BaseAdapter | None:
        if isinstance(base_layer, nn.Linear):
            lora_linear = LoraLinearAdapter(
                cast(nn.Linear, base_layer), cast(BaseModel, self), **kwargs
            )
            # init
            lora_a_init_method = lora_a_init_method or lora_init_method
            lora_b_init_method = lora_b_init_method or lora_init_method
            if lora_a_init_method:
                lora_a_init_method(lora_linear.lora_A.weight)
            if lora_b_init_method:
                lora_b_init_method(lora_linear.lora_B.weight)
                if lora_linear.is_bias:
                    lora_b_init_method(lora_linear.lora_B.bias)
                    
            return lora_linear
        return None

    def dispatch_adapted_layer(
        self, fqname: str, base_layer: nn.Module, *args, **kwargs
    ) -> BaseAdaptedLayer:
        return LoraLinearAdaptedLayer(base_layer,*args,**kwargs)
