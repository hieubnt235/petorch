from collections.abc import Callable
from typing import Unpack

from pydantic import PositiveInt, NonNegativeFloat, PositiveFloat
from torch import nn

from petorch.adapter import (
    BaseModelAdaptionConfig,
    BaseAdaptedLayer,
    ValidateConfigKwargs,
)
from petorch.prebuilt.adapters.lora import (
    LoraLinear,
    LoraAdaptedLayer,
    BaseLoraAdapter,
    LoraEmbedding,
    LoraConv1d,
    LoraConv2d,
    LoraConv3d,
)
from petorch.utilities import TorchInitMethod

MODULE_ADAPTER_CLASSES_MAP = {
    nn.Linear: LoraLinear,
    nn.Conv1d: LoraConv1d,
    nn.Conv2d: LoraConv2d,
    nn.Conv3d: LoraConv3d,
    nn.Embedding: LoraEmbedding,
    nn.Module: None,
}


class LoraConfig(BaseModelAdaptionConfig):
    rank: PositiveInt = 8
    alpha: PositiveInt = 16
    dropout: NonNegativeFloat = 0.1
    bias: bool = False
    scale: PositiveFloat = 1.0

    def dispatch_adapter(
        self,
        fqname: str,
        base_layer: nn.Module,
        *args,
        lora_init_method: TorchInitMethod | Callable | None = None,
        lora_a_init_method: TorchInitMethod | Callable | None = None,
        lora_b_init_method: TorchInitMethod | Callable | None = None,
        **kwargs: Unpack[ValidateConfigKwargs]
    ) -> BaseLoraAdapter | None:
        adapter_cls = MODULE_ADAPTER_CLASSES_MAP.get(type(base_layer), None)

        if adapter_cls is not None:
            assert issubclass(adapter_cls, BaseLoraAdapter)
            adapter = adapter_cls(base_layer, self, **kwargs)
            # init
            lora_a_init_method = lora_a_init_method or lora_init_method
            lora_b_init_method = lora_b_init_method or lora_init_method
            if lora_a_init_method:
                lora_a_init_method(adapter.lora_A.weight)
            if lora_b_init_method:
                lora_b_init_method(adapter.lora_B.weight)
                if adapter.is_bias:
                    lora_b_init_method(adapter.lora_B.bias)
            return adapter

        return None

    def dispatch_adapted_layer(
        self, fqname: str, base_layer: nn.Module, *args, **kwargs
    ) -> BaseAdaptedLayer:
        return LoraAdaptedLayer(base_layer, *args, **kwargs)
