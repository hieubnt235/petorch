import types
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from pydantic import BaseModel
from torch import nn
from torchao import quantize_
from torchao.quantization import Int8WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig
from torchao.core.config import AOBaseConfig
from torchao.dtypes import to_nf4, NF4Tensor
from torchao.quantization import register_quantize_module_handler
from torchao.dtypes._nf4tensor_api import NF4WeightOnlyConfig
from .core import BaseModelQuantizationConfig, BaseQuantizer

# Model can only quantize one time, next time quantize will raise an error.

"""
For now, using the standard, simple torchao.quantize_ under the hood
"""

__QUANT_API_ATTR__ = "__PETORCH__QUANTIZATION_ATTRIBUTE_NAME__"


@dataclass
class NF4Config(AOBaseConfig):
    block_size: int = 64
    scaler_block_size: int = 256


def linear_module_repr(module: nn.Linear):
    return f"in_features={module.weight.shape[1]}, out_features={module.weight.shape[0]}, weight={module.weight}, dtype={module.weight.dtype}"


# For using with `quantize_` api
@register_quantize_module_handler(NF4Config)
def _nf4_weight_only_transform(
    module: torch.nn.Module,
    config: NF4Config,
) -> torch.nn.Module:
    new_weight = to_nf4(module.weight, config.block_size, config.scaler_block_size)
    module.weight = nn.Parameter(new_weight, requires_grad=False)  # Freeze
    module.extra_repr = types.MethodType(linear_module_repr, module)
    return module


class NF4Quantizer(BaseQuantizer):

    def _quantize(self, fqname: str, layer: nn.Module) -> nn.Module | None:
        nf4config = NF4Config(block_size=16, scaler_block_size=16)
        quantize_(layer, nf4config)

        return None


class TorchAONF4Config(BaseModelQuantizationConfig):

    def dispatch_quantizer(
        self, fpname: str, base_layer: nn.Module, *args, **kwargs
    ) -> BaseQuantizer | None:
        pass


class QuantizationAPI:

    @staticmethod
    def quantize(model: torch.nn.Module, config: BaseModelQuantizationConfig):
        """This method adapts to `torchao.quantize_`.


        Args:
            model:
            config:

        Returns:

        """
        pass

    @staticmethod
    def dequantize(self):
        pass

    @staticmethod
    def load_and_quantize_model():
        # TODO: after QLora example.
        pass
