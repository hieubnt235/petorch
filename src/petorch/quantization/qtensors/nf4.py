from typing import Any, Self, Callable, Sequence

import torch
from torch import Tensor
from torchao.quantization import Int8WeightOnlyConfig
from torchao.dtypes import NF4Tensor as _NF4Tensor, AffineQuantizedTensor
from bitsandbytes.nn import Params4bit
from pydantic import PositiveInt
from .base import QTensor, QTensorConfig, QConfig_T, QTensorState


# TODO, create the abstract for quantization, and the dispatch API


class NF4Config(QTensorConfig):
    block_size: PositiveInt = 64
    scaler_block_size: PositiveInt = 256

class NF4State(QTensorState):
    pass

class NF4Tensor(QTensor[NF4Config, QTensorState]):
    q_config_cls = NF4Config
    q_state_cls = NF4State

    @classmethod
    def __torch_function__(
        cls, func: Callable[..., Any], types: Sequence[type[Any]], args=(), kwargs=None
    ) -> Any:
        pass
    
    @classmethod
    def from_high_precision(cls, hp_tensor: Tensor, config: NF4Config) -> Self:
        return cls()
    
    def get_high_precision(self) -> Tensor:
        pass

    