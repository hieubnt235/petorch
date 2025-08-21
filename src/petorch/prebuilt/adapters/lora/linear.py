import math
from typing import cast

import torch
from torch import nn

from .base import BaseLoraAdapter


class LoraLinear(BaseLoraAdapter):
    base_layer_class = nn.Linear

    @property
    def base_layer(self) -> nn.Linear:
        # Override for typed hint.
        return cast(nn.Linear, super().base_layer)

    def _init_lora_layers(self) -> None:
        self.lora_A = nn.Linear(self.base_layer.in_features, self.rank, bias=False)
        """lora_A.weight has shape (rank, base_layer.in_features)"""

        self.lora_B = nn.Linear(
            self.rank, self.base_layer.out_features, bias=self.is_bias
        )
        """lora_B.weight has shape (base_layer.out_features, rank)"""
    
    
    def get_delta_weight(self) -> torch.Tensor:
        # delta_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        delta_weight = (
            torch.einsum("or, ri->oi", self.lora_B.weight, self.lora_A.weight)
            * self.scaling
        )
        assert delta_weight.shape == self.base_layer.weight.shape
        return delta_weight
