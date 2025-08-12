from typing import Type, cast

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from .base import BaseLoraAdapter


class LoraConvNd(BaseLoraAdapter):
    base_layer_class: Type[_ConvNd]

    @property
    def base_layer(self) -> _ConvNd:
        return cast(_ConvNd, super().base_layer)

    @classmethod
    def _init_subclass_(cls, **kwargs):
        assert (
            issubclass(cls.base_layer_class, _ConvNd)
            and cls.base_layer_class != _ConvNd
        ), f"`base_layer_class` must be the subclass of `{_ConvNd}`(not be it also). Got `{cls.base_layer_class}`."

    @property
    def kernel_dim(self) -> int:
        return self.base_layer.weight.dim()

    def _init_lora_layers(self) -> None:
        bl = self.base_layer
        self.lora_A = self.base_layer_class(
            bl.in_channels, self.rank, bl.kernel_size, bl.stride, bl.padding, bias=False
        )
        kernel_size = stride = (1,) * (self.kernel_dim - 2)
        self.lora_B = self.base_layer_class(
            self.rank, bl.out_channels, kernel_size, stride, bias=self.is_bias
        )

    def get_delta_weight(self) -> torch.Tensor:
        assert isinstance(self.lora_A, self.base_layer_class) and isinstance(
            self.lora_B, self.base_layer_class
        )
        delta_weight = (
            torch.einsum(
                "o r ..., r i ... -> o i ...", self.lora_B.weight, self.lora_A.weight
            )
            * self.scaling
        )
        assert delta_weight.shape == self.base_layer.weight.shape
        return delta_weight


class LoraConv1d(LoraConvNd):
    base_layer_class = nn.Conv1d


class LoraConv2d(LoraConvNd):
    base_layer_class = nn.Conv2d


class LoraConv3d(LoraConvNd):
    base_layer_class = nn.Conv3d
