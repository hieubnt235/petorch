from typing import cast

import torch
from pydantic import PositiveInt, NonNegativeFloat, BaseModel
from torch import nn

from petorch.adapter import BaseAdaptedModelConfig, BaseAdapter, AdapterConfig


class DummyAdapterConfig(AdapterConfig):
    rank: PositiveInt
    alpha: PositiveInt
    dropout: NonNegativeFloat
    adapter_name: str


class DummyAdapter(BaseAdapter):
    """
    Dummy LinearLora
    """
    config_class = DummyAdapterConfig
    
    def __init__(self, base_layer: nn.Linear, config: "BaseAdaptedModelConfig"):
        assert isinstance(
            base_layer, nn.Linear
        ), f"Base layer must has type {nn.Linear}, got {type(base_layer)}."
        super().__init__(base_layer, config)

        self.lora_A = nn.Linear(base_layer.in_features, self.rank)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features)
        self.lora_dropout = nn.Dropout(self.dropout)

        self.scale = getattr(self.config, "scale", None) or 1

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
    def scaling(self) -> float:
        return self.scale * self.alpha / self.rank

    def forward(self, batch_input: torch.Tensor, **kwargs) -> torch.Tensor:
        output = self.base_layer(batch_input)
        return (
            output
            + self.lora_B(self.lora_A(self.lora_dropout(batch_input))) * self.scaling
        )


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (8, 8)
        self.conv = nn.Conv2d(3, 3, 3, padding="same")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 3, 256)
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x: torch.Tensor):
        assert x.shape[2:] == self.input_shape, f"{x.shape[:2]}-{self.input_shape}"
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class NestedDummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (8, 8)
        self.conv = nn.Conv2d(3, 3, 3, padding="same")
        self.sub_model = Dummy()
        self.fc = nn.Linear(100, 16)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.sub_model(x)
        x = self.fc(x)
        return x


class DummyModelConfig(BaseAdaptedModelConfig):
    rank: PositiveInt = 8
    alpha: PositiveInt = 16
    dropout: NonNegativeFloat = 0.1

    def dispatch_adapter(
        self, fpname: str, base_layer: nn.Module, *args, **kwargs
    ) -> BaseAdapter | None:
        if isinstance(base_layer, nn.Linear):
            return DummyAdapter(cast(nn.Linear, base_layer), self)
