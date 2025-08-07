from typing import Unpack, cast

import torch
from mlflow.types.chat import BaseModel
from pydantic import PositiveInt, NonNegativeFloat
from torch import nn

from petorch.adapter import (
    BaseAdapter,
    ValidateConfigKwargs,
    AdapterConfig,
    BaseAdaptedLayer,
)


# from peft.tuners.lora import Linear, LoraModel
# from peft import PeftModel


class LoraLinearAdapterConfig(AdapterConfig):
    rank: PositiveInt = 8
    alpha: PositiveInt = 16
    dropout: NonNegativeFloat = 0.1
    bias: bool = False


class LoraLinearAdapter(BaseAdapter):

    config_class = LoraLinearAdapterConfig

    def __init__(
        self,
        base_layer: nn.Linear,
        config: dict | BaseModel,
        **kwargs: Unpack[ValidateConfigKwargs],
    ):
        assert isinstance(
            base_layer, nn.Linear
        ), f"Base layer must has type {nn.Linear}, got {type(base_layer)}."
        super().__init__(base_layer, config, **kwargs)

        self.scale = getattr(self.config, "scale", None) or 1

        # Modules
        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        """lora_A.weight has shape (rank, base_layer.in_features)"""
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=self.is_bias)
        """lora_B.weight has shape (base_layer.out_features, rank)"""
        self.lora_dropout = (
            nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity()
        )

    @property
    def is_bias(self) -> bool:
        return self.config.bias and (cast(nn.Linear, self.base_layer).bias is not None)

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

    def get_delta(self, batch_input: torch.Tensor) -> torch.tensor:
        return self.lora_B(self.lora_A(self.lora_dropout(batch_input))) * self.scaling

    def get_delta_weight(self) -> torch.Tensor:
        """
        Get lora delta weight, Note that this is weight only, the merging process need also bias.
        Returns:
            Tensor with shape (base_layer.out_features, base_layer.in_features)
        """
        delta_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        assert delta_weight.shape == self.base_layer.weight.shape
        return delta_weight

    def get_delta_bias(self) -> torch.Tensor | None:
        if self.is_bias:
            return self.lora_B.bias * self.scaling
        return None


class LoraLinearAdaptedLayer(BaseAdaptedLayer):

    def forward(self, batch_input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        If merged, return `base_layer.forward` only, else all `active_adapters` will be added to the final results.
        Args:
            batch_input:
            *args:
            **kwargs:

        Returns:
            Result typed torch.Tensor
        """
        assert len(self.adapter_names) > 0
        base_output = self.base_layer(batch_input)

        # If merged, return
        if self.is_merged:
            return base_output
        else:
            for adapter in self.active_adapters.values():
                assert isinstance(adapter, LoraLinearAdapter)
                base_output += adapter.get_delta(batch_input)

        return base_output

    def _validate_adapter(self, adapter: BaseAdapter, *args, **kwargs) -> BaseAdapter:
        if isinstance(adapter, LoraLinearAdapter):
            return adapter
        else:
            raise ValueError(
                f"Adapter `{adapter}` is not the valid or supported one for AdaptedLayer `{self.__class__}`."
            )

    def _merge(
        self, adapter_names: str | list[str] | None = None, *args, **kwargs
    ) -> None:
        adapter_names = adapter_names or self.adapter_names
        adapter_names = (
            [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
        )

        for name in adapter_names:
            if name not in self._merged_adapter_names:
                adapter = self.get_adapter(name)
                assert isinstance(adapter, LoraLinearAdapter)
                assert self.base_layer == adapter.base_layer

                base_layer = cast(nn.Linear, self.base_layer)
                base_layer.weight += adapter.get_delta_weight()

                if (delta_bias := adapter.get_delta_bias()) is not None:
                    base_layer.bias += delta_bias
                self._merged_adapter_names.append(name)

    def _unmerge(
        self, adapter_names: str | list[str] | None = None, *args, **kwargs
    ) -> None:
        adapter_names = adapter_names or self.adapter_names
        adapter_names = (
            [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
        )
        # pre-check
        for name in adapter_names:
            if name not in self._merged_adapter_names:
                raise ValueError(f"Adapter named `{name}` haven't merged.")

        for name in adapter_names:
            adapter = self.get_adapter(name)
            assert isinstance(adapter, LoraLinearAdapter)

            base_layer = cast(nn.Linear, self.base_layer)
            base_layer.weight -= adapter.get_delta_weight()

            if (delta_bias := adapter.get_delta_bias()) is not None:
                base_layer.bias -= delta_bias
            self._merged_adapter_names.remove(name)
