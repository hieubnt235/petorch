import warnings
from abc import ABC, abstractmethod
from typing import Unpack, cast, Type

import torch
from pydantic import PositiveInt, NonNegativeFloat, BaseModel, PositiveFloat
from torch import nn

from petorch.adapter import (
    BaseAdapter,
    ValidateConfigKwargs,
    AdapterConfig,
    BaseAdaptedLayer,
)


class LoraAdapterConfig(AdapterConfig):
    rank: PositiveInt = 8
    alpha: PositiveFloat = 16
    dropout: NonNegativeFloat = 0.1
    bias: bool = False
    scale: PositiveFloat = 1.0


class BaseLoraAdapter(BaseAdapter, ABC):

    config_class = LoraAdapterConfig
    base_layer_class: Type[nn.Module]

    def __init__(
        self,
        base_layer: nn.Module,
        config: dict | BaseModel,
        **kwargs: Unpack[ValidateConfigKwargs],
    ):
        assert (
            issubclass((blc := getattr(self, "base_layer_class", nn.Module)), nn.Module)
            and type(blc) != nn.Module
        ), f"Subclass must declare `base_layer_class` as a subclass of {nn.Module} but not be it. Got {blc}"

        assert isinstance(
            base_layer, self.base_layer_class
        ), f"Base layer must has type {self.base_layer_class}, got {type(base_layer)}."

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

    # ---Abstract methods---

    @abstractmethod
    def _init_lora_layers(self) -> None:
        """Override this method to change `self.lora_A` and `self.lora_B`.
        The `bias` of lora B should depend on the `is_bias` attribute.
        """

    @abstractmethod
    def get_delta_weight(self) -> torch.Tensor:
        """
        Get lora delta weight, Note that this is weight only, the merging process need also bias.
        Returns:
            Tensor with shape (base_layer.out_features, base_layer.in_features)
        """
        pass

    # ---Optional override---

    def get_delta_bias(self) -> torch.Tensor | None:
        if self.is_bias:
            return self.lora_B.bias * self.scaling
        return None

    def get_delta(self, batch_input: torch.Tensor) -> torch.tensor:
        return self.lora_B(self.lora_A(self.lora_dropout(batch_input))) * self.scaling

    # ---Properties---

    @property
    def config(self) -> LoraAdapterConfig:
        return cast(LoraAdapterConfig, super().config)

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


class LoraAdaptedLayer(BaseAdaptedLayer):

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
                assert isinstance(adapter, BaseLoraAdapter)
                base_output += adapter.get_delta(batch_input)

            return base_output

    def _validate_adapter(self, adapter: BaseAdapter, *args, **kwargs) -> BaseAdapter:
        if isinstance(adapter, BaseLoraAdapter) and (
            adapter.base_layer == self.base_layer
        ):
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
                assert isinstance(adapter, BaseLoraAdapter)
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
            assert isinstance(adapter, BaseLoraAdapter)

            base_layer = cast(nn.Linear, self.base_layer)
            base_layer.weight -= adapter.get_delta_weight()

            if (delta_bias := adapter.get_delta_bias()) is not None:
                base_layer.bias -= delta_bias
            self._merged_adapter_names.remove(name)
