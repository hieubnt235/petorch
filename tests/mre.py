from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from random import randint
from typing import Sequence, Any, TypedDict, Type
from typing import Unpack

import torch
from mlflow.types.chat import BaseModel
from pydantic import ConfigDict, Field
from pydantic import PositiveInt, NonNegativeFloat, BaseModel
from torch import nn
from torch.nn.init import (
    normal_,
    uniform_,
    constant_,
    ones_,
    zeros_,
    eye_,
    dirac_,
    xavier_uniform_,
    xavier_normal_,
    kaiming_uniform_,
    kaiming_normal_,
    trunc_normal_,
    orthogonal_,
    sparse_,
)


class TorchInitMethod(Enum):
    uniform = uniform_
    normal = normal_
    constant = constant_
    ones = ones_
    zeros = zeros_
    eye = eye_
    dirac = dirac_
    xavier_uniform = xavier_uniform_
    xavier_normal = xavier_normal_
    kaiming_uniform = kaiming_uniform_
    kaiming_normal = kaiming_normal_
    trunc_normal = trunc_normal_
    orthogonal = orthogonal_
    sparse = sparse_


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


class ValidateConfigKwargs(TypedDict):
    strict: bool | None
    from_attributes: bool | None
    context: Any | None
    by_alias: bool | None
    by_name: bool | None


class AdapterConfig(BaseModel):
    adapter_name: str


class BaseAdapter(nn.Module):
    """
    The abstract class for one layer adapter.
    For example, it contains loraA and loraB matrices.

    Features:
        - Adapter declare dynamic adjustable hyperparameters (dahparams), such as scale, for higher-level set it (use set it).
        - Adapter know how to use that (dhparams) in forward method.

    """

    config_class: Type[AdapterConfig]

    def __init_subclass__(cls, **kwargs):
        assert issubclass(
            cls.config_class, AdapterConfig
        ), f"`config_class` must be the subclass of `AdapterConfig`. Got `{cls.config_class}`."

        if (
            field_info := cls.config_class.model_fields.get("adapter_name", None)
        ) is None:
            raise ValueError("The config_class must have attribute `adapter_name`.")

        if not issubclass(field_info.annotation, str):
            raise ValueError(
                f"Type hint of `adapter_name` in `AdapterConfig` must be `str`."
            )

    def __init__(
        self,
        base_layer: nn.Module,
        config: BaseModel | dict,
        **kwargs: Unpack[ValidateConfigKwargs],
    ):
        super().__init__()
        self._base_layer = [base_layer]  # Wrap to a list for not exposed as module
        self._config = self.config_class.model_validate(
            config, **self._get_validate_kwargs(config, **kwargs)
        )
        self._adapter_name: str = self._config.adapter_name

    @property
    def config(self) -> AdapterConfig:
        return self._config

    @property
    def base_layer(self) -> nn.Module:
        return self._base_layer[0]

    @property
    def name(self) -> str:
        return self._adapter_name

    def merge(self, *args, **kwargs):
        """This method will merge this adapter to base layer."""
        raise NotImplementedError

    def unmerge(self):
        """This method will unmerge this adapter to base layer."""
        raise NotImplementedError

    def __getitem__(self, item):
        return getattr(self.config, item)

    # noinspection PyMethodMayBeStatic
    def _get_validate_kwargs(self, config, **kwargs: dict):
        from_attributes = kwargs.pop("from_attributes", None)
        validate_kwargs = dict(
            strict=kwargs.pop("strict", None),
            from_attributes=True if not isinstance(config, dict) else from_attributes,
            context=kwargs.pop("context", None),
            by_alias=kwargs.pop("by_alias", None),
            by_name=kwargs.pop("by_name", None),
        )
        return validate_kwargs

    def _update_config(
        self,
        new_config: BaseModel | dict,
        **validate_kwargs: Unpack[ValidateConfigKwargs],
    ):
        new_config = self.config_class.model_validate(
            new_config, **self._get_validate_kwargs(new_config, **validate_kwargs)
        )
        assert (
            new_config.adapter_name == self.name
        ), f"New config's `adapter_name` does not match with the current one. `{new_config.adapter_name}` and `{self.name}`."
        self._config = new_config


class BaseAdaptedLayer(nn.Module, ABC):
    """
    This class will wrap original layer with `Adapters`, and then replace the original layer to this.

    This class must not exist without any adapter. It will be replaced to the base layer when no adapter in it.

    Subclass should define these methods:
        - **forward**: Calculate result base on `base_layer` and `adapters`.
        - **merge** :
        - **unmerge**:
        - **validate_adapter**:

    The default behavior will call one active adapter if exists any, or call base_layer.

    """

    def __init__(self, base_layer: nn.Module, *args, **kwargs):
        super().__init__()

        self.base_layer = base_layer
        self.active_adapters = nn.ModuleDict()
        self.non_active_adapters = nn.ModuleDict()

        self._merged_adapter_names: list[str] = []
        """`merge` and `unmerge` method will modify this list only."""

    # ---Abstract---
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        This method will never be called when there's no adapter.
        So that it should check `assert len(self.adapter_names) >0` at the very first beginning.
        """

    @abstractmethod
    def _validate_adapter(self, adapter: BaseAdapter, *args, **kwargs) -> BaseAdapter:
        """
        This method will validate if adapter is correct for this class. Or maybe change, rebuild an adapter.

        Args:
            adapter:
            args: Arguments passed by `API.add_adapter`.
            kwargs: Arguments passed by `API.add_adapter`.
        Returns:
            BaseAdapter object if the adapter is valid. If not, should raise an error.
        """

    # ---Abstract (Optional)---

    def _merge(
        self, adapter_names: str | list[str] | None = None, *args, **kwargs
    ) -> Any:
        """
        Implement the merging process. Should store merge adapter names in `self._merged_adapter_names`.
        Args:
            adapter_names:
            *args: Arguments passed by `API.add_adapter`.
            **kwargs:Arguments passed by `API.add_adapter`.
        """
        raise NotImplementedError

    def _unmerge(
        self, adapter_names: str | list[str] | None = None, *args, **kwargs
    ) -> Any:
        raise NotImplementedError

    def _validate_after_merge_or_unmerge(self, *args, **kwargs) -> None:
        """
        This method will be call after merge or unmerge, intentionally used for checking valid parameters.
        The default behavior is checking there are no Nan, Inf values in parameters of base_layer. Raise if there is one.
        Args:
            *args: Arguments passed by API.
            **kwargs: Argument passed by API.

        Returns:
            None
        """
        for name, param in self.base_layer.named_parameters():
            if not torch.isfinite(param).all():
                raise ValueError(
                    f"base_layer has non-finite value in parameter `{name}`"
                )

    # ---Public---

    def get_adapter(
        self, adapter_name: str, *, raise_if_not_found: bool = True
    ) -> BaseAdapter | None:
        """
        Get a BaseAdapter object with specified name.
        Args:
            adapter_name:
            raise_if_not_found:
        Returns:
            BaseAdapter object with name `adapter_name` or None if not found.

        """
        for adapters in [self.active_adapters, self.non_active_adapters]:
            try:
                return cast(BaseAdapter, adapters[adapter_name])
            except KeyError:
                pass
        if raise_if_not_found:
            raise ValueError(f"Adapter named {adapter_name} not found.")
        return None

    @property
    def adapter_names(self) -> list[str]:
        return list(self.active_adapters.keys()) + list(self.non_active_adapters.keys())

    @property
    def merged_adapter_names(self) -> list[str]:
        """
        Returns:
            A list of merged adapter name, not that this list is a deepcopy (for non-changeable).
             So that you must modify again to capture the update.
        """
        return deepcopy(self._merged_adapter_names)

    @property
    def is_merged(self) -> bool:
        return len(self._merged_adapter_names) > 0

    # ---Private---

    def _add_adapters(
        self, adapters: BaseAdapter | Sequence[BaseAdapter], *, activate: bool = False
    ):
        """Add adapters to non_activate_adapters by default. Raise error if adapter already exists."""
        adapters = [adapters] if isinstance(adapters, BaseAdapter) else list(adapters)
        for adapter in adapters:
            assert isinstance(adapter, BaseAdapter)
            assert adapter.name not in self.adapter_names
            adapter = self._validate_adapter(adapter)
            assert isinstance(adapter, BaseAdapter)

            if activate:
                self.active_adapters[adapter.name] = adapter
            else:
                self.non_active_adapters[adapter.name] = adapter

    def _remove_adapters(self, adapter_names: str | Sequence[str]) -> list[str]:
        """

        Args:
            adapter_names:

        Returns:
            A list of adapter names that have been found and  removed in this layer.
        """
        adapter_names = (
            [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
        )

        removed_adapters: list[str] = []
        for name in adapter_names:
            rm_adt = None
            for adapters in [self.active_adapters, self.non_active_adapters]:
                try:
                    rm_adt = adapters.pop(
                        name
                    )  # Will raise key error if not found adapter.
                    assert isinstance(
                        rm_adt, BaseAdapter
                    )  # Check if dict contains only BaseAdapter.
                    assert (
                        name not in removed_adapters
                    )  # Check if name exists in both dicts.
                    removed_adapters.append(name)
                    if name in self._merged_adapter_names:
                        self._merged_adapter_names.remove(name)
                except KeyError:
                    pass
        return removed_adapters

    @dataclass
    class _ActivateFlags:
        change: list[str] = field(default_factory=list)
        not_change: list[str] = field(default_factory=list)
        not_found: list[str] = field(default_factory=list)

        @property
        def total(self) -> int:
            return len(self.change) + len(self.not_change) + len(self.not_found)

    def _activate_adapters(
        self, adapter_names: str | Sequence[str], *, activate: bool = True
    ) -> _ActivateFlags:
        """

        Args:
            adapter_names:
            activate: Flag indicate to activate or deactivate

        Returns:
            A _ActivateFlags object indicate process flags, including keys `change`, `not_change` and `not_found`.

        """
        # Useful flags.
        activate_adapters = BaseAdaptedLayer._ActivateFlags()

        if activate:
            from_ = self.non_active_adapters
            to_ = self.active_adapters
        else:
            from_ = self.active_adapters
            to_ = self.non_active_adapters

        adapter_names = (
            [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
        )
        for name in adapter_names:
            if name not in to_.keys():
                if name not in from_.keys():
                    activate_adapters.not_found.append(name)
                    continue
                adapter = from_.pop(name)
                assert isinstance(adapter, BaseAdapter) and adapter.name == name
                to_[name] = adapter
                activate_adapters.change.append(name)
            else:
                activate_adapters.not_change.append(name)
            assert name in to_.keys()

        return activate_adapters


class BaseAdaptedModelConfig(BaseModel, ABC):
    """
    Config that bind with one model adapter.

    Config has two roles:
    1. Contain all arguments for adapter layer to use.
    2. Contain dispatch logic for construct LayerAdapter and replace it the base_layer .
    """

    model_config = ConfigDict(validate_assignment=True)
    adapter_name: str = Field("default")

    @abstractmethod
    def dispatch_adapter(
        self, fpname: str, base_layer: nn.Module, *args, **kwargs
    ) -> BaseAdapter | None:
        """
        This method will construct and dispatch an adapter for each base layer.
        Args:
            fpname: Fully qualified name of the module.
            base_layer: The original torch.nn.Module instance of the model.
            args: Addition arguments passed by `add_adapter` method.
            kwargs: Addition keyword arguments passed by `add_adapter` method.
        Returns:
            BaseAdapter object, or None when does not add adapter to layer.

        """
        pass

    def dispatch_adapted_layer(
        self, fpname: str, base_layer: nn.Module, *args, **kwargs
    ) -> BaseAdaptedLayer:
        cast(BaseAdaptedModelConfig, self)  # For disable warning
        return BaseAdaptedLayer(base_layer)


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
                base_layer.weight.data += adapter.get_delta_weight()

                if (delta_bias := adapter.get_delta_bias()) is not None:
                    base_layer.bias.data += delta_bias
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
            base_layer.weight.data -= adapter.get_delta_weight()

            if (delta_bias := adapter.get_delta_bias()) is not None:
                base_layer.bias.data -= delta_bias
            self._merged_adapter_names.remove(name)


from collections.abc import Callable
from typing import cast, Unpack

from pydantic import PositiveInt, NonNegativeFloat, BaseModel
from torch import nn


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
        **kwargs: Unpack[ValidateConfigKwargs],
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
        return LoraLinearAdaptedLayer(base_layer, *args, **kwargs)


__ADT_NAMES_ATTR__ = "__PETORCH_ADAPTER_NAMES_ATTRIBUTE_NAME__"


class AdapterAPI:
    class __AdapterNameList__:
        """
        This class will be injected to model to record adapter names.
        """

        def __init__(self):
            self._private_list = []

        def append(self, v):
            assert v not in self._private_list
            self._private_list.append(v)

        def remove(self, v):
            self._private_list.remove(v)

    @staticmethod
    @torch.no_grad()
    def get_adapter_names(model: nn.Module) -> list[str] | None:
        """
        Get current Petorch adapter names that added to model.
        Args:
            model:

        Returns:
            list of adapter names with len always >0 or None if no adapter is added (Model is now the raw, original model.).

        """
        adt_names = getattr(model, __ADT_NAMES_ATTR__, None)
        if adt_names is not None:
            assert isinstance(adt_names, AdapterAPI.__AdapterNameList__)
            assert (
                len(adt_names._private_list) > 0
            )  # Object with len=0 must be deleted and getattr return None.
            return deepcopy(adt_names._private_list)
        return None

    @staticmethod
    @torch.no_grad()
    def add_adapter(
        model: nn.Module, config: BaseAdaptedModelConfig, *args, **kwargs
    ) -> list[str]:
        """
        Add adapter to Pytorch model.
        Args:
            model:
            config:
            args: Will be passed to **config.dispatch_layer_adapter** and **config.dispatch_adapted_layer**.
            kwargs: Will be passed to **config.dispatch_layer_adapter** and **config.dispatch_adapted_layer**.
        Returns:
            list of Fully qualified names that adapter is added.
            If blank, adapter haven't been added to model because of `dispatch_layer_adapter` always return None .

        Raises:
            ValueError: If adapter name already exists.

        Notes:
            The config added will take a deepcopy.


        """
        if (adt_names := AdapterAPI.get_adapter_names(model)) is not None:
            if config.adapter_name in adt_names:
                raise ValueError(
                    f"Adapter named `{config.adapter_name}` already added to the model."
                )

        def is_wrapped_by_adapted_layer(model, fqname: str) -> bool:
            splits = fqname.split(".")
            for i, _ in enumerate(splits):
                fqn = ".".join(splits[:i])
                if isinstance(model.get_submodule(fqn), BaseAdaptedLayer):
                    return True
            return False

        # Collect
        layer_collections = []
        adapted_fqnames: list[str] = []  # Return module path for debug
        for fqname, layer in model.named_modules():
            # Only collect BaseAdaptedLayer, or layer satisfies conditions:
            # 1. Not be wrapped by another BaseAdaptedLayer.
            # 2. config.dispatch_adapter return BaseAdapter.
            # 3. Returned adapter pass adapted_layer._validate_adapter.

            # Also ensure that no nested adapted layer happened.

            is_adapted_layer = isinstance(layer, BaseAdaptedLayer)

            # Skip all non-BaseAdaptedLayer layers that wrapped by BaseAdaptedLayer.
            if not is_adapted_layer:
                if is_wrapped_by_adapted_layer(model, fqname):
                    continue
            else:
                # If the layer is BaseAdaptedLayer, The parents must not BaseAdaptedLayer.
                assert not is_wrapped_by_adapted_layer(
                    model, fqname
                ), f"Nested Adapted layer detected. fqname={fqname}"

            base_layer = layer.base_layer if is_adapted_layer else layer

            if (
                adapter := config.dispatch_adapter(fqname, base_layer, *args, **kwargs)
            ) is not None:
                assert isinstance(adapter, BaseAdapter)
                adapted_layer = (
                    layer
                    if is_adapted_layer
                    else config.dispatch_adapted_layer(
                        fqname, base_layer, *args, **kwargs
                    )
                )
                assert isinstance(adapted_layer, BaseAdaptedLayer)

                # Check if the new adapter is valid for the layer
                adapter = adapted_layer._validate_adapter(adapter, *args, **kwargs)
                assert isinstance(
                    adapter, BaseAdapter
                ), f"The adapter `{adapter}` is not the valid one for `{adapted_layer}`."

                layer_collections.append(
                    (fqname, adapted_layer, adapter, is_adapted_layer)
                )
                adapted_fqnames.append(fqname)

        # Modify
        for fqname, adapted_layer, adapter, is_adapted_layer in layer_collections:
            adapted_layer._add_adapters(adapter, activate=False)

            # If a new AdaptedLayer created, swap the base layer with it.
            if not is_adapted_layer:
                model.set_submodule(fqname, adapted_layer)

        # Update model.__ADT_NAMES_ATTR__, create new if not exists.
        if len(adapted_fqnames) > 0:
            if not hasattr(model, __ADT_NAMES_ATTR__):
                adt_name_list = AdapterAPI.__AdapterNameList__()
                adt_name_list.append(config.adapter_name)
                setattr(model, __ADT_NAMES_ATTR__, adt_name_list)
            else:
                getattr(model, __ADT_NAMES_ATTR__).append(config.adapter_name)

        model.train(model.training)
        return adapted_fqnames

    @staticmethod
    @torch.no_grad()
    def update_adapter(
        model: nn.Module,
        config: BaseModel | dict,
        **validate_kwargs: Unpack[ValidateConfigKwargs],
    ) -> None:
        """
        Update Adapter to a new config. So that model will be used new attributes new config.

        Notes:
            This method does not update adapter schema, it just only add config to current adapter layer.
            The config added will take a deepcopy.

        Args:
            model:
            config:

        Raises:
            ValueError: If Adapter does not found in model.

        """
        updated_layers: list[str] = []
        if (adt_names := AdapterAPI.get_adapter_names(model)) is None:
            raise ValueError(f"Model does not have any adapter.")
        if config.adapter_name not in adt_names:
            raise ValueError(
                f"Model does not have adapter named `{config.adapter_name}`."
            )

        # Loop for all BaseAdaptedLayers
        for fqname, layer in model.named_modules():
            if not isinstance(layer, BaseAdaptedLayer):
                continue

            if (adapter := layer.get_adapter(config.adapter_name)) is not None:
                assert isinstance(adapter, BaseAdapter)
                # Change underlying config
                adapter._update_config(config, **validate_kwargs)
                updated_layers.append(fqname)

        # There must be at least one layer be updated.
        assert len(updated_layers) > 0
        model.train(model.training)

    @staticmethod
    @torch.no_grad()
    def activate_adapter(
        model: nn.Module,
        adapter_names: str | Sequence[str] | None = None,
        *,
        activate: bool = True,
    ) -> list[str]:
        """
        Activate adapters in model. Already activated adapters will stay the same.
        Args:
            model:
            adapter_names: None for all adapter in model.
            activate:

        Returns:
            A list of adapter name that just be activated or deactivated (change state).

        Raises:
            ValueError: If there's any adapter that does not exist.

        """
        change_adapters: set[str] = set()
        not_change_adapters: set[str] = set()
        not_found_adapters: set[str] = set()

        adapter_names = AdapterAPI._resolve_adapter_names(model, adapter_names)

        # Collect fqnames of BaseAdaptedLayers
        activate_fqnames: list[str] = []
        for fqname, layer in model.named_modules():
            if isinstance(layer, BaseAdaptedLayer):
                activate_fqnames.append(fqname)

        # Modify
        for fqname in activate_fqnames:
            layer = model.get_submodule(fqname)
            assert isinstance(layer, BaseAdaptedLayer)
            flags = layer._activate_adapters(adapter_names, activate=activate)
            assert flags.total == len(adapter_names)

            # Add to sets
            for change_adt in flags.change:
                change_adapters.add(change_adt)
            for not_change_adt in flags.not_change:
                not_change_adapters.add(not_change_adt)
            for not_found_adt in flags.not_found:
                not_found_adapters.add(not_found_adt)

        assert (l1 := len(not_change_adapters)) + (l2 := len(not_found_adapters)) + (
            l3 := len(change_adapters)
        ) == (l4 := len(adapter_names)), f"{l1}+{l2}+{l3}!={l4}"

        # All adapters must be found.
        for nf in not_found_adapters:
            if nf in change_adapters or nf in not_change_adapters:
                not_found_adapters.remove(nf)
        assert (
            len(not_found_adapters) == 0
        ), f"c:{change_adapters}, nc:{not_change_adapters}, nf:{not_found_adapters} "

        # Not allow adapter exists in two modes at the same time.
        for c_adt in change_adapters:
            assert c_adt not in not_change_adapters

        model.train(model.training)
        return list(change_adapters)

    @staticmethod
    @torch.no_grad()
    def remove_adapter(
        model: nn.Module, adapter_names: str | Sequence[str] | None = None
    ) -> None:
        """
        Remove adapters of this config from adapted layer, if after remove there's no adapters remain, it will switch
        adapted layer to base layer.
        Args:
            model:
            adapter_names:

        Raises:
            ValueError: If there's a not found adapter.

        """

        adapter_names = AdapterAPI._resolve_adapter_names(model, adapter_names)

        not_found_adapter_names = (
            {adapter_names} if isinstance(adapter_names, str) else set(adapter_names)
        )

        # Collect fqnames of BaseAdaptedLayers
        remove_fqnames: list[str] = []
        for fqname, layer in model.named_modules():
            if isinstance(layer, BaseAdaptedLayer):
                remove_fqnames.append(fqname)

        # Remove
        for fqname in remove_fqnames:
            layer = model.get_submodule(fqname)
            assert isinstance(layer, BaseAdaptedLayer)
            rm_adt = layer._remove_adapters(adapter_names)
            for removed_name in rm_adt:
                if removed_name in not_found_adapter_names:
                    not_found_adapter_names.remove(removed_name)

            if len(rm_adt) > 0:
                # If there's no adapter in module after removing, switch it to base layer.
                if len(layer.adapter_names) == 0:
                    model.set_submodule(fqname, layer.base_layer)
            else:
                # If rm_adt ==[], Then adapted layer must have adapters.
                # Because when after removing and no adapters remain, it already switched to baselayer, no more adapted layer.
                assert len(layer.adapter_names) > 0

        # All adapters must be found in model (there's at least one adapted layer has specific adapter).
        assert len(not_found_adapter_names) == 0

        _adapter_name_list = cast(
            AdapterAPI.__AdapterNameList__, getattr(model, __ADT_NAMES_ATTR__)
        )
        for adt_name in adapter_names:
            _adapter_name_list.remove(adt_name)
        if len(_adapter_name_list._private_list) == 0:
            delattr(model, __ADT_NAMES_ATTR__)

        model.train(model.training)

    @staticmethod
    @torch.no_grad()
    def merge(
        model: nn.Module, adapter_names: str | list[str] | None = None, *args, **kwargs
    ):
        return AdapterAPI._merge_or_unmerge_(
            model, adapter_names, *args, merge=True, **kwargs
        )

    @staticmethod
    @torch.no_grad()
    def unmerge(
        model: nn.Module, adapter_names: str | list[str] | None = None, *args, **kwargs
    ):
        return AdapterAPI._merge_or_unmerge_(
            model, adapter_names, *args, merge=False, **kwargs
        )

    @staticmethod
    def _merge_or_unmerge_(
        model: nn.Module,
        adapter_names: str | list[str] | None = None,
        *args,
        merge: bool,
        **kwargs,
    ) -> Any:
        """Not call this method directly, it does not ensure for no_grad."""
        adapter_names = AdapterAPI._resolve_adapter_names(model, adapter_names)
        merge_fqnames: list[str] = []

        # Collect first for avoiding the case that _merge implementation change the architecture.
        for fqname, layer in model.named_modules():
            if isinstance(layer, BaseAdaptedLayer):
                merge_fqnames.append(fqname)
        assert len(merge_fqnames) > 0

        for fqname in merge_fqnames:
            layer = model.get_submodule(fqname)
            assert isinstance(layer, BaseAdaptedLayer)
            if merge:
                layer._merge(adapter_names, *args, **kwargs)
            else:
                layer._unmerge(adapter_names, *args, **kwargs)

            layer._validate_after_merge_or_unmerge(*args, **kwargs)
        model.train(model.training)

    @staticmethod
    def _resolve_adapter_names(
        model: nn.Module, adapter_names: str | list[str] | None = None
    ) -> list[str]:
        if (adt_names := AdapterAPI.get_adapter_names(model)) is None:
            raise ValueError(f"Model does not have any adapter.")

        if adapter_names is not None:
            adapter_names = (
                [adapter_names]
                if isinstance(adapter_names, str)
                else list(adapter_names)
            )
            for adt_name in adapter_names:
                if adt_name not in adt_names:
                    raise ValueError(f"Model does not have adapter named `{adt_name}`.")
        else:
            adapter_names = adt_names
        assert len(adapter_names) > 0  # Not allow blank list
        return adapter_names


def _randint_list(n_list: int, l_len: int, a: int, b: int):
    for _ in range(n_list):
        l = []
        for _ in range(l_len):
            l.append(randint(a, b))
        yield l


def _make_adapter_configs(num: int) -> list[LoraLinearModelConfig]:
    return [
        LoraLinearModelConfig(
            adapter_name=f"adapter_{i}",
            rank=4 * rank,
            alpha=8 * rank,
            bias=bias < 5,
            scale=scale,
        )
        for i, [rank, bias, scale] in enumerate(_randint_list(num, 3, 2, 10))
    ]


if __name__ == "__main__":
    model = NestedDummy()
    sample_size = (3, 8, 8)

    configs = _make_adapter_configs(3)

    sample = torch.randn((2,) + sample_size)
    org_model = deepcopy(model)
    model.eval()
    org_model.eval()

    base_output: torch.Tensor = org_model(sample)
    assert torch.isfinite(base_output).all()

    adapter_names = [config.adapter_name for config in configs]
    adapted_fqn :list[str] = []
    for config in configs:
        adapted_fqn = AdapterAPI.add_adapter(
            model, config, lora_init_method=TorchInitMethod.normal
        )

    # IMPORTANT, PRONE OF ERROR
    assert model.training == False
    model.eval()

    def _assert_correct_adapted_layer(*, is_activated: bool, is_merged: bool = False):
        # Check for adapted layers is correct.
        current_adapted_fqn = 0
        for name, module in model.named_modules():
            # Check for all current adapted layers
            if isinstance(module, LoraLinearAdaptedLayer):
                assert name in adapted_fqn
                if is_activated:
                    assert (
                        len(module.active_adapters)
                        == len(adapter_names)
                        == len(module.adapter_names)
                    )
                    assert len(module.non_active_adapters) == 0
                else:
                    assert (
                        len(module.non_active_adapters)
                        == len(adapter_names)
                        == len(module.adapter_names)
                    )
                    assert len(module.active_adapters) == 0
                if is_merged:
                    assert module.is_merged
                    assert (
                        len(module.merged_adapter_names)
                        == len(adapter_names)
                        == len(module.adapter_names)
                    )

                current_adapted_fqn += 1
            else:
                assert name not in adapted_fqn

        assert current_adapted_fqn == len(adapted_fqn)

    # After add adapters
    _assert_correct_adapted_layer(is_activated=False)

    assert isinstance(
        adapted_layer := model.get_submodule(
            adapted_fqn[randint(0, len(adapted_fqn) - 1)]
        ),
        LoraLinearAdaptedLayer,
    )  # Get any adapted layer. Note that this layer is already tested in `test_api`.Just get for monitoring.

    # None activated adapter cases
    assert (
        len(AdapterAPI.get_adapter_names(model))
        == len(adapter_names)
        == len(c := adapted_layer.non_active_adapters)
        == len(adapted_layer.adapter_names)
        and len(adapted_layer.active_adapters) == 0
    ), f"{c}"
    assert torch.allclose(model(sample), base_output)

    # Activate
    activated_names = AdapterAPI.activate_adapter(model, adapter_names, activate=True)
    model.eval()
    _assert_correct_adapted_layer(is_activated=True)
    assert not torch.allclose(model(sample), base_output)
    
    model.eval()
    activated_output = model(sample)
    assert torch.isfinite(base_output).all()

    # Merge
    assert not adapted_layer.is_merged
    AdapterAPI.merge(model, adapter_names)
    model.eval()
    _assert_correct_adapted_layer(is_activated=True, is_merged=True)

    # Now the base model output is equal to activated_output
    # print(model)
    
    # Here is the fail, and max_abs varies from 2.5-99999 (small to very large, but not below than zeros)
    model.eval()
    assert torch.allclose(
        o := model(sample), activated_output
    ), f"max_abs = {(activated_output-o).abs().max()}"
