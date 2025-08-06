from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Sequence, cast, Any, TypedDict, Type, Unpack

from pydantic import BaseModel, ConfigDict, Field
from torch import nn

"""
Problem from peft:

All redundant non-use adapters and layers still exists in repr.
We should drop that, or some how flag it to disable for easy debug

adapter know which module to inject (through config)
or


Adapter itself SHOULD NOT know about it's activate or scale,..., it the role of layer.

**Inject adapter API**:
Each layer type know what layer to add adapter (the first is win or raise if duplicate), or Dispatcher API ?
Adapter know how to use `scale`, and adapter layer set it (user set it).


1. We have BaseModule that we want to inject adapter.
2. After inject adapter,


Pytorch model should keepin pytorch model, we provide function that manipulate adapter

"""

"""

Aim:
1. Need the simple and flexibility.
2. Create framework (simple to dev) instead of library (easy to use), so that allow user to easy to build their own methods.
3. Native for every pytorch model, easy to integrate Pytorch base model of other libraries like transformers, diffusers, timm,...
4. Unify two concepts of quantization and peft to a single place.
5. Intensity in sanity checking.

Design:
1. Pytorch model should keepin pytorch model, we provide function that manipulate adapter,
So that user just use pytorch model be default.

2. One model adapter can be mixing of multiple adapter, for example some layers use LORA, some layers
use I3A etc?

Usages:
1. Build AbstractLayerAdapter subclasses
2. Build Config for dispatcher and all adapter parameters
3. Define AdaptedLayer.forward method for the usage of multi adapters.
4. Call AdapterManagerAPI.


"""


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

    def __init__(self, base_layer: nn.Module):
        super().__init__()

        self.base_layer = base_layer
        self.active_adapters = nn.ModuleDict()
        self.non_active_adapters = nn.ModuleDict()

        self._merged_adapter_names: list[str] = []
        """`merge` and `unmerge` method will modify this list only."""

    # ---Abstract---
    @abstractmethod
    def forward(self, *args, **kwargs)->Any:
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
    
    def _merge(self, adapter_names: str | list[str] | None = None, *args, **kwargs) -> Any:
        """
        Implement the merging process. Should store merge adapter names in `self._merged_adapter_names`.
        Args:
            adapter_names:
            *args: Arguments passed by `API.add_adapter`.
            **kwargs:Arguments passed by `API.add_adapter`.
        """
        raise NotImplementedError

    def _unmerge(self, adapter_names: str | list[str] | None = None, *args, **kwargs) -> Any:
        raise NotImplementedError


    # ---Public---

    def get_adapter(
        self, adapter_name:str, *, raise_if_not_found: bool = True
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
    def is_merged(self)->bool:
        return len(self._merged_adapter_names)>0
    
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
        This method will construct and dispatch adapter for each layer.
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
