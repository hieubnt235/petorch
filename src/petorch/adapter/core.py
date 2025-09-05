from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Sequence,
    cast,
    Any,
    TypedDict,
    Type,
    Unpack,
    TypeVar,
    Generic,
    Literal, ClassVar,
)

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

# noinspection PyProtectedMember
from torch.nn.modules.module import _IncompatibleKeys

from petorch.utilities import fake_use


class ValidateConfigKwargs(TypedDict):
    strict: bool | None
    from_attributes: bool | None
    context: Any | None
    by_alias: bool | None
    by_name: bool | None


class AdapterConfig(BaseModel):
    adapter_name: str


BaseModule_T = TypeVar("BaseModule_T", bound=nn.Module)
Config_T = TypeVar("Config_T", bound=AdapterConfig)


class BaseAdapter(nn.Module, Generic[BaseModule_T, Config_T], ABC):
    """
    The abstract class for one layer adapter.
    For example, it contains loraA and loraB matrices.

    The only requirement of the subclasses of this method is the ` config_class ` class attribute.


    Features:
        - Adapter declares dynamic adjustable hyperparameters, such as scale, for higher-level set it (use set it).
        - Adapter knows how to use these hyperparameters in the forward method.

    """

    base_layer_class: Type[BaseModule_T]
    config_class: Type[Config_T]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        assert issubclass(
            cfg_cls := getattr(cls, "config_class"), AdapterConfig
        ), f"`config_class` must be the subclass of `AdapterConfig`. Got `{cfg_cls}`."

        if not issubclass(cfg_cls.model_fields["adapter_name"].annotation, str):
            raise ValueError(
                f"Type hint of `adapter_name` in `AdapterConfig` must be `str`."
            )

        blc = getattr(cls, "base_layer_class", None)
        assert issubclass(
            cast(type, blc), nn.Module
        ), f"Subclass must declare `base_layer_class` as a subclass of {nn.Module} but not be it. Got {blc}"

    def __init__(
        self,
        base_layer: BaseModule_T,
        config: BaseModel | dict[str, Any],
        **kwargs: Unpack[ValidateConfigKwargs],
    ):
        super().__init__()
        assert isinstance(
            base_layer, self.base_layer_class
        ), f"Base layer must has type {self.base_layer_class}, got {type(base_layer)}."

        self._base_layer = [base_layer]  # Wrap to a list for not exposed as a module
        self._config = self.config_class.model_validate(
            config, **self._get_validate_kwargs(config, **kwargs)
        )
        self._adapter_name: str = self._config.adapter_name

    @abstractmethod
    def get_delta_weight(self) -> torch.Tensor:
        """
        Get adapter delta weight, Note that this is weight only, the merging process need also bias.
        Make sure you scale all the values in this method if needed, because the usage will scale anymore.
        Returns:
            Tensor with shape (base_layer.out_features, base_layer.in_features)
        """
        ...

    # noinspection PyTypeHints
    @abstractmethod
    def get_delta_bias(self) -> torch.Tensor | None: ...

    @abstractmethod
    def get_delta(self, batch_input: torch.Tensor) -> torch.Tensor: ...

    @property
    def config(self) -> Config_T:
        return self._config

    @property
    def base_layer(self) -> BaseModule_T:
        return self._base_layer[0]

    @property
    def name(self) -> str:
        return self._adapter_name

    # noinspection PyMethodMayBeStatic
    def _get_validate_kwargs(
        self, config: dict[str, Any] | BaseModel, **kwargs: Any
    ) -> dict[str, Any]:
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
        new_config: BaseModel | dict[str, Any],
        **validate_kwargs: Unpack[ValidateConfigKwargs],
    ) -> None:
        new_config = self.config_class.model_validate(
            new_config, **self._get_validate_kwargs(new_config, **validate_kwargs)
        )
        assert (
            new_config.adapter_name == self.name
        ), f"New config's `adapter_name` does not match with the current one. `{new_config.adapter_name}` and `{self.name}`."
        self._config = new_config


class BaseAdaptedLayer(nn.Module, ABC, Generic[BaseModule_T]):
    """
    This class will wrap the original layer with `Adapters`, and then replace the original layer to this.

    This class must not exist without any adapter. It will be replaced to the base layer when no adapter in it.

    Subclass should define these methods:
        - **forward**: Calculate result base on `base_layer` and `adapters`.
        - **merge**:
        - **unmerge**:
        - **validate_adapter**:

    The default behavior will call one active adapter if exists any, or call base_layer.

    """

    act_adt_key: str = "active_adapters"
    non_act_adt_key: str = "non_active_adapters"
    
    @abstractmethod
    def _validate_base_layer(self, base_layer: nn.Module):
        """
        Implement the logic for validating the base layer. Such as ensuring base_layer must have the attribute `weight`.
        Args:
            base_layer:
        """
        ...

    @abstractmethod
    def _update_base_layer(
        self,
        delta_weight: torch.Tensor,
        delta_bias: None | torch.Tensor,
    ) -> None:
        """
        Implement how `self.base_layer` is updated with adapter weights and biases.
        This method is called after calculated merged or unmerged adapter weights.
        It can be updated inline or swap self.base_layer to another.

        Args:
            delta_weight: Sum of the weights for all adapters that intent to be merged.
            delta_bias: Sum of the biases for all adapters that intent to be merged or unmerge.

        """
        ...
    
    def __init__(self, base_layer: BaseModule_T) -> None:
        super().__init__()
        self._validate_base_layer(base_layer)
        self.base_layer = base_layer
        self.active_adapters = nn.ModuleDict()
        self.non_active_adapters = nn.ModuleDict()

        assert getattr(self, self.act_adt_key) == self.active_adapters
        assert getattr(self, self.non_act_adt_key) == self.non_active_adapters

        self._merged_adapter_names: list[str] = []
        """`merge` and `unmerge` method will modify this list only."""

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

    # ---Public---

    def forward(
        self, batch_input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        If merged, return `base_layer.forward` only else all `active_adapters` will be added to the final results.
        Args:
            batch_input:
            *args:
            **kwargs:

        Returns:
            Result typed torch.Tensor
        """
        fake_use(args, kwargs)
        assert len(self.adapter_names) > 0
        base_output = cast(torch.Tensor, self.base_layer(batch_input))

        # If merged, return
        if self.is_merged:
            return base_output
        else:
            for adapter in self.active_adapters.values():
                adapter = self._validate_adapter(
                    cast(BaseAdapter[BaseModule_T, AdapterConfig], adapter)
                )
                base_output += adapter.get_delta(batch_input)

            return base_output

    def get_adapter(
        self, adapter_name: str, *, raise_if_not_found: bool = True
    ) -> BaseAdapter[BaseModule_T, AdapterConfig] | None:
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
                return cast(
                    BaseAdapter[BaseModule_T, AdapterConfig], adapters[adapter_name]
                )
            except KeyError:
                pass
        if raise_if_not_found:
            raise ValueError(f"Adapter named {adapter_name} not found.")
        return None

    def __getattr__(self, item: str) -> Any:
        try:
            return super().__getattr__(item)
        except AttributeError:
            base_layer = super().__getattr__("base_layer")
            return getattr(base_layer, item)

    # ---Private---

    def _add_adapters(
        self,
        adapters: (
            BaseAdapter[BaseModule_T, Config_T]
            | Sequence[BaseAdapter[BaseModule_T, Config_T]]
        ),
        *,
        activate: bool = False,
    ) -> None:
        """Add adapters to non_activate_adapters by default. Raise error if adapter already exists."""
        adapters = [adapters] if isinstance(adapters, BaseAdapter) else list(adapters)
        for adapter in adapters:
            assert isinstance(adapter, BaseAdapter)
            assert (
                adapter.name not in self.adapter_names
            ), f"Adapter {adapter.name} already exists."
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
            for adapters in [self.active_adapters, self.non_active_adapters]:
                try:
                    rm_adt = adapters.pop(
                        name
                    )  # Will raise a key error if not found adapter.
                    assert isinstance(
                        rm_adt, BaseAdapter
                    )  # Check if the dict contains only BaseAdapter.
                    assert (
                        name not in removed_adapters
                    )  # Check if the name exists in both dicts.
                    removed_adapters.append(name)
                    if name in self._merged_adapter_names:
                        self._merged_adapter_names.remove(name)
                except KeyError:
                    pass
        return removed_adapters

    def _get_adapter_state_dict(
        self,
        adapter_names: str | Sequence[str],
        *,
        active_only: bool = False,
        non_active_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Get layer state dict, remove base_layer and filter adapter.
        Args:
            adapter_names:
            active_only:
            non_active_only:

        Returns:
            A state dictionary whose adapter keys are started with "active_adapters" or "non_active_adapters"
             and addition keys if exists.

        """
        # todo: support load state dict with adapter name.in format {adapter_name}.lora_A

        if active_only and non_active_only:
            raise ValueError(
                "`active_only` and `non_active_only` cannot be set at the same time."
            )
        adapter_names = (
            [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
        )

        # Raw state_dict
        state_dict = self.state_dict()
        # logger.debug(f"Raw AdaptedLayer state_dict keys: {'\n'.join(list(state_dict.keys()))}")

        # Collect keys to delete
        del_keys = []
        k_del_start = None
        if active_only:
            k_del_start = self.non_act_adt_key
        if non_active_only:
            k_del_start = self.act_adt_key
        if k_del_start:
            del_keys.extend([k for k in state_dict.keys() if k.startswith(k_del_start)])

        for key in state_dict.keys():
            # Delete base_layer keys
            if key.startswith("base_layer"):
                del_keys.append(key)
                continue
            else:
                # Delete keys that not contain adapter name in it.
                is_contain_name = False
                for name in adapter_names:
                    if name in key:
                        is_contain_name = True
                        break
                if not is_contain_name:
                    del_keys.append(key)

        for key in del_keys:
            # logger.debug(f"del_key: `{key}`")
            del state_dict[key]

        return state_dict

    def _load_adapter_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict_activation: bool = False,
        strict_load: bool = False,
        assign: bool = False,
    ) -> _IncompatibleKeys:
        """

        Args:
            state_dict: Adapter keys must be start with "non_active_adapter." or "active_adapter.".
            strict_activation: Only load when the state dict activation state is matched.
            strict_load: pass to the ` strict ` argument of `Module.load_state_dict`.
        Returns:
            (From Module.load_state_dict): `NamedTuple` with ``missing_keys`` and ``unexpected_keys`` fields:

                * **missing_keys** is a list of str containing any keys that are expected
                    by this module but missing from the provided ``state_dict``.
                * **unexpected_keys** is a list of str containing the keys that are not
                    expected by this module but present in the provided ``state_dict``.
        """

        final_state_dict = {}
        for key, weight in state_dict.items():
            if (
                key.startswith(self.non_act_adt_key + ".")
                or key.startswith(self.act_adt_key + ".")
            ) and not strict_activation:
                splits = key.split(".")
                attr_key, adt_key = splits[0], splits[1]
                if adt_key in self.active_adapters.keys():
                    key = f"{self.act_adt_key}.{".".join(splits[1:])}"
                elif adt_key in self.non_active_adapters.keys():
                    key = f"{self.non_act_adt_key}.{".".join(splits[1:])}"

            final_state_dict[key] = weight

        # Get base layer state dict, for not raise error because of base_layer key not available.
        base_layer_state_dict = self.base_layer.state_dict(prefix="base_layer.")
        for k, v in base_layer_state_dict.items():
            # Check that adapter state dict overrides the base layer state dict; this is unexpected behavior.
            assert k not in final_state_dict
            final_state_dict[k] = v

        return cast(
            _IncompatibleKeys,
            self.load_state_dict(final_state_dict, strict=strict_load, assign=assign),
        )

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

    def _validate_adapter(
        self, adapter: BaseAdapter[BaseModule_T, Config_T], *args: Any, **kwargs: Any
    ) -> BaseAdapter[BaseModule_T, Config_T]:
        """
        This method will validate if the adapter is correct for this class. Or maybe change, rebuild an adapter.

        Args:
            adapter:
            args: Arguments passed by `API.add_adapter`.
            kwargs: Arguments passed by `API.add_adapter`.
        Returns:
            BaseAdapter object if the adapter is valid. If not, should raise an error.
        """
        fake_use(args, kwargs)
        if isinstance(adapter, BaseAdapter) and (adapter.base_layer == self.base_layer):
            return adapter
        else:
            raise ValueError(
                f"Adapter `{adapter}` is not the valid or supported one for AdaptedLayer `{self.__class__}`."
            )

    def _merge(
        self, adapter_names: str | list[str] | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """
        Implement the merging process. Should store merge adapter names in `self._merged_adapter_names`.
        Args:
            adapter_names:
            *args: Arguments passed by `API.add_adapter`.
            **kwargs:Arguments passed by `API.add_adapter`.
        """
        fake_use(args, kwargs)
        # Resolve adapter names
        adapter_names = adapter_names or self.adapter_names
        adapter_names = (
            [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
        )
        # Pre-check
        for name in adapter_names:
            if name in self._merged_adapter_names:
                raise ValueError(f"Adapter named `{name}` was already merged.")

        # Update weight
        self._update_base_layer_with_adapter_weights(
            adapter_names, merge_or_unmerge="merge"
        )

        #  Update records
        for name in adapter_names:
            self._merged_adapter_names.append(name)

    def _unmerge(
        self, adapter_names: str | list[str] | None = None, *args: Any, **kwargs: Any
    ) -> None:
        fake_use(args, kwargs)
        # Resolve adapter names
        adapter_names = adapter_names or self.adapter_names
        adapter_names = (
            [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
        )
        # Pre-check
        for name in adapter_names:
            if name not in self._merged_adapter_names:
                raise ValueError(f"Adapter named `{name}` haven't merged.")

        # Update weight
        self._update_base_layer_with_adapter_weights(
            adapter_names, merge_or_unmerge="unmerge"
        )

        #  Update records
        for name in adapter_names:
            self._merged_adapter_names.remove(name)

    def _update_base_layer_with_adapter_weights(
        self, adapter_names: list[str], *, merge_or_unmerge: Literal["merge", "unmerge"]
    ) -> None:

        adapter_weights: int | torch.Tensor = 0
        adapter_bias: None | torch.Tensor = None
        assert merge_or_unmerge in ["merge", "unmerge"]
        op = torch.add if merge_or_unmerge == "merge" else torch.sub

        for name in adapter_names:
            adapter = self.get_adapter(name)
            assert adapter is not None
            adapter = self._validate_adapter(adapter)

            adapter_weights = op(adapter_weights, adapter.get_delta_weight())
            if (delta_bias := adapter.get_delta_bias()) is not None:
                if adapter_bias is None:
                    adapter_bias = delta_bias
                else:
                    adapter_bias = op(adapter_bias, delta_bias)

        assert isinstance(adapter_weights, torch.Tensor)
        self._update_base_layer(adapter_weights, adapter_bias)

    def _validate_after_merge_or_unmerge(self, *args: Any, **kwargs: Any) -> None:
        """
        This method will be call after merge or unmerge, intentionally used for checking valid parameters.
        The default behavior is checking there are no Nan, Inf values in parameters of base_layer. Raise if there is one.
        Args:
            *args: Arguments passed by API.
            **kwargs: Argument passed by API.

        Returns:
            None
        """
        fake_use(args, kwargs)
        for name, param in self.base_layer.named_parameters():
            if not torch.isfinite(param).all():
                raise ValueError(
                    f"base_layer has non-finite value in parameter `{name}`"
                )

class AdaptedLayer(BaseAdaptedLayer[BaseModule_T]):
    
    def _validate_base_layer(self, base_layer: nn.Module):
        assert isinstance(base_layer,nn.Module)
        assert hasattr(base_layer,"weight")
        assert isinstance(base_layer.weight, torch.Tensor)
        
    def _update_base_layer(self, delta_weight: torch.Tensor, delta_bias: None | torch.Tensor) -> None:
        if isinstance(self.base_layer.)


class BaseModelAdaptionConfig(BaseModel, ABC):
    """
    Config that bind with the model to one adapter.

    Config has two roles:
    1. Contain all arguments for the adapter layer to use. So that this config must has all required attributes of all adapter layers it dispatches,
     which is defined in `BaseAdapter.config_class`

    2. Contain dispatch logic for construct LayerAdapter and replace it the base_layer .
    """

    model_config:ClassVar[ConfigDict] = ConfigDict(
        validate_assignment=True, validate_default=True, arbitrary_types_allowed=True
    )
    adapter_name: str = Field("default")

    @abstractmethod
    def dispatch_adapter(
        self, fpname: str, base_layer: nn.Module, *args: Any, **kwargs: Any
    ) -> BaseAdapter[BaseModule_T, Config_T] | None:
        """
        This method will construct and dispatch an adapter for each base layer.
        Args:
            fpname: Fully qualified name of the module.
            base_layer: The original torch.nn.Module instance of the model.
            args: Addition arguments passed by the ` add_adapter ` method.
            kwargs: Addition keyword arguments passed by the ` add_adapter ` method.
        Returns:
            BaseAdapter object, or None when does not add adapter to layer.

        """
        pass

    def dispatch_adapted_layer(
        self, fpname: str, base_layer: BaseModule_T, *args: Any, **kwargs: Any
    ) -> BaseAdaptedLayer[BaseModule_T]:
        """
        This method will dispatch a BaseAdaptedLayer instance for the layer that is being added adapter for the first time only.
        In other words, it will be called by API when:
            1. The base layer is the candidate to add an adapter (`dispatch_adapter` return `BaseAdapter` instance)
            2. The base layer is not yet AdaptedLayer instance.

        Args:
            fpname:
            base_layer:
            *args:
            **kwargs:

        Returns:

        """
        _ = cast(BaseModelAdaptionConfig, self)  # For disable warning
        return BaseAdaptedLayer(base_layer)
