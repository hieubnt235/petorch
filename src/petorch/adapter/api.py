from copy import deepcopy
from typing import Sequence, cast, Unpack, Any, Self

import torch
from pydantic import BaseModel
from torch import nn

from .core import (
    BaseAdapter,
    BaseModelAdaptionConfig,
    BaseAdaptedLayer,
    ValidateConfigKwargs,
)


def _resolve_adapter_names(
    model: nn.Module, adapter_names: str | list[str] | None = None
) -> list[str]:
    """
    Check if given adapter names are available in a model, else raise an error.
    If `adapter_names` is None, return all available adapter names in the module.
    Args:
        model:
        adapter_names:

    Returns:
        A list of all valid adapter names.

    """
    if (adt_names := AdapterAPI.get_adapter_names(model)) is None:
        raise ValueError(f"Model does not have any adapter.")

    if adapter_names is not None:
        adapter_names = (
            [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
        )
        for adt_name in adapter_names:
            if adt_name not in adt_names:
                raise ValueError(f"Model does not have adapter named `{adt_name}`.")
    else:
        adapter_names = adt_names
    assert len(adapter_names) > 0  # Not allow a blank list
    return adapter_names


def _merge_or_unmerge_(
    model: nn.Module,
    adapter_names: str | list[str] | None = None,
    *args,
    merge: bool,
    **kwargs,
) -> Any:
    """Not call this method directly, it does not ensure for no_grad."""
    adapter_names = _resolve_adapter_names(model, adapter_names)
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


class __AdapterMeta__:
    """
    This class will be injected to the model to record adapter meta.
    """

    class MetaModel(BaseModel):
        adapted_fqnames: list[str]
        config: BaseModelAdaptionConfig

    __ADT_META_ATTR__ = "__PETORCH_META_ATTRIBUTE_KEY__"

    def __init__(self):
        self._private_dict: dict[str, self.MetaModel] = {}
        """adapter_name:(list_of_fqname, config)"""

    @property
    def _adapted_fqns(self) -> list[str]:
        fqn = []
        for meta in self._private_dict.values():
            fqn.extend(meta.adapted_fqnames)
        return list(set(fqn))

    @property
    def _adaption_configs(self) -> dict[str, BaseModelAdaptionConfig]:
        configs = {}
        for name, meta in self._private_dict.items():
            configs[name] = meta.config
        return configs

    def _add_meta(self, m: MetaModel) -> Self:
        assert m.config.adapter_name not in self._private_dict.keys()

        assert isinstance(m, self.MetaModel) and len(m.adapted_fqnames) > 0
        self._private_dict[m.config.adapter_name] = m
        return self

    def _remove_meta(self, name: str) -> MetaModel:
        return self._private_dict.pop(name)

    def __getattr__(self, item):
        return getattr(self._private_dict, item)

    def __len__(self):
        return len(self._private_dict)

    @classmethod
    def _get_from_obj(cls, obj) -> Self | None:
        meta = getattr(obj, cls.__ADT_META_ATTR__, None)
        assert isinstance(meta, cls) or (meta is None)
        return meta

    @classmethod
    def _remove_from_obj(cls, obj, safe_check: bool = True) -> Self:
        if safe_check:
            assert isinstance(cls._get_from_obj(obj), cls)
            delattr(obj, cls.__ADT_META_ATTR__)
            assert cls._get_from_obj(obj) is None
        else:
            delattr(obj, cls.__ADT_META_ATTR__)

    @classmethod
    def _set_to_obj(cls, self, obj) -> Self:
        assert self._get_from_obj(obj) is None
        setattr(obj, self.__ADT_META_ATTR__, self)
        return self

    def _set_self_to_obj(self, obj) -> Self:
        return self._set_to_obj(self, obj)


class AdapterAPI:

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
        adt_meta = __AdapterMeta__._get_from_obj(model)
        if adt_meta is not None:
            assert (
                len(adt_meta) > 0
            )  # Object with len=0 must be deleted and getattr return None.
            return list(adt_meta.keys())

        return None

    @staticmethod
    @torch.no_grad()
    def get_adapted_fqnames(
        model: nn.Module, sort: bool = True, *args, **kwargs
    ) -> list[str] | None:
        """

        Args:
            model:
            sort:

        Returns:
            A list of adapted layer's fq names with len always >0 or None if the model does not have any adapter.

        """
        meta = __AdapterMeta__._get_from_obj(model)
        if meta is not None:
            assert isinstance(meta, __AdapterMeta__)
            assert len(m_fqn := meta._adapted_fqns) > 0
            if sort:
                m_fqn.sort(
                    key=kwargs.pop("key", None), reverse=kwargs.pop("reverse", False)
                )
            return m_fqn
        return None

    @staticmethod
    @torch.no_grad()
    def get_adaption_configs(
        model: nn.Module,
    ) -> dict[str, BaseModelAdaptionConfig] | None:
        """
        Args:
            model:

        Returns:
            A dict of all configs as values and keys are adapter_name with length always >0.
             Return None if no adapter is added (Model is now the raw, original model.).

        """
        meta = __AdapterMeta__._get_from_obj(model)
        if meta is not None:
            assert isinstance(meta, __AdapterMeta__)
            assert len(configs := meta._adaption_configs) > 0
            return configs
        return None

    @staticmethod
    @torch.no_grad()
    def add_adapter(
        model: nn.Module, config: BaseModelAdaptionConfig, *args, activate:bool=False, **kwargs
    ) -> list[str]:
        """
        Add adapter to Pytorch model.
        Args:
            activate: Is activate adapter. Default to False.
            model:
            config:
            args: Will be passed to **config.dispatch_layer_adapter** and **config.dispatch_adapted_layer**.
            kwargs: Will be passed to **config.dispatch_layer_adapter** and **config.dispatch_adapted_layer**.
        Returns:
            list of Fully qualified names that adapter is added.
            If blank, adapter haven't been added to model because of `dispatch_layer_adapter` always return None.

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
            adapted_layer._add_adapters(adapter, activate=activate)

            # If a new AdaptedLayer created, swap the base layer with it.
            if not is_adapted_layer:
                model.set_submodule(fqname, adapted_layer, strict=True)

        # Update model.__ADT_NAMES_ATTR__, create new if not exists.
        if len(adapted_fqnames) > 0:
            adt_meta = __AdapterMeta__._get_from_obj(model)
            if adt_meta is None:
                adt_meta = (
                    __AdapterMeta__()
                    ._add_meta(
                        __AdapterMeta__.MetaModel(
                            adapted_fqnames=adapted_fqnames, config=deepcopy(config)
                        )
                    )
                    ._set_self_to_obj(model)
                )
            else:
                adt_meta._add_meta(
                    __AdapterMeta__.MetaModel(
                        adapted_fqnames=adapted_fqnames, config=deepcopy(config)
                    )
                )

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
        This method is intentionally change attributes, properties for the model to work (with forward or merge),
        not change the model itself.

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

        adapter_names = _resolve_adapter_names(model, adapter_names)

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
    def deactivate_adapter(
        model: nn.Module,
        adapter_names: str | Sequence[str] | None = None,
    ) -> list[str]:
        return AdapterAPI.activate_adapter(model, adapter_names, activate=False)

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

        adapter_names = _resolve_adapter_names(model, adapter_names)

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
                # If there's no adapter in the module after removing, switch it to base layer.
                if len(layer.adapter_names) == 0:
                    model.set_submodule(fqname, layer.base_layer, strict=True)
            else:
                # If rm_adt ==[], Then adapted layer must have adapters.
                # Because when after removing and no adapters remain, it already switched to baselayer, no more adapted layer.
                assert len(layer.adapter_names) > 0

        # All adapters must be found in the model (there's at least one adapted layer has specific adapter).
        assert len(not_found_adapter_names) == 0

        adt_meta = __AdapterMeta__._get_from_obj(model)

        for adt_name in adapter_names:
            adt_meta._remove_meta(adt_name)
        if len(adt_meta) == 0:
            __AdapterMeta__._remove_from_obj(model)
        model.train(model.training)

    @staticmethod
    @torch.no_grad()
    def get_adapter_state_dict(
        model,
        adapter_names: str | list[str] | None = None,
        *args,
        from_meta: bool = True,
        validate_meta: bool = True,
        active_only: bool = False,
        non_active_only: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """

        Args:
            model:
            adapter_names:
            *args:

            # --- Below arguments usually be default almost cases. ---
            non_active_only:
            active_only:
            from_meta: Whether to extract adapter from meta, or loop and extract the entire model.
             Default is `True` and should be `True`.
            validate_meta: Whether to validate that state dict is sync with the meta. Only be used when `from_meta=False`.
             Default is `True` and should be `True`.
            **kwargs:

        Returns:
            A dictionary whose keys are fqname of base layer and values are a dict of adapter names and their state dicts.
        """
        adapter_names = _resolve_adapter_names(model, adapter_names)
        adapted_fqn = AdapterAPI.get_adapted_fqnames(model)
        assert isinstance(adapted_fqn, list)  # Model must have one adapter.

        state_dict: dict[str, torch.Tensor] = {}

        def _get_and_update_state_dict(adapted_layer: BaseAdaptedLayer, pre_fqn):
            layer_state_dict = adapted_layer._get_adapter_state_dict(
                adapter_names,
                active_only=active_only,
                non_active_only=non_active_only,
            )
            state_dict.update(
                {f"{pre_fqn}.{k}": v for k, v in layer_state_dict.items()}
            )

        if from_meta:
            for fqn in adapted_fqn:
                adapted_layer = model.get_submodule(fqn)
                assert isinstance(adapted_layer, BaseAdaptedLayer)
                _get_and_update_state_dict(adapted_layer, fqn)
        else:
            manual_fqn: list[str] = []
            for fqn, module in model.named_modules():
                if isinstance(module, BaseAdaptedLayer):
                    manual_fqn.append(fqn)
                    adapted_layer = cast(BaseAdaptedLayer, module)
                    _get_and_update_state_dict(adapted_layer, fqn)
            if validate_meta:
                manual_fqn.sort()
                adapted_fqn.sort()
                assert manual_fqn == adapted_fqn

        return state_dict

    @staticmethod
    @torch.no_grad()
    def load_adapter_state_dict(
        model,
        state_dict: dict[str, torch.Tensor],
        *args,
        from_meta: bool = True,
        validate_meta: bool = True,
        strict_activation: bool = False,
        strict_load: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, list[str]]]:
        """

        Args:
            model:
            state_dict:
            *args:

            # --- Below arguments usually be default almost cases. ---
            strict_activation: Only load when the state dict activation state is matched.
            strict_load: pass to the ` strict ` argument of `Module.load_state_dict`.
            validate_meta:
            from_meta:
            **kwargs:

        Returns:
            A dict whose keys are fqn to the base layer,
            values are a dict with `missing_keys` and `unexpected_keys` returned from `Module.load_state_dict`.
        """

        adapter_state_dict: dict[str, torch.Tensor] = {}
        adapted_fqn = AdapterAPI.get_adapted_fqnames(model)
        assert isinstance(adapted_fqn, list)  # Model must have one adapter.

        not_load_keys: dict[str, dict[str, list[str]]] = {}

        def _compose_and_load_adapter_state_dict(
            adapted_layer: BaseAdaptedLayer, fqn: str
        ):
            # Compose
            new_state_dict = {}
            for key, weight in state_dict.items():
                if key.startswith(fqn):
                    new_state_dict[key.removeprefix(fqn + ".")] = weight

            if new_state_dict != {}:
                missing_keys, unexpected_keys = adapted_layer._load_adapter_state_dict(
                    new_state_dict, strict_activation, strict_load
                )
                if missing_keys or unexpected_keys:
                    not_load_keys[fqn] = dict(
                        missing_keys=missing_keys, unexpected_keys=unexpected_keys
                    )

        if from_meta:
            for fqn in adapted_fqn:
                adapted_layer = model.get_submodule(fqn)
                assert isinstance(adapted_layer, BaseAdaptedLayer)
                _compose_and_load_adapter_state_dict(adapted_layer, fqn)

        else:
            manual_fqn: list[str] = []
            for fqn, module in model.named_modules():
                if isinstance(module, BaseAdaptedLayer):
                    manual_fqn.append(fqn)
                    adapted_layer = cast(BaseAdaptedLayer, module)
                    _compose_and_load_adapter_state_dict(adapted_layer, fqn)

            if validate_meta:
                manual_fqn.sort()
                adapted_fqn.sort()
                assert manual_fqn == adapted_fqn

        return not_load_keys

    @staticmethod
    @torch.no_grad()
    def merge(
        model: nn.Module, adapter_names: str | list[str] | None = None, *args, **kwargs
    ):
        return _merge_or_unmerge_(model, adapter_names, *args, merge=True, **kwargs)

    @staticmethod
    @torch.no_grad()
    def unmerge(
        model: nn.Module, adapter_names: str | list[str] | None = None, *args, **kwargs
    ):
        return _merge_or_unmerge_(model, adapter_names, *args, merge=False, **kwargs)
