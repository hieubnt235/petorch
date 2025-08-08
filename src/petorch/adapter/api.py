from copy import deepcopy
from typing import Sequence, cast, Unpack, Any

import torch
from pydantic import BaseModel
from torch import nn

from .base import (
    BaseAdapter,
    BaseAdaptedModelConfig,
    BaseAdaptedLayer,
    ValidateConfigKwargs,
)

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

            layer._validate_after_merge_or_unmerge(*args,**kwargs)
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
