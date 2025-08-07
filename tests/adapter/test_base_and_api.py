from copy import deepcopy
from typing import cast

import pytest
import torch
from torch import Tensor

from petorch.adapter import BaseAdapter, AdapterAPI, BaseAdaptedLayer
from petorch.dummy import NestedDummy, Dummy
from petorch.prebuilt.configs import LoraLinearModelConfig
from petorch.prebuilt.lora import LoraLinearAdapter


@pytest.mark.parametrize("model_cls", [NestedDummy, Dummy])
def test_api(model_cls):
    adapter_name = "test_adapter"
    model = model_cls()
    org_model = deepcopy(model)
    config = LoraLinearModelConfig(adapter_name=adapter_name)

    sample = torch.rand([2, 3, 8, 8])

    output1 = cast(Tensor, org_model(sample))

    # --add_adapter--
    target_fqn = []
    for name, module in model.named_modules():
        if (adapter := config.dispatch_adapter(name, module)) is not None:
            assert isinstance(adapter, BaseAdapter)
            target_fqn.append(name)
    target_fqn.sort()

    fqn = AdapterAPI.add_adapter(model, config)
    fqn.sort()
    assert fqn == target_fqn
    assert AdapterAPI.get_adapter_names(model)[0] == adapter_name

    with pytest.raises(ValueError, match="already added"):
        fqn = AdapterAPI.add_adapter(model, config)

    assert isinstance(
        adapted_layer := model.get_submodule(target_fqn[0]), BaseAdaptedLayer
    )
    assert adapted_layer.adapter_names[0] == adapter_name
    assert adapter_name in adapted_layer.non_active_adapters
    assert torch.all(output1 == model(sample))  # Test not activate adapter

    # --activate_adapter--
    with pytest.raises(ValueError, match="does not have adapter"):
        AdapterAPI.activate_adapter(model, "random_other_adapter_name")

    # Activate fail, nothing change, just like not activating
    assert adapter_name in adapted_layer.non_active_adapters
    assert torch.all(output1 == model(sample))  # Test not activate adapter

    # After activate successfully, the model output will be changed
    activated_adapter_names = AdapterAPI.activate_adapter(model, adapter_name)
    assert activated_adapter_names[0] == adapter_name
    assert adapter_name in adapted_layer.active_adapters
    assert not torch.all(output1 == model(sample))

    # --update_adapter--
    # noinspection PyTypeChecker
    assert isinstance(
        (activated_adapter := adapted_layer.active_adapters[adapter_name]),
        LoraLinearAdapter,
    )

    # Test that base_layer is not the module of adapter
    for name, module in activated_adapter.named_modules():
        assert not cast(str, name).endswith("base_layer")
        assert module != activated_adapter.base_layer

    def cal_scaling(_adapter, _config):
        return _adapter.scale * _config.alpha / _config.rank

    # Test deep copy
    assert id(activated_adapter.scaling) != id(config)
    assert activated_adapter.scaling == cal_scaling(activated_adapter, config)
    config.alpha = 32
    assert activated_adapter.scaling != cal_scaling(activated_adapter, config)
    AdapterAPI.update_adapter(model, config)
    assert activated_adapter.scaling == cal_scaling(activated_adapter, config)

    with pytest.raises(ValueError, match="Model does not have adapter named"):
        new_config = deepcopy(config)
        new_config.alpha = 64
        new_config.adapter_name = "new_adapter_name"
        AdapterAPI.update_adapter(model, new_config)
    # If update fail, config hold the same, test with old config
    assert activated_adapter.scaling == cal_scaling(activated_adapter, config)
    assert activated_adapter.scaling != cal_scaling(activated_adapter, new_config)

    # --remove_adapter--
    with pytest.raises(ValueError, match="does not have adapter named"):
        AdapterAPI.remove_adapter(
            model, [adapter_name, "another_not_added_adapter_name"]
        )

    # Adapter is still not change after remove fail
    assert not torch.all(output1 == model(sample))

    # output now back to the original
    AdapterAPI.remove_adapter(model, [adapter_name])
    assert torch.all(output1 == model(sample))
    assert AdapterAPI.get_adapter_names(model) is None
