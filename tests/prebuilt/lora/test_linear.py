import pytest
import torch
from torch import nn

from petorch.prebuilt.configs import LoraLinearModelConfig
from petorch.prebuilt.lora import LoraLinearAdaptedLayer, LoraLinearAdapter
from petorch.utilities import TorchInitMethod

adapter_name = "test_prebuild_linear_lora_adapter"
fqname = "base_layer_fqname"
linear_size = (32, 32)


@torch.no_grad
@pytest.mark.parametrize(
    "config",
    [
        LoraLinearModelConfig(adapter_name=adapter_name, bias=False),
        LoraLinearModelConfig(adapter_name=adapter_name, bias=True),
    ],
)
@pytest.mark.parametrize("base_layer_bias", [True, False])
@pytest.mark.parametrize("dropout", [0, 0.5])
@pytest.mark.parametrize(
    "sample", [torch.randn([2, 32]) for _ in range(3)]
)  # Test many random tensors for ensuring it's general for all cases.
def test_lora_linear(config, base_layer_bias, dropout, sample):
    base_layer = nn.Linear(*linear_size, bias=base_layer_bias)
    config.dropout = dropout
    adapted_layer = config.dispatch_adapted_layer(fqname, base_layer)
    adapted_layer.eval()
    zero_init_adapter = config.dispatch_adapter(
        fqname, base_layer, lora_init_method=TorchInitMethod.zeros
    )
    ones_init_adapter = config.dispatch_adapter(
        fqname, base_layer, lora_init_method=TorchInitMethod.ones
    )
    assert isinstance(zero_init_adapter, LoraLinearAdapter)
    assert isinstance(ones_init_adapter, LoraLinearAdapter)
    
    # ---Zero init---
    adapted_layer._add_adapters(zero_init_adapter, activate=False)
    adapted_layer.eval()  # IMPORTANT: MUST SWAP TO EVAL AFTER CHANGE ARCHITECTURE.

    assert isinstance(adapted_layer, LoraLinearAdaptedLayer)

    # For non-activated adapter case, adapted_layer and base_layer output is the same
    output = adapted_layer(sample)
    assert torch.allclose(output, adapted_layer.base_layer(sample))

    # No active adapter, the forward is the same as base_layer forward
    assert torch.allclose(output, base_layer(sample))

    # Still not change, because all lora layer is zeros init.
    adapted_layer._activate_adapters(adapter_name, activate=True)
    assert torch.allclose(
        output,
        o := adapted_layer(sample),
    ), f"abs = {(o-output).abs().max()}"  # ~9e-6
    adapted_layer._remove_adapters(adapter_name)

    # ---Ones init---
    adapted_layer._add_adapters(ones_init_adapter, activate=False)
    adapted_layer.eval()  # IMPORTANT: MUST SWAP TO EVAL AFTER CHANGE ARCHITECTURE.

    assert torch.allclose(output, adapted_layer(sample))  # None activated case

    adapted_layer._activate_adapters(adapter_name, activate=True)
    assert len(adapted_layer.active_adapters) == 1
    # Now it changes.
    assert not torch.allclose(output, adapted_layer(sample))

    # ---Merge---
    activated_adapter_output = adapted_layer(sample)  # Activated

    # Merge
    assert not adapted_layer.is_merged
    adapted_layer._merge(adapter_name)
    assert adapted_layer.is_merged
    assert (
        len(adapted_layer.merged_adapter_names) == 1
        and adapter_name in adapted_layer.merged_adapter_names
    )

    # Deactivate
    adapted_layer._activate_adapters(adapter_name, activate=False)
    assert len(adapted_layer.active_adapters) == 0
    assert len(adapted_layer.non_active_adapters) == 1

    # Check merged-unactivated case == unmerged-activated case
    # Notes: Sometime error because of out `xtol`(approx 5e-5)
    assert torch.allclose(
        o := adapted_layer(sample), activated_adapter_output
    ), f"max_abs = {(activated_adapter_output-o).abs().max()}"

    assert torch.allclose(
        adapted_layer.base_layer(sample), activated_adapter_output
    )  # No activated adapter case

    # Unmerged, deactivate case.
    adapted_layer._unmerge(adapter_name)
    assert len(adapted_layer.merged_adapter_names) == 0
    assert not torch.allclose(adapted_layer(sample), activated_adapter_output)

    # Everything like the original output
    assert torch.allclose(
        o := adapted_layer(sample), output, atol=1e-5
    ), f"max_abs = {(o-output).abs().max()}"  # ~9e-6
    assert torch.allclose(o, adapted_layer.base_layer(sample))
