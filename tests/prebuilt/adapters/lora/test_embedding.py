from random import randint
from typing import cast

import pytest
import torch
from torch import nn

from petorch.prebuilt.adapters.lora import LoraAdaptedLayer, LoraEmbedding
from petorch.prebuilt.configs import LoraConfig
from petorch.utilities import TorchInitMethod

adapter_name = "test_prebuild_linear_lora_adapter"
fqname = "base_layer_fqname"
emb_size = (1000, 500)

"""Vary these variableS to the large numbers for ensuring the testing is True for most cases. But the test will run for a longer time."""

NUM_TEST_SAMPLES = 10
"""Number of sample input to test."""

NUM_TEST_CONFIG_LISTS = 10
"""Number of config lists, where each list has length==`NUM_CONCURRENT_ADAPTERS`.  """

NUM_CONCURRENT_ADAPTERS = 10
"""For testing with multi adapters, this is the number of adapter model have at the same time. """


@torch.no_grad()
@pytest.mark.parametrize(
    "config",
    [
        LoraConfig(adapter_name=adapter_name),
        LoraConfig(adapter_name=adapter_name),
    ],
)
@pytest.mark.parametrize(
    "sample",
    [
        torch.tensor([randint(0, 1000) for _ in range(3)])
        for _ in range(NUM_TEST_SAMPLES)
    ],
)
def test_lora_embedding_single_adapter(config: LoraConfig, sample: torch.Tensor):
    base_layer = nn.Embedding(*emb_size)
    adapted_layer = config.dispatch_adapted_layer(fqname, base_layer)
    adapted_layer.eval()
    zero_init_adapter = config.dispatch_adapter(
        fqname, base_layer, lora_init_method=TorchInitMethod.zeros
    )
    ones_init_adapter = config.dispatch_adapter(
        fqname, base_layer, lora_init_method=TorchInitMethod.ones
    )
    assert isinstance(zero_init_adapter, LoraEmbedding)
    assert isinstance(ones_init_adapter, LoraEmbedding)

    # ---Zero init---
    adapted_layer._add_adapters(zero_init_adapter, activate=False)
    adapted_layer.eval()  # IMPORTANT: MUST SWAP TO EVAL AFTER CHANGE ARCHITECTURE.

    assert isinstance(adapted_layer, LoraAdaptedLayer)

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
    assert torch.allclose(
        o := adapted_layer(sample), activated_adapter_output, atol=5e-5
    ), f"max_abs = {(activated_adapter_output-o).abs().max()}"

    # No activated adapter case
    assert torch.allclose(
        adapted_layer.base_layer(sample), activated_adapter_output, atol=5e-5
    ), f"max_abs = {(activated_adapter_output-o).abs().max()}"

    # Unmerged, deactivate case.
    adapted_layer._unmerge(adapter_name)
    assert len(adapted_layer.merged_adapter_names) == 0
    assert not torch.allclose(adapted_layer(sample), activated_adapter_output)

    # Everything like the original output
    # Tested for 500 times, and the maximum max_abs for all times is 1.13e-5, so I let 1.2e-5 here.
    assert torch.allclose(
        o := adapted_layer(sample), output, atol=1.2e-5
    ), f"max_abs = {(o-output).abs().max()}"

    assert torch.allclose(o, adapted_layer.base_layer(sample))


def _randint_list(n_list: int, l_len: int, a: int, b: int):
    for _ in range(n_list):
        l = []
        for _ in range(l_len):
            l.append(randint(a, b))
        yield l


def _make_adapter_configs(num_current_adapters: int) -> list[LoraConfig]:
    return [
        LoraConfig(
            adapter_name=f"adapter_{i}",
            rank=4 * rank,
            alpha=8 * rank,
            scale=scale / num_current_adapters,
        )
        for i, [rank, scale] in enumerate(_randint_list(num_current_adapters, 2, 2, 10))
    ]


@torch.no_grad()
@pytest.mark.parametrize(
    "configs",
    [
        _make_adapter_configs(NUM_CONCURRENT_ADAPTERS)
        for _ in range(NUM_TEST_CONFIG_LISTS)
    ],
)
@pytest.mark.parametrize(
    "sample",
    [
        torch.tensor([randint(0, 1000) for _ in range(3)])
        for _ in range(NUM_TEST_SAMPLES)
    ],
)
def test_lora_embedding_multi_adapters(sample, configs):
    base_layer = nn.Embedding(*emb_size)
    adapters = [
        cast(LoraConfig, config).dispatch_adapter(fqname, base_layer, lora_init_method=TorchInitMethod.normal)
        for config in configs
    ]
    assert None not in adapters
    adapter_names = [adapter.name for adapter in adapters]

    adapted_layer = cast(
        LoraAdaptedLayer, configs[0].dispatch_adapted_layer(fqname, base_layer)
    )
    assert isinstance(adapted_layer, LoraAdaptedLayer)

    # None activated adapter cases
    adapted_layer._add_adapters(adapters, activate=False)
    assert (
        len(adapted_layer.non_active_adapters)
        == len(adapters)
        == len(adapted_layer.adapter_names)
    )

    adapted_layer.eval()

    output = adapted_layer(sample)
    """Non activate output"""
    assert torch.isfinite(output).all()
    assert torch.allclose(adapted_layer.base_layer(sample), output)

    # Activated case
    adapted_layer._activate_adapters(adapter_names, activate=True)
    assert (
        len(adapted_layer.active_adapters)
        == len(adapters)
        == len(adapted_layer.adapter_names)
    )

    activated_output = adapted_layer(sample)
    assert torch.isfinite(activated_output).all()

    assert not torch.allclose(
        activated_output, output
    ), f"max_abs = {(activated_output-output).abs().max()}"

    assert torch.allclose(
        adapted_layer.base_layer(sample), output
    )  # The base layer still doesn't change

    # Merge
    assert not adapted_layer.is_merged
    adapted_layer._merge(adapter_names)
    assert adapted_layer.is_merged
    assert len(adapted_layer.merged_adapter_names) == len(adapter_names)

    # After merge, the base layer is used only, and output equal to the activated adapters case
    assert torch.allclose(
        o := adapted_layer(sample), activated_output, atol=3e-5
    ), f"max_abs = {(activated_output-o).abs().max()}"
    assert torch.allclose(
        o := adapted_layer.base_layer(sample), activated_output, atol=3e-5
    ), f"max_abs = {(activated_output-o).abs().max()}"

    # Unmerge
    adapted_layer._unmerge(adapter_names)
    assert not adapted_layer.is_merged

    # It's still activated now, so
    assert torch.allclose(
        o := adapted_layer(sample), activated_output, atol=1e-5
    ), f"max_abs = {(activated_output-o).abs().max()}"

    # But only base does NOT equal
    assert not torch.allclose(
        o := adapted_layer.base_layer(sample), activated_output
    ), f"max_abs = {(activated_output-o).abs().max()}"

    # Deactivate
    adapted_layer._activate_adapters(adapter_names, activate=False)
    assert (
        len(adapted_layer.non_active_adapters)
        == len(adapters)
        == len(adapted_layer.adapter_names)
    )
    assert len(adapted_layer.active_adapters) == 0

    # Check with non-activate output
    assert torch.allclose(
        o := adapted_layer(sample), output, atol=3e-5
    ), f"max_abs = {(output-o).abs().max()}"

    assert torch.allclose(
        o := adapted_layer.base_layer(sample), output, atol=3e-5
    ), f"max_abs = {(output-o).abs().max()}"
