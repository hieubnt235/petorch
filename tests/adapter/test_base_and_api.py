from copy import deepcopy
from random import randint
from typing import cast

import pytest
import torch
from loguru import logger
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from petorch.adapter import BaseAdapter, AdapterAPI, BaseAdaptedLayer
from petorch.prebuilt.adapters.lora import LoraLinear, LoraAdaptedLayer
from petorch.prebuilt.configs import LoraLinearModelConfig
from petorch.utilities import TorchInitMethod
from petorch.utilities.dummy import NestedDummy, Dummy

sample_size = (3, 8, 8)


@pytest.mark.parametrize("model_cls", [NestedDummy, Dummy])
def test_api(model_cls):
    adapter_name = "test_adapter"
    model = model_cls()
    org_model = deepcopy(model)
    config = LoraLinearModelConfig(adapter_name=adapter_name)

    sample = torch.rand((2,) + sample_size)

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
        LoraLinear,
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
    # If update fails, config will be the same, test it with old config.
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


"""Vary these variableS to the large numbers for ensuring the testing is True for most cases. But the test will run for a longer time."""

NUM_TEST_SAMPLES = 5
"""Number of sample input to test."""

NUM_TEST_CONFIG_LISTS = 5
"""Number of config lists, where each list has length==`NUM_CONCURRENT_ADAPTERS`.  """

NUM_CONCURRENT_ADAPTERS = 5
"""For testing with multi adapters, this is the number of adapter model have at the same time.
 Because of tolerances accumulations, more adapters make the model from reconstruct process (remove, unmerge) more different than original model.
 So that if using too much (>3), refer reload model instead of reconstruct.
 Here set to 5 for stress test. See the different detail in tests.
 """


def _randint_list(n_list: int, l_len: int, a: int, b: int):
    for _ in range(n_list):
        l = []
        for _ in range(l_len):
            l.append(randint(a, b))
        yield l


def _make_adapter_configs(num_current_adapters: int) -> list[LoraLinearModelConfig]:
    return [
        LoraLinearModelConfig(
            adapter_name=f"adapter_{i}",
            rank=4 * rank,
            alpha=8 * rank,
            bias=bias < 5,
            scale=scale/num_current_adapters, # For a stable test when added too many adapters.
        )
        for i, [rank, bias, scale] in enumerate(_randint_list(num_current_adapters, 3, 2, 10))
    ]


@torch.no_grad()
@pytest.mark.parametrize(
    "configs",
    [
        _make_adapter_configs(NUM_CONCURRENT_ADAPTERS)
        for _ in range(NUM_TEST_CONFIG_LISTS)
    ],
)
@pytest.mark.parametrize("model_cls", [NestedDummy, Dummy])
def test_manipulate_multi_adapters(configs, model_cls):
    model = NestedDummy()
    org_model = deepcopy(model)
    sample = torch.randn((2,) + sample_size)
    base_output: torch.Tensor = org_model(sample)

    # Add adapters
    adapted_fqn: list[str] = []
    for config in configs:
        adapted_fqn = AdapterAPI.add_adapter(
            model, config, lora_init_method=TorchInitMethod.normal
        )
        # print(model)
        # print("=="*100)
    adapter_names = [config.adapter_name for config in configs]
    assert isinstance(
        adapted_layer := model.get_submodule(
            adapted_fqn[randint(0, len(adapted_fqn) - 1)]
        ),
        LoraAdaptedLayer,
    )  # Get any adapted layer. Note that this layer is already tested in `test_api`.Just get for monitoring.

    # None activated adapter cases
    assert (
        len(AdapterAPI.get_adapter_names(model))
        == len(adapter_names)
        == len(adapted_layer.non_active_adapters)
        == len(adapted_layer.adapter_names)
        and len(adapted_layer.active_adapters) == 0
    )
    assert torch.allclose(model(sample), base_output)

    # Activate and ensure all adapted layers are activated
    AdapterAPI.activate_adapter(model, adapter_names, activate=True)
    for fqn in adapted_fqn:
        al = model.get_submodule(fqn)
        assert isinstance(al, LoraAdaptedLayer)
        assert (
            len(adapted_layer.active_adapters)
            == len(adapted_layer.adapter_names)
            == len(adapter_names)
            and len(adapted_layer.non_active_adapters) == 0
        )

    # No equal anymore
    assert not torch.allclose(model(sample), base_output, atol=1e-2)

    # Deactivate
    AdapterAPI.activate_adapter(model, adapter_names, activate=False)
    assert torch.allclose(model(sample), base_output)

    # Remove adapter, model are completely fresh
    AdapterAPI.remove_adapter(
        model,
        adapter_names,
    )
    with pytest.raises(ValueError, match="Model does not have any adapter"):
        AdapterAPI.activate_adapter(model)

    for name, param in model.named_parameters():
        org_param = org_model.get_parameter(name)
        assert torch.allclose(param, org_param)
    assert torch.allclose(model(sample), org_model(sample))


def assert_models_are_identical(model_a: nn.Module, model_b: nn.Module):
    names_a = AdapterAPI.get_adapter_names(model_a)
    names_b = AdapterAPI.get_adapter_names(model_b)
    m = f"Model architectures not match in adapters. Got `{names_a}` and `{names_b}`."
    if names_a is None:
        assert names_b is None, m
    else:
        assert names_b is not None, m
        names_a.sort()
        names_b.sort()
        assert names_a == names_b, m

    modules_a = {name: module for name, module in model_a.named_modules()}
    modules_b = {name: module for name, module in model_b.named_modules()}
    params_a = {name: param for name, param in model_a.named_parameters()}
    params_b = {name: param for name, param in model_b.named_parameters()}

    assert (l1 := len(modules_a)) == (
        l2 := len(modules_b)
    ), f"Model architectures not match in number of modules. Got {l1} and {l2}."
    assert (l1 := len(params_a)) == (
        l2 := len(params_b)
    ), f"Model architectures not match in number of parameters. Got {l1} and {l2}."

    for name, module_a in modules_a.items():
        module_b = model_b.get_submodule(name)
        assert (a := type(module_a)) == (
            b := type(module_b)
        ), f"Model architectures not match in module type at fqn={name}. Got{a}!={b}. "

    for name, param_a in params_a.items():
        param_b = model_b.get_parameter(name)
        assert torch.allclose(param_a, param_b, atol=3e-5), (
            f"Params {name} does not match. "
            f"max_abs = {(param_a-param_b).abs().max()},"
            f" max={max(param_a.max().item(),param_b.max().item())}"
        )


def register_record_hook(
    model: nn.Module, record_dict: dict, fqns: list[str]
) -> list[RemovableHandle]:
    """
    Format:
    {"fqn":{"input":tensor1, "output":tensor}
    """
    handles: list[RemovableHandle] = []
    for fqn in fqns:
        # Note, use default arg for capture, if not, it takes the reference of fqn, then error will happened later.
        def record(module, args, output, captured_fqn=fqn):
            record_dict[captured_fqn] = {"input": args, "output": output}

        handles.append(model.get_submodule(fqn).register_forward_hook(record))

    return handles


@torch.no_grad()
@pytest.mark.parametrize(
    "configs",
    [
        _make_adapter_configs(NUM_CONCURRENT_ADAPTERS)
        for _ in range(NUM_TEST_CONFIG_LISTS)
    ],
)
@pytest.mark.parametrize(
    "sample", [torch.randn((2,) + sample_size) for _ in range(NUM_TEST_SAMPLES)]
)
@pytest.mark.parametrize(
    "model_cls",
    [NestedDummy, Dummy],
)
def test_merge_api(
    configs: list[LoraLinearModelConfig], sample: torch.Tensor, model_cls
):
    model = model_cls()
    org_model = deepcopy(model)
    model.eval()
    org_model.eval()

    base_output: torch.Tensor = org_model(sample)
    assert torch.isfinite(base_output).all()

    adapter_names = [config.adapter_name for config in configs]
    adapted_fqns: list[str] = []
    for config in configs:
        adapted_fqns = AdapterAPI.add_adapter(
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
            if isinstance(module, LoraAdaptedLayer):
                assert name in adapted_fqns
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
                else:
                    assert not module.is_merged
                    assert len(module.merged_adapter_names) == 0

                current_adapted_fqn += 1
            else:
                assert name not in adapted_fqns

        assert current_adapted_fqn == len(adapted_fqns)

    # After add adapters
    _assert_correct_adapted_layer(is_activated=False)

    assert isinstance(
        adapted_layer := model.get_submodule(
            adapted_fqns[randint(0, len(adapted_fqns) - 1)]
        ),
        LoraAdaptedLayer,
    )  # Get any adapted layer. Note that this layer is already tested in `test_api`.Just get for monitoring.

    # None-activated-adapter case
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

    # Record all the adapted layers.
    activated_al_records = {}
    aal_handles = register_record_hook(model, activated_al_records, adapted_fqns)
    model.eval()
    activated_output = model(sample)
    assert torch.isfinite(base_output).all()
    for handle in aal_handles:
        handle.remove()

    # Merge
    assert not adapted_layer.is_merged
    AdapterAPI.merge(model, adapter_names)
    model.eval()
    _assert_correct_adapted_layer(is_activated=True, is_merged=True)

    # Check the equality between merged case and activated adapters case, both must the as close as possible.
    merged_al_records = {}
    mal_handle = register_record_hook(model, merged_al_records, adapted_fqns)
    model.eval()
    merged_output = model(sample)
    assert torch.isfinite(base_output).all()
    for handle in mal_handle:
        handle.remove()

    assert (
        len(aal_handles)
        == len(mal_handle)
        == len(activated_al_records)
        == len(merged_al_records)
        == len(adapted_fqns)
    )

    # Test that all adapted layer inputs and outputs, in both activate case and merge case, are the same.
    for fqn in adapted_fqns:
        # Note that this allows the higher tolorances, because the input is not exactly the same as the original also, it variates
        # from the previous layer.
        assert torch.allclose(
            a := activated_al_records[fqn]["input"][0],
            m := merged_al_records[fqn]["input"][0],
            atol=5e-5
        ), f"max_abs = {(a-m).abs().max()}, max={max(a.max(),m.max())}"

        assert torch.allclose(
            a := activated_al_records[fqn]["output"],
            m := merged_al_records[fqn]["output"],
            atol=5e-4,rtol=1e-4
        ), f"max_abs = {(a-m).abs().max()}, a_max={a.max()}, m_max={m.max()}"

    assert torch.allclose(
        merged_output, activated_output, atol=5e-6
    ), f"max_abs = {(activated_output-merged_output).abs().max()}"

    # Unmerge, still activated
    AdapterAPI.unmerge(model, adapter_names)
    _assert_correct_adapted_layer(is_activated=True, is_merged=False)
    # Now the base model output is equal to activated_output
    assert torch.allclose(
        o := model(sample), activated_output, atol=2e-6
    ), f"max_abs = {(o-activated_output).abs().max()}, max={max(o.max(),activated_output.max())}"

    # Deactivate
    AdapterAPI.activate_adapter(model, adapter_names, activate=False)
    _assert_correct_adapted_layer(is_activated=False, is_merged=False)
    # Now the base model output is equal to the original output
    assert torch.allclose(
        o := model(sample), b := base_output, atol=5e-4,rtol=1e-4
    ), f"max_abs = {(o-base_output).abs().max()}, o_max={o.max()},b_max = {b.max().item()}"

    # Remove adapters
    with pytest.raises(AssertionError, match="Model architectures not match"):
        assert_models_are_identical(model, org_model)
    AdapterAPI.remove_adapter(model, adapter_names)
    assert_models_are_identical(model, org_model)

    assert str(org_model) == str(model)
    logger.debug(
        f"\nOriginal model:\n{org_model}" f"\nProcessed and cleaned up model:\n{model}"
    )
