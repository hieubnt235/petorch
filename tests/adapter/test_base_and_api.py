from copy import deepcopy
from random import randint
from typing import cast, Type

import pytest
import torch
from petorch import logger
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from petorch.adapter import BaseAdapter, AdapterAPI, BaseAdaptedLayer
from petorch.prebuilt.adapters.lora import LoraLinear, LoraAdaptedLayer, BaseLoraAdapter
from petorch.prebuilt.configs import LoraConfig
from petorch.utilities import TorchInitMethod, DummyV2
from petorch.utilities.modules import NestedDummy, Dummy

sample_size = (3, 8, 8)
emb_idx = torch.randint(0, 100, [2, 192])


@pytest.mark.parametrize("model_cls", [NestedDummy, DummyV2])
def test_add_remove_activate_update_apis(model_cls):
    adapter_name = "test_adapter"
    model = model_cls()
    org_model = deepcopy(model)
    config = LoraConfig(adapter_name=adapter_name)

    sample = torch.rand((2,) + sample_size)

    output1 = cast(Tensor, org_model(sample, emb_idx))

    # --add_adapter--
    target_fqn = []
    for name, module in model.named_modules():
        if (adapter := config.dispatch_adapter(name, module)) is not None:
            assert isinstance(adapter, BaseAdapter)
            target_fqn.append(name)
    target_fqn.sort()

    fqn = AdapterAPI.add_adapter(model, config, lora_init_method=TorchInitMethod.kaiming_uniform)
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
    assert torch.all(output1 == model(sample, emb_idx))  # Test not activate adapter

    # --activate_adapter--
    with pytest.raises(ValueError, match="does not have adapter"):
        AdapterAPI.activate_adapter(model, "random_other_adapter_name")

    # Activate fail, nothing change, just like not activating
    assert adapter_name in adapted_layer.non_active_adapters
    assert torch.all(output1 == model(sample, emb_idx))  # Test not activate adapter

    # After activate successfully, the model output will be changed
    activated_adapter_names = AdapterAPI.activate_adapter(model, adapter_name)
    assert activated_adapter_names[0] == adapter_name
    assert adapter_name in adapted_layer.active_adapters
    assert not torch.all(output1 == model(sample, emb_idx))

    # --update_adapter--
    # noinspection PyTypeChecker
    assert isinstance(
        (activated_adapter := adapted_layer.active_adapters[adapter_name]),
        BaseLoraAdapter,
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
    assert not torch.all(output1 == model(sample, emb_idx))

    # output now back to the original
    AdapterAPI.remove_adapter(model, [adapter_name])
    assert torch.all(output1 == model(sample, emb_idx))
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


def _make_adapter_configs(num_current_adapters: int) -> list[LoraConfig]:
    return [
        LoraConfig(
            adapter_name=f"adapter_{i}",
            rank=4 * rank,
            alpha=8 * rank,
            bias=bias < 5,
            scale=scale
            / num_current_adapters,  # For a stable test when added too many adapters.
        )
        for i, [rank, bias, scale] in enumerate(
            _randint_list(num_current_adapters, 3, 2, 10)
        )
    ]


@torch.no_grad()
@pytest.mark.parametrize(
    "configs",
    [
        _make_adapter_configs(NUM_CONCURRENT_ADAPTERS)
        for _ in range(NUM_TEST_CONFIG_LISTS)
    ],
)
@pytest.mark.parametrize("model_cls", [NestedDummy, DummyV2])
def test_manipulate_multi_adapters(configs, model_cls):
    model = NestedDummy()
    org_model = deepcopy(model)
    sample = torch.randn((2,) + sample_size)
    base_output: torch.Tensor = org_model(sample, emb_idx)

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
    assert torch.allclose(model(sample, emb_idx), base_output)

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
    assert not torch.allclose(model(sample, emb_idx), base_output, atol=1e-2)

    # Deactivate
    AdapterAPI.activate_adapter(model, adapter_names, activate=False)
    assert torch.allclose(model(sample, emb_idx), base_output)

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
    assert torch.allclose(model(sample, emb_idx), org_model(sample, emb_idx))


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
        try:
            module_b = model_b.get_submodule(name)
        except AttributeError:
            raise AssertionError(
                f"Model architectures not match in schema. `model_b` does not have module at `{name}`."
            )

        assert (a := type(module_a)) == (
            b := type(module_b)
        ), f"Model architectures not match in module type at fqn={name}. Got{a}!={b}. "

    for name, param_a in params_a.items():
        param_b = model_b.get_parameter(name)
        assert torch.allclose(param_a, param_b, atol=5e-5), (
            f"Params does not match: `{name}`. "
            f"max_abs = {(param_a-param_b).abs().max()},"
            f" max={max(param_a.max().item(),param_b.max().item())}"
        )


def z(x: Tensor, epsilon=1e-8):
    mean = x.float().mean()
    std = x.float().std()
    return (x - mean) / (std + epsilon)


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


def weights_init(m: nn.Module):

    if isinstance(getattr(m, "weight", None), nn.Parameter):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.normal_(m.weight)
    if isinstance(getattr(m, "bias", None), nn.Parameter):
        nn.init.constant_(m.bias, 0)


_debugged_models = []


def debug_model(text: str, model: nn.Module, stop_debug: bool = False):
    if type(model) not in _debugged_models:
        logger.debug(f"\n{text}:{"~"*100}\n{model}\n{"="*100}")
    if stop_debug:
        _debugged_models.append(type(model))


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
    [NestedDummy, DummyV2],
)
def test_merge_api(configs: list[LoraConfig], sample: torch.Tensor, model_cls):

    model = model_cls()
    model.apply(weights_init)
    org_model = deepcopy(model)
    model.eval()
    org_model.eval()

    base_output: torch.Tensor = org_model(sample, emb_idx)
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
    assert torch.allclose(model(sample, emb_idx), base_output)

    # Activate
    activated_names = AdapterAPI.activate_adapter(model, adapter_names, activate=True)
    model.eval()
    _assert_correct_adapted_layer(is_activated=True)
    assert not torch.allclose(model(sample, emb_idx), base_output)

    # Record all the adapted layers.
    activated_al_records = {}
    aal_handles = register_record_hook(model, activated_al_records, adapted_fqns)
    model.eval()
    activated_output = model(sample, emb_idx)
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
    merged_output: Tensor = model(sample, emb_idx)
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
        # Note that this allows the higher tolerances, because the input is not exactly the same as the original also, it variates
        # from the previous layer. Also, the outputs are often not stable because of not norm.
        assert torch.allclose(
            a := z(activated_al_records[fqn]["input"][0]),
            m := z(merged_al_records[fqn]["input"][0]),
            atol=2e-5,
            # rtol=1e-4,
        ), f"`{fqn}`max_abs = {(a-m).abs().max()}, a_max={a.max()}, m_max={m.max()}"

        assert torch.allclose(
            a := z(activated_al_records[fqn]["output"]),
            m := z(merged_al_records[fqn]["output"]),
            atol=1e-5,
            # rtol=1e-4,
        ), f"`{fqn}`:max_abs = {(a-m).abs().max()}, a_max={a.max()}, m_max={m.max()}"

    assert torch.allclose(
        merged_output, activated_output, atol=2e-5
    ), f"max_abs = {(merged_output-activated_output).abs().max()}, m_max={merged_output.max()} a_max={activated_output.max()}"

    # Unmerge, still activated
    AdapterAPI.unmerge(model, adapter_names)
    _assert_correct_adapted_layer(is_activated=True, is_merged=False)

    # Now the base model output is equal to activated_output
    assert torch.allclose(
        o := model(sample, emb_idx), activated_output, atol=1e-5
    ), f"max_abs = {(o-activated_output).abs().max()}, max={max(o.max(),activated_output.max())}"

    # Deactivate
    AdapterAPI.activate_adapter(model, adapter_names, activate=False)
    _assert_correct_adapted_layer(is_activated=False, is_merged=False)

    # Now the base model output is equal to the original output (reconstruction will have more differences than tests before).
    # The atol is 5e-3 somehow high, but the mean is often 5e-4.
    assert torch.allclose(
        o := model(sample, emb_idx), b := base_output, atol=5e-3
    ), f"max_abs = {(o-base_output).abs().max()},mean={(o-base_output).abs().mean()} o_max={o.max()},b_max = {b.max()}"

    debug_model(
        "Original model",
        org_model,
    )
    debug_model(
        "Adapted model",
        model,
    )

    # Remove adapters
    with pytest.raises(AssertionError, match="Model architectures not match"):
        assert_models_are_identical(model, org_model)
    AdapterAPI.remove_adapter(model, adapter_names)
    assert_models_are_identical(model, org_model)

    assert str(org_model) == str(model)
    debug_model("Processed and cleaned up model", model, True)


@torch.no_grad()
@pytest.mark.parametrize(
    "configs",
    [
        _make_adapter_configs(NUM_CONCURRENT_ADAPTERS)
        for _ in range(NUM_TEST_CONFIG_LISTS)
    ],
)
@pytest.mark.parametrize(
    "model_cls",
    [NestedDummy, DummyV2],
)
def test_get_and_load_state_dict_apis(
    configs: list[LoraConfig], model_cls: type(nn.Module)
):
    model: nn.Module = model_cls()
    model.apply(weights_init)
    org_model = deepcopy(model)

    adapter_names = [config.adapter_name for config in configs]
    adapted_fqns: list[str] = []

    # All adapters has zero weights init
    for config in configs:
        adapted_fqns = AdapterAPI.add_adapter(
            model, config, lora_init_method=TorchInitMethod.zeros
        )
    adapted_fqns.sort()
    assert adapted_fqns == AdapterAPI.get_adapted_fqnames(model)

    adt_state_dict = AdapterAPI.get_adapter_state_dict(model)
    adt_sd_keys = list(adt_state_dict.keys())
    # logger.debug(
    #     f"{"\n".join([f"{k}:{v.shape}" for k , v in adt_state_dict.items()  ])}"
    # )

    # All adapter weights according to the adapted fqn are available in state_dict.
    key_fmt = "{adt_fqn}.{adt_key}.{adt_name}.{lora_X}.{wnb}"
    adt_keys = [LoraAdaptedLayer.act_adt_key, LoraAdaptedLayer.non_act_adt_key]

    configs_dict = {config.adapter_name: config for config in configs}
    assert configs_dict == AdapterAPI.get_adaption_configs(model)

    available_state_keys: list[str] = []

    def _assert_keys(adt_fqn, adt_keys, adt_name, model: nn.Module):
        check_keys = []
        for lx in ["lora_A", "lora_B"]:
            check_keys.append(
                key_fmt.format(
                    adt_fqn=adt_fqn,
                    adt_key=adt_keys,
                    adt_name=adt_name,
                    lora_X=lx,
                    wnb="weight",
                )
            )
            # Only `lora_B` with base_layer is not nn.Embedding can have bias.
            if (
                configs_dict[adt_name].bias
                and lx == "lora_B"
                and (
                    not isinstance(
                        m := model.get_submodule(adt_fqn).base_layer, nn.Embedding
                    )
                )
            ):
                check_keys.append(
                    key_fmt.format(
                        adt_fqn=adt_fqn,
                        adt_key=adt_keys,
                        adt_name=adt_name,
                        lora_X=lx,
                        wnb="bias",
                    )
                )
                # logger.debug(f"Append `{check_keys[-1]}` to check_keys. m={m},adt_fqn={adt_fqn}")

        assert all(
            [k in adt_sd_keys for k in check_keys]
        ), f"\n{check_keys}\n {"\n".join([k for k in adt_sd_keys])}"

        available_state_keys.extend(check_keys)

    for adt_fqn in adapted_fqns:
        for adt_name in adapter_names:
            if (
                key_fmt.format(
                    adt_fqn=adt_fqn,
                    adt_key=adt_keys[0],
                    adt_name=adt_name,
                    lora_X="lora_A",
                    wnb="weight",
                )
                in adt_sd_keys
            ):
                _assert_keys(adt_fqn, adt_keys[0], adt_name, model)
            elif (
                key_fmt.format(
                    adt_fqn=adt_fqn,
                    adt_key=adt_keys[1],
                    adt_name=adt_name,
                    lora_X="lora_A",
                    wnb="weight",
                )
                in adt_sd_keys
            ):
                _assert_keys(adt_fqn, adt_keys[1], adt_name, model)
            else:
                raise AssertionError(
                    f"{adt_fqn}.{adt_keys}.{adt_name}.lora_X.wnb not exists in state_dict."
                )
    available_state_keys.sort()
    adt_sd_keys.sort()
    assert (
        available_state_keys == adt_sd_keys
    ), f"{len(available_state_keys)}-{len(adt_sd_keys)}"

    # Add adapters to org_model, but different init method than before (now: ones, before: zeros)
    model2 = deepcopy(org_model)
    for config in configs:
        adapted_fqns = AdapterAPI.add_adapter(
            model2, config, lora_init_method=TorchInitMethod.ones
        )
    adapted_fqns.sort()
    assert adapted_fqns == AdapterAPI.get_adapted_fqnames(model)

    with pytest.raises(AssertionError, match="Params does not match"):
        assert_models_are_identical(model, model2)

    # Now identical
    not_load_keys = AdapterAPI.load_adapter_state_dict(
        model2, adt_state_dict, strict_load=True
    )
    assert not_load_keys == {}, not_load_keys
    assert_models_are_identical(model, model2)

    # Test fuse activate and non activate
    model3 = deepcopy(org_model)
    for config in configs:
        adapted_fqns = AdapterAPI.add_adapter(
            model3, config, lora_init_method=TorchInitMethod.ones
        )
    with pytest.raises(AssertionError, match="Params does not match"):
        assert_models_are_identical(model, model3)

    # Activate adapter
    AdapterAPI.activate_adapter(model3, adapter_names[0], activate=True)
    with pytest.raises(AssertionError, match="Model architectures not match in schema"):
        assert_models_are_identical(model, model3)

    # Load, deactivate and now identical
    not_load_keys = AdapterAPI.load_adapter_state_dict(
        model3, adt_state_dict, strict_load=True, strict_activation=False  # Default
    )
    AdapterAPI.deactivate_adapter(model3)
    assert not_load_keys == {}, not_load_keys
    assert_models_are_identical(model, model3)
