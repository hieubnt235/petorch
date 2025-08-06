from copy import deepcopy

import pytest
import torch

from petorch.adapter import AdapterAPI
from petorch.prebuilt.dummy import NestedDummy
from petorch.prebuilt.configs import LoraLinearModelConfig
from petorch.utilities import TorchInitMethod


@pytest.fixture
def sample():
    return torch.rand([2, 3, 8, 8])


def test_initial_output(sample):
    adapter_name = "test_initial_output_adapter_name"
    org_model = NestedDummy()
    model = deepcopy(org_model)
    config = LoraLinearModelConfig(adapter_name=adapter_name)

    AdapterAPI.add_adapter(model, config, lora_init_method=TorchInitMethod.zeros)
    # Get outputs from both layers
    with torch.no_grad():
        assert torch.allclose(model(sample), org_model(sample))

        AdapterAPI.activate_adapter(model, adapter_name, activate=True)
        # Still equal, because the lora initialize weights is zeros.
        assert torch.allclose(model(sample), org_model(sample))
