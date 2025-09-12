import faulthandler

from torch import Tensor

faulthandler.enable()

import torch
from petorch.quantization.qtensors.nf4 import (
    F4QTensor,
    F4QConfig,
)  # or your module path


def make_nf4(tensor: Tensor, config: F4QConfig | None = None) -> F4QTensor:
    config = config or F4QConfig(quant_type="nf4")
    nf4 = F4QTensor.from_high_precision(tensor, config)
    assert torch.all(torch.isfinite(nf4.get_high_precision()))
    return nf4


t = torch.randn([16, 16], dtype=torch.float32, device="cpu")
print("before make_nf4")
q = make_nf4(t)
print("after make_nf4", type(q))
