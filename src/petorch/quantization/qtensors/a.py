import  os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from torch import Tensor

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
from bitsandbytes.backends.default.ops import *

def _abc():
    t = torch.randn([16, 16], device="cpu")
    q = make_nf4(torch.zeros_like(t))# This will cause seg fault

    for _,ts in q.iter_tensors():
        print(ts.isfinite().all(), ts.abs().max())

_abc()