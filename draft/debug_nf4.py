import torch

from petorch.quantization.qtensors.nf4 import F4QConfig, F4QTensor, QTensor
a = torch.rand([256,256],).float()

config = F4QConfig()
quant = F4QTensor.from_high_precision(a, config)
print(isinstance(quant, F4QTensor))
print(isinstance(quant, QTensor))
print(isinstance(quant,torch.Tensor))
for name, ts in quant.quant_tensors.items():
    print(name, ts.device, ts.dtype, ts.numel(), ts.element_size())

# import torch
# from bitsandbytes.nn import Params4bit
# from bitsandbytes.backends.cpu.ops import *
# from torch import nn
#
# dim = 256
#
# m_bnb = nn.Linear(dim, dim, dtype=torch.float16)
# m_bnb.weight = Params4bit(
#     m_bnb.weight.data,
#     compress_statistics=True,
#     quant_type="nf4",
#     quant_storage=torch.uint8,
# ).to(device="cpu")
