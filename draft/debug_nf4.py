import os
os.environ["LOG_LEVEL"] = "DEBUG"
import torch
from petorch.quantization.qtensors.nf4 import F4QConfig, F4QTensor, QTensor
from torch.autograd.function import Function
from torch import nn
# from bitsandbytes.backends.default.ops import *
# torch.mm()
# torch.matmul()
# nn.functional.mse_loss()
# torch.sum()
# torch.optim.Adam.zero_grad
import numpy as np

def metrics(ref: torch.Tensor, test: torch.Tensor, name: str):
    diff = ref.float().cpu() - test.float().cpu()
    ref_np, test_np, diff_np = ref.numpy(), test.numpy(), diff.numpy()

    print(f"\n=== {name} vs Original ===")
    print(f"L1 norm:            {torch.norm(diff, p=1).item():.6e}")
    print(f"L2 norm:            {torch.norm(diff, p=2).item():.6e}")
    print(f"Mean Squared Error: {torch.mean(diff**2).item():.6e}")
    print(f"Root MSE:           {torch.sqrt(torch.mean(diff**2)).item():.6e}")
    print(f"Mean Abs Error:     {torch.mean(torch.abs(diff)).item():.6e}")
    print(f"Max Abs Error:      {torch.max(torch.abs(diff)).item():.6e}")

    # relative error
    print(f"Relative L2 error:  {(torch.norm(diff,2)/torch.norm(ref,2)).item():.6e}")

    # cosine similarity
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), test.flatten(), dim=0)
    print(f"Cosine Similarity:  {cos.item():.6f}")

    # correlation (Pearson)
    corr = np.corrcoef(ref_np.flatten(), test_np.flatten())[0,1]
    print(f"Pearson Corr:       {corr:.6f}")

    # SNR
    signal_power = torch.mean(ref**2).item()
    noise_power = torch.mean(diff**2).item()
    snr = 10 * np.log10(signal_power / (noise_power + 1e-12))
    print(f"Signal-to-Noise:    {snr:.2f} dB")


a = torch.randn([256,256],).to(torch.bfloat16)
config = F4QConfig()
quant = F4QTensor.from_high_precision(a, config)
print(isinstance(quant, F4QTensor))
print(isinstance(quant, QTensor))
print(isinstance(quant,torch.Tensor))
for name, ts in quant.quant_tensors.items():
    print(name, ts.device, ts.dtype, ts.numel(), ts.element_size())
    
hp = quant.get_high_precision()
assert hp.dtype == quant.dtype
assert hp.device == quant.device

a = a.float()
hp = hp.float()
# To
quant_to = quant.to(dtype=torch.float, device = "cuda")
hp_to = quant_to.get_high_precision()
assert quant_to.device.type == hp_to.device.type ==  "cuda"

# metrics(a.float(),hp.float(),"nf4")

def similarity(ref: torch.Tensor, t: torch.Tensor):
    dot = torch.dot(ref.view(-1), t.view(-1))
    norm_sq = torch.norm(ref).pow(2) + torch.norm(t).pow(2)

    return 2*dot/norm_sq


atol = 5e-3
rtol = 5e-3
simi = similarity(a, hp)
print(f"{simi=}:{atol=}:{rtol=} - {torch.allclose(simi,torch.tensor(1.0),rtol=rtol,atol=atol)}")

# import matplotlib.pyplot as plt
# strides = 1000
# error = (a-hp)
# abs_err = error.abs()
# print(f"{abs_err.max()=}, {abs_err.min()=}, {abs_err.mean()=}")
# plt.plot(error[::strides].reshape(-1).cpu().numpy(), label = "error")
# plt.plot(a[::strides].reshape(-1).cpu().numpy(), label = "original")
# plt.plot(hp[::strides].reshape(-1).cpu().numpy(), label="dequantized")
# plt.legend()
#
# plt.show()

