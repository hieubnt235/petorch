import torch

from .core import BaseAdaptedLayer

# from torchao.quantization import Int8WeightOnlyConfig, quantize_
# from torchao.dtypes import NF4Tensor
# from bitsandbytes.nn import LinearNF4
# from torch.ao.nn.quantized.modules import Conv2d
# torch.nn.Conv2d
# torch.library.register_kernel()
# from ctypes import CDLL


class AdaptedLayer(BaseAdaptedLayer):

    def _update_base_layer(
        self, delta_weight: torch.Tensor, delta_bias: None | torch.Tensor
    ) -> None:
        pass  # TODO
