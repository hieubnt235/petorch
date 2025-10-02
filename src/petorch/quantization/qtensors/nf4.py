__all__ = ["F4QTensor", "F4QConfig", "F4QState"]
from functools import lru_cache
from typing import Literal, override, Self

import torch
from bitsandbytes import functional as F
from pydantic import Field
from torch import Tensor

from petorch import logger
from .base import QTensor, QState, _WrapperState, QConfig


class F4QConfig(QConfig):
    """Config for F4QTensor, the default is already good, skip if you don't know what it is."""

    block_size: int = Field(default=64, gt=0)
    scale_block_size: int = Field(default=256, gt=0)
    quant_type: Literal["fp4", "nf4"] = "nf4"

    scales_dtype: torch.dtype = torch.float32
    """Dtype of scales storage, only float32 (default) has c++ kernel that increase performance for CPU."""


class F4QState(QState[F4QConfig]):
    quant_data: Tensor
    """The quantized of scaled data (ex: data is scaled to range [-1,1]). Hold index of the codebook."""

    quant_scale_res: Tensor
    """Quantized tensor of the data scale residuals (data_scales - data_scales_offset)"""

    residual_scales: Tensor
    """The block-wise scale of the `quant_scale_res` ."""

    data_scales_offset: Tensor
    """The means of the data scales."""

    def _tensor_to(
        self,
        tensor_name: str,
        tensor: Tensor,
        device: str | torch.device | int | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> Tensor:

        # No use dtype
        return tensor.to(
            device=device,
            non_blocking=non_blocking,
            copy=copy,
            memory_format=memory_format,
        )


class F4QTensor(QTensor[F4QState]):
    q_state_cls = F4QState

    @override
    @classmethod
    def from_high_precision(cls, hp_tensor: Tensor, config: F4QConfig) -> Self:
        """Double quantizes the high-precision tensor to 4-bits code-wise format."""

        # 0. Validation
        assert torch.is_floating_point(hp_tensor)
        assert hp_tensor.isfinite().all()
        assert hp_tensor.is_contiguous()

        # 1. Quantize data -> data, data_scales
        # bitsandbytes.backends.default.ops.quantize_4bit
        # Flatten and quantize. Note that this method use CODE internally.

        # TODO: case for all elements in one block is zeros.
        # Problem: When all elements in 1 block is zeros. So that the scale is 0, then the dev return nan in bnb code.
        # Note that this nan tensor will be convert to uint8, and all become zeros so program cannot detect the nan,
        # until it reach C++ code.
        # Solution:
        # 1. Prevalidating
        # 2. Hack (Treat the all-zero blocks case (cause absmax=0) separately to the bnb kernel)
        # 3. Rewrite the kernel
        
        
        
        deq_state: tuple[Tensor, Tensor] = torch.ops.bitsandbytes.quantize_4bit.default(
            hp_tensor,
            config.block_size,
            config.quant_type,
            torch.uint8,
        )
        # shape (flatten_shape, 1), (ceil(flatten_shape/block_size), )
        quant_data, data_scales = deq_state

        assert len(data_scales.shape) == 1
        assert quant_data.device == hp_tensor.device == data_scales.device

        # 2. Quantize data scales residual ->  quant_scale_res, residual_scales, data_scales_offset
        # bitsandbytes.backends.default.ops.quantize_blockwise
        # Quantize the scale residuals
        device = data_scales.device
        scales_dtype = config.scales_dtype

        # Residual can be zero if num_block is 1 ( hp_tensor.numel()<=block_size)
        # so that the offset scale is scalar and is close to zero, that cause segment fault.
        # https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1744
        # I treat this case separately.

        data_scales_offset = data_scales.mean()
        scale_res = data_scales - data_scales_offset

        assert scale_res.shape == data_scales.shape
        deq_state2: tuple[Tensor, Tensor] = (
            torch.ops.bitsandbytes.quantize_blockwise.default(
                scale_res.to(scales_dtype),  # Residual
                cls._get_double_quantize_code(device).to(scales_dtype),
                config.scale_block_size,
            )
        )
        quant_scale_res, residual_scales = deq_state2

        # 3. Store states, no need to store the data_scales for double quantization.
        del data_scales
        del scale_res
        states = F4QState(
            quant_data=quant_data,
            quant_scale_res=quant_scale_res,
            residual_scales=residual_scales.to(scales_dtype),
            data_scales_offset=data_scales_offset.to(scales_dtype),
            config=config,
        )
        states._wrapper_state = _WrapperState.from_data(hp_tensor)
        return cls(_state=states)

    @override
    def _dequantize_high_precision(self) -> Tensor:
        quant_state = self.quant_state
        config = self.config
        scales_dtype = config.scales_dtype
        device = quant_state.residual_scales.device
        # 1. Dequantize the scale residuals -> scale_res
        # bitsandbytes.backends.default.ops dequantize_blockwise
        scale_res: Tensor = torch.ops.bitsandbytes.dequantize_blockwise.default(
            quant_state.quant_scale_res,
            quant_state.residual_scales,
            self._get_double_quantize_code(device).to(scales_dtype),
            config.scale_block_size,
            scales_dtype,
        )

        # 2. Offset the scale residuals -> data_scales
        data_scales = scale_res + self.quant_state.data_scales_offset

        # 3. Dequantize the data -> data (original data)
        # bitsandbytes.backends.default.ops dequantize_blockwise
        data: Tensor = torch.ops.bitsandbytes.dequantize_4bit.default(
            quant_state.quant_data,
            data_scales,
            config.block_size,
            config.quant_type,
            self.shape,
            self.dtype,
        )
        return data

    @classmethod
    @lru_cache(maxsize=None)
    def _get_double_quantize_code(cls, device: torch.device | str | None = None):
        # the default creates 256 values from [-1,1]
        # create_dynamic_map returns a CPU tensor, then move to the device
        tensor = F.create_dynamic_map().to(device=device)
        logger.debug(
            f"Double quantize code created. Shape = {tensor.shape}, device={tensor.device}"
        )
        return tensor
