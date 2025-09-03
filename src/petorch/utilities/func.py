import base64
import itertools
from typing import Any, Iterator, cast

import torch
from torch import nn


def get_model_size_in_bytes(model: nn.Module, ignore_embeddings=False):

    def flat_size(tensor):
        if hasattr(tensor, "__tensor_flatten__"):
            size = 0
            # 0th element is a list of attributes that
            # hold tensors
            for attr_name in tensor.__tensor_flatten__()[0]:
                sub_tensor = getattr(tensor, attr_name)
                size += flat_size(sub_tensor)
            return size
        else:
            return tensor.numel() * tensor.element_size()

    def loop_params_non_recurse(m: nn.Module):
        size = 0
        if not (isinstance(m, torch.nn.Embedding) and ignore_embeddings):
            for p in itertools.chain(
                m.buffers(recurse=False),
                cast(Iterator[torch.Tensor], m.parameters(recurse=False)),
            ):
                size += flat_size(p)
        return size

    model_size = 0
    for name, sub_m in model.named_modules():
        model_size += loop_params_non_recurse(sub_m)

    return model_size


def get_module_num_parameters(model: nn.Module) -> tuple[int, int]:
    """

    Args:
        model:

    Returns:
        Tuple of trainable and non-trainable parameters, (train_params, non_train_params).
    """
    train_params = 0
    non_train_params = 0
    for param in model.parameters():
        n = param.numel()
        if param.requires_grad:
            train_params += n
        else:
            non_train_params += n
    return train_params, non_train_params


def freeze_module(module: nn.Module) -> int:
    """
    Freeze module in place.
    Args:
        module:

    Returns:
        Number of parameters were frozen by this method.
    """
    n = 0
    for param in module.parameters():
        if param.requires_grad:
            param.requires_grad_(False)
            n += param.numel()
    return n


def b64encode(x: str) -> str:
    """Encode directory string into base64-safe string."""
    return base64.urlsafe_b64encode(x.encode()).decode()


def b64decode(y: str) -> str:
    """Decode base64-safe string back into original string."""
    return base64.urlsafe_b64decode(y.encode()).decode()


def fake_use(*args: Any, **kwargs: Any) -> Any:
    return args, kwargs
