from torch import nn


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

def freeze_module(module: nn.Module)->int:
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
            n+=param.numel()
    return n