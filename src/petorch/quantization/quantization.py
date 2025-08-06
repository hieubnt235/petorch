import types
from typing import cast

import torch
from accelerate import init_empty_weights
from diffusers import ModelMixin
from safetensors import safe_open
from torch import nn
from torchao import quantize_
from torchao.core.config import AOBaseConfig
from torchao.dtypes import NF4Tensor, to_nf4
from torchao.dtypes.nf4tensor import linear_nf4
from torchao.quantization.quant_api import _is_linear
from tqdm import tqdm
from transformers.utils.hub import cached_files
from torchao.quantization.quant_api import ap


def get_weight_file_path(path_or_repo_id:str, filenames:list[str]|str, subfolder, **kwargs)-> str :
    for wn in weight_names:
        files = cached_files(path_or_repo_id,filenames,subfolder=subfolder,**kwargs)
        if files:
            return files[0]
        else:
            raise ValueError(f"No files {path_or_repo_id}-{filenames}-{subfolder}")
        
def quantize_filter_fn(module: nn.Module,module_name: str)->bool:
    # Make sure the weight is only quantized once
    is_linear = _is_linear(module, module_name)
    passed = is_linear and (not isinstance(module.weight, NF4Tensor))
    if passed:
        print(f"Quantize module {module.__class__.__name__}, Shape = {module.weight.shape}")
    return passed


def quantize(
    model: torch.nn.Module | ModelMixin,
    weight_path: str,
    *,
    config: AOBaseConfig ,
    compute_dtype: torch.dtype = None,
    device: str = "cpu",
):
    """
    
    Args:
        model:
        weight_path:
        config:
        compute_dtype:
        device:

    Returns:

    """
    dtype = compute_dtype or next(model.parameters()).dtype
    
    with safe_open(weight_path, framework="pt", device="cpu") as f:
        f = cast(safe_open, f)

        # Because of the change in model architect of diffuser's library, this happened, see `ModelMixin.from_pretrain`.
        keys = {k: k for k in f.keys()}

        if isinstance(model, ModelMixin):
            model._fix_state_dict_keys_on_load(keys)

        for model_key, ckpt_key in tqdm(keys.items(), desc="Quantizing"):
            if "." in model_key:
                module_name, param_name = model_key.rsplit(".", 1)
                module = model.get_submodule(module_name)
            else:
                module = model
                param_name = model_key
            module.register_parameter(
                param_name,
                nn.Parameter(
                    t := f.get_tensor(ckpt_key).to(dtype=dtype, device=device)
                ),
            )
            # print(f"Add params `{model_key}` --- {t.shape}")
            quantize_(module, nf4config, filter_fn=quantize_filter_fn)

if __name__=="__main__":
    from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME
    from diffusers import AutoencoderKL, ModelMixin
    from dataclasses import dataclass
    from torchao.quantization import register_quantize_module_handler
    
    model_id = "stabilityai/stable-diffusion-2-1"

    weight_names = [SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME]
    
    vae_subfolder = "vae"
    vae_config = AutoencoderKL.load_config(model_id, subfolder=vae_subfolder, # variant='fp16'
    )
    
    @dataclass
    class NF4Config(AOBaseConfig):
        block_size: int = 64
        scaler_block_size: int = 256
    
    def linear_module_repr(module: nn.Linear):
        return f"in_features={module.weight.shape[1]}, out_features={module.weight.shape[0]}, weight={module.weight}, dtype={module.weight.dtype}"
    
    # For using with `quantize_` api
    @register_quantize_module_handler(NF4Config)
    def _nf4_weight_only_transform(module: torch.nn.Module, config: NF4Config, ) -> torch.nn.Module:
        new_weight = to_nf4(module.weight, config.block_size, config.scaler_block_size)
        module.weight = nn.Parameter(new_weight, requires_grad=False)  # Freeze
        module.extra_repr = types.MethodType(linear_module_repr, module)
        return module
    
    nf4config = NF4Config(block_size=16, scaler_block_size=16)

    with init_empty_weights():
        vae = AutoencoderKL.from_config(vae_config)
        vae = cast(AutoencoderKL, vae).to(dtype=torch.bfloat16)

