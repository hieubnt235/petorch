from typing import cast, Literal

import torch
import torch.nn.functional as F
from pydantic import model_validator
from torch import nn

from petorch.utilities import ParamWrapper
from .base import BaseLoraAdapter, LoraAdapterConfig


class LoraEmbedding(BaseLoraAdapter):
    base_layer_class = nn.Embedding

    @property
    def base_layer(self) -> nn.Embedding:
        return cast(nn.Embedding, super().base_layer)

    def _init_lora_layers(self) -> None:
        bl = self.base_layer
        self.lora_A = ParamWrapper(weight=torch.empty([bl.num_embeddings, self.rank]))
        self.lora_B = ParamWrapper(weight=torch.empty([self.rank, bl.embedding_dim]))
        assert (getattr(self.lora_A, "bias"), getattr(self.lora_B, "bias")) == (
            None,
            None,
        )
        self.config.bias=False # Todo: this step should be in init method for more clean in purpose of this method.
        assert not self.is_bias

    def reset_parameters(self):
        nn.init.zeros_(self.lora_A.weight)
        nn.init.normal_(self.lora_B.weight)
        

    def get_delta_weight(self) -> torch.Tensor:
        # The weight shape of nn.Embedding is (num_embeddings, embedding_dim) and forward is
        # return F.embedding(input, self.weight,...)

        delta_weight = (
            torch.einsum(
                "n r, r d -> n d",
                self.lora_A.weight,
                self.lora_B.weight,
            )
            * self.scaling
        )
        assert delta_weight.shape == self.base_layer.weight.shape
        return delta_weight

    def get_delta(self, batch_input: torch.Tensor) -> torch.tensor:
        bl = self.base_layer
        # noinspection PyTypeChecker
        # return (
        #     F.embedding(
        #         batch_input,
        #         bl.get_delta_weight(),
        #         bl.padding_idx,
        #         bl.max_norm,
        #         bl.norm_type,
        #         bl.scale_grad_by_freq,
        #         bl.sparse,
        #     )
        # )

        # This method is much faster (x10) than the above method, sees the notebook.
        return (
            F.embedding(
                batch_input,
                self.lora_A.weight,
                bl.padding_idx,
                bl.max_norm,
                bl.norm_type,
                bl.scale_grad_by_freq,
                bl.sparse,
            )
            @ self.lora_B.weight
        ) * self.scaling
