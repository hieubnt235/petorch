from typing import Any

import torch
from torch import nn


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (8, 8)
        self.conv = nn.Conv2d(3, 3, 3, padding="same")
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Fully connected part
        self.fc1 = nn.Linear(8 * 8 * 3, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 100)
        self.ln2 = nn.LayerNorm(100)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        assert x.shape[2:] == self.input_shape

        x = self.relu(self.bn(self.conv(x)))
        x = self.flatten(x)

        x = self.ln1(self.relu(self.fc1(x)))

        x = self.ln2(self.fc2(x))
        return x


class NestedDummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding="same")
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
        self.sub_model = Dummy()
        self.fc = nn.Linear(100, 16)
        self.ln = nn.LayerNorm(16)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = self.relu(self.bn(self.conv(x)))
        x = self.sub_model(x)

        x = self.ln(self.relu(self.fc(x)))
        return x


class DummyV2(nn.Module):
    """
    The simplest stable model that accepts a float tensor of shape [N, 3, 8, 8]
    and contains Embedding, Conv2d, and Linear layers.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same')
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        conv_output_size = 16 * (3 * 8 * 8) * embedding_dim
        self.linear = nn.Linear(conv_output_size, 10)  # Example 10 output classes
        self.ln = nn.LayerNorm(10)

    def forward(self, x: torch.Tensor, emb_idx: torch.Tensor, *args, **kwargs):
        # Input x shape: [batch, channels, height, width] e.g., [2, 3, 8, 8]
        # emb_idx is Long Tensor with shape [2,192] from 0 to vocab_size
        x = self.embedding(emb_idx)  # -> [2, 192, 32]
        x = x.unsqueeze(1)  # -> [2, 1, 192, 32]
        x = self.relu(self.bn(self.conv(x)))
        x = self.flatten(x)
        x = self.linear(x)
        return self.ln(self.relu(x))


class ParamWrapper(nn.Module):

    def __init__(
        self,
        weight: torch.Tensor | nn.Parameter,
        bias: torch.Tensor | nn.Parameter | None = None,
    ):
        super().__init__()
        
        self.weight = (
            nn.Parameter(weight) if isinstance(weight, torch.Tensor) else weight
        )
        if bias:
            self.bias = nn.Parameter(bias) if isinstance(bias, torch.Tensor) else bias
        else:
            self.register_parameter("bias", None)

        assert isinstance(self.weight, nn.Parameter)
        assert isinstance(self.bias, nn.Parameter) or (self.bias is None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

