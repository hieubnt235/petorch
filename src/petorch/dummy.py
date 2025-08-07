import torch
from torch import nn


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (8, 8)
        self.conv = nn.Conv2d(3, 3, 3, padding="same")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 3, 256)
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x: torch.Tensor):
        assert x.shape[2:] == self.input_shape, f"{x.shape[:2]}-{self.input_shape}"
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class NestedDummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (8, 8)
        self.conv = nn.Conv2d(3, 3, 3, padding="same")
        self.sub_model = Dummy()
        self.fc = nn.Linear(100, 16)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.sub_model(x)
        x = self.fc(x)
        return x
