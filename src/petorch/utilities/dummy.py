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

    def forward(self, x: torch.Tensor):
        assert x.shape[2:] == self.input_shape, f"{x.shape[:2]}-{self.input_shape}"
        # Apply normalization and activation
        x = self.relu(self.bn(self.conv(x)))
        x = self.flatten(x)
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))  # No activation on the final output
        return x


class NestedDummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding="same")
        self.bn = nn.BatchNorm2d(3)  # <-- ADDED
        self.relu = nn.ReLU()

        self.sub_model = Dummy()  # <-- Uses the stable version

        self.fc = nn.Linear(100, 16)
        self.ln = nn.LayerNorm(16)  # <-- ADDED

    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn(self.conv(x)))
        x = self.sub_model(x)
        x = self.ln(self.fc(x))
        return x
