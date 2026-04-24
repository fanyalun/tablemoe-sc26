import torch
from torch import nn


class OffloadLinearWrapper(nn.Module):
    def __init__(self, linear_module: nn.Linear, device: torch.device):
        super().__init__()
        self.linear_module = linear_module
        self.storage = self._setup_storage(linear_module, device)

    def _setup_storage(self, layer: nn.Linear, device: torch.device):
        if not layer.weight.is_contiguous():
            with torch.no_grad():
                layer.weight.data = layer.weight.data.contiguous()
        if layer.weight.device.type != device.type:
            with torch.no_grad():
                layer.weight.data = layer.weight.data.to(device)

        if hasattr(layer.weight, "untyped_storage"):
            return layer.weight.untyped_storage()
        return layer.weight.storage()

    def forward(self, x):
        return self.linear_module(x)
