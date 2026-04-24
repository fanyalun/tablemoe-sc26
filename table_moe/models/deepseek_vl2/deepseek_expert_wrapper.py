import torch
from torch import nn

from ...offload.expert_wrapper import BaseExpertWrapper


class DeepSeekExpertWrapper(BaseExpertWrapper):
    def __init__(self, expert_module: nn.Module, device: torch.device):
        super().__init__(expert_module=expert_module, device=device, act_fn=nn.SiLU())
