import torch
from torch import nn

from ..ops import maybe_run_fp16_fused_expert
from .linear_wrapper import OffloadLinearWrapper


class BaseExpertWrapper(nn.Module):
    def __init__(self, expert_module: nn.Module, device: torch.device, act_fn: nn.Module):
        super().__init__()
        self.w1 = OffloadLinearWrapper(expert_module.gate_proj, device)
        self.w3 = OffloadLinearWrapper(expert_module.up_proj, device)
        self.w2 = OffloadLinearWrapper(expert_module.down_proj, device)
        self.act_fn = act_fn

        self.w1_event = torch.cuda.Event()
        self.w2_event = torch.cuda.Event()
        self.w3_event = torch.cuda.Event()
        self.compute_event = torch.cuda.Event()

        self.load = False
        self.free = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.load:
            self.w1_event.wait()
            current_hidden_states = self.act_fn(self.w1(hidden_states))
            self.w3_event.wait()
            current_hidden_states = current_hidden_states * self.w3(hidden_states)
            self.w2_event.wait()
            current_hidden_states = self.w2(current_hidden_states)
            self.load = False
        else:
            current_hidden_states = maybe_run_fp16_fused_expert(self, hidden_states)
            if current_hidden_states is None:
                current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
                current_hidden_states = self.w2(current_hidden_states)

        self.compute_event.record(torch.cuda.current_stream())
        return current_hidden_states
