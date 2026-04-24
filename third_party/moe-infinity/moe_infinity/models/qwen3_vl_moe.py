import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3VLMoeExpertMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False
        )

    def forward(self, x):
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class SyncQwen3VLMoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)

        self.gate = nn.Linear(
            config.hidden_size, config.num_experts, bias=False
        )
        self.experts = nn.ModuleList(
            [
                Qwen3VLMoeExpertMLP(
                    config.hidden_size,
                    config.moe_intermediate_size,
                )
                for _ in range(self.num_experts)
            ]
        )

        self.lib = None
        self.expert_executor = None
        self.layer_id = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        router_mask, routing_weights_mask = self.lib.topk_softmax(
            router_logits
        )
        target_device = hidden_states.device
        if router_mask.device != target_device:
            router_mask = router_mask.to(target_device, non_blocking=True)
        if routing_weights_mask.device != target_device:
            routing_weights_mask = routing_weights_mask.to(
                target_device, non_blocking=True
            )

        self.expert_executor.dispatch_local(
            self.layer_id,
            hidden_states,
            router_mask,
            routing_weights_mask,
        )
        final_hidden_states = self.expert_executor.wait_dispatch_local()

        final_hidden_states = final_hidden_states.view(
            batch_size,
            sequence_length,
            hidden_dim,
        ).to(hidden_states.dtype)
        return final_hidden_states
