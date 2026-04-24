import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from pregated_moe.memory import LayerRoute


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
        self.pregated_route = None
        self.gate_tensor_ids = []
        self.debug_enabled = os.getenv("PREGATED_DEBUG", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _debug(self, message: str) -> None:
        if self.debug_enabled:
            print(
                f"[PregatedQwenDebug][layer={self.layer_id}] {message}",
                flush=True,
            )

    def _prepare_route(self, hidden_states: torch.Tensor) -> LayerRoute:
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
        # topk_softmax returns views backed by a reused native workspace.
        # Clone here so the saved route is not overwritten by the next gate call.
        router_mask = router_mask.clone()
        routing_weights_mask = routing_weights_mask.clone()
        active_experts = torch.nonzero(
            router_mask.view(-1, self.num_experts).any(dim=0),
            as_tuple=False,
        ).flatten()
        return LayerRoute(
            router_mask=router_mask,
            routing_weights_mask=routing_weights_mask,
            expert_ids=active_experts.cpu().tolist(),
            router_logits=router_logits,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.pregated_route is None:
            current_route = self._prepare_route(hidden_states)
        else:
            request_id = self.pregated_route.begin_request_if_needed(
                self.layer_id
            )
            if self.pregated_route.is_first_layer(self.layer_id):
                current_route = self._prepare_route(hidden_states)
                self._debug(
                    f"current_route experts={current_route.expert_ids[:16]} total={len(current_route.expert_ids)} request={request_id}"
                )
                self.pregated_route.submit_layer(
                    request_id,
                    self.layer_id,
                    current_route.expert_ids,
                )
            else:
                current_route = self.pregated_route.pop_pending_route(
                    self.layer_id
                )
                if current_route.router_mask.device != hidden_states.device:
                    current_route.router_mask = current_route.router_mask.to(
                        hidden_states.device,
                        non_blocking=True,
                    )
                if (
                    current_route.routing_weights_mask.device
                    != hidden_states.device
                ):
                    current_route.routing_weights_mask = (
                        current_route.routing_weights_mask.to(
                            hidden_states.device,
                            non_blocking=True,
                        )
                    )
                self._debug(
                    f"pending_current_route experts={current_route.expert_ids[:16]} total={len(current_route.expert_ids)} request={request_id}"
                )

            next_module = self.pregated_route.get_next_module(self.layer_id)
            if next_module is not None:
                next_gate_device = self.pregated_route.fetch_gate_tensors(
                    request_id,
                    next_module,
                )
                next_hidden_states = hidden_states
                if next_hidden_states.device != next_gate_device:
                    next_hidden_states = next_hidden_states.to(
                        next_gate_device,
                        non_blocking=True,
                    )
                next_route = next_module._prepare_route(next_hidden_states)
                self._debug(
                    f"next_route layer={next_module.layer_id} experts={next_route.expert_ids[:16]} total={len(next_route.expert_ids)} request={request_id}"
                )
                self.pregated_route.set_pending_route(
                    next_module.layer_id,
                    next_route,
                )
                self.pregated_route.submit_layer(
                    request_id,
                    next_module.layer_id,
                    next_route.expert_ids,
                )

            self.pregated_route.wait_layer_ready(request_id, self.layer_id)
            self._debug(
                f"dispatch experts={current_route.expert_ids[:16]} total={len(current_route.expert_ids)} request={request_id}"
            )

        self.expert_executor.dispatch_local(
            self.layer_id,
            hidden_states,
            current_route.router_mask,
            current_route.routing_weights_mask,
        )
        final_hidden_states = self.expert_executor.wait_dispatch_local()

        if self.pregated_route is not None:
            self.pregated_route.release_layer(request_id, self.layer_id)
            self.pregated_route.finish_request_if_last(
                request_id,
                self.layer_id,
            )

        final_hidden_states = final_hidden_states.view(
            batch_size,
            sequence_length,
            hidden_dim,
        ).to(hidden_states.dtype)
        return final_hidden_states
