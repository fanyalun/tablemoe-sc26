import math
from typing import Dict

import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F

from pregated_moe.memory import LayerRoute


def _uses_deepseek_vl2_gate_semantics(config) -> bool:
    return (
        getattr(config, "scoring_func", "softmax") == "sigmoid"
        or getattr(config, "topk_method", "greedy") == "noaux_tc"
    )


class DeepseekMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = getattr(
            config, "routed_scaling_factor", 1.0
        )
        self.norm_topk_prob = getattr(config, "norm_topk_prob", False)
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )

    def forward(self, hidden_states):
        """
        Forward pass for the MoE gate.
        :param hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size).
        :return: Gating logits of shape (batch_size, sequence_length, n_routed_experts).
        """
        # Compute the gating logits
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )
        scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = (
                topk_weight / denominator * self.routed_scaling_factor
            )
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight, None


class DeepseekVL2MoEGateAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts,))
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            )
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len,
                    self.n_group,
                    self.n_routed_experts // self.n_group,
                )
                .reshape(bsz * seq_len, -1)
            )
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(
                bsz * seq_len, -1
            ) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len,
                    self.n_group,
                    self.n_routed_experts // self.n_group,
                )
                .reshape(bsz * seq_len, -1)
            )
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
            _, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(
                f"Unsupported DeepSeek-VL2 topk method: {self.topk_method}"
            )

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = (
                topk_weight / denominator * self.routed_scaling_factor
            )
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        aux_loss = None
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(
                        bsz, seq_len * aux_topk, device=hidden_states.device
                    ),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1),
                    num_classes=self.n_routed_experts,
                )
                ce = mask_ce.float().mean(0)
                pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (pi * fi).sum() * self.alpha

        return topk_idx, topk_weight, aux_loss


class DeepseekMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_expert = config.n_routed_experts

        if self.config.model_type == "deepseek_v2":
            from .modeling_deepseek_v2 import DeepseekV2MLP, MoEGate

            self.mlp_cls = DeepseekV2MLP
            self.gate = (
                DeepseekVL2MoEGateAdapter(config)
                if _uses_deepseek_vl2_gate_semantics(config)
                else MoEGate(config)
            )
        if self.config.model_type == "deepseek_v3":
            from .modeling_deepseek_v3 import DeepseekV3MLP, MoEGate

            self.mlp_cls = DeepseekV3MLP
            self.gate = MoEGate(config)

        self.experts = nn.ModuleList(
            [
                self.mlp_cls(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for i in range(config.n_routed_experts)
            ]
        )

        if not hasattr(self, "gate"):
            self.gate = DeepseekMoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = (
                config.moe_intermediate_size * config.n_shared_experts
            )
            self.shared_experts = self.mlp_cls(
                config=config, intermediate_size=intermediate_size
            )

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Dict[int, int] = None
        self.pregated_route = None
        self.gate_tensor_ids = []

    @nvtx.annotate("DeepSeekPrepare", color="blue")
    def _prepare_route(self, hidden_states):
        gate_output = self.gate(hidden_states)
        if len(gate_output) == 3:
            topk_idx, topk_weight, _ = gate_output
        else:
            topk_idx, topk_weight = gate_output

        router_mask = F.one_hot(
            topk_idx,
            num_classes=self.num_expert,
        ).permute(0, 2, 1)
        routing_weights_mask = (
            topk_weight[:, :, None] * router_mask.permute(0, 2, 1)
        ).permute(0, 2, 1)
        routing_weights_mask = torch.sum(
            routing_weights_mask,
            dim=-1,
        )
        router_mask = torch.any(router_mask, dim=-1)
        active_experts = torch.nonzero(
            router_mask.view(-1, self.num_expert).any(dim=0),
            as_tuple=False,
        ).flatten()
        return LayerRoute(
            router_mask=router_mask,
            routing_weights_mask=routing_weights_mask,
            expert_ids=active_experts.cpu().tolist(),
            topk_idx=topk_idx,
            topk_weight=topk_weight,
        )

    @nvtx.annotate(message="DeepseekMoEBlock", color="blue")
    def forward(self, hidden_states):
        identity = hidden_states
        if self.pregated_route is None:
            current_route = self._prepare_route(hidden_states)
        else:
            request_id = self.pregated_route.begin_request_if_needed(
                self.layer_id
            )
            if self.pregated_route.is_first_layer(self.layer_id):
                current_route = self._prepare_route(hidden_states)
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

        batch_size, sequence_length, hidden_dim = identity.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

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
            batch_size, sequence_length, hidden_dim
        ).to(hidden_states.dtype)
        if self.config.n_shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(
                identity
            )
        return final_hidden_states
