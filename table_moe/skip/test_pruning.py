import unittest

import torch
from torch import nn

from table_moe.models.deepseek_vl2.deepseek_layers import DeepSeekMoeWrapperSkipBaseline
from table_moe.models.deepseek_vl2.deepseek_layers import DeepSeekMoeWrapperSkipOffload
from table_moe.models.qwen3_vl_moe.custom_layers import QwenMoeWrapperSkipBaseline, QwenMoeWrapperSkipOffload
from table_moe.skip import build_fixed_keep_mask, renormalize_surviving_weights


class DummyQwenConfig:
    hidden_size = 3
    num_experts = 8
    num_experts_per_tok = 8


class DummyDeepSeekConfig:
    hidden_size = 2
    n_routed_experts = 6
    n_shared_experts = 1
    num_experts_per_tok = 6
    routed_scaling_factor = 2.0


class FixedLinear(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        rows = hidden_states.shape[0]
        return self.logits[:rows].to(device=hidden_states.device, dtype=hidden_states.dtype)


class RecordingQwenExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_hidden_states = None
        self.last_routing_weights = None
        self.last_router_indices = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        router_indices: torch.Tensor,
    ) -> torch.Tensor:
        self.last_hidden_states = hidden_states.detach().clone()
        self.last_routing_weights = routing_weights.detach().clone()
        self.last_router_indices = router_indices.detach().clone()
        return hidden_states


class FixedDeepSeekGate(nn.Module):
    def __init__(self, selected_experts: torch.Tensor, routing_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("selected_experts", selected_experts)
        self.register_buffer("routing_weights", routing_weights)

    def forward(self, hidden_states: torch.Tensor):
        rows = hidden_states.shape[0] * hidden_states.shape[1]
        return (
            self.selected_experts[:rows].to(hidden_states.device),
            self.routing_weights[:rows].to(device=hidden_states.device, dtype=hidden_states.dtype),
            None,
        )


class ZeroSharedExperts(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


class ConstantSharedExperts(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.full_like(hidden_states, self.scale)


class ConstantExpert(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.full_like(hidden_states, self.scale)


class FakeExpertCache:
    def __init__(self, experts: dict[tuple[int, int], nn.Module]):
        self.experts = experts
        self.requested_uids = []
        self.prefetch_uids = []

    def load_experts(self, *uids, unordered=False, uids_to_prefetch=None):
        self.requested_uids = list(uids)
        self.prefetch_uids = list(uids_to_prefetch or [])
        return [(uid, self.experts[uid]) for uid in uids]

    def check(self, layer_idx, expert_ids):
        return [(layer_idx, expert_id) for expert_id in expert_ids]


class SkipPruningTest(unittest.TestCase):
    def test_build_fixed_keep_mask(self):
        selected_experts = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        keep_mask = build_fixed_keep_mask(selected_experts, 2)
        expected = torch.tensor([[True, True, False, False], [True, True, False, False]])
        self.assertTrue(torch.equal(keep_mask, expected))

    def test_renormalize_to_one(self):
        routing_weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
        keep_mask = torch.tensor([[True, True, False, False]])
        effective_weights = renormalize_surviving_weights(routing_weights, keep_mask)
        expected = torch.tensor([[4.0 / 7.0, 3.0 / 7.0, 0.0, 0.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(effective_weights, expected, atol=1e-6))

    def test_qwen_skip_wrapper_prefill_uses_fixed_top5(self):
        logits = torch.tensor(
            [
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                [1.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
            ],
            dtype=torch.float32,
        )
        gate = FixedLinear(logits)
        experts = RecordingQwenExperts()
        wrapper = QwenMoeWrapperSkipBaseline(
            text_config=DummyQwenConfig(),
            layer_id=0,
            gate=gate,
            experts=experts,
            skip_keep_k=5,
            decode_skip_keep_k=4,
        )

        hidden_states = torch.randn(1, 2, DummyQwenConfig.hidden_size, dtype=torch.float32)
        output = wrapper(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(experts.last_router_indices.shape, (2, 8))
        nonzero_counts = experts.last_routing_weights.ne(0).sum(dim=-1)
        self.assertTrue(torch.equal(nonzero_counts, torch.tensor([5, 5])))
        self.assertTrue(
            torch.allclose(
                experts.last_routing_weights.sum(dim=-1),
                torch.ones(2, dtype=experts.last_routing_weights.dtype),
                atol=1e-6,
            )
        )

    def test_qwen_skip_wrapper_decode_uses_fixed_top4(self):
        logits = torch.tensor([[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
        gate = FixedLinear(logits)
        experts = RecordingQwenExperts()
        wrapper = QwenMoeWrapperSkipBaseline(
            text_config=DummyQwenConfig(),
            layer_id=0,
            gate=gate,
            experts=experts,
            skip_keep_k=5,
            decode_skip_keep_k=4,
        )

        hidden_states = torch.randn(1, 1, DummyQwenConfig.hidden_size, dtype=torch.float32)
        output = wrapper(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        nonzero_counts = experts.last_routing_weights.ne(0).sum(dim=-1)
        self.assertTrue(torch.equal(nonzero_counts, torch.tensor([4])))
        self.assertTrue(
            torch.allclose(
                experts.last_routing_weights.sum(dim=-1),
                torch.ones(1, dtype=experts.last_routing_weights.dtype),
                atol=1e-6,
            )
        )

    def test_deepseek_skip_wrapper_prefill_uses_fixed_top5(self):
        selected_experts = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        )
        routing_weights = torch.tensor(
            [
                [0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
                [0.6, 0.3, 0.2, 0.1, 0.05, 0.02],
            ],
            dtype=torch.float32,
        )
        gate = FixedDeepSeekGate(selected_experts, routing_weights)
        experts = nn.ModuleList([ConstantExpert(i + 1) for i in range(6)])
        wrapper = DeepSeekMoeWrapperSkipBaseline(
            lang_config=DummyDeepSeekConfig(),
            layer_id=1,
            gate=gate,
            shared_experts=ZeroSharedExperts(),
            expert_cache=None,
            experts=experts,
            skip_keep_k=5,
            decode_skip_keep_k=4,
        )

        hidden_states = torch.randn(1, 2, DummyDeepSeekConfig.hidden_size, dtype=torch.float32)
        output = wrapper(hidden_states)

        kept_weights = routing_weights[:, :5]
        expected_weights = kept_weights * (
            routing_weights.sum(dim=-1, keepdim=True) / kept_weights.sum(dim=-1, keepdim=True)
        )
        expected_scales = (expected_weights * torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)).sum(dim=-1)
        expected_output = torch.stack(
            [torch.full((DummyDeepSeekConfig.hidden_size,), scale.item(), dtype=torch.float32) for scale in expected_scales],
            dim=0,
        ).reshape(1, 2, DummyDeepSeekConfig.hidden_size)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_deepseek_skip_wrapper_decode_uses_fixed_top4(self):
        selected_experts = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
        routing_weights = torch.tensor([[0.5, 0.4, 0.3, 0.2, 0.1, 0.05]], dtype=torch.float32)
        gate = FixedDeepSeekGate(selected_experts, routing_weights)
        experts = nn.ModuleList([ConstantExpert(i + 1) for i in range(6)])
        wrapper = DeepSeekMoeWrapperSkipBaseline(
            lang_config=DummyDeepSeekConfig(),
            layer_id=1,
            gate=gate,
            shared_experts=ZeroSharedExperts(),
            expert_cache=None,
            experts=experts,
            skip_keep_k=5,
            decode_skip_keep_k=4,
        )

        hidden_states = torch.randn(1, 1, DummyDeepSeekConfig.hidden_size, dtype=torch.float32)
        output = wrapper(hidden_states)

        kept_weights = routing_weights[:, :4]
        expected_weights = kept_weights * (
            routing_weights.sum(dim=-1, keepdim=True) / kept_weights.sum(dim=-1, keepdim=True)
        )
        expected_scale = (expected_weights * torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)).sum()
        expected_output = torch.full_like(output, expected_scale.item())
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_qwen_skip_offload_only_loads_surviving_experts(self):
        logits = torch.tensor(
            [
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                [1.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
            ],
            dtype=torch.float32,
        )
        gate = FixedLinear(logits)
        cache = FakeExpertCache({(0, idx): ConstantExpert(idx + 1) for idx in range(8)})
        wrapper = QwenMoeWrapperSkipOffload(
            text_config=DummyQwenConfig(),
            layer_id=0,
            gate=gate,
            expert_cache=cache,
            skip_keep_k=5,
            decode_skip_keep_k=4,
        )

        hidden_states = torch.randn(1, 2, DummyQwenConfig.hidden_size, dtype=torch.float32)
        output = wrapper(hidden_states)

        routing_weights = torch.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, DummyQwenConfig.num_experts_per_tok, dim=-1)
        keep_mask = build_fixed_keep_mask(topk_indices, 5)
        expected_weights = renormalize_surviving_weights(topk_weights, keep_mask)
        expected_scale = (expected_weights * (topk_indices.to(torch.float32) + 1.0)).sum(dim=-1)
        expected_output = expected_scale[:, None].expand(-1, DummyQwenConfig.hidden_size).reshape_as(output)

        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))
        self.assertEqual(cache.requested_uids, [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])

    def test_deepseek_skip_offload_only_loads_surviving_experts(self):
        selected_experts = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        )
        routing_weights = torch.tensor(
            [
                [0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
                [0.6, 0.3, 0.2, 0.1, 0.05, 0.02],
            ],
            dtype=torch.float32,
        )
        gate = FixedDeepSeekGate(selected_experts, routing_weights)
        cache = FakeExpertCache({(1, idx): ConstantExpert(idx + 1) for idx in range(6)})
        wrapper = DeepSeekMoeWrapperSkipOffload(
            lang_config=DummyDeepSeekConfig(),
            layer_id=1,
            gate=gate,
            shared_experts=ConstantSharedExperts(10.0),
            expert_cache=cache,
            skip_keep_k=5,
            decode_skip_keep_k=4,
        )

        hidden_states = torch.randn(1, 2, DummyDeepSeekConfig.hidden_size, dtype=torch.float32)
        output = wrapper(hidden_states)

        keep_mask = build_fixed_keep_mask(selected_experts, 5)
        expected_weights = renormalize_surviving_weights(
            routing_weights,
            keep_mask,
            target_row_sum=routing_weights.sum(dim=-1, keepdim=True),
        )
        expected_scale = (expected_weights[:, :5] * torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)).sum(dim=-1)
        expected_output = torch.stack(
            [torch.full((DummyDeepSeekConfig.hidden_size,), 10.0 + scale.item(), dtype=torch.float32) for scale in expected_scale],
            dim=0,
        ).reshape_as(output)

        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))
        self.assertEqual(cache.requested_uids, [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)])


if __name__ == "__main__":
    unittest.main()
