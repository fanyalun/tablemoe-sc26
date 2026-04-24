import unittest
from unittest import mock

import torch

from table_moe.cache_engine.config import CacheConfig
from table_moe.cache_engine.search import VectorSearchEngine
from table_moe.cache_engine import search as search_module


class DummyManager:
    def __init__(self):
        self.device = torch.device("cpu")
        self.keys_buffer = torch.zeros((2, 2, 2, 2), dtype=torch.float32, device=self.device)
        self.raw_keys_buffer = torch.zeros((2, 2, 2, 2), dtype=torch.float32, device=self.device)
        self.pca = {
            "vision": {
                "proj": torch.eye(2, dtype=torch.float32, device=self.device),
                "bias": torch.zeros(2, dtype=torch.float32, device=self.device),
            },
            "text": {
                "proj": torch.eye(2, dtype=torch.float32, device=self.device),
                "bias": torch.zeros(2, dtype=torch.float32, device=self.device),
            },
        }
        self.online_expert_ids = None


class OfflineDotThresholdTest(unittest.TestCase):
    def setUp(self):
        self.manager = DummyManager()
        self.engine = VectorSearchEngine(self.manager)
        self.old_threshold = float(CacheConfig.OFFLINE_DOT_THRESHOLD)
        CacheConfig.OFFLINE_DOT_THRESHOLD = 0.0

        self.manager.raw_keys_buffer[0, 0] = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0]],
            dtype=torch.float32,
        )
        self.manager.raw_keys_buffer[1, 0] = torch.tensor(
            [[0.0, 0.0], [-1.0, 0.0]],
            dtype=torch.float32,
        )

    def tearDown(self):
        CacheConfig.OFFLINE_DOT_THRESHOLD = self.old_threshold

    def test_search_prefill_offline_dot_requires_positive_max_dot(self):
        hidden_states = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0]],
            dtype=torch.float32,
        )
        selected_experts = torch.tensor([[0], [1]], dtype=torch.long)
        is_vision_mask = torch.tensor([True, True], dtype=torch.bool)
        reuse_mask = torch.tensor([[True], [True]], dtype=torch.bool)

        off_hit_mask, off_cluster_indices = self.engine.search_prefill_offline_dot(
            layer_idx=0,
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            is_vision_mask=is_vision_mask,
            reuse_mask=reuse_mask,
        )

        self.assertTrue(bool(off_hit_mask[0, 0].item()))
        self.assertEqual(int(off_cluster_indices[0, 0].item()), 0)
        self.assertFalse(bool(off_hit_mask[1, 0].item()))

    def test_search_decode_offline_dot_requires_positive_max_dot(self):
        hidden_states = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0]],
            dtype=torch.float32,
        )
        selected_experts = torch.tensor([[0], [1]], dtype=torch.long)
        is_vision_mask = torch.tensor([True, True], dtype=torch.bool)
        reuse_mask = torch.tensor([[True], [True]], dtype=torch.bool)

        with mock.patch.object(search_module, "HAS_TRITON", False):
            off_hit_mask, off_cluster_indices = self.engine.search_decode_offline_dot(
                layer_idx=0,
                hidden_states=hidden_states,
                selected_experts=selected_experts,
                is_vision_mask=is_vision_mask,
                reuse_mask=reuse_mask,
            )

        self.assertTrue(bool(off_hit_mask[0, 0].item()))
        self.assertEqual(int(off_cluster_indices[0, 0].item()), 0)
        self.assertFalse(bool(off_hit_mask[1, 0].item()))


if __name__ == "__main__":
    unittest.main()
