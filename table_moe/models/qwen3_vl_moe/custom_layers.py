import torch
from torch import nn

from ...ops import qwen_topk_softmax
from ...skip import (
    build_fixed_keep_mask,
    renormalize_surviving_weights,
)


class QwenMoeWrapperBaseline(nn.Module):
    def __init__(
        self,
        text_config,
        layer_id: int,
        gate: nn.Linear,
        expert_cache,
        layers=None,
        global_config=None,
    ):
        super().__init__()
        self.hidden_dim = text_config.hidden_size
        self.num_experts = text_config.num_experts
        self.top_k = text_config.num_experts_per_tok
        self.layer_id = layer_id
        self.gate = gate
        self.experts = expert_cache
        self.layers = list(layers) if layers is not None else None
        self.num_layers = len(self.layers) if self.layers is not None else 0

    def _get_next_layer_prefetch(self, hidden_states_flat, seq_length):
        """
        计算下一层需要预取的专家 ID。
        区分 Prefill 和 Decode 阶段，并根据 offload_config 截断。
        """
        uids_next = []
        next_layer_idx = self.layer_id + 1
        
        # 懒加载配置，避免循环引用
        from .offload_config import get_offload_config
        config = get_offload_config()
        if config.get('prefetch',True) is False:
            return uids_next
        # === Case 1: Prefill 阶段 (SeqLen > 1) ===
        # 策略：激进预取，检查下一层所有专家，加载其中已 Offload 的，直到填满 limit
        if seq_length > 1 and next_layer_idx < self.num_layers:
            # 检查下一层所有专家中，哪些在 Offload 状态
            candidates = self.experts.check(next_layer_idx, list(range(self.num_experts)))
            
            limit = config.get('prefetch_limit_prefill', 20)
            uids_next = candidates[:limit]
            return uids_next

        # === Case 2: Decode 阶段 (SeqLen == 1) ===
        # 策略：基于 Gate 预测 TopK，仅预取最可能的专家
        if self.layers is not None and next_layer_idx < self.num_layers:
            limit = config.get('prefetch_limit_decode', 4)
            if limit <= 0:
                return uids_next
            next_layer = self.layers[next_layer_idx]
            next_moe = getattr(next_layer, "mlp", None)
            
            if next_moe is not None and hasattr(next_moe, "gate"):
                with torch.no_grad():
                    # 预测下一层 Gate 输出
                    next_logits = next_moe.gate(hidden_states_flat).float()
                    _, next_selected = torch.topk(next_logits, self.top_k, dim=-1)
                    next_selected_unique = torch.unique(next_selected).tolist()
                    
                    if next_selected_unique:
                        # 检查这些专家是否 Offload
                        candidates = self.experts.check(next_layer_idx, next_selected_unique)
                        
                        limit = config.get('prefetch_limit_decode', 4)
                        uids_next = candidates[:limit]
                        
        return uids_next

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # # 启动 MoE 层计时
        # collector = get_perf_stats()
        # moe_timer = CudaTimer(enabled=collector.enabled)
        # moe_timer.__enter__()
        
        batch_size, seq_length, hidden_dim = hidden_states.shape
        bs = batch_size * seq_length        
        hidden_states_flat = hidden_states.reshape(bs, hidden_dim)

        # -------- 1. 本层路由 (Ground Truth) --------
        router_logits = self.gate(hidden_states_flat)
        routing_weights, selected_experts = qwen_topk_softmax(
            router_logits,
            self.top_k,
            normalize_topk=True,
        )
        routing_weights = routing_weights.to(router_logits.dtype)

        # [Optimized] 1. 展平并排序所有选中的专家 ID
        flat_experts = selected_experts.flatten()
        sort_idx = torch.argsort(flat_experts)
        sorted_experts = flat_experts[sort_idx]
        
        # [Optimized] 2. 生成对应的行列索引 (Row=TokenIdx, Col=SlotIdx)
        rows = torch.arange(bs, device=hidden_states.device).repeat_interleave(self.top_k)
        cols = torch.arange(self.top_k, device=hidden_states.device).repeat(bs)
        
        sorted_rows = rows[sort_idx]
        sorted_cols = cols[sort_idx]
        
        # [Optimized] 3. 统计每个专家分配到的 Token 数量，并提取去重后的专家列表
        unique_e_tensor, counts_tensor = torch.unique_consecutive(sorted_experts, return_counts=True)
        active_experts = unique_e_tensor.tolist()
        
        # [Optimized] 4. 构建索引映射表: expert_id -> (token_indices, slot_indices)
        expert_to_idx_map = {}
        curr_offset = 0
        counts_list = counts_tensor.tolist()
        
        for e_id, cnt in zip(active_experts, counts_list):
            expert_to_idx_map[e_id] = (
                sorted_rows[curr_offset : curr_offset + cnt],
                sorted_cols[curr_offset : curr_offset + cnt]
            )
            curr_offset += cnt
    
        # -------- 2. 预测下一层预取 --------
        uids_to_prefetch = self._get_next_layer_prefetch(hidden_states_flat, seq_length)

        # # 启动专家计算计时
        # expert_timer = CudaTimer(enabled=collector.enabled)
        # expert_timer.__enter__()

        # -------- 3. 加载专家 --------
        expert_iterator = self.experts.load_experts(
            *((self.layer_id, e) for e in active_experts), unordered=True, uids_to_prefetch=uids_to_prefetch
        )
        
        # # 统计激活专家和计算专家数量
        # if collector.enabled:
        #     phase = "prefill" if seq_length > 1 else "decode"
        #     # Active Experts 和 Computed Experts 在 baseline 中相同（没有缓存，都是去重后的专家数）
        #     collector.get_stats("baseline", phase, self.layer_id).add_active_experts(len(active_experts))
        #     collector.get_stats("baseline", phase, self.layer_id).add_computed_experts(len(active_experts))

        final_hidden_states = torch.zeros(
            (bs, hidden_dim), dtype=hidden_states_flat.dtype, device=hidden_states_flat.device,
        )
        
        for (_layer_index, expert_idx), expert_layer in expert_iterator:
            # [Optimized] 直接查表获取索引，无需在循环中做掩码计算
            if expert_idx not in expert_to_idx_map: continue
            
            token_idx, slot_idx = expert_to_idx_map[expert_idx]
            
            current_state = hidden_states_flat.index_select(0, token_idx)
            current_weight = routing_weights[token_idx, slot_idx].unsqueeze(-1)

            current_hidden_states = expert_layer(hidden_states=current_state) * current_weight
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        
        # # 结束专家计算计时
        # expert_timer.__exit__()
        # if collector.enabled:
        #     expert_compute_time = expert_timer.elapsed_ms()
        #     phase = "prefill" if seq_length > 1 else "decode"
        #     collector.get_stats("baseline", phase, self.layer_id).add_expert_compute(expert_compute_time)
        
        # # 结束 MoE 层计时
        # moe_timer.__exit__()
        # if collector.enabled:
        #     moe_layer_time = moe_timer.elapsed_ms()
        #     phase = "prefill" if seq_length > 1 else "decode"
        #     collector.get_stats("baseline", phase, self.layer_id).add_moe_layer(moe_layer_time)

        return final_hidden_states.reshape(batch_size, seq_length, hidden_dim)

class QwenMoeWrapperSkipOffload(QwenMoeWrapperBaseline):
    def __init__(
        self,
        *args,
        skip_keep_k: int = 5,
        decode_skip_keep_k: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.skip_keep_k = skip_keep_k
        self.decode_skip_keep_k = skip_keep_k if decode_skip_keep_k is None else decode_skip_keep_k

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_dim = hidden_states.shape
        bs = batch_size * seq_length
        hidden_states_flat = hidden_states.reshape(bs, hidden_dim)

        router_logits = self.gate(hidden_states_flat)
        routing_weights, selected_experts = qwen_topk_softmax(
            router_logits,
            self.top_k,
            normalize_topk=False,
        )
        keep_k = self.skip_keep_k if seq_length > 1 else self.decode_skip_keep_k
        keep_mask = build_fixed_keep_mask(selected_experts, keep_k)
        routing_weights = renormalize_surviving_weights(routing_weights, keep_mask).to(router_logits.dtype)

        rows, cols = torch.where(keep_mask)
        final_hidden_states = torch.zeros(
            (bs, hidden_dim), dtype=hidden_states_flat.dtype, device=hidden_states_flat.device
        )
        if rows.numel() == 0:
            return final_hidden_states.reshape(batch_size, seq_length, hidden_dim)

        flat_experts = selected_experts[rows, cols]
        sort_idx = torch.argsort(flat_experts)
        sorted_experts = flat_experts[sort_idx]
        sorted_rows = rows[sort_idx]
        sorted_cols = cols[sort_idx]

        unique_e_tensor, counts_tensor = torch.unique_consecutive(sorted_experts, return_counts=True)
        active_experts = unique_e_tensor.tolist()
        expert_to_idx_map = {}
        curr_offset = 0
        for e_id, cnt in zip(active_experts, counts_tensor.tolist()):
            expert_to_idx_map[e_id] = (
                sorted_rows[curr_offset : curr_offset + cnt],
                sorted_cols[curr_offset : curr_offset + cnt],
            )
            curr_offset += cnt

        uids_to_prefetch = self._get_next_layer_prefetch(hidden_states_flat, seq_length)
        expert_iterator = self.experts.load_experts(
            *((self.layer_id, e) for e in active_experts), unordered=True, uids_to_prefetch=uids_to_prefetch
        )

        for (_layer_index, expert_idx), expert_layer in expert_iterator:
            if expert_idx not in expert_to_idx_map:
                continue

            token_idx, slot_idx = expert_to_idx_map[expert_idx]
            current_state = hidden_states_flat.index_select(0, token_idx)
            current_weight = routing_weights[token_idx, slot_idx].unsqueeze(-1)
            current_hidden_states = expert_layer(hidden_states=current_state) * current_weight
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.reshape(batch_size, seq_length, hidden_dim)


class QwenMoeWrapperSkipBaseline(nn.Module):
    def __init__(
        self,
        text_config,
        layer_id: int,
        gate: nn.Linear,
        experts: nn.Module,
        skip_keep_k: int = 5,
        decode_skip_keep_k: int | None = None,
        layers=None,
        global_config=None,
    ):
        super().__init__()
        self.hidden_dim = text_config.hidden_size
        self.num_experts = text_config.num_experts
        self.top_k = text_config.num_experts_per_tok
        self.layer_id = layer_id
        self.gate = gate
        self.experts = experts
        self.skip_keep_k = skip_keep_k
        self.decode_skip_keep_k = skip_keep_k if decode_skip_keep_k is None else decode_skip_keep_k
        self.layers = list(layers) if layers is not None else None
        self.num_layers = len(self.layers) if self.layers is not None else 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, self.hidden_dim)

        router_logits = self.gate(hidden_states_flat)
        routing_weights, router_indices = qwen_topk_softmax(
            router_logits,
            self.top_k,
            normalize_topk=False,
        )
        keep_k = self.skip_keep_k if seq_length > 1 else self.decode_skip_keep_k
        keep_mask = build_fixed_keep_mask(router_indices, keep_k)
        effective_weights = renormalize_surviving_weights(routing_weights, keep_mask)
        effective_weights = effective_weights.to(hidden_states_flat.dtype)

        dense_routing_weights = torch.zeros_like(router_logits)
        dense_routing_weights.scatter_(1, router_indices, effective_weights.to(dense_routing_weights.dtype))

        hidden_states = hidden_states_flat.reshape(batch_size, -1, self.hidden_dim)
        return self.experts(hidden_states, dense_routing_weights, router_indices)
