import torch
from torch import nn

from ...cache_engine.config import CacheConfig
from ...skip import (
    build_decode_keep_mask,
    build_fixed_keep_mask,
    build_prefill_keep_mask,
    renormalize_surviving_weights,
)
from ...utils.modality import ModalityContext


class DeepSeekMoeWrapperBaseline(nn.Module):
    """
    DeepSeek-VL2 MoE Layer Wrapper (Baseline Offload)
    处理 shared experts 常驻 GPU + routed experts offload
    复用官方 MoEGate 计算 routing，输出公式:
    shared_output + routed_output
    其中 routed_scaling_factor 已由 MoEGate 内部应用。
    """
    def __init__(
        self,
        lang_config,
        layer_id: int,
        gate: nn.Module,
        shared_experts: nn.Module,
        expert_cache,
        layers=None,
        global_config=None,
    ):
        super().__init__()
        self.hidden_dim = lang_config.hidden_size
        self.num_routed_experts = lang_config.n_routed_experts  # 72
        self.num_shared_experts = lang_config.n_shared_experts  # 2
        self.top_k = lang_config.num_experts_per_tok  # 6
        self.routed_scaling_factor = lang_config.routed_scaling_factor  # 2.0
        self.scoring_func = getattr(lang_config, 'scoring_func', 'sigmoid')
        self.norm_topk_prob = getattr(lang_config, 'norm_topk_prob', True)

        self.layer_id = layer_id
        self.gate = gate  # nn.Linear，包含 e_score_correction_bias 属性
        self.shared_experts = shared_experts  # 常驻 GPU
        self.experts = expert_cache
        self.layers = list(layers) if layers is not None else None
        self.num_layers = len(self.layers) if self.layers is not None else 0

    def _compute_routing(self, hidden_states_flat):
        """
        兼容旧调试脚本接口。
        输入: [tokens, hidden]
        输出: (topk_weight, topk_idx)
        实际计算仍完全复用官方 MoEGate.forward。
        """
        hidden_states = hidden_states_flat.unsqueeze(1)
        topk_idx, topk_weight, _ = self.gate(hidden_states)
        return topk_weight, topk_idx

    def _get_next_layer_prefetch(self, hidden_states, seq_length):
        """计算下一层需要预取的专家 ID，跳过 Layer 0 (dense)"""
        uids_next = []
        next_layer_idx = self.layer_id + 1

        from .offload_config import get_deepseek_offload_config
        config = get_deepseek_offload_config()
        if not config.get('prefetch', True):
            return uids_next

        first_k_dense = config.get('first_k_dense_replace', 1)

        # 跳过 dense 层
        if next_layer_idx < first_k_dense or next_layer_idx >= self.num_layers:
            return uids_next

        # Prefill 阶段：激进预取
        if seq_length > 1:
            candidates = self.experts.check(next_layer_idx, list(range(self.num_routed_experts)))
            limit = config.get('prefetch_limit_prefill', 2)
            return candidates[:limit]

        # Decode 阶段：基于 Gate 预测
        limit = config.get('prefetch_limit_decode', 0)
        if limit <= 0:
            return uids_next

        if self.layers is not None:
            next_layer = self.layers[next_layer_idx]
            next_moe = getattr(next_layer, 'mlp', None)
            if next_moe is not None and hasattr(next_moe, 'gate'):
                with torch.no_grad():
                    next_selected, _, _ = next_moe.gate(hidden_states)
                    next_selected_unique = torch.unique(next_selected).tolist()
                    if next_selected_unique:
                        candidates = self.experts.check(next_layer_idx, next_selected_unique)
                        uids_next = candidates[:limit]

        return uids_next

    def _plan_full_routed_execution(self, selected_experts, hidden_states_flat):
        """
        构造与官方 moe_infer 对齐的 routed expert 执行计划。
        返回:
        - idxs: top-k flatten 后的逆置换索引
        - sorted_tokens: 按 expert 分组后的 token 输入
        - active_experts: 有 token 的 expert 列表（升序）
        - token_ranges: expert_idx -> (start_idx, end_idx)
        """
        cnts = selected_experts.new_zeros((selected_experts.shape[0], self.num_routed_experts))
        cnts.scatter_(1, selected_experts, 1)
        tokens_per_expert = cnts.sum(dim=0).cpu().tolist()
        idxs = selected_experts.view(-1).argsort()
        sorted_tokens = hidden_states_flat[idxs // selected_experts.shape[1]]

        active_experts = []
        token_ranges = {}
        start_idx = 0
        for expert_idx, num_tokens in enumerate(tokens_per_expert):
            if num_tokens == 0:
                continue
            end_idx = start_idx + num_tokens
            active_experts.append(expert_idx)
            token_ranges[expert_idx] = (start_idx, end_idx)
            start_idx = end_idx

        return idxs, sorted_tokens, active_experts, token_ranges

    def _plan_masked_slot_execution(self, selected_experts, slot_mask):
        """
        为 cached 路径构造 slot 对齐的 expert 执行计划。
        返回:
        - unique_experts: 有未命中 token 的 expert 列表
        - expert_to_idx_map: expert_idx -> (token_rows, topk_cols)
        """
        rows, cols = torch.where(slot_mask)
        if rows.numel() == 0:
            return [], {}

        flat_experts = selected_experts[rows, cols]
        sort_idx = torch.argsort(flat_experts)
        sorted_experts = flat_experts[sort_idx]
        sorted_rows = rows[sort_idx]
        sorted_cols = cols[sort_idx]

        unique_e_tensor, counts_tensor = torch.unique_consecutive(sorted_experts, return_counts=True)
        expert_to_idx_map = {}
        curr_offset = 0
        for e_id, cnt in zip(unique_e_tensor.tolist(), counts_tensor.tolist()):
            expert_to_idx_map[e_id] = (
                sorted_rows[curr_offset: curr_offset + cnt],
                sorted_cols[curr_offset: curr_offset + cnt],
            )
            curr_offset += cnt

        return unique_e_tensor.tolist(), expert_to_idx_map

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_dim = hidden_states.shape
        bs = batch_size * seq_length
        hidden_states_flat = hidden_states.reshape(bs, hidden_dim)

        # 1. Gate 计算：完全复用官方 MoEGate
        selected_experts, routing_weights, _ = self.gate(hidden_states)

        # 2. Shared Experts 计算（始终激活，常驻 GPU）
        shared_output = self.shared_experts(hidden_states_flat)

        # 3. Routed Experts 计算：严格对齐官方 moe_infer 的重排/聚合顺序。
        idxs, sorted_tokens, active_experts, token_ranges = self._plan_full_routed_execution(
            selected_experts,
            hidden_states_flat,
        )

        # 4. 预取下一层
        uids_to_prefetch = self._get_next_layer_prefetch(hidden_states, seq_length)

        # 5. 加载并计算 routed experts
        expert_iterator = self.experts.load_experts(
            *((self.layer_id, e) for e in active_experts),
            unordered=True,
            uids_to_prefetch=uids_to_prefetch
        )
        # 保持与官方 moe_infer 一致的逆置换布局，确保最终加权求和顺序完全一致。
        flat_size = selected_experts.numel()
        new_x = hidden_states_flat.new_empty((flat_size, hidden_dim))

        for (_, expert_idx), expert_layer in expert_iterator:
            start_idx, end_idx = token_ranges[expert_idx]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert_layer(hidden_states=tokens_for_this_expert)
            new_x[idxs[start_idx:end_idx]] = expert_out
        routed_output = (
            new_x.view(*selected_experts.shape, -1)
            .type(routing_weights.dtype)
            .mul_(routing_weights.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )

        # 6. 合并输出: shared + routed (routed_scaling_factor 已在 gate 中应用)
        final_output = shared_output + routed_output

        return final_output.reshape(batch_size, seq_length, hidden_dim)

class DeepSeekMoeWrapperSkipBaseline(DeepSeekMoeWrapperBaseline):
    """
    DeepSeek-VL2 MoE Layer Wrapper (Skip Baseline)
    全量模型常驻 GPU，固定保留前若干 routed experts 重算，剩余 routed experts 直接跳过。
    """

    def __init__(self, *args, experts=None, skip_keep_k: int = 4, decode_skip_keep_k: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.routed_experts = experts
        self.skip_keep_k = skip_keep_k
        self.decode_skip_keep_k = skip_keep_k if decode_skip_keep_k is None else decode_skip_keep_k

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_dim = hidden_states.shape
        bs = batch_size * seq_length
        hidden_states_flat = hidden_states.reshape(bs, hidden_dim)

        selected_experts, routing_weights, _ = self.gate(hidden_states)
        shared_output = self.shared_experts(hidden_states_flat)
        keep_k = self.skip_keep_k if seq_length > 1 else self.decode_skip_keep_k
        keep_mask = build_fixed_keep_mask(selected_experts, keep_k)

        effective_weights = renormalize_surviving_weights(
            routing_weights,
            keep_mask,
            target_row_sum=routing_weights.sum(dim=-1, keepdim=True),
        )
        collected_expert_outputs = torch.zeros(
            (bs, self.top_k, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        unique_experts, expert_to_idx_map = self._plan_masked_slot_execution(selected_experts, keep_mask)
        for expert_idx in unique_experts:
            token_idx, slot_idx = expert_to_idx_map[expert_idx]
            curr_inputs = hidden_states_flat.index_select(0, token_idx)
            expert_out = self.routed_experts[expert_idx](curr_inputs)
            collected_expert_outputs[token_idx, slot_idx] = expert_out

        routed_output = (
            collected_expert_outputs
            .to(effective_weights.dtype)
            .mul_(effective_weights.unsqueeze(dim=-1))
            .sum(dim=1)
            .to(hidden_states_flat.dtype)
        )

        final_output = shared_output + routed_output
        return final_output.reshape(batch_size, seq_length, hidden_dim)


class DeepSeekMoeWrapperSkipOffload(DeepSeekMoeWrapperBaseline):
    """
    DeepSeek-VL2 MoE Layer Wrapper (Skip Offload)
    复用 baseline 的 expert cache，仅对保留槽位执行 routed experts。
    """

    def __init__(self, *args, skip_keep_k: int = 4, decode_skip_keep_k: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_keep_k = skip_keep_k
        self.decode_skip_keep_k = skip_keep_k if decode_skip_keep_k is None else decode_skip_keep_k

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_dim = hidden_states.shape
        bs = batch_size * seq_length
        hidden_states_flat = hidden_states.reshape(bs, hidden_dim)

        selected_experts, routing_weights, _ = self.gate(hidden_states)
        shared_output = self.shared_experts(hidden_states_flat)
        keep_k = self.skip_keep_k if seq_length > 1 else self.decode_skip_keep_k
        keep_mask = build_fixed_keep_mask(selected_experts, keep_k)
        effective_weights = renormalize_surviving_weights(
            routing_weights,
            keep_mask,
            target_row_sum=routing_weights.sum(dim=-1, keepdim=True),
        )
        collected_expert_outputs = torch.zeros(
            (bs, self.top_k, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        unique_experts, expert_to_idx_map = self._plan_masked_slot_execution(selected_experts, keep_mask)
        if unique_experts:
            uids_to_prefetch = self._get_next_layer_prefetch(hidden_states, seq_length)
            expert_iterator = self.experts.load_experts(
                *((self.layer_id, e) for e in unique_experts),
                unordered=True,
                uids_to_prefetch=uids_to_prefetch,
            )
            for (_layer_index, expert_idx), expert_layer in expert_iterator:
                if expert_idx not in expert_to_idx_map:
                    continue
                token_idx, slot_idx = expert_to_idx_map[expert_idx]
                curr_inputs = hidden_states_flat.index_select(0, token_idx)
                expert_out = expert_layer(hidden_states=curr_inputs)
                collected_expert_outputs[token_idx, slot_idx] = expert_out

        routed_output = (
            collected_expert_outputs
            .to(effective_weights.dtype)
            .mul_(effective_weights.unsqueeze(dim=-1))
            .sum(dim=1)
            .to(hidden_states_flat.dtype)
        )

        final_output = shared_output + routed_output
        return final_output.reshape(batch_size, seq_length, hidden_dim)


class DeepSeekMoeWrapperCached(DeepSeekMoeWrapperBaseline):
    """
    DeepSeek-VL2 MoE Layer Wrapper (Hybrid: Offload + Similarity Cache)
    继承 baseline，添加相似度缓存逻辑
    """
    def __init__(
        self,
        *args,
        cache_manager=None,
        prefill_keep_strategy="importance",
        decode_search_strategy="hybrid",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cache_manager = cache_manager
        from ...cache_engine.search import VectorSearchEngine
        self.search_engine = VectorSearchEngine(cache_manager)
        self.prefill_keep_strategy = prefill_keep_strategy
        self.decode_search_strategy = decode_search_strategy
        self.search_event = torch.cuda.Event()
        self.fetch_stream = torch.cuda.Stream()
        self.fetch_event = torch.cuda.Event()
        self.decode_workspace = torch.zeros(
            (1, self.top_k, self.hidden_dim),
            dtype=self.gate.weight.dtype,
            device=self.gate.weight.device,
        )
        self.fetch_results_buffer = {}
        self.decode_bs1_on_hit = None
        self.decode_bs1_off_hit = None
        self.decode_bs1_on_slot = None
        self.decode_bs1_off_idx = None
        self.decode_bs1_on_idx = None
        self.decode_bs1_survivor_mask = None
        self.decode_bs1_weight_work = None
        self.decode_bs1_effective_weights = None

    def _get_prefill_keep_k(self):
        return getattr(CacheConfig, "PREFILL_KEEP_K", CacheConfig.KEEP_K)

    def _get_decode_keep_k(self):
        return getattr(CacheConfig, "DECODE_KEEP_K", CacheConfig.KEEP_K)

    def _build_compute_mask(self, selected_experts, routing_weights, is_prefill):
        if is_prefill and self.prefill_keep_strategy == "importance":
            return build_prefill_keep_mask(
                selected_experts=selected_experts,
                routing_weights=routing_weights,
                attn_scores=ModalityContext.get_attn_weights(),
                num_experts=self.num_routed_experts,
                keep_rate=CacheConfig.KEEP_RATE,
            )

        if is_prefill and self.prefill_keep_strategy == "fixed_keep_k":
            return build_decode_keep_mask(selected_experts, self._get_prefill_keep_k())
        if not is_prefill:
            return build_decode_keep_mask(selected_experts, self._get_decode_keep_k())

        raise ValueError(f"Unsupported prefill_keep_strategy: {self.prefill_keep_strategy}")

    def _ensure_decode_bs1_buffers(self, hidden_states_flat, routing_dtype):
        hidden_dim = hidden_states_flat.shape[-1]
        device = hidden_states_flat.device
        hidden_dtype = hidden_states_flat.dtype
        needs_realloc = (
            self.decode_workspace is None
            or self.decode_workspace.shape[0] < 1
            or self.decode_workspace.shape[1] != self.top_k
            or self.decode_workspace.shape[2] != hidden_dim
            or self.decode_workspace.dtype != hidden_dtype
            or self.decode_workspace.device != device
        )
        if needs_realloc:
            self.decode_workspace = torch.zeros(
                (1, self.top_k, hidden_dim),
                dtype=hidden_dtype,
                device=device,
            )

        small_shape = (self.top_k,)
        small_device_mismatch = (
            self.decode_bs1_on_hit is None
            or self.decode_bs1_on_hit.shape != small_shape
            or self.decode_bs1_on_hit.device != device
        )
        if small_device_mismatch:
            self.decode_bs1_on_hit = torch.zeros(small_shape, dtype=torch.bool, device=device)
            self.decode_bs1_off_hit = torch.zeros(small_shape, dtype=torch.bool, device=device)
            self.decode_bs1_on_slot = torch.zeros(small_shape, dtype=torch.long, device=device)
            self.decode_bs1_off_idx = torch.zeros(small_shape, dtype=torch.long, device=device)
            self.decode_bs1_on_idx = torch.full((1,), -1, dtype=torch.long, device=device)
            self.decode_bs1_survivor_mask = torch.zeros(small_shape, dtype=torch.bool, device=device)

        if (
            self.decode_bs1_weight_work is None
            or self.decode_bs1_weight_work.shape != small_shape
            or self.decode_bs1_weight_work.dtype != routing_dtype
            or self.decode_bs1_weight_work.device != device
        ):
            self.decode_bs1_weight_work = torch.zeros(small_shape, dtype=routing_dtype, device=device)

        if (
            self.decode_bs1_effective_weights is None
            or self.decode_bs1_effective_weights.shape != small_shape
            or self.decode_bs1_effective_weights.dtype != routing_dtype
            or self.decode_bs1_effective_weights.device != device
        ):
            self.decode_bs1_effective_weights = torch.zeros(small_shape, dtype=routing_dtype, device=device)

        return self.decode_workspace[0]

    def _build_decode_bs1_expert_plan(self, selected_experts_cpu, keep_k):
        unique_experts = []
        expert_to_slots = {}
        for slot in range(keep_k):
            expert_id = int(selected_experts_cpu[slot])
            if expert_id not in expert_to_slots:
                expert_to_slots[expert_id] = []
                unique_experts.append(expert_id)
            expert_to_slots[expert_id].append(slot)
        return unique_experts, expert_to_slots

    def _compute_decode_bs1_effective_weights(self, routing_weights_row, survivor_mask):
        weights_work = self.decode_bs1_weight_work
        effective_weights = self.decode_bs1_effective_weights

        weights_work.copy_(routing_weights_row)
        if bool(survivor_mask.all().item()):
            effective_weights.copy_(weights_work)
            return effective_weights

        if not bool(survivor_mask.any().item()):
            effective_weights.zero_()
            return effective_weights

        target_row_sum = weights_work.sum()
        weights_work.masked_fill_(~survivor_mask, 0)
        kept_sum = weights_work.sum().clamp_min(1e-20)
        effective_weights.copy_(weights_work * (target_row_sum / kept_sum))
        return effective_weights

    def _forward_decode_bs1(self, hidden_states, hidden_states_flat, routing_weights, selected_experts, is_vision):
        collected_expert_outputs = self._ensure_decode_bs1_buffers(hidden_states_flat, routing_weights.dtype)
        collected_expert_outputs.zero_()

        on_hit = self.decode_bs1_on_hit
        off_hit = self.decode_bs1_off_hit
        on_slot = self.decode_bs1_on_slot
        off_idx = self.decode_bs1_off_idx
        on_idx = self.decode_bs1_on_idx
        survivor_mask = self.decode_bs1_survivor_mask
        keep_k = max(0, min(int(self._get_decode_keep_k()), self.top_k))

        on_hit.zero_()
        off_hit.zero_()
        on_slot.zero_()
        off_idx.zero_()
        on_idx.fill_(-1)

        selected_experts_row = selected_experts[0]
        keep_experts_cpu = selected_experts_row[:keep_k].detach().cpu().tolist()
        unique_experts, expert_to_slots = self._build_decode_bs1_expert_plan(keep_experts_cpu, keep_k)

        uids_next = self._get_next_layer_prefetch(hidden_states, 1)
        expert_iterator = self.experts.load_experts(
            *((self.layer_id, expert_id) for expert_id in unique_experts),
            unordered=True,
            uids_to_prefetch=uids_next,
        )

        if self.decode_search_strategy == "hybrid":
            self.search_engine.search_decode_hybrid_bs1_into(
                hidden_state=hidden_states_flat[0],
                selected_experts_row=selected_experts_row,
                is_vision=is_vision[0],
                keep_k=keep_k,
                out_on_hit=on_hit,
                out_on_idx=on_idx,
                out_off_hit=off_hit,
                out_off_idx=off_idx,
                out_on_slot=on_slot,
            )
        elif self.decode_search_strategy == "offline":
            self.search_engine.search_decode_offline_bs1_into(
                hidden_state=hidden_states_flat[0],
                selected_experts_row=selected_experts_row,
                is_vision=is_vision[0],
                keep_k=keep_k,
                out_off_hit=off_hit,
                out_off_idx=off_idx,
            )
        else:
            raise ValueError(f"Unsupported decode_search_strategy: {self.decode_search_strategy}")

        if keep_k < self.top_k:
            hit_slots = torch.nonzero(on_hit[keep_k:], as_tuple=False).flatten()
            if hit_slots.numel() > 0:
                hit_slots = hit_slots + keep_k
                cache_rows = on_idx.expand(hit_slots.numel())
                collected_expert_outputs.index_copy_(
                    0,
                    hit_slots,
                    self.cache_manager.online_values[cache_rows, on_slot.index_select(0, hit_slots)],
                )

        off_slots = torch.nonzero(off_hit[keep_k:], as_tuple=False).flatten()
        need_offline_fetch = off_slots.numel() > 0
        if need_offline_fetch:
            off_slots = off_slots + keep_k
            modality_id = 0 if bool(is_vision[0].item()) else 1
            current_stream = torch.cuda.current_stream()
            with torch.cuda.stream(self.fetch_stream):
                self.fetch_stream.wait_stream(current_stream)
                self.cache_manager.gather_values_decode_bs1(
                    expert_ids=selected_experts_row.index_select(0, off_slots),
                    modality_id=modality_id,
                    cluster_ids=off_idx.index_select(0, off_slots),
                    slot_ids=off_slots,
                    out_buffer=collected_expert_outputs,
                )

        shared_output = self.shared_experts(hidden_states_flat)

        if expert_iterator:
            for (_, expert_idx), expert_layer in expert_iterator:
                expert_out = expert_layer(hidden_states=hidden_states_flat)[0]
                for slot in expert_to_slots[expert_idx]:
                    collected_expert_outputs[slot].copy_(expert_out)

        if need_offline_fetch:
            torch.cuda.current_stream().wait_stream(self.fetch_stream)

        survivor_mask.zero_()
        if keep_k > 0:
            survivor_mask[:keep_k] = True
        if keep_k < self.top_k:
            survivor_mask[keep_k:] = on_hit[keep_k:] | off_hit[keep_k:]

        effective_weights = self._compute_decode_bs1_effective_weights(routing_weights[0], survivor_mask)
        routed_output = (
            collected_expert_outputs
            .to(effective_weights.dtype)
            .mul_(effective_weights.unsqueeze(-1))
            .sum(dim=0)
            .to(hidden_states_flat.dtype)
        )

        if self.cache_manager.enable_online:
            self.cache_manager.update_online_cache_bs1(
                hidden_state=hidden_states_flat[0],
                expert_ids=selected_experts_row,
                expert_outputs=collected_expert_outputs,
                valid_slot_mask=survivor_mask,
            )

        final_output = shared_output[0] + routed_output
        return final_output.view(1, 1, -1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_dim = hidden_states.shape
        bs = batch_size * seq_length
        hidden_states_flat = hidden_states.reshape(bs, hidden_dim)

        # 1. Gate 计算：完全复用官方 MoEGate
        selected_experts, routing_weights, _ = self.gate(hidden_states)

        is_prefill = seq_length > 1
        is_vision = ModalityContext.get_modality_mask(hidden_states.device, seq_length).reshape(-1)

        if not is_prefill and batch_size == 1:
            return self._forward_decode_bs1(
                hidden_states=hidden_states,
                hidden_states_flat=hidden_states_flat,
                routing_weights=routing_weights,
                selected_experts=selected_experts,
                is_vision=is_vision,
            )

        # 2. Shared Experts 计算（始终激活）
        shared_output = self.shared_experts(hidden_states_flat)

        # 3. 准备 collected_expert_outputs buffer
        if is_prefill:
            collected_expert_outputs = torch.zeros(
                (bs, self.top_k, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
        else:
            needs_realloc = (
                self.decode_workspace is None
                or self.decode_workspace.shape[0] < bs
                or self.decode_workspace.shape[1] != self.top_k
                or self.decode_workspace.shape[2] != hidden_dim
                or self.decode_workspace.dtype != hidden_states.dtype
                or self.decode_workspace.device != hidden_states.device
            )
            if needs_realloc:
                new_bs = max(1, bs)
                self.decode_workspace = torch.zeros(
                    (new_bs, self.top_k, hidden_dim),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                collected_expert_outputs = self.decode_workspace[:bs]
            else:
                collected_expert_outputs = self.decode_workspace[:bs]
                collected_expert_outputs.zero_()
        # cached 路径需要保持 [token, slot, hidden] 布局，便于融合在线/离线命中与实时 expert 输出。

        # 4. 先确定固定保留集合，并提前启动专家加载
        self.fetch_results_buffer.clear()
        fetch_results = self.fetch_results_buffer

        on_hit_mask = torch.zeros((bs, self.top_k), dtype=torch.bool, device=hidden_states.device)
        off_hit_mask = torch.zeros((bs, self.top_k), dtype=torch.bool, device=hidden_states.device)
        on_buffer_indices = None
        on_slot_indices = torch.zeros((bs, self.top_k), dtype=torch.long, device=hidden_states.device)
        off_cluster_indices = torch.zeros((bs, self.top_k), dtype=torch.long, device=hidden_states.device)

        compute_mask = self._build_compute_mask(selected_experts, routing_weights, is_prefill)
        search_mask = ~compute_mask

        # 定义 fetch 函数
        def perform_value_fetch():
            with torch.cuda.stream(self.fetch_stream):
                self.search_event.wait(self.fetch_stream)
                self.experts.load_event.wait(self.fetch_stream)
                rows, cols = torch.where(off_hit_mask)
                if rows.numel() > 0:
                    eids = selected_experts[rows, cols].cpu()
                    cids = off_cluster_indices[rows, cols].cpu()
                    mids = torch.where(is_vision[rows], 0, 1).cpu()
                    vals = self.cache_manager.gather_values(eids, mids, cids).to(hidden_states.device, non_blocking=True)
                    fetch_results['offline'] = (rows, cols, vals)
                self.fetch_event.record()

        # 5. 固定保留集合直接启动专家加载，search 不再决定是否重算
        uids_next = self._get_next_layer_prefetch(hidden_states, seq_length)
        unique_experts, expert_to_idx_map = self._plan_masked_slot_execution(selected_experts, compute_mask)

        expert_iterator = self.experts.load_experts(
            *((self.layer_id, e) for e in unique_experts),
            unordered=True,
            uids_to_prefetch=uids_next,
            values_fetcher=perform_value_fetch,
            values_event=self.fetch_event,
        )

        # 6. Search 回到当前流上执行，只保留 event 给 fetch_stream 做依赖。
        if search_mask.any():
            if is_prefill:
                if self.decode_search_strategy == "offline":
                    off_hit_mask, off_cluster_indices = self.search_engine.search_prefill_offline_dot(
                        self.layer_id, hidden_states_flat, selected_experts, is_vision, search_mask
                    )
                else:
                    off_hit_mask, off_cluster_indices = self.search_engine.search_prefill_offline(
                        self.layer_id, hidden_states_flat, selected_experts, is_vision, search_mask
                    )
            else:
                if self.decode_search_strategy == "hybrid":
                    on_hit_mask, on_buffer_indices, off_hit_mask, off_cluster_indices, on_slot_indices = self.search_engine.search_decode_hybrid(
                        self.layer_id,
                        hidden_states_flat,
                        selected_experts,
                        is_vision,
                        keep_k=self._get_decode_keep_k(),
                    )
                elif self.decode_search_strategy == "offline":
                    off_hit_mask, off_cluster_indices = self.search_engine.search_decode_offline_dot(
                        self.layer_id, hidden_states_flat, selected_experts, is_vision, search_mask
                    )
                else:
                    raise ValueError(f"Unsupported decode_search_strategy: {self.decode_search_strategy}")
            self.search_event.record()
        else:
            self.search_event.record()

        if expert_iterator:
            for (_, expert_idx), expert_layer in expert_iterator:
                token_idx, slot_idx = expert_to_idx_map[expert_idx]
                curr_inputs = hidden_states_flat.index_select(0, token_idx)
                expert_out = expert_layer(hidden_states=curr_inputs)
                collected_expert_outputs[token_idx, slot_idx] = expert_out

        reuse_hit_mask = off_hit_mask

        # 7. 填充在线缓存结果
        if on_buffer_indices is not None:
            rows, cols = torch.where(on_hit_mask)
            if rows.numel() > 0:
                hist_token_idxs = on_buffer_indices[rows]
                slot_idxs = on_slot_indices[rows, cols]
                vals = self.cache_manager.online_values[hist_token_idxs, slot_idxs]
                collected_expert_outputs[rows, cols] = vals
            reuse_hit_mask = on_hit_mask | off_hit_mask

        # 8. 填充离线缓存结果
        if 'offline' in fetch_results:
            rows, cols, vals = fetch_results['offline']
            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(self.fetch_stream)
            vals.record_stream(current_stream)
            collected_expert_outputs[rows, cols] = vals

        survivor_mask = compute_mask | reuse_hit_mask
        effective_weights = renormalize_surviving_weights(
            routing_weights,
            survivor_mask,
            target_row_sum=routing_weights.sum(dim=-1, keepdim=True),
        )

        # 9. 合并 routed 输出：对 compute + reuse_hit 的 surviving 权重重新归一化后聚合
        routed_output = (
            collected_expert_outputs
            .to(effective_weights.dtype)
            .mul_(effective_weights.unsqueeze(dim=-1))
            .sum(dim=1)
            .to(hidden_states_flat.dtype)
        )

        # 10. 更新在线缓存
        if self.cache_manager.enable_online:
            ONLINE_SIZE = self.cache_manager.online_size
            start_cache_idx = max(0, bs - ONLINE_SIZE) if is_prefill else 0
            self.cache_manager.update_online_cache(
                hidden_states_flat[start_cache_idx:],
                selected_experts[start_cache_idx:],
                collected_expert_outputs[start_cache_idx:],
                valid_slot_mask=survivor_mask[start_cache_idx:],
            )

        # 11. 最终输出: shared + routed (routed_scaling_factor 已在 gate 中应用)
        final_output = shared_output + routed_output

        return final_output.reshape(batch_size, seq_length, hidden_dim)
