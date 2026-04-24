import torch
from contextlib import nullcontext
from torch import nn

from ...ops import qwen_topk_softmax
from ...cache_engine.config import CacheConfig
from ...cache_engine.search import VectorSearchEngine
from ...skip import (
    build_decode_keep_mask,
    build_prefill_keep_mask,
    renormalize_surviving_weights,
)
from ...utils.modality import ModalityContext
from .custom_layers import QwenMoeWrapperBaseline


class QwenMoeWrapperCached(QwenMoeWrapperBaseline):
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
        self.search_engine = VectorSearchEngine(cache_manager)
        self.prefill_keep_strategy = prefill_keep_strategy
        self.decode_search_strategy = decode_search_strategy
        self.search_event = torch.cuda.Event()
        self.fetch_stream = torch.cuda.Stream()
        self.fetch_event = torch.cuda.Event()
        # self.update_stream = torch.cuda.Stream()
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
        self.profiler = None

    def set_perf_profiler(self, profiler):
        self.profiler = profiler

    def _measure_cuda(self, key):
        if self.profiler is None or not self.profiler.is_active():
            return nullcontext()
        return self.profiler.measure_cuda(
            key,
            layer_id=self.layer_id,
            device=self.gate.weight.device,
        )

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
                num_experts=self.num_experts,
                keep_rate=CacheConfig.KEEP_RATE,
            )

        if is_prefill and self.prefill_keep_strategy == "fixed_keep_k":
            return build_decode_keep_mask(selected_experts, self._get_prefill_keep_k())
        if not is_prefill:
            return build_decode_keep_mask(selected_experts, self._get_decode_keep_k())

        raise ValueError(f"Unsupported prefill_keep_strategy: {self.prefill_keep_strategy}")

    def _ensure_decode_bs1_buffers(self, hidden_states_flat):
        hidden_dim = hidden_states_flat.shape[-1]
        device = hidden_states_flat.device
        dtype = hidden_states_flat.dtype
        needs_realloc = (
            self.decode_workspace is None
            or self.decode_workspace.shape[0] < 1
            or self.decode_workspace.shape[1] != self.top_k
            or self.decode_workspace.shape[2] != hidden_dim
            or self.decode_workspace.dtype != dtype
            or self.decode_workspace.device != device
        )
        if needs_realloc:
            self.decode_workspace = torch.zeros(
                (1, self.top_k, hidden_dim),
                dtype=dtype,
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
            self.decode_bs1_weight_work = torch.zeros(small_shape, dtype=torch.float32, device=device)

        if (
            self.decode_bs1_effective_weights is None
            or self.decode_bs1_effective_weights.shape != small_shape
            or self.decode_bs1_effective_weights.dtype != dtype
            or self.decode_bs1_effective_weights.device != device
        ):
            self.decode_bs1_effective_weights = torch.zeros(small_shape, dtype=dtype, device=device)

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
        weights_work.div_(weights_work.sum().clamp_min(1e-20))

        if bool(survivor_mask.all().item()):
            effective_weights.copy_(weights_work.to(dtype=effective_weights.dtype))
            return effective_weights

        if not bool(survivor_mask.any().item()):
            effective_weights.zero_()
            return effective_weights

        weights_work.masked_fill_(~survivor_mask, 0)
        kept_sum = weights_work.sum().clamp_min(1e-20)
        effective_weights.copy_((weights_work / kept_sum).to(dtype=effective_weights.dtype))
        return effective_weights

    def _forward_decode_bs1(self, hidden_states_flat, routing_weights, selected_experts, is_vision):
        collected_expert_outputs = self._ensure_decode_bs1_buffers(hidden_states_flat)
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

        uids_next = self._get_next_layer_prefetch(hidden_states_flat, 1)
        expert_iterator = self.experts.load_experts(
            *((self.layer_id, expert_id) for expert_id in unique_experts),
            unordered=True,
            uids_to_prefetch=uids_next,
        )

        if self.decode_search_strategy == "hybrid":
            with self._measure_cuda("qwen.search.decode.hybrid"):
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

        if expert_iterator:
            for (_, expert_idx), expert_layer in expert_iterator:
                with self._measure_cuda("qwen.expert_compute.decode"):
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
        final_hidden_states = torch.matmul(effective_weights, collected_expert_outputs)

        if self.cache_manager.enable_online:
            self.cache_manager.update_online_cache_bs1(
                hidden_state=hidden_states_flat[0],
                expert_ids=selected_experts_row,
                expert_outputs=collected_expert_outputs,
                valid_slot_mask=survivor_mask,
            )

        return final_hidden_states.view(1, 1, -1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # collector = get_perf_stats()
        # moe_timer = CudaTimer(enabled=collector.enabled)
        # moe_timer.__enter__()
        
        batch_size, seq_length, hidden_dim = hidden_states.shape
        bs = batch_size * seq_length
        hidden_states_flat = hidden_states.reshape(bs, hidden_dim)

        # 1) 路由
        router_logits = self.gate(hidden_states_flat)
        routing_weights, selected_experts = qwen_topk_softmax(
            router_logits,
            self.top_k,
            normalize_topk=False,
        )
        
        is_prefill = seq_length > 1
        is_vision = ModalityContext.get_modality_mask(hidden_states.device, seq_length).reshape(-1)
        expert_compute_key = "qwen.expert_compute.prefill" if is_prefill else "qwen.expert_compute.decode"

        if not is_prefill and batch_size == 1:
            return self._forward_decode_bs1(
                hidden_states_flat=hidden_states_flat,
                routing_weights=routing_weights,
                selected_experts=selected_experts,
                is_vision=is_vision,
            )
        
        # Buffer 复用
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

        # 2) 先确定固定保留集合，并提前启动专家加载
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

        # 3) 固定保留集合直接启动专家加载，search 不再决定是否重算
        active_slots = selected_experts[compute_mask]
        uids_next = self._get_next_layer_prefetch(hidden_states_flat, seq_length)
        unique_experts = torch.unique(active_slots).tolist()

        expert_iterator = self.experts.load_experts(
            *((self.layer_id, e) for e in unique_experts),
            unordered=True,
            uids_to_prefetch=uids_next,
            values_fetcher=perform_value_fetch,
            values_event=self.fetch_event,
        )

        # 4) Search 回到当前流上执行，只保留 event 给 fetch_stream 做依赖。
        if search_mask.any():
            if is_prefill:
                with self._measure_cuda("qwen.search.prefill.offline"):
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
                    with self._measure_cuda("qwen.search.decode.hybrid"):
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

        # === 优化开始 ===
        # 1. 一次性获取所有需要计算的坐标 (Row, Col)
        c_rows, c_cols = torch.where(compute_mask)
        
        if c_rows.numel() > 0:
            # 2. 获取这些位置对应的专家 ID
            c_experts = selected_experts[c_rows, c_cols]
            
            # 3. 对专家 ID 进行排序，将相同专家的任务聚集在一起
            sort_idx = torch.argsort(c_experts)
            sorted_experts = c_experts[sort_idx]
            sorted_rows = c_rows[sort_idx]
            sorted_cols = c_cols[sort_idx]
            
            # 4. 计算每个专家的任务数量 (Counts)
            unique_e_tensor, counts_tensor = torch.unique_consecutive(sorted_experts, return_counts=True)
            
            # 5. 构建专家索引字典: expert_id -> (token_idx, slot_idx)
            expert_to_idx_map = {}
            u_e_list = unique_e_tensor.tolist()
            counts_list = counts_tensor.tolist()
            
            curr_offset = 0
            for e_id, cnt in zip(u_e_list, counts_list):
                # 切片获取该专家对应的所有行和列索引
                expert_to_idx_map[e_id] = (
                    sorted_rows[curr_offset : curr_offset + cnt],
                    sorted_cols[curr_offset : curr_offset + cnt]
                )
                curr_offset += cnt
        else:
            expert_to_idx_map = {}

        if expert_iterator:
            for (_, expert_idx), expert_layer in expert_iterator:
                token_idx, slot_idx = expert_to_idx_map[expert_idx]
                curr_inputs = hidden_states_flat.index_select(0, token_idx)
                with self._measure_cuda(expert_compute_key):
                    expert_out = expert_layer(hidden_states=curr_inputs)
                collected_expert_outputs[token_idx, slot_idx] = expert_out

        reuse_hit_mask = off_hit_mask

        # A. 填充在线结果 (仅 Decode 阶段有效，直接使用 Kernel 返回的 slot index)
        if on_buffer_indices is not None:
            rows, cols = torch.where(on_hit_mask)
            if rows.numel() > 0:
                hist_token_idxs = on_buffer_indices[rows]
                slot_idxs = on_slot_indices[rows, cols] # 直接查表，Zero Overhead
                vals = self.cache_manager.online_values[hist_token_idxs, slot_idxs]
                collected_expert_outputs[rows, cols] = vals
            reuse_hit_mask = on_hit_mask | off_hit_mask

        # 5) 融合复用结果
        # expert_timer = CudaTimer(enabled=collector.enabled)
        # expert_timer.__enter__()
            
        # B. 填充离线结果 (异步等待)
        if "offline" in fetch_results:
            rows, cols, vals = fetch_results["offline"]
            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(self.fetch_stream)
            vals.record_stream(current_stream)
            collected_expert_outputs[rows, cols] = vals

        survivor_mask = compute_mask | reuse_hit_mask
        effective_weights = renormalize_surviving_weights(routing_weights, survivor_mask).to(router_logits.dtype)
        final_hidden_states = torch.bmm(effective_weights.unsqueeze(1), collected_expert_outputs).squeeze(1)

        # expert_timer.__exit__()
        # if collector.enabled:
        #     phase = "prefill" if seq_length > 1 else "decode"
        #     collector.get_stats("cached", phase, self.layer_id).add_expert_compute(expert_timer.elapsed_ms())

        # 5) 同步写回在线缓存
        if self.cache_manager.enable_online:
            ONLINE_SIZE = self.cache_manager.online_size
            start_cache_idx = max(0, bs - ONLINE_SIZE) if is_prefill else 0
            self.cache_manager.update_online_cache(
                hidden_states_flat[start_cache_idx:],
                selected_experts[start_cache_idx:],
                collected_expert_outputs[start_cache_idx:],
                valid_slot_mask=survivor_mask[start_cache_idx:],
            )

        # moe_timer.__exit__()
        # if collector.enabled:
        #     phase = "prefill" if seq_length > 1 else "decode"
        #     collector.get_stats("cached", phase, self.layer_id).add_moe_layer(moe_timer.elapsed_ms())

        return final_hidden_states.reshape(batch_size, seq_length, hidden_dim)
