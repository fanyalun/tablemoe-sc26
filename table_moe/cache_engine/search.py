import torch

from .config import CacheConfig


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _normalize_last_dim(x: torch.Tensor, eps: float, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    denom = norms.clamp_min(eps)
    normalized = x / denom
    zero_mask = norms <= eps
    if zero_mask.any():
        normalized = normalized.masked_fill(zero_mask, 0.0)
    if out_dtype is not None:
        normalized = normalized.to(dtype=out_dtype)
    return normalized


try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[Warning] Triton not installed. Fused kernels disabled.")


if HAS_TRITON:
    @triton.jit
    def prefill_offline_search_kernel(
        Query_Low_ptr,
        Off_Centroids_ptr,
        Target_EID_ptr,
        Is_Vision_Mask_ptr,
        Reuse_Mask_ptr,
        Out_Off_Hit_ptr,
        Out_Off_Idx_ptr,
        stride_ql_b,
        stride_ql_d,
        stride_oc_e,
        stride_oc_m,
        stride_oc_k,
        stride_oc_d,
        stride_te_b,
        stride_te_s,
        stride_rm_b,
        stride_rm_s,
        stride_out_b,
        stride_out_s,
        threshold_vision,
        threshold_text,
        L_DIM: tl.constexpr,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        TOP_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_idx = pid // TOP_K
        slot_idx = pid % TOP_K

        reuse = tl.load(Reuse_Mask_ptr + batch_idx * stride_rm_b + slot_idx * stride_rm_s)
        if not reuse:
            tl.store(Out_Off_Hit_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, False)
            tl.store(Out_Off_Idx_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, 0)
            return

        target_eid = tl.load(Target_EID_ptr + batch_idx * stride_te_b + slot_idx * stride_te_s)
        is_vision = tl.load(Is_Vision_Mask_ptr + batch_idx).to(tl.int1)
        mod_id = tl.where(is_vision, 0, 1)

        offs_l = tl.arange(0, L_DIM)
        x_low = tl.load(Query_Low_ptr + batch_idx * stride_ql_b + offs_l * stride_ql_d, mask=offs_l < L_DIM, other=0.0)

        c_ptr_base = Off_Centroids_ptr + target_eid * stride_oc_e + mod_id * stride_oc_m
        offs_k = tl.arange(0, BLOCK_K)
        c_ptrs = c_ptr_base + (offs_k[:, None] * stride_oc_k + offs_l[None, :] * stride_oc_d)
        mask_k = offs_k[:, None] < K
        c_block = tl.load(c_ptrs, mask=mask_k, other=0.0)

        scores = tl.sum(x_low[None, :] * c_block, axis=1)
        scores = tl.where(offs_k < K, scores, -65504.0)
        max_val, max_idx = tl.max(scores, axis=0, return_indices=True)

        threshold = tl.where(is_vision, threshold_vision, threshold_text)
        hit = max_val >= threshold
        tl.store(Out_Off_Hit_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, hit)
        tl.store(Out_Off_Idx_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, max_idx)

    @triton.jit
    def decode_offline_search_kernel(
        X_High_ptr,
        Proj_Vis_ptr,
        Bias_Vis_ptr,
        Proj_Txt_ptr,
        Bias_Txt_ptr,
        Off_Centroids_ptr,
        Target_EID_ptr,
        Is_Vision_Mask_ptr,
        Reuse_Mask_ptr,
        Out_Off_Hit_ptr,
        Out_Off_Idx_ptr,
        stride_xh_b,
        stride_xh_d,
        stride_pv_h,
        stride_pv_l,
        stride_pt_h,
        stride_pt_l,
        stride_oc_e,
        stride_oc_m,
        stride_oc_k,
        stride_oc_d,
        stride_te_b,
        stride_te_s,
        stride_rm_b,
        stride_rm_s,
        stride_out_b,
        stride_out_s,
        H_DIM: tl.constexpr,
        L_DIM: tl.constexpr,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
        TOP_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_idx = pid // TOP_K
        slot_idx = pid % TOP_K

        reuse = tl.load(Reuse_Mask_ptr + batch_idx * stride_rm_b + slot_idx * stride_rm_s)
        if not reuse:
            tl.store(Out_Off_Hit_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, False)
            tl.store(Out_Off_Idx_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, 0)
            return

        target_eid = tl.load(Target_EID_ptr + batch_idx * stride_te_b + slot_idx * stride_te_s)
        is_vision = tl.load(Is_Vision_Mask_ptr + batch_idx).to(tl.int1)
        mod_id = tl.where(is_vision, 0, 1)

        x_low_acc = tl.zeros([L_DIM], dtype=tl.float32)
        for k in range(0, H_DIM, BLOCK_H):
            offs_h = k + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H_DIM
            x_chunk = tl.load(X_High_ptr + batch_idx * stride_xh_b + offs_h, mask=mask_h, other=0.0)

            offs_l = tl.arange(0, L_DIM)
            p_vis_ptrs = Proj_Vis_ptr + (offs_h[:, None] * stride_pv_h + offs_l[None, :] * stride_pv_l)
            p_txt_ptrs = Proj_Txt_ptr + (offs_h[:, None] * stride_pt_h + offs_l[None, :] * stride_pt_l)
            p_ptrs = tl.where(is_vision, p_vis_ptrs, p_txt_ptrs)
            proj_chunk = tl.load(p_ptrs, mask=mask_h[:, None], other=0.0)
            x_low_acc += tl.sum(x_chunk[:, None] * proj_chunk, axis=0)

        offs_l = tl.arange(0, L_DIM)
        b_ptr = tl.where(is_vision, Bias_Vis_ptr + offs_l, Bias_Txt_ptr + offs_l)
        x_low = (x_low_acc + tl.load(b_ptr)).to(tl.float16)

        c_ptr_base = Off_Centroids_ptr + target_eid * stride_oc_e + mod_id * stride_oc_m
        offs_k = tl.arange(0, BLOCK_K)
        c_ptrs = c_ptr_base + (offs_k[:, None] * stride_oc_k + offs_l[None, :] * stride_oc_d)
        mask_k = offs_k[:, None] < K
        c_block = tl.load(c_ptrs, mask=mask_k, other=0.0)

        scores = tl.sum(x_low[None, :] * c_block, axis=1)
        scores = tl.where(offs_k < K, scores, -65504.0)
        max_val, max_idx = tl.max(scores, axis=0, return_indices=True)

        hit = max_val > 0.0
        tl.store(Out_Off_Hit_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, hit)
        tl.store(Out_Off_Idx_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, max_idx)

    @triton.jit
    def decode_offline_dot_search_kernel(
        X_High_ptr,
        Proj_Vis_ptr,
        Bias_Vis_ptr,
        Proj_Txt_ptr,
        Bias_Txt_ptr,
        Off_Keys_Raw_ptr,
        Target_EID_ptr,
        Is_Vision_Mask_ptr,
        Reuse_Mask_ptr,
        Out_Off_Hit_ptr,
        Out_Off_Idx_ptr,
        stride_xh_b,
        stride_xh_d,
        stride_pv_h,
        stride_pv_l,
        stride_pt_h,
        stride_pt_l,
        stride_ok_e,
        stride_ok_m,
        stride_ok_k,
        stride_ok_d,
        stride_te_b,
        stride_te_s,
        stride_rm_b,
        stride_rm_s,
        stride_out_b,
        stride_out_s,
        threshold_dot,
        H_DIM: tl.constexpr,
        L_DIM: tl.constexpr,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
        TOP_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_idx = pid // TOP_K
        slot_idx = pid % TOP_K

        reuse = tl.load(Reuse_Mask_ptr + batch_idx * stride_rm_b + slot_idx * stride_rm_s)
        if not reuse:
            tl.store(Out_Off_Hit_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, False)
            tl.store(Out_Off_Idx_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, 0)
            return

        target_eid = tl.load(Target_EID_ptr + batch_idx * stride_te_b + slot_idx * stride_te_s)
        is_vision = tl.load(Is_Vision_Mask_ptr + batch_idx).to(tl.int1)
        mod_id = tl.where(is_vision, 0, 1)

        x_low_acc = tl.zeros([L_DIM], dtype=tl.float32)
        for k in range(0, H_DIM, BLOCK_H):
            offs_h = k + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H_DIM
            x_chunk = tl.load(X_High_ptr + batch_idx * stride_xh_b + offs_h, mask=mask_h, other=0.0)

            offs_l = tl.arange(0, L_DIM)
            p_vis_ptrs = Proj_Vis_ptr + (offs_h[:, None] * stride_pv_h + offs_l[None, :] * stride_pv_l)
            p_txt_ptrs = Proj_Txt_ptr + (offs_h[:, None] * stride_pt_h + offs_l[None, :] * stride_pt_l)
            p_ptrs = tl.where(is_vision, p_vis_ptrs, p_txt_ptrs)
            proj_chunk = tl.load(p_ptrs, mask=mask_h[:, None], other=0.0)
            x_low_acc += tl.sum(x_chunk[:, None] * proj_chunk, axis=0)

        offs_l = tl.arange(0, L_DIM)
        b_ptr = tl.where(is_vision, Bias_Vis_ptr + offs_l, Bias_Txt_ptr + offs_l)
        x_low = (x_low_acc + tl.load(b_ptr)).to(tl.float16)

        k_ptr_base = Off_Keys_Raw_ptr + target_eid * stride_ok_e + mod_id * stride_ok_m
        offs_k = tl.arange(0, BLOCK_K)
        k_ptrs = k_ptr_base + (offs_k[:, None] * stride_ok_k + offs_l[None, :] * stride_ok_d)
        mask_k = offs_k[:, None] < K
        k_block = tl.load(k_ptrs, mask=mask_k, other=0.0)

        scores = tl.sum(x_low[None, :] * k_block, axis=1)
        scores = tl.where(offs_k < K, scores, -65504.0)
        max_val, max_idx = tl.max(scores, axis=0, return_indices=True)

        hit = max_val > threshold_dot
        tl.store(Out_Off_Hit_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, hit)
        tl.store(Out_Off_Idx_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, max_idx)

    @triton.jit
    def decode_hybrid_search_kernel(
        X_High_ptr,
        Proj_Vis_ptr,
        Bias_Vis_ptr,
        Proj_Txt_ptr,
        Bias_Txt_ptr,
        On_Key_ptr,
        On_EID_ptr,
        Off_Centroids_ptr,
        Target_EID_ptr,
        Is_Vision_Mask_ptr,
        Out_On_Hit_ptr,
        Out_On_Idx_ptr,
        Out_On_Slot_ptr,
        Out_Off_Hit_ptr,
        Out_Off_Idx_ptr,
        stride_xh_b,
        stride_xh_d,
        stride_pv_h,
        stride_pv_l,
        stride_pt_h,
        stride_pt_l,
        stride_ok_b,
        stride_ok_d,
        stride_oe_b,
        stride_oe_s,
        stride_oc_e,
        stride_oc_m,
        stride_oc_k,
        stride_oc_d,
        stride_te_b,
        stride_te_s,
        stride_out_b,
        stride_out_s,
        H_DIM: tl.constexpr,
        L_DIM: tl.constexpr,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_ON: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ON_TOPK: tl.constexpr,
        ON_TOPK_PAD: tl.constexpr,
        valid_on_size,
        COL_OFFSET: tl.constexpr,
        NUM_SEARCH: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_idx = pid // NUM_SEARCH
        slot_idx = pid % NUM_SEARCH

        target_eid = tl.load(Target_EID_ptr + batch_idx * stride_te_b + (slot_idx + COL_OFFSET) * stride_te_s)
        is_vision = tl.load(Is_Vision_Mask_ptr + batch_idx).to(tl.int1)
        mod_id = tl.where(is_vision, 0, 1)

        online_hit = False
        best_on_idx = -1

        scores_on = tl.zeros([BLOCK_ON], dtype=tl.float32)
        x_ptr_base = X_High_ptr + batch_idx * stride_xh_b
        for k in range(0, H_DIM, BLOCK_H):
            offs_h = k + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H_DIM
            x_chunk = tl.load(x_ptr_base + offs_h, mask=mask_h, other=0.0)

            offs_n = tl.arange(0, BLOCK_ON)
            k_ptrs = On_Key_ptr + (offs_n[:, None] * stride_ok_b + offs_h[None, :] * stride_ok_d)
            k_chunk = tl.load(k_ptrs, mask=mask_h[None, :], other=0.0)
            scores_on += tl.sum(x_chunk[None, :] * k_chunk, axis=1)

        mask_valid = tl.arange(0, BLOCK_ON) < valid_on_size
        scores_on = tl.where(mask_valid, scores_on, -float("inf"))
        max_on_sim, max_idx = tl.max(scores_on, axis=0, return_indices=True)

        if max_on_sim > 0.0:
            offs_tk = tl.arange(0, ON_TOPK_PAD)
            mask_tk = offs_tk < ON_TOPK
            cached_eids = tl.load(
                On_EID_ptr + max_idx * stride_oe_b + offs_tk * stride_oe_s,
                mask=mask_tk,
                other=-1,
            )
            matches = cached_eids == target_eid
            if tl.sum(matches.to(tl.int32)) > 0:
                online_hit = True
                best_on_idx = max_idx
                slot_idx_in_cache = tl.argmax(matches.to(tl.int32), axis=0)
                tl.store(Out_On_Slot_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, slot_idx_in_cache)

        tl.store(Out_On_Hit_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, online_hit)
        if slot_idx == 0:
            tl.store(Out_On_Idx_ptr + batch_idx, best_on_idx)

        if online_hit:
            tl.store(Out_Off_Hit_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, False)
            tl.store(Out_Off_Idx_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, 0)
            return

        x_low_acc = tl.zeros([L_DIM], dtype=tl.float32)
        for k in range(0, H_DIM, BLOCK_H):
            offs_h = k + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H_DIM
            x_chunk = tl.load(X_High_ptr + batch_idx * stride_xh_b + offs_h, mask=mask_h, other=0.0)

            offs_l = tl.arange(0, L_DIM)
            p_vis_ptrs = Proj_Vis_ptr + (offs_h[:, None] * stride_pv_h + offs_l[None, :] * stride_pv_l)
            p_txt_ptrs = Proj_Txt_ptr + (offs_h[:, None] * stride_pt_h + offs_l[None, :] * stride_pt_l)
            p_ptrs = tl.where(is_vision, p_vis_ptrs, p_txt_ptrs)
            proj_chunk = tl.load(p_ptrs, mask=mask_h[:, None], other=0.0)
            x_low_acc += tl.sum(x_chunk[:, None] * proj_chunk, axis=0)

        offs_l = tl.arange(0, L_DIM)
        b_ptr = tl.where(is_vision, Bias_Vis_ptr + offs_l, Bias_Txt_ptr + offs_l)
        x_low = (x_low_acc + tl.load(b_ptr)).to(tl.float16)

        c_ptr_base = Off_Centroids_ptr + target_eid * stride_oc_e + mod_id * stride_oc_m
        offs_k = tl.arange(0, BLOCK_K)
        c_ptrs = c_ptr_base + (offs_k[:, None] * stride_oc_k + offs_l[None, :] * stride_oc_d)
        mask_k = offs_k[:, None] < K
        c_block = tl.load(c_ptrs, mask=mask_k, other=0.0)

        scores = tl.sum(x_low[None, :] * c_block, axis=1)
        scores = tl.where(offs_k < K, scores, -65504.0)
        max_val, max_idx = tl.max(scores, axis=0, return_indices=True)

        hit = max_val > 0.0
        tl.store(Out_Off_Hit_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, hit)
        tl.store(Out_Off_Idx_ptr + batch_idx * stride_out_b + slot_idx * stride_out_s, max_idx)


class VectorSearchEngine:
    def __init__(self, manager):
        self.manager = manager
        self.device = manager.device
        self.workspace_capacity = 0
        self.workspace = {}
        self.prefill_query_workspace_capacity = 0
        self.prefill_query_workspace = None
        self.decode_query_workspace_capacity = 0
        self.decode_query_workspace = None
        self.search_dims_cache = None

    def _get_pca_pair(self):
        pca_map = getattr(self.manager, "pca", None)
        if not isinstance(pca_map, dict) or not pca_map:
            raise ValueError("manager.pca must be a non-empty dict")

        if "vision" in pca_map and "text" in pca_map:
            return pca_map["vision"], pca_map["text"]

        first_key = next(iter(pca_map))
        shared = pca_map[first_key]
        return shared, shared

    def _get_search_dims(self):
        if self.search_dims_cache is not None:
            return self.search_dims_cache

        pca_vis, pca_txt = self._get_pca_pair()
        compressed_dim = int(pca_vis["proj"].shape[1])
        if int(pca_txt["proj"].shape[1]) != compressed_dim:
            raise ValueError("vision/text PCA projection dims must match")

        fixed_k = int(self.manager.keys_buffer.shape[2])
        online_topk = None
        if getattr(self.manager, "online_expert_ids", None) is not None:
            online_topk = int(self.manager.online_expert_ids.shape[1])
        self.search_dims_cache = {
            "pca_vis": pca_vis,
            "pca_txt": pca_txt,
            "compressed_dim": compressed_dim,
            "fixed_k": fixed_k,
            "online_topk": online_topk,
        }
        return self.search_dims_cache

    def _ensure_workspace(self, batch_size, top_k):
        if batch_size > self.workspace_capacity:
            new_cap = max(batch_size, 128)
            new_cap = ((new_cap + 127) // 128) * 128
            self.workspace["on_hit"] = torch.zeros((new_cap, top_k), dtype=torch.bool, device=self.device)
            self.workspace["off_hit"] = torch.zeros((new_cap, top_k), dtype=torch.bool, device=self.device)
            self.workspace["off_idx"] = torch.zeros((new_cap, top_k), dtype=torch.long, device=self.device)
            self.workspace["on_idx"] = torch.full((new_cap,), -1, dtype=torch.long, device=self.device)
            self.workspace["on_slot"] = torch.zeros((new_cap, top_k), dtype=torch.long, device=self.device)
            self.workspace["reuse"] = torch.zeros((new_cap, top_k), dtype=torch.bool, device=self.device)
            self.workspace_capacity = new_cap

    def _ensure_prefill_query_workspace(self, batch_size, compressed_dim):
        dtype = self.manager.keys_buffer.dtype
        needs_alloc = (
            self.prefill_query_workspace is None
            or batch_size > self.prefill_query_workspace_capacity
            or self.prefill_query_workspace.shape[1] != compressed_dim
            or self.prefill_query_workspace.dtype != dtype
        )
        if needs_alloc:
            new_cap = max(batch_size, 128)
            new_cap = ((new_cap + 127) // 128) * 128
            self.prefill_query_workspace = torch.zeros((new_cap, compressed_dim), dtype=dtype, device=self.device)
            self.prefill_query_workspace_capacity = new_cap
        return self.prefill_query_workspace[:batch_size]

    def _ensure_decode_query_workspace(self, batch_size, compressed_dim):
        dtype = self.manager.keys_buffer.dtype
        needs_alloc = (
            self.decode_query_workspace is None
            or batch_size > self.decode_query_workspace_capacity
            or self.decode_query_workspace.shape[1] != compressed_dim
            or self.decode_query_workspace.dtype != dtype
        )
        if needs_alloc:
            new_cap = max(batch_size, 128)
            new_cap = ((new_cap + 127) // 128) * 128
            self.decode_query_workspace = torch.zeros((new_cap, compressed_dim), dtype=dtype, device=self.device)
            self.decode_query_workspace_capacity = new_cap
        return self.decode_query_workspace[:batch_size]

    def _project_queries_into(self, hidden_states, is_vision_mask, search_dims, query_low, normalize):
        target_dtype = self.manager.keys_buffer.dtype
        eps = float(CacheConfig.OFFLINE_COSINE_EPS)
        vision_mask = is_vision_mask.bool()
        text_mask = ~vision_mask

        if vision_mask.any():
            pca_vis = search_dims["pca_vis"]
            x_vis_low = torch.addmm(pca_vis["bias"], hidden_states[vision_mask], pca_vis["proj"])
            if normalize:
                query_low[vision_mask] = _normalize_last_dim(x_vis_low, eps, out_dtype=target_dtype)
            else:
                query_low[vision_mask] = x_vis_low.to(dtype=target_dtype)

        if text_mask.any():
            pca_txt = search_dims["pca_txt"]
            x_txt_low = torch.addmm(pca_txt["bias"], hidden_states[text_mask], pca_txt["proj"])
            if normalize:
                query_low[text_mask] = _normalize_last_dim(x_txt_low, eps, out_dtype=target_dtype)
            else:
                query_low[text_mask] = x_txt_low.to(dtype=target_dtype)

        return query_low

    def _build_prefill_queries(self, hidden_states, is_vision_mask, search_dims):
        batch_size = hidden_states.shape[0]
        query_low = self._ensure_prefill_query_workspace(batch_size, search_dims["compressed_dim"])
        return self._project_queries_into(hidden_states, is_vision_mask, search_dims, query_low, normalize=True)

    def _build_prefill_queries_offline_dot(self, hidden_states, is_vision_mask, search_dims):
        batch_size = hidden_states.shape[0]
        query_low = self._ensure_prefill_query_workspace(batch_size, search_dims["compressed_dim"])
        return self._project_queries_into(hidden_states, is_vision_mask, search_dims, query_low, normalize=False)

    def _build_decode_queries(self, hidden_states, is_vision_mask, search_dims):
        batch_size = hidden_states.shape[0]
        query_low = self._ensure_decode_query_workspace(batch_size, search_dims["compressed_dim"])
        return self._project_queries_into(hidden_states, is_vision_mask, search_dims, query_low, normalize=False)

    def _search_prefill_offline_into(self, query_low, selected_experts, is_vision_mask, reuse_mask, out_hit_mask, out_cluster_indices):
        if not reuse_mask.any():
            return

        search_dims = self._get_search_dims()
        threshold_vision = float(CacheConfig.OFFLINE_COSINE_THRESHOLD_VISION)
        threshold_text = float(CacheConfig.OFFLINE_COSINE_THRESHOLD_TEXT)

        if HAS_TRITON:
            _, top_k = selected_experts.shape
            grid = (selected_experts.shape[0] * top_k,)
            prefill_offline_search_kernel[grid](
                Query_Low_ptr=query_low,
                Off_Centroids_ptr=self.manager.keys_buffer,
                Target_EID_ptr=selected_experts,
                Is_Vision_Mask_ptr=is_vision_mask,
                Reuse_Mask_ptr=reuse_mask,
                Out_Off_Hit_ptr=out_hit_mask,
                Out_Off_Idx_ptr=out_cluster_indices,
                stride_ql_b=query_low.stride(0),
                stride_ql_d=query_low.stride(1),
                stride_oc_e=self.manager.keys_buffer.stride(0),
                stride_oc_m=self.manager.keys_buffer.stride(1),
                stride_oc_k=self.manager.keys_buffer.stride(2),
                stride_oc_d=self.manager.keys_buffer.stride(3),
                stride_te_b=selected_experts.stride(0),
                stride_te_s=selected_experts.stride(1),
                stride_rm_b=reuse_mask.stride(0),
                stride_rm_s=reuse_mask.stride(1),
                stride_out_b=out_hit_mask.stride(0),
                stride_out_s=out_hit_mask.stride(1),
                threshold_vision=threshold_vision,
                threshold_text=threshold_text,
                L_DIM=search_dims["compressed_dim"],
                K=search_dims["fixed_k"],
                BLOCK_K=256,
                TOP_K=top_k,
            )
            return

        self._search_decode_offline_from_low_into(
            query_low=query_low,
            selected_experts=selected_experts,
            is_vision_mask=is_vision_mask,
            reuse_mask=reuse_mask,
            out_hit_mask=out_hit_mask,
            out_cluster_indices=out_cluster_indices,
            threshold_vision=threshold_vision,
            threshold_text=threshold_text,
        )

    def _search_decode_offline_from_low_into(
        self,
        query_low,
        selected_experts,
        is_vision_mask,
        reuse_mask,
        out_hit_mask,
        out_cluster_indices,
        threshold_vision=0.0,
        threshold_text=0.0,
    ):
        if not reuse_mask.any():
            return

        query_rows, query_cols = torch.where(reuse_mask)
        if query_rows.numel() == 0:
            return

        query_eids = selected_experts[query_rows, query_cols]
        query_mods = torch.where(is_vision_mask[query_rows], 0, 1)
        query_low_rows = query_low[query_rows]

        for mod_id, threshold in ((0, threshold_vision), (1, threshold_text)):
            mod_mask = query_mods == mod_id
            if not mod_mask.any():
                continue

            mod_idx = torch.where(mod_mask)[0]
            cents = self.manager.keys_buffer[query_eids[mod_idx], mod_id]
            scores = torch.bmm(cents, query_low_rows[mod_idx].unsqueeze(2)).squeeze(2)
            max_s, max_i = scores.max(dim=1)
            hits = max_s >= threshold
            if hits.any():
                rows = query_rows[mod_idx][hits]
                cols = query_cols[mod_idx][hits]
                out_hit_mask[rows, cols] = True
                out_cluster_indices[rows, cols] = max_i[hits]

    def _search_decode_offline_into(self, hidden_states, selected_experts, is_vision_mask, reuse_mask, out_hit_mask, out_cluster_indices):
        if not reuse_mask.any():
            return

        search_dims = self._get_search_dims()
        if HAS_TRITON:
            _, top_k = selected_experts.shape
            pca_vis = search_dims["pca_vis"]
            pca_txt = search_dims["pca_txt"]
            grid = (selected_experts.shape[0] * top_k,)
            decode_offline_search_kernel[grid](
                X_High_ptr=hidden_states,
                Proj_Vis_ptr=pca_vis["proj"],
                Bias_Vis_ptr=pca_vis["bias"],
                Proj_Txt_ptr=pca_txt["proj"],
                Bias_Txt_ptr=pca_txt["bias"],
                Off_Centroids_ptr=self.manager.keys_buffer,
                Target_EID_ptr=selected_experts,
                Is_Vision_Mask_ptr=is_vision_mask,
                Reuse_Mask_ptr=reuse_mask,
                Out_Off_Hit_ptr=out_hit_mask,
                Out_Off_Idx_ptr=out_cluster_indices,
                stride_xh_b=hidden_states.stride(0),
                stride_xh_d=hidden_states.stride(1),
                stride_pv_h=pca_vis["proj"].stride(0),
                stride_pv_l=pca_vis["proj"].stride(1),
                stride_pt_h=pca_txt["proj"].stride(0),
                stride_pt_l=pca_txt["proj"].stride(1),
                stride_oc_e=self.manager.keys_buffer.stride(0),
                stride_oc_m=self.manager.keys_buffer.stride(1),
                stride_oc_k=self.manager.keys_buffer.stride(2),
                stride_oc_d=self.manager.keys_buffer.stride(3),
                stride_te_b=selected_experts.stride(0),
                stride_te_s=selected_experts.stride(1),
                stride_rm_b=reuse_mask.stride(0),
                stride_rm_s=reuse_mask.stride(1),
                stride_out_b=out_hit_mask.stride(0),
                stride_out_s=out_hit_mask.stride(1),
                H_DIM=hidden_states.shape[-1],
                L_DIM=search_dims["compressed_dim"],
                K=search_dims["fixed_k"],
                BLOCK_K=256,
                BLOCK_H=128,
                TOP_K=top_k,
            )
            return

        query_low = self._build_decode_queries(hidden_states, is_vision_mask, search_dims)
        self._search_decode_offline_from_low_into(
            query_low=query_low,
            selected_experts=selected_experts,
            is_vision_mask=is_vision_mask,
            reuse_mask=reuse_mask,
            out_hit_mask=out_hit_mask,
            out_cluster_indices=out_cluster_indices,
        )

    def _search_offline_dot_from_low_into(
        self,
        query_low,
        selected_experts,
        is_vision_mask,
        reuse_mask,
        out_hit_mask,
        out_cluster_indices,
    ):
        if not reuse_mask.any():
            return

        query_rows, query_cols = torch.where(reuse_mask)
        if query_rows.numel() == 0:
            return

        query_eids = selected_experts[query_rows, query_cols]
        query_mods = torch.where(is_vision_mask[query_rows], 0, 1)
        query_low_rows = query_low[query_rows]
        threshold_dot = float(CacheConfig.OFFLINE_DOT_THRESHOLD)

        for mod_id in (0, 1):
            mod_mask = query_mods == mod_id
            if not mod_mask.any():
                continue

            mod_idx = torch.where(mod_mask)[0]
            keys = self.manager.raw_keys_buffer[query_eids[mod_idx], mod_id]
            scores = torch.bmm(keys, query_low_rows[mod_idx].unsqueeze(2)).squeeze(2)
            max_s, max_i = scores.max(dim=1)
            hits = max_s > threshold_dot

            if hits.any():
                rows = query_rows[mod_idx][hits]
                cols = query_cols[mod_idx][hits]
                out_hit_mask[rows, cols] = True
                out_cluster_indices[rows, cols] = max_i[hits]

    def search_prefill_offline_dot(self, layer_idx, hidden_states, selected_experts, is_vision_mask, reuse_mask):
        bs, top_k = selected_experts.shape
        off_hit_mask = torch.zeros((bs, top_k), dtype=torch.bool, device=self.device)
        off_cluster_indices = torch.zeros((bs, top_k), dtype=torch.long, device=self.device)

        query_low = self._build_prefill_queries_offline_dot(hidden_states, is_vision_mask, self._get_search_dims())
        self._search_offline_dot_from_low_into(
            query_low=query_low,
            selected_experts=selected_experts,
            is_vision_mask=is_vision_mask,
            reuse_mask=reuse_mask,
            out_hit_mask=off_hit_mask,
            out_cluster_indices=off_cluster_indices,
        )
        return off_hit_mask, off_cluster_indices

    def search_decode_offline_dot(self, layer_idx, hidden_states, selected_experts, is_vision_mask, reuse_mask):
        bs, top_k = selected_experts.shape
        off_hit_mask = torch.zeros((bs, top_k), dtype=torch.bool, device=self.device)
        off_cluster_indices = torch.zeros((bs, top_k), dtype=torch.long, device=self.device)

        if not reuse_mask.any():
            return off_hit_mask, off_cluster_indices

        search_dims = self._get_search_dims()
        if HAS_TRITON:
            pca_vis = search_dims["pca_vis"]
            pca_txt = search_dims["pca_txt"]
            threshold_dot = float(CacheConfig.OFFLINE_DOT_THRESHOLD)
            grid = (selected_experts.shape[0] * top_k,)
            decode_offline_dot_search_kernel[grid](
                X_High_ptr=hidden_states,
                Proj_Vis_ptr=pca_vis["proj"],
                Bias_Vis_ptr=pca_vis["bias"],
                Proj_Txt_ptr=pca_txt["proj"],
                Bias_Txt_ptr=pca_txt["bias"],
                Off_Keys_Raw_ptr=self.manager.raw_keys_buffer,
                Target_EID_ptr=selected_experts,
                Is_Vision_Mask_ptr=is_vision_mask,
                Reuse_Mask_ptr=reuse_mask,
                Out_Off_Hit_ptr=off_hit_mask,
                Out_Off_Idx_ptr=off_cluster_indices,
                stride_xh_b=hidden_states.stride(0),
                stride_xh_d=hidden_states.stride(1),
                stride_pv_h=pca_vis["proj"].stride(0),
                stride_pv_l=pca_vis["proj"].stride(1),
                stride_pt_h=pca_txt["proj"].stride(0),
                stride_pt_l=pca_txt["proj"].stride(1),
                stride_ok_e=self.manager.raw_keys_buffer.stride(0),
                stride_ok_m=self.manager.raw_keys_buffer.stride(1),
                stride_ok_k=self.manager.raw_keys_buffer.stride(2),
                stride_ok_d=self.manager.raw_keys_buffer.stride(3),
                stride_te_b=selected_experts.stride(0),
                stride_te_s=selected_experts.stride(1),
                stride_rm_b=reuse_mask.stride(0),
                stride_rm_s=reuse_mask.stride(1),
                stride_out_b=off_hit_mask.stride(0),
                stride_out_s=off_hit_mask.stride(1),
                threshold_dot=threshold_dot,
                H_DIM=hidden_states.shape[-1],
                L_DIM=search_dims["compressed_dim"],
                K=search_dims["fixed_k"],
                BLOCK_K=256,
                BLOCK_H=128,
                TOP_K=top_k,
            )
            return off_hit_mask, off_cluster_indices

        query_low = self._build_decode_queries(hidden_states, is_vision_mask, self._get_search_dims())
        self._search_offline_dot_from_low_into(
            query_low=query_low,
            selected_experts=selected_experts,
            is_vision_mask=is_vision_mask,
            reuse_mask=reuse_mask,
            out_hit_mask=off_hit_mask,
            out_cluster_indices=off_cluster_indices,
        )
        return off_hit_mask, off_cluster_indices

    def _search_decode_online_into(self, hidden_states, target_eids, out_on_hit, out_on_idx, out_on_slot):
        valid_online_size = self.manager.online_size if self.manager.online_full else self.manager.online_ptr
        has_online_cache = (
            getattr(self.manager, "online_keys", None) is not None
            and getattr(self.manager, "online_expert_ids", None) is not None
            and valid_online_size > 0
            and target_eids.shape[1] > 0
        )
        if not has_online_cache:
            return False

        online_keys = self.manager.online_keys[:valid_online_size]
        online_expert_ids = self.manager.online_expert_ids[:valid_online_size]

        online_scores = torch.matmul(hidden_states, online_keys.transpose(0, 1))
        max_on_sim, max_idx = online_scores.max(dim=1)
        valid_batches = max_on_sim > 0
        out_on_idx[valid_batches] = max_idx[valid_batches]
        if not valid_batches.any():
            return True

        cached_eids = online_expert_ids[max_idx]
        matches = target_eids.unsqueeze(-1) == cached_eids.unsqueeze(1)
        slot_hits = matches.any(dim=-1) & valid_batches.unsqueeze(1)
        out_on_hit.copy_(slot_hits)
        if slot_hits.any():
            slot_indices = torch.argmax(matches.to(torch.int32), dim=-1)
            out_on_slot[slot_hits] = slot_indices[slot_hits]
        return True

    def search_prefill_offline(self, layer_idx, hidden_states, selected_experts, is_vision_mask, reuse_mask):
        bs, top_k = selected_experts.shape
        off_hit_mask = torch.zeros((bs, top_k), dtype=torch.bool, device=self.device)
        off_cluster_indices = torch.zeros((bs, top_k), dtype=torch.long, device=self.device)

        query_low = self._build_prefill_queries(hidden_states, is_vision_mask, self._get_search_dims())
        self._search_prefill_offline_into(
            query_low=query_low,
            selected_experts=selected_experts,
            is_vision_mask=is_vision_mask,
            reuse_mask=reuse_mask,
            out_hit_mask=off_hit_mask,
            out_cluster_indices=off_cluster_indices,
        )
        return off_hit_mask, off_cluster_indices

    def search_decode_offline(self, layer_idx, hidden_states, selected_experts, is_vision_mask, reuse_mask):
        bs, top_k = selected_experts.shape
        off_hit_mask = torch.zeros((bs, top_k), dtype=torch.bool, device=self.device)
        off_cluster_indices = torch.zeros((bs, top_k), dtype=torch.long, device=self.device)

        self._search_decode_offline_into(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            is_vision_mask=is_vision_mask,
            reuse_mask=reuse_mask,
            out_hit_mask=off_hit_mask,
            out_cluster_indices=off_cluster_indices,
        )
        return off_hit_mask, off_cluster_indices

    def search_decode_hybrid(self, layer_idx, hidden_states, selected_experts, is_vision_mask, keep_k=None):
        batch_size, top_k = selected_experts.shape
        if keep_k is None:
            keep_k = CacheConfig.KEEP_K
        keep_k = max(0, min(int(keep_k), top_k))
        num_search = max(top_k - keep_k, 0)

        self._ensure_workspace(batch_size, top_k)
        full_on_hit = self.workspace["on_hit"][:batch_size, :top_k]
        full_off_hit = self.workspace["off_hit"][:batch_size, :top_k]
        full_off_idx = self.workspace["off_idx"][:batch_size, :top_k]
        full_on_idx = self.workspace["on_idx"][:batch_size]
        full_on_slot = self.workspace["on_slot"][:batch_size, :top_k]
        full_reuse = self.workspace["reuse"][:batch_size, :top_k]

        full_on_hit.zero_()
        full_off_hit.zero_()
        full_off_idx.zero_()
        full_on_idx.fill_(-1)
        full_on_slot.zero_()
        full_reuse.zero_()

        if num_search <= 0:
            return full_on_hit, full_on_idx, full_off_hit, full_off_idx, full_on_slot

        valid_online_size = self.manager.online_size if self.manager.online_full else self.manager.online_ptr
        has_online_cache = (
            getattr(self.manager, "online_keys", None) is not None
            and getattr(self.manager, "online_expert_ids", None) is not None
            and valid_online_size > 0
        )
        if HAS_TRITON and has_online_cache:
            search_dims = self._get_search_dims()
            pca_vis = search_dims["pca_vis"]
            pca_txt = search_dims["pca_txt"]
            online_topk = search_dims["online_topk"]
            online_topk_pad = _next_power_of_two(online_topk)
            grid = (batch_size * num_search,)
            decode_hybrid_search_kernel[grid](
                X_High_ptr=hidden_states,
                Proj_Vis_ptr=pca_vis["proj"],
                Bias_Vis_ptr=pca_vis["bias"],
                Proj_Txt_ptr=pca_txt["proj"],
                Bias_Txt_ptr=pca_txt["bias"],
                On_Key_ptr=self.manager.online_keys,
                On_EID_ptr=self.manager.online_expert_ids,
                Off_Centroids_ptr=self.manager.keys_buffer,
                Target_EID_ptr=selected_experts,
                Is_Vision_Mask_ptr=is_vision_mask,
                Out_On_Hit_ptr=full_on_hit[:, keep_k:],
                Out_On_Idx_ptr=full_on_idx,
                Out_On_Slot_ptr=full_on_slot[:, keep_k:],
                Out_Off_Hit_ptr=full_off_hit[:, keep_k:],
                Out_Off_Idx_ptr=full_off_idx[:, keep_k:],
                stride_xh_b=hidden_states.stride(0),
                stride_xh_d=hidden_states.stride(1),
                stride_pv_h=pca_vis["proj"].stride(0),
                stride_pv_l=pca_vis["proj"].stride(1),
                stride_pt_h=pca_txt["proj"].stride(0),
                stride_pt_l=pca_txt["proj"].stride(1),
                stride_ok_b=self.manager.online_keys.stride(0),
                stride_ok_d=self.manager.online_keys.stride(1),
                stride_oe_b=self.manager.online_expert_ids.stride(0),
                stride_oe_s=self.manager.online_expert_ids.stride(1),
                stride_oc_e=self.manager.keys_buffer.stride(0),
                stride_oc_m=self.manager.keys_buffer.stride(1),
                stride_oc_k=self.manager.keys_buffer.stride(2),
                stride_oc_d=self.manager.keys_buffer.stride(3),
                stride_te_b=selected_experts.stride(0),
                stride_te_s=selected_experts.stride(1),
                stride_out_b=full_on_hit[:, keep_k:].stride(0),
                stride_out_s=full_on_hit[:, keep_k:].stride(1),
                H_DIM=hidden_states.shape[-1],
                L_DIM=search_dims["compressed_dim"],
                K=search_dims["fixed_k"],
                BLOCK_K=256,
                BLOCK_ON=64,
                BLOCK_H=128,
                ON_TOPK=online_topk,
                ON_TOPK_PAD=online_topk_pad,
                valid_on_size=valid_online_size,
                COL_OFFSET=keep_k,
                NUM_SEARCH=num_search,
            )
            return full_on_hit, full_on_idx, full_off_hit, full_off_idx, full_on_slot

        target_tail = selected_experts[:, keep_k:]
        online_hit_tail = full_on_hit[:, keep_k:]
        online_slot_tail = full_on_slot[:, keep_k:]
        reuse_tail = full_reuse[:, keep_k:]

        has_online_cache = self._search_decode_online_into(
            hidden_states=hidden_states,
            target_eids=target_tail,
            out_on_hit=online_hit_tail,
            out_on_idx=full_on_idx,
            out_on_slot=online_slot_tail,
        )
        if has_online_cache:
            reuse_tail.copy_(~online_hit_tail)
        else:
            reuse_tail.fill_(True)

        if reuse_tail.any():
            search_dims = self._get_search_dims()
            query_low = self._build_decode_queries(hidden_states, is_vision_mask, search_dims)
            self._search_decode_offline_from_low_into(
                query_low=query_low,
                selected_experts=selected_experts,
                is_vision_mask=is_vision_mask,
                reuse_mask=full_reuse,
                out_hit_mask=full_off_hit,
                out_cluster_indices=full_off_idx,
            )

        return full_on_hit, full_on_idx, full_off_hit, full_off_idx, full_on_slot

    def search_decode_offline_bs1_into(
        self,
        hidden_state,
        selected_experts_row,
        is_vision,
        keep_k,
        out_off_hit,
        out_off_idx,
    ):
        top_k = int(selected_experts_row.shape[0])
        out_off_hit.zero_()
        out_off_idx.zero_()

        if top_k == 0 or keep_k >= top_k:
            return

        self._ensure_workspace(1, top_k)
        reuse_mask = self.workspace["reuse"][:1, :top_k]
        reuse_mask.zero_()
        reuse_mask[:, keep_k:] = True

        off_hit_mask, off_cluster_indices = self.search_decode_offline_dot(
            layer_idx=None,
            hidden_states=hidden_state.view(1, -1),
            selected_experts=selected_experts_row.view(1, top_k),
            is_vision_mask=is_vision.view(1),
            reuse_mask=reuse_mask,
        )
        out_off_hit.copy_(off_hit_mask.view(-1))
        out_off_idx.copy_(off_cluster_indices.view(-1))

    def search_decode_hybrid_bs1_into(
        self,
        hidden_state,
        selected_experts_row,
        is_vision,
        keep_k,
        out_on_hit,
        out_on_idx,
        out_off_hit,
        out_off_idx,
        out_on_slot,
    ):
        top_k = int(selected_experts_row.shape[0])
        out_on_hit.zero_()
        out_on_idx.fill_(-1)
        out_off_hit.zero_()
        out_off_idx.zero_()
        out_on_slot.zero_()

        num_search = max(top_k - keep_k, 0)
        if num_search <= 0:
            return

        hidden_states = hidden_state.view(1, -1)
        selected_experts = selected_experts_row.view(1, top_k)
        is_vision_mask = is_vision.view(1)

        valid_online_size = self.manager.online_size if self.manager.online_full else self.manager.online_ptr
        has_online_cache = (
            getattr(self.manager, "online_keys", None) is not None
            and getattr(self.manager, "online_expert_ids", None) is not None
            and valid_online_size > 0
        )
        if HAS_TRITON and has_online_cache:
            search_dims = self._get_search_dims()
            pca_vis = search_dims["pca_vis"]
            pca_txt = search_dims["pca_txt"]
            online_topk = search_dims["online_topk"]
            online_topk_pad = _next_power_of_two(online_topk)
            out_on_hit_tail = out_on_hit[keep_k:].view(1, num_search)
            out_on_slot_tail = out_on_slot[keep_k:].view(1, num_search)
            out_off_hit_tail = out_off_hit[keep_k:].view(1, num_search)
            out_off_idx_tail = out_off_idx[keep_k:].view(1, num_search)
            decode_hybrid_search_kernel[(num_search,)](
                X_High_ptr=hidden_states,
                Proj_Vis_ptr=pca_vis["proj"],
                Bias_Vis_ptr=pca_vis["bias"],
                Proj_Txt_ptr=pca_txt["proj"],
                Bias_Txt_ptr=pca_txt["bias"],
                On_Key_ptr=self.manager.online_keys,
                On_EID_ptr=self.manager.online_expert_ids,
                Off_Centroids_ptr=self.manager.keys_buffer,
                Target_EID_ptr=selected_experts,
                Is_Vision_Mask_ptr=is_vision_mask,
                Out_On_Hit_ptr=out_on_hit_tail,
                Out_On_Idx_ptr=out_on_idx,
                Out_On_Slot_ptr=out_on_slot_tail,
                Out_Off_Hit_ptr=out_off_hit_tail,
                Out_Off_Idx_ptr=out_off_idx_tail,
                stride_xh_b=hidden_states.stride(0),
                stride_xh_d=hidden_states.stride(1),
                stride_pv_h=pca_vis["proj"].stride(0),
                stride_pv_l=pca_vis["proj"].stride(1),
                stride_pt_h=pca_txt["proj"].stride(0),
                stride_pt_l=pca_txt["proj"].stride(1),
                stride_ok_b=self.manager.online_keys.stride(0),
                stride_ok_d=self.manager.online_keys.stride(1),
                stride_oe_b=self.manager.online_expert_ids.stride(0),
                stride_oe_s=self.manager.online_expert_ids.stride(1),
                stride_oc_e=self.manager.keys_buffer.stride(0),
                stride_oc_m=self.manager.keys_buffer.stride(1),
                stride_oc_k=self.manager.keys_buffer.stride(2),
                stride_oc_d=self.manager.keys_buffer.stride(3),
                stride_te_b=selected_experts.stride(0),
                stride_te_s=selected_experts.stride(1),
                stride_out_b=out_on_hit_tail.stride(0),
                stride_out_s=out_on_hit_tail.stride(1),
                H_DIM=hidden_states.shape[-1],
                L_DIM=search_dims["compressed_dim"],
                K=search_dims["fixed_k"],
                BLOCK_K=256,
                BLOCK_ON=64,
                BLOCK_H=128,
                ON_TOPK=online_topk,
                ON_TOPK_PAD=online_topk_pad,
                valid_on_size=valid_online_size,
                COL_OFFSET=keep_k,
                NUM_SEARCH=num_search,
            )
            return

        self._ensure_workspace(1, top_k)
        reuse_mask = self.workspace["reuse"][:1, :top_k]
        reuse_mask.zero_()
        reuse_tail = reuse_mask[:, keep_k:]
        out_on_hit_tail = out_on_hit[keep_k:].view(1, num_search)
        out_on_slot_tail = out_on_slot[keep_k:].view(1, num_search)

        has_online_cache = self._search_decode_online_into(
            hidden_states=hidden_states,
            target_eids=selected_experts[:, keep_k:],
            out_on_hit=out_on_hit_tail,
            out_on_idx=out_on_idx.view(1),
            out_on_slot=out_on_slot_tail,
        )
        if has_online_cache:
            reuse_tail.copy_(~out_on_hit_tail)
        else:
            reuse_tail.fill_(True)

        if reuse_tail.any():
            self._search_decode_offline_into(
                hidden_states=hidden_states,
                selected_experts=selected_experts,
                is_vision_mask=is_vision_mask,
                reuse_mask=reuse_mask,
                out_hit_mask=out_off_hit.view(1, top_k),
                out_cluster_indices=out_off_idx.view(1, top_k),
            )
