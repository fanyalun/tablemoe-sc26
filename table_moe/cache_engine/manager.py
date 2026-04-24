import glob
import os

import torch

from .config import CacheConfig


def _resolve_cache_dtype():
    dtype = getattr(CacheConfig, "MODEL_DTYPE", torch.bfloat16)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        normalized = dtype.removeprefix("torch.").lower()
        if normalized == "float16":
            return torch.float16
        if normalized == "bfloat16":
            return torch.bfloat16
        if normalized == "float32":
            return torch.float32
    raise ValueError(f"Unsupported cache dtype: {dtype!r}")


def _normalize_last_dim(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.float()
    denom = torch.linalg.vector_norm(x_fp32, dim=-1, keepdim=True).clamp_min(eps)
    normalized = x_fp32 / denom
    zero_mask = torch.linalg.vector_norm(x_fp32, dim=-1, keepdim=True) <= eps
    if zero_mask.any():
        normalized = normalized.masked_fill(zero_mask, 0.0)
    return normalized.to(dtype=x.dtype)


class HybridStorageManager:
    def __init__(self, layer_idx, num_experts, hidden_dim, device="cuda:0"):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = _resolve_cache_dtype()

        self.pca = self._load_pca()

        self.keys_buffer = torch.zeros(
            (num_experts, 2, CacheConfig.FIXED_K, CacheConfig.COMPRESSED_DIM),
            dtype=self.dtype,
            device=device,
        )
        self.raw_keys_buffer = torch.zeros(
            (num_experts, 2, CacheConfig.FIXED_K, CacheConfig.COMPRESSED_DIM),
            dtype=self.dtype,
            device=device,
        )
        self.values_buffer = torch.empty(
            (num_experts, 2, CacheConfig.FIXED_K, hidden_dim),
            dtype=self.dtype,
            pin_memory=True,
        )

        self._load_cache_data()

        self.enable_online = (
            CacheConfig.ONLINE_MIN_LAYER_IDX <= layer_idx <= CacheConfig.ONLINE_MAX_LAYER_IDX
        )
        self.online_ptr = 0
        self.online_size = 0
        self.online_full = False
        self.online_topk = 0

        if self.enable_online:
            self.online_size = CacheConfig.ONLINE_CACHE_SIZE
            self.online_topk = CacheConfig.ONLINE_TOP_K

            self.online_keys = torch.zeros(
                (self.online_size, self.hidden_dim), dtype=self.dtype, device=device
            )
            self.online_values = torch.zeros(
                (self.online_size, self.online_topk, hidden_dim), dtype=self.dtype, device=device
            )
            self.online_expert_ids = torch.full(
                (self.online_size, self.online_topk), -1, dtype=torch.long, device=device
            )
        else:
            self.online_keys = None
            self.online_values = None
            self.online_expert_ids = None

    @torch.no_grad()
    def update_online_cache_bs1(self, hidden_state, expert_ids, expert_outputs, valid_slot_mask=None):
        if not self.enable_online:
            return

        slot_idx = self.online_ptr
        self.online_keys[slot_idx].copy_(hidden_state)
        self.online_expert_ids[slot_idx].copy_(expert_ids)
        if valid_slot_mask is not None:
            self.online_expert_ids[slot_idx][~valid_slot_mask] = -1

        self.online_values[slot_idx].copy_(expert_outputs)
        if valid_slot_mask is not None:
            self.online_values[slot_idx][~valid_slot_mask] = 0

        self.online_ptr = (slot_idx + 1) % self.online_size
        if self.online_ptr == 0:
            self.online_full = True

    @torch.no_grad()
    def update_online_cache(self, hidden_states, expert_ids, expert_outputs, valid_slot_mask=None):
        if not self.enable_online:
            return

        batch_size = hidden_states.shape[0]
        if batch_size == 0:
            return

        if valid_slot_mask is not None:
            expert_ids = expert_ids.clone()
            expert_outputs = expert_outputs.clone()
            invalid_mask = ~valid_slot_mask
            expert_ids[invalid_mask] = -1
            expert_outputs[invalid_mask] = 0

        if batch_size > self.online_size:
            hidden_states = hidden_states[-self.online_size:]
            expert_ids = expert_ids[-self.online_size:]
            expert_outputs = expert_outputs[-self.online_size:]
            if valid_slot_mask is not None:
                valid_slot_mask = valid_slot_mask[-self.online_size:]
            batch_size = self.online_size

        start = self.online_ptr
        end = start + batch_size

        if end <= self.online_size:
            self.online_keys[start:end] = hidden_states
            self.online_expert_ids[start:end] = expert_ids
            self.online_values[start:end] = expert_outputs
            self.online_ptr = end % self.online_size
        else:
            first_chunk = self.online_size - start
            second_chunk = batch_size - first_chunk

            self.online_keys[start:] = hidden_states[:first_chunk]
            self.online_expert_ids[start:] = expert_ids[:first_chunk]
            self.online_values[start:] = expert_outputs[:first_chunk]

            self.online_keys[:second_chunk] = hidden_states[first_chunk:]
            self.online_expert_ids[:second_chunk] = expert_ids[first_chunk:]
            self.online_values[:second_chunk] = expert_outputs[first_chunk:]

            self.online_ptr = second_chunk
            self.online_full = True

        if self.online_ptr == 0:
            self.online_full = True

    def _load_pca(self):
        pca_path = os.path.join(CacheConfig.PCA_DIR, f"layer_{self.layer_idx}")
        pca_dict = {}
        for mod in ["vision", "text"]:
            fpath = os.path.join(pca_path, f"L{self.layer_idx}_{mod}_pca.pt")
            if os.path.exists(fpath):
                data = torch.load(fpath, map_location=self.device)
                mean = data["mean"].to(device=self.device, dtype=self.dtype)
                proj = data["proj"].to(device=self.device, dtype=self.dtype)
                bias = -torch.matmul(mean, proj)
                pca_dict[mod] = {"proj": proj, "bias": bias}
            else:
                pca_dict[mod] = {
                    "proj": torch.zeros(
                        (self.hidden_dim, CacheConfig.COMPRESSED_DIM),
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    "bias": torch.zeros(
                        CacheConfig.COMPRESSED_DIM,
                        device=self.device,
                        dtype=self.dtype,
                    ),
                }
        return pca_dict

    def _load_cache_data(self):
        layer_dir = os.path.join(CacheConfig.CACHE_DIR, f"layer_{self.layer_idx}")
        files = glob.glob(os.path.join(layer_dir, "*_cache.pt"))
        if not files:
            return

        for fpath in files:
            try:
                fname = os.path.basename(fpath)
                parts = fname.replace(".pt", "").split("_")
                eid = int(parts[1][1:])
                modality_str = parts[2]
                mid = CacheConfig.MODALITY_MAP.get(modality_str, -1)

                if mid == -1 or eid >= self.num_experts:
                    continue

                data = torch.load(fpath, map_location="cpu")
                keys = data["key"].to(dtype=self.dtype)
                normalized_keys = _normalize_last_dim(keys, float(CacheConfig.OFFLINE_COSINE_EPS))
                values = data["value"].to(dtype=self.dtype)

                actual_k = keys.shape[0]
                self.raw_keys_buffer[eid, mid, :actual_k] = keys.to(device=self.device, dtype=self.dtype)
                self.keys_buffer[eid, mid, :actual_k] = normalized_keys.to(device=self.device, dtype=self.dtype)
                self.values_buffer[eid, mid, :actual_k] = values
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

    def gather_values(self, expert_ids, modality_ids, cluster_ids):
        return self.values_buffer[expert_ids, modality_ids, cluster_ids]

    @torch.no_grad()
    def gather_values_decode_bs1(self, expert_ids, modality_id, cluster_ids, slot_ids, out_buffer):
        count = int(expert_ids.numel())
        if count == 0:
            return

        for idx in range(count):
            slot = int(slot_ids[idx])
            expert_id = int(expert_ids[idx])
            cluster_id = int(cluster_ids[idx])
            out_buffer[slot].copy_(
                self.values_buffer[expert_id, modality_id, cluster_id],
                non_blocking=True,
            )
