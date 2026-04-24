from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OFFLINE_ROOT = REPO_ROOT / "offline_table" / "ds_fp16"
DEFAULT_BENCHMARK = "MMBench_DEV_EN_V11"
SUPPORTED_CACHE_RATIOS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
DEFAULT_CACHE_RATIO = 0.5


DEEPSEEK_CACHE_SIZE_PRESETS = {
    0.1: [0] + [8] * 15 + [7] * 14,
    0.25: [0] + [18] * 29,
    0.5: [0] + [34] * 15 + [33] * 14,
    0.75: [0] + [54] * 29,
    0.9: [0] + [65] * 26 + [64] * 3,
    1.0: [0] + [72] * 29,
}


OFFLOAD_CONFIG = {
    "cache_size_per_layer": [4] * 48,
    "buffer_size": 128,
    "prefetch": False,
    "prefetch_limit_prefill": 2,
    "prefetch_limit_decode": 0,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "model_dtype": torch.float16,
}


DEEPSEEK_OFFLOAD_CONFIG = {
    "cache_ratio": DEFAULT_CACHE_RATIO,
    "cache_size_per_layer": list(DEEPSEEK_CACHE_SIZE_PRESETS[DEFAULT_CACHE_RATIO]),
    "buffer_size": 72,
    "prefetch": False,
    "prefetch_limit_prefill": 2,
    "prefetch_limit_decode": 0,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "model_dtype": torch.float16,
    "num_routed_experts": 72,
    "num_shared_experts": 2,
    "routed_scaling_factor": 2.0,
    "first_k_dense_replace": 1,
    "pca_dir": str(DEFAULT_OFFLINE_ROOT / "clustering_results" / f"{DEFAULT_BENCHMARK}_LayerPCA_256"),
    "cache_dir": str(DEFAULT_OFFLINE_ROOT / "offline_table" / f"{DEFAULT_BENCHMARK}_LayerPCA_256"),
    "compressed_dim": 64,
    "fixed_k": 256,
    "keep_k": 3,
    "skip_keep_k": 5,
    "decode_skip_keep_k": 4,
    "keep_rate": 0.6,
    "modality_map": {"vision": 0, "text": 1},
    "online_cache_size": 64,
    "online_top_k": 6,
    "online_min_layer_idx": 1,
    "online_max_layer_idx": None,
}


def _normalize_cache_ratio(value):
    if value is None:
        return None
    normalized = round(float(value), 2)
    if normalized not in SUPPORTED_CACHE_RATIOS:
        raise ValueError(
            f"Unsupported cache_ratio: {value}. Expected one of {SUPPORTED_CACHE_RATIOS}"
        )
    return normalized


def _cache_size_from_ratio(cache_ratio: float):
    return list(DEEPSEEK_CACHE_SIZE_PRESETS[_normalize_cache_ratio(cache_ratio)])


def update_offload_config(**kwargs):
    cache_ratio = kwargs.pop("cache_ratio", None)
    recomp_ratio = kwargs.pop("recomp_ratio", None)
    if recomp_ratio is not None and "keep_rate" not in kwargs:
        kwargs["keep_rate"] = recomp_ratio
    if cache_ratio is not None:
        normalized_ratio = _normalize_cache_ratio(cache_ratio)
        DEEPSEEK_OFFLOAD_CONFIG["cache_ratio"] = normalized_ratio
        DEEPSEEK_OFFLOAD_CONFIG["cache_size_per_layer"] = _cache_size_from_ratio(normalized_ratio)
    OFFLOAD_CONFIG.update(kwargs)
    DEEPSEEK_OFFLOAD_CONFIG.update(kwargs)


def update_deepseek_offload_config(**kwargs):
    cache_ratio = kwargs.pop("cache_ratio", None)
    recomp_ratio = kwargs.pop("recomp_ratio", None)
    if recomp_ratio is not None and "keep_rate" not in kwargs:
        kwargs["keep_rate"] = recomp_ratio
    if cache_ratio is not None:
        normalized_ratio = _normalize_cache_ratio(cache_ratio)
        DEEPSEEK_OFFLOAD_CONFIG["cache_ratio"] = normalized_ratio
        DEEPSEEK_OFFLOAD_CONFIG["cache_size_per_layer"] = _cache_size_from_ratio(normalized_ratio)
    DEEPSEEK_OFFLOAD_CONFIG.update(kwargs)


def get_offload_config():
    return OFFLOAD_CONFIG.copy()


def get_deepseek_offload_config():
    return DEEPSEEK_OFFLOAD_CONFIG.copy()


def get_deepseek_cache_config(num_layers=None, top_k=None):
    config = get_deepseek_offload_config()
    online_max_layer_idx = config["online_max_layer_idx"]
    if online_max_layer_idx is None:
        online_max_layer_idx = (num_layers - 1) if num_layers is not None else 29

    online_top_k = config["online_top_k"]
    if online_top_k is None:
        online_top_k = top_k if top_k is not None else 6

    keep_k = config["keep_k"]
    if keep_k is None:
        keep_k = max(1, top_k // 2) if top_k is not None else 3
    prefill_keep_k = config["skip_keep_k"]
    decode_keep_k = config["decode_skip_keep_k"]

    return {
        "PCA_DIR": config["pca_dir"],
        "CACHE_DIR": config["cache_dir"],
        "COMPRESSED_DIM": config["compressed_dim"],
        "FIXED_K": config["fixed_k"],
        "KEEP_K": keep_k,
        "PREFILL_KEEP_K": prefill_keep_k,
        "DECODE_KEEP_K": decode_keep_k,
        "KEEP_RATE": config["keep_rate"],
        "MODALITY_MAP": config["modality_map"],
        "ONLINE_CACHE_SIZE": config["online_cache_size"],
        "ONLINE_TOP_K": online_top_k,
        "ONLINE_MIN_LAYER_IDX": config["online_min_layer_idx"],
        "ONLINE_MAX_LAYER_IDX": online_max_layer_idx,
        "MODEL_DTYPE": config["model_dtype"],
    }
