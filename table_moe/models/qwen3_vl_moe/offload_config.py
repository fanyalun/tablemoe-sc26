import os
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OFFLINE_ROOT = REPO_ROOT / "offline_table" / "qwen_fp16"
DEFAULT_BENCHMARK = "MMBench_DEV_EN_V11"
SUPPORTED_CACHE_RATIOS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
DEFAULT_CACHE_RATIO = 0.5


QWEN_CACHE_SIZE_PRESETS = {
    0.1: [11] * 6 + [10] * 42,
    0.25: [30] * 16 + [29] * 32,
    0.5: [62] * 16 + [61] * 32,
    0.75: [94] * 16 + [93] * 32,
    0.9: [113] * 25 + [112] * 23,
    1.0: [128] * 48,
}


OFFLOAD_CONFIG = {
    "cache_ratio": DEFAULT_CACHE_RATIO,
    "cache_size_per_layer": list(QWEN_CACHE_SIZE_PRESETS[DEFAULT_CACHE_RATIO]),
    "buffer_size": 128,
    "prefetch": False,
    "prefetch_limit_prefill": 2,
    "prefetch_limit_decode": 0,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "model_dtype": torch.float16,
    "pca_dir": str(DEFAULT_OFFLINE_ROOT / "clustering_results" / f"{DEFAULT_BENCHMARK}_LayerPCA_256"),
    "cache_dir": str(DEFAULT_OFFLINE_ROOT / "offline_table" / f"{DEFAULT_BENCHMARK}_LayerPCA_256"),
    "compressed_dim": 64,
    "fixed_k": 256,
    "keep_k": None,
    "skip_keep_k": 5,
    "decode_skip_keep_k": 4,
    "keep_rate": 0.6,
    "modality_map": {"vision": 0, "text": 1},
    "online_cache_size": 64,
    "online_top_k": None,
    "online_min_layer_idx": 0,
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
    return list(QWEN_CACHE_SIZE_PRESETS[_normalize_cache_ratio(cache_ratio)])


def _parse_env_bool(name: str):
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean env for {name}: {raw!r}")


def _parse_env_int(name: str):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def _parse_env_float(name: str):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return float(raw)


def _parse_env_str(name: str):
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if value == "":
        return None
    return value


def _apply_env_overrides(config: dict) -> dict:
    overridden = config.copy()

    prefetch = _parse_env_bool("TABLEMOE_QWEN_PREFETCH")
    if prefetch is not None:
        overridden["prefetch"] = prefetch

    prefill_limit = _parse_env_int("TABLEMOE_QWEN_PREFETCH_LIMIT_PREFILL")
    if prefill_limit is not None:
        overridden["prefetch_limit_prefill"] = prefill_limit

    decode_limit = _parse_env_int("TABLEMOE_QWEN_PREFETCH_LIMIT_DECODE")
    if decode_limit is not None:
        overridden["prefetch_limit_decode"] = decode_limit

    fixed_k = _parse_env_int("TABLEMOE_QWEN_FIXED_K")
    if fixed_k is not None:
        overridden["fixed_k"] = fixed_k

    keep_rate = _parse_env_float("TABLEMOE_QWEN_KEEP_RATE")
    if keep_rate is not None:
        overridden["keep_rate"] = keep_rate

    recomp_ratio = _parse_env_float("TABLEMOE_QWEN_RECOMP_RATIO")
    if recomp_ratio is not None:
        overridden["keep_rate"] = recomp_ratio

    cache_ratio = _parse_env_float("TABLEMOE_QWEN_CACHE_RATIO")
    if cache_ratio is not None:
        overridden["cache_ratio"] = _normalize_cache_ratio(cache_ratio)
        overridden["cache_size_per_layer"] = _cache_size_from_ratio(cache_ratio)

    pca_dir = _parse_env_str("TABLEMOE_QWEN_PCA_DIR")
    if pca_dir is not None:
        overridden["pca_dir"] = pca_dir

    cache_dir = _parse_env_str("TABLEMOE_QWEN_CACHE_DIR")
    if cache_dir is not None:
        overridden["cache_dir"] = cache_dir

    return overridden


def update_offload_config(**kwargs):
    cache_ratio = kwargs.pop("cache_ratio", None)
    recomp_ratio = kwargs.pop("recomp_ratio", None)
    if recomp_ratio is not None and "keep_rate" not in kwargs:
        kwargs["keep_rate"] = recomp_ratio
    if cache_ratio is not None:
        normalized_ratio = _normalize_cache_ratio(cache_ratio)
        OFFLOAD_CONFIG["cache_ratio"] = normalized_ratio
        OFFLOAD_CONFIG["cache_size_per_layer"] = _cache_size_from_ratio(normalized_ratio)
    OFFLOAD_CONFIG.update(kwargs)


def get_offload_config():
    return _apply_env_overrides(OFFLOAD_CONFIG)


def get_hybrid_cache_config(num_layers=None, top_k=None):
    config = get_offload_config()
    keep_k = config["keep_k"]
    if keep_k is None:
        keep_k = max(1, top_k // 2) if top_k is not None else 3
    prefill_keep_k = config["skip_keep_k"]
    decode_keep_k = config["decode_skip_keep_k"]

    online_top_k = config["online_top_k"]
    if online_top_k is None:
        online_top_k = top_k if top_k is not None else 6

    online_max_layer_idx = config["online_max_layer_idx"]
    if online_max_layer_idx is None:
        online_max_layer_idx = (num_layers - 1) if num_layers is not None else 47

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
