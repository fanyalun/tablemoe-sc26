import os
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFLINE_ROOT = REPO_ROOT / "offline_table" / "ds_fp16"
DEFAULT_BENCHMARK = "MMBench_DEV_EN_V11"


def _default_cache_config():
    return {
        "PCA_DIR": os.getenv(
            "DS_CACHE_PCA_DIR",
            str(DEFAULT_OFFLINE_ROOT / "clustering_results" / f"{DEFAULT_BENCHMARK}_LayerPCA_256"),
        ),
        "CACHE_DIR": os.getenv(
            "DS_CACHE_DIR",
            str(DEFAULT_OFFLINE_ROOT / "offline_table" / f"{DEFAULT_BENCHMARK}_LayerPCA_256"),
        ),
        "COMPRESSED_DIM": 64,
        "FIXED_K": 256,
        "KEEP_K": 3,
        "PREFILL_KEEP_K": 3,
        "DECODE_KEEP_K": 3,
        "KEEP_RATE": 0.6,
        "MODALITY_MAP": {"vision": 0, "text": 1},
        "ONLINE_CACHE_SIZE": 64,
        "ONLINE_TOP_K": 6,
        "ONLINE_MIN_LAYER_IDX": 1,
        "ONLINE_MAX_LAYER_IDX": 47,
        "MODEL_DTYPE": torch.bfloat16,
        "OFFLINE_COSINE_THRESHOLD_VISION": 0.6,
        "OFFLINE_COSINE_THRESHOLD_TEXT": 0.7,
        "OFFLINE_DOT_THRESHOLD": 0.0,
        "OFFLINE_COSINE_EPS": 1e-6,
    }


_ACTIVE_CACHE_CONFIG = _default_cache_config()


def set_active_cache_config(config):
    global _ACTIVE_CACHE_CONFIG
    merged = _default_cache_config()
    if config is not None:
        merged.update(config)
    _ACTIVE_CACHE_CONFIG = merged


def update_active_cache_config(**kwargs):
    _ACTIVE_CACHE_CONFIG.update(kwargs)


def get_active_cache_config():
    return _ACTIVE_CACHE_CONFIG.copy()


class _CacheConfigMeta(type):
    def __getattr__(cls, name):
        if name in _ACTIVE_CACHE_CONFIG:
            return _ACTIVE_CACHE_CONFIG[name]
        raise AttributeError(name)

    def __setattr__(cls, name, value):
        if name in _ACTIVE_CACHE_CONFIG:
            _ACTIVE_CACHE_CONFIG[name] = value
            return
        super().__setattr__(name, value)


class CacheConfig(metaclass=_CacheConfigMeta):
    pass
