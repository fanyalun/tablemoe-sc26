from .adapter import Qwen3VLMoeAdapter


def build_offload_model_baseline(*args, **kwargs):
    from .builder import build_offload_model_baseline as impl

    return impl(*args, **kwargs)


def build_offload_model_skip_offload(*args, **kwargs):
    from .builder import build_offload_model_skip_offload as impl

    return impl(*args, **kwargs)


def build_offload_model_hybrid(*args, **kwargs):
    from .builder import build_offload_model_hybrid as impl

    return impl(*args, **kwargs)


def build_offload_model_offline(*args, **kwargs):
    from .builder import build_offload_model_offline as impl

    return impl(*args, **kwargs)


def build_offload_model_online(*args, **kwargs):
    from .builder import build_offload_model_online as impl

    return impl(*args, **kwargs)


def build_full_model_skip(*args, **kwargs):
    from .builder import build_full_model_skip as impl

    return impl(*args, **kwargs)


def __getattr__(name):
    if name in {"OFFLOAD_CONFIG", "get_hybrid_cache_config", "get_offload_config", "update_offload_config"}:
        from . import offload_config

        return getattr(offload_config, name)
    raise AttributeError(name)

__all__ = [
    "OFFLOAD_CONFIG",
    "Qwen3VLMoeAdapter",
    "build_full_model_skip",
    "build_offload_model_baseline",
    "build_offload_model_skip_offload",
    "build_offload_model_hybrid",
    "build_offload_model_offline",
    "build_offload_model_online",
    "get_hybrid_cache_config",
    "get_offload_config",
    "update_offload_config",
]
