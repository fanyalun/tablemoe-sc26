from .adapter import DeepSeekV2Adapter


def build_offload_model_deepseek_baseline(*args, **kwargs):
    from .builder import build_offload_model_deepseek_baseline as impl

    return impl(*args, **kwargs)


def build_offload_model_deepseek_skip_offload(*args, **kwargs):
    from .builder import build_offload_model_deepseek_skip_offload as impl

    return impl(*args, **kwargs)


def build_offload_model_deepseek_hybrid(*args, **kwargs):
    from .builder import build_offload_model_deepseek_hybrid as impl

    return impl(*args, **kwargs)


def build_offload_model_deepseek_offline(*args, **kwargs):
    from .builder import build_offload_model_deepseek_offline as impl

    return impl(*args, **kwargs)


def build_offload_model_deepseek_online(*args, **kwargs):
    from .builder import build_offload_model_deepseek_online as impl

    return impl(*args, **kwargs)


def build_full_model_deepseek_skip(*args, **kwargs):
    from .builder import build_full_model_deepseek_skip as impl

    return impl(*args, **kwargs)


def __getattr__(name):
    if name in {
        "DEEPSEEK_OFFLOAD_CONFIG",
        "OFFLOAD_CONFIG",
        "get_deepseek_cache_config",
        "get_deepseek_offload_config",
        "get_offload_config",
        "update_deepseek_offload_config",
        "update_offload_config",
    }:
        from . import offload_config

        return getattr(offload_config, name)
    raise AttributeError(name)

__all__ = [
    "DEEPSEEK_OFFLOAD_CONFIG",
    "DeepSeekV2Adapter",
    "OFFLOAD_CONFIG",
    "build_full_model_deepseek_skip",
    "build_offload_model_deepseek_baseline",
    "build_offload_model_deepseek_skip_offload",
    "build_offload_model_deepseek_hybrid",
    "build_offload_model_deepseek_offline",
    "build_offload_model_deepseek_online",
    "get_deepseek_cache_config",
    "get_deepseek_offload_config",
    "get_offload_config",
    "update_deepseek_offload_config",
    "update_offload_config",
]
