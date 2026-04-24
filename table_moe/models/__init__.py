from .deepseek_vl2.adapter import DeepSeekV2Adapter
from .qwen3_vl_moe.adapter import Qwen3VLMoeAdapter


_ADAPTERS = {
    "qwen3_vl_moe": Qwen3VLMoeAdapter(),
    "deepseek_vl2": DeepSeekV2Adapter(),
}


def get_model_adapter(model_family: str):
    if model_family not in _ADAPTERS:
        raise ValueError(f"Unsupported model_family: {model_family}, expected one of {sorted(_ADAPTERS)}")
    return _ADAPTERS[model_family]


def list_supported_models():
    return sorted(_ADAPTERS)
