from .models import get_model_adapter, list_supported_models


def build_model(model_family: str, mode: str, model_id: str, **kwargs):
    adapter = get_model_adapter(model_family)
    normalized_mode = str(mode).strip().lower().replace("-", "_")

    if normalized_mode == "adapmoe":
        return adapter.build_baseline(model_id=model_id, **kwargs)
    if normalized_mode == "skip":
        return adapter.build_skip_offload(model_id=model_id, **kwargs)
    if normalized_mode == "tablemoe":
        return adapter.build_hybrid(model_id=model_id, **kwargs)
    if normalized_mode == "offline":
        return adapter.build_offline(model_id=model_id, **kwargs)
    if normalized_mode == "online":
        return adapter.build_online(model_id=model_id, **kwargs)
    if normalized_mode == "full_skip":
        return adapter.build_skip(model_id=model_id, **kwargs)
    raise ValueError(
        f"Unsupported mode: {mode}, expected one of ['adapmoe', 'skip', 'offline', 'online', 'tablemoe']"
    )
