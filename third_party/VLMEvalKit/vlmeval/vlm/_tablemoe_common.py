from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def ensure_repo_root_on_path() -> Path:
    repo_root = get_repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def get_lmu_data_root() -> Path:
    return get_repo_root() / "LMUData"


def configure_lmu_data_root(lmu_data_root: str | Path | None = None) -> str:
    root = Path(lmu_data_root) if lmu_data_root is not None else get_lmu_data_root()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["LMUData"] = str(root)
    return os.environ["LMUData"]


def configure_tablemoe_cache_paths(
    pca_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, str]:
    overrides = {}

    if pca_dir is not None:
        resolved_pca_dir = str(Path(pca_dir))
        os.environ["DS_CACHE_PCA_DIR"] = resolved_pca_dir
        overrides["pca_dir"] = resolved_pca_dir

    if cache_dir is not None:
        resolved_cache_dir = str(Path(cache_dir))
        os.environ["DS_CACHE_DIR"] = resolved_cache_dir
        overrides["cache_dir"] = resolved_cache_dir

    return overrides


def ensure_deepseek_vl2_repo():
    repo_root = ensure_repo_root_on_path()
    deepseek_repo = repo_root / "third_party" / "DeepSeek-VL2"
    deepseek_repo_str = str(deepseek_repo)
    if deepseek_repo_str not in sys.path:
        sys.path.insert(0, deepseek_repo_str)

    try:
        from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
        from deepseek_vl2.utils.io import load_pil_images
    except ImportError as exc:
        raise RuntimeError(f"Failed to import deepseek_vl2 from {deepseek_repo}") from exc

    return DeepseekVLV2Processor, DeepseekVLV2ForCausalLM, load_pil_images


def resolve_torch_dtype(torch_dtype):
    if torch_dtype is None or torch_dtype == "auto":
        return None
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if isinstance(torch_dtype, str):
        normalized = torch_dtype.lower()
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if normalized in mapping:
            return mapping[normalized]
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def get_model_device(model) -> torch.device:
    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    if isinstance(model_device, str):
        return torch.device(model_device)

    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    raise RuntimeError("Unable to infer model device")


def cast_floating_tensors(value, float_dtype):
    if float_dtype is None:
        return value
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(dtype=float_dtype)
        return value
    if hasattr(value, "keys") and hasattr(value, "__getitem__") and hasattr(value, "__setitem__"):
        for key in list(value.keys()):
            value[key] = cast_floating_tensors(value[key], float_dtype)
        return value
    if isinstance(value, list):
        return [cast_floating_tensors(item, float_dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(cast_floating_tensors(item, float_dtype) for item in value)
    if isinstance(value, dict):
        return {key: cast_floating_tensors(item, float_dtype) for key, item in value.items()}
    return value


def move_batch_to_device(batch, device, float_dtype=None):
    if hasattr(batch, "to"):
        if float_dtype is not None:
            try:
                batch = batch.to(device, dtype=float_dtype)
                return batch
            except TypeError:
                batch = batch.to(device)
        else:
            batch = batch.to(device)

    if float_dtype is None:
        return batch

    if hasattr(batch, "items") and hasattr(batch, "__setitem__"):
        for key, value in list(batch.items()):
            batch[key] = cast_floating_tensors(value, float_dtype)
        return batch

    if hasattr(batch, "keys") and hasattr(batch, "__getitem__") and hasattr(batch, "__setitem__"):
        for key in list(batch.keys()):
            batch[key] = cast_floating_tensors(batch[key], float_dtype)
        return batch

    return cast_floating_tensors(batch, float_dtype)
