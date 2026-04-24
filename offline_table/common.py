import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from table_moe.utils.dataset import (
    load_ai2d_dataset,
    load_mmbench_dataset,
    load_pope_dataset,
    load_realworldqa_dataset,
    load_scienceqa_dataset,
)
from table_moe.utils.prompt import (
    build_ai2d_prompt,
    build_ai2d_prompt_deepseekvl2,
    build_mmbench_prompt,
    build_mmbench_prompt_deepseekvl2,
    build_pope_prompt,
    build_pope_prompt_deepseekvl2,
    build_realworldqa_prompt,
    build_realworldqa_prompt_deepseekvl2,
    build_scienceqa_prompt,
    build_scienceqa_prompt_deepseekvl2,
)

OFFLINE_TABLE_ROOT = REPO_ROOT / "offline_table"
COMPRESSED_DIM = 64
FIXED_K = 256
MODEL_DTYPE = torch.bfloat16
DEFAULT_CLUSTER_SIZE = FIXED_K
DEFAULT_DTYPE_NAME = "bf16"
MODEL_CHOICES = ("qwen3vlmoe", "deepseekvl2")
DTYPE_CHOICES = ("bf16", "fp16")
QWEN_IMAGE_TOKEN_ID = 151655
QWEN_VIDEO_TOKEN_ID = 151656

DATASET_LOADERS = {
    "realworldqa": load_realworldqa_dataset,
    "mmbench": load_mmbench_dataset,
    "ai2d": load_ai2d_dataset,
    "scienceqa": load_scienceqa_dataset,
    "pope": load_pope_dataset,
}

QWEN_PROMPT_BUILDERS = {
    "realworldqa": build_realworldqa_prompt,
    "mmbench": build_mmbench_prompt,
    "ai2d": build_ai2d_prompt,
    "scienceqa": build_scienceqa_prompt,
    "pope": build_pope_prompt,
}

DEEPSEEK_PROMPT_BUILDERS = {
    "realworldqa": build_realworldqa_prompt_deepseekvl2,
    "mmbench": build_mmbench_prompt_deepseekvl2,
    "ai2d": build_ai2d_prompt_deepseekvl2,
    "scienceqa": build_scienceqa_prompt_deepseekvl2,
    "pope": build_pope_prompt_deepseekvl2,
}

DEFAULT_MIN_TEXT_PER_SAMPLE = {
    "qwen3vlmoe": 8,
    "deepseekvl2": 8,
}


def ensure_repo_root_on_path():
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def ensure_deepseek_repo_on_path():
    ensure_repo_root_on_path()
    deepseek_repo = REPO_ROOT / "third_party" / "DeepSeek-VL2"
    deepseek_repo_str = str(deepseek_repo)
    if deepseek_repo_str not in sys.path:
        sys.path.insert(0, deepseek_repo_str)
    return deepseek_repo


def load_deepseek_components():
    ensure_deepseek_repo_on_path()
    from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
    from deepseek_vl2.utils.io import load_pil_images

    return DeepseekVLV2Processor, DeepseekVLV2ForCausalLM, load_pil_images


def detect_dataset_key(data_path: str) -> str:
    lower = data_path.lower()
    for key in DATASET_LOADERS:
        if key in lower:
            return key
    raise ValueError(f"Unknown dataset in path: {data_path}")


def get_dataset_loader(dataset_key: str):
    return DATASET_LOADERS[dataset_key]


def get_prompt_builder(model_name: str, dataset_key: str):
    if model_name == "qwen3vlmoe":
        return QWEN_PROMPT_BUILDERS[dataset_key]
    if model_name == "deepseekvl2":
        return DEEPSEEK_PROMPT_BUILDERS[dataset_key]
    raise ValueError(f"Unsupported model: {model_name}")


def normalize_dtype_name(dtype_name: str) -> str:
    normalized = str(dtype_name).strip().lower()
    aliases = {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return aliases[normalized]


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = normalize_dtype_name(dtype_name)
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def get_model_root_name(model_name: str, dtype_name: str = DEFAULT_DTYPE_NAME) -> str:
    normalized_dtype = normalize_dtype_name(dtype_name)
    if model_name == "qwen3vlmoe":
        return f"qwen_{normalized_dtype}"
    if model_name == "deepseekvl2":
        return f"ds_{normalized_dtype}"
    raise ValueError(f"Unsupported model: {model_name}")


def get_model_root(model_name: str, dtype_name: str = DEFAULT_DTYPE_NAME) -> Path:
    return OFFLINE_TABLE_ROOT / get_model_root_name(model_name, dtype_name)


def get_default_step_root(model_name: str, step_name: str, dtype_name: str = DEFAULT_DTYPE_NAME) -> Path:
    root = get_model_root(model_name, dtype_name)
    if step_name == "step1":
        return root / "profiling_results"
    if step_name == "step2":
        return root / "clustering_results"
    if step_name == "step3":
        return root / "offline_table"
    raise ValueError(f"Unsupported step name: {step_name}")
def get_default_min_text_per_sample(model_name: str) -> int:
    return DEFAULT_MIN_TEXT_PER_SAMPLE[model_name]


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
    if isinstance(value, list):
        return [cast_floating_tensors(item, float_dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(cast_floating_tensors(item, float_dtype) for item in value)
    if isinstance(value, dict):
        return {key: cast_floating_tensors(item, float_dtype) for key, item in value.items()}
    return value


def move_batch_to_device(batch, device, float_dtype=None):
    if hasattr(batch, "to"):
        batch = batch.to(device)
    if float_dtype is None:
        return batch
    if hasattr(batch, "items") and hasattr(batch, "__setitem__"):
        for key, value in list(batch.items()):
            batch[key] = cast_floating_tensors(value, float_dtype)
        return batch
    return cast_floating_tensors(batch, float_dtype)


def get_qwen_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Could not locate Qwen layers")


def get_deepseek_layers(model):
    if hasattr(model, "language") and hasattr(model.language, "model") and hasattr(model.language.model, "layers"):
        return model.language.model.layers
    raise ValueError("Could not locate DeepSeek layers")


def get_routed_experts(moe_module):
    experts = getattr(moe_module, "experts", None)
    if experts is None:
        raise ValueError("MoE module does not expose routed experts")
    return experts, len(experts)


def build_deepseek_conversation(prompt_items):
    content = []
    images = []
    for item in prompt_items:
        if item["type"] == "image":
            images.append(item["value"])
            content.append("<image>\n")
        elif item["type"] == "text":
            content.append(item["value"])
    return [
        {"role": "<|User|>", "content": "".join(content), "images": images},
        {"role": "<|Assistant|>", "content": ""},
    ]


def get_special_token_ids(model_name: str, processor=None):
    if model_name == "qwen3vlmoe":
        return QWEN_IMAGE_TOKEN_ID, QWEN_VIDEO_TOKEN_ID
    if model_name == "deepseekvl2":
        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("DeepSeek processor does not expose image_token_id")
        return image_token_id, None
    raise ValueError(f"Unsupported model: {model_name}")


def get_threshold(model_name: str, modality: str, layer_idx: int, num_layers: int | None = None) -> float:
    if modality not in {"vision", "text"}:
        raise ValueError(f"Unsupported modality: {modality}")

    if model_name == "qwen3vlmoe":
        if modality == "vision":
            if 0 <= layer_idx <= 4:
                return 0.6325
            if 5 <= layer_idx <= 42:
                return 0.7746
            if 43 <= layer_idx <= 47:
                return 0.6325
        if modality == "text":
            if 0 <= layer_idx <= 4:
                return 0.3162
            if 5 <= layer_idx <= 42:
                return 0.4472
            if 43 <= layer_idx <= 47:
                return 0.3162
        return 0.3162

    if num_layers is None or num_layers <= 0:
        num_layers = max(layer_idx + 1, 1)

    edge_count = min(5, max(1, num_layers // 6))
    mid_start = edge_count
    mid_end = max(mid_start, num_layers - edge_count - 1)

    if modality == "vision":
        if layer_idx < mid_start or layer_idx > mid_end:
            return 0.6325
        return 0.7746

    if layer_idx < mid_start or layer_idx > mid_end:
        return 0.3162
    return 0.4472


def resolve_runtime_device(device_arg: str) -> str:
    if torch.cuda.is_available():
        return device_arg
    return "cpu"


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def infer_benchmark_name(data_path: str) -> str:
    path = Path(data_path)
    if path.suffix:
        return path.stem
    return path.name


def is_safe_cleanup_path(candidate: Path, expected_root: Path) -> bool:
    try:
        candidate.resolve().relative_to(expected_root.resolve())
    except ValueError:
        return False
    return True
