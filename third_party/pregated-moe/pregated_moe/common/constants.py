import sys
from pathlib import Path

from transformers import (
    MixtralForCausalLM,
    NllbMoeForConditionalGeneration,
    OPTForCausalLM,
    PretrainedConfig,
    SwitchTransformersForConditionalGeneration,
)

try:
    from transformers import Qwen3MoeForCausalLM

    _has_qwen3_moe = True
except ImportError:
    Qwen3MoeForCausalLM = None
    _has_qwen3_moe = False

from ..models.modeling_arctic import (
    ArcticForCausalLM,
)  # TODO: Replace this with huggingface transformers
from ..models.modeling_deepseek_v2 import DeepseekV2ForCausalLM
from ..models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ..models.modeling_grok.modeling_grok1 import (
    Grok1ModelForCausalLM,
)  # TODO: Replace this with huggingface transformers


def ensure_local_deepseek_vl2_repo():
    repo_root = Path(__file__).resolve().parents[3] / "DeepSeek-VL2"
    if not repo_root.exists():
        return None

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    return repo_root


try:
    if ensure_local_deepseek_vl2_repo() is not None:
        from deepseek_vl2.models import DeepseekVLV2ForCausalLM

        _has_deepseek_vl2 = True
    else:
        _has_deepseek_vl2 = False
except ImportError:
    _has_deepseek_vl2 = False

try:
    from transformers import Qwen3VLMoeForConditionalGeneration

    _has_qwen3_vl_moe = True
except ImportError:
    _has_qwen3_vl_moe = False

MODEL_MAPPING_NAMES = {
    "switch": SwitchTransformersForConditionalGeneration,
    "nllb": NllbMoeForConditionalGeneration,
    "mixtral": MixtralForCausalLM,
    "opt": OPTForCausalLM,
    "grok": Grok1ModelForCausalLM,
    "arctic": ArcticForCausalLM,
    "deepseek": DeepseekV2ForCausalLM,
    "deepseek_v3": DeepseekV3ForCausalLM,
}

if _has_qwen3_moe:
    MODEL_MAPPING_NAMES["qwen3"] = Qwen3MoeForCausalLM

if _has_deepseek_vl2:
    MODEL_MAPPING_NAMES["deepseek_vl2"] = DeepseekVLV2ForCausalLM

if _has_qwen3_vl_moe:
    MODEL_MAPPING_NAMES["qwen3vlmoe"] = (
        Qwen3VLMoeForConditionalGeneration
    )

MODEL_MAPPING_TYPES = {
    "switch": 0,
    "nllb": 2,
    "mixtral": 4,
    "grok": 4,
    "arctic": 4,
    "deepseek": 5,
    "deepseek_v3": 5,
    "deepseek_vl2": 5,
    "qwen3vlmoe": 5,
}

if _has_qwen3_moe:
    MODEL_MAPPING_TYPES["qwen3"] = 5


def get_architecture_name(config: PretrainedConfig) -> str:
    architectures = getattr(config, "architectures", None)
    if architectures:
        return architectures[0].lower()

    model_type = getattr(config, "model_type", None)
    if model_type:
        return str(model_type).lower()

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        architectures = getattr(text_config, "architectures", None)
        if architectures:
            return architectures[0].lower()
        model_type = getattr(text_config, "model_type", None)
        if model_type:
            return str(model_type).lower()

    language_config = getattr(config, "language_config", None)
    if language_config is not None:
        architectures = getattr(language_config, "architectures", None)
        if architectures:
            return architectures[0].lower()
        model_type = getattr(language_config, "model_type", None)
        if model_type:
            return str(model_type).lower()

    raise RuntimeError("Unable to infer model architecture from config.")


def resolve_model_architecture(config: PretrainedConfig) -> str:
    architecture = get_architecture_name(config)

    if any(
        name in architecture
        for name in ("deepseek_vl_v2", "deepseekvl2", "deepseekvlv2")
    ):
        return "deepseek_vl2"
    if any(
        name in architecture
        for name in ("qwen3_vl_moe", "qwen3vlmoe")
    ):
        return "qwen3vlmoe"

    for supp_arch in MODEL_MAPPING_NAMES:
        if supp_arch in architecture:
            return supp_arch

    raise RuntimeError(
        f"The `load_checkpoint_and_dispatch` function does not support the architecture {architecture}. "
        f"Please provide a model that is supported by the function. "
        f"Supported architectures are {list(MODEL_MAPPING_NAMES.keys())}."
    )


def parse_expert_type(config: PretrainedConfig) -> int:
    arch = resolve_model_architecture(config)
    return MODEL_MAPPING_TYPES[arch]
