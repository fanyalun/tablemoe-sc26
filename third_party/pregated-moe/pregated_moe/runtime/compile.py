import os

import torch
from transformers.models.mixtral.modeling_mixtral import (
    MixtralBlockSparseTop2MLP,
)
from transformers.models.nllb_moe.modeling_nllb_moe import (
    NllbMoeDenseActDense,
)

try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

    _has_qwen3_moe = True
except ImportError:
    Qwen3MoeMLP = None
    _has_qwen3_moe = False

# from pregated_moe.models.modeling_grok import MoeMLP as GrokMoeMLP
from pregated_moe.models.modeling_arctic import ArcticMLP
from pregated_moe.models.modeling_deepseek_v2 import DeepseekV2MLP
from pregated_moe.models.modeling_deepseek_v3 import DeepseekV3MLP

EXPERT_CLS = {
    # "grok": GrokMoeMLP,
    "arctic": ArcticMLP,
    "deepseek_v2": DeepseekV2MLP,
    "deepseek_v3": DeepseekV3MLP,
    "mixtral": MixtralBlockSparseTop2MLP,
    # "nllb_moe": NllbMoeDenseActDense,
}

if _has_qwen3_moe:
    EXPERT_CLS["qwen3_moe"] = Qwen3MoeMLP


# compile a single expert
def script_expert(save_dir, expert_type, config, **kwargs):
    """
    Compile a single expert.
    """
    # get argument list from the expert class
    # expert_cls = EXPERT_CLS[expert_type]
    # expert_args = expert_cls.__init__.__code__.co_varnames

    if expert_type not in EXPERT_CLS:
        raise RuntimeError(
            f"Unsupported or unavailable expert_type `{expert_type}`. "
            f"Available expert types: {sorted(EXPERT_CLS.keys())}"
        )

    expert_instance = EXPERT_CLS[expert_type](config, **kwargs)
    # compile the forward function of the expert
    module = torch.jit.script(expert_instance)
    torch.jit.save(
        module,
        os.path.join(save_dir, "expert.pt"),
    )
