# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from .arctic import ArcticConfig, SyncArcticMoeBlock
from .deepseek import DeepseekMoEBlock
from .grok import SyncGrokMoeBlock
from .mixtral import SyncMixtralSparseMoeBlock
from .model_utils import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_deepseek,
    rotate_half,
)
from .nllb_moe import SyncNllbMoeSparseMLP
from .qwen3_vl_moe import SyncQwen3VLMoeSparseMoeBlock
from .switch_transformers import SyncSwitchTransformersSparseMLP

try:
    from .qwen import Qwen3MoEBlock
except ImportError:
    class Qwen3MoEBlock:  # type: ignore[no-redef]
        pass
