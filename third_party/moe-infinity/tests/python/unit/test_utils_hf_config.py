from types import SimpleNamespace
from typing import cast

import pytest
import torch
from transformers import PretrainedConfig

from moe_infinity.common.constants import resolve_model_architecture
from moe_infinity.utils.hf_config import (
    parse_expert_dtype,
    parse_expert_id,
    parse_moe_param,
)


def _cfg(**kwargs) -> PretrainedConfig:
    return cast(PretrainedConfig, SimpleNamespace(**kwargs))


def test_parse_expert_dtype_supported():
    cfg = _cfg(torch_dtype=torch.bfloat16)
    assert parse_expert_dtype(cfg) == 0

    cfg = _cfg(torch_dtype=torch.float32)
    assert parse_expert_dtype(cfg) == 1

    cfg = _cfg(torch_dtype=torch.float16)
    assert parse_expert_dtype(cfg) == 2

    composite = _cfg(
        torch_dtype=None,
        text_config=_cfg(torch_dtype=torch.float16),
    )
    assert parse_expert_dtype(composite) == 2


def test_parse_expert_dtype_unsupported():
    cfg = _cfg(torch_dtype=torch.int32)
    with pytest.raises(AssertionError):
        parse_expert_dtype(cfg)


def test_parse_moe_param_switch_and_nllb():
    switch = _cfg(
        architectures=["SwitchTransformers"],
        num_sparse_encoder_layers=2,
        num_sparse_decoder_layers=3,
        num_experts=8,
    )
    assert parse_moe_param(switch) == (5, 8, 2)

    nllb = _cfg(
        architectures=["NllbMoe"],
        encoder_layers=12,
        decoder_layers=6,
        encoder_sparse_step=2,
        decoder_sparse_step=3,
        num_experts=4,
    )
    assert parse_moe_param(nllb) == (8, 4, 6)


def test_parse_moe_param_mixtral_and_deepseek():
    mixtral = _cfg(
        architectures=["Mixtral"],
        num_hidden_layers=7,
        num_local_experts=6,
    )
    assert parse_moe_param(mixtral) == (7, 6, 0)

    deepseek = _cfg(
        architectures=["DeepSeek"],
        num_hidden_layers=9,
        n_routed_experts=12,
    )
    assert parse_moe_param(deepseek) == (9, 12, 0)


def test_parse_moe_param_qwen3_and_qwen3vlmoe():
    qwen3 = _cfg(
        architectures=["Qwen3MoeForCausalLM"],
        num_hidden_layers=8,
        num_experts=16,
    )
    assert parse_moe_param(qwen3) == (8, 16, 0)

    qwen3vl = _cfg(
        architectures=["Qwen3VLMoeForConditionalGeneration"],
        torch_dtype=None,
        text_config=_cfg(
            torch_dtype=torch.bfloat16,
            num_hidden_layers=6,
            num_experts=8,
            mlp_only_layers=[0, 3],
            decoder_sparse_step=2,
        ),
    )
    assert parse_moe_param(qwen3vl) == (2, 8, 0)


def test_parse_moe_param_deepseek_vl2_uses_language_config():
    deepseek_vl2 = _cfg(
        architectures=["DeepseekVLV2ForCausalLM"],
        language_config=_cfg(
            num_hidden_layers=10,
            n_routed_experts=32,
        ),
    )
    assert parse_moe_param(deepseek_vl2) == (10, 32, 0)


def test_parse_expert_id_switch_and_mixtral():
    switch = _cfg(
        architectures=["Switch"],
        num_sparse_encoder_layers=2,
        num_sparse_decoder_layers=2,
        num_experts=4,
        encoder_sparse_step=2,
        decoder_sparse_step=2,
    )

    layer_id, expert_id = parse_expert_id(
        "decoder.block.1.layer.2.mlp.experts.expert_3.wi.weight",
        switch,
    )
    assert (layer_id, expert_id) == (2, 3)

    mixtral = _cfg(
        architectures=["Mixtral"],
        num_hidden_layers=4,
        num_local_experts=8,
    )
    layer_id, expert_id = parse_expert_id(
        "model.layers.2.block_sparse_moe.experts.5.w1.weight",
        mixtral,
    )
    assert (layer_id, expert_id) == (2, 5)


def test_parse_expert_id_grok_and_deepseek():
    grok = _cfg(
        architectures=["Grok"],
        num_hidden_layers=3,
        num_experts=2,
    )
    layer_id, expert_id = parse_expert_id(
        "model.layers.1.moe_block.experts.0.linear_1.weight",
        grok,
    )
    assert (layer_id, expert_id) == (1, 0)

    deepseek = _cfg(
        architectures=["Deepseek"],
        num_hidden_layers=3,
        n_routed_experts=4,
    )
    layer_id, expert_id = parse_expert_id(
        "model.layers.2.mlp.experts.3.gate_proj.weight",
        deepseek,
    )
    assert (layer_id, expert_id) == (2, 3)


def test_parse_expert_id_qwen3_and_qwen3vlmoe():
    qwen3 = _cfg(
        architectures=["Qwen3MoeForCausalLM"],
        num_hidden_layers=8,
        num_experts=16,
    )
    layer_id, expert_id = parse_expert_id(
        "model.layers.5.mlp.experts.7.gate_proj.weight",
        qwen3,
    )
    assert (layer_id, expert_id) == (5, 7)

    qwen3vl = _cfg(
        architectures=["Qwen3VLMoeForConditionalGeneration"],
        text_config=_cfg(
            num_hidden_layers=6,
            num_experts=8,
            mlp_only_layers=[0, 3],
            decoder_sparse_step=2,
        ),
    )
    layer_id, expert_id = parse_expert_id(
        "model.language_model.layers.5.mlp.experts.7.gate_proj.weight",
        qwen3vl,
    )
    assert (layer_id, expert_id) == (1, 7)


def test_resolve_model_architecture_multimodal_aliases():
    deepseek_vl2 = _cfg(architectures=["DeepseekVLV2ForCausalLM"])
    assert resolve_model_architecture(deepseek_vl2) == "deepseek_vl2"

    qwen3vl = _cfg(architectures=["Qwen3VLMoeForConditionalGeneration"])
    assert resolve_model_architecture(qwen3vl) == "qwen3vlmoe"
