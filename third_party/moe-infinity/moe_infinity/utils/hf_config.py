import re
from typing import Optional, Tuple

import torch
from transformers import PretrainedConfig


def _get_text_like_config(config: PretrainedConfig) -> PretrainedConfig:
    if hasattr(config, "text_config"):
        return config.text_config
    if hasattr(config, "language_config"):
        return config.language_config
    return config


def _get_architecture_name(config: PretrainedConfig) -> str:
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


def _resolve_model_architecture(config: PretrainedConfig) -> str:
    architecture = _get_architecture_name(config)

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

    supported_architectures = (
        "switch",
        "nllb",
        "mixtral",
        "opt",
        "grok",
        "arctic",
        "deepseek",
        "deepseek_v3",
        "qwen3",
    )
    for supp_arch in supported_architectures:
        if supp_arch in architecture:
            return supp_arch

    raise RuntimeError(
        f"The `load_checkpoint_and_dispatch` function does not support the architecture {architecture}. "
        "Please provide a model that is supported by the function."
    )


def parse_expert_dtype(config: PretrainedConfig) -> int:
    dtype = config.torch_dtype
    if dtype is None and (
        hasattr(config, "text_config") or hasattr(config, "language_config")
    ):
        dtype = _get_text_like_config(config).torch_dtype
    if dtype == torch.bfloat16:
        dtype = 0
    elif dtype == torch.float32:
        dtype = 1
    elif dtype == torch.float16:
        dtype = 2
    else:
        assert False, "Unknown dtype %s" % dtype

    return dtype


def parse_moe_param(config: PretrainedConfig) -> Tuple[int, int, int]:
    arch = _resolve_model_architecture(config)

    if "switch" in arch:
        num_encoder_layers = config.num_sparse_encoder_layers
        num_decoder_layers = config.num_sparse_decoder_layers
        num_layers = num_encoder_layers + num_decoder_layers
        num_experts = config.num_experts
    elif "nllb" in arch:
        num_encoder_layers = config.encoder_layers // config.encoder_sparse_step
        num_decoder_layers = config.decoder_layers // config.decoder_sparse_step
        num_layers = num_encoder_layers + num_decoder_layers
        num_experts = config.num_experts
    elif "mixtral" in arch or "arctic" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_local_experts
    elif "grok" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_experts
    elif "deepseek" in arch:
        deepseek_config = _get_text_like_config(config)
        num_encoder_layers = 0
        num_decoder_layers = deepseek_config.num_hidden_layers
        num_layers = deepseek_config.num_hidden_layers
        num_experts = deepseek_config.n_routed_experts
    elif "qwen3vlmoe" in arch:
        text_config = _get_text_like_config(config)
        num_encoder_layers = 0
        num_decoder_layers = text_config.num_hidden_layers
        num_layers = sum(
            1
            for i in range(text_config.num_hidden_layers)
            if (i not in text_config.mlp_only_layers)
            and (text_config.num_experts > 0)
            and ((i + 1) % text_config.decoder_sparse_step == 0)
        )
        num_experts = text_config.num_experts
    elif "qwen3" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_experts
    else:
        raise RuntimeError(f"Unsupported architecture {arch}")

    return num_layers, num_experts, num_encoder_layers


def parse_expert_id(
    param_name: str, config: PretrainedConfig
) -> Tuple[Optional[int], Optional[int]]:
    arch = _resolve_model_architecture(config)
    _, _, num_encoder_layers = parse_moe_param(config)
    result = []

    if "switch" in arch or "nllb" in arch:
        # example "decoder.block.1.layer.2.mlp.experts.expert_100.wi.weight"
        encoder_sparse_step = config.encoder_sparse_step
        decoder_sparse_step = config.decoder_sparse_step

        result = re.findall(
            r"(encoder|decoder)\.[a-z]+\.(\d+).*expert_(\d+)", param_name
        )

        if result:
            layer_type, layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)

    elif "mixtral" in arch or "arctic" in arch:
        encoder_sparse_step = None
        decoder_sparse_step = 1
        layer_type = "decoder"

        # example "model.layers.0.block_sparse_moe.experts.0.w1.weight"
        result = re.findall(
            r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    elif "grok" in arch:
        encoder_sparse_step = None
        decoder_sparse_step = 1
        layer_type = "decoder"

        # example "model.layers.0.moe_block.experts.0.linear_1.weight"
        result = re.findall(
            r"layers\.(\d+)\.moe_block\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            # print(f"layer_id: {layer_id}, expert_id: {expert_id}")
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    elif "deepseek" in arch:
        encoder_sparse_step = None
        decoder_sparse_step = 1
        layer_type = "decoder"

        # example "model.layers.1.mlp.experts.0.gate_proj.weight"
        result = re.findall(
            r"(?:language\.)?model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.",
            param_name,
        )
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    elif "qwen3vlmoe" in arch:
        encoder_sparse_step = None
        decoder_sparse_step = 1
        layer_type = "decoder"

        result = re.findall(
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)

            text_config = _get_text_like_config(config)
            moe_layer_idx = sum(
                1
                for i in range(layer_id)
                if (i not in text_config.mlp_only_layers)
                and (text_config.num_experts > 0)
                and ((i + 1) % text_config.decoder_sparse_step == 0)
            )
            layer_id = moe_layer_idx
    elif "qwen3" in arch:
        encoder_sparse_step = None
        decoder_sparse_step = 1
        layer_type = "decoder"

        result = re.findall(
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)

    if result:
        if layer_type == "decoder":
            layer_id = layer_id // decoder_sparse_step + num_encoder_layers
        elif layer_type == "encoder":
            layer_id = layer_id // encoder_sparse_step
        else:
            raise ValueError(f"Unsupported layer type {layer_type}")

        return layer_id, expert_id

    return None, None
