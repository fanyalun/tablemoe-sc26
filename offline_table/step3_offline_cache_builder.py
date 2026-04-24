import argparse
import gc
import glob
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_table.common import (
    DEFAULT_DTYPE_NAME,
    DTYPE_CHOICES,
    MODEL_CHOICES,
    get_deepseek_layers,
    get_qwen_layers,
    get_routed_experts,
    get_torch_dtype,
    load_deepseek_components,
)


def safe_matmul(input_tensor, weight, bias=None):
    in_features = input_tensor.shape[-1]
    if weight.shape[0] == in_features:
        output = torch.matmul(input_tensor, weight)
    elif weight.shape[1] == in_features:
        output = torch.matmul(input_tensor, weight.t())
    else:
        raise RuntimeError(f"Shape mismatch: input {input_tensor.shape}, weight {weight.shape}")
    if bias is not None:
        output += bias
    return output


def forward_single_qwen_expert(experts_module, expert_idx, inputs):
    gate_up_weights = experts_module.gate_up_proj if isinstance(experts_module.gate_up_proj, torch.nn.Parameter) else experts_module.gate_up_proj.weight
    down_weights = experts_module.down_proj if isinstance(experts_module.down_proj, torch.nn.Parameter) else experts_module.down_proj.weight
    current_gate_up_w = gate_up_weights[expert_idx]
    current_down_w = down_weights[expert_idx]
    gate_up_bias = getattr(experts_module.gate_up_proj, "bias", None)
    down_bias = getattr(experts_module.down_proj, "bias", None)
    current_gate_up_b = gate_up_bias[expert_idx] if gate_up_bias is not None else None
    current_down_b = down_bias[expert_idx] if down_bias is not None else None
    gate_up_out = safe_matmul(inputs, current_gate_up_w, current_gate_up_b)
    gate, up = gate_up_out.chunk(2, dim=-1)
    act_out = F.silu(gate) * up
    return safe_matmul(act_out, current_down_w, current_down_b)


def forward_single_deepseek_expert(expert_module, inputs):
    gate_out = expert_module.gate_proj(inputs)
    up_out = expert_module.up_proj(inputs)
    return expert_module.down_proj(expert_module.act_fn(gate_out) * up_out)


def load_qwen_model(model_path, model_dtype):
    from transformers import Qwen3VLMoeForConditionalGeneration

    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()
    return model, get_qwen_layers(model)


def load_deepseek_model(model_path, model_dtype):
    _, DeepseekVLV2ForCausalLM, _ = load_deepseek_components()
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
    ).to("cpu").eval()
    return model, get_deepseek_layers(model)


def build_offline_cache(args):
    model_dtype = get_torch_dtype(args.dtype)
    print(f"Loading Model from {args.model_path} | model={args.model} | dtype={args.dtype}")

    if args.model == "qwen3vlmoe":
        model, layers = load_qwen_model(args.model_path, model_dtype)
    else:
        model, layers = load_deepseek_model(args.model_path, model_dtype)

    os.makedirs(args.save_dir, exist_ok=True)
    layer_dirs = sorted(glob.glob(os.path.join(args.cluster_dir, "layer_*")))
    progress = tqdm(layer_dirs, desc="Processing Layers")

    for layer_dir in progress:
        try:
            layer_idx = int(os.path.basename(layer_dir).split("_")[1])
        except Exception:
            continue
        if layer_idx >= len(layers):
            continue

        if args.model == "qwen3vlmoe":
            try:
                current_experts = layers[layer_idx].mlp.experts.to(args.device)
                num_experts = model.config.text_config.num_experts
            except AttributeError:
                continue
        else:
            try:
                current_experts, num_experts = get_routed_experts(layers[layer_idx].mlp)
                current_experts = current_experts.to(args.device)
            except AttributeError:
                continue

        layer_save_dir = os.path.join(args.save_dir, f"layer_{layer_idx}")
        os.makedirs(layer_save_dir, exist_ok=True)
        cluster_files = glob.glob(os.path.join(layer_dir, "*_clusters.pt"))
        saved_count = 0

        for cluster_file in cluster_files:
            file_name = os.path.basename(cluster_file)
            parts = file_name.replace(".pt", "").split("_")
            if len(parts) < 4:
                continue
            expert_id = int(parts[1][1:])
            modality = parts[2]
            if expert_id >= num_experts:
                continue

            cluster_data = torch.load(cluster_file, map_location="cpu")
            low_dim_keys = cluster_data.get("key")
            high_dim_inputs = cluster_data.get("value")
            if low_dim_keys is None or high_dim_inputs is None:
                continue

            inputs = high_dim_inputs.to(args.device).to(model_dtype)
            with torch.no_grad():
                if args.model == "qwen3vlmoe":
                    outputs = forward_single_qwen_expert(current_experts, expert_id, inputs)
                else:
                    outputs = forward_single_deepseek_expert(current_experts[expert_id], inputs)

            torch.save(
                {
                    "id": torch.arange(low_dim_keys.shape[0], dtype=torch.int32),
                    "key": low_dim_keys.to(model_dtype).cpu(),
                    "value": outputs.to(model_dtype).cpu(),
                },
                os.path.join(layer_save_dir, f"L{layer_idx}_E{expert_id}_{modality}_cache.pt"),
            )
            saved_count += 1

        progress.set_postfix({"Saved": saved_count})
        current_experts.cpu()
        del current_experts
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nOffline Cache Build Complete! Saved to {args.save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Step3 offline expert table builder")
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--cluster-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default=DEFAULT_DTYPE_NAME)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    build_offline_cache(parse_args())
