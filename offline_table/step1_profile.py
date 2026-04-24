import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
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
    build_deepseek_conversation,
    detect_dataset_key,
    ensure_dir,
    get_dataset_loader,
    get_default_min_text_per_sample,
    get_model_device,
    get_prompt_builder,
    get_special_token_ids,
    get_torch_dtype,
    infer_benchmark_name,
    load_deepseek_components,
    move_batch_to_device,
)


class ExpertDataCollector:
    def __init__(
        self,
        model_name,
        save_dir,
        max_vision_tokens=100000,
        max_text_tokens=40000,
        buffer_limit=100000,
        save_dtype="float32",
        per_expert_token_cap=0,
    ):
        self.model_name = model_name
        self.save_dir = save_dir
        self.limits = {"vision": max_vision_tokens, "text": max_text_tokens}
        self.counts = {"vision": 0, "text": 0}
        self.buffer_limit = buffer_limit
        self.save_dtype = save_dtype
        self.save_torch_dtype = torch.float16 if save_dtype == "float16" else torch.float32
        self.per_expert_token_cap = per_expert_token_cap if per_expert_token_cap > 0 else None
        self.current_buffer_size = 0
        self.buffers = {}
        self.file_indices = {}
        self.stored_counts = {}
        self.current_input_ids = None
        self.current_attention_mask = None
        self.current_selected_masks = None
        self.image_token_id = None
        self.video_token_id = None
        os.makedirs(save_dir, exist_ok=True)

    def set_special_tokens(self, image_token_id, video_token_id=None):
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

    def set_current_step(self, input_ids, attention_mask, selected_masks):
        self.current_input_ids = input_ids.detach().cpu().numpy().flatten()
        if attention_mask is None:
            self.current_attention_mask = np.ones_like(self.current_input_ids, dtype=bool)
        else:
            self.current_attention_mask = attention_mask.detach().cpu().numpy().flatten().astype(bool)
        self.current_selected_masks = selected_masks

    def is_finished(self):
        return (self.counts["vision"] >= self.limits["vision"]) and (self.counts["text"] >= self.limits["text"])

    def _ensure_expert_state(self, layer_idx, exp_id):
        if layer_idx not in self.buffers:
            self.buffers[layer_idx] = {}
            self.file_indices[layer_idx] = {}
            self.stored_counts[layer_idx] = {}
        if exp_id not in self.buffers[layer_idx]:
            self.buffers[layer_idx][exp_id] = {"text": [], "vision": []}
            self.file_indices[layer_idx][exp_id] = {"text": 0, "vision": 0}
            self.stored_counts[layer_idx][exp_id] = {"text": 0, "vision": 0}

    def _collect_qwen_routes(self, module, hidden_states_flat, min_len):
        router_logits = module.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        selected_experts = torch.topk(routing_weights, module.top_k, dim=-1)[1]
        return selected_experts[:min_len].cpu().numpy(), router_logits.shape[-1]

    def _collect_deepseek_routes(self, module, hidden_states, hidden_states_flat, min_len):
        hidden_states_for_gate = hidden_states if hidden_states.dim() == 3 else hidden_states_flat.unsqueeze(0)
        topk_idx, _, _ = module.gate(hidden_states_for_gate)
        selected_experts = topk_idx.reshape(-1, topk_idx.shape[-1])[:min_len]
        return selected_experts.cpu().numpy(), len(module.experts)

    def hook_fn(self, module, inputs, layer_idx):
        if self.is_finished():
            return
        if self.current_input_ids is None or self.current_selected_masks is None:
            return
        if not hasattr(module, "gate"):
            return

        hidden_states = inputs[0].detach()
        hidden_states_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        min_len = min(self.current_input_ids.shape[0], hidden_states_flat.shape[0])
        current_ids = self.current_input_ids[:min_len]
        hidden_states_flat = hidden_states_flat[:min_len]
        selected_masks = {key: value[:min_len] for key, value in self.current_selected_masks.items()}

        is_vision_token = current_ids == self.image_token_id
        if self.video_token_id is not None:
            is_vision_token = is_vision_token | (current_ids == self.video_token_id)

        vision_mask = selected_masks.get("vision", np.zeros_like(current_ids, dtype=bool)) & is_vision_token
        text_mask = selected_masks.get("text", np.zeros_like(current_ids, dtype=bool)) & (~is_vision_token)
        if (not np.any(vision_mask)) and (not np.any(text_mask)):
            return

        if self.model_name == "qwen3vlmoe":
            selected_experts_cpu, num_experts = self._collect_qwen_routes(module, hidden_states_flat, min_len)
        else:
            selected_experts_cpu, num_experts = self._collect_deepseek_routes(module, hidden_states, hidden_states_flat, min_len)

        hidden_states_cpu = hidden_states_flat.to(self.save_torch_dtype).cpu().numpy()

        for modality, target_mask in (("vision", vision_mask), ("text", text_mask)):
            if not np.any(target_mask):
                continue
            for exp_id in range(num_experts):
                routed_to_expert = np.any(selected_experts_cpu == exp_id, axis=1)
                final_mask = target_mask & routed_to_expert
                if not np.any(final_mask):
                    continue
                expert_tokens = hidden_states_cpu[final_mask]
                self._ensure_expert_state(layer_idx, exp_id)
                if self.per_expert_token_cap is not None:
                    stored_count = self.stored_counts[layer_idx][exp_id][modality]
                    remaining_capacity = self.per_expert_token_cap - stored_count
                    if remaining_capacity <= 0:
                        continue
                    if expert_tokens.shape[0] > remaining_capacity:
                        expert_tokens = expert_tokens[:remaining_capacity]
                if expert_tokens.shape[0] == 0:
                    continue
                self.buffers[layer_idx][exp_id][modality].append(expert_tokens)
                self.stored_counts[layer_idx][exp_id][modality] += expert_tokens.shape[0]
                self.current_buffer_size += expert_tokens.shape[0]

        if self.current_buffer_size >= self.buffer_limit:
            self.flush()

    def flush(self):
        if self.current_buffer_size == 0:
            return

        print(
            f"Flushing... (Processed: Vision {self.counts['vision']}/{self.limits['vision']}, "
            f"Text {self.counts['text']}/{self.limits['text']})"
        )

        for layer_idx, experts in self.buffers.items():
            for exp_id, modalities in experts.items():
                for modality, data_list in modalities.items():
                    if not data_list:
                        continue
                    data = np.concatenate(data_list, axis=0)
                    save_dir = os.path.join(self.save_dir, f"layer_{layer_idx}", f"expert_{exp_id}", modality)
                    os.makedirs(save_dir, exist_ok=True)
                    part_idx = self.file_indices[layer_idx][exp_id][modality]
                    np.save(os.path.join(save_dir, f"vectors_part_{part_idx}.npy"), data)
                    self.file_indices[layer_idx][exp_id][modality] += 1
                    self.buffers[layer_idx][exp_id][modality] = []

        self.current_buffer_size = 0


def resolve_hf_device_map(device: str):
    if device == "auto":
        return "auto"
    return {"": device}


def compute_adaptive_quota(avail, remaining_budget, remaining_samples, history_lengths, min_per_sample, max_per_sample):
    if avail <= 0 or remaining_budget <= 0:
        return 0

    median_len = float(np.median(history_lengths)) if history_lengths else float(avail)
    median_len = max(median_len, 1.0)
    weight = np.sqrt(max(float(avail), 1.0) / median_len)
    base = remaining_budget / max(1, remaining_samples)
    quota = int(round(base * weight))
    quota = max(min_per_sample, quota)
    quota = min(max_per_sample, quota)
    quota = min(avail, quota, remaining_budget)
    return max(0, quota)


def register_hooks(model_name, model, collector):
    hook_count = 0
    for name, module in model.named_modules():
        if model_name == "qwen3vlmoe":
            valid = name.endswith("mlp") and hasattr(module, "gate")
        else:
            valid = hasattr(module, "gate") and hasattr(module, "experts") and hasattr(module, "shared_experts")
        if not valid:
            continue
        parts = name.split(".")
        try:
            layer_idx = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            continue
        module.register_forward_pre_hook(lambda m, i, l=layer_idx: collector.hook_fn(m, i, l))
        hook_count += 1
    print(f"Registered {hook_count} hooks")


def prepare_qwen_runner(args):
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    model_dtype = get_torch_dtype(args.dtype)
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map=resolve_hf_device_map(args.device),
        trust_remote_code=True,
        attn_implementation='flash_attention_2'
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    device = get_model_device(model)
    image_token_id, video_token_id = get_special_token_ids("qwen3vlmoe")

    def prepare_inputs(prompt_builder, row, img_root):
        messages = prompt_builder(row, img_root)
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        if videos:
            videos = [video[0] for video in videos]
        inputs = processor(
            text=prompt_text,
            images=images,
            videos=videos,
            do_resize=True,
            max_pixels=5120 * 28 * 28,
            min_pixels=768 * 28 * 28,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        return move_batch_to_device(inputs, device, float_dtype=getattr(model, "dtype", model_dtype))

    def run_forward(inputs):
        with torch.no_grad():
            model(**inputs)

    return model, processor, prepare_inputs, run_forward, image_token_id, video_token_id


def prepare_deepseek_runner(args):
    DeepseekVLV2Processor, DeepseekVLV2ForCausalLM, load_pil_images = load_deepseek_components()
    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    model_dtype = get_torch_dtype(args.dtype)
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map=resolve_hf_device_map(args.device),
    ).eval()
    device = get_model_device(model)
    image_token_id, video_token_id = get_special_token_ids("deepseekvl2", processor=processor)

    def prepare_inputs(prompt_builder, row, img_root):
        prompt_items = prompt_builder(row, img_root)
        conversation = build_deepseek_conversation(prompt_items)
        pil_images = load_pil_images(conversation)
        inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        )
        return move_batch_to_device(inputs, device, float_dtype=getattr(model, "dtype", model_dtype))

    def run_forward(inputs):
        with torch.no_grad():
            inputs_embeds = model.prepare_inputs_embeds(**inputs)
            model.language.model(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
            )

    return model, processor, prepare_inputs, run_forward, image_token_id, video_token_id


def build_runner(args):
    if args.model == "qwen3vlmoe":
        return prepare_qwen_runner(args)
    if args.model == "deepseekvl2":
        return prepare_deepseek_runner(args)
    raise ValueError(f"Unsupported model: {args.model}")


def split_train_test_indices(num_samples, seed, train_ratio=0.7, no_test_split=False):
    all_indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_indices)
    if no_test_split:
        return rng, all_indices, np.array([], dtype=all_indices.dtype)
    train_size = int(num_samples * train_ratio)
    train_size = max(1, min(train_size, num_samples))
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:]
    return rng, train_indices, test_indices


def save_eval_ids(dataset, test_indices, data_path, seed):
    benchmark = infer_benchmark_name(data_path)
    eval_ids_dir = REPO_ROOT / "offline_table" / "eval_ids"
    ensure_dir(eval_ids_dir)

    test_ids = dataset.iloc[test_indices]["id"].tolist() if len(test_indices) > 0 else []
    save_path = eval_ids_dir / benchmark
    payload = {
        "benchmark": benchmark,
        "seed": seed,
        "split_ratio": {"train": 0.7, "test": 0.3},
        "num_test_samples": len(test_ids),
        "test_ids": test_ids,
    }
    with open(save_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
    print(f"Saved test ids to {save_path} (count={len(test_ids)})")


def run_collection(args):
    print(
        f"Start collection | model={args.model} | Vision Limit: {args.max_vision_tokens} | "
        f"Text Limit: {args.max_text_tokens} | model_dtype={args.dtype} | save_dtype={args.save_dtype}"
    )

    model, processor, prepare_inputs_fn, run_forward_fn, image_token_id, video_token_id = build_runner(args)
    collector = ExpertDataCollector(
        args.model,
        args.output_dir,
        max_vision_tokens=args.max_vision_tokens,
        max_text_tokens=args.max_text_tokens,
        save_dtype=args.save_dtype,
        per_expert_token_cap=args.per_expert_token_cap,
    )
    collector.set_special_tokens(image_token_id, video_token_id)
    register_hooks(args.model, model, collector)

    dataset_key = detect_dataset_key(args.data_dir)
    dataset = get_dataset_loader(dataset_key)(args.data_dir)
    prompt_builder = get_prompt_builder(args.model, dataset_key)
    print(f"Loaded {len(dataset)} samples")

    rng, train_indices, test_indices = split_train_test_indices(
        len(dataset),
        args.seed,
        train_ratio=0.7,
        no_test_split=args.no_test_split,
    )
    if args.no_test_split:
        print(f"Full-dataset training with seed={args.seed}: train={len(train_indices)}, test=0")
    else:
        save_eval_ids(dataset, test_indices, args.data_dir, args.seed)
        print(f"Train/Test split with seed={args.seed}: train={len(train_indices)}, test={len(test_indices)}")

    parent_dir = os.path.dirname(args.data_dir)
    img_root = os.path.join(parent_dir, "images")

    pbar_vis = tqdm(total=args.max_vision_tokens, desc="Vision", position=0)
    pbar_txt = tqdm(total=args.max_text_tokens, desc="Text", position=1)
    vis_hist = []
    txt_hist = []
    total_samples = len(train_indices)
    processed_samples = 0

    for idx in train_indices:
        if collector.is_finished():
            break

        row = dataset.iloc[idx]
        try:
            inputs = prepare_inputs_fn(prompt_builder, row, img_root)
            input_ids_np = inputs.input_ids.detach().cpu().numpy().flatten()
            is_vision_token = input_ids_np == image_token_id
            if video_token_id is not None:
                is_vision_token = is_vision_token | (input_ids_np == video_token_id)

            vis_candidates = np.where(is_vision_token)[0]
            txt_candidates = np.where(~is_vision_token)[0]
            remaining_vis = max(0, collector.limits["vision"] - collector.counts["vision"])
            remaining_txt = max(0, collector.limits["text"] - collector.counts["text"])
            remaining_samples = max(1, total_samples - processed_samples)

            take_vis = compute_adaptive_quota(
                avail=len(vis_candidates),
                remaining_budget=remaining_vis,
                remaining_samples=remaining_samples,
                history_lengths=vis_hist,
                min_per_sample=args.min_vision_per_sample,
                max_per_sample=args.max_vision_per_sample,
            )
            take_txt = compute_adaptive_quota(
                avail=len(txt_candidates),
                remaining_budget=remaining_txt,
                remaining_samples=remaining_samples,
                history_lengths=txt_hist,
                min_per_sample=args.min_text_per_sample,
                max_per_sample=args.max_text_per_sample,
            )

            vis_hist.append(len(vis_candidates))
            txt_hist.append(len(txt_candidates))
            selected_vis = np.zeros_like(is_vision_token, dtype=bool)
            selected_txt = np.zeros_like(is_vision_token, dtype=bool)

            if take_vis > 0:
                chosen_vis = vis_candidates if take_vis == len(vis_candidates) else rng.choice(vis_candidates, size=take_vis, replace=False)
                selected_vis[chosen_vis] = True

            if take_txt > 0:
                chosen_txt = txt_candidates if take_txt == len(txt_candidates) else rng.choice(txt_candidates, size=take_txt, replace=False)
                selected_txt[chosen_txt] = True

            if (not np.any(selected_vis)) and (not np.any(selected_txt)):
                processed_samples += 1
                continue

            collector.set_current_step(
                inputs.input_ids,
                getattr(inputs, "attention_mask", None),
                selected_masks={"vision": selected_vis, "text": selected_txt},
            )

            vis_inc = int(selected_vis.sum())
            txt_inc = int(selected_txt.sum())
            collector.counts["vision"] += vis_inc
            collector.counts["text"] += txt_inc
            pbar_vis.update(vis_inc)
            pbar_txt.update(txt_inc)

            run_forward_fn(inputs)
            processed_samples += 1
        except Exception as exc:
            print(f"Sample {idx} Error: {exc}")
            processed_samples += 1

    pbar_vis.close()
    pbar_txt.close()
    collector.flush()
    print(f"Finished! Collected: Vision {collector.counts['vision']}, Text {collector.counts['text']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Step1 expert token profiling")
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default=DEFAULT_DTYPE_NAME)
    parser.add_argument("--save-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--min-vision-per-sample", type=int, default=1)
    parser.add_argument("--min-text-per-sample", type=int, default=None)
    parser.add_argument("--max-vision-per-sample", type=int, default=64)
    parser.add_argument("--max-text-per-sample", type=int, default=128)
    parser.add_argument("--max-vision-tokens", type=int, default=100000)
    parser.add_argument("--max-text-tokens", type=int, default=40000)
    parser.add_argument("--per-expert-token-cap", type=int, default=0)
    parser.add_argument("--no-test-split", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.min_text_per_sample is None:
        args.min_text_per_sample = get_default_min_text_per_sample(args.model)
    return args


if __name__ == "__main__":
    run_collection(parse_args())
