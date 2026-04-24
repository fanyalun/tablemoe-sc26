import argparse
import gc
import json
import os
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, REPO_ROOT_STR)

from table_moe import build_model
from table_moe.utils import dataset as dataset_module
from table_moe.utils import prompt as prompt_module
from table_moe.utils.eval import eval_mme, eval_mmbench, eval_realworldqa
from table_moe.utils.modality import ModalityContext
from table_moe.utils.perf_profile import PerfProfileRecorder
from table_moe.utils.timestreamer import TimingStreamer
from runtime_env import configure_runtime_env


MODEL_SPECS = {
    "qwen3vlmoe": {
        "family": "qwen3_vl_moe",
        "label": "Qwen3-VL-MoE",
        "default_attn": "flash_attention_2",
    },
    "deepseekvl2": {
        "family": "deepseek_vl2",
        "label": "DeepSeek-VL2",
        "default_attn": "flash_attention_2",
    },
}


def _resolve_attr(module, candidate_names):
    for name in candidate_names:
        value = getattr(module, name, None)
        if callable(value):
            return value
    raise ImportError(f"Missing callable in {module.__name__}: {candidate_names}")


def _resolve_prompt_builder(base_name):
    fn = _resolve_attr(
        prompt_module,
        [
            base_name,
            f"{base_name}_qwen3vl",
            f"{base_name}_deepseekvl2",
            f"{base_name}_vlmeval",
        ],
    )

    def wrapped(line, img_root, _fn=fn):
        messages = _fn(line, img_root)
        if (
            isinstance(messages, list)
            and messages
            and isinstance(messages[0], dict)
            and "type" in messages[0]
            and "value" in messages[0]
        ):
            to_hf = getattr(prompt_module, "_to_hf_chat_messages", None)
            if callable(to_hf):
                return to_hf(messages)
        return messages

    return wrapped


DATASET_LOADERS = {
    "mmbench": _resolve_attr(dataset_module, ["load_mmbench_dataset"]),
    "hallusionbench": _resolve_attr(dataset_module, ["load_hallusionbench_dataset"]),
    "ai2d": _resolve_attr(dataset_module, ["load_ai2d_dataset"]),
    "mme": _resolve_attr(dataset_module, ["load_mme_dataset"]),
    "realworldqa": _resolve_attr(dataset_module, ["load_realworldqa_dataset"]),
    "scienceqa": _resolve_attr(dataset_module, ["load_scienceqa_dataset"]),
    "pope": _resolve_attr(dataset_module, ["load_pope_dataset"]),
}

PROMPT_BUILDERS = {
    "mmbench": _resolve_prompt_builder("build_mmbench_prompt"),
    "hallusionbench": _resolve_prompt_builder("build_hallusionbench_prompt"),
    "ai2d": _resolve_prompt_builder("build_ai2d_prompt"),
    "mme": _resolve_prompt_builder("build_mme_prompt"),
    "realworldqa": _resolve_prompt_builder("build_realworldqa_prompt"),
    "scienceqa": _resolve_prompt_builder("build_scienceqa_prompt"),
    "pope": _resolve_prompt_builder("build_pope_prompt"),
}

EVAL_FNS = {
    "mmbench": eval_mmbench,
    "hallusionbench": eval_realworldqa,
    "ai2d": eval_realworldqa,
    "mme": eval_mme,
    "realworldqa": eval_realworldqa,
    "scienceqa": eval_realworldqa,
    "pope": eval_realworldqa,
}

MULTI_DATASET_FILES = {
    "MMBench_DEV_EN_V11": "MMBench_DEV_EN_V11.tsv",
    "AI2D_TEST": "AI2D_TEST.tsv",
    "RealWorldQA": "RealWorldQA.tsv",
    "ScienceQA_TEST": "ScienceQA_TEST.tsv",
    "POPE": "POPE.tsv",
}
DEFAULT_MULTI_DATASETS = ",".join(MULTI_DATASET_FILES.keys())
MULTI_DATASET_ALIASES = {
    "mmbench": "MMBench_DEV_EN_V11",
    "mmbenchdevenv11": "MMBench_DEV_EN_V11",
    "ai2d": "AI2D_TEST",
    "ai2dtest": "AI2D_TEST",
    "realworldqa": "RealWorldQA",
    "scienceqa": "ScienceQA_TEST",
    "scienceqatest": "ScienceQA_TEST",
    "pope": "POPE",
}


def _normalize_dataset_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _canonicalize_dataset_name(name: str) -> str:
    normalized = _normalize_dataset_name(name)
    canonical = MULTI_DATASET_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Unsupported dataset `{name}`. Expected one of: {', '.join(MULTI_DATASET_FILES.keys())}"
        )
    return canonical


def resolve_model_name(model_name: str | None, model_path: str, model_key: str) -> str:
    if model_name:
        return model_name
    path_name = Path(model_path).name
    return path_name if path_name else model_key


def resolve_dataset_specs(data_dir: str | None, lmudata_dir: str | None, datasets: str) -> list[dict]:
    if data_dir:
        benchmark = Path(data_dir).stem
        try:
            benchmark = _canonicalize_dataset_name(benchmark)
        except ValueError:
            pass
        return [{"benchmark": benchmark, "data_file": data_dir}]

    if not lmudata_dir:
        raise ValueError("Either data_dir or lmudata_dir must be provided")

    dataset_names = [item.strip() for item in datasets.split(",") if item.strip()]
    if not dataset_names:
        raise ValueError("datasets must not be empty when lmudata_dir is provided")

    specs = []
    for dataset_name in dataset_names:
        benchmark = _canonicalize_dataset_name(dataset_name)
        specs.append(
            {
                "benchmark": benchmark,
                "data_file": os.path.join(lmudata_dir, MULTI_DATASET_FILES[benchmark]),
            }
        )
    return specs


def build_dataset_output_dir(output_dir: str, model_name: str, benchmark: str, multi_dataset: bool) -> str:
    if not multi_dataset:
        return output_dir
    return os.path.join(output_dir, f"{model_name}_{benchmark}")


def write_run_outputs(output_dir: str, run_config: dict, results: list, metrics: list):
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False, default=json_default)
    with open(os.path.join(output_dir, "output.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)


def detect_dataset_key(data_dir: str) -> str:
    lower = data_dir.lower()
    dataset_aliases = {
        "mmbench": ["mmbench", "mmbench_dev_en_v11"],
        "hallusionbench": ["hallusionbench"],
        "ai2d": ["ai2d_test", "ai2d"],
        "mme": ["mme"],
        "realworldqa": ["realworldqa"],
        "scienceqa": ["scienceqa_test", "scienceqa"],
        "pope": ["pope"],
    }
    for key, aliases in dataset_aliases.items():
        if any(alias in lower for alias in aliases):
            return key
    raise ValueError(f"Unknown dataset in {data_dir}")


def sample_dataset_for_benchmark(dataset, sample_ratio: float, sample_seed: int):
    total = len(dataset)
    if total == 0:
        return dataset
    sample_size = max(1, int(total * sample_ratio))
    sample_size = min(total, sample_size)
    sampled = dataset.sample(n=sample_size, random_state=sample_seed).reset_index(drop=True)
    print(f"Benchmark sampling: {sample_size}/{total} samples ({sample_ratio:.1%}), seed={sample_seed}")
    return sampled


def _resolve_sample_id(row, idx):
    return row.get("id", row.get("index", idx))


def _resolve_source_index(row, idx):
    return row.get("__source_index__", idx)


def _maybe_restrict_dataset_for_profile(dataset, args):
    profile_sample_id = getattr(args, "profile_sample_id", None)
    if not getattr(args, "profile_timing", False) or profile_sample_id is None:
        return dataset

    target = str(profile_sample_id)
    matched_indices = []
    available_ids = []
    for idx, row in dataset.iterrows():
        sample_id = _resolve_sample_id(row, idx)
        sample_id_str = str(sample_id)
        if len(available_ids) < 10:
            available_ids.append(sample_id_str)
        if sample_id_str == target:
            matched_indices.append(idx)

    if not matched_indices:
        preview = ", ".join(available_ids) if available_ids else "<empty>"
        raise ValueError(
            f"profile_sample_id={target} not found in sampled dataset. "
            f"Available sampled ids (first 10): {preview}"
        )

    restricted = dataset.loc[matched_indices].reset_index(drop=True)
    print(
        "Profile timing active: restricting benchmark to "
        f"sample_id={target} ({len(restricted)}/{len(dataset)} sampled rows)"
    )
    return restricted


def _build_layer_ratio_map(numerator_entry, denominator_entry):
    numerator_layers = (numerator_entry or {}).get("layers", {})
    denominator_layers = (denominator_entry or {}).get("layers", {})
    layer_ids = sorted(
        set(numerator_layers.keys()) | set(denominator_layers.keys()),
        key=lambda item: int(item),
    )
    if not layer_ids:
        return None

    layer_ratios = {}
    for layer_id in layer_ids:
        numerator_ms = numerator_layers.get(layer_id, {}).get("total_ms")
        denominator_ms = denominator_layers.get(layer_id, {}).get("total_ms")
        entry = {
            "numerator_ms": numerator_ms,
            "denominator_ms": denominator_ms,
        }
        if numerator_ms is None:
            entry["ratio"] = None
            entry["reason"] = "missing numerator"
        elif denominator_ms is None:
            entry["ratio"] = None
            entry["reason"] = "missing denominator"
        elif float(denominator_ms) <= 0:
            entry["ratio"] = None
            entry["reason"] = "denominator is zero"
        else:
            entry["ratio"] = float(numerator_ms) / float(denominator_ms)
        layer_ratios[str(layer_id)] = entry
    return layer_ratios


def _build_ratio_summary(timings, numerator_key, denominator_key):
    numerator_entry = timings.get(numerator_key)
    denominator_entry = timings.get(denominator_key)
    numerator_ms = None if numerator_entry is None else numerator_entry.get("total_ms")
    denominator_ms = None if denominator_entry is None else denominator_entry.get("total_ms")

    summary = {
        "numerator_key": numerator_key,
        "denominator_key": denominator_key,
        "numerator_ms": numerator_ms,
        "denominator_ms": denominator_ms,
    }
    if numerator_ms is None:
        summary["ratio"] = None
        summary["reason"] = "missing numerator"
    elif denominator_ms is None:
        summary["ratio"] = None
        summary["reason"] = "missing denominator"
    elif float(denominator_ms) <= 0:
        summary["ratio"] = None
        summary["reason"] = "denominator is zero"
    else:
        summary["ratio"] = float(numerator_ms) / float(denominator_ms)

    layer_ratios = _build_layer_ratio_map(numerator_entry, denominator_entry)
    if layer_ratios:
        summary["layers"] = layer_ratios
    return summary


def _attach_overhead_summary(profile_result):
    if not profile_result:
        return profile_result

    timings = profile_result.get("timings", {})
    profile_result["overhead_summary"] = {
        "hybrid_attention_over_flashattention2": _build_ratio_summary(
            timings,
            "qwen.hybrid_attention.total",
            "qwen.hybrid_attention.flash_attn2",
        ),
        "prefill_search_offline_over_expert_compute": _build_ratio_summary(
            timings,
            "qwen.search.prefill.offline",
            "qwen.expert_compute.prefill",
        ),
        "decode_search_hybrid_over_expert_compute": _build_ratio_summary(
            timings,
            "qwen.search.decode.hybrid",
            "qwen.expert_compute.decode",
        ),
    }
    return profile_result


def _write_profile_output(profile_results, profile_output_path):
    payload = profile_results[0] if len(profile_results) == 1 else profile_results
    with open(profile_output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_default)
    print(f"Saved timing profile to {profile_output_path}")


def _attach_qwen_perf_profiler(model, profiler):
    if profiler is None:
        return

    outer_model = getattr(model, "model", None)
    language_model = getattr(outer_model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        return

    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        if hasattr(self_attn, "set_perf_profiler"):
            self_attn.set_perf_profiler(profiler)
        mlp = getattr(layer, "mlp", None)
        if hasattr(mlp, "set_perf_profiler"):
            mlp.set_perf_profiler(profiler)


def json_default(obj):
    if isinstance(obj, (torch.device, torch.dtype)):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _cast_floating_tensors(value, float_dtype):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(dtype=float_dtype)
        return value
    if isinstance(value, list):
        return [_cast_floating_tensors(item, float_dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_cast_floating_tensors(item, float_dtype) for item in value)
    if isinstance(value, dict):
        return {key: _cast_floating_tensors(item, float_dtype) for key, item in value.items()}
    return value


def _move_batch_to_device(batch, device, float_dtype=None):
    if hasattr(batch, "to"):
        batch = batch.to(device)

    if float_dtype is None:
        return batch

    if hasattr(batch, "items") and hasattr(batch, "__setitem__"):
        for key, value in list(batch.items()):
            batch[key] = _cast_floating_tensors(value, float_dtype)
        return batch

    return _cast_floating_tensors(batch, float_dtype)


def _build_deepseek_conversation(messages):
    conversation = []
    for msg in messages:
        role = msg.get("role", "user")
        content_list = msg.get("content", [])

        images = []
        text_parts = []
        for item in content_list:
            if isinstance(item, dict):
                if item.get("type") == "image":
                    images.append(item.get("image"))
                    text_parts.append("<image>")
                elif item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)

        conversation.append(
            {
                "role": "<|User|>" if role.lower() == "user" else "<|Assistant|>",
                "content": "".join(text_parts),
                "images": images if images else [],
            }
        )

    if not conversation or conversation[-1]["role"] != "<|Assistant|>":
        conversation.append({"role": "<|Assistant|>", "content": ""})
    return conversation


def _set_cache_env(args):
    if getattr(args, "cache_dir", None):
        os.environ["DS_CACHE_DIR"] = args.cache_dir
        os.environ["TABLEMOE_QWEN_CACHE_DIR"] = args.cache_dir
    if getattr(args, "pca_dir", None):
        os.environ["DS_CACHE_PCA_DIR"] = args.pca_dir
        os.environ["TABLEMOE_QWEN_PCA_DIR"] = args.pca_dir


def _apply_public_offload_overrides(args, mode: str):
    if mode not in {"adapmoe", "skip", "offline", "online", "tablemoe"}:
        return

    overrides = {}
    cache_ratio = getattr(args, "cache_ratio", None)
    keep_rate = getattr(args, "keep_rate", None)
    recomp_ratio = getattr(args, "recomp_ratio", None)
    if cache_ratio is not None:
        overrides["cache_ratio"] = cache_ratio
    if mode == "tablemoe":
        if keep_rate is not None:
            overrides["keep_rate"] = keep_rate
        elif recomp_ratio is not None:
            overrides["recomp_ratio"] = recomp_ratio
    if getattr(args, "cache_dir", None):
        overrides["cache_dir"] = args.cache_dir
    if getattr(args, "pca_dir", None):
        overrides["pca_dir"] = args.pca_dir

    if not overrides:
        return

    if args.model == "qwen3vlmoe":
        from table_moe.models.qwen3_vl_moe import update_offload_config

        update_offload_config(**overrides)
        return

    from table_moe.models.deepseek_vl2 import update_deepseek_offload_config

    update_deepseek_offload_config(**overrides)


def _default_output_dir(mode: str, model_key: str) -> str:
    return str(REPO_ROOT / "perf_results" / mode / model_key)


def _ensure_deepseek_repo(repo_path: str):
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    try:
        from deepseek_vl2.models import DeepseekVLV2Processor
        from deepseek_vl2.utils.io import load_pil_images
    except ImportError as exc:
        raise RuntimeError(f"Failed to import deepseek_vl2 from {repo_path}") from exc

    return DeepseekVLV2Processor, load_pil_images


def _build_runtime(args, mode: str):
    model_key = args.model
    spec = MODEL_SPECS[model_key]
    kwargs = {"device_map": "auto"}
    profiler = PerfProfileRecorder(
        enabled=bool(getattr(args, "profile_timing", False)),
        sample_id=getattr(args, "profile_sample_id", None),
    )

    _apply_public_offload_overrides(args, mode)

    if args.attn_implementation:
        if model_key == "qwen3vlmoe":
            kwargs["attn_implementation"] = args.attn_implementation
        elif model_key == "deepseekvl2" and mode in {"adapmoe", "skip"}:
            kwargs["attn_implementation"] = args.attn_implementation

    model, lang_cfg, expert_cache = build_model(
        model_family=spec["family"],
        mode=mode,
        model_id=args.model_path,
        **kwargs,
    )
    model.eval()

    if model_key == "qwen3vlmoe":
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        _attach_qwen_perf_profiler(model, profiler)
        return {
            "kind": model_key,
            "model": model,
            "lang_cfg": lang_cfg,
            "expert_cache": expert_cache,
            "processor": processor,
            "tokenizer": processor.tokenizer,
            "profiler": profiler,
            "profile_no_streamer_sync": bool(getattr(args, "profile_no_streamer_sync", False)),
        }

    DeepseekVLV2Processor, load_pil_images = _ensure_deepseek_repo(args.deepseek_repo)
    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    return {
        "kind": model_key,
        "model": model,
        "lang_cfg": lang_cfg,
        "expert_cache": expert_cache,
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "load_pil_images": load_pil_images,
        "profiler": profiler,
        "profile_no_streamer_sync": bool(getattr(args, "profile_no_streamer_sync", False)),
    }


def _get_offload_config_dump(model_key: str):
    if model_key == "qwen3vlmoe":
        from table_moe.models.qwen3_vl_moe import get_offload_config

        return get_offload_config()

    from table_moe.models.deepseek_vl2 import get_deepseek_offload_config

    return get_deepseek_offload_config()


def _get_cache_config_dump():
    from table_moe.cache_engine.config import CacheConfig

    return {k: v for k, v in vars(CacheConfig).items() if k.isupper() and not k.startswith("_")}


def _cleanup_runtime(runtime):
    if not runtime:
        return

    expert_cache = runtime.get("expert_cache")
    if expert_cache is not None and hasattr(expert_cache, "close"):
        try:
            expert_cache.close()
        except Exception as exc:
            print(f"[WARN] expert_cache cleanup failed: {exc}")

    model = runtime.get("model")
    if model is not None:
        del model

    for key in list(runtime.keys()):
        runtime[key] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _run_qwen_warmup(runtime, dataset, build_prompt, img_root, mode: str, warmup_samples: int):
    from qwen_vl_utils import process_vision_info

    model = runtime["model"]
    processor = runtime["processor"]
    num_warmup = min(warmup_samples, len(dataset))
    for idx in range(num_warmup):
        warmup_row = dataset.iloc[idx]
        warmup_msgs = build_prompt(warmup_row, img_root)
        warmup_text = processor.apply_chat_template(warmup_msgs, tokenize=False, add_generation_prompt=True)
        warmup_images, _ = process_vision_info(warmup_msgs)
        warmup_inputs = processor(
            text=[warmup_text],
            images=warmup_images,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        if mode in {"tablemoe", "offline", "online", "skip"}:
            ModalityContext.set_input_ids(warmup_inputs.input_ids)
        try:
            _ = model.generate(**warmup_inputs, max_new_tokens=10)
        finally:
            if mode in {"tablemoe", "offline", "online", "skip"}:
                ModalityContext.clear()


def _run_deepseek_warmup(runtime, dataset, build_prompt, img_root, mode: str, warmup_samples: int):
    model = runtime["model"]
    processor = runtime["processor"]
    tokenizer = runtime["tokenizer"]
    load_pil_images = runtime["load_pil_images"]

    num_warmup = min(warmup_samples, len(dataset))
    for idx in range(num_warmup):
        warmup_row = dataset.iloc[idx]
        warmup_msgs = build_prompt(warmup_row, img_root)

        warmup_conv = _build_deepseek_conversation(warmup_msgs)
        warmup_pil = load_pil_images(warmup_conv)
        warmup_inputs = processor(
            conversations=warmup_conv,
            images=warmup_pil,
            force_batchify=True,
            system_prompt="",
        ).to(model.device, dtype=model.dtype)

        if mode in {"tablemoe", "offline", "online", "skip"}:
            ModalityContext.set_input_ids(getattr(warmup_inputs, "input_ids", None))
        try:
            warmup_embeds = model.prepare_inputs_embeds(**warmup_inputs)
            _ = model.generate(
                inputs_embeds=warmup_embeds,
                attention_mask=warmup_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=10,
                do_sample=False,
            )
        finally:
            if mode in {"tablemoe", "offline", "online", "skip"}:
                ModalityContext.clear()


def _run_qwen_sample(runtime, row, build_prompt, img_root, mode: str, max_new_tokens: int, sample_id):
    from qwen_vl_utils import process_vision_info

    model = runtime["model"]
    processor = runtime["processor"]
    profiler = runtime.get("profiler")
    streamer_sync = True
    if profiler is not None and profiler.enabled:
        streamer_sync = not runtime.get("profile_no_streamer_sync", False)

    with profiler.measure("sample.prepare_inputs") if profiler is not None else nullcontext():
        messages = build_prompt(row, img_root)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_resize=True,
            max_pixels=5120 * 28 * 28,
            min_pixels=768 * 28 * 28,
            return_tensors="pt",
            **(video_kwargs or {}),
        ).to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    streamer = TimingStreamer(
        profiler=profiler,
        synchronize_cuda=streamer_sync,
    )
    if profiler is not None and profiler.is_active():
        profiler.set_value("sample.prompt_length", int(inputs.input_ids.shape[1]))

    if mode in {"tablemoe", "offline", "online", "skip"}:
        ModalityContext.set_input_ids(inputs.input_ids)
    try:
        with profiler.measure("sample.generate_total") if profiler is not None else nullcontext():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                streamer=streamer,
            )
    finally:
        if mode in {"tablemoe", "offline", "online", "skip"}:
            ModalityContext.clear()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    ttft, tpot, num_new_tokens = streamer.get_metrics()
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    with profiler.measure("sample.batch_decode") if profiler is not None else nullcontext():
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    if profiler is not None and profiler.is_active():
        profiler.set_value("streamer.ttft_ms", float(ttft) * 1000.0)
        profiler.set_value("streamer.tpot_ms", float(tpot) * 1000.0)
        profiler.set_value("streamer.num_new_tokens", int(num_new_tokens))

    return (
        {
            "id": int(sample_id) if isinstance(sample_id, int) else str(sample_id),
            "question": row.get("question", ""),
            "gt_answer": row.get("answer", ""),
            "prediction": output_text,
        },
        {
            "id": str(sample_id),
            "ttft": ttft,
            "tpot": tpot,
            "prompt_length": inputs.input_ids.shape[1],
            "num_new_tokens": num_new_tokens,
        },
    )


def _run_deepseek_sample(runtime, row, build_prompt, img_root, mode: str, max_new_tokens: int, sample_id):
    model = runtime["model"]
    processor = runtime["processor"]
    tokenizer = runtime["tokenizer"]
    load_pil_images = runtime["load_pil_images"]
    profiler = runtime.get("profiler")
    streamer_sync = True
    if profiler is not None and profiler.enabled:
        streamer_sync = not runtime.get("profile_no_streamer_sync", False)

    with profiler.measure("sample.prepare_inputs") if profiler is not None else nullcontext():
        messages = build_prompt(row, img_root)
        conversation = _build_deepseek_conversation(messages)
        pil_images = load_pil_images(conversation)

        prepare_inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        ).to(model.device, dtype=model.dtype)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    streamer = TimingStreamer(
        profiler=profiler,
        synchronize_cuda=streamer_sync,
    )
    if profiler is not None and profiler.is_active():
        profiler.set_value("sample.prompt_length", int(prepare_inputs.attention_mask.shape[1]))

    if mode in {"tablemoe", "offline", "online", "skip"}:
        ModalityContext.set_input_ids(getattr(prepare_inputs, "input_ids", None))

    try:
        with profiler.measure("sample.prepare_inputs_embeds") if profiler is not None else nullcontext():
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        with profiler.measure("sample.generate_total") if profiler is not None else nullcontext():
            generated_ids = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                streamer=streamer,
            )
    finally:
        if mode in {"tablemoe", "offline", "online", "skip"}:
            ModalityContext.clear()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    ttft, tpot, num_new_tokens = streamer.get_metrics()
    with profiler.measure("sample.batch_decode") if profiler is not None else nullcontext():
        output_text = tokenizer.decode(generated_ids[0].cpu().tolist(), skip_special_tokens=True)
    if profiler is not None and profiler.is_active():
        profiler.set_value("streamer.ttft_ms", float(ttft) * 1000.0)
        profiler.set_value("streamer.tpot_ms", float(tpot) * 1000.0)
        profiler.set_value("streamer.num_new_tokens", int(num_new_tokens))

    return (
        {
            "id": int(sample_id) if isinstance(sample_id, int) else str(sample_id),
            "question": row.get("question", ""),
            "gt_answer": row.get("answer", ""),
            "prediction": output_text,
        },
        {
            "id": str(sample_id),
            "ttft": ttft,
            "tpot": tpot,
            "prompt_length": prepare_inputs.attention_mask.shape[1],
            "num_new_tokens": num_new_tokens,
        },
    )


def build_parser(mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--lmudata-dir", type=str, default=None)
    parser.add_argument("--datasets", type=str, default=DEFAULT_MULTI_DATASETS)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--profile-sample-id", type=str, default=None)
    parser.add_argument("--profile-no-streamer-sync", action="store_true")
    parser.add_argument("--profile-output-json", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--cache-ratio", type=float, default=None)
    parser.add_argument("--keep-rate", type=float, default=None)
    parser.add_argument("--recomp-ratio", type=float, default=None)
    parser.add_argument("--deepseek-repo", type=str, default=str(REPO_ROOT / "third_party" / "DeepSeek-VL2"))
    if mode in {"tablemoe", "offline", "online"}:
        parser.add_argument("--cache-dir", type=str, required=True)
        parser.add_argument("--pca-dir", type=str, default=None)
    return parser


def _resolve_available_dataset_specs(dataset_specs: list[dict], multi_dataset: bool) -> list[dict]:
    available_specs = []
    for dataset_spec in dataset_specs:
        data_file = dataset_spec["data_file"]
        if os.path.exists(data_file):
            available_specs.append(dataset_spec)
            continue
        if multi_dataset:
            print(f"[WARN] dataset file not found, skip: {data_file}")
            continue
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    return available_specs


def _run_dataset_benchmark(
    runtime,
    args,
    mode: str,
    runtime_hardware_config,
    dataset_spec: dict,
    dataset_index: int,
    total_datasets: int,
    multi_dataset: bool,
):
    benchmark = dataset_spec["benchmark"]
    data_file = dataset_spec["data_file"]
    output_dir = build_dataset_output_dir(args.output_dir, args.model_name, benchmark, multi_dataset)
    os.makedirs(output_dir, exist_ok=True)

    dataset_key = detect_dataset_key(data_file)
    load_dataset = DATASET_LOADERS[dataset_key]
    build_prompt = PROMPT_BUILDERS[dataset_key]
    eval_fn = EVAL_FNS[dataset_key]

    print("")
    print("=" * 80)
    print(f"[{dataset_index}/{total_datasets}] {MODEL_SPECS[args.model]['label']} {mode} | Benchmark: {benchmark}")
    print("=" * 80)
    print(f"Data file: {data_file}")
    print(f"Output dir: {output_dir}")
    print(f"Loading {dataset_key} dataset from {data_file}...")
    dataset = load_dataset(data_file)
    if "__source_index__" not in dataset.columns:
        dataset = dataset.copy()
        dataset["__source_index__"] = dataset.index
    print(f"Loaded {len(dataset)} samples")
    profiler = runtime.get("profiler")
    if profiler is not None:
        profiler.clear_results()
    runtime["_profile_benchmark"] = benchmark

    parent_dir = os.path.dirname(data_file)
    img_root = os.path.join(parent_dir, "images")
    os.makedirs(img_root, exist_ok=True)

    if len(dataset) > 0 and args.warmup_samples > 0:
        try:
            if args.model == "qwen3vlmoe":
                _run_qwen_warmup(runtime, dataset, build_prompt, img_root, mode, args.warmup_samples)
            else:
                _run_deepseek_warmup(runtime, dataset, build_prompt, img_root, mode, args.warmup_samples)
            print("Warmup completed")
        except Exception as exc:
            print(f"Warmup failed (ignoring): {exc}")
            ModalityContext.clear()

    dataset = sample_dataset_for_benchmark(dataset, args.sample_ratio, args.sample_seed)
    dataset = _maybe_restrict_dataset_for_profile(dataset, args)
    results = []
    metrics = []
    profile_results = []

    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"{args.model}-{mode}-{benchmark}"):
        sample_id = _resolve_sample_id(row, idx)
        source_index = _resolve_source_index(row, idx)
        is_profile_active = False
        if profiler is not None:
            metadata = {
                "model": args.model,
                "mode": mode,
                "benchmark": benchmark,
                "sample_index": idx,
                "sample_id": str(sample_id),
                "source_index": int(source_index) if isinstance(source_index, int) else str(source_index),
                "max_new_tokens": int(args.max_new_tokens),
            }
            if mode in {"tablemoe", "offline", "online", "skip"}:
                metadata["cache_engine_config"] = _get_cache_config_dump()
            is_profile_active = profiler.begin_sample(sample_id, metadata=metadata)
        try:
            if args.model == "qwen3vlmoe":
                result_item, metric_item = _run_qwen_sample(
                    runtime,
                    row,
                    build_prompt,
                    img_root,
                    mode,
                    args.max_new_tokens,
                    sample_id,
                )
            else:
                result_item, metric_item = _run_deepseek_sample(
                    runtime,
                    row,
                    build_prompt,
                    img_root,
                    mode,
                    args.max_new_tokens,
                    sample_id,
                )

            results.append(result_item)
            if is_profile_active:
                profile_result = _attach_overhead_summary(profiler.finish_sample(status="ok"))
                if profile_result is not None:
                    metric_item["timing_profile"] = profile_result
                    profile_results.append(profile_result)
            metrics.append(metric_item)
        except Exception as exc:
            if is_profile_active:
                profiler.finish_sample(status="error", error=exc)
            print(
                f"Error processing sample {idx} "
                f"(sample_id={sample_id}, source_index={source_index}): {exc}"
            )
            traceback.print_exc()
            ModalityContext.clear()
            continue

    run_arguments = vars(args).copy()
    run_arguments["data_dir"] = data_file
    run_arguments["benchmark"] = benchmark
    run_arguments["output_dir"] = output_dir

    run_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "model": args.model,
        "model_name": args.model_name,
        "benchmark": benchmark,
        "model_family": MODEL_SPECS[args.model]["family"],
        "arguments": run_arguments,
        "runtime_hardware_config": runtime_hardware_config,
        "offload_system_config": _get_offload_config_dump(args.model),
    }
    if mode in {"tablemoe", "offline", "online", "skip"}:
        run_config["cache_engine_config"] = _get_cache_config_dump()

    write_run_outputs(output_dir, run_config, results, metrics)
    if args.profile_timing and profile_results:
        profile_output_path = args.profile_output_json
        if profile_output_path is None:
            profile_output_path = os.path.join(output_dir, "timing_profile.json")
        elif multi_dataset:
            base, ext = os.path.splitext(profile_output_path)
            suffix = ext or ".json"
            profile_output_path = f"{base}_{benchmark}{suffix}"
        _write_profile_output(profile_results, profile_output_path)

    print("Evaluating results...")
    eval_fn(results=results, output_dir=output_dir)


def run(mode: str):
    parser = build_parser(mode)
    args = parser.parse_args()
    runtime_hardware_config = configure_runtime_env()

    if not (0 < args.sample_ratio <= 1):
        raise ValueError(f"sample_ratio must be in (0, 1], got {args.sample_ratio}")
    if bool(args.data_dir) == bool(args.lmudata_dir):
        raise ValueError("Specify exactly one of --data-dir or --lmudata-dir")

    if not (0 < float(args.keep_rate if args.keep_rate is not None else args.recomp_ratio or 0.6) <= 1):
        raise ValueError("keep_rate/recomp_ratio must be in (0, 1]")

    if mode in {"tablemoe", "offline", "online"}:
        _set_cache_env(args)

    args.model_name = resolve_model_name(args.model_name, args.model_path, args.model)
    args.output_dir = args.output_dir or _default_output_dir(mode, args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    multi_dataset = args.lmudata_dir is not None
    dataset_specs = resolve_dataset_specs(args.data_dir, args.lmudata_dir, args.datasets)
    dataset_specs = _resolve_available_dataset_specs(dataset_specs, multi_dataset)
    if not dataset_specs:
        raise FileNotFoundError("No available dataset files to benchmark")

    print("\n" + "=" * 80)
    print(f"{MODEL_SPECS[args.model]['label']} {mode} test")
    print("=" * 80)
    print(f"model_name: {args.model_name}")
    print(f"output_dir: {args.output_dir}")
    print(f"datasets: {', '.join(spec['benchmark'] for spec in dataset_specs)}")
    print("")

    total_datasets = len(dataset_specs)
    runtime = None
    try:
        runtime = _build_runtime(args, mode)
        for dataset_index, dataset_spec in enumerate(dataset_specs, start=1):
            _run_dataset_benchmark(
                runtime,
                args,
                mode,
                runtime_hardware_config,
                dataset_spec,
                dataset_index,
                total_datasets,
                multi_dataset,
            )
    finally:
        _cleanup_runtime(runtime)
