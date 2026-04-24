import gc
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor

from common import MODEL_SPECS, _ensure_deepseek_repo
from moe_infinity_common import (
    _apply_model_dtype_to_config,
    _collect_checkpoint_weight_stats,
    _print_weight_stats,
    _repair_deepseek_direct_params,
    _repair_deepseek_resident_module_params,
    _resolve_memory_budget,
    resolve_image_root,
    run_moe_sample,
    run_moe_warmup,
    save_runtime_trace,
    validate_moe_runtime_args,
)
from table_moe.utils.perf_profile import PerfProfileRecorder


REPO_ROOT = Path(__file__).resolve().parents[1]
PREGATED_MOE_ROOT = REPO_ROOT / "third_party" / "pregated-moe"

PREGATED_MOE_DEFAULTS = {
    "qwen3vlmoe": {
        "offload_path": str(REPO_ROOT / "third_party" / "moe-infinity-qwen3vl"),
        "dtype": torch.float16,
    },
    "deepseekvl2": {
        "offload_path": str(REPO_ROOT / "third_party" / "moe-infinity-deepseekvl2"),
        "dtype": torch.float16,
    },
}


def _ensure_pregated_moe_path():
    path_str = str(PREGATED_MOE_ROOT)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def apply_pregated_runtime_defaults(args):
    if getattr(args, "offload_path", None) is None:
        args.offload_path = PREGATED_MOE_DEFAULTS[args.model]["offload_path"]
    if not getattr(args, "attn_implementation", None):
        args.attn_implementation = MODEL_SPECS[args.model]["default_attn"]
    if not hasattr(args, "prefetch"):
        args.prefetch = False
    if not hasattr(args, "trace_path"):
        args.trace_path = None
    if not hasattr(args, "trace_enabled"):
        args.trace_enabled = False
    if not hasattr(args, "deepseek_repo"):
        args.deepseek_repo = str(REPO_ROOT / "third_party" / "DeepSeek-VL2")
    return args


def cleanup_pregated_moe_runtime(runtime):
    if not runtime:
        return

    engine = runtime.get("engine")

    try:
        if engine is not None:
            engine.clean_up()
    except Exception as exc:
        print(f"[WARN] Pregated-MoE clean_up failed: {exc}")

    try:
        from pregated_moe.distributed import expert_executor as expert_executor_module

        if hasattr(expert_executor_module, "_expert_dispatcher"):
            expert_executor_module._expert_dispatcher = None
    except Exception as exc:
        print(f"[WARN] Pregated-MoE expert dispatcher cleanup failed: {exc}")

    try:
        if (
            engine is not None
            and getattr(engine, "archer_engine", None) is not None
            and hasattr(engine.archer_engine, "clean_up_resources")
        ):
            engine.archer_engine.clean_up_resources()
    except Exception as exc:
        print(f"[WARN] Pregated-MoE native resource cleanup failed: {exc}")

    for key in list(runtime.keys()):
        runtime[key] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_pregated_moe_runtime(args):
    apply_pregated_runtime_defaults(args)
    _ensure_pregated_moe_path()
    from pregated_moe.common.constants import (
        MODEL_MAPPING_NAMES,
        ensure_local_deepseek_vl2_repo,
        resolve_model_architecture,
    )
    from pregated_moe.runtime import OffloadEngine
    from pregated_moe.utils import ArcherConfig, get_checkpoint_paths

    model_dtype = PREGATED_MOE_DEFAULTS[args.model]["dtype"]
    if args.model == "deepseekvl2":
        ensure_local_deepseek_vl2_repo()
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import (
            DeepseekVLV2Config,
            DeepseekVLV2ForCausalLM,
        )

        model_config = DeepseekVLV2Config.from_pretrained(
            args.model_path,
        )
        _apply_model_dtype_to_config(model_config, model_dtype)
        arch = "deepseek_vl2"
        model_cls = DeepseekVLV2ForCausalLM
    else:
        model_config = AutoConfig.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        _apply_model_dtype_to_config(model_config, model_dtype)
        arch = resolve_model_architecture(model_config)
        model_cls = MODEL_MAPPING_NAMES[arch]

    checkpoint_paths = get_checkpoint_paths(args.model_path)
    weight_stats = _collect_checkpoint_weight_stats(checkpoint_paths, args.model)
    budget_meta = _resolve_memory_budget(args, weight_stats)
    args.device_memory_ratio = float(budget_meta["device_memory_ratio"])

    config_json = {
        "offload_path": args.offload_path,
        "device_memory_ratio": budget_meta["device_memory_ratio"],
        "trace_capacity": args.trace_capacity,
        "prefetch": False,
    }
    if getattr(args, "trace_path", None):
        config_json["trace_path"] = args.trace_path
    if getattr(args, "num_threads", None) is not None:
        config_json["num_threads"] = args.num_threads

    engine_config = ArcherConfig.load_from_json(config_json)
    engine = OffloadEngine(engine_config.trace_capacity, model_config)
    profiler = PerfProfileRecorder(
        enabled=bool(getattr(args, "profile_timing", False)),
        sample_id=getattr(args, "profile_sample_id", None),
    )
    engine.perf_profiler = profiler
    engine.ckpt_files = checkpoint_paths

    print("\n" + "=" * 80)
    print(f"Building Pregated-MoE runtime for {MODEL_SPECS[args.model]['label']}")
    print("=" * 80)
    print(f"model_path: {args.model_path}")
    print(f"offload_path: {args.offload_path}")
    print("device_map: auto")
    print(f"attn_implementation: {args.attn_implementation}")
    print(f"model_dtype: {model_dtype}")
    if budget_meta["requested_expert_cache_ratio"] is not None:
        print(
            f"requested expert_cache_ratio: {budget_meta['requested_expert_cache_ratio']}"
        )
    print(f"device_memory_ratio: {budget_meta['device_memory_ratio']}")
    print(f"num_threads: {engine_config.num_threads}")

    with engine.init(cls=model_cls, ar_config=engine_config):
        model = model_cls.from_pretrained(
            args.model_path,
            attn_implementation=args.attn_implementation,
            is_flash_attn_available=(
                args.attn_implementation == "flash_attention_2"
            ),
            torch_dtype=model_dtype,
            trust_remote_code=(args.model != "deepseekvl2"),
            device_map="auto",
        )

    if args.model == "deepseekvl2":
        _repair_deepseek_direct_params(model, checkpoint_paths)
        _repair_deepseek_resident_module_params(model, checkpoint_paths)

    model.eval()
    _print_weight_stats(weight_stats, budget_meta)

    runtime_meta = {
        "architecture": arch,
        "device_map": "auto",
        "attn_implementation": args.attn_implementation,
        "num_threads": engine_config.num_threads,
        "device_memory_ratio": budget_meta["device_memory_ratio"],
        "expert_cache_ratio": budget_meta["expert_cache_ratio"],
        "prefetch": False,
        "trace_enabled": False,
        "trace_path": getattr(args, "trace_path", None),
        "trace_capacity": args.trace_capacity,
        "gpu_total_bytes": budget_meta["gpu_total_bytes"],
        "gpu_total_source": budget_meta["gpu_total_source"],
        "profile_timing": bool(getattr(args, "profile_timing", False)),
        "profile_sample_id": getattr(args, "profile_sample_id", None),
        "profile_no_streamer_sync": bool(
            getattr(args, "profile_no_streamer_sync", False)
        ),
        "simulated_pregated": True,
        "route_variant": "dual_gate_proxy",
        "prepare_mode": "layer_ordered_exact_queue",
    }

    if args.model == "qwen3vlmoe":
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        return {
            "kind": args.model,
            "model": model,
            "processor": processor,
            "tokenizer": processor.tokenizer,
            "engine": engine,
            "profiler": profiler,
            "weight_stats": weight_stats,
            "runtime_meta": runtime_meta,
        }

    DeepseekVLV2Processor, load_pil_images = _ensure_deepseek_repo(
        args.deepseek_repo
    )
    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    return {
        "kind": args.model,
        "model": model,
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "load_pil_images": load_pil_images,
        "engine": engine,
        "profiler": profiler,
        "weight_stats": weight_stats,
        "runtime_meta": runtime_meta,
    }
