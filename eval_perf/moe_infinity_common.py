import json
import math
import os
import struct
import sys
import time
import gc
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor

from common import (
    MODEL_SPECS,
    _build_deepseek_conversation,
    _ensure_deepseek_repo,
)
from table_moe.utils.perf_profile import PerfProfileRecorder
from table_moe.utils.timestreamer import StopWatch


REPO_ROOT = Path(__file__).resolve().parents[1]
MOE_INFINITY_ROOT = REPO_ROOT / "third_party" / "moe-infinity"
DEFAULT_GPU_TOTAL_MEMORY_MIB = 81920

MOE_INFINITY_DEFAULTS = {
    "qwen3vlmoe": {
        "offload_path": str(REPO_ROOT / "third_party" / "moe-infinity-qwen3vl"),
        "dtype": torch.float16,
        "sparse_name_key": ".mlp.experts.",
        "dense_force_keys": [],
    },
    "deepseekvl2": {
        "offload_path": str(REPO_ROOT / "third_party" / "moe-infinity-deepseekvl2"),
        "dtype": torch.float16,
        "sparse_name_key": ".mlp.experts.",
        "dense_force_keys": [
            "shared_experts",
            "vision",
            "projector",
            "image_newline",
            "view_seperator",
            "aligner",
        ],
    },
}

SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

EXPERT_DISPATCHER_PERF_DURATION_KEYS = [
    "expert_dispatcher.enqueue_node_lock_wait",
    "expert_dispatcher.fetch_queue_wait",
    "expert_dispatcher.fetch_cache_evict",
    "expert_dispatcher.fetch_set_device",
    "expert_dispatcher.fetch_slice_input",
    "expert_dispatcher.exec_queue_wait",
    "expert_dispatcher.exec_forward",
    "expert_dispatcher.output_to_device",
    "expert_dispatcher.output_queue_push",
]

EXPERT_DISPATCHER_PERF_COUNTER_KEYS = [
    "expert_dispatcher.cache_hit",
    "expert_dispatcher.cache_miss",
    "expert_dispatcher.evict",
    "expert_dispatcher.output_count",
]


def apply_moe_runtime_defaults(args):
    if getattr(args, "offload_path", None) is None:
        args.offload_path = MOE_INFINITY_DEFAULTS[args.model]["offload_path"]
    if not getattr(args, "attn_implementation", None):
        args.attn_implementation = MODEL_SPECS[args.model]["default_attn"]
    if not hasattr(args, "prefetch"):
        args.prefetch = False
    if not hasattr(args, "trace_path"):
        args.trace_path = None
    if not hasattr(args, "trace_enabled"):
        args.trace_enabled = bool(
            getattr(args, "prefetch", False)
            or getattr(args, "save_trace_path", None)
        )
    if not hasattr(args, "deepseek_repo"):
        args.deepseek_repo = str(REPO_ROOT / "third_party" / "DeepSeek-VL2")
    return args


def validate_moe_runtime_args(args):
    sample_ratio = getattr(args, "sample_ratio", None)
    if sample_ratio is not None and not (0 < sample_ratio <= 1):
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")

    expert_cache_ratio = getattr(args, "expert_cache_ratio", None)
    if expert_cache_ratio is not None and not (0 <= expert_cache_ratio <= 1):
        raise ValueError(
            f"expert_cache_ratio must be in [0, 1], got {expert_cache_ratio}"
        )

    if expert_cache_ratio is None:
        device_memory_ratio = getattr(args, "device_memory_ratio", None)
        if device_memory_ratio is None or not (0 < device_memory_ratio <= 1):
            raise ValueError(
                f"device_memory_ratio must be in (0, 1], got {device_memory_ratio}"
            )

    gpu_total_memory_mib = getattr(args, "gpu_total_memory_mib", None)
    if gpu_total_memory_mib is not None and gpu_total_memory_mib <= 0:
        raise ValueError(
            f"gpu_total_memory_mib must be positive, got {gpu_total_memory_mib}"
        )

    trace_capacity = getattr(args, "trace_capacity", None)
    if trace_capacity is not None and trace_capacity <= 0:
        raise ValueError(f"trace_capacity must be positive, got {trace_capacity}")

    max_new_tokens = getattr(args, "max_new_tokens", None)
    if max_new_tokens is not None and max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")


def resolve_image_root(data_path: str) -> str:
    parent_dir = os.path.dirname(data_path)
    img_root = os.path.join(parent_dir, "images")
    os.makedirs(img_root, exist_ok=True)
    return img_root


def save_runtime_trace(runtime, path: str, num_entries: int | None = None):
    tracer = runtime["engine"].expert_tracer
    if num_entries is None:
        data = tracer.trace_collection.cpu().numpy()
    else:
        if num_entries < 0 or num_entries > int(tracer.trace_collection.shape[0]):
            raise ValueError(
                f"Invalid trace entry count {num_entries}, capacity is "
                f"{int(tracer.trace_collection.shape[0])}"
            )
        data = tracer.trace_collection[:num_entries].cpu().numpy()

    np.save(path, data)
    print(f"Saved trace ({data.shape}) to {path}", flush=True)


def _begin_runtime_generation_trace(runtime, batch_size: int):
    engine = runtime.get("engine")
    if engine is None or not hasattr(engine, "begin_generation_trace"):
        return None
    return engine.begin_generation_trace(batch_size)


def _end_runtime_generation_trace(runtime, seq_id_list, persist: bool):
    engine = runtime.get("engine")
    if engine is None or not hasattr(engine, "end_generation_trace"):
        return
    engine.end_generation_trace(seq_id_list, persist=persist)


def _get_expert_executor(runtime):
    engine = runtime.get("engine")
    if engine is None:
        return None
    return getattr(engine, "expert_executor", None)


def _clear_expert_dispatcher_perf_stats(runtime):
    executor = _get_expert_executor(runtime)
    if executor is not None and hasattr(executor, "clear_perf_stats"):
        executor.clear_perf_stats()


def _record_expert_dispatcher_perf_stats(runtime, profiler):
    if profiler is None or not profiler.is_active():
        return

    executor = _get_expert_executor(runtime)
    if executor is None or not hasattr(executor, "get_perf_stats"):
        return

    stats = executor.get_perf_stats()
    if not stats:
        return

    expected_size = len(EXPERT_DISPATCHER_PERF_DURATION_KEYS) + len(
        EXPERT_DISPATCHER_PERF_COUNTER_KEYS
    )
    if len(stats) < expected_size:
        return

    for idx, key in enumerate(EXPERT_DISPATCHER_PERF_DURATION_KEYS):
        profiler.add_duration(key, float(stats[idx]) / 1_000_000.0)

    counter_offset = len(EXPERT_DISPATCHER_PERF_DURATION_KEYS)
    for idx, key in enumerate(EXPERT_DISPATCHER_PERF_COUNTER_KEYS):
        profiler.increment_counter(key, amount=int(stats[counter_offset + idx]))

    cache_hit = int(stats[counter_offset + 0])
    cache_miss = int(stats[counter_offset + 1])
    cache_total = cache_hit + cache_miss
    profiler.set_value("expert_dispatcher.cache_total", cache_total)
    if cache_total > 0:
        profiler.set_value(
            "expert_dispatcher.cache_hit_rate",
            float(cache_hit) / float(cache_total),
        )


def clear_runtime_trace(runtime):
    tracer = runtime["engine"].expert_tracer
    tracer.trace.clear()
    tracer.trace_collection.zero_()
    tracer.collection_access.fill(0)
    tracer.persistent_capacity = 0


def cleanup_moe_runtime(runtime):
    if not runtime:
        return

    engine = runtime.get("engine")

    try:
        if engine is not None:
            engine.clean_up()
    except Exception as exc:
        print(f"[WARN] MoE-Infinity clean_up failed: {exc}")

    try:
        from moe_infinity.distributed import expert_executor as expert_executor_module

        if hasattr(expert_executor_module, "_expert_dispatcher"):
            expert_executor_module._expert_dispatcher = None
    except Exception as exc:
        print(f"[WARN] MoE-Infinity expert dispatcher cleanup failed: {exc}")

    try:
        if (
            engine is not None
            and getattr(engine, "archer_engine", None) is not None
            and hasattr(engine.archer_engine, "clean_up_resources")
        ):
            engine.archer_engine.clean_up_resources()
    except Exception as exc:
        print(f"[WARN] MoE-Infinity native resource cleanup failed: {exc}")

    for key in list(runtime.keys()):
        runtime[key] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _apply_model_dtype_to_config(config, model_dtype: torch.dtype):
    if hasattr(config, "torch_dtype"):
        config.torch_dtype = model_dtype
    for attr_name in ("text_config", "language_config", "vision_config"):
        nested = getattr(config, attr_name, None)
        if nested is not None and hasattr(nested, "torch_dtype"):
            nested.torch_dtype = model_dtype


def _ensure_moe_infinity_path():
    path_str = str(MOE_INFINITY_ROOT)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _format_bytes(num_bytes: int) -> str:
    gib = num_bytes / (1024 ** 3)
    mib = num_bytes / (1024 ** 2)
    return f"{gib:.3f} GiB ({mib:.1f} MiB, {num_bytes} bytes)"


def _is_sparse_tensor_name(model_key: str, name: str) -> bool:
    defaults = MOE_INFINITY_DEFAULTS[model_key]
    if any(key in name for key in defaults["dense_force_keys"]):
        return False
    return defaults["sparse_name_key"] in name


def _iter_safetensors_entries(shard_path: str):
    with open(shard_path, "rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))

    for name, info in header.items():
        if name == "__metadata__":
            continue
        yield name, info


def _entry_nbytes(entry_info) -> int:
    dtype = str(entry_info["dtype"]).upper()
    item_size = SAFETENSORS_DTYPE_BYTES.get(dtype)
    if item_size is None:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")
    return int(math.prod(entry_info["shape"])) * item_size


def _collect_checkpoint_weight_stats(checkpoint_paths, model_key: str):
    stats = {
        "dense_bytes": 0,
        "sparse_bytes": 0,
        "total_bytes": 0,
        "dense_tensor_count": 0,
        "sparse_tensor_count": 0,
        "total_tensor_count": 0,
        "source": "checkpoint",
    }

    for shard_path in checkpoint_paths:
        if not str(shard_path).endswith(".safetensors"):
            continue
        for name, entry_info in _iter_safetensors_entries(shard_path):
            tensor_bytes = _entry_nbytes(entry_info)
            stats["total_bytes"] += tensor_bytes
            stats["total_tensor_count"] += 1

            if _is_sparse_tensor_name(model_key, name):
                stats["sparse_bytes"] += tensor_bytes
                stats["sparse_tensor_count"] += 1
            else:
                stats["dense_bytes"] += tensor_bytes
                stats["dense_tensor_count"] += 1

    return stats


def _resolve_gpu_total_memory_bytes(args):
    if getattr(args, "gpu_total_memory_mib", None) is not None:
        mib = int(args.gpu_total_memory_mib)
        return mib * (1024 ** 2), f"arg:{mib}MiB"

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        visible_total = sum(
            torch.cuda.get_device_properties(idx).total_memory
            for idx in range(torch.cuda.device_count())
        )
        return int(visible_total), "visible_cuda"

    return int(DEFAULT_GPU_TOTAL_MEMORY_MIB * (1024 ** 2)), (
        f"default:{DEFAULT_GPU_TOTAL_MEMORY_MIB}MiB"
    )


def _resolve_memory_budget(args, weight_stats):
    gpu_total_bytes, gpu_total_source = _resolve_gpu_total_memory_bytes(args)
    expert_cache_ratio = getattr(args, "expert_cache_ratio", None)

    if expert_cache_ratio is None:
        device_memory_ratio = float(args.device_memory_ratio)
        pooled_budget_bytes = int(round(gpu_total_bytes * device_memory_ratio))
        target_sparse_cache_bytes = max(
            0,
            pooled_budget_bytes - int(weight_stats["dense_bytes"]),
        )
        effective_expert_cache_ratio = (
            target_sparse_cache_bytes / float(weight_stats["sparse_bytes"])
            if weight_stats["sparse_bytes"] > 0
            else 0.0
        )
        effective_expert_cache_ratio = max(0.0, min(1.0, effective_expert_cache_ratio))
        return {
            "mode": "device_memory_ratio",
            "device_memory_ratio": device_memory_ratio,
            "expert_cache_ratio": effective_expert_cache_ratio,
            "requested_expert_cache_ratio": None,
            "gpu_total_bytes": gpu_total_bytes,
            "gpu_total_source": gpu_total_source,
            "pooled_budget_bytes": pooled_budget_bytes,
            "target_sparse_cache_bytes": target_sparse_cache_bytes,
        }

    dense_bytes = int(weight_stats["dense_bytes"])
    sparse_bytes = int(weight_stats["sparse_bytes"])
    target_sparse_cache_bytes = int(round(float(expert_cache_ratio) * sparse_bytes))
    pooled_budget_bytes = dense_bytes + target_sparse_cache_bytes
    if pooled_budget_bytes > gpu_total_bytes:
        raise ValueError(
            "Requested expert_cache_ratio exceeds available GPU memory budget: "
            f"dense={_format_bytes(dense_bytes)}, "
            f"target_sparse={_format_bytes(target_sparse_cache_bytes)}, "
            f"gpu_total={_format_bytes(gpu_total_bytes)}"
        )

    device_memory_ratio = pooled_budget_bytes / float(gpu_total_bytes)
    return {
        "mode": "expert_cache_ratio",
        "device_memory_ratio": device_memory_ratio,
        "expert_cache_ratio": float(expert_cache_ratio),
        "requested_expert_cache_ratio": float(expert_cache_ratio),
        "gpu_total_bytes": gpu_total_bytes,
        "gpu_total_source": gpu_total_source,
        "pooled_budget_bytes": pooled_budget_bytes,
        "target_sparse_cache_bytes": target_sparse_cache_bytes,
    }


def _print_weight_stats(weight_stats, budget_meta):
    print("\n" + "=" * 80)
    print("MoE-Infinity Weight Stats")
    print("=" * 80)
    print(f"Dense weights : {_format_bytes(weight_stats['dense_bytes'])}")
    print(f"Sparse weights: {_format_bytes(weight_stats['sparse_bytes'])}")
    print(f"Total weights : {_format_bytes(weight_stats['total_bytes'])}")
    print(
        "Tensor counts : "
        f"dense={weight_stats['dense_tensor_count']}, "
        f"sparse={weight_stats['sparse_tensor_count']}, "
        f"total={weight_stats['total_tensor_count']}"
    )
    if weight_stats.get("source"):
        print(f"Weight source  : {weight_stats['source']}")

    gpu_total_bytes = int(budget_meta["gpu_total_bytes"])
    print(
        f"GPU total memory : {_format_bytes(gpu_total_bytes)} "
        f"(source={budget_meta['gpu_total_source']})"
    )
    print(f"Budget mode : {budget_meta['mode']}")
    print(f"device_memory_ratio : {float(budget_meta['device_memory_ratio']):.6f}")
    print(f"effective expert_cache_ratio : {float(budget_meta['expert_cache_ratio']):.6f}")
    print(f"Dense planning bytes  : {_format_bytes(weight_stats['dense_bytes'])}")
    print(f"Target sparse cache   : {_format_bytes(budget_meta['target_sparse_cache_bytes'])}")
    print(f"Pool capacity target  : {_format_bytes(budget_meta['pooled_budget_bytes'])}")
    print("")


def _align_deepseek_direct_params(model, target_device):
    target_dtype = getattr(model, "dtype", None)
    direct_param_names = ("image_newline", "view_seperator", "tile_indicators")
    for name in direct_param_names:
        param = getattr(model, name, None)
        if param is None or not isinstance(param, torch.nn.Parameter):
            continue
        if param.device == target_device and (
            target_dtype is None or param.dtype == target_dtype
        ):
            continue
        param.data = param.data.to(device=target_device, dtype=target_dtype)


def _is_deepseek_resident_name(name: str) -> bool:
    fixed_prefixes = (
        "vision",
        "projector",
        "aligner",
    )
    if any(name == prefix or name.startswith(f"{prefix}.") for prefix in fixed_prefixes):
        return True

    return False


def _is_deepseek_placeholder_tensor(tensor) -> bool:
    return bool(getattr(tensor, "is_meta", False) or tensor.numel() <= 1)


def _get_deepseek_language_device(model):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")

    language = getattr(model, "language", None)
    if language is not None:
        embeddings = language.get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        if torch.is_tensor(weight):
            return weight.device
        try:
            return next(language.parameters()).device
        except StopIteration:
            pass
    return getattr(model, "device", torch.device("cpu"))


def _align_deepseek_resident_module_params(
    model,
    target_device,
    target_dtype=None,
):
    for name, param in model.named_parameters():
        if not _is_deepseek_resident_name(name):
            continue
        if param.device == target_device and (
            target_dtype is None or param.dtype == target_dtype
        ):
            continue
        param.data = param.data.to(device=target_device, dtype=target_dtype)

    for name, buffer in model.named_buffers():
        if not _is_deepseek_resident_name(name):
            continue
        if buffer.device == target_device and (
            target_dtype is None or buffer.dtype == target_dtype
        ):
            continue
        buffer.data = buffer.data.to(device=target_device, dtype=target_dtype)


def _load_named_tensors_from_checkpoints(checkpoint_paths, target_names):
    loaded_tensors = {}
    remaining = set(target_names)

    for ckpt in checkpoint_paths:
        if not remaining:
            break

        if str(ckpt).endswith(".safetensors"):
            for name, _ in _iter_safetensors_entries(ckpt):
                matched_name = next(
                    (
                        target
                        for target in remaining
                        if name == target or name.endswith(f".{target}")
                    ),
                    None,
                )
                if matched_name is None:
                    continue
                from safetensors import safe_open

                with safe_open(ckpt, framework="pt", device="cpu") as handle:
                    loaded_tensors[matched_name] = handle.get_tensor(name)
                remaining.remove(matched_name)
        else:
            state_dict = torch.load(ckpt, map_location="cpu")
            for name in list(remaining):
                matched_name = next(
                    (
                        key
                        for key in state_dict.keys()
                        if key == name or key.endswith(f".{name}")
                    ),
                    None,
                )
                if matched_name is None:
                    continue
                loaded_tensors[name] = state_dict[matched_name]
                remaining.remove(name)

    return loaded_tensors


def _get_deepseek_expected_direct_param_shapes(model):
    config = model.config
    tile_tag = getattr(model, "tile_tag", getattr(config, "tile_tag", None))
    expected_shapes = {}

    if tile_tag == "2D":
        n_embed = int(config.projector_config.n_embed)
        expected_shapes["image_newline"] = (n_embed,)
        expected_shapes["view_seperator"] = (n_embed,)
    elif tile_tag == "1D":
        tile_variants_num = len(config.candidate_resolutions)
        n_embed = int(config.aligner.params.n_embed)
        expected_shapes["tile_indicators"] = (tile_variants_num + 1, n_embed)
    else:
        raise RuntimeError(f"Unsupported DeepSeek-VL2 tile_tag: {tile_tag}")

    return expected_shapes


def _repair_deepseek_direct_params(model, checkpoint_paths):
    expected_shapes = _get_deepseek_expected_direct_param_shapes(model)
    tensors_to_restore = []

    for name, expected_shape in expected_shapes.items():
        param = getattr(model, name, None)
        if not isinstance(param, torch.nn.Parameter):
            raise RuntimeError(
                f"DeepSeek-VL2 model is missing direct parameter `{name}`."
            )
        if tuple(param.shape) != tuple(expected_shape):
            tensors_to_restore.append(name)

    if tensors_to_restore:
        restored = _load_named_tensors_from_checkpoints(
            checkpoint_paths,
            tuple(tensors_to_restore),
        )
        for name in tensors_to_restore:
            if name not in restored:
                raise RuntimeError(
                    f"Missing DeepSeek-VL2 direct parameter `{name}` in checkpoints."
                )
            tensor = restored[name]
            expected_shape = expected_shapes[name]
            if tuple(tensor.shape) != tuple(expected_shape):
                raise RuntimeError(
                    f"DeepSeek-VL2 direct parameter `{name}` has checkpoint shape "
                    f"{tuple(tensor.shape)}, expected {tuple(expected_shape)}."
                )
            param = getattr(model, name)
            param.data = tensor.to(device=param.device, dtype=param.dtype)

    debug_shapes = {
        name: tuple(getattr(model, name).shape)
        for name in expected_shapes
    }
    for name, expected_shape in expected_shapes.items():
        if debug_shapes[name] != tuple(expected_shape):
            raise RuntimeError(
                f"DeepSeek-VL2 direct parameter `{name}` restored to shape "
                f"{debug_shapes[name]}, expected {tuple(expected_shape)}."
            )
    print(f"[DeepSeek] direct param shapes: {debug_shapes}")


def _repair_deepseek_resident_module_params(model, checkpoint_paths):
    always_restore_prefixes = (
        "vision",
        "projector",
        "aligner",
    )
    params_to_restore = {}

    for name, param in model.named_parameters():
        if not _is_deepseek_resident_name(name):
            continue
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in always_restore_prefixes
        ) or _is_deepseek_placeholder_tensor(param):
            params_to_restore[name] = param

    buffers_to_restore = {}
    for name, buffer in model.named_buffers():
        if not _is_deepseek_resident_name(name):
            continue
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in always_restore_prefixes
        ) or _is_deepseek_placeholder_tensor(buffer):
            buffers_to_restore[name] = buffer

    target_names = tuple(params_to_restore) + tuple(buffers_to_restore)
    if not target_names:
        return

    restored = _load_named_tensors_from_checkpoints(
        checkpoint_paths,
        target_names,
    )

    missing = [
        name for name in target_names if name not in restored
    ]
    if missing:
        raise RuntimeError(
            "Missing DeepSeek-VL2 resident parameter(s) in checkpoints: "
            + ", ".join(sorted(missing[:10]))
            + (" ..." if len(missing) > 10 else "")
        )

    model_device = _get_deepseek_language_device(model)
    restored_shapes = {}

    for name, param in params_to_restore.items():
        tensor = restored[name]
        target_device = model_device
        param.data = tensor.to(device=target_device, dtype=param.dtype)
        restored_shapes[name] = tuple(param.shape)

    for name, buffer in buffers_to_restore.items():
        tensor = restored[name]
        target_device = model_device
        buffer.data = tensor.to(device=target_device, dtype=buffer.dtype)
        restored_shapes[name] = tuple(buffer.shape)

    print(
        "[DeepSeek] restored resident tensors: "
        f"{len(restored_shapes)}"
    )


def build_moe_runtime(args):
    apply_moe_runtime_defaults(args)
    _ensure_moe_infinity_path()
    from moe_infinity.common.constants import (
        MODEL_MAPPING_NAMES,
        ensure_local_deepseek_vl2_repo,
        resolve_model_architecture,
    )
    from moe_infinity.runtime import OffloadEngine
    from moe_infinity.utils import ArcherConfig, get_checkpoint_paths

    if getattr(args, "trace_capacity", None) is None:
        raise ValueError("trace_capacity must be set before building MoE-Infinity runtime.")

    trace_path = getattr(args, "trace_path", None)
    prefetch = bool(getattr(args, "prefetch", False))
    trace_enabled = bool(getattr(args, "trace_enabled", False))
    num_threads = getattr(args, "num_threads", None)

    model_dtype = MOE_INFINITY_DEFAULTS[args.model]["dtype"]
    if args.model == "deepseekvl2":
        ensure_local_deepseek_vl2_repo()
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import (
            DeepseekVLV2Config,
            DeepseekVLV2ForCausalLM,
        )

        model_config = DeepseekVLV2Config.from_pretrained(
            args.model_path,
            trust_remote_code=True,
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
        "prefetch": prefetch,
    }
    if trace_path is not None:
        config_json["trace_path"] = trace_path
    if num_threads is not None:
        config_json["num_threads"] = num_threads
    engine_config = ArcherConfig.load_from_json(config_json)
    engine = OffloadEngine(engine_config.trace_capacity, model_config)
    profiler = PerfProfileRecorder(
        enabled=bool(getattr(args, "profile_timing", False)),
        sample_id=getattr(args, "profile_sample_id", None),
    )
    engine.perf_profiler = profiler
    engine.ckpt_files = checkpoint_paths

    print("\n" + "=" * 80)
    print(f"Building MoE-Infinity runtime for {MODEL_SPECS[args.model]['label']}")
    print("=" * 80)
    print(f"model_path: {args.model_path}")
    print(f"offload_path: {args.offload_path}")
    print("device_map: auto")
    print(f"attn_implementation: {args.attn_implementation}")
    print(f"model_dtype: {model_dtype}")
    if budget_meta["requested_expert_cache_ratio"] is not None:
        print(f"requested expert_cache_ratio: {budget_meta['requested_expert_cache_ratio']}")
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
            trust_remote_code=True,
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
        "prefetch": prefetch,
        "trace_enabled": trace_enabled,
        "trace_path": trace_path,
        "trace_capacity": args.trace_capacity,
        "gpu_total_bytes": budget_meta["gpu_total_bytes"],
        "gpu_total_source": budget_meta["gpu_total_source"],
        "profile_timing": bool(getattr(args, "profile_timing", False)),
        "profile_sample_id": getattr(args, "profile_sample_id", None),
        "profile_no_streamer_sync": bool(
            getattr(args, "profile_no_streamer_sync", False)
        ),
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


def _prepare_qwen_inputs(runtime, row, build_prompt, img_root, padding: bool):
    from qwen_vl_utils import process_vision_info

    model = runtime["model"]
    processor = runtime["processor"]
    profiler = runtime.get("profiler")
    profile_enabled = profiler is not None and profiler.is_active()

    messages = build_prompt(row, img_root)
    start_time = time.perf_counter()
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if profile_enabled:
        profiler.add_duration(
            "qwen_prepare.apply_chat_template",
            time.perf_counter() - start_time,
        )
    start_time = time.perf_counter()
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if profile_enabled:
        profiler.add_duration(
            "qwen_prepare.process_vision_info",
            time.perf_counter() - start_time,
        )

    video_metadatas = None
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos = list(videos)
        video_metadatas = list(video_metadatas)

    start_time = time.perf_counter()
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        do_resize=True,
        max_pixels=5120 * 28 * 28,
        min_pixels=768 * 28 * 28,
        padding=padding,
        return_tensors="pt",
        **(video_kwargs or {}),
    )
    if profile_enabled:
        profiler.add_duration(
            "qwen_prepare.processor",
            time.perf_counter() - start_time,
        )

    start_time = time.perf_counter()
    inputs = inputs.to(model.device)
    if profile_enabled:
        profiler.add_duration(
            "qwen_prepare.inputs_to_device",
            time.perf_counter() - start_time,
        )
    return inputs


def _prepare_deepseek_inputs(runtime, row, build_prompt, img_root):
    model = runtime["model"]
    processor = runtime["processor"]
    load_pil_images = runtime["load_pil_images"]
    profiler = runtime.get("profiler")
    profile_enabled = profiler is not None and profiler.is_active()

    messages = build_prompt(row, img_root)
    conversation = _build_deepseek_conversation(messages)
    start_time = time.perf_counter()
    pil_images = load_pil_images(conversation)
    if profile_enabled:
        profiler.add_duration(
            "deepseek_prepare.load_pil_images",
            time.perf_counter() - start_time,
        )
    start_time = time.perf_counter()
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt="",
    )
    if profile_enabled:
        profiler.add_duration(
            "deepseek_prepare.processor",
            time.perf_counter() - start_time,
        )
    start_time = time.perf_counter()
    runtime_device = _get_deepseek_language_device(model)
    prepare_inputs = prepare_inputs.to(runtime_device, dtype=model.dtype)
    if profile_enabled:
        profiler.add_duration(
            "deepseek_prepare.inputs_to_device",
            time.perf_counter() - start_time,
        )
    _align_deepseek_direct_params(model, prepare_inputs.input_ids.device)
    _align_deepseek_resident_module_params(
        model,
        prepare_inputs.input_ids.device,
        getattr(model, "dtype", None),
    )
    return prepare_inputs


def _run_qwen_warmup(runtime, dataset, build_prompt, img_root, warmup_samples):
    model = runtime["model"]
    num_warmup = min(warmup_samples, len(dataset))
    for idx in range(num_warmup):
        row = dataset.iloc[idx]
        inputs = _prepare_qwen_inputs(
            runtime,
            row,
            build_prompt,
            img_root,
            padding=True,
        )

        seq_id_list = _begin_runtime_generation_trace(
            runtime,
            int(inputs.input_ids.shape[0]),
        )
        with torch.no_grad():
            try:
                _ = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                )
            finally:
                _end_runtime_generation_trace(
                    runtime,
                    seq_id_list,
                    persist=False,
                )


def _run_deepseek_warmup(runtime, dataset, build_prompt, img_root, warmup_samples):
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    num_warmup = min(warmup_samples, len(dataset))

    for idx in range(num_warmup):
        row = dataset.iloc[idx]
        prepare_inputs = _prepare_deepseek_inputs(
            runtime,
            row,
            build_prompt,
            img_root,
        )

        seq_id_list = _begin_runtime_generation_trace(
            runtime,
            int(prepare_inputs.input_ids.shape[0]),
        )
        with torch.no_grad():
            try:
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                _align_deepseek_direct_params(model, inputs_embeds.device)
                _align_deepseek_resident_module_params(
                    model,
                    inputs_embeds.device,
                    getattr(model, "dtype", None),
                )
                _ = model.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=prepare_inputs.input_ids,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=8,
                    do_sample=False,
                    use_cache=True,
                )
            finally:
                _end_runtime_generation_trace(
                    runtime,
                    seq_id_list,
                    persist=False,
                )


def run_moe_warmup(runtime, dataset, build_prompt, img_root, warmup_samples):
    if runtime["kind"] == "qwen3vlmoe":
        _run_qwen_warmup(runtime, dataset, build_prompt, img_root, warmup_samples)
        return
    _run_deepseek_warmup(runtime, dataset, build_prompt, img_root, warmup_samples)


def _run_qwen_trace_sample(runtime, row, build_prompt, img_root, max_new_tokens):
    model = runtime["model"]
    inputs = _prepare_qwen_inputs(
        runtime,
        row,
        build_prompt,
        img_root,
        padding=False,
    )
    seq_id_list = _begin_runtime_generation_trace(
        runtime,
        int(inputs.input_ids.shape[0]),
    )
    generation_succeeded = False
    with torch.no_grad():
        try:
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generation_succeeded = True
        finally:
            _end_runtime_generation_trace(
                runtime,
                seq_id_list,
                persist=generation_succeeded,
            )


def _run_deepseek_trace_sample(runtime, row, build_prompt, img_root, max_new_tokens):
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    prepare_inputs = _prepare_deepseek_inputs(
        runtime,
        row,
        build_prompt,
        img_root,
    )
    seq_id_list = _begin_runtime_generation_trace(
        runtime,
        int(prepare_inputs.input_ids.shape[0]),
    )
    generation_succeeded = False
    with torch.no_grad():
        try:
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
            _align_deepseek_direct_params(model, inputs_embeds.device)
            _align_deepseek_resident_module_params(
                model,
                inputs_embeds.device,
                getattr(model, "dtype", None),
            )
            _ = model.generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            generation_succeeded = True
        finally:
            _end_runtime_generation_trace(
                runtime,
                seq_id_list,
                persist=generation_succeeded,
            )


def run_moe_trace_sample(runtime, row, build_prompt, img_root, max_new_tokens):
    if runtime["kind"] == "qwen3vlmoe":
        _run_qwen_trace_sample(runtime, row, build_prompt, img_root, max_new_tokens)
        return
    _run_deepseek_trace_sample(runtime, row, build_prompt, img_root, max_new_tokens)


def _run_qwen_sample(runtime, row, build_prompt, img_root, args, sample_id):
    processor = runtime["processor"]
    profiler = runtime.get("profiler")
    profile_enabled = profiler is not None and profiler.is_active()
    start_time = time.perf_counter()
    inputs = _prepare_qwen_inputs(
        runtime,
        row,
        build_prompt,
        img_root,
        padding=False,
    )
    if profile_enabled:
        profiler.add_duration(
            "sample.prepare_inputs",
            time.perf_counter() - start_time,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    streamer = StopWatch(
        runtime["engine"],
        profiler=profiler,
        synchronize_cuda=not bool(
            getattr(args, "profile_no_streamer_sync", False)
        ),
    )

    seq_id_list = _begin_runtime_generation_trace(
        runtime,
        int(inputs.input_ids.shape[0]),
    )
    should_persist_trace = bool(runtime["runtime_meta"].get("trace_enabled"))
    generation_succeeded = False
    generate_start = time.perf_counter()
    try:
        with torch.no_grad():
            generated_ids = runtime["model"].generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                streamer=streamer,
            )
        generation_succeeded = True
    finally:
        _end_runtime_generation_trace(
            runtime,
            seq_id_list,
            persist=should_persist_trace and generation_succeeded,
        )
    generate_total = time.perf_counter() - generate_start

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    ttft, tpot, num_new_tokens = streamer.get_metrics()
    if profile_enabled:
        profiler.add_duration("sample.generate_total", generate_total)
        profiler.set_value("streamer.ttft_ms", float(ttft) * 1000.0)
        profiler.set_value("streamer.tpot_ms", float(tpot) * 1000.0)
        profiler.set_value("streamer.num_new_tokens", int(num_new_tokens))
        profiler.set_value(
            "streamer.synchronize_cuda",
            not bool(getattr(args, "profile_no_streamer_sync", False)),
        )
        profiler.set_value(
            "sample.prompt_length",
            int(inputs.input_ids.shape[1]),
        )

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    decode_start = time.perf_counter()
    output_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    if profile_enabled:
        profiler.add_duration(
            "sample.batch_decode",
            time.perf_counter() - decode_start,
        )

    return (
        {
            "id": int(sample_id) if isinstance(sample_id, int) else str(sample_id),
            "question": row.get("question", ""),
            "gt_answer": row.get("answer", ""),
            "prediction": output_text,
        },
        {
            "id": str(sample_id),
            "ttft": float(ttft),
            "tpot": float(tpot),
            "prompt_length": int(inputs.input_ids.shape[1]),
            "num_new_tokens": int(num_new_tokens),
        },
    )


def _run_deepseek_sample(runtime, row, build_prompt, img_root, args, sample_id):
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    profiler = runtime.get("profiler")
    profile_enabled = profiler is not None and profiler.is_active()
    start_time = time.perf_counter()
    prepare_inputs = _prepare_deepseek_inputs(
        runtime,
        row,
        build_prompt,
        img_root,
    )
    if profile_enabled:
        profiler.add_duration(
            "sample.prepare_inputs",
            time.perf_counter() - start_time,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    streamer = StopWatch(
        runtime["engine"],
        profiler=profiler,
        synchronize_cuda=not bool(
            getattr(args, "profile_no_streamer_sync", False)
        ),
    )

    seq_id_list = _begin_runtime_generation_trace(
        runtime,
        int(prepare_inputs.input_ids.shape[0]),
    )
    should_persist_trace = bool(runtime["runtime_meta"].get("trace_enabled"))
    generation_succeeded = False
    embed_start = time.perf_counter()
    try:
        with torch.no_grad():
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
            _align_deepseek_direct_params(model, inputs_embeds.device)
            _align_deepseek_resident_module_params(
                model,
                inputs_embeds.device,
                getattr(model, "dtype", None),
            )
        if profile_enabled:
            profiler.add_duration(
                "deepseek_prepare.prepare_inputs_embeds",
                time.perf_counter() - embed_start,
            )

        generate_start = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                streamer=streamer,
            )
        generation_succeeded = True
    finally:
        _end_runtime_generation_trace(
            runtime,
            seq_id_list,
            persist=should_persist_trace and generation_succeeded,
        )
    generate_total = time.perf_counter() - generate_start

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    ttft, tpot, num_new_tokens = streamer.get_metrics()
    if profile_enabled:
        profiler.add_duration("sample.generate_total", generate_total)
        profiler.set_value("streamer.ttft_ms", float(ttft) * 1000.0)
        profiler.set_value("streamer.tpot_ms", float(tpot) * 1000.0)
        profiler.set_value("streamer.num_new_tokens", int(num_new_tokens))
        profiler.set_value(
            "streamer.synchronize_cuda",
            not bool(getattr(args, "profile_no_streamer_sync", False)),
        )
        profiler.set_value("sample.prompt_length", int(prepare_inputs.input_ids.shape[1]))
    input_len = int(prepare_inputs.input_ids.shape[1])
    output_ids = generated_ids[0][input_len:]
    decode_start = time.perf_counter()
    output_text = tokenizer.decode(output_ids.tolist(), skip_special_tokens=True)
    if profile_enabled:
        profiler.add_duration(
            "sample.batch_decode",
            time.perf_counter() - decode_start,
        )

    return (
        {
            "id": int(sample_id) if isinstance(sample_id, int) else str(sample_id),
            "question": row.get("question", ""),
            "gt_answer": row.get("answer", ""),
            "prediction": output_text,
        },
        {
            "id": str(sample_id),
            "ttft": float(ttft),
            "tpot": float(tpot),
            "prompt_length": int(input_len),
            "num_new_tokens": int(num_new_tokens),
        },
    )


def run_moe_sample(runtime, row, build_prompt, img_root, args, sample_id):
    profiler = runtime.get("profiler")
    if profiler is not None:
        is_profile_active = profiler.begin_sample(
            sample_id,
            metadata={
                "model": runtime["kind"],
                "benchmark": runtime.get("_profile_benchmark"),
                "max_new_tokens": int(args.max_new_tokens),
            },
        )
        if is_profile_active:
            _clear_expert_dispatcher_perf_stats(runtime)

    try:
        if runtime["kind"] == "qwen3vlmoe":
            result_item, metric_item = _run_qwen_sample(
                runtime,
                row,
                build_prompt,
                img_root,
                args,
                sample_id,
            )
        else:
            result_item, metric_item = _run_deepseek_sample(
                runtime,
                row,
                build_prompt,
                img_root,
                args,
                sample_id,
            )
    except Exception as exc:
        if profiler is not None:
            _record_expert_dispatcher_perf_stats(runtime, profiler)
            profiler.finish_sample(status="error", error=exc)
        raise

    if profiler is not None:
        _record_expert_dispatcher_perf_stats(runtime, profiler)
        profile_result = profiler.finish_sample(status="ok")
        if profile_result is not None:
            metric_item["timing_profile"] = profile_result

    return result_item, metric_item
