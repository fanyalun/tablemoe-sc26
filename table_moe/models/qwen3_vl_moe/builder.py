import gc
import inspect
import json
import os
import random
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple

import psutil
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration

from ...cache_engine.config import set_active_cache_config
from ...cache_engine.manager import HybridStorageManager
from ...offload.linear_cache import LinearCache

from .cached_layers import QwenMoeWrapperCached
from .custom_layers import (
    QwenMoeWrapperBaseline,
    QwenMoeWrapperSkipBaseline,
    QwenMoeWrapperSkipOffload,
)
from .expert_wrapper import QwenExpertWrapper
from .hybrid_attention import HybridQwen3Attention
from .offload_config import get_hybrid_cache_config, get_offload_config

random.seed(42)


@contextmanager
def with_default_dtype(dtype: torch.dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def print_memory(step: str):
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    print(
        f"[Memory] {step}: RSS={mem.rss / 1024**3:.2f}GB, "
        f"VMS={mem.vms / 1024**3:.2f}GB, Shared={getattr(mem, 'shared', 0) / 1024**3:.2f}GB"
    )


@contextmanager
def qwen_moe_skeleton_context():
    """
    Preserve the official MoE block shape and gate module, but omit routed expert
    parameter allocation so routed weights only live in LinearCache.
    """
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock

    original_init = Qwen3VLMoeTextSparseMoeBlock.__init__

    def skeleton_init(self, config):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = None

    Qwen3VLMoeTextSparseMoeBlock.__init__ = skeleton_init
    try:
        yield
    finally:
        Qwen3VLMoeTextSparseMoeBlock.__init__ = original_init


def _normalize_attn_implementation(attn_implementation: str) -> str:
    supported = {"eager", "sdpa", "flash_attention_2"}
    if attn_implementation not in supported:
        raise ValueError(
            f"Unsupported attention implementation: {attn_implementation}, "
            f"expected one of {sorted(supported)}"
        )
    return attn_implementation


def _apply_qwen_dtype_to_config(full_config: Qwen3VLMoeConfig, model_dtype: torch.dtype):
    full_config.torch_dtype = model_dtype
    full_config.text_config.dtype = model_dtype
    if hasattr(full_config, "vision_config") and hasattr(full_config.vision_config, "dtype"):
        full_config.vision_config.dtype = model_dtype


def _resolve_runtime_device(device_map, fallback_device: torch.device) -> torch.device:
    if device_map is None:
        return fallback_device
    if isinstance(device_map, torch.device):
        return device_map
    if isinstance(device_map, str):
        if device_map == "auto":
            if torch.cuda.is_available():
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            return torch.device("cpu")
        return torch.device(device_map)
    raise TypeError(f"Unsupported device_map for Qwen offload builder: {device_map!r}")


def _is_routed_expert_key(key: str) -> bool:
    return ".mlp.experts.gate_up_proj" in key or ".mlp.experts.down_proj" in key


def _load_non_routed_weight_map(model_path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return {}, {}

    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    filtered_map: Dict[str, str] = {}
    fname_to_keys: Dict[str, List[str]] = defaultdict(list)
    for key, fname in weight_map.items():
        if _is_routed_expert_key(key):
            continue
        filtered_map[key] = fname
        fname_to_keys[fname].append(key)
    return filtered_map, fname_to_keys


def _load_trunk_weights_qwen_to_model(model, model_path: str, model_device: torch.device):
    """
    Materialize non-routed tensors directly onto the target device. This matches
    HF-style load semantics better than building on CPU and calling model.to(...).
    """
    try:
        from accelerate.utils import set_module_tensor_to_device
    except ImportError as e:
        raise RuntimeError(
            "Loading Qwen offload trunk requires accelerate.utils.set_module_tensor_to_device"
        ) from e

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    has_dtype_arg = "dtype" in inspect.signature(set_module_tensor_to_device).parameters

    def load_tensor_to_model(key: str, tensor: torch.Tensor) -> bool:
        target = params.get(key)
        if target is None:
            target = buffers.get(key)
        if target is None:
            return False

        if torch.is_floating_point(target):
            value = tensor.to(dtype=target.dtype)
            target_dtype = target.dtype
        else:
            value = tensor
            target_dtype = None

        kwargs = {"value": value}
        if has_dtype_arg and target_dtype is not None:
            kwargs["dtype"] = target_dtype
        set_module_tensor_to_device(model, key, model_device, **kwargs)
        return True

    filtered_map, fname_to_keys = _load_non_routed_weight_map(model_path)

    missing_in_model: List[str] = []
    loaded_count = 0

    if not filtered_map:
        safepath = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(safepath):
            raise FileNotFoundError(
                f"Cannot find model.safetensors.index.json or model.safetensors under {model_path}"
            )

        full_sd = load_file(safepath, device="cpu")
        with torch.no_grad():
            for key, tensor in full_sd.items():
                if _is_routed_expert_key(key):
                    continue
                if not load_tensor_to_model(key, tensor):
                    missing_in_model.append(key)
                    continue
                loaded_count += 1
        del full_sd
        gc.collect()
    else:
        shard_count = len(fname_to_keys)
        for shard_idx, (fname, keys) in enumerate(sorted(fname_to_keys.items()), start=1):
            shard_path = os.path.join(model_path, fname)
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                with torch.no_grad():
                    for key in keys:
                        if not load_tensor_to_model(key, shard.get_tensor(key)):
                            missing_in_model.append(key)
                            continue
                        loaded_count += 1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[offload] Loaded trunk shard {shard_idx}/{shard_count}: {fname} ({len(keys)} tensors)")

    if missing_in_model:
        raise KeyError(f"Non-routed checkpoint keys missing in model: {missing_in_model[:10]}")
    return loaded_count


def _materialize_nonpersistent_buffers(model, model_device: torch.device):
    """
    Meta-init keeps certain constructor-created buffers outside the checkpoint load path.
    RoPE `inv_freq` is the critical example for Qwen. Align them to the target device
    after trunk loading so the runtime matches HF's `from_pretrained(device_map=...)`.
    """
    try:
        from accelerate.utils import set_module_tensor_to_device
    except ImportError as e:
        raise RuntimeError(
            "Materializing Qwen non-persistent buffers requires accelerate.utils.set_module_tensor_to_device"
        ) from e

    moved = []
    for name, buffer in model.named_buffers():
        if buffer.device == model_device:
            continue

        kwargs = {}
        if not getattr(buffer, "is_meta", False):
            kwargs["value"] = buffer
        set_module_tensor_to_device(model, name, model_device, **kwargs)
        moved.append(name)

    # Keep alias attributes such as `original_inv_freq` consistent after buffer replacement.
    for module in model.modules():
        if hasattr(module, "inv_freq") and hasattr(module, "original_inv_freq"):
            module.original_inv_freq = module.inv_freq

    if moved:
        print(f"[offload] Materialized {len(moved)} non-persistent buffers to {model_device}")
    return moved


def _assert_no_meta_tensors(model):
    meta_tensors = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if getattr(tensor, "is_meta", False):
            meta_tensors.append(name)
            if len(meta_tensors) >= 10:
                break
    if meta_tensors:
        raise RuntimeError(f"Model still contains meta tensors after loading: {meta_tensors}")


def _build_qwen_skeleton_model(
    full_config: Qwen3VLMoeConfig, device: torch.device, model_dtype: torch.dtype
):
    try:
        from accelerate import init_empty_weights
        from transformers.modeling_utils import no_init_weights
    except ImportError as e:
        raise RuntimeError(
            "Building Qwen offload skeleton requires accelerate.init_empty_weights "
            "and transformers.modeling_utils.no_init_weights"
        ) from e

    print(
        f"[offload] Creating Qwen skeleton with meta init "
        f"(target device: {device}, default_dtype={model_dtype})..."
    )
    with qwen_moe_skeleton_context(), no_init_weights(), init_empty_weights(), with_default_dtype(model_dtype):
        model = Qwen3VLMoeForConditionalGeneration(full_config)
    return model


def _resolve_cache_sizes(raw_cache_config, num_layers: int) -> List[int]:
    if isinstance(raw_cache_config, int):
        cache_sizes = [raw_cache_config] * num_layers
        print(f"[offload] Cache Config: Uniform size {raw_cache_config} for all {num_layers} layers.")
        return cache_sizes

    if isinstance(raw_cache_config, list):
        if len(raw_cache_config) != num_layers:
            raise ValueError(
                f"cache_size_per_layer list length ({len(raw_cache_config)}) "
                f"must match num_layers ({num_layers})"
            )
        print(f"[offload] Cache Config: Custom list per layer (min={min(raw_cache_config)}, max={max(raw_cache_config)}).")
        return raw_cache_config

    raise TypeError("cache_size_per_layer must be int or List[int]")


def _build_expert_cache(text_cfg_full, device: torch.device, model_dtype: torch.dtype, total_gpu_experts: int, total_offload_experts: int, buffer_size: int):
    hidden_size = text_cfg_full.hidden_size
    inter_size = text_cfg_full.moe_intermediate_size

    def make_module():
        class MockExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.utils.skip_init(
                    nn.Linear, hidden_size, inter_size, bias=False, device=device, dtype=model_dtype
                )
                self.up_proj = torch.nn.utils.skip_init(
                    nn.Linear, hidden_size, inter_size, bias=False, device=device, dtype=model_dtype
                )
                self.down_proj = torch.nn.utils.skip_init(
                    nn.Linear, inter_size, hidden_size, bias=False, device=device, dtype=model_dtype
                )
                self.act_fn = nn.SiLU()

        return QwenExpertWrapper(MockExpert(), device)

    return LinearCache(
        make_module=make_module,
        main_size=total_gpu_experts,
        offload_size=total_offload_experts,
        buffer_size=buffer_size,
    )


def _load_experts_into_cache(
    model_id: str,
    text_cfg_full,
    cache_sizes: List[int],
    expert_cache: LinearCache,
    model_dtype: torch.dtype,
):
    print("[offload] Loading routed expert weights into LinearCache...")
    index_path = os.path.join(model_id, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    num_layers = text_cfg_full.num_hidden_layers
    num_experts = text_cfg_full.num_experts
    intermediate_size = text_cfg_full.moe_intermediate_size

    file_to_layers: Dict[str, List[int]] = defaultdict(list)
    for layer_idx in range(num_layers):
        gate_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        if gate_key in weight_map:
            file_to_layers[weight_map[gate_key]].append(layer_idx)

    total_files = len(file_to_layers)
    for files_processed, (fname, layers_in_this_file) in enumerate(sorted(file_to_layers.items()), start=1):
        fpath = os.path.join(model_id, fname)
        print(f"[offload] Processing expert shard {files_processed}/{total_files}: {fname}")

        with safe_open(fpath, framework="pt", device="cpu") as shard:
            for layer_idx in layers_in_this_file:
                gate_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
                down_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"

                current_layer_k = cache_sizes[layer_idx]
                gpu_expert_indices = set(random.sample(range(num_experts), min(current_layer_k, num_experts)))

                all_gate_up = shard.get_tensor(gate_key).to(dtype=model_dtype)
                all_down = shard.get_tensor(down_key).to(dtype=model_dtype)

                for expert_idx in range(num_experts):
                    gate_up_i = all_gate_up[expert_idx]
                    down_w = all_down[expert_idx]

                    gate_w = gate_up_i[:, :intermediate_size]
                    up_w = gate_up_i[:, intermediate_size:]

                    w1 = gate_w.T.contiguous()
                    w3 = up_w.T.contiguous()
                    w2 = down_w.T.contiguous()

                    expert_cache.add_linear_storage(
                        uid=(layer_idx, expert_idx),
                        storage=[w1.untyped_storage(), w2.untyped_storage(), w3.untyped_storage()],
                        eviction_group=layer_idx,
                        offload=(expert_idx not in gpu_expert_indices),
                    )

                del all_gate_up, all_down
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_offload_model_baseline(
    model_id: str,
    attn_implementation: str = "flash_attention_2",
    device_map="auto",
):
    config = get_offload_config()
    device = _resolve_runtime_device(device_map, config["device"])
    raw_cache_config = config["cache_size_per_layer"]
    buffer_size = config["buffer_size"]
    model_dtype = config["model_dtype"]
    attn_implementation = _normalize_attn_implementation(attn_implementation)

    print("\n" + "=" * 80)
    print(f"[offload] Build Qwen3-VL-MoE skeleton (baseline) from: {model_id}")
    print("=" * 80)
    print(f"[offload] device_map={device_map!r}, target_device={device}")
    print_memory("Start")

    full_config: Qwen3VLMoeConfig = Qwen3VLMoeConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    full_config._attn_implementation = attn_implementation
    _apply_qwen_dtype_to_config(full_config, model_dtype)

    text_cfg_full = full_config.text_config
    num_layers = text_cfg_full.num_hidden_layers
    num_experts = text_cfg_full.num_experts
    cache_sizes = _resolve_cache_sizes(raw_cache_config, num_layers)

    total_gpu_experts = sum(cache_sizes)
    total_offload_experts = num_layers * num_experts

    print(f"[offload] Global attention implementation: {full_config._attn_implementation}")
    model = _build_qwen_skeleton_model(full_config, device, model_dtype)

    expert_cache = _build_expert_cache(
        text_cfg_full=text_cfg_full,
        device=device,
        model_dtype=model_dtype,
        total_gpu_experts=total_gpu_experts,
        total_offload_experts=total_offload_experts,
        buffer_size=buffer_size,
    )

    print_memory("Before Wrapper Replace")
    text_model = model.model.language_model
    layers_list = list(text_model.layers)
    print("[offload] Replacing text MoE layers with QwenMoeWrapperBaseline...")
    for layer_idx, layer in enumerate(text_model.layers):
        gate = getattr(layer.mlp, "gate", None)
        if gate is None:
            raise ValueError(f"Layer {layer_idx} is not a sparse MoE layer; cannot build offload wrapper.")
        layer.mlp = QwenMoeWrapperBaseline(
            text_config=text_cfg_full,
            layer_id=layer_idx,
            gate=gate,
            expert_cache=expert_cache,
            layers=layers_list,
            global_config=full_config,
        )
        layer.mlp.top_k = text_cfg_full.num_experts_per_tok

    print_memory("Before Trunk Load")
    print(f"[offload] Loading non-routed trunk weights directly to target device {device}...")
    loaded_count = _load_trunk_weights_qwen_to_model(model, model_id, device)
    _materialize_nonpersistent_buffers(model, device)
    print(f"[offload] Loaded {loaded_count} non-routed tensors")
    _assert_no_meta_tensors(model)
    print_memory("After Trunk Load")

    _load_experts_into_cache(
        model_id=model_id,
        text_cfg_full=text_cfg_full,
        cache_sizes=cache_sizes,
        expert_cache=expert_cache,
        model_dtype=model_dtype,
    )

    print_memory("After Expert Load")
    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, text_cfg_full, expert_cache


def _build_offload_model_cached(
    model_id: str,
    attn_implementation: str = "flash_attention_2",
    device_map="auto",
    *,
    mode_label: str,
    prefill_keep_strategy: str,
    decode_search_strategy: str,
    replace_attention: bool,
):
    config = get_offload_config()
    device = _resolve_runtime_device(device_map, config["device"])
    raw_cache_config = config["cache_size_per_layer"]
    buffer_size = config["buffer_size"]
    model_dtype = config["model_dtype"]
    attn_implementation = _normalize_attn_implementation(attn_implementation)

    print("\n" + "=" * 80)
    print(f"[offload] Build Qwen3-VL-MoE skeleton ({mode_label}) from: {model_id}")
    print("=" * 80)
    print(f"[offload] device_map={device_map!r}, target_device={device}")
    print_memory("Start")

    full_config: Qwen3VLMoeConfig = Qwen3VLMoeConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    full_config._attn_implementation = attn_implementation
    _apply_qwen_dtype_to_config(full_config, model_dtype)

    text_cfg_full = full_config.text_config
    num_layers = text_cfg_full.num_hidden_layers
    num_experts = text_cfg_full.num_experts
    cache_sizes = _resolve_cache_sizes(raw_cache_config, num_layers)
    cache_config = get_hybrid_cache_config(
        num_layers=num_layers,
        top_k=text_cfg_full.num_experts_per_tok,
    )
    if decode_search_strategy == "offline":
        cache_config["ONLINE_MAX_LAYER_IDX"] = -1
    set_active_cache_config(cache_config)

    total_gpu_experts = sum(cache_sizes)
    total_offload_experts = num_layers * num_experts

    print(
        f"[offload] Expert cache allocation:\n"
        f"  - Total Layers: {num_layers}\n"
        f"  - Total GPU Slots: {total_gpu_experts}\n"
        f"  - Total Offload Experts: {total_offload_experts}"
    )
    print(f"[offload] Global attention implementation: {full_config._attn_implementation}")

    model = _build_qwen_skeleton_model(full_config, device, model_dtype)
    expert_cache = _build_expert_cache(
        text_cfg_full=text_cfg_full,
        device=device,
        model_dtype=model_dtype,
        total_gpu_experts=total_gpu_experts,
        total_offload_experts=total_offload_experts,
        buffer_size=buffer_size,
    )

    print_memory("Before Wrapper Replace")
    print(f"[{mode_label}] Initializing Hybrid Storage Managers...")
    text_model = model.model.language_model
    layers_list = list(text_model.layers)
    for layer_idx, layer in enumerate(text_model.layers):
        gate = getattr(layer.mlp, "gate", None)
        if gate is None:
            raise ValueError(f"Layer {layer_idx} is not a sparse MoE layer; cannot build offload wrapper.")

        layer_cache_manager = HybridStorageManager(
            layer_idx=layer_idx,
            num_experts=text_cfg_full.num_experts,
            hidden_dim=text_cfg_full.hidden_size,
            device=device,
        )

        layer.mlp = QwenMoeWrapperCached(
            text_config=text_cfg_full,
            layer_id=layer_idx,
            gate=gate,
            expert_cache=expert_cache,
            cache_manager=layer_cache_manager,
            prefill_keep_strategy=prefill_keep_strategy,
            decode_search_strategy=decode_search_strategy,
            layers=layers_list,
            global_config=full_config,
        )
        layer.mlp.top_k = text_cfg_full.num_experts_per_tok

    print_memory("Before Trunk Load")
    print(f"[offload] Loading non-routed trunk weights directly to target device {device}...")
    loaded_count = _load_trunk_weights_qwen_to_model(model, model_id, device)
    _materialize_nonpersistent_buffers(model, device)
    print(f"[offload] Loaded {loaded_count} non-routed tensors")
    _assert_no_meta_tensors(model)
    print_memory("After Trunk Load")

    _load_experts_into_cache(
        model_id=model_id,
        text_cfg_full=text_cfg_full,
        cache_sizes=cache_sizes,
        expert_cache=expert_cache,
        model_dtype=model_dtype,
    )

    if replace_attention:
        print("[hybrid] Replacing attention modules for importance sampling...")
        for layer in text_model.layers:
            layer.self_attn = HybridQwen3Attention(layer.self_attn)

    print_memory("After Expert Load")
    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, text_cfg_full, expert_cache


def build_offload_model_skip_offload(
    model_id: str,
    attn_implementation: str = "flash_attention_2",
    device_map="auto",
):
    config = get_offload_config()
    device = _resolve_runtime_device(device_map, config["device"])
    raw_cache_config = config["cache_size_per_layer"]
    buffer_size = config["buffer_size"]
    model_dtype = config["model_dtype"]
    skip_keep_k = config["skip_keep_k"]
    decode_skip_keep_k = config["decode_skip_keep_k"]
    attn_implementation = _normalize_attn_implementation(attn_implementation)

    print("\n" + "=" * 80)
    print(f"[skip-offload] Build Qwen3-VL-MoE skeleton (skip offload) from: {model_id}")
    print("=" * 80)
    print(f"[skip-offload] device_map={device_map!r}, target_device={device}")
    print_memory("Start")

    full_config: Qwen3VLMoeConfig = Qwen3VLMoeConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    full_config._attn_implementation = attn_implementation
    _apply_qwen_dtype_to_config(full_config, model_dtype)

    text_cfg_full = full_config.text_config
    num_layers = text_cfg_full.num_hidden_layers
    num_experts = text_cfg_full.num_experts
    cache_sizes = _resolve_cache_sizes(raw_cache_config, num_layers)

    total_gpu_experts = sum(cache_sizes)
    total_offload_experts = num_layers * num_experts

    print(f"[skip-offload] Global attention implementation: {full_config._attn_implementation}")
    model = _build_qwen_skeleton_model(full_config, device, model_dtype)

    expert_cache = _build_expert_cache(
        text_cfg_full=text_cfg_full,
        device=device,
        model_dtype=model_dtype,
        total_gpu_experts=total_gpu_experts,
        total_offload_experts=total_offload_experts,
        buffer_size=buffer_size,
    )

    print_memory("Before Wrapper Replace")
    text_model = model.model.language_model
    layers_list = list(text_model.layers)
    print("[skip-offload] Replacing text MoE layers with QwenMoeWrapperSkipOffload...")
    for layer_idx, layer in enumerate(text_model.layers):
        gate = getattr(layer.mlp, "gate", None)
        if gate is None:
            raise ValueError(f"Layer {layer_idx} is not a sparse MoE layer; cannot build offload wrapper.")
        layer.mlp = QwenMoeWrapperSkipOffload(
            text_config=text_cfg_full,
            layer_id=layer_idx,
            gate=gate,
            expert_cache=expert_cache,
            skip_keep_k=skip_keep_k,
            decode_skip_keep_k=decode_skip_keep_k,
            layers=layers_list,
            global_config=full_config,
        )
        layer.mlp.top_k = text_cfg_full.num_experts_per_tok

    print_memory("Before Trunk Load")
    print(f"[skip-offload] Loading non-routed trunk weights directly to target device {device}...")
    loaded_count = _load_trunk_weights_qwen_to_model(model, model_id, device)
    _materialize_nonpersistent_buffers(model, device)
    print(f"[skip-offload] Loaded {loaded_count} non-routed tensors")
    _assert_no_meta_tensors(model)
    print_memory("After Trunk Load")

    _load_experts_into_cache(
        model_id=model_id,
        text_cfg_full=text_cfg_full,
        cache_sizes=cache_sizes,
        expert_cache=expert_cache,
        model_dtype=model_dtype,
    )

    print_memory("After Expert Load")
    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, text_cfg_full, expert_cache


def build_offload_model_hybrid(
    model_id: str,
    attn_implementation: str = "flash_attention_2",
    device_map="auto",
):
    return _build_offload_model_cached(
        model_id=model_id,
        attn_implementation=attn_implementation,
        device_map=device_map,
        mode_label="hybrid",
        prefill_keep_strategy="importance",
        decode_search_strategy="hybrid",
        replace_attention=True,
    )


def build_offload_model_offline(
    model_id: str,
    attn_implementation: str = "flash_attention_2",
    device_map="auto",
):
    return _build_offload_model_cached(
        model_id=model_id,
        attn_implementation=attn_implementation,
        device_map=device_map,
        mode_label="offline",
        prefill_keep_strategy="fixed_keep_k",
        decode_search_strategy="offline",
        replace_attention=False,
    )


def build_offload_model_online(
    model_id: str,
    attn_implementation: str = "flash_attention_2",
    device_map="auto",
):
    return _build_offload_model_cached(
        model_id=model_id,
        attn_implementation=attn_implementation,
        device_map=device_map,
        mode_label="online",
        prefill_keep_strategy="fixed_keep_k",
        decode_search_strategy="hybrid",
        replace_attention=False,
    )


def build_full_model_skip(
    model_id: str,
    attn_implementation: str = "flash_attention_2",
    device_map="auto",
):
    config = get_offload_config()
    device = _resolve_runtime_device(device_map, config["device"])
    model_dtype = config["model_dtype"]
    skip_keep_k = config["skip_keep_k"]
    decode_skip_keep_k = config["decode_skip_keep_k"]
    attn_implementation = _normalize_attn_implementation(attn_implementation)

    print("\n" + "=" * 80)
    print(f"[skip] Build Qwen3-VL-MoE full model (skip baseline) from: {model_id}")
    print("=" * 80)
    print(f"[skip] device_map={device_map!r}, target_device={device}")
    print_memory("Start")

    full_config: Qwen3VLMoeConfig = Qwen3VLMoeConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    full_config._attn_implementation = attn_implementation
    _apply_qwen_dtype_to_config(full_config, model_dtype)

    text_cfg_full = full_config.text_config
    set_active_cache_config(
        get_hybrid_cache_config(
            num_layers=text_cfg_full.num_hidden_layers,
            top_k=text_cfg_full.num_experts_per_tok,
        )
    )

    print(f"[skip] Global attention implementation: {full_config._attn_implementation}")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_id,
        config=full_config,
        torch_dtype=model_dtype,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )

    text_model = model.model.language_model
    layers_list = list(text_model.layers)
    print("[skip] Replacing text MoE layers with QwenMoeWrapperSkipBaseline...")
    for layer_idx, layer in enumerate(text_model.layers):
        original_mlp = getattr(layer, "mlp", None)
        gate = getattr(original_mlp, "gate", None)
        experts = getattr(original_mlp, "experts", None)
        if gate is None or experts is None:
            raise ValueError(f"Layer {layer_idx} is not a sparse MoE layer; cannot build skip wrapper.")
        layer.mlp = QwenMoeWrapperSkipBaseline(
            text_config=text_cfg_full,
            layer_id=layer_idx,
            gate=gate,
            experts=experts,
            skip_keep_k=skip_keep_k,
            decode_skip_keep_k=decode_skip_keep_k,
            layers=layers_list,
            global_config=full_config,
        )
        layer.mlp.top_k = text_cfg_full.num_experts_per_tok

    print_memory("After Wrapper Replace")
    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, text_cfg_full, None
