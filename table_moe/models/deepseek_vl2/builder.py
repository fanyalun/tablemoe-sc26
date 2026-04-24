import os
import gc
import json
import inspect
import random
import psutil
import sys
from collections import defaultdict
from typing import Dict, List, Tuple
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file
from safetensors import safe_open

from ...cache_engine.config import set_active_cache_config
from ...cache_engine.manager import HybridStorageManager
from ...offload.linear_cache import LinearCache
from .deepseek_expert_wrapper import DeepSeekExpertWrapper
from .deepseek_layers import (
    DeepSeekMoeWrapperBaseline,
    DeepSeekMoeWrapperCached,
    DeepSeekMoeWrapperSkipBaseline,
    DeepSeekMoeWrapperSkipOffload,
)
from .offload_config import get_deepseek_cache_config, get_deepseek_offload_config
from .deepseek_attention import HybridDeepSeekAttention

random.seed(42)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEEPSEEK_VL2_REPO = _REPO_ROOT / "third_party" / "DeepSeek-VL2"


@contextmanager
def with_default_dtype(dtype):
    """上下文管理器：临时设置 torch 默认 dtype"""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def print_memory(step):
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    print(f"[Memory] {step}: RSS={mem.rss/1024**3:.2f}GB, VMS={mem.vms/1024**3:.2f}GB")


@contextmanager
def deepseek_moe_skeleton_context():
    """
    临时将 DeepseekV2MoE 替换为不实例化 routed experts 的 skeleton。
    保留 gate/shared_experts 的真实形状，后续再用 offload wrapper 替换整层 mlp。
    """
    from deepseek_vl2.models.modeling_deepseek import DeepseekV2MoE, DeepseekV2MLP, MoEGate

    original_init = DeepseekV2MoE.__init__

    def skeleton_init(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.ep_size = 1
        self.experts_per_rank = config.n_routed_experts
        self.ep_rank = 0
        self.experts = nn.ModuleList()
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                config=config,
                intermediate_size=intermediate_size,
            )

    DeepseekV2MoE.__init__ = skeleton_init
    try:
        yield
    finally:
        DeepseekV2MoE.__init__ = original_init


def _load_non_routed_weight_map(model_path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return {}, {}

    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    filtered_map = {}
    fname_to_keys: Dict[str, List[str]] = defaultdict(list)
    for key, fname in weight_map.items():
        if _is_routed_expert_key(key):
            continue
        filtered_map[key] = fname
        fname_to_keys[fname].append(key)
    return filtered_map, fname_to_keys


def _load_trunk_weights_deepseek_to_model(model, model_path: str, model_device: torch.device):
    """
    将非 routed expert 权重按 shard 直接 materialize 到目标 device。
    支持 meta-init skeleton；不经过完整 CPU trunk_state 聚合。
    """
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    try:
        from accelerate.utils import set_module_tensor_to_device
    except ImportError as e:
        raise RuntimeError(
            "加载 DeepSeek offload trunk 需要 accelerate.utils.set_module_tensor_to_device"
        ) from e

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

    if not filtered_map:
        safepath = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(safepath):
            raise FileNotFoundError(f"Cannot find model.safetensors.index.json or model.safetensors under {model_path}")
        full_sd = load_file(safepath, device="cpu")
        missing_in_model = []
        loaded_count = 0
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
        if missing_in_model:
            raise KeyError(f"Non-routed checkpoint keys missing in model: {missing_in_model[:10]}")
        return loaded_count, 0

    missing_in_model = []
    loaded_count = 0
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
        print(f"  Loaded trunk shard {shard_idx}/{shard_count}: {fname} ({len(keys)} tensors)")

    if missing_in_model:
        raise KeyError(f"Non-routed checkpoint keys missing in model: {missing_in_model[:10]}")
    return loaded_count, len(filtered_map) - loaded_count


def _is_routed_expert_key(key: str) -> bool:
    """判断是否是 routed expert 权重键"""
    # 格式: language.model.layers.{i}.mlp.experts.{j}.gate_proj/up_proj/down_proj
    if ".mlp.experts." not in key:
        return False
    parts = key.split(".mlp.experts.")
    if len(parts) < 2:
        return False
    # 取 experts. 后面的第一个部分
    after_experts = parts[1].split(".")[0]
    return after_experts.isdigit()


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
    raise TypeError(f"Unsupported device_map for DeepSeek offload builder: {device_map!r}")


def _build_deepseek_skeleton_model(DeepseekVLV2ForCausalLM, full_config, device: torch.device, model_dtype: torch.dtype):
    """
    用真实 config 构建 HF 风格 skeleton：
    - Layer 0 保持 dense MLP
    - Layer 1-29 仅保留 gate/shared_experts，不实例化 routed experts
    - 参数走 meta-init，非持久 buffer 保持官方 from_pretrained(device_map=...) 的构造语义
    """
    try:
        from accelerate import init_empty_weights
        from transformers.modeling_utils import no_init_weights
    except ImportError as e:
        raise RuntimeError(
            "构建 DeepSeek offload skeleton 需要 accelerate.init_empty_weights 和 transformers.modeling_utils.no_init_weights"
        ) from e

    print(f"[DeepSeek] Creating skeleton model with meta init (target device: {device}, default_dtype={model_dtype})...")
    with deepseek_moe_skeleton_context(), no_init_weights(), init_empty_weights(), with_default_dtype(model_dtype):
        model = DeepseekVLV2ForCausalLM(full_config)
    return model


def _normalize_attn_implementation(attn_implementation: str) -> str:
    supported = {"eager", "flash_attention_2"}
    if attn_implementation not in supported:
        raise ValueError(f"Unsupported attention implementation: {attn_implementation}, expected one of {sorted(supported)}")
    return attn_implementation


def build_offload_model_deepseek_baseline(
    model_id: str,
    attn_implementation: str = "flash_attention_2",
    device_map="auto",
):
    """构建 DeepSeek-VL2 Baseline Offload 模型"""
    config = get_deepseek_offload_config()
    device = _resolve_runtime_device(device_map, config['device'])
    raw_cache_config = config['cache_size_per_layer']
    buffer_size = config['buffer_size']
    model_dtype = config['model_dtype']
    first_k_dense = config['first_k_dense_replace']
    attn_implementation = _normalize_attn_implementation(attn_implementation)

    print("\n" + "=" * 80)
    print(f"[DeepSeek Offload] Building Baseline Model from: {model_id}")
    print("=" * 80)
    print(f"[DeepSeek] device_map={device_map!r}, target_device={device}")
    print_memory("Start")

    # 1. 从 DeepSeek-VL2 仓库导入自定义类
    deepseek_vl2_repo = str(_DEEPSEEK_VL2_REPO)
    if deepseek_vl2_repo not in sys.path:
        sys.path.insert(0, deepseek_vl2_repo)

    # 2. 导入 DeepSeek-VL2 的配置和模型类
    try:
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config, DeepseekVLV2ForCausalLM
        from deepseek_vl2.models.modeling_deepseek import DeepseekV2ForCausalLM
        print("[DeepSeek] Loaded custom model classes from DeepSeek-VL2 repo")
    except ImportError as e:
        print(f"[DeepSeek] Failed to load from DeepSeek-VL2 repo: {e}")
        raise RuntimeError(f"请确保 DeepSeek-VL2 仓库已存在于 {_DEEPSEEK_VL2_REPO}")

    # 3. 加载配置（指定 torch_dtype）
    full_config = DeepseekVLV2Config.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
        trust_remote_code=True
    )
    full_config._attn_implementation = attn_implementation

    print(f"[DeepSeek] Attention implementation: {full_config._attn_implementation}")

    lang_cfg = full_config.language_config
    num_layers = lang_cfg.num_hidden_layers
    num_routed_experts = lang_cfg.n_routed_experts  # 从配置读取，不是硬编码

    print(f"[DeepSeek] Model config: {num_layers} layers, {num_routed_experts} routed experts/layer")

    # 4. 用真实 config 构建 skeleton：
    # trunk/gate/shared_experts 直接在 GPU 上初始化，routed experts 不进入主模型参数树。
    model = _build_deepseek_skeleton_model(DeepseekVLV2ForCausalLM, full_config, device, model_dtype)
    print("[DeepSeek] Created skeleton model (routed experts omitted from main model)")

    # 5. 处理 cache_sizes
    if isinstance(raw_cache_config, int):
        cache_sizes = [raw_cache_config] * num_layers
    elif isinstance(raw_cache_config, list):
        if len(raw_cache_config) != num_layers:
            raise ValueError(f"cache_size_per_layer length mismatch: {len(raw_cache_config)} vs {num_layers}")
        cache_sizes = raw_cache_config
    else:
        raise TypeError("cache_size_per_layer must be int or List[int]")

    print(f"[DeepSeek] Cache sizes: {cache_sizes[:5]}... (first 5 layers)")

    # 6. 计算总 GPU 槽位（跳过 dense 层）
    total_gpu_experts = sum(cache_sizes[first_k_dense:])
    total_offload_experts = (num_layers - first_k_dense) * num_routed_experts

    print(f"[DeepSeek] Expert cache allocation:")
    print(f"  - Total Layers: {num_layers} (Layer 0-{first_k_dense-1}: dense, Layer {first_k_dense}-{num_layers-1}: MoE)")
    print(f"  - Total GPU Slots: {total_gpu_experts}")
    print(f"  - Total Offload Experts: {total_offload_experts}")

    # 6. 先加载 trunk 权重（让 gate 和 shared_experts 有正确的大小）
    print_memory("Before Trunk Load")
    print(f"[DeepSeek] Loading trunk weights directly to target device {device}...")
    loaded_count, missing_count = _load_trunk_weights_deepseek_to_model(model, model_id, device)
    print(f"[DeepSeek] ✅ Loaded {loaded_count} non-routed tensors directly to target device {device} (missing={missing_count})")
    print_memory("After Trunk Load")
    torch.cuda.empty_cache()

    # 7. 构造 LinearCache（routed experts 只进入 cache，不进入主模型）
    def make_module():
        hidden_size = lang_cfg.hidden_size
        inter_size = lang_cfg.moe_intermediate_size

        class MockExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.utils.skip_init(
                    nn.Linear, hidden_size, inter_size,
                    bias=False, device=device, dtype=model_dtype
                )
                self.up_proj = torch.nn.utils.skip_init(
                    nn.Linear, hidden_size, inter_size,
                    bias=False, device=device, dtype=model_dtype
                )
                self.down_proj = torch.nn.utils.skip_init(
                    nn.Linear, inter_size, hidden_size,
                    bias=False, device=device, dtype=model_dtype
                )

        return DeepSeekExpertWrapper(MockExpert(), device)

    expert_cache = LinearCache(
        make_module=make_module,
        main_size=total_gpu_experts,
        offload_size=total_offload_experts,
        buffer_size=buffer_size,
    )

    print_memory("After LinearCache")

    # 8. 再替换 MoE 层（此时 gate 和 shared_experts 已经有正确的权重）
    print("[DeepSeek] Replacing MoE layers...")
    lang_model = model.language.model  # model.language.model.layers
    layers_list = list(lang_model.layers)

    for layer_idx in range(num_layers):
        layer = lang_model.layers[layer_idx]

        if layer_idx < first_k_dense:
            # Layer 0: Dense FFN，不需要替换
            print(f"  Layer {layer_idx}: Dense FFN (skip)")
            continue

        # Layers 1-29: 提取 gate 和 shared_experts
        original_mlp = layer.mlp
        gate = original_mlp.gate
        shared_experts = original_mlp.shared_experts

        layer.mlp = DeepSeekMoeWrapperBaseline(
            lang_config=lang_cfg,
            layer_id=layer_idx,
            gate=gate,
            shared_experts=shared_experts,
            expert_cache=expert_cache,
            layers=layers_list,
            global_config=full_config,
        )

    print(f"[DeepSeek] Replaced {num_layers - first_k_dense} MoE layers")

    # 9. 加载 routed expert 权重
    print("[DeepSeek] Loading routed expert weights...")
    index_path = os.path.join(model_id, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    _load_routed_experts(
        model_id, weight_map, expert_cache, cache_sizes,
        num_layers, num_routed_experts, first_k_dense, model_dtype
    )

    torch.cuda.empty_cache()

    print_memory("After Expert Load")
    model.eval()

    gc.collect()
    torch.cuda.empty_cache()

    return model, lang_cfg, expert_cache


def build_offload_model_deepseek_skip_offload(
    model_id: str,
    attn_implementation: str = "flash_attention_2",
    device_map="auto",
):
    """构建 DeepSeek-VL2 Skip-Offload 模型"""
    config = get_deepseek_offload_config()
    device = _resolve_runtime_device(device_map, config['device'])
    raw_cache_config = config['cache_size_per_layer']
    buffer_size = config['buffer_size']
    model_dtype = config['model_dtype']
    skip_keep_k = config["skip_keep_k"]
    decode_skip_keep_k = config["decode_skip_keep_k"]
    first_k_dense = config['first_k_dense_replace']
    attn_implementation = _normalize_attn_implementation(attn_implementation)

    print("\n" + "=" * 80)
    print(f"[DeepSeek Skip-Offload] Building Model from: {model_id}")
    print("=" * 80)
    print(f"[DeepSeek Skip-Offload] device_map={device_map!r}, target_device={device}")
    print_memory("Start")

    deepseek_vl2_repo = str(_DEEPSEEK_VL2_REPO)
    if deepseek_vl2_repo not in sys.path:
        sys.path.insert(0, deepseek_vl2_repo)

    try:
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config, DeepseekVLV2ForCausalLM
        from deepseek_vl2.models.modeling_deepseek import DeepseekV2ForCausalLM
        print("[DeepSeek Skip-Offload] Loaded custom model classes from DeepSeek-VL2 repo")
    except ImportError as e:
        print(f"[DeepSeek Skip-Offload] Failed to load from DeepSeek-VL2 repo: {e}")
        raise RuntimeError(f"请确保 DeepSeek-VL2 仓库已存在于 {_DEEPSEEK_VL2_REPO}")

    full_config = DeepseekVLV2Config.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
        trust_remote_code=True
    )
    full_config._attn_implementation = attn_implementation

    print(f"[DeepSeek Skip-Offload] Attention implementation: {full_config._attn_implementation}")

    lang_cfg = full_config.language_config
    num_layers = lang_cfg.num_hidden_layers
    num_routed_experts = lang_cfg.n_routed_experts

    print(f"[DeepSeek Skip-Offload] Model config: {num_layers} layers, {num_routed_experts} routed experts/layer")

    model = _build_deepseek_skeleton_model(DeepseekVLV2ForCausalLM, full_config, device, model_dtype)
    print("[DeepSeek Skip-Offload] Created skeleton model (routed experts omitted from main model)")

    if isinstance(raw_cache_config, int):
        cache_sizes = [raw_cache_config] * num_layers
    elif isinstance(raw_cache_config, list):
        if len(raw_cache_config) != num_layers:
            raise ValueError(f"cache_size_per_layer length mismatch: {len(raw_cache_config)} vs {num_layers}")
        cache_sizes = raw_cache_config
    else:
        raise TypeError("cache_size_per_layer must be int or List[int]")

    print(f"[DeepSeek Skip-Offload] Cache sizes: {cache_sizes[:5]}... (first 5 layers)")

    total_gpu_experts = sum(cache_sizes[first_k_dense:])
    total_offload_experts = (num_layers - first_k_dense) * num_routed_experts

    print(f"[DeepSeek Skip-Offload] Expert cache allocation:")
    print(f"  - Total Layers: {num_layers} (Layer 0-{first_k_dense-1}: dense, Layer {first_k_dense}-{num_layers-1}: MoE)")
    print(f"  - Total GPU Slots: {total_gpu_experts}")
    print(f"  - Total Offload Experts: {total_offload_experts}")

    print_memory("Before Trunk Load")
    print(f"[DeepSeek Skip-Offload] Loading trunk weights directly to target device {device}...")
    loaded_count, missing_count = _load_trunk_weights_deepseek_to_model(model, model_id, device)
    print(
        f"[DeepSeek Skip-Offload] Loaded {loaded_count} non-routed tensors directly to target device {device} "
        f"(missing={missing_count})"
    )
    print_memory("After Trunk Load")
    torch.cuda.empty_cache()

    def make_module():
        hidden_size = lang_cfg.hidden_size
        inter_size = lang_cfg.moe_intermediate_size

        class MockExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.utils.skip_init(
                    nn.Linear, hidden_size, inter_size,
                    bias=False, device=device, dtype=model_dtype
                )
                self.up_proj = torch.nn.utils.skip_init(
                    nn.Linear, hidden_size, inter_size,
                    bias=False, device=device, dtype=model_dtype
                )
                self.down_proj = torch.nn.utils.skip_init(
                    nn.Linear, inter_size, hidden_size,
                    bias=False, device=device, dtype=model_dtype
                )

        return DeepSeekExpertWrapper(MockExpert(), device)

    expert_cache = LinearCache(
        make_module=make_module,
        main_size=total_gpu_experts,
        offload_size=total_offload_experts,
        buffer_size=buffer_size,
    )

    print_memory("After LinearCache")

    print("[DeepSeek Skip-Offload] Replacing MoE layers...")
    lang_model = model.language.model
    layers_list = list(lang_model.layers)

    for layer_idx in range(num_layers):
        layer = lang_model.layers[layer_idx]

        if layer_idx < first_k_dense:
            print(f"  Layer {layer_idx}: Dense FFN (skip-offload)")
            continue

        original_mlp = layer.mlp
        gate = original_mlp.gate
        shared_experts = original_mlp.shared_experts

        layer.mlp = DeepSeekMoeWrapperSkipOffload(
            lang_config=lang_cfg,
            layer_id=layer_idx,
            gate=gate,
            shared_experts=shared_experts,
            expert_cache=expert_cache,
            skip_keep_k=skip_keep_k,
            decode_skip_keep_k=decode_skip_keep_k,
            layers=layers_list,
            global_config=full_config,
        )

    print(f"[DeepSeek Skip-Offload] Replaced {num_layers - first_k_dense} MoE layers")

    print("[DeepSeek Skip-Offload] Loading routed expert weights...")
    index_path = os.path.join(model_id, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    _load_routed_experts(
        model_id, weight_map, expert_cache, cache_sizes,
        num_layers, num_routed_experts, first_k_dense, model_dtype
    )

    torch.cuda.empty_cache()

    print_memory("After Expert Load")
    model.eval()

    gc.collect()
    torch.cuda.empty_cache()

    return model, lang_cfg, expert_cache


def _load_routed_experts(
    model_id, weight_map, expert_cache, cache_sizes,
    num_layers, num_routed_experts, first_k_dense, model_dtype
):
    """加载 routed expert 权重到 LinearCache"""
    # 构建 expert -> {weight_type: file} 映射
    expert_files = {}
    for layer_idx in range(first_k_dense, num_layers):
        for expert_idx in range(num_routed_experts):
            gate_key = f"language.model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
            up_key = f"language.model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
            down_key = f"language.model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"

            if gate_key in weight_map and up_key in weight_map and down_key in weight_map:
                expert_files[(layer_idx, expert_idx)] = {
                    'gate': weight_map[gate_key],
                    'up': weight_map[up_key],
                    'down': weight_map[down_key]
                }

    # 构建 file -> experts 映射（用于显示进度）
    file_to_expert_count = defaultdict(int)
    for files_dict in expert_files.values():
        for fname in set(files_dict.values()):
            file_to_expert_count[fname] += 1

    # 预先确定哪些 expert 放在 GPU
    gpu_experts = {}
    for layer_idx in range(first_k_dense, num_layers):
        current_layer_k = cache_sizes[layer_idx]
        gpu_experts[layer_idx] = set(random.sample(range(num_routed_experts), min(current_layer_k, num_routed_experts)))

    # 按文件分组加载，但需要处理跨文件的 expert
    files_to_load = sorted(set(fname for files_dict in expert_files.values() for fname in files_dict.values()))
    file_handles = {}

    # 打开所有需要的文件
    for fname in files_to_load:
        fpath = os.path.join(model_id, fname)
        file_handles[fname] = safe_open(fpath, framework="pt", device="cpu")

    try:
        files_processed = 0
        total_files = len(files_to_load)

        # 按 expert 加载（而不是按文件）
        for (layer_idx, expert_idx), files_dict in expert_files.items():
            gate_key = f"language.model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
            up_key = f"language.model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
            down_key = f"language.model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"

            try:
                gate_w = file_handles[files_dict['gate']].get_tensor(gate_key).to(dtype=model_dtype)
                up_w = file_handles[files_dict['up']].get_tensor(up_key).to(dtype=model_dtype)
                down_w = file_handles[files_dict['down']].get_tensor(down_key).to(dtype=model_dtype)
            except Exception as e:
                print(f"    Error loading expert {expert_idx} in layer {layer_idx}: {e}")
                continue

            # LinearCache 期望 [out, in] 格式
            w1 = gate_w.contiguous()
            w3 = up_w.contiguous()
            w2 = down_w.contiguous()

            expert_cache.add_linear_storage(
                uid=(layer_idx, expert_idx),
                storage=[w1.untyped_storage(), w2.untyped_storage(), w3.untyped_storage()],
                eviction_group=layer_idx,
                offload=(expert_idx not in gpu_experts[layer_idx])
            )

        # 显示加载进度
        for i, fname in enumerate(files_to_load, 1):
            print(f"  Processing shard {i}/{total_files}: {fname} ({file_to_expert_count[fname]} experts)")

    finally:
        # 关闭所有文件句柄
        for handle in file_handles.values():
            del handle
        gc.collect()


def _build_offload_model_deepseek_cached(
    model_id: str,
    device_map="auto",
    *,
    mode_label: str,
    prefill_keep_strategy: str,
    decode_search_strategy: str,
    replace_attention: bool,
):
    """构建 DeepSeek-VL2 Cached 系列模型"""
    config = get_deepseek_offload_config()
    device = _resolve_runtime_device(device_map, config['device'])
    raw_cache_config = config['cache_size_per_layer']
    buffer_size = config['buffer_size']
    model_dtype = config['model_dtype']
    first_k_dense = config['first_k_dense_replace']

    print("\n" + "=" * 80)
    print(f"[DeepSeek {mode_label.capitalize()}] Building Model from: {model_id}")
    print("=" * 80)
    print(f"[DeepSeek] device_map={device_map!r}, target_device={device}")
    print_memory("Start")

    # 1. 从 DeepSeek-VL2 仓库导入自定义类
    deepseek_vl2_repo = str(_DEEPSEEK_VL2_REPO)
    if deepseek_vl2_repo not in sys.path:
        sys.path.insert(0, deepseek_vl2_repo)

    # 2. 导入 DeepSeek-VL2 的配置和模型类
    try:
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config, DeepseekVLV2ForCausalLM
        from deepseek_vl2.models.modeling_deepseek import DeepseekV2ForCausalLM
        print("[DeepSeek] Loaded custom model classes from DeepSeek-VL2 repo")
    except ImportError as e:
        print(f"[DeepSeek] Failed to load from DeepSeek-VL2 repo: {e}")
        raise RuntimeError(f"请确保 DeepSeek-VL2 仓库已存在于 {_DEEPSEEK_VL2_REPO}")

    # 3. 加载配置（指定 torch_dtype）
    full_config = DeepseekVLV2Config.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
        trust_remote_code=True
    )
    full_config._attn_implementation = "flash_attention_2"

    lang_cfg = full_config.language_config
    num_layers = lang_cfg.num_hidden_layers
    num_routed_experts = lang_cfg.n_routed_experts  # 从配置读取
    cache_config = get_deepseek_cache_config(
        num_layers=num_layers,
        top_k=lang_cfg.num_experts_per_tok,
    )
    if decode_search_strategy == "offline":
        cache_config["ONLINE_MAX_LAYER_IDX"] = -1
    set_active_cache_config(cache_config)

    print(
        f"[DeepSeek] Model config: {num_layers} layers, {num_routed_experts} routed experts/layer"
    )
    print(f"[DeepSeek] Attention implementation: {full_config._attn_implementation}")

    # 4. 用真实 config 构建 skeleton：
    # trunk/gate/shared_experts 直接在 GPU 上初始化，routed experts 不进入主模型参数树。
    model = _build_deepseek_skeleton_model(DeepseekVLV2ForCausalLM, full_config, device, model_dtype)
    print("[DeepSeek] Created skeleton model (routed experts omitted from main model)")

    # 3. 处理 cache_sizes
    if isinstance(raw_cache_config, int):
        cache_sizes = [raw_cache_config] * num_layers
    elif isinstance(raw_cache_config, list):
        if len(raw_cache_config) != num_layers:
            raise ValueError(f"cache_size_per_layer length mismatch")
        cache_sizes = raw_cache_config
    else:
        raise TypeError("cache_size_per_layer must be int or List[int]")

    total_gpu_experts = sum(cache_sizes[first_k_dense:])
    total_offload_experts = (num_layers - first_k_dense) * num_routed_experts
    print(f"[DeepSeek] GPU Slots: {total_gpu_experts}, Offload Experts: {total_offload_experts}")

    # 5. 先加载 trunk 权重（让 gate 和 shared_experts 有正确的大小）
    print_memory("Before Trunk Load")
    print(f"[DeepSeek] Loading trunk weights directly to target device {device}...")
    loaded_count, missing_count = _load_trunk_weights_deepseek_to_model(model, model_id, device)
    print(f"[DeepSeek] ✅ Loaded {loaded_count} non-routed tensors directly to target device {device} (missing={missing_count})")
    print_memory("After Trunk Load")
    torch.cuda.empty_cache()

    # 6. 构造 LinearCache（routed experts 只进入 cache，不进入主模型）
    def make_module():
        hidden_size = lang_cfg.hidden_size
        inter_size = lang_cfg.moe_intermediate_size

        class MockExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.utils.skip_init(
                    nn.Linear, hidden_size, inter_size,
                    bias=False, device=device, dtype=model_dtype
                )
                self.up_proj = torch.nn.utils.skip_init(
                    nn.Linear, hidden_size, inter_size,
                    bias=False, device=device, dtype=model_dtype
                )
                self.down_proj = torch.nn.utils.skip_init(
                    nn.Linear, inter_size, hidden_size,
                    bias=False, device=device, dtype=model_dtype
                )

        return DeepSeekExpertWrapper(MockExpert(), device)

    expert_cache = LinearCache(
        make_module=make_module,
        main_size=total_gpu_experts,
        offload_size=total_offload_experts,
        buffer_size=buffer_size,
    )

    print_memory("After LinearCache")

    # 7. 再替换 MoE 层（使用 Cached Wrapper）
    print(f"[DeepSeek {mode_label.capitalize()}] Replacing MoE layers with Cached Wrapper...")
    lang_model = model.language.model  # model.language.model.layers
    layers_list = list(lang_model.layers)

    for layer_idx in range(num_layers):
        layer = lang_model.layers[layer_idx]

        if layer_idx < first_k_dense:
            print(f"  Layer {layer_idx}: Dense FFN (skip)")
            continue

        original_mlp = layer.mlp
        gate = original_mlp.gate
        shared_experts = original_mlp.shared_experts

        # 初始化 HybridStorageManager
        layer_cache_manager = HybridStorageManager(
            layer_idx=layer_idx,
            num_experts=num_routed_experts,
            hidden_dim=lang_cfg.hidden_size,
            device=device
        )

        layer.mlp = DeepSeekMoeWrapperCached(
            lang_config=lang_cfg,
            layer_id=layer_idx,
            gate=gate,
            shared_experts=shared_experts,
            expert_cache=expert_cache,
            cache_manager=layer_cache_manager,
            prefill_keep_strategy=prefill_keep_strategy,
            decode_search_strategy=decode_search_strategy,
            layers=layers_list,
            global_config=full_config,
        )

    print(f"[DeepSeek] Replaced {num_layers - first_k_dense} MoE layers with Cached Wrapper")

    # 8. 加载 routed expert 权重
    print("[DeepSeek] Loading routed expert weights...")
    index_path = os.path.join(model_id, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    _load_routed_experts(
        model_id, weight_map, expert_cache, cache_sizes,
        num_layers, num_routed_experts, first_k_dense, model_dtype
    )

    torch.cuda.empty_cache()

    # 9. 替换 Attention 模块（用于 Importance Sampling）
    if replace_attention:
        print("[DeepSeek Hybrid] Replacing Attention modules...")
        for i in range(first_k_dense, num_layers):
            layer = lang_model.layers[i]
            layer.self_attn = HybridDeepSeekAttention(layer.self_attn)
        print(f"✓ Replaced attention in {num_layers - first_k_dense} layers")

    print_memory("After Expert Load")
    model.eval()

    gc.collect()
    torch.cuda.empty_cache()

    return model, lang_cfg, expert_cache


def build_offload_model_deepseek_hybrid(model_id: str, device_map="auto"):
    """构建 DeepSeek-VL2 Hybrid 模型 (Offload + Similarity Cache)"""
    return _build_offload_model_deepseek_cached(
        model_id=model_id,
        device_map=device_map,
        mode_label="hybrid",
        prefill_keep_strategy="importance",
        decode_search_strategy="hybrid",
        replace_attention=True,
    )


def build_offload_model_deepseek_offline(model_id: str, device_map="auto"):
    """构建 DeepSeek-VL2 Offline 消融模型"""
    return _build_offload_model_deepseek_cached(
        model_id=model_id,
        device_map=device_map,
        mode_label="offline",
        prefill_keep_strategy="fixed_keep_k",
        decode_search_strategy="offline",
        replace_attention=False,
    )


def build_offload_model_deepseek_online(model_id: str, device_map="auto"):
    """构建 DeepSeek-VL2 Online 消融模型"""
    return _build_offload_model_deepseek_cached(
        model_id=model_id,
        device_map=device_map,
        mode_label="online",
        prefill_keep_strategy="fixed_keep_k",
        decode_search_strategy="hybrid",
        replace_attention=False,
    )


def build_full_model_deepseek_skip(model_id: str, device_map="auto"):
    """构建 DeepSeek-VL2 Skip Baseline 模型（全量 GPU + fixed-slot skip）"""
    config = get_deepseek_offload_config()
    device = _resolve_runtime_device(device_map, config['device'])
    model_dtype = config['model_dtype']
    skip_keep_k = config["skip_keep_k"]
    decode_skip_keep_k = config["decode_skip_keep_k"]
    first_k_dense = config['first_k_dense_replace']

    print("\n" + "=" * 80)
    print(f"[DeepSeek Skip] Building Full Model from: {model_id}")
    print("=" * 80)
    print(f"[DeepSeek Skip] device_map={device_map!r}, target_device={device}")
    print_memory("Start")

    deepseek_vl2_repo = str(_DEEPSEEK_VL2_REPO)
    if deepseek_vl2_repo not in sys.path:
        sys.path.insert(0, deepseek_vl2_repo)

    try:
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config, DeepseekVLV2ForCausalLM
        print("[DeepSeek Skip] Loaded custom model classes from DeepSeek-VL2 repo")
    except ImportError as e:
        print(f"[DeepSeek Skip] Failed to load from DeepSeek-VL2 repo: {e}")
        raise RuntimeError(f"请确保 DeepSeek-VL2 仓库已存在于 {_DEEPSEEK_VL2_REPO}")

    full_config = DeepseekVLV2Config.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    full_config._attn_implementation = "flash_attention_2"

    lang_cfg = full_config.language_config
    num_layers = lang_cfg.num_hidden_layers
    set_active_cache_config(
        get_deepseek_cache_config(
            num_layers=num_layers,
            top_k=lang_cfg.num_experts_per_tok,
        )
    )

    print(f"[DeepSeek Skip] Attention implementation: {full_config._attn_implementation}")
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        model_id,
        config=full_config,
        torch_dtype=model_dtype,
        trust_remote_code=True,
        device_map=device_map,
    )

    lang_model = model.language.model
    layers_list = list(lang_model.layers)
    print("[DeepSeek Skip] Replacing MoE layers with Skip Wrapper...")
    for layer_idx in range(num_layers):
        layer = lang_model.layers[layer_idx]

        if layer_idx < first_k_dense:
            print(f"  Layer {layer_idx}: Dense FFN (skip)")
            continue

        original_mlp = layer.mlp
        gate = original_mlp.gate
        shared_experts = original_mlp.shared_experts
        experts = original_mlp.experts

        layer.mlp = DeepSeekMoeWrapperSkipBaseline(
            lang_config=lang_cfg,
            layer_id=layer_idx,
            gate=gate,
            shared_experts=shared_experts,
            expert_cache=None,
            experts=experts,
            skip_keep_k=skip_keep_k,
            decode_skip_keep_k=decode_skip_keep_k,
            layers=layers_list,
            global_config=full_config,
        )

    print_memory("After Wrapper Replace")
    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, lang_cfg, None
