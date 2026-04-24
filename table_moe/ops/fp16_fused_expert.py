import os
import threading
import warnings
from pathlib import Path
from typing import Optional

import torch

from ._prebuilt import get_arch_build_dir, load_prebuilt_extension


_ENV_NAME = "TABLEMOE_USE_FP16_FUSED_EXPERT"
_MAX_TOKENS_ENV_NAME = "TABLEMOE_FP16_FUSED_MAX_TOKENS"
_BUILD_LOCK = threading.Lock()
_EXTENSION_CACHE = {}
_FAILURE_ONCE = set()


def _env_enabled() -> bool:
    raw = os.getenv(_ENV_NAME)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _warn_once(key: str, message: str) -> None:
    if key in _FAILURE_ONCE:
        return
    _FAILURE_ONCE.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _max_tokens() -> int:
    raw = os.getenv(_MAX_TOKENS_ENV_NAME)
    if raw is None:
        return 1
    try:
        value = int(raw)
    except ValueError:
        _warn_once(
            "invalid_max_tokens",
            f"Invalid {_MAX_TOKENS_ENV_NAME}={raw!r}, falling back to 1.",
        )
        return 1
    if value <= 0:
        _warn_once(
            "nonpositive_max_tokens",
            f"Invalid {_MAX_TOKENS_ENV_NAME}={raw!r}, falling back to 1.",
        )
        return 1
    return value


def _resolve_cutlass_dir() -> Optional[Path]:
    cutlass_dir = Path(os.path.expanduser(os.getenv("CUTLASS_DIR", "~/cutlass")))
    include_root = cutlass_dir / "include" / "cutlass" / "cutlass.h"
    if include_root.exists():
        return cutlass_dir
    _warn_once(
        "cutlass_missing",
        f"FP16 fused expert disabled: CUTLASS not found at {cutlass_dir}. "
        "Set CUTLASS_DIR or install CUTLASS under ~/cutlass.",
    )
    return None


def _arch_key(device: torch.device) -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    if major < 8:
        _warn_once(
            f"unsupported_sm_{major}{minor}",
            f"FP16 fused expert disabled: unsupported CUDA capability sm_{major}{minor}.",
        )
        return None
    return f"{major}{minor}"


def _extension_name(arch: str) -> str:
    return f"tablemoe_fp16_fused_expert_sm{arch}"


def _arch_list_value(arch: str) -> str:
    return f"{arch[0]}.{arch[1:]}"


def build_precompiled_extension(arch: str, *, verbose: bool = False):
    cutlass_dir = _resolve_cutlass_dir()
    if cutlass_dir is None:
        return None

    source_root = Path(__file__).resolve().parent / "csrc" / "fp16_fused_expert"
    sources = [
        str(source_root / "fp16_fused_moe_mlp_bindings.cpp"),
        str(source_root / "fp16_fused_moe_mlp.cu"),
    ]
    include_dirs = [
        str(source_root),
        str(cutlass_dir / "include"),
        str(cutlass_dir / "tools" / "util" / "include"),
    ]
    build_directory = get_arch_build_dir(arch)
    build_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list_value(arch))

    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
    ]

    from torch.utils.cpp_extension import load

    ext_name = _extension_name(arch)
    return load(
        name=ext_name,
        sources=sources,
        build_directory=str(build_directory),
        extra_include_paths=include_dirs,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=["-O3", "-std=c++17", "-fPIC"],
        verbose=verbose,
    )


def _load_extension(device: torch.device):
    if not _env_enabled():
        return None
    if not torch.cuda.is_available():
        return None

    arch = _arch_key(device)
    if arch is None:
        return None

    with _BUILD_LOCK:
        if arch in _EXTENSION_CACHE:
            return _EXTENSION_CACHE[arch]

        ext_name = _extension_name(arch)
        try:
            extension = load_prebuilt_extension(ext_name, arch)
        except Exception as exc:
            _warn_once(
                f"prebuilt_failed_sm{arch}",
                f"FP16 fused expert prebuilt load failed for sm_{arch}, falling back to JIT: {exc}",
            )
            extension = None

        if extension is not None:
            _EXTENSION_CACHE[arch] = extension
            return extension

        try:
            extension = build_precompiled_extension(arch, verbose=False)
        except Exception as exc:
            _warn_once(
                f"build_failed_sm{arch}",
                f"FP16 fused expert disabled: JIT build failed for sm_{arch}: {exc}",
            )
            extension = None

        _EXTENSION_CACHE[arch] = extension
        return extension


def _eligible_weight(linear_wrapper) -> Optional[torch.Tensor]:
    linear_module = getattr(linear_wrapper, "linear_module", None)
    if linear_module is None or getattr(linear_module, "bias", None) is not None:
        return None
    weight = linear_module.weight
    if not weight.is_cuda or weight.dtype != torch.float16 or not weight.is_contiguous():
        return None
    return weight


def _ensure_workspace(owner, hidden_states: torch.Tensor, intermediate_dim: int):
    device = hidden_states.device
    dtype = hidden_states.dtype
    gate_shape = (hidden_states.shape[0], intermediate_dim)
    output_shape = tuple(hidden_states.shape)

    gate_buf = getattr(owner, "_fp16_fused_gate_buf", None)
    fused_buf = getattr(owner, "_fp16_fused_buf", None)
    output_buf = getattr(owner, "_fp16_fused_output_buf", None)

    if (
        gate_buf is None
        or gate_buf.shape != gate_shape
        or gate_buf.device != device
        or gate_buf.dtype != dtype
    ):
        gate_buf = torch.empty(gate_shape, device=device, dtype=dtype)
        owner._fp16_fused_gate_buf = gate_buf

    if (
        fused_buf is None
        or fused_buf.shape != gate_shape
        or fused_buf.device != device
        or fused_buf.dtype != dtype
    ):
        fused_buf = torch.empty(gate_shape, device=device, dtype=dtype)
        owner._fp16_fused_buf = fused_buf

    if (
        output_buf is None
        or tuple(output_buf.shape) != output_shape
        or output_buf.device != device
        or output_buf.dtype != dtype
    ):
        output_buf = torch.empty(output_shape, device=device, dtype=dtype)
        owner._fp16_fused_output_buf = output_buf

    return gate_buf, fused_buf, output_buf


def maybe_run_fp16_fused_expert(owner, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
    if not _env_enabled():
        return None
    if hidden_states.device.type != "cuda":
        return None
    if hidden_states.dtype != torch.float16:
        return None
    if hidden_states.ndim != 2:
        return None
    if hidden_states.shape[0] <= 0 or hidden_states.shape[0] > _max_tokens():
        return None
    if not hidden_states.is_contiguous():
        return None

    w1 = _eligible_weight(owner.w1)
    w3 = _eligible_weight(owner.w3)
    w2 = _eligible_weight(owner.w2)
    if w1 is None or w3 is None or w2 is None:
        return None

    extension = _load_extension(hidden_states.device)
    if extension is None:
        return None

    gate_buf, fused_buf, output_buf = _ensure_workspace(owner, hidden_states, w1.shape[0])
    extension.fused_moe_ffn_fp16_into(
        hidden_states,
        w1,
        w3,
        w2,
        gate_buf,
        fused_buf,
        output_buf,
    )
    return output_buf
