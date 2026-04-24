import os
import threading
import warnings
from pathlib import Path

import torch

from ._prebuilt import get_arch_build_dir, load_prebuilt_extension


_ENV_NAME = "TABLEMOE_USE_QWEN_TOPK_SOFTMAX"
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


def _arch_key(device: torch.device) -> str | None:
    if not torch.cuda.is_available():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return f"{major}{minor}"


def _extension_name(arch: str) -> str:
    return f"tablemoe_qwen_topk_softmax_sm{arch}"


def _arch_list_value(arch: str) -> str:
    return f"{arch[0]}.{arch[1:]}"


def build_precompiled_extension(arch: str, *, verbose: bool = False):
    source_root = Path(__file__).resolve().parent / "csrc" / "qwen_topk_softmax"
    sources = [
        str(source_root / "qwen_topk_softmax_bindings.cpp"),
        str(source_root / "qwen_topk_softmax_kernels.cu"),
    ]
    build_directory = get_arch_build_dir(arch)
    build_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list_value(arch))

    from torch.utils.cpp_extension import load

    ext_name = _extension_name(arch)
    return load(
        name=ext_name,
        sources=sources,
        build_directory=str(build_directory),
        extra_include_paths=[str(source_root)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
        ],
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
                f"Qwen topk_softmax prebuilt load failed for sm_{arch}, falling back to JIT: {exc}",
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
                f"Qwen topk_softmax disabled: JIT build failed for sm_{arch}: {exc}",
            )
            extension = None

        _EXTENSION_CACHE[arch] = extension
        return extension


def _fallback_topk_softmax(router_logits: torch.Tensor, top_k: int):
    probs = torch.softmax(router_logits, dim=-1, dtype=torch.float)
    return torch.topk(probs, top_k, dim=-1)


def qwen_topk_softmax(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    normalize_topk: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if router_logits.dim() != 2:
        raise ValueError(f"qwen_topk_softmax expects 2D logits, got {tuple(router_logits.shape)}")

    weights = None
    indices = None

    if router_logits.device.type == "cuda" and _env_enabled():
        extension = _load_extension(router_logits.device)
        if extension is not None:
            logits_fp32 = router_logits.to(dtype=torch.float32)
            if not logits_fp32.is_contiguous():
                logits_fp32 = logits_fp32.contiguous()
            try:
                weights, indices = extension.qwen_topk_softmax(logits_fp32, int(top_k))
            except Exception as exc:
                _warn_once(
                    "runtime_failed",
                    f"Qwen topk_softmax disabled at runtime, falling back to torch ops: {exc}",
                )

    if weights is None or indices is None:
        weights, indices = _fallback_topk_softmax(router_logits, top_k)

    if normalize_topk:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)

    return weights, indices
