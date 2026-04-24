import argparse
from pathlib import Path

import torch

try:
    from ._prebuilt import find_prebuilt_shared_object, get_arch_build_dir
    from .fp16_fused_expert import build_precompiled_extension as build_fp16_fused_expert
    from .fp16_fused_expert import _extension_name as fp16_fused_expert_name
    from .qwen_topk_softmax import build_precompiled_extension as build_qwen_topk_softmax
    from .qwen_topk_softmax import _extension_name as qwen_topk_softmax_name
except ImportError:
    from table_moe.ops._prebuilt import find_prebuilt_shared_object, get_arch_build_dir
    from table_moe.ops.fp16_fused_expert import build_precompiled_extension as build_fp16_fused_expert
    from table_moe.ops.fp16_fused_expert import _extension_name as fp16_fused_expert_name
    from table_moe.ops.qwen_topk_softmax import build_precompiled_extension as build_qwen_topk_softmax
    from table_moe.ops.qwen_topk_softmax import _extension_name as qwen_topk_softmax_name


def _detect_arch() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot prebuild CUDA extensions.")
    index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return f"{major}{minor}"


def _resolve_arch(args_arch: str | None) -> str:
    if args_arch:
        return args_arch.replace(".", "")
    return _detect_arch()


def _report_shared_object(ext_name: str, arch: str) -> Path:
    so_path = find_prebuilt_shared_object(ext_name, arch)
    if so_path is None:
        raise RuntimeError(f"Built extension {ext_name} for sm{arch} but no shared object was found.")
    return so_path


def main():
    parser = argparse.ArgumentParser(description="Prebuild tablemoe CUDA extensions.")
    parser.add_argument("--arch", type=str, default=None, help="CUDA arch like 80 or 8.0. Defaults to current GPU.")
    parser.add_argument(
        "--only",
        choices=["all", "fp16_fused_expert", "qwen_topk_softmax"],
        default="all",
        help="Build only one extension.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose build output.")
    args = parser.parse_args()

    arch = _resolve_arch(args.arch)
    build_dir = get_arch_build_dir(arch)
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build_extensions] Target arch: sm{arch}")
    print(f"[build_extensions] Output dir: {build_dir}")

    if args.only in {"all", "qwen_topk_softmax"}:
        name = qwen_topk_softmax_name(arch)
        print(f"[build_extensions] Building {name} ...")
        build_qwen_topk_softmax(arch, verbose=args.verbose)
        print(f"[build_extensions] Built {name}: {_report_shared_object(name, arch)}")

    if args.only in {"all", "fp16_fused_expert"}:
        name = fp16_fused_expert_name(arch)
        print(f"[build_extensions] Building {name} ...")
        module = build_fp16_fused_expert(arch, verbose=args.verbose)
        if module is None:
            raise RuntimeError("Failed to build fp16_fused_expert: CUTLASS is not available.")
        print(f"[build_extensions] Built {name}: {_report_shared_object(name, arch)}")


if __name__ == "__main__":
    main()
