import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional


def get_prebuilt_root() -> Path:
    raw = os.getenv("TABLEMOE_EXTENSIONS_DIR")
    if raw:
        return Path(os.path.expanduser(raw)).resolve()
    return Path(__file__).resolve().parent / "prebuilt"


def get_arch_build_dir(arch: str) -> Path:
    return get_prebuilt_root() / f"sm{arch}"


def find_prebuilt_shared_object(ext_name: str, arch: str) -> Optional[Path]:
    build_dir = get_arch_build_dir(arch)
    if not build_dir.exists():
        return None

    direct_matches = sorted(build_dir.glob(f"{ext_name}*.so"))
    if direct_matches:
        return direct_matches[0]

    nested_matches = sorted(build_dir.glob(f"**/{ext_name}*.so"))
    if nested_matches:
        return nested_matches[0]

    return None


def load_prebuilt_extension(ext_name: str, arch: str):
    so_path = find_prebuilt_shared_object(ext_name, arch)
    if so_path is None:
        return None

    if ext_name in sys.modules:
        return sys.modules[ext_name]

    spec = importlib.util.spec_from_file_location(ext_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {so_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[ext_name] = module
    spec.loader.exec_module(module)
    return module
