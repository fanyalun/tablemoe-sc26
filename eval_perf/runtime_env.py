import os
from typing import Dict, Optional

import torch


_RUNTIME_CONFIG: Optional[Dict[str, object]] = None


def _read_int_env(*names: str) -> Optional[int]:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return None


def _format_cpu_affinity() -> Optional[str]:
    if not hasattr(os, "sched_getaffinity"):
        return None

    try:
        cpus = sorted(os.sched_getaffinity(0))
    except OSError:
        return None

    if not cpus:
        return ""

    ranges = []
    start = cpus[0]
    prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = cpu
        prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def configure_runtime_env() -> Dict[str, object]:
    global _RUNTIME_CONFIG
    if _RUNTIME_CONFIG is not None:
        return _RUNTIME_CONFIG

    config: Dict[str, object] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selected_gpu": os.environ.get("TABLEMOE_SELECTED_GPU"),
        "gpu_bus_id": os.environ.get("TABLEMOE_GPU_BUS_ID"),
        "numa_node": os.environ.get("TABLEMOE_NUMA_NODE"),
        "mem_node": os.environ.get("TABLEMOE_MEM_NODE"),
        "cpu_list": os.environ.get("TABLEMOE_CPU_LIST"),
        "bind_source": os.environ.get("TABLEMOE_BIND_SOURCE"),
        "bind_strategy": os.environ.get("TABLEMOE_BIND_STRATEGY"),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numexpr_num_threads_env": os.environ.get("NUMEXPR_NUM_THREADS"),
        "tablemoe_cpu_threads_env": os.environ.get("TABLEMOE_CPU_THREADS"),
        "tablemoe_interop_threads_env": os.environ.get("TABLEMOE_INTEROP_THREADS"),
        "moe_io_threads_env": os.environ.get("MOE_IO_THREADS"),
        "tablemoe_aio_threads_env": os.environ.get("TABLEMOE_AIO_THREADS"),
        "tablemoe_init_threads_env": os.environ.get("TABLEMOE_INIT_THREADS"),
        "tablemoe_fetch_threads_env": os.environ.get("TABLEMOE_FETCH_THREADS"),
        "tablemoe_use_physical_cores": os.environ.get("TABLEMOE_USE_PHYSICAL_CORES"),
        "tablemoe_numa_core_limit": os.environ.get("TABLEMOE_NUMA_CORE_LIMIT"),
        "tablemoe_hardware_preset_name": os.environ.get("TABLEMOE_HARDWARE_PRESET_NAME"),
        "tablemoe_hardware_preset_path": os.environ.get("TABLEMOE_HARDWARE_PRESET_PATH"),
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
        "omp_places": os.environ.get("OMP_PLACES"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "cpu_affinity": _format_cpu_affinity(),
    }

    cpu_threads = _read_int_env("TABLEMOE_CPU_THREADS", "OMP_NUM_THREADS")
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    config["torch_num_threads"] = int(torch.get_num_threads())

    interop_threads = _read_int_env("TABLEMOE_INTEROP_THREADS")
    if interop_threads is not None:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError as exc:
            config["torch_num_interop_threads_error"] = str(exc)
    config["torch_num_interop_threads"] = int(torch.get_num_interop_threads())

    print("\n[RuntimeEnv] Hardware binding")
    for key in (
        "cuda_visible_devices",
        "selected_gpu",
        "gpu_bus_id",
        "numa_node",
        "mem_node",
        "cpu_list",
        "bind_source",
        "bind_strategy",
        "omp_num_threads_env",
        "mkl_num_threads_env",
        "openblas_num_threads_env",
        "numexpr_num_threads_env",
        "tablemoe_cpu_threads_env",
        "tablemoe_interop_threads_env",
        "moe_io_threads_env",
        "tablemoe_aio_threads_env",
        "tablemoe_init_threads_env",
        "tablemoe_fetch_threads_env",
        "tablemoe_use_physical_cores",
        "tablemoe_numa_core_limit",
        "tablemoe_hardware_preset_name",
        "tablemoe_hardware_preset_path",
        "omp_proc_bind",
        "omp_places",
        "tokenizers_parallelism",
        "cpu_affinity",
        "torch_num_threads",
        "torch_num_interop_threads",
    ):
        print(f"  {key}: {config.get(key)}")
    if "torch_num_interop_threads_error" in config:
        print(f"  torch_num_interop_threads_error: {config['torch_num_interop_threads_error']}")
    print("")

    _RUNTIME_CONFIG = config
    return config
