import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR_STR = str(SCRIPT_DIR)
if SCRIPT_DIR_STR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR_STR)

from common import (
    DEFAULT_MULTI_DATASETS,
    DATASET_LOADERS,
    EVAL_FNS,
    MODEL_SPECS,
    PROMPT_BUILDERS,
    _default_output_dir,
    build_dataset_output_dir,
    detect_dataset_key,
    resolve_dataset_specs,
    resolve_model_name,
    sample_dataset_for_benchmark,
    write_run_outputs,
)
from moe_infinity_common import (
    apply_moe_runtime_defaults,
    build_moe_runtime,
    cleanup_moe_runtime,
    resolve_image_root,
    run_moe_sample,
    run_moe_warmup,
    save_runtime_trace,
    validate_moe_runtime_args,
)
from runtime_env import configure_runtime_env


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--lmudata-dir", type=str, default=None)
    parser.add_argument("--datasets", type=str, default=DEFAULT_MULTI_DATASETS)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--offload-path", type=str, default=None)
    parser.add_argument("--device-memory-ratio", type=float, default=0.1)
    parser.add_argument("--expert-cache-ratio", type=float, default=0.5)
    parser.add_argument("--gpu-total-memory-mib", type=int, default=None)
    parser.add_argument("--trace-capacity", type=int, default=1000)
    parser.add_argument("--trace-path", type=str, default=None)
    parser.add_argument("--save-trace-path", type=str, default=None)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--profile-sample-id", type=str, default=None)
    parser.add_argument("--profile-no-streamer-sync", action="store_true")
    parser.add_argument("--profile-output-json", type=str, default=None)
    parser.add_argument(
        "--deepseek-repo",
        type=str,
        default=str(REPO_ROOT / "third_party" / "DeepSeek-VL2"),
    )
    return parser


def _resolve_available_dataset_specs(dataset_specs, multi_dataset):
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


def _resolve_sample_id(row, idx):
    return row.get("id", row.get("index", idx))


def _resolve_source_index(row, idx):
    return row.get("__source_index__", idx)


def _is_cuda_device_assert(exc: Exception) -> bool:
    message = str(exc).lower()
    return "device-side assert triggered" in message


def _dump_failed_sample(output_dir, benchmark, idx, sample_id, source_index, row, exc):
    payload = {
        "benchmark": benchmark,
        "sample_index": idx,
        "sample_id": str(sample_id),
        "source_index": int(source_index)
        if isinstance(source_index, int)
        else str(source_index),
        "error": str(exc),
        "row": row.to_dict() if hasattr(row, "to_dict") else dict(row),
    }
    path = os.path.join(output_dir, f"failed_sample_{idx}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    print(f"Saved failed sample context to {path}")


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


def _run_dataset_benchmark(
    runtime,
    args,
    runtime_hardware_config,
    dataset_spec,
    dataset_index,
    total_datasets,
    multi_dataset,
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
    print(f"[{dataset_index}/{total_datasets}] {MODEL_SPECS[args.model]['label']} MoE-Infinity | Benchmark: {benchmark}")
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

    img_root = resolve_image_root(data_file)

    if len(dataset) > 0 and args.warmup_samples > 0:
        try:
            run_moe_warmup(
                runtime,
                dataset,
                build_prompt,
                img_root,
                args.warmup_samples,
            )
            print("Warmup completed")
        except Exception as exc:
            print(f"Warmup failed (ignoring): {exc}")

    dataset = sample_dataset_for_benchmark(
        dataset,
        args.sample_ratio,
        args.sample_seed,
    )
    dataset = _maybe_restrict_dataset_for_profile(dataset, args)

    results = []
    metrics = []
    for idx, row in tqdm(
        dataset.iterrows(),
        total=len(dataset),
        desc=f"{args.model}-moe_infinity-{benchmark}",
    ):
        sample_id = _resolve_sample_id(row, idx)
        source_index = _resolve_source_index(row, idx)
        try:
            result_item, metric_item = run_moe_sample(
                runtime,
                row,
                build_prompt,
                img_root,
                args,
                sample_id,
            )
            results.append(result_item)
            metrics.append(metric_item)
        except Exception as exc:
            print(
                f"Error processing sample {idx} "
                f"(sample_id={sample_id}, source_index={source_index}): {exc}"
            )
            _dump_failed_sample(
                output_dir,
                benchmark,
                idx,
                sample_id,
                source_index,
                row,
                exc,
            )
            traceback.print_exc()
            if _is_cuda_device_assert(exc):
                raise RuntimeError(
                    f"CUDA device-side assert on benchmark={benchmark}, "
                    f"sample_index={idx}, sample_id={sample_id}, "
                    f"source_index={source_index}"
                ) from exc
            continue

    run_arguments = vars(args).copy()
    run_arguments["data_dir"] = data_file
    run_arguments["benchmark"] = benchmark
    run_arguments["output_dir"] = output_dir

    run_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "moe_infinity",
        "model": args.model,
        "model_name": args.model_name,
        "benchmark": benchmark,
        "model_family": MODEL_SPECS[args.model]["family"],
        "arguments": run_arguments,
        "runtime_hardware_config": runtime_hardware_config,
        "runtime_meta": runtime["runtime_meta"],
        "weight_stats": runtime["weight_stats"],
    }

    write_run_outputs(output_dir, run_config, results, metrics)

    if args.profile_timing and profiler is not None and profiler.results:
        profile_output_path = args.profile_output_json
        if profile_output_path is None:
            profile_output_path = os.path.join(output_dir, "timing_profile.json")
        elif multi_dataset:
            base, ext = os.path.splitext(profile_output_path)
            suffix = ext or ".json"
            profile_output_path = f"{base}_{benchmark}{suffix}"

        payload = profiler.results[0] if len(profiler.results) == 1 else profiler.results
        with open(profile_output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"Saved timing profile to {profile_output_path}")

    print("Evaluating results...")
    eval_fn(results=results, output_dir=output_dir)


def main():
    parser = _build_parser()
    args = parser.parse_args()
    runtime_hardware_config = configure_runtime_env()

    apply_moe_runtime_defaults(args)
    validate_moe_runtime_args(args)

    if bool(args.data_dir) == bool(args.lmudata_dir):
        raise ValueError("Specify exactly one of --data-dir or --lmudata-dir")

    if args.output_dir is None:
        args.output_dir = _default_output_dir("moe_infinity", args.model)

    args.model_name = resolve_model_name(args.model_name, args.model_path, args.model)
    multi_dataset = args.lmudata_dir is not None
    dataset_specs = resolve_dataset_specs(args.data_dir, args.lmudata_dir, args.datasets)
    dataset_specs = _resolve_available_dataset_specs(dataset_specs, multi_dataset)
    if not dataset_specs:
        raise FileNotFoundError("No available dataset files to benchmark")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.offload_path, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"{MODEL_SPECS[args.model]['label']} MoE-Infinity test")
    print("=" * 80)
    print(f"model_name: {args.model_name}")
    print(f"output_dir: {args.output_dir}")
    print(f"datasets: {', '.join(spec['benchmark'] for spec in dataset_specs)}")
    print("")

    runtime = None
    try:
        runtime = build_moe_runtime(args)
        if args.prefetch and args.trace_path is None:
            print("[WARNING] prefetch 已开启，但未提供 trace_path，将使用冷启动轨迹。")

        total_datasets = len(dataset_specs)
        for dataset_index, dataset_spec in enumerate(dataset_specs, start=1):
            _run_dataset_benchmark(
                runtime,
                args,
                runtime_hardware_config,
                dataset_spec,
                dataset_index,
                total_datasets,
                multi_dataset,
            )

        if args.save_trace_path is not None:
            save_runtime_trace(runtime, args.save_trace_path)
            print(f"Saved trace to {args.save_trace_path}")
    finally:
        cleanup_moe_runtime(runtime)


if __name__ == "__main__":
    main()
