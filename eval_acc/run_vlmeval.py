import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VLM_EVAL_ROOT = REPO_ROOT / "third_party" / "VLMEvalKit"

METHOD_LABELS = {
    "transformers": "Transformers",
    "skip": "AdapMoE(+gating)",
    "offline": "+ALUT",
    "online": "+WINDOW",
    "tablemoe": "TableMoE",
}

DATASET_SPECS = {
    "MMBench_DEV_EN_V11": {
        "class": "ImageMCQDataset",
        "dataset": "MMBench_DEV_EN_V11",
    },
    "RealWorldQA": {
        "class": "ImageMCQDataset",
        "dataset": "RealWorldQA",
    },
    "AI2D_TEST": {
        "class": "ImageMCQDataset",
        "dataset": "AI2D_TEST",
    },
    "ScienceQA_TEST": {
        "class": "ImageMCQDataset",
        "dataset": "ScienceQA_TEST",
    },
    "POPE": {
        "class": "ImageYORNDataset",
        "dataset": "POPE",
    },
}

DATASET_ALIASES = {
    "mmbench": "MMBench_DEV_EN_V11",
    "mmbenchdevenv11": "MMBench_DEV_EN_V11",
    "realworldqa": "RealWorldQA",
    "ai2d": "AI2D_TEST",
    "ai2dtest": "AI2D_TEST",
    "scienceqa": "ScienceQA_TEST",
    "scienceqatest": "ScienceQA_TEST",
    "pope": "POPE",
}

DATASET_ENV_PREFIX = {
    "MMBench_DEV_EN_V11": "MMBENCH",
    "RealWorldQA": "REALWORLDQA",
    "AI2D_TEST": "AI2D",
    "ScienceQA_TEST": "SCIENCEQA",
    "POPE": "POPE",
}

MODEL_ALIASES = {
    "qwen": "qwen3vlmoe",
    "qwen3vl": "qwen3vlmoe",
    "qwen3vlmoe": "qwen3vlmoe",
    "qwen3vl30ba3binstruct": "qwen3vlmoe",
    "qwenqwen3vl30ba3binstruct": "qwen3vlmoe",
    "deepseek": "deepseekvl2",
    "deepseekvl2": "deepseekvl2",
}

METHOD_ALIASES = {
    "transformers": "transformers",
    "skip": "skip",
    "offline": "offline",
    "online": "online",
    "tablemoe": "tablemoe",
}

MODEL_SPECS = {
    "qwen3vlmoe": {
        "offline_root": "qwen_fp16",
        "methods": {
            "transformers": {
                "model_name": "qwen3_vl_transformers",
                "class": "Qwen3VLChat",
            },
            "skip": {
                "model_name": "qwen3_vl_adapmoe_gating",
                "class": "Qwen3VLSkipBaseline",
            },
            "tablemoe": {
                "model_name": "qwen3_vl_tablemoe",
                "class": "Qwen3VLTableMoE",
            },
            "offline": {
                "model_name": "qwen3_vl_tablemoe_offline",
                "class": "Qwen3VLOfflineTableMoE",
            },
            "online": {
                "model_name": "qwen3_vl_tablemoe_online",
                "class": "Qwen3VLOnlineTableMoE",
            },
        },
    },
    "deepseekvl2": {
        "offline_root": "ds_fp16",
        "methods": {
            "transformers": {
                "model_name": "deepseek_vl2_transformers",
                "class": "DeepSeekVL2",
            },
            "skip": {
                "model_name": "deepseek_vl2_adapmoe_gating",
                "class": "DeepSeekVL2SkipBaseline",
            },
            "tablemoe": {
                "model_name": "deepseek_vl2_tablemoe",
                "class": "DeepSeekVL2TableMoE",
            },
            "offline": {
                "model_name": "deepseek_vl2_tablemoe_offline",
                "class": "DeepSeekVL2OfflineTableMoE",
            },
            "online": {
                "model_name": "deepseek_vl2_tablemoe_online",
                "class": "DeepSeekVL2OnlineTableMoE",
            },
        },
    },
}


def normalize_token(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def canonicalize_model(model: str) -> str:
    normalized = normalize_token(model)
    resolved = MODEL_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            "Unsupported model. Expected one of: Qwen3-VL-30B-A3B-Instruct, DeepSeek-VL2"
        )
    return resolved


def canonicalize_method(method: str) -> str:
    normalized = normalize_token(method)
    resolved = METHOD_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError("Unsupported method. Expected one of: transformers, skip, offline, online, tablemoe")
    return resolved


def canonicalize_dataset(dataset: str) -> str:
    normalized = normalize_token(dataset)
    resolved = DATASET_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            "Unsupported dataset. Expected one of: "
            + ", ".join(DATASET_SPECS.keys())
        )
    return resolved


def parse_datasets(raw: str) -> list[str]:
    tokens = [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
    if not tokens:
        raise ValueError("datasets must not be empty")

    datasets = []
    for token in tokens:
        canonical = canonicalize_dataset(token)
        if canonical not in datasets:
            datasets.append(canonical)
    return datasets


def getenv_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def resolve_tablemoe_dirs(model: str, benchmark: str) -> tuple[Path, Path]:
    prefix = DATASET_ENV_PREFIX[benchmark]
    cache_override = getenv_path(f"{prefix}_CACHE_DIR")
    pca_override = getenv_path(f"{prefix}_PCA_DIR")

    offline_root = REPO_ROOT / "offline_table" / MODEL_SPECS[model]["offline_root"]
    cache_root = getenv_path("CACHE_ROOT") or (offline_root / "offline_table")
    pca_root = getenv_path("PCA_ROOT") or (offline_root / "clustering_results")
    cache_suffix = os.environ.get("CACHE_DIR_SUFFIX", "_LayerPCA_256")
    pca_suffix = os.environ.get("PCA_DIR_SUFFIX", "_LayerPCA_256")

    cache_dir = cache_override or (cache_root / f"{benchmark}{cache_suffix}")
    pca_dir = pca_override or (pca_root / f"{benchmark}{pca_suffix}")
    return pca_dir.resolve(), cache_dir.resolve()


def ensure_tablemoe_dirs(pca_dir: Path, cache_dir: Path, benchmark: str) -> None:
    if not pca_dir.is_dir():
        raise FileNotFoundError(
            f"PCA dir not found for {benchmark}: {pca_dir}. "
            "Please download the published offline table with "
            "offline_table/download_offline_table.sh or build it locally with "
            "offline_table/run_offline_table.sh."
        )
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Cache dir not found for {benchmark}: {cache_dir}. "
            "Please download the published offline table with "
            "offline_table/download_offline_table.sh or build it locally with "
            "offline_table/run_offline_table.sh."
        )


def build_dataset_config(benchmarks: list[str]) -> dict:
    return {benchmark: DATASET_SPECS[benchmark] for benchmark in benchmarks}


def build_model_config(
    model: str,
    method: str,
    model_path: str,
    lmu_data_root: Path,
    max_new_tokens: int,
    cache_ratio: float | None,
    keep_rate: float | None,
    benchmark: str | None = None,
) -> dict:
    method_spec = MODEL_SPECS[model]["methods"][method]
    config = {
        "class": method_spec["class"],
        "model_path": model_path,
        "torch_dtype": "fp16",
        "max_new_tokens": max_new_tokens,
    }

    if model == "qwen3vlmoe":
        config["top_k"] = 1
        if method in {"skip", "offline", "online", "tablemoe"}:
            config["lmu_data_root"] = str(lmu_data_root)
            if cache_ratio is not None:
                config["cache_ratio"] = float(cache_ratio)
            if keep_rate is not None and method == "tablemoe":
                config["keep_rate"] = float(keep_rate)
            if method in {"offline", "online", "tablemoe"} and benchmark is not None:
                pca_dir, cache_dir = resolve_tablemoe_dirs(model, benchmark)
                ensure_tablemoe_dirs(pca_dir, cache_dir, benchmark)
                config["pca_dir"] = str(pca_dir)
                config["cache_dir"] = str(cache_dir)
    else:
        config["do_sample"] = False
        if method == "transformers":
            config["attn_implementation"] = "flash_attention_2"
            config["device_map"] = "cuda:0"
        else:
            config["lmu_data_root"] = str(lmu_data_root)
            if cache_ratio is not None:
                config["cache_ratio"] = float(cache_ratio)
            if keep_rate is not None and method == "tablemoe":
                config["keep_rate"] = float(keep_rate)
            if method in {"offline", "online", "tablemoe"} and benchmark is not None:
                pca_dir, cache_dir = resolve_tablemoe_dirs(model, benchmark)
                ensure_tablemoe_dirs(pca_dir, cache_dir, benchmark)
                config["pca_dir"] = str(pca_dir)
                config["cache_dir"] = str(cache_dir)

    return config


def write_config(config: dict, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_run_command(
    config_path: Path,
    work_dir: Path,
    run_mode: str,
    reuse: bool,
    verbose: bool,
    judge_model: str | None,
    judge_args: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "run.py",
        "--config",
        str(config_path),
        "--work-dir",
        str(work_dir),
        "--mode",
        run_mode,
    ]
    if reuse:
        cmd.append("--reuse")
    if verbose:
        cmd.append("--verbose")
    if judge_model:
        cmd.extend(["--judge", judge_model])
    if judge_args:
        cmd.extend(["--judge-args", judge_args])
    return cmd


def run_vlmeval(
    cmd: list[str],
    lmu_data_root: Path,
    output_model_name: str,
    benchmarks: list[str],
    extra_info: dict | None = None,
) -> None:
    env = os.environ.copy()
    env["LMUData"] = str(lmu_data_root)

    python_path_parts = [str(REPO_ROOT), str(VLM_EVAL_ROOT)]
    existing_python_path = env.get("PYTHONPATH", "")
    if existing_python_path:
        python_path_parts.append(existing_python_path)
    env["PYTHONPATH"] = os.pathsep.join(python_path_parts)

    print("=" * 80)
    print(f"Model name: {output_model_name}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"LMUData: {lmu_data_root}")
    if extra_info:
        for key, value in extra_info.items():
            print(f"{key}: {value}")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, cwd=VLM_EVAL_ROOT, env=env, check=True)


def find_latest_result_file(model_dir: Path, output_model_name: str, benchmark: str) -> Path | None:
    candidates = []
    for suffix in ("acc.csv", "score.csv"):
        pattern = f"{output_model_name}_{benchmark}*{suffix}"
        candidates.extend(model_dir.rglob(pattern))

    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_overall_score(result_file: Path) -> float | None:
    with result_file.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows or "Overall" not in rows[0]:
        return None

    overall = float(rows[0]["Overall"])
    if overall <= 1.0:
        overall *= 100.0
    return round(overall, 4)


def collect_summary(
    work_dir: Path,
    output_model_name: str,
    model_key: str,
    method: str,
    benchmarks: list[str],
    metric_type: str,
    judge_model: str | None,
) -> tuple[Path, Path]:
    model_dir = work_dir / output_model_name
    summary_json = work_dir / f"{output_model_name}_summary.json"
    summary_md = work_dir / f"{output_model_name}_summary.md"

    results = {}
    missing_files = []

    for benchmark in benchmarks:
        result_file = find_latest_result_file(model_dir, output_model_name, benchmark)
        entry = {
            "score": None,
            "result_file": None,
        }

        if result_file is None:
            missing_files.append(benchmark)
            results[benchmark] = entry
            continue

        entry["result_file"] = str(result_file)
        entry["score"] = parse_overall_score(result_file)
        results[benchmark] = entry

    payload = {
        "model_key": model_key,
        "model": output_model_name,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "metric_type": metric_type,
        "judge_model": judge_model,
        "benchmarks": results,
        "missing_files": missing_files,
    }
    summary_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# {output_model_name} Accuracy Summary",
        "",
        f"Method: {METHOD_LABELS[method]}",
        f"Metric: {metric_type}",
        "",
        "| Dataset | Score (%) | Result File |",
        "| --- | ---: | --- |",
    ]
    for benchmark in benchmarks:
        entry = results[benchmark]
        score_value = entry["score"]
        score_display = "N/A" if score_value is None else f"{score_value:.4f}"
        result_file = entry["result_file"] or "N/A"
        lines.append(f"| {benchmark} | {score_display} | {result_file} |")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("")
    print("[SUMMARY] Accuracy results:")
    for benchmark in benchmarks:
        score_value = results[benchmark]["score"]
        score_display = "N/A" if score_value is None else f"{score_value:.4f}%"
        print(f"  {benchmark}: {score_display}")
    print(f"[SUMMARY] JSON: {summary_json}")
    print(f"[SUMMARY] MD:   {summary_md}")

    return summary_json, summary_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified VLMEvalKit accuracy entrypoint for transformers/skip/offline/online/tablemoe."
    )
    parser.add_argument("--method", required=True, help="transformers, skip, offline, online, tablemoe")
    parser.add_argument(
        "--model",
        required=True,
        help="Qwen3-VL-30B-A3B-Instruct or DeepSeek-VL2",
    )
    parser.add_argument("--model-path", required=True, help="Model path on the target server")
    parser.add_argument("--model-name", help="Override VLMEvalKit output model name")
    parser.add_argument("--datasets", required=True, help="Comma or whitespace separated dataset list")
    parser.add_argument("--work-dir", type=Path, required=True, help="Accuracy output root")
    parser.add_argument("--lmu-data-root", type=Path, required=True, help="LMUData root")
    parser.add_argument("--run-mode", default="all", choices=["all", "infer", "eval"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cache-ratio", type=float, default=None)
    parser.add_argument("--keep-rate", type=float, default=None)
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--judge-args", type=str, default=None)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method = canonicalize_method(args.method)
    model = canonicalize_model(args.model)
    benchmarks = parse_datasets(args.datasets)

    args.work_dir = args.work_dir.resolve()
    args.lmu_data_root = args.lmu_data_root.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if not args.lmu_data_root.is_dir():
        raise FileNotFoundError(f"LMUData root not found: {args.lmu_data_root}")

    output_model_name = args.model_name or MODEL_SPECS[model]["methods"][method]["model_name"]
    generated_config_dir = args.work_dir / "generated_configs"
    metric_type = "judge" if args.judge_model else "exact_match"

    print("==============================================")
    print("Unified VLMEval Accuracy Runner")
    print("==============================================")
    print(f"Method:          {METHOD_LABELS[method]}")
    print(f"Model:           {model}")
    print(f"Model path:      {args.model_path}")
    print(f"Output model:    {output_model_name}")
    print(f"Benchmarks:      {', '.join(benchmarks)}")
    print(f"Work dir:        {args.work_dir}")
    print(f"LMUData root:    {args.lmu_data_root}")
    print(f"Run mode:        {args.run_mode}")
    print(f"Max new tokens:  {args.max_new_tokens}")
    print(f"Cache ratio:     {args.cache_ratio}")
    print(f"Keep rate:       {args.keep_rate}")
    print(f"Judge model:     {args.judge_model or '<none>'}")
    print(f"Reuse:           {args.reuse}")
    print(f"Verbose:         {args.verbose}")

    if method in {"transformers", "skip"}:
        config = {
            "model": {
                output_model_name: build_model_config(
                    model=model,
                    method=method,
                    model_path=args.model_path,
                    lmu_data_root=args.lmu_data_root,
                    max_new_tokens=args.max_new_tokens,
                    cache_ratio=args.cache_ratio,
                    keep_rate=args.keep_rate,
                )
            },
            "data": build_dataset_config(benchmarks),
        }
        config_path = generated_config_dir / f"{output_model_name}_all.json"
        write_config(config, config_path)
        cmd = build_run_command(
            config_path=config_path,
            work_dir=args.work_dir,
            run_mode=args.run_mode,
            reuse=args.reuse,
            verbose=args.verbose,
            judge_model=args.judge_model,
            judge_args=args.judge_args,
        )
        run_vlmeval(
            cmd=cmd,
            lmu_data_root=args.lmu_data_root,
            output_model_name=output_model_name,
            benchmarks=benchmarks,
        )
    else:
        for benchmark in benchmarks:
            pca_dir, cache_dir = resolve_tablemoe_dirs(model, benchmark)
            config = {
                "model": {
                    output_model_name: build_model_config(
                        model=model,
                        method=method,
                        model_path=args.model_path,
                        lmu_data_root=args.lmu_data_root,
                        max_new_tokens=args.max_new_tokens,
                        cache_ratio=args.cache_ratio,
                        keep_rate=args.keep_rate,
                        benchmark=benchmark,
                    )
                },
                "data": build_dataset_config([benchmark]),
            }
            config_path = generated_config_dir / f"{output_model_name}_{benchmark}.json"
            write_config(config, config_path)
            cmd = build_run_command(
                config_path=config_path,
                work_dir=args.work_dir,
                run_mode=args.run_mode,
                reuse=args.reuse,
                verbose=args.verbose,
                judge_model=args.judge_model,
                judge_args=args.judge_args,
            )
            run_vlmeval(
                cmd=cmd,
                lmu_data_root=args.lmu_data_root,
                output_model_name=output_model_name,
                benchmarks=[benchmark],
                extra_info={
                    "PCA dir": pca_dir,
                    "Cache dir": cache_dir,
                },
            )

    summary_json, summary_md = collect_summary(
        work_dir=args.work_dir,
        output_model_name=output_model_name,
        model_key=model,
        method=method,
        benchmarks=benchmarks,
        metric_type=metric_type,
        judge_model=args.judge_model,
    )

    print("")
    print("[INFO] Finished.")
    print(f"[INFO] Model dir:     {args.work_dir / output_model_name}")
    print(f"[INFO] Summary JSON: {summary_json}")
    print(f"[INFO] Summary MD:   {summary_md}")


if __name__ == "__main__":
    main()
