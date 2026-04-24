import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = ["adapmoe", "skip", "offline", "online", "tablemoe", "pregated-moe", "moe-infinity"]
METHOD_LABELS = {
    "adapmoe": "AdapMoE",
    "skip": "AdapMoE(+gating)",
    "offline": "+ALUT",
    "online": "+WINDOW",
    "pregated-moe": "Pregated-MoE",
    "moe-infinity": "MoE-Infinity",
    "tablemoe": "TableMoE",
}
DATASET_ORDER = [
    "RealWorldQA",
    "MMBench_DEV_EN_V11",
    "AI2D_TEST",
    "ScienceQA_TEST",
    "POPE",
]


def _load_metrics(metrics_path: Path) -> tuple[float | None, float | None, int]:
    if not metrics_path.exists():
        return None, None, 0

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        return None, None, 0

    ttfts = [float(item["ttft"]) for item in payload if "ttft" in item]
    tpots = [float(item["tpot"]) for item in payload if "tpot" in item]
    samples = len(payload)
    avg_ttft = None if not ttfts else sum(ttfts) / len(ttfts)
    avg_tpot = None if not tpots else sum(tpots) / len(tpots)
    return avg_ttft, avg_tpot, samples


def _load_accuracy(accuracy_path: Path) -> tuple[float | None, str | None, float | None]:
    if not accuracy_path.exists():
        return None, None, None

    payload = json.loads(accuracy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None, None, None

    accuracy = payload.get("accuracy")
    sample_accuracy = payload.get("sample_accuracy")
    metric_name = "accuracy"
    if sample_accuracy is not None:
        metric_name = "circular_accuracy"

    if accuracy is not None:
        accuracy = float(accuracy)
    if sample_accuracy is not None:
        sample_accuracy = float(sample_accuracy)
    return accuracy, metric_name, sample_accuracy


def _resolve_method_dir(method_dir: Path) -> str:
    return method_dir.name


def _resolve_method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _load_record(method_dir: Path, run_dir: Path) -> dict | None:
    run_config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "metrics.json"
    accuracy_path = run_dir / "accuracy.json"
    if not run_config_path.exists():
        return None

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    avg_ttft, avg_tpot, samples = _load_metrics(metrics_path)
    accuracy, simple_metric_name, sample_accuracy = _load_accuracy(accuracy_path)
    args = run_config.get("arguments", {})
    method = _resolve_method_dir(method_dir)
    cache_ratio = args.get("cache_ratio")
    keep_rate = args.get("keep_rate")
    if keep_rate is None:
        keep_rate = args.get("recomp_ratio")
    if method in {"pregated-moe", "moe-infinity"}:
        cache_ratio = args.get("expert_cache_ratio", cache_ratio)
        keep_rate = "n/a"

    return {
        "model": run_config.get("model"),
        "model_name": run_config.get("model_name"),
        "method": method,
        "method_label": _resolve_method_label(method),
        "dataset": run_config.get("benchmark"),
        "cache_ratio": cache_ratio,
        "keep_rate": keep_rate,
        "ttft": avg_ttft,
        "tpot": avg_tpot,
        "samples": samples,
        "sample_ratio": args.get("sample_ratio"),
        "metrics_file": str(metrics_path) if metrics_path.exists() else None,
        "simple_accuracy": accuracy,
        "simple_metric_name": simple_metric_name,
        "simple_sample_accuracy": sample_accuracy,
        "accuracy_file": str(accuracy_path) if accuracy_path.exists() else None,
    }


def load_records(result_root: Path) -> list[dict]:
    records = []
    for method_dir in sorted(result_root.iterdir()):
        if not method_dir.is_dir() or method_dir.name == "summary":
            continue
        for run_dir in sorted(method_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            record = _load_record(method_dir, run_dir)
            if record is not None:
                records.append(record)
    return records


def write_json(records: list[dict], output_path: Path):
    output_path.write_text(
        json.dumps({"records": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(records: list[dict], output_path: Path):
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "model_name",
                "method",
                "method_label",
                "dataset",
                "cache_ratio",
                "keep_rate",
                "ttft",
                "tpot",
                "samples",
                "sample_ratio",
                "metrics_file",
                "simple_accuracy",
                "simple_metric_name",
                "simple_sample_accuracy",
                "accuracy_file",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def write_markdown(records: list[dict], output_path: Path):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["model"]].append(record)

    lines = ["# Performance Table", ""]
    for model_key, items in sorted(grouped.items()):
        lines.append(f"## {model_key}")
        lines.append("")
        lines.append("| Method | " + " | ".join(DATASET_ORDER) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(DATASET_ORDER)) + " |")
        lookup = {(item["method"], item["dataset"]): item for item in items}
        for method in METHOD_ORDER:
            row = [_resolve_method_label(method)]
            for dataset in DATASET_ORDER:
                item = lookup.get((method, dataset))
                if item is None or item["ttft"] is None or item["tpot"] is None:
                    row.append("N/A")
                    continue
                row.append(f"{float(item['ttft']):.4f} / {float(item['tpot']):.4f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_simple_accuracy_json(records: list[dict], output_path: Path):
    accuracy_records = []
    for record in records:
        accuracy_records.append(
            {
                "model": record["model"],
                "model_name": record["model_name"],
                "method": record["method"],
                "method_label": record["method_label"],
                "dataset": record["dataset"],
                "simple_accuracy": record["simple_accuracy"],
                "simple_metric_name": record["simple_metric_name"],
                "simple_sample_accuracy": record["simple_sample_accuracy"],
                "accuracy_file": record["accuracy_file"],
            }
        )
    output_path.write_text(
        json.dumps({"records": accuracy_records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_simple_accuracy_csv(records: list[dict], output_path: Path):
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "model_name",
                "method",
                "method_label",
                "dataset",
                "simple_accuracy",
                "simple_metric_name",
                "simple_sample_accuracy",
                "accuracy_file",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "model": record["model"],
                    "model_name": record["model_name"],
                    "method": record["method"],
                    "method_label": record["method_label"],
                    "dataset": record["dataset"],
                    "simple_accuracy": record["simple_accuracy"],
                    "simple_metric_name": record["simple_metric_name"],
                    "simple_sample_accuracy": record["simple_sample_accuracy"],
                    "accuracy_file": record["accuracy_file"],
                }
            )


def write_simple_accuracy_markdown(records: list[dict], output_path: Path):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["model"]].append(record)

    lines = ["# Simple Accuracy Table", ""]
    lines.append(
        "This table is generated from the lightweight accuracy checks already executed in `eval_perf`."
    )
    lines.append(
        "Use `eval_acc` with VLMEvalKit for the full paper-grade accuracy reproduction."
    )
    lines.append("")

    for model_key, items in sorted(grouped.items()):
        lines.append(f"## {model_key}")
        lines.append("")
        lines.append("| Method | " + " | ".join(DATASET_ORDER) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(DATASET_ORDER)) + " |")
        lookup = {(item["method"], item["dataset"]): item for item in items}
        for method in METHOD_ORDER:
            row = [_resolve_method_label(method)]
            for dataset in DATASET_ORDER:
                item = lookup.get((method, dataset))
                if item is None or item["simple_accuracy"] is None:
                    row.append("N/A")
                    continue

                metric_name = item.get("simple_metric_name") or "accuracy"
                if metric_name == "circular_accuracy" and item.get("simple_sample_accuracy") is not None:
                    row.append(
                        f"{float(item['simple_accuracy']):.4f}"
                        f" (sample {float(item['simple_sample_accuracy']):.4f})"
                    )
                else:
                    row.append(f"{float(item['simple_accuracy']):.4f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.result_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(records, args.output_dir / "perf_table.json")
    write_csv(records, args.output_dir / "perf_table.csv")
    write_markdown(records, args.output_dir / "perf_table.md")
    write_simple_accuracy_json(records, args.output_dir / "simple_accuracy_table.json")
    write_simple_accuracy_csv(records, args.output_dir / "simple_accuracy_table.csv")
    write_simple_accuracy_markdown(records, args.output_dir / "simple_accuracy_table.md")


if __name__ == "__main__":
    main()
