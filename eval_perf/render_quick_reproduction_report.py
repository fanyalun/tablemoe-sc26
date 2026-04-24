import argparse
import json
from pathlib import Path


PERF_METHODS = ["adapmoe", "skip", "offline", "online", "tablemoe"]
METHOD_LABELS = {
    "adapmoe": "AdapMoE",
    "skip": "AdapMoE(+gating)",
    "offline": "+ALUT",
    "online": "+WINDOW",
    "tablemoe": "TableMoE",
}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_perf_records(perf_root: Path):
    payload = load_json(perf_root / "method_comparison" / "summary" / "perf_table.json")
    return payload.get("records", [])


def load_simple_accuracy_records(perf_root: Path):
    payload = load_json(perf_root / "method_comparison" / "summary" / "simple_accuracy_table.json")
    return payload.get("records", [])


def load_cache_hit_summary(perf_root: Path):
    payload = load_json(perf_root / "cache_hit" / "decode_cache_hit_summary.json")
    return payload.get("summary_by_method", {})


def find_record(records, method: str, dataset: str):
    matches = [record for record in records if record.get("method") == method and record.get("dataset") == dataset]
    if len(matches) != 1:
        raise ValueError(f"Expected one record for method={method}, dataset={dataset}, got {len(matches)}")
    return matches[0]


def format_metric(value):
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{float(value) * 100.0:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Render the final quick reproduction markdown report.")
    parser.add_argument("--perf-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="MMBench_DEV_EN_V11")
    parser.add_argument("--sample-ratio", type=float, required=True)
    parser.add_argument("--cache-ratio", type=str, default="0.5")
    parser.add_argument("--keep-rate", type=str, default="0.6")
    parser.add_argument("--cache-hit-max-new-tokens", type=str, default="7")
    args = parser.parse_args()

    perf_root = args.perf_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    perf_records = load_perf_records(perf_root)
    simple_accuracy_records = load_simple_accuracy_records(perf_root)
    cache_hit_by_method = load_cache_hit_summary(perf_root)

    report_path = output_dir / "quick_reproduction_report.md"
    manifest_path = output_dir / "quick_reproduction_manifest.json"

    lines = [
        "# Quick Reproduction Report",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- sample_ratio: `{args.sample_ratio}`",
        f"- cache_ratio: `{args.cache_ratio}`",
        f"- TableMoE keep_rate: `{args.keep_rate}`",
        "",
        "## 1. Performance Comparison",
        "",
        "| Method | TTFT | TPOT | Samples |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in PERF_METHODS:
        record = find_record(perf_records, method, args.dataset)
        lines.append(
            f"| {METHOD_LABELS[method]} | {format_metric(record.get('ttft'))} | "
            f"{format_metric(record.get('tpot'))} | {record.get('samples', 'N/A')} |"
        )

    lines.extend(
        [
            "",
            "## 2. Decoding Cache Hit Rate Comparison",
            "",
            "| Method | Hit Rate (%) | Hits | Misses | Total |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in PERF_METHODS:
        summary = cache_hit_by_method.get(method)
        if summary is None:
            raise ValueError(f"Missing cache-hit summary for method={method}")
        lines.append(
            f"| {METHOD_LABELS[method]} | {format_percent(summary.get('hit_rate'))} | "
            f"{summary.get('hits', 'N/A')} | {summary.get('misses', 'N/A')} | "
            f"{summary.get('total', 'N/A')} |"
        )

    lines.extend(
        [
            "",
            "## 3. Lightweight Accuracy Comparison",
            "",
            "| Method | Accuracy | Metric |",
            "| --- | ---: | --- |",
        ]
    )
    for method in PERF_METHODS:
        record = find_record(simple_accuracy_records, method, args.dataset)
        lines.append(
            f"| {METHOD_LABELS[method]} | {format_metric(record.get('simple_accuracy'))} | "
            f"{record.get('simple_metric_name', 'N/A')} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "dataset": args.dataset,
        "sample_ratio": args.sample_ratio,
        "cache_ratio": args.cache_ratio,
        "keep_rate": args.keep_rate,
        "perf_methods": PERF_METHODS,
        "cache_hit_methods": PERF_METHODS,
        "simple_accuracy_methods": PERF_METHODS,
        "perf_root": str(perf_root / "method_comparison"),
        "cache_hit_root": str(perf_root / "cache_hit"),
        "simple_accuracy_root": str(perf_root / "method_comparison"),
        "report_path": str(report_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote report: {report_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
