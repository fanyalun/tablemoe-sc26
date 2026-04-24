import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = ["transformers", "skip", "offline", "online", "tablemoe"]
DATASET_ORDER = [
    "RealWorldQA",
    "MMBench_DEV_EN_V11",
    "AI2D_TEST",
    "ScienceQA_TEST",
    "POPE",
]


def load_records(result_root: Path) -> list[dict]:
    records = []
    for summary_path in sorted(result_root.rglob("*_summary.json")):
        if summary_path.parent.name == "summary":
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for dataset, entry in payload.get("benchmarks", {}).items():
            records.append(
                {
                    "model_key": payload.get("model_key"),
                    "model": payload.get("model"),
                    "method": payload.get("method"),
                    "method_label": payload.get("method_label"),
                    "dataset": dataset,
                    "metric_type": payload.get("metric_type", "exact_match"),
                    "score": entry.get("score"),
                    "result_file": entry.get("result_file"),
                }
            )
    return records


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_key",
                "model",
                "method",
                "method_label",
                "dataset",
                "metric_type",
                "score",
                "result_file",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def write_json(records: list[dict], path: Path) -> None:
    path.write_text(
        json.dumps({"records": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_markdown(records: list[dict], path: Path) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["model_key"], record["metric_type"])].append(record)

    lines = ["# Accuracy Table", ""]
    for (model_key, metric_type), items in sorted(grouped.items()):
        lines.append(f"## {model_key} / {metric_type}")
        lines.append("")
        lines.append("| Method | " + " | ".join(DATASET_ORDER) + " |")
        lines.append("| --- | " + " | ".join(["---:"] * len(DATASET_ORDER)) + " |")
        lookup = {
            (item["method"], item["dataset"]): item["score"]
            for item in items
        }
        for method in METHOD_ORDER:
            label = next(
                (item["method_label"] for item in items if item["method"] == method),
                method,
            )
            row = [label]
            for dataset in DATASET_ORDER:
                value = lookup.get((method, dataset))
                row.append("N/A" if value is None else f"{float(value):.4f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.result_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(records, args.output_dir / "accuracy_table.json")
    write_csv(records, args.output_dir / "accuracy_table.csv")
    write_markdown(records, args.output_dir / "accuracy_table.md")


if __name__ == "__main__":
    main()
