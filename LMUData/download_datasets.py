import argparse
import hashlib
import shutil
import ssl
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

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

DATASET_SPECS = {
    "MMBench_DEV_EN_V11": {
        "url": "https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN_V11.tsv",
        "md5": "30c05be8f2f347a50be25aa067248184",
    },
    "RealWorldQA": {
        "url": "https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv",
        "md5": "4de008f55dc4fd008ca9e15321dc44b7",
    },
    "AI2D_TEST": {
        "url": "https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv",
        "md5": "0f593e0d1c7df9a3d69bf1f947e71975",
    },
    "ScienceQA_TEST": {
        "url": "https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_TEST.tsv",
        "md5": "e42e9e00f9c59a80d8a5db35bc32b71f",
    },
    "POPE": {
        "url": "https://opencompass.openxlab.space/utils/VLMEval/POPE.tsv",
        "md5": "c12f5acb142f2ef1f85a26ba2fbe41d5",
    },
}


def normalize_token(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def canonicalize_dataset(dataset: str) -> str:
    resolved = DATASET_ALIASES.get(normalize_token(dataset))
    if resolved is None:
        raise ValueError(
            "Unsupported dataset. Expected one of: "
            "MMBench_DEV_EN_V11, RealWorldQA, AI2D_TEST, ScienceQA_TEST, POPE"
        )
    return resolved


def parse_datasets(raw: str) -> list[str]:
    datasets = []
    for token in raw.replace(",", " ").split():
        canonical = canonicalize_dataset(token)
        if canonical not in datasets:
            datasets.append(canonical)
    if not datasets:
        raise ValueError("datasets must not be empty")
    return datasets


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=context) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)


def ensure_dataset(dataset: str, lmu_data_root: Path) -> Path:
    spec = DATASET_SPECS[dataset]
    target = lmu_data_root / f"{dataset}.tsv"

    if target.is_file() and file_md5(target) == spec["md5"]:
        return target

    if target.is_file():
        target.unlink()

    download_file(spec["url"], target)
    current_md5 = file_md5(target)
    if current_md5 != spec["md5"]:
        raise RuntimeError(
            f"MD5 mismatch for {dataset}: expected {spec['md5']}, got {current_md5}"
        )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the TSV files for the paper datasets."
    )
    parser.add_argument(
        "--datasets",
        default="MMBench_DEV_EN_V11",
        help="Comma- or space-separated datasets",
    )
    parser.add_argument(
        "--lmu-data-root",
        type=Path,
        default=REPO_ROOT / "LMUData",
        help="Directory where TSV files will be stored",
    )
    args = parser.parse_args()

    datasets = parse_datasets(args.datasets)
    args.lmu_data_root.mkdir(parents=True, exist_ok=True)

    print("==============================================")
    print("Dataset TSV Download")
    print("==============================================")
    print(f"LMUData root: {args.lmu_data_root.resolve()}")
    print(f"Datasets:     {', '.join(datasets)}")

    for dataset in datasets:
        target = ensure_dataset(dataset, args.lmu_data_root)
        print(f"[OK]   {dataset}: {target}")


if __name__ == "__main__":
    main()
