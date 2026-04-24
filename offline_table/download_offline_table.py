import argparse
import re
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_ALIASES = {
    "qwen": "qwen3vlmoe",
    "qwen3vl": "qwen3vlmoe",
    "qwen3vlmoe": "qwen3vlmoe",
    "qwen3vl30ba3binstruct": "qwen3vlmoe",
    "qwenqwen3vl30ba3binstruct": "qwen3vlmoe",
    "deepseek": "deepseekvl2",
    "deepseekvl2": "deepseekvl2",
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

SUPPORTED_OFFLINE_TABLES = {
    "qwen3vlmoe": {
        "offline_root": "qwen_fp16",
        "datasets": {
            "RealWorldQA",
            "MMBench_DEV_EN_V11",
            "AI2D_TEST",
        },
    },
}

SPLIT_ZIP_RE = re.compile(r"\.z\d{2}$", re.IGNORECASE)


def normalize_token(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def canonicalize_model(model: str) -> str:
    resolved = MODEL_ALIASES.get(normalize_token(model))
    if resolved is None:
        raise ValueError(
            "Unsupported model. Expected one of: Qwen3-VL-30B-A3B-Instruct, DeepSeek-VL2"
        )
    return resolved


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


def ensure_supported(model: str, datasets: list[str]) -> tuple[str, list[str]]:
    model_spec = SUPPORTED_OFFLINE_TABLES.get(model)
    if model_spec is None:
        raise ValueError(
            "Prebuilt offline tables are currently published only for "
            "Qwen3-VL-30B-A3B-Instruct."
        )

    unsupported = [dataset for dataset in datasets if dataset not in model_spec["datasets"]]
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(
            f"Prebuilt offline tables are not published for: {joined}. "
            "Please build them locally with offline_table/run_offline_table.sh."
        )
    return model_spec["offline_root"], datasets


def require_hf_bucket_api() -> tuple[object, object]:
    try:
        from huggingface_hub import download_bucket_files, list_bucket_tree
    except ImportError as err:
        raise RuntimeError(
            "download_offline_table.py requires a recent huggingface_hub with bucket support. "
            "Please temporarily upgrade huggingface_hub before running this downloader."
        ) from err
    return download_bucket_files, list_bucket_tree


def list_bucket_paths(bucket_id: str) -> list[str]:
    _, list_bucket_tree = require_hf_bucket_api()
    return [
        item.path
        for item in list_bucket_tree(bucket_id, recursive=True)
        if getattr(item, "type", None) == "file"
    ]


def collect_remote_files(bucket_paths: list[str], offline_root: str, dataset: str) -> list[str]:
    suffix = f"{dataset}_LayerPCA_256"
    clustering_prefix = f"offline_table/{offline_root}/clustering_results/{suffix}/"
    archive_prefix = f"offline_table/{offline_root}/offline_table/{suffix}"

    clustering_files = sorted(path for path in bucket_paths if path.startswith(clustering_prefix))
    archive_files = sorted(
        path
        for path in bucket_paths
        if path == f"{archive_prefix}.zip"
        or SPLIT_ZIP_RE.search(path)
        and path.startswith(archive_prefix)
    )

    if not clustering_files:
        raise FileNotFoundError(
            f"No clustering_results files found in bucket for dataset {dataset}."
        )
    if not any(path.endswith(".zip") for path in archive_files):
        raise FileNotFoundError(
            f"No split zip root file found in bucket for dataset {dataset}."
        )

    return clustering_files + archive_files


def download_files(bucket_id: str, remote_paths: list[str], local_dir: Path) -> None:
    download_bucket_files, _ = require_hf_bucket_api()
    download_pairs = []
    for remote_path in remote_paths:
        local_path = local_dir / remote_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        download_pairs.append((remote_path, str(local_path)))
    download_bucket_files(bucket_id, files=download_pairs)


def require_7zip() -> str:
    for candidate in ("7zz", "7z"):
        tool = shutil.which(candidate)
        if tool is not None:
            return tool

    raise RuntimeError(
        "Extracting the published offline-table split zip archives requires `7z` or `7zz` in PATH. "
        "Please install 7-Zip in the runtime environment and retry."
    )


def dataset_paths(local_dir: Path, offline_root: str, dataset: str) -> tuple[str, Path, Path, Path, Path]:
    suffix = f"{dataset}_LayerPCA_256"
    offline_dir = local_dir / "offline_table" / offline_root
    pca_dir = offline_dir / "clustering_results" / suffix
    cache_root = offline_dir / "offline_table"
    cache_dir = cache_root / suffix
    zip_path = cache_root / f"{suffix}.zip"
    return suffix, pca_dir, cache_root, cache_dir, zip_path


def extract_dataset_cache(local_dir: Path, offline_root: str, dataset: str) -> tuple[Path, Path]:
    _, pca_dir, cache_root, cache_dir, zip_path = dataset_paths(local_dir, offline_root, dataset)

    if not pca_dir.is_dir():
        raise FileNotFoundError(f"Missing clustering_results directory after download: {pca_dir}")

    if cache_dir.is_dir():
        return pca_dir, cache_dir

    if not zip_path.is_file() or zip_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing split zip root file after download: {zip_path}")

    extractor = require_7zip()
    subprocess.run([extractor, "x", "-y", str(zip_path), f"-o{cache_root}"], check=True)

    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Offline table cache restore incomplete for {dataset}: {cache_dir}"
        )
    return pca_dir, cache_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download published TableMoE offline tables from a Hugging Face bucket."
    )
    parser.add_argument("--model", default="Qwen3-VL-30B-A3B-Instruct", help="Model name")
    parser.add_argument(
        "--datasets",
        default="MMBench_DEV_EN_V11",
        help="Comma- or space-separated datasets",
    )
    parser.add_argument("--repo-id", default="fanyafanya/ALUTs", help="Hugging Face bucket id")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=REPO_ROOT,
        help="Repository root where offline_table/ will be restored",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download files even if the offline table has already been restored locally",
    )
    args = parser.parse_args()

    model = canonicalize_model(args.model)
    datasets = parse_datasets(args.datasets)
    offline_root, datasets = ensure_supported(model, datasets)
    local_dir = args.local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    require_7zip()

    print("==============================================")
    print("Offline Table Download")
    print("==============================================")
    print(f"HF bucket: {args.repo_id}")
    print(f"Model:     {model}")
    print(f"Datasets:  {', '.join(datasets)}")
    print(f"Local dir: {local_dir}")

    all_ready = True
    for dataset in datasets:
        _, pca_dir, _, cache_dir, _ = dataset_paths(local_dir, offline_root, dataset)
        if not (pca_dir.is_dir() and cache_dir.is_dir()) or args.force_download:
            all_ready = False
            break

    if all_ready:
        for dataset in datasets:
            _, pca_dir, _, cache_dir, _ = dataset_paths(local_dir, offline_root, dataset)
            print(f"[SKIP] {dataset}: already restored at {pca_dir} | {cache_dir}")
        return

    bucket_paths = list_bucket_paths(args.repo_id)

    for dataset in datasets:
        _, pca_dir, _, cache_dir, _ = dataset_paths(local_dir, offline_root, dataset)
        if pca_dir.is_dir() and cache_dir.is_dir() and not args.force_download:
            print(f"[SKIP] {dataset}: already restored at {pca_dir} | {cache_dir}")
            continue

        remote_paths = collect_remote_files(bucket_paths, offline_root, dataset)
        print(f"[INFO] {dataset}: downloading {len(remote_paths)} files")
        download_files(args.repo_id, remote_paths, local_dir)
        pca_dir, cache_dir = extract_dataset_cache(local_dir, offline_root, dataset)
        print(f"[OK] {dataset}: {pca_dir} | {cache_dir}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as err:
        raise SystemExit(str(err)) from err
