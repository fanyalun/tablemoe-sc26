import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_ALIASES = {
    "qwen": "qwen3vlmoe",
    "qwen3vl": "qwen3vlmoe",
    "qwen3vlmoe": "qwen3vlmoe",
    "qwen3vl30ba3binstruct": "qwen3vlmoe",
    "qwenqwen3vl30ba3binstruct": "qwen3vlmoe",
}

SUPPORTED_MODELS = {
    "qwen3vlmoe": {
        "repo_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "local_subdir": "Qwen3-VL-30B-A3B-Instruct",
    }
}


def normalize_token(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def canonicalize_model(model: str) -> str:
    resolved = MODEL_ALIASES.get(normalize_token(model))
    if resolved is None:
        raise ValueError(
            "Unsupported model download target. "
            "Only Qwen3-VL-30B-A3B-Instruct is supported by this helper."
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the default Qwen model checkpoint for Tablemoe reproduction."
    )
    parser.add_argument(
        "--model",
        default="Qwen3-VL-30B-A3B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory where the model checkpoint will be stored",
    )
    args = parser.parse_args()

    model = canonicalize_model(args.model)
    model_spec = SUPPORTED_MODELS[model]
    target_dir = args.local_dir / model_spec["local_subdir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    print("==============================================")
    print("Model Download")
    print("==============================================")
    print(f"HF repo:   {model_spec['repo_id']}")
    print(f"Local dir: {target_dir.resolve()}")

    snapshot_download(
        repo_id=model_spec["repo_id"],
        local_dir=str(target_dir.resolve()),
    )

    config_path = target_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model download incomplete: {config_path} is missing")

    print(f"[OK] Model ready: {target_dir.resolve()}")


if __name__ == "__main__":
    main()
