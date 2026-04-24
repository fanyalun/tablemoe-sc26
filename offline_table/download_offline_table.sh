#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  bash offline_table/download_offline_table.sh

Environment variables:
  MODEL                   Default: Qwen3-VL-30B-A3B-Instruct
  DATASETS                Default: MMBench_DEV_EN_V11
  HF_OFFLINE_TABLE_REPO   Default: fanyafanya/ALUTs (Hugging Face bucket id)
  LOCAL_DIR               Default: <repo_root>
  FORCE_DOWNLOAD          Default: 0. Set to 1 to re-download files even if they already exist locally.
  BUCKET_HF_HUB_VERSION   Default: 1.11.0. Temporary huggingface_hub version used during bucket download.
  RUNTIME_HF_HUB_VERSION  Default: 0.36.2. Version restored after the download completes.
  PYTHON_BIN              Default: python3

Requirements:
  - Python environment with `pip`
  - `7z` or `7zz` in PATH (required for split zip archives)

Behavior:
  - Temporarily upgrades `huggingface_hub` to the bucket-enabled version above.
  - Restores `huggingface_hub` to the runtime version below after the download finishes.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL="${MODEL:-Qwen3-VL-30B-A3B-Instruct}"
DATASETS="${DATASETS:-MMBench_DEV_EN_V11}"
HF_OFFLINE_TABLE_REPO="${HF_OFFLINE_TABLE_REPO:-fanyafanya/ALUTs}"
LOCAL_DIR="${LOCAL_DIR:-$PROJECT_DIR}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
BUCKET_HF_HUB_VERSION="${BUCKET_HF_HUB_VERSION:-1.11.0}"
RUNTIME_HF_HUB_VERSION="${RUNTIME_HF_HUB_VERSION:-0.36.2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN=("$PYTHON_BIN" -m pip)
NEED_RESTORE=0

get_hf_hub_version() {
  "$PYTHON_BIN" - <<'PY'
try:
    import huggingface_hub
except ImportError:
    print("__ABSENT__")
else:
    print(huggingface_hub.__version__)
PY
}

require_7zip() {
  if command -v 7zz >/dev/null 2>&1 || command -v 7z >/dev/null 2>&1; then
    return 0
  fi

  cat >&2 <<'EOF'
Extracting the published offline-table split zip archives requires `7z` or `7zz` in PATH.
Install 7-Zip in the runtime environment before running this downloader, for example:
  apt-get update && apt-get install -y 7zip
If your environment is non-root, prefix the apt-get commands with sudo.
EOF
  return 1
}

restore_hf_hub() {
  if [[ "$NEED_RESTORE" != "1" ]]; then
    return 0
  fi

  echo "[INFO] Restoring huggingface_hub to $RUNTIME_HF_HUB_VERSION"
  "${PIP_BIN[@]}" install "huggingface_hub==$RUNTIME_HF_HUB_VERSION"
}

cleanup() {
  local status=$?
  if ! restore_hf_hub; then
    echo "[WARN] Failed to restore huggingface_hub automatically. Please restore it manually." >&2
    status=1
  fi
  exit "$status"
}

trap cleanup EXIT

require_7zip

CURRENT_HF_HUB_VERSION="$(get_hf_hub_version)"
if [[ "$BUCKET_HF_HUB_VERSION" != "$RUNTIME_HF_HUB_VERSION" ]]; then
  NEED_RESTORE=1
fi

if [[ "$CURRENT_HF_HUB_VERSION" != "$BUCKET_HF_HUB_VERSION" ]]; then
  echo "[INFO] Switching huggingface_hub from $CURRENT_HF_HUB_VERSION to $BUCKET_HF_HUB_VERSION for bucket download"
  "${PIP_BIN[@]}" install "huggingface_hub==$BUCKET_HF_HUB_VERSION"
fi

cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/download_offline_table.py"
  --model "$MODEL"
  --datasets "$DATASETS"
  --repo-id "$HF_OFFLINE_TABLE_REPO"
  --local-dir "$LOCAL_DIR"
)

if [[ "$FORCE_DOWNLOAD" == "1" ]]; then
  cmd+=(--force-download)
fi

"${cmd[@]}"
