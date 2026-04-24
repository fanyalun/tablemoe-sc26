#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_model.sh

Environment variables:
  MODEL         Default: Qwen3-VL-30B-A3B-Instruct
  LOCAL_DIR     Default: <repo_root>/models

Supported models:
  Qwen3-VL-30B-A3B-Instruct
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL="${MODEL:-Qwen3-VL-30B-A3B-Instruct}"
LOCAL_DIR="${LOCAL_DIR:-$PROJECT_DIR/models}"

python3 "$SCRIPT_DIR/download_model.py" \
  --model "$MODEL" \
  --local-dir "$LOCAL_DIR"
