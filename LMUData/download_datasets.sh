#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash LMUData/download_datasets.sh

Environment variables:
  DATASETS       Default: MMBench_DEV_EN_V11
  LMUDATA_DIR    Default: <repo_root>/LMUData

Supported datasets:
  RealWorldQA
  MMBench_DEV_EN_V11
  AI2D_TEST
  ScienceQA_TEST
  POPE
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATASETS="${DATASETS:-MMBench_DEV_EN_V11}"
LMUDATA_DIR="${LMUDATA_DIR:-$PROJECT_DIR/LMUData}"

python3 "$SCRIPT_DIR/download_datasets.py" \
  --datasets "$DATASETS" \
  --lmu-data-root "$LMUDATA_DIR"
