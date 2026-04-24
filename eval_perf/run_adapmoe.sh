#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=adapmoe exec bash "$SCRIPT_DIR/eval_perf.sh" "$@"
