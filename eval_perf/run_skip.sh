#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHODS=skip exec bash "$SCRIPT_DIR/eval_perf.sh" "$@"
