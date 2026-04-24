#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=skip exec bash "$SCRIPT_DIR/eval_acc.sh" "$@"
