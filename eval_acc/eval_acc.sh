#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_DIR/scripts/asset_helpers.sh"

DEFAULT_DATASETS="MMBench_DEV_EN_V11"
DEFAULT_METHODS="tablemoe"
DEFAULT_MODEL="Qwen3-VL-30B-A3B-Instruct"
DEFAULT_JUDGE_LABEL="Qwen3-VL-30B-A3B-Instruct"

usage() {
  cat <<'EOF'
Usage:
  MODEL_PATH=/path/to/model \
  bash eval_acc/eval_acc.sh

Environment variables:
  MODEL_PATH                Optional for the default Qwen model if it exists
                            under <repo_root>/models/Qwen3-VL-30B-A3B-Instruct
  MODEL                     Default: Qwen3-VL-30B-A3B-Instruct
  METHODS                   Default: tablemoe. Choices: transformers, skip, offline, online, tablemoe
  DATASETS                  Default: MMBench_DEV_EN_V11
  LMUDATA_DIR               Default: <repo_root>/LMUData
  RESULT_ROOT               Default: <repo_root>/acc_results/default
  RESULT_DIR_NAME           Default: default
  RUN_MODE                  Default: all
  MAX_NEW_TOKENS            Default: 128
  CACHE_RATIO               Default: 0.5
  KEEP_RATE                 Default: 0.6
  ENABLE_JUDGE              Default: 0
  AUTO_DOWNLOAD_DATASETS    Default: 1
  HF_AUTO_DOWNLOAD          Default: 1
  HF_OFFLINE_TABLE_REPO     Default: fanyafanya/ALUTs
  REUSE=1
  VERBOSE=1
EOF
}

normalize_token() {
  echo "${1:-}" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]'
}

normalize_model() {
  local raw="${1:-}"
  local token
  token="$(normalize_token "$raw")"
  case "$token" in
    qwen|qwen3vl|qwen3vlmoe|qwen3vl30ba3binstruct|qwenqwen3vl30ba3binstruct)
      echo "qwen3vlmoe"
      ;;
    deepseek|deepseekvl2)
      echo "deepseekvl2"
      ;;
    *)
      echo ""
      ;;
  esac
}

display_model_name() {
  local model_key="${1:-}"
  case "$model_key" in
    qwen3vlmoe)
      echo "Qwen3-VL-30B-A3B-Instruct"
      ;;
    deepseekvl2)
      echo "DeepSeek-VL2"
      ;;
    *)
      echo "$model_key"
      ;;
  esac
}

normalize_method() {
  local raw="${1:-}"
  case "${raw,,}" in
    transformers)
      echo "transformers"
      ;;
    skip)
      echo "skip"
      ;;
    offline)
      echo "offline"
      ;;
    online)
      echo "online"
      ;;
    tablemoe)
      echo "tablemoe"
      ;;
    *)
      echo ""
      ;;
  esac
}

resolve_model() {
  local model_path="${1:-}"
  local config_path="$model_path/config.json"
  local lowered="${model_path,,}"

  if [[ -f "$config_path" ]]; then
    if grep -q '"model_type"[[:space:]]*:[[:space:]]*"qwen3_vl_moe"' "$config_path"; then
      echo "qwen3vlmoe"
      return
    fi
    if grep -q '"model_type"[[:space:]]*:[[:space:]]*"deepseek_vl_v2"' "$config_path"; then
      echo "deepseekvl2"
      return
    fi
  fi

  if [[ "$lowered" == *qwen* ]]; then
    echo "qwen3vlmoe"
    return
  fi
  if [[ "$lowered" == *deepseek* ]]; then
    echo "deepseekvl2"
    return
  fi
}

canonicalize_dataset_name() {
  local raw="$1"
  local normalized="${raw,,}"
  normalized="${normalized//[^a-z0-9]/}"
  case "$normalized" in
    mmbench|mmbenchdevenv11)
      echo "MMBench_DEV_EN_V11"
      ;;
    ai2d|ai2dtest)
      echo "AI2D_TEST"
      ;;
    realworldqa)
      echo "RealWorldQA"
      ;;
    scienceqa|scienceqatest)
      echo "ScienceQA_TEST"
      ;;
    pope)
      echo "POPE"
      ;;
    *)
      echo ""
      ;;
  esac
}

using_default_tablemoe_dirs() {
  [[ -z "${CACHE_ROOT:-}" ]] \
    && [[ -z "${PCA_ROOT:-}" ]] \
    && [[ -z "${MMBENCH_CACHE_DIR:-}" ]] \
    && [[ -z "${AI2D_CACHE_DIR:-}" ]] \
    && [[ -z "${REALWORLDQA_CACHE_DIR:-}" ]] \
    && [[ -z "${SCIENCEQA_CACHE_DIR:-}" ]] \
    && [[ -z "${POPE_CACHE_DIR:-}" ]] \
    && [[ -z "${MMBENCH_PCA_DIR:-}" ]] \
    && [[ -z "${AI2D_PCA_DIR:-}" ]] \
    && [[ -z "${REALWORLDQA_PCA_DIR:-}" ]] \
    && [[ -z "${SCIENCEQA_PCA_DIR:-}" ]] \
    && [[ -z "${POPE_PCA_DIR:-}" ]]
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 && -z "${MODEL_PATH:-}" && "${1:-}" != -* ]]; then
  MODEL_PATH="$1"
  shift
fi

MODEL_PATH="${MODEL_PATH:-}"
MODEL="${MODEL:-}"
METHODS_RAW="${METHODS:-${MODE:-$DEFAULT_METHODS}}"
DATASETS="${DATASETS:-$DEFAULT_DATASETS}"
LMUDATA_DIR="${LMUDATA_DIR:-$PROJECT_DIR/LMUData}"
RESULT_DIR_NAME="${RESULT_DIR_NAME:-default}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/acc_results/$RESULT_DIR_NAME}"
RUN_MODE="${RUN_MODE:-all}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
CACHE_RATIO="${CACHE_RATIO:-0.5}"
KEEP_RATE="${KEEP_RATE:-${RECOMP_RATIO:-0.6}}"
ENABLE_JUDGE="${ENABLE_JUDGE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AUTO_DOWNLOAD_DATASETS="${AUTO_DOWNLOAD_DATASETS:-1}"
HF_AUTO_DOWNLOAD="${HF_AUTO_DOWNLOAD:-1}"
HF_OFFLINE_TABLE_REPO="${HF_OFFLINE_TABLE_REPO:-fanyafanya/ALUTs}"

tablemoe_ensure_eval_acc_requirements "$PROJECT_DIR" "$PYTHON_BIN"

if [[ -z "$MODEL" ]]; then
  if [[ -n "$MODEL_PATH" ]]; then
    MODEL="$(resolve_model "$MODEL_PATH")"
  else
    MODEL="$DEFAULT_MODEL"
  fi
fi
MODEL="${MODEL:-$DEFAULT_MODEL}"
MODEL="$(normalize_model "$MODEL")"
if [[ -z "$MODEL" ]]; then
  echo "[ERROR] Unsupported MODEL. Expected Qwen3-VL-30B-A3B-Instruct or DeepSeek-VL2"
  exit 1
fi

RESOLVED_MODEL_PATH="$(tablemoe_resolve_model_path "$PROJECT_DIR" "$MODEL" "$MODEL_PATH" || true)"
if [[ -z "$RESOLVED_MODEL_PATH" ]]; then
  tablemoe_print_model_path_help "$PROJECT_DIR" "$MODEL"
  exit 1
fi
MODEL_PATH="$RESOLVED_MODEL_PATH"

if [[ "$ENABLE_JUDGE" == "1" && -z "${JUDGE_MODEL:-}" ]]; then
  JUDGE_MODEL="$DEFAULT_JUDGE_LABEL"
fi

IFS=', ' read -r -a RAW_METHOD_ARRAY <<< "$METHODS_RAW"
METHOD_ARRAY=()
for raw_method in "${RAW_METHOD_ARRAY[@]}"; do
  [[ -z "$raw_method" ]] && continue
  method="$(normalize_method "$raw_method")"
  if [[ -z "$method" ]]; then
    echo "[ERROR] Unsupported method: $raw_method"
    exit 1
  fi
  METHOD_ARRAY+=("$method")
done

if [[ "${#METHOD_ARRAY[@]}" -eq 0 ]]; then
  echo "[ERROR] No valid methods selected"
  exit 1
fi

SELECTED_DATASETS=()
IFS=', ' read -r -a RAW_DATASETS <<< "$DATASETS"
for raw_dataset in "${RAW_DATASETS[@]}"; do
  [[ -z "$raw_dataset" ]] && continue
  benchmark="$(canonicalize_dataset_name "$raw_dataset")"
  if [[ -z "$benchmark" ]]; then
    echo "[ERROR] Unsupported dataset: $raw_dataset"
    exit 1
  fi
  SELECTED_DATASETS+=("$benchmark")
done

if [[ "${#SELECTED_DATASETS[@]}" -eq 0 ]]; then
  echo "[ERROR] No valid datasets selected"
  exit 1
fi

tablemoe_ensure_dataset_files "$PROJECT_DIR" "$LMUDATA_DIR" "${SELECTED_DATASETS[@]}"

TABLEMOE_SELECTED=0
for method in "${METHOD_ARRAY[@]}"; do
  if [[ "$method" == "tablemoe" || "$method" == "offline" || "$method" == "online" ]]; then
    TABLEMOE_SELECTED=1
    break
  fi
done

if [[ "$TABLEMOE_SELECTED" == "1" ]] && using_default_tablemoe_dirs; then
  tablemoe_ensure_offline_tables "$PROJECT_DIR" "$MODEL" "${SELECTED_DATASETS[@]}"
fi

mkdir -p "$RESULT_ROOT"

echo "=============================================="
echo "Unified Accuracy Evaluation"
echo "=============================================="
echo "MODEL:                          $(display_model_name "$MODEL")"
echo "MODEL_PATH:                     $MODEL_PATH"
echo "METHODS:                        ${METHOD_ARRAY[*]}"
echo "DATASETS:                       ${SELECTED_DATASETS[*]}"
echo "LMUDATA_DIR:                    $LMUDATA_DIR"
echo "RESULT_ROOT:                    $RESULT_ROOT"
echo "RUN_MODE:                       $RUN_MODE"
echo "MAX_NEW_TOKENS:                 $MAX_NEW_TOKENS"
echo "CACHE_RATIO:                    $CACHE_RATIO"
echo "KEEP_RATE:                      $KEEP_RATE"
echo "ENABLE_JUDGE:                   $ENABLE_JUDGE"
echo "AUTO_DOWNLOAD_DATASETS:         $AUTO_DOWNLOAD_DATASETS"
echo "HF_AUTO_DOWNLOAD:               $HF_AUTO_DOWNLOAD"
echo "HF_OFFLINE_TABLE_REPO:          $HF_OFFLINE_TABLE_REPO"

for method in "${METHOD_ARRAY[@]}"; do
  method_output_dir="$RESULT_ROOT/$method"
  mkdir -p "$method_output_dir"
  cmd=(
    "$PYTHON_BIN"
    "$SCRIPT_DIR/run_vlmeval.py"
    --method "$method"
    --model "$MODEL"
    --model-path "$MODEL_PATH"
    --datasets "$DATASETS"
    --work-dir "$method_output_dir"
    --lmu-data-root "$LMUDATA_DIR"
    --run-mode "$RUN_MODE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --cache-ratio "$CACHE_RATIO"
    --keep-rate "$KEEP_RATE"
  )
  if [[ -n "${MODEL_NAME:-}" ]]; then
    cmd+=(--model-name "$MODEL_NAME")
  fi
  if [[ -n "${JUDGE_MODEL:-}" ]]; then
    cmd+=(--judge-model "$JUDGE_MODEL")
  fi
  if [[ -n "${JUDGE_ARGS:-}" ]]; then
    cmd+=(--judge-args "$JUDGE_ARGS")
  fi
  if [[ "${REUSE:-0}" == "1" ]]; then
    cmd+=(--reuse)
  fi
  if [[ "${VERBOSE:-0}" == "1" ]]; then
    cmd+=(--verbose)
  fi

  echo ""
  echo "[RUN] method=$method"
  "${cmd[@]}"
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_accuracy.py" \
  --result-root "$RESULT_ROOT" \
  --output-dir "$RESULT_ROOT/summary"
