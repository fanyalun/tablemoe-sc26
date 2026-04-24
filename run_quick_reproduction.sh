#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
source "$PROJECT_DIR/scripts/asset_helpers.sh"

DEFAULT_MODEL="Qwen3-VL-30B-A3B-Instruct"
DEFAULT_MODEL_KEY="qwen3vlmoe"
DEFAULT_DATASET="MMBench_DEV_EN_V11"
DEFAULT_RESULT_DIR_NAME="quick_reproduction"
DEFAULT_SAMPLE_RATIO="0.01"
DEFAULT_CACHE_RATIO="0.5"
DEFAULT_ACC_CACHE_RATIO="1.0"
DEFAULT_KEEP_RATE="0.6"
DEFAULT_CACHE_HIT_MAX_NEW_TOKENS="7"
DEFAULT_PERF_METHODS="adapmoe,skip,offline,online,tablemoe"
DEFAULT_CACHE_HIT_METHODS="$DEFAULT_PERF_METHODS"
DEFAULT_ACC_METHODS="skip,offline,online,tablemoe,transformers"
DEFAULT_STEPS="perf,cache_hit,report"
DEFAULT_WARMUP_SAMPLES="5"

usage() {
  cat <<'USAGE'
Usage:
  CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path/to/model bash run_quick_reproduction.sh

Environment variables:
  MODEL_PATH                 Optional if the default Qwen model exists under <repo_root>/models.
  LMUDATA_DIR                Default: <repo_root>/LMUData
  RESULT_DIR_NAME            Default: quick_reproduction
  SAMPLE_RATIO               Default: 0.01
  SAMPLE_SEED                Default: 42
  CACHE_RATIO                Default: 0.5 for eval_perf and cache_hit
  ACC_CACHE_RATIO            Default: 1.0 when STEPS includes acc
  KEEP_RATE                  Default: 0.6, used by TableMoE only
  CACHE_HIT_MAX_NEW_TOKENS   Default: 7
  WARMUP_SAMPLES             Default: 5
  STEPS                      Default: perf,cache_hit,report. Optional subset of perf, cache_hit, acc, report.
USAGE
}

step_enabled() {
  local step="$1"
  if [[ "${STEPS,,}" == "all" ]]; then
    return 0
  fi

  local token=""
  IFS=', ' read -r -a tokens <<< "$STEPS"
  for token in "${tokens[@]}"; do
    [[ "${token,,}" == "$step" ]] && return 0
  done
  return 1
}

resolve_default_cache_dir() {
  tablemoe_resolve_cache_dir "$DEFAULT_DATASET"
}

resolve_default_pca_dir() {
  tablemoe_resolve_pca_dir "$DEFAULT_DATASET"
}

run_perf_case() {
  local result_root="$1"

  MODEL="$DEFAULT_MODEL" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" \
  DATASETS="$DEFAULT_DATASET" \
  METHODS="$PERF_METHODS" \
  LMUDATA_DIR="$LMUDATA_DIR" \
  RESULT_ROOT="$result_root" \
  CACHE_RATIO="$CACHE_RATIO" \
  KEEP_RATE="$KEEP_RATE" \
  SAMPLE_RATIO="$SAMPLE_RATIO" \
  SAMPLE_SEED="$SAMPLE_SEED" \
  WARMUP_SAMPLES="$WARMUP_SAMPLES" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
  AUTO_DOWNLOAD_DATASETS="$AUTO_DOWNLOAD_DATASETS" \
  HF_AUTO_DOWNLOAD="$HF_AUTO_DOWNLOAD" \
  HF_OFFLINE_TABLE_REPO="$HF_OFFLINE_TABLE_REPO" \
  bash "$PROJECT_DIR/eval_perf/eval_perf.sh"
}

run_acc_case() {
  local result_root="$1"

  MODEL="$DEFAULT_MODEL" \
  MODEL_PATH="$MODEL_PATH" \
  DATASETS="$DEFAULT_DATASET" \
  METHODS="$ACC_METHODS" \
  LMUDATA_DIR="$LMUDATA_DIR" \
  RESULT_ROOT="$result_root" \
  CACHE_RATIO="$ACC_CACHE_RATIO" \
  KEEP_RATE="$KEEP_RATE" \
  AUTO_DOWNLOAD_DATASETS="$AUTO_DOWNLOAD_DATASETS" \
  HF_AUTO_DOWNLOAD="$HF_AUTO_DOWNLOAD" \
  HF_OFFLINE_TABLE_REPO="$HF_OFFLINE_TABLE_REPO" \
  bash "$PROJECT_DIR/eval_acc/eval_acc.sh"
}

run_cache_hit_case() {
  local result_root="$1"
  local cmd=(
    "$PYTHON_BIN"
    "$PROJECT_DIR/eval_perf/collect_tablemoe_cache_hit.py"
    --model "$DEFAULT_MODEL_KEY"
    --model-path "$MODEL_PATH"
    --model-name "$MODEL_NAME"
    --data-dir "$DATA_FILE"
    --output-dir "$result_root"
    --cache-dir "$CACHE_DIR"
    --pca-dir "$PCA_DIR"
    --methods "$CACHE_HIT_METHODS"
    --cache-ratio "$CACHE_RATIO"
    --keep-rate "$KEEP_RATE"
    --sample-ratio "$SAMPLE_RATIO"
    --sample-seed "$SAMPLE_SEED"
    --warmup-samples "$WARMUP_SAMPLES"
    --max-new-tokens "$CACHE_HIT_MAX_NEW_TOKENS"
  )
  if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
    cmd+=(--attn-implementation "$ATTN_IMPLEMENTATION")
  fi
  "${cmd[@]}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_PATH="${MODEL_PATH:-}"
LMUDATA_DIR="${LMUDATA_DIR:-$PROJECT_DIR/LMUData}"
RESULT_DIR_NAME="${RESULT_DIR_NAME:-$DEFAULT_RESULT_DIR_NAME}"
SAMPLE_RATIO="${SAMPLE_RATIO:-$DEFAULT_SAMPLE_RATIO}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
CACHE_RATIO="${CACHE_RATIO:-$DEFAULT_CACHE_RATIO}"
ACC_CACHE_RATIO="${ACC_CACHE_RATIO:-$DEFAULT_ACC_CACHE_RATIO}"
KEEP_RATE="${KEEP_RATE:-$DEFAULT_KEEP_RATE}"
CACHE_HIT_MAX_NEW_TOKENS="${CACHE_HIT_MAX_NEW_TOKENS:-$DEFAULT_CACHE_HIT_MAX_NEW_TOKENS}"
PERF_METHODS="${PERF_METHODS:-$DEFAULT_PERF_METHODS}"
CACHE_HIT_METHODS="${CACHE_HIT_METHODS:-$DEFAULT_CACHE_HIT_METHODS}"
ACC_METHODS="${ACC_METHODS:-$DEFAULT_ACC_METHODS}"
STEPS="${STEPS:-$DEFAULT_STEPS}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-$DEFAULT_WARMUP_SAMPLES}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-}"
AUTO_DOWNLOAD_DATASETS="${AUTO_DOWNLOAD_DATASETS:-1}"
HF_AUTO_DOWNLOAD="${HF_AUTO_DOWNLOAD:-1}"
HF_OFFLINE_TABLE_REPO="${HF_OFFLINE_TABLE_REPO:-fanyafanya/ALUTs}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"

RESOLVED_MODEL_PATH="$(tablemoe_resolve_model_path "$PROJECT_DIR" "$DEFAULT_MODEL_KEY" "$MODEL_PATH" || true)"
if [[ -z "$RESOLVED_MODEL_PATH" ]]; then
  tablemoe_print_model_path_help "$PROJECT_DIR" "$DEFAULT_MODEL_KEY"
  exit 1
fi
MODEL_PATH="$RESOLVED_MODEL_PATH"
if [[ -z "$MODEL_NAME" ]]; then
  MODEL_NAME="$(basename "$MODEL_PATH")"
fi

DEFAULT_CACHE_BASE="$(tablemoe_default_cache_base_for_model "$PROJECT_DIR" "$DEFAULT_MODEL_KEY")"
if [[ -n "$DEFAULT_CACHE_BASE" ]]; then
  : "${CACHE_ROOT:=$DEFAULT_CACHE_BASE/offline_table}"
  : "${PCA_ROOT:=$DEFAULT_CACHE_BASE/clustering_results}"
fi

PERF_ROOT="$PROJECT_DIR/perf_results/$RESULT_DIR_NAME"
ACC_ROOT="$PROJECT_DIR/acc_results/$RESULT_DIR_NAME"
SUMMARY_ROOT="$PERF_ROOT/summary"
DATA_FILE="$LMUDATA_DIR/${DEFAULT_DATASET}.tsv"
CACHE_DIR="$(resolve_default_cache_dir)"
PCA_DIR="$(resolve_default_pca_dir)"

mkdir -p "$PERF_ROOT" "$SUMMARY_ROOT"

tablemoe_ensure_dataset_files "$PROJECT_DIR" "$LMUDATA_DIR" "$DEFAULT_DATASET"
if [[ ! -d "$CACHE_DIR" || ! -d "$PCA_DIR" ]]; then
  tablemoe_ensure_offline_tables "$PROJECT_DIR" "$DEFAULT_MODEL_KEY" "$DEFAULT_DATASET"
  CACHE_DIR="$(resolve_default_cache_dir)"
  PCA_DIR="$(resolve_default_pca_dir)"
fi

echo "=============================================="
echo "Quick Reproduction"
echo "=============================================="
echo "MODEL:                          $DEFAULT_MODEL"
echo "MODEL_PATH:                     $MODEL_PATH"
echo "DATASET:                        $DEFAULT_DATASET"
echo "LMUDATA_DIR:                    $LMUDATA_DIR"
echo "PERF_ROOT:                      $PERF_ROOT"
echo "ACC_ROOT (optional):            $ACC_ROOT"
echo "SAMPLE_RATIO:                   $SAMPLE_RATIO"
echo "SAMPLE_SEED:                    $SAMPLE_SEED"
echo "CACHE_RATIO (perf/cache_hit):   $CACHE_RATIO"
echo "ACC_CACHE_RATIO:                $ACC_CACHE_RATIO"
echo "KEEP_RATE:                      $KEEP_RATE"
echo "CACHE_HIT_MAX_NEW_TOKENS:       $CACHE_HIT_MAX_NEW_TOKENS"
echo "PERF_METHODS:                   $PERF_METHODS"
echo "CACHE_HIT_METHODS:              $CACHE_HIT_METHODS"
echo "ACC_METHODS:                    $ACC_METHODS"
echo "STEPS:                          $STEPS"

if step_enabled perf; then
  echo ""
  echo "[RUN] eval_perf method comparison | methods=$PERF_METHODS"
  run_perf_case "$PERF_ROOT/method_comparison"
fi

if step_enabled cache_hit; then
  echo ""
  echo "[RUN] decode cache-hit comparison | methods=$CACHE_HIT_METHODS | max_new_tokens=$CACHE_HIT_MAX_NEW_TOKENS"
  run_cache_hit_case "$PERF_ROOT/cache_hit"
fi

if step_enabled acc; then
  echo ""
  echo "[RUN] eval_acc method comparison | methods=$ACC_METHODS | cache_ratio=$ACC_CACHE_RATIO"
  mkdir -p "$ACC_ROOT"
  run_acc_case "$ACC_ROOT/method_comparison"
fi

if step_enabled report; then
  "$PYTHON_BIN" "$PROJECT_DIR/eval_perf/render_quick_reproduction_report.py" \
    --perf-root "$PERF_ROOT" \
    --output-dir "$SUMMARY_ROOT" \
    --dataset "$DEFAULT_DATASET" \
    --sample-ratio "$SAMPLE_RATIO" \
    --cache-ratio "$CACHE_RATIO" \
    --keep-rate "$KEEP_RATE" \
    --cache-hit-max-new-tokens "$CACHE_HIT_MAX_NEW_TOKENS"

  echo ""
  echo "[DONE] Quick reproduction report: $SUMMARY_ROOT/quick_reproduction_report.md"
fi
