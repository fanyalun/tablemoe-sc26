#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/hardware_env.sh"
source "$PROJECT_DIR/scripts/asset_helpers.sh"

DEFAULT_DATASETS="MMBench_DEV_EN_V11"
DEFAULT_METHODS="tablemoe"
DEFAULT_MODEL="Qwen3-VL-30B-A3B-Instruct"
DEFAULT_CACHE_RATIO="0.5"
DEFAULT_KEEP_RATE="0.6"
DEFAULT_SAMPLE_RATIO="0.01"

usage() {
  cat <<'EOF'
Usage:
  MODEL_PATH=/path/to/model \
  bash eval_perf/eval_perf.sh

Environment variables:
  MODEL_PATH                Optional for the default Qwen model if it exists
                            under <repo_root>/models/Qwen3-VL-30B-A3B-Instruct
  MODEL                     Default: Qwen3-VL-30B-A3B-Instruct
  METHODS                   Default: tablemoe. Choices: adapmoe, skip, offline, online, tablemoe
  DATASETS                  Default: MMBench_DEV_EN_V11
  LMUDATA_DIR               Default: <repo_root>/LMUData
  RESULT_ROOT               Default: <repo_root>/perf_results/default
  RESULT_DIR_NAME           Default: default
  CACHE_RATIO               Default: 0.5
  KEEP_RATE                 Default: 0.6
  RECOMP_RATIO              Compatibility alias for KEEP_RATE
  SAMPLE_RATIO              Default: 0.01
  WARMUP_SAMPLES            Default: 5
  MAX_NEW_TOKENS            Use the Python entrypoint default if unset
  AUTO_DOWNLOAD_DATASETS    Default: 1
  HF_AUTO_DOWNLOAD          Default: 1
  HF_OFFLINE_TABLE_REPO     Default: fanyafanya/ALUTs

TableMoE path overrides:
  CACHE_ROOT / PCA_ROOT
  MMBENCH_CACHE_DIR / AI2D_CACHE_DIR / REALWORLDQA_CACHE_DIR / SCIENCEQA_CACHE_DIR / POPE_CACHE_DIR
  MMBENCH_PCA_DIR   / AI2D_PCA_DIR   / REALWORLDQA_PCA_DIR   / SCIENCEQA_PCA_DIR   / POPE_PCA_DIR

Optional third-party baseline env overrides:
  MOE_INFINITY_CONDA_ENV / MOE_INFINITY_OFFLOAD_PATH
  PREGATED_MOE_CONDA_ENV / PREGATED_MOE_OFFLOAD_PATH
  If unset, the current python3 is used directly.
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
    adapmoe)
      echo "adapmoe"
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
    pregatedmoe|pregated-moe|pregated_moe)
      echo "pregated-moe"
      ;;
    moeinfinity|moe-infinity|moe_infinity)
      echo "moe-infinity"
      ;;
    *)
      echo ""
      ;;
  esac
}

method_label() {
  local method="$1"
  case "$method" in
    adapmoe)
      echo "AdapMoE"
      ;;
    skip)
      echo "AdapMoE(+gating)"
      ;;
    offline)
      echo "+ALUT"
      ;;
    online)
      echo "+WINDOW"
      ;;
    tablemoe)
      echo "TableMoE"
      ;;
    pregated-moe)
      echo "Pregated-MoE"
      ;;
    moe-infinity)
      echo "MoE-Infinity"
      ;;
    *)
      echo "$method"
      ;;
  esac
}

python_script_for_method() {
  local method="$1"
  case "$method" in
    adapmoe)
      echo "$SCRIPT_DIR/test_adapmoe.py"
      ;;
    skip)
      echo "$SCRIPT_DIR/test_skip.py"
      ;;
    offline)
      echo "$SCRIPT_DIR/test_tablemoe_offline.py"
      ;;
    online)
      echo "$SCRIPT_DIR/test_tablemoe_online.py"
      ;;
    tablemoe)
      echo "$SCRIPT_DIR/test_tablemoe.py"
      ;;
    pregated-moe)
      echo "$SCRIPT_DIR/test_pregated_moe.py"
      ;;
    moe-infinity)
      echo "$SCRIPT_DIR/test_moe_infinity.py"
      ;;
    *)
      return 1
      ;;
  esac
}

infer_model_from_path() {
  local model_path="${1:-}"
  local normalized="${model_path,,}"
  if [[ "$normalized" == *qwen* && "$normalized" != *deepseek* ]]; then
    echo "qwen3vlmoe"
    return
  fi
  if [[ "$normalized" == *deepseek* && "$normalized" != *qwen* ]]; then
    echo "deepseekvl2"
    return
  fi
  echo ""
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

cache_env_name() {
  local benchmark="$1"
  case "$benchmark" in
    MMBench_DEV_EN_V11)
      echo "MMBENCH_CACHE_DIR"
      ;;
    AI2D_TEST)
      echo "AI2D_CACHE_DIR"
      ;;
    RealWorldQA)
      echo "REALWORLDQA_CACHE_DIR"
      ;;
    ScienceQA_TEST)
      echo "SCIENCEQA_CACHE_DIR"
      ;;
    POPE)
      echo "POPE_CACHE_DIR"
      ;;
  esac
}

pca_env_name() {
  local benchmark="$1"
  case "$benchmark" in
    MMBench_DEV_EN_V11)
      echo "MMBENCH_PCA_DIR"
      ;;
    AI2D_TEST)
      echo "AI2D_PCA_DIR"
      ;;
    RealWorldQA)
      echo "REALWORLDQA_PCA_DIR"
      ;;
    ScienceQA_TEST)
      echo "SCIENCEQA_PCA_DIR"
      ;;
    POPE)
      echo "POPE_PCA_DIR"
      ;;
  esac
}

default_cache_base_for_model() {
  local model_key="$1"
  case "$model_key" in
    qwen3vlmoe)
      echo "$PROJECT_DIR/offline_table/qwen_fp16"
      ;;
    deepseekvl2)
      echo "$PROJECT_DIR/offline_table/ds_fp16"
      ;;
    *)
      echo ""
      ;;
  esac
}

resolve_cache_dir() {
  local benchmark="$1"
  local env_name
  env_name="$(cache_env_name "$benchmark")"
  if [[ -n "${!env_name:-}" ]]; then
    echo "${!env_name}"
    return
  fi
  if [[ -n "${CACHE_ROOT:-}" ]]; then
    echo "${CACHE_ROOT}/${benchmark}${CACHE_DIR_SUFFIX:-_LayerPCA_256}"
    return
  fi
  echo ""
}

resolve_pca_dir() {
  local benchmark="$1"
  local env_name
  env_name="$(pca_env_name "$benchmark")"
  if [[ -n "${!env_name:-}" ]]; then
    echo "${!env_name}"
    return
  fi
  if [[ -n "${PCA_ROOT:-}" ]]; then
    echo "${PCA_ROOT}/${benchmark}${PCA_DIR_SUFFIX:-_LayerPCA_256}"
    return
  fi
  echo ""
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

append_optional_arg() {
  local -n target_array="$1"
  local flag="$2"
  local value="${3:-}"
  if [[ -n "$value" ]]; then
    target_array+=("$flag" "$value")
  fi
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
LMUDATA_DIR="${LMUDATA_DIR:-$PROJECT_DIR/LMUData}"
DATASETS="${DATASETS:-$DEFAULT_DATASETS}"
RESULT_DIR_NAME="${RESULT_DIR_NAME:-default}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/perf_results/$RESULT_DIR_NAME}"
MODEL_NAME="${MODEL_NAME:-}"
CACHE_RATIO="${CACHE_RATIO:-$DEFAULT_CACHE_RATIO}"
KEEP_RATE="${KEEP_RATE:-${RECOMP_RATIO:-$DEFAULT_KEEP_RATE}}"
SAMPLE_RATIO="${SAMPLE_RATIO:-$DEFAULT_SAMPLE_RATIO}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-5}"
DEEPSEEK_REPO="${DEEPSEEK_REPO:-$PROJECT_DIR/third_party/DeepSeek-VL2}"
MOE_INFINITY_CONDA_ENV="${MOE_INFINITY_CONDA_ENV:-}"
PREGATED_MOE_CONDA_ENV="${PREGATED_MOE_CONDA_ENV:-}"
AUTO_DOWNLOAD_DATASETS="${AUTO_DOWNLOAD_DATASETS:-1}"
HF_AUTO_DOWNLOAD="${HF_AUTO_DOWNLOAD:-1}"
HF_OFFLINE_TABLE_REPO="${HF_OFFLINE_TABLE_REPO:-fanyafanya/ALUTs}"

if [[ -z "$MODEL" ]]; then
  if [[ -n "$MODEL_PATH" ]]; then
    MODEL="$(infer_model_from_path "$MODEL_PATH")"
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

if [[ -z "$MODEL_NAME" ]]; then
  MODEL_NAME="$(basename "$MODEL_PATH")"
fi

DEFAULT_CACHE_BASE="$(default_cache_base_for_model "$MODEL")"
if [[ -n "$DEFAULT_CACHE_BASE" ]]; then
  : "${CACHE_ROOT:=$DEFAULT_CACHE_BASE/offline_table}"
  : "${PCA_ROOT:=$DEFAULT_CACHE_BASE/clustering_results}"
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

declare -a SELECTED_DATASETS=()
IFS=',' read -r -a RAW_DATASETS <<< "$DATASETS"
for raw_dataset in "${RAW_DATASETS[@]}"; do
  raw_dataset="${raw_dataset// /}"
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

if [[ "$KEEP_RATE" != "$DEFAULT_KEEP_RATE" ]]; then
  for method in "${METHOD_ARRAY[@]}"; do
    if [[ "$method" == "pregated-moe" || "$method" == "moe-infinity" ]]; then
      echo "[ERROR] keep_rate is only used by tablemoe; pregated-moe and moe-infinity do not accept non-default keep rates"
      exit 1
    fi
  done
fi

mkdir -p "$RESULT_ROOT"

echo "=============================================="
echo "Unified Performance Evaluation"
echo "=============================================="
echo "MODEL:                          $(display_model_name "$MODEL")"
echo "MODEL_PATH:                     $MODEL_PATH"
echo "MODEL_NAME:                     $MODEL_NAME"
echo "METHODS:                        ${METHOD_ARRAY[*]}"
echo "DATASETS:                       ${SELECTED_DATASETS[*]}"
echo "LMUDATA_DIR:                    $LMUDATA_DIR"
echo "RESULT_ROOT:                    $RESULT_ROOT"
echo "CACHE_RATIO:                    $CACHE_RATIO"
echo "KEEP_RATE:                      $KEEP_RATE"
echo "SAMPLE_RATIO:                   $SAMPLE_RATIO"
echo "DEEPSEEK_REPO:                  $DEEPSEEK_REPO"
echo "AUTO_DOWNLOAD_DATASETS:         $AUTO_DOWNLOAD_DATASETS"
echo "HF_AUTO_DOWNLOAD:               $HF_AUTO_DOWNLOAD"
echo "HF_OFFLINE_TABLE_REPO:          $HF_OFFLINE_TABLE_REPO"

for method in "${METHOD_ARRAY[@]}"; do
  method_label_value="$(method_label "$method")"
  method_script="$(python_script_for_method "$method")"
  method_result_root="$RESULT_ROOT/$method"
  mkdir -p "$method_result_root"

  dataset_total="${#SELECTED_DATASETS[@]}"
  dataset_index=0
  for benchmark in "${SELECTED_DATASETS[@]}"; do
    dataset_index=$((dataset_index + 1))
    data_file="$LMUDATA_DIR/${benchmark}.tsv"
    output_dir="$method_result_root/${MODEL_NAME}_${benchmark}"

    echo ""
    echo "====================================================================="
    echo "[$dataset_index/$dataset_total] Method: $method_label_value | Benchmark: $benchmark"
    echo "Data file: $data_file"
    echo "Output dir: $output_dir"

    cmd=()
    if [[ "$method" == "moe-infinity" ]]; then
      if [[ -n "$MOE_INFINITY_CONDA_ENV" ]]; then
        cmd+=(conda run --no-capture-output -n "$MOE_INFINITY_CONDA_ENV")
      fi
      cmd+=(python3 -u "$method_script")
    elif [[ "$method" == "pregated-moe" ]]; then
      if [[ -n "$PREGATED_MOE_CONDA_ENV" ]]; then
        cmd+=(conda run --no-capture-output -n "$PREGATED_MOE_CONDA_ENV")
      fi
      cmd+=(python3 -u "$method_script")
    else
      cmd+=(python3 -u "$method_script")
    fi

    cmd+=(
      --model "$MODEL"
      --model-path "$MODEL_PATH"
      --model-name "$MODEL_NAME"
      --data-dir "$data_file"
      --output-dir "$output_dir"
      --deepseek-repo "$DEEPSEEK_REPO"
      --sample-ratio "$SAMPLE_RATIO"
      --warmup-samples "$WARMUP_SAMPLES"
    )

    append_optional_arg cmd --sample-seed "${SAMPLE_SEED:-}"
    append_optional_arg cmd --max-new-tokens "${MAX_NEW_TOKENS:-}"
    append_optional_arg cmd --attn-implementation "${ATTN_IMPLEMENTATION:-}"

    if [[ "$method" == "adapmoe" || "$method" == "skip" || "$method" == "offline" || "$method" == "online" || "$method" == "tablemoe" ]]; then
      cmd+=(--cache-ratio "$CACHE_RATIO")
    fi

    if [[ "$method" == "tablemoe" ]]; then
      cmd+=(--keep-rate "$KEEP_RATE")
    fi

    if [[ "$method" == "tablemoe" || "$method" == "offline" || "$method" == "online" ]]; then
      cache_dir="$(resolve_cache_dir "$benchmark")"
      pca_dir="$(resolve_pca_dir "$benchmark")"
      if [[ -z "$cache_dir" || ! -d "$cache_dir" ]]; then
        echo "[ERROR] cache dir not found for $benchmark: $cache_dir"
        echo "[ERROR] Try downloading the published offline table or build it locally:"
        echo "MODEL=$(tablemoe_display_model_name "$MODEL") DATASETS=\"$benchmark\" bash offline_table/download_offline_table.sh"
        echo "MODEL_PATH=<path-to-model> MODEL_NAME=$(tablemoe_display_model_name "$MODEL") DATASETS=\"$benchmark\" bash offline_table/run_offline_table.sh"
        exit 1
      fi
      if [[ -z "$pca_dir" || ! -d "$pca_dir" ]]; then
        echo "[ERROR] pca dir not found for $benchmark: $pca_dir"
        echo "[ERROR] Try downloading the published offline table or build it locally:"
        echo "MODEL=$(tablemoe_display_model_name "$MODEL") DATASETS=\"$benchmark\" bash offline_table/download_offline_table.sh"
        echo "MODEL_PATH=<path-to-model> MODEL_NAME=$(tablemoe_display_model_name "$MODEL") DATASETS=\"$benchmark\" bash offline_table/run_offline_table.sh"
        exit 1
      fi
      cmd+=(--cache-dir "$cache_dir" --pca-dir "$pca_dir")
    fi

    if [[ "$method" == "moe-infinity" ]]; then
      append_optional_arg cmd --offload-path "${MOE_INFINITY_OFFLOAD_PATH:-}"
      cmd+=(--expert-cache-ratio "$CACHE_RATIO")
      append_optional_arg cmd --device-memory-ratio "${DEVICE_MEMORY_RATIO:-}"
      append_optional_arg cmd --gpu-total-memory-mib "${GPU_TOTAL_MEMORY_MIB:-}"
      append_optional_arg cmd --num-threads "${MOE_INFINITY_NUM_THREADS:-}"
    fi

    if [[ "$method" == "pregated-moe" ]]; then
      append_optional_arg cmd --offload-path "${PREGATED_MOE_OFFLOAD_PATH:-}"
      cmd+=(--expert-cache-ratio "$CACHE_RATIO")
      append_optional_arg cmd --device-memory-ratio "${DEVICE_MEMORY_RATIO:-}"
      append_optional_arg cmd --gpu-total-memory-mib "${GPU_TOTAL_MEMORY_MIB:-}"
      append_optional_arg cmd --num-threads "${PREGATED_MOE_NUM_THREADS:-${MOE_INFINITY_NUM_THREADS:-}}"
    fi

    run_with_hardware_env "${cmd[@]}"
  done
done

python3 "$SCRIPT_DIR/summarize_perf.py" \
  --result-root "$RESULT_ROOT" \
  --output-dir "$RESULT_ROOT/summary"
