#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/asset_helpers.sh"
cd "$SCRIPT_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen3-VL-30B-A3B-Instruct}"
DTYPE="${DTYPE:-fp16}"
DEVICE="${DEVICE:-cuda:0}"
DATASETS="${DATASETS:-MMBench_DEV_EN_V11}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/LMUData}"
SEED="${SEED:-42}"
CLUSTER_SIZES="${CLUSTER_SIZES:-256}"

MAX_VISION_TOKENS="${MAX_VISION_TOKENS:-100000}"
MAX_TEXT_TOKENS="${MAX_TEXT_TOKENS:-40000}"
MIN_VISION_PER_SAMPLE="${MIN_VISION_PER_SAMPLE:-8}"
MAX_VISION_PER_SAMPLE="${MAX_VISION_PER_SAMPLE:-64}"
MAX_TEXT_PER_SAMPLE="${MAX_TEXT_PER_SAMPLE:-128}"
STEP1_SAVE_DTYPE="${STEP1_SAVE_DTYPE:-}"
PER_EXPERT_TOKEN_CAP="${PER_EXPERT_TOKEN_CAP:-0}"
MIN_SAMPLES_PER_EXPERT="${MIN_SAMPLES_PER_EXPERT:-}"
NO_TEST_SPLIT="${NO_TEST_SPLIT:-1}"
REUSE_PCA_FROM_FIRST_CLUSTER="${REUSE_PCA_FROM_FIRST_CLUSTER:-1}"

SKIP_STEP1="${SKIP_STEP1:-0}"
CLEAN_STEP1_AFTER_STEP2="${CLEAN_STEP1_AFTER_STEP2:-1}"
CLEAN_STEP2_CLUSTERS_AFTER_STEP3="${CLEAN_STEP2_CLUSTERS_AFTER_STEP3:-1}"

usage() {
  cat <<'EOF'
Usage:
  MODEL_PATH=/path/to/model \
  bash offline_table/run_offline_table.sh

Environment variables:
  MODEL_PATH                Optional for the default Qwen model if it exists
                            under <repo_root>/models/Qwen3-VL-30B-A3B-Instruct
  MODEL_NAME                Default: Qwen3-VL-30B-A3B-Instruct
  DTYPE                     Default: fp16
  DATASETS                  Default: MMBench_DEV_EN_V11
  DATA_ROOT                 Default: <repo_root>/LMUData
  DEVICE                    Default: cuda:0
  CLUSTER_SIZES             Default: 256
  AUTO_DOWNLOAD_DATASETS    Default: 1
  STEP1_ROOT / STEP2_ROOT / STEP3_ROOT
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

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_NAME="$(normalize_model "$MODEL_NAME")"
if [[ -z "$MODEL_NAME" ]]; then
  echo "[ERROR] Unsupported MODEL_NAME. Expected Qwen3-VL-30B-A3B-Instruct or DeepSeek-VL2"
  exit 1
fi

case "$MODEL_NAME:$DTYPE" in
  qwen3vlmoe:bf16)
    DEFAULT_MIN_TEXT_PER_SAMPLE="8"
    MODEL_ROOT="$SCRIPT_DIR/qwen_bf16"
    ;;
  qwen3vlmoe:fp16)
    DEFAULT_MIN_TEXT_PER_SAMPLE="8"
    MODEL_ROOT="$SCRIPT_DIR/qwen_fp16"
    ;;
  deepseekvl2:bf16)
    DEFAULT_MIN_TEXT_PER_SAMPLE="8"
    MODEL_ROOT="$SCRIPT_DIR/ds_bf16"
    ;;
  deepseekvl2:fp16)
    DEFAULT_MIN_TEXT_PER_SAMPLE="8"
    MODEL_ROOT="$SCRIPT_DIR/ds_fp16"
    ;;
  *)
    echo "[ERROR] Unsupported MODEL_NAME/DTYPE: $MODEL_NAME / $DTYPE"
    exit 1
    ;;
esac

MODEL_PATH="${MODEL_PATH:-}"
MIN_TEXT_PER_SAMPLE="${MIN_TEXT_PER_SAMPLE:-$DEFAULT_MIN_TEXT_PER_SAMPLE}"
STEP1_ROOT="${STEP1_ROOT:-$MODEL_ROOT/profiling_results}"
STEP2_ROOT="${STEP2_ROOT:-$MODEL_ROOT/clustering_results}"
STEP3_ROOT="${STEP3_ROOT:-$MODEL_ROOT/offline_table}"
AUTO_DOWNLOAD_DATASETS="${AUTO_DOWNLOAD_DATASETS:-1}"

RESOLVED_MODEL_PATH="$(tablemoe_resolve_model_path "$REPO_ROOT" "$MODEL_NAME" "$MODEL_PATH" || true)"
if [[ -z "$RESOLVED_MODEL_PATH" ]]; then
  tablemoe_print_model_path_help "$REPO_ROOT" "$MODEL_NAME"
  exit 1
fi
MODEL_PATH="$RESOLVED_MODEL_PATH"

if [[ -z "$STEP1_SAVE_DTYPE" ]]; then
  if [[ "$DTYPE" == "fp16" ]]; then
    STEP1_SAVE_DTYPE="float16"
  else
    STEP1_SAVE_DTYPE="float32"
  fi
fi

echo "[INFO] ========================================="
echo "[INFO] Unified Offline Table Pipeline"
echo "[INFO] ========================================="
echo "[INFO] MODEL_NAME : $(display_model_name "$MODEL_NAME")"
echo "[INFO] MODEL_PATH : $MODEL_PATH"
echo "[INFO] DTYPE      : $DTYPE"
echo "[INFO] STEP1_DTYPE: $STEP1_SAVE_DTYPE"
echo "[INFO] DEVICE     : $DEVICE"
echo "[INFO] DATA_ROOT  : $DATA_ROOT"
echo "[INFO] DATASETS   : $DATASETS"
echo "[INFO] K VALUES   : $CLUSTER_SIZES"
echo "[INFO] NO_TEST_SPLIT : $NO_TEST_SPLIT"
echo "[INFO] EXPERT_CAP : $PER_EXPERT_TOKEN_CAP"
echo "[INFO] STEP1_ROOT : $STEP1_ROOT"
echo "[INFO] STEP2_ROOT : $STEP2_ROOT"
echo "[INFO] STEP3_ROOT : $STEP3_ROOT"

action_check_file() {
  local file_path="$1"
  if [[ ! -f "$file_path" ]]; then
    echo "[ERROR] File not found: $file_path"
    exit 1
  fi
}

action_check_file "$SCRIPT_DIR/step1_profile.py"
action_check_file "$SCRIPT_DIR/step2_pca_cluster.py"
action_check_file "$SCRIPT_DIR/step3_offline_cache_builder.py"

DATASET_ARRAY=()
IFS=', ' read -r -a raw_datasets <<< "$DATASETS"
for raw_dataset in "${raw_datasets[@]}"; do
  [[ -z "$raw_dataset" ]] && continue
  benchmark="$(canonicalize_dataset_name "$raw_dataset")"
  if [[ -z "$benchmark" ]]; then
    echo "[ERROR] Unsupported dataset: $raw_dataset"
    exit 1
  fi
  DATASET_ARRAY+=("$benchmark")
done

if [[ "${#DATASET_ARRAY[@]}" -eq 0 ]]; then
  echo "[ERROR] No valid datasets selected"
  exit 1
fi

tablemoe_ensure_dataset_files "$REPO_ROOT" "$DATA_ROOT" "${DATASET_ARRAY[@]}"

run_one_dataset() {
  local benchmark="$1"
  local data_path="$DATA_ROOT/${benchmark}.tsv"
  local step1_out="$STEP1_ROOT/${benchmark}_modality_split"

  echo
  echo "[INFO] ==============================="
  echo "[INFO] Benchmark: $benchmark"
  echo "[INFO] DATA     : $data_path"
  echo "[INFO] STEP1    : $step1_out"
  echo "[INFO] ==============================="

  action_check_file "$data_path"

  if [[ "$SKIP_STEP1" == "1" ]]; then
    echo "[INFO] Step1 skipped"
    if [[ ! -d "$step1_out" ]]; then
      echo "[ERROR] Missing Step1 directory: $step1_out"
      exit 1
    fi
  else
    local -a step1_cmd=(
      python3 "$SCRIPT_DIR/step1_profile.py"
      --model "$MODEL_NAME"
      --model-path "$MODEL_PATH"
      --data-dir "$data_path"
      --output-dir "$step1_out"
      --dtype "$DTYPE"
      --save-dtype "$STEP1_SAVE_DTYPE"
      --device "$DEVICE"
      --min-vision-per-sample "$MIN_VISION_PER_SAMPLE"
      --min-text-per-sample "$MIN_TEXT_PER_SAMPLE"
      --max-vision-per-sample "$MAX_VISION_PER_SAMPLE"
      --max-text-per-sample "$MAX_TEXT_PER_SAMPLE"
      --max-vision-tokens "$MAX_VISION_TOKENS"
      --max-text-tokens "$MAX_TEXT_TOKENS"
      --per-expert-token-cap "$PER_EXPERT_TOKEN_CAP"
      --seed "$SEED"
    )
    if [[ "$NO_TEST_SPLIT" == "1" ]]; then
      step1_cmd+=(--no-test-split)
    fi
    "${step1_cmd[@]}"
  fi

  local reuse_pca_dir=""
  local cluster_size min_samples step2_out step3_out
  local pca_count cdf_count cluster_count cache_count
  for cluster_size in $CLUSTER_SIZES; do
    step2_out="$STEP2_ROOT/${benchmark}_LayerPCA_${cluster_size}"
    step3_out="$STEP3_ROOT/${benchmark}_LayerPCA_${cluster_size}"
    min_samples="$MIN_SAMPLES_PER_EXPERT"
    if [[ -z "$min_samples" ]]; then
      min_samples="$cluster_size"
    fi

    echo
    echo "[INFO] -------------------------------"
    echo "[INFO] Benchmark : $benchmark"
    echo "[INFO] Cluster K : $cluster_size"
    echo "[INFO] STEP2     : $step2_out"
    echo "[INFO] STEP3     : $step3_out"
    echo "[INFO] -------------------------------"

    local -a step2_cmd=(
      python3 "$SCRIPT_DIR/step2_pca_cluster.py"
      --model "$MODEL_NAME"
      --data-dir "$step1_out"
      --save-dir "$step2_out"
      --dtype "$DTYPE"
      --cluster-size "$cluster_size"
      --min-samples-per-expert "$min_samples"
      --seed "$SEED"
      --device "$DEVICE"
    )
    if [[ "$REUSE_PCA_FROM_FIRST_CLUSTER" == "1" && -n "$reuse_pca_dir" ]]; then
      step2_cmd+=(--reuse-pca-dir "$reuse_pca_dir")
    fi
    "${step2_cmd[@]}"

    python3 "$SCRIPT_DIR/step3_offline_cache_builder.py" \
      --model "$MODEL_NAME" \
      --model-path "$MODEL_PATH" \
      --cluster-dir "$step2_out" \
      --save-dir "$step3_out" \
      --dtype "$DTYPE" \
      --device "$DEVICE"

    if [[ -z "$reuse_pca_dir" ]]; then
      reuse_pca_dir="$step2_out"
    fi

    if [[ "$CLEAN_STEP2_CLUSTERS_AFTER_STEP3" == "1" && -d "$step2_out" ]]; then
      case "$step2_out" in
        "$MODEL_ROOT"/clustering_results/*)
          echo "[INFO] Removing Step2 cluster tensors while keeping PCA and CDF"
          find "$step2_out" -type f -name '*_clusters.pt' -delete || true
          ;;
        *)
          echo "[WARN] Skip Step2 cleanup for unsafe path: $step2_out"
          ;;
      esac
    fi

    pca_count=$(find "$step2_out" -type f -name '*_pca.pt' | wc -l || true)
    cdf_count=$(find "$step2_out" -type f -name '*_cdf.png' | wc -l || true)
    cluster_count=$(find "$step2_out" -type f -name '*_clusters.pt' | wc -l || true)
    cache_count=$(find "$step3_out" -type f -name '*_cache.pt' | wc -l || true)

    echo "[INFO] [$benchmark][K=$cluster_size] PCA files    : $pca_count"
    echo "[INFO] [$benchmark][K=$cluster_size] CDF images   : $cdf_count"
    echo "[INFO] [$benchmark][K=$cluster_size] Cluster files: $cluster_count"
    echo "[INFO] [$benchmark][K=$cluster_size] Cache files  : $cache_count"
  done

  if [[ "$CLEAN_STEP1_AFTER_STEP2" == "1" && -d "$step1_out" ]]; then
    case "$step1_out" in
      "$MODEL_ROOT"/profiling_results/*)
        echo "[INFO] Removing Step1 intermediate directory: $step1_out"
        rm -rf "$step1_out"
        ;;
      *)
        echo "[WARN] Skip Step1 cleanup for unsafe path: $step1_out"
        ;;
    esac
  fi
}

for benchmark in "${DATASET_ARRAY[@]}"; do
  run_one_dataset "$benchmark"
done

echo
echo "[INFO] Offline table pipeline finished for: $DATASETS"
