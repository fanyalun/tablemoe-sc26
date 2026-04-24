#!/usr/bin/env bash

tablemoe_display_model_name() {
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

tablemoe_default_model_dir() {
  local repo_root="$1"
  local model_key="$2"

  case "$model_key" in
    qwen3vlmoe)
      echo "$repo_root/models/Qwen3-VL-30B-A3B-Instruct"
      ;;
    deepseekvl2)
      echo "$repo_root/models/DeepSeek-VL2"
      ;;
    *)
      echo ""
      ;;
  esac
}

tablemoe_resolve_model_path() {
  local repo_root="$1"
  local model_key="$2"
  local requested_path="${3:-}"

  if [[ -n "$requested_path" ]]; then
    printf '%s\n' "$requested_path"
    return 0
  fi

  local default_dir=""
  default_dir="$(tablemoe_default_model_dir "$repo_root" "$model_key")"
  if [[ -n "$default_dir" && -d "$default_dir" ]]; then
    printf '%s\n' "$default_dir"
    return 0
  fi

  return 1
}

tablemoe_print_model_path_help() {
  local repo_root="$1"
  local model_key="$2"
  local default_dir=""
  default_dir="$(tablemoe_default_model_dir "$repo_root" "$model_key")"

  echo "[ERROR] MODEL_PATH is required."
  if [[ -n "$default_dir" ]]; then
    echo "[ERROR] Default model directory not found: $default_dir"
  fi

  if [[ "$model_key" == "qwen3vlmoe" ]]; then
    echo "[ERROR] Download the default Qwen checkpoint first:"
    echo "bash scripts/download_model.sh"
    echo "[ERROR] Or set MODEL_PATH=<path-to-model>"
  else
    echo "[ERROR] Set MODEL_PATH=<path-to-model>"
  fi
}

tablemoe_collect_missing_eval_acc_modules() {
  local python_bin="${1:-python3}"

  "$python_bin" - <<'PY'
import importlib.metadata
import importlib.util

required = {
    "dotenv": "dotenv",
    "google.genai": "google-genai",
    "gradio": "gradio",
    "ipdb": "ipdb",
    "math_verify": "math-verify",
    "clip": "openai-clip",
    "torchmetrics": "torchmetrics",
}

for module_name, package_name in required.items():
    if importlib.util.find_spec(module_name) is None:
        print(package_name)

try:
    importlib.metadata.version("setuptools")
except importlib.metadata.PackageNotFoundError:
    print("setuptools")
PY
}

tablemoe_ensure_eval_acc_requirements() {
  local repo_root="$1"
  local python_bin="${2:-python3}"
  local missing=()

  mapfile -t missing < <(tablemoe_collect_missing_eval_acc_modules "$python_bin")
  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "[ERROR] Missing eval_acc dependencies: ${missing[*]}"
  echo "[ERROR] Install them before running accuracy evaluation:"
  echo "pip install -r $repo_root/requirements.txt"
  if printf '%s\n' "${missing[@]}" | grep -qx 'setuptools'; then
    echo "[ERROR] If setuptools is already installed but pkg_resources is still missing, run:"
    echo "python -m pip install --force-reinstall setuptools"
  fi
  return 1
}

tablemoe_resolve_tablemoe_dirs() {
  local repo_root="$1"
  local model_key="$2"
  local benchmark="$3"
  local offline_root=""

  case "$model_key" in
    qwen3vlmoe)
      offline_root="$repo_root/offline_table/qwen_fp16"
      ;;
    deepseekvl2)
      offline_root="$repo_root/offline_table/ds_fp16"
      ;;
    *)
      return 1
      ;;
  esac

  printf '%s\n%s\n' \
    "$offline_root/clustering_results/${benchmark}_LayerPCA_256" \
    "$offline_root/offline_table/${benchmark}_LayerPCA_256"
}

tablemoe_cache_env_name() {
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

tablemoe_pca_env_name() {
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

tablemoe_default_cache_base_for_model() {
  local repo_root="$1"
  local model_key="$2"
  case "$model_key" in
    qwen3vlmoe)
      echo "$repo_root/offline_table/qwen_fp16"
      ;;
    deepseekvl2)
      echo "$repo_root/offline_table/ds_fp16"
      ;;
    *)
      echo ""
      ;;
  esac
}

tablemoe_resolve_cache_dir() {
  local benchmark="$1"
  local env_name=""
  env_name="$(tablemoe_cache_env_name "$benchmark")"
  if [[ -n "${env_name}" && -n "${!env_name:-}" ]]; then
    echo "${!env_name}"
    return
  fi
  if [[ -n "${CACHE_ROOT:-}" ]]; then
    echo "${CACHE_ROOT}/${benchmark}${CACHE_DIR_SUFFIX:-_LayerPCA_256}"
    return
  fi
  echo ""
}

tablemoe_resolve_pca_dir() {
  local benchmark="$1"
  local env_name=""
  env_name="$(tablemoe_pca_env_name "$benchmark")"
  if [[ -n "${env_name}" && -n "${!env_name:-}" ]]; then
    echo "${!env_name}"
    return
  fi
  if [[ -n "${PCA_ROOT:-}" ]]; then
    echo "${PCA_ROOT}/${benchmark}${PCA_DIR_SUFFIX:-_LayerPCA_256}"
    return
  fi
  echo ""
}

tablemoe_collect_missing_datasets() {
  local lmu_data_dir="$1"
  shift

  local benchmark=""
  for benchmark in "$@"; do
    if [[ ! -f "$lmu_data_dir/${benchmark}.tsv" ]]; then
      printf '%s\n' "$benchmark"
    fi
  done
}

tablemoe_ensure_dataset_files() {
  local repo_root="$1"
  local lmu_data_dir="$2"
  shift 2

  mkdir -p "$lmu_data_dir"

  local missing=()
  local benchmark=""
  for benchmark in "$@"; do
    if [[ ! -f "$lmu_data_dir/${benchmark}.tsv" ]]; then
      missing+=("$benchmark")
    fi
  done

  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi

  if [[ "${AUTO_DOWNLOAD_DATASETS:-1}" == "1" ]]; then
    echo "[INFO] Missing dataset TSV files: ${missing[*]}"
    echo "[INFO] Downloading dataset TSV files via LMUData/download_datasets.sh"
    if ! DATASETS="${missing[*]}" \
      LMUDATA_DIR="$lmu_data_dir" \
      bash "$repo_root/LMUData/download_datasets.sh"; then
      echo "[WARN] Automatic dataset download failed. Please download the TSV files manually."
    fi
  fi

  local remaining=()
  for benchmark in "${missing[@]}"; do
    if [[ ! -f "$lmu_data_dir/${benchmark}.tsv" ]]; then
      remaining+=("$benchmark")
    fi
  done

  if [[ "${#remaining[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "[ERROR] Missing dataset TSV files: ${remaining[*]}"
  echo "[ERROR] Download them first, for example:"
  echo "DATASETS=\"${remaining[*]}\" bash LMUData/download_datasets.sh"
  return 1
}

tablemoe_supports_hf_offline_table() {
  local model_key="$1"
  local benchmark="$2"

  if [[ "$model_key" != "qwen3vlmoe" ]]; then
    return 1
  fi

  case "$benchmark" in
    RealWorldQA|MMBench_DEV_EN_V11|AI2D_TEST)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

tablemoe_collect_missing_offline_tables() {
  local repo_root="$1"
  local model_key="$2"
  shift 2

  local benchmark=""
  local pca_dir=""
  local cache_dir=""
  local dirs=()
  for benchmark in "$@"; do
    mapfile -t dirs < <(tablemoe_resolve_tablemoe_dirs "$repo_root" "$model_key" "$benchmark")
    pca_dir="${dirs[0]}"
    cache_dir="${dirs[1]}"
    if [[ ! -d "$pca_dir" || ! -d "$cache_dir" ]]; then
      printf '%s\n' "$benchmark"
    fi
  done
}

tablemoe_try_hf_offline_download() {
  local repo_root="$1"
  local model_key="$2"
  shift 2

  if [[ "${HF_AUTO_DOWNLOAD:-1}" != "1" ]]; then
    return 0
  fi

  local downloadable=()
  local benchmark=""
  for benchmark in "$@"; do
    if tablemoe_supports_hf_offline_table "$model_key" "$benchmark"; then
      downloadable+=("$benchmark")
    fi
  done

  if [[ "${#downloadable[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "[INFO] Downloading prebuilt offline tables from Hugging Face for: ${downloadable[*]}"
  if ! MODEL="$model_key" \
    DATASETS="${downloadable[*]}" \
    LOCAL_DIR="$repo_root" \
    HF_OFFLINE_TABLE_REPO="${HF_OFFLINE_TABLE_REPO:-fanyafanya/ALUTs}" \
    bash "$repo_root/offline_table/download_offline_table.sh"; then
    echo "[WARN] Automatic offline table download failed. Falling back to local checks."
  fi
}

tablemoe_ensure_offline_tables() {
  local repo_root="$1"
  local model_key="$2"
  shift 2

  local missing=()
  mapfile -t missing < <(tablemoe_collect_missing_offline_tables "$repo_root" "$model_key" "$@")
  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi

  tablemoe_try_hf_offline_download "$repo_root" "$model_key" "${missing[@]}"

  mapfile -t missing < <(tablemoe_collect_missing_offline_tables "$repo_root" "$model_key" "$@")
  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "[ERROR] Missing TableMoE offline tables for: ${missing[*]}"
  if [[ "$model_key" == "qwen3vlmoe" ]]; then
    local downloadable=()
    local build_required=()
    local benchmark=""
    for benchmark in "${missing[@]}"; do
      if tablemoe_supports_hf_offline_table "$model_key" "$benchmark"; then
        downloadable+=("$benchmark")
      else
        build_required+=("$benchmark")
      fi
    done
    if [[ "${#downloadable[@]}" -gt 0 ]]; then
      echo "[ERROR] Try downloading the published offline tables first:"
      echo "MODEL=Qwen3-VL-30B-A3B-Instruct DATASETS=\"${downloadable[*]}\" bash offline_table/download_offline_table.sh"
    fi
    if [[ "${#build_required[@]}" -gt 0 ]]; then
      echo "[ERROR] Build the remaining offline tables locally:"
      echo "MODEL_PATH=<path-to-model> MODEL_NAME=$(tablemoe_display_model_name "$model_key") DATASETS=\"${build_required[*]}\" bash offline_table/run_offline_table.sh"
    fi
  else
    echo "[ERROR] Build the offline tables locally:"
    echo "MODEL_PATH=<path-to-model> MODEL_NAME=$(tablemoe_display_model_name "$model_key") DATASETS=\"${missing[*]}\" bash offline_table/run_offline_table.sh"
  fi
  return 1
}
