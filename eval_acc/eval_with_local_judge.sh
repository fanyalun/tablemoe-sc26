#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLM_EVAL_ROOT="${PROJECT_DIR}/third_party/VLMEvalKit"
ACC_ROOT_RAW="${ACC_ROOT:-${PROJECT_DIR}/acc_results}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
DEFAULT_JUDGE_ARGS='{"do_sample": false}'
JUDGE_ARGS_RAW="${JUDGE_ARGS:-$DEFAULT_JUDGE_ARGS}"
LOG_DIR_RAW="${LOG_DIR:-${ACC_ROOT_RAW}/re_eval_logs}"
TS="$(date +%Y%m%d_%H%M%S)"
ACC_ROOT=""
LOG_DIR=""
LOG_FILE=""

TARGET_DATASETS=(
  "RealWorldQA"
  "MMBench_DEV_EN_V11"
  "ScienceQA_TEST"
  "POPE"
  "AI2D_TEST"
)

usage() {
 cat <<'EOF'
用法:
  bash eval_acc/eval_with_local_judge.sh [EXP_DIR ...]

说明:
  1. EXP_DIR 目录级别应为 acc_results/<result_dir_name>/<method>/<model_name>，例如:
     <repo_root>/acc_results/default/tablemoe/qwen3_vl_tablemoe
  2. 每个 EXP_DIR 下应直接包含历史结果目录。
  3. 若不传位置参数，也不设置 EXP_DIRS / EXP_DIRS_RAW，
     脚本会自动扫描 acc_results 下所有符合条件的模型目录。
  4. 重评完成后，脚本会自动刷新对应 result_root 下的 accuracy_table 汇总表。

常用环境变量:
  ACC_ROOT
  EXP_DIR_LIST
  EXP_DIRS
  EXP_DIRS_RAW
  JUDGE_MODEL
  JUDGE_ARGS
  LOG_DIR

建议:
  EXP_DIRS 使用换行、逗号或分号分隔多个目录。
EOF
}

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

looks_like_model_result_dir() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -type f \( -name "*.xlsx" -o -name "*.pkl" \) | grep -q .
}

parse_exp_dirs_raw() {
  local raw="${1:-}"
  [[ -n "$raw" ]] || return 0

  while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -n "$line" ]] || continue
    printf '%s\n' "$line"
  done < <(printf '%s\n' "$raw" | tr ',;' '\n\n')
}

resolve_abs_path() {
  local input_path="$1"
  python3 - "$PROJECT_DIR" "$input_path" <<'PY'
import os
import sys

project_dir = os.path.abspath(sys.argv[1])
input_path = sys.argv[2]
if os.path.isabs(input_path):
    print(os.path.abspath(input_path))
else:
    print(os.path.abspath(os.path.join(project_dir, input_path)))
PY
}

normalize_exp_dir() {
  local exp_dir="$1"
  local abs_dir
  abs_dir="$(resolve_abs_path "$exp_dir")"

  if [[ ! -d "$abs_dir" ]]; then
    printf '%s\n' "$abs_dir"
    return 0
  fi

  local parent_dir
  local grandparent_dir
  parent_dir="$(dirname "$abs_dir")"
  grandparent_dir="$(dirname "$parent_dir")"

  case "$(basename "$grandparent_dir")" in
    transformers|skip|tablemoe)
      if looks_like_model_result_dir "$abs_dir"; then
        printf '%s\n' "$parent_dir"
        return 0
      fi
      ;;
  esac

  printf '%s\n' "$abs_dir"
}

collect_default_exp_dirs() {
  [[ -d "${ACC_ROOT}" ]] || return 0
  while IFS= read -r dir; do
    [[ -n "$dir" ]] || continue
    if looks_like_model_result_dir "$dir"; then
      printf '%s\n' "$dir"
    fi
  done < <(find "${ACC_ROOT}" -type d ! -name generated_configs ! -name summary | sort)
}

has_prediction_for_dataset() {
  local exp_dir="$1"
  local model_name="$2"
  local dataset="$3"

  find "${exp_dir}" -type f \( \
      -name "${model_name}_${dataset}.xlsx" -o \
      -name "${model_name}_${dataset}.pkl" -o \
      -name "${model_name}_${dataset}_"'*_result.pkl' -o \
      -name "01_${dataset}.pkl" \
    \) | grep -q .
}

update_model_summary() {
  local exp_dir="$1"
  local method_dir
  local method
  local model_name

  method_dir="$(dirname "$exp_dir")"
  method="$(basename "$method_dir")"
  model_name="$(basename "$exp_dir")"

  python3 - "$PROJECT_DIR" "$method_dir" "$model_name" "$method" "$JUDGE_MODEL" <<'PY'
import json
import sys
from pathlib import Path

project_dir = Path(sys.argv[1]).resolve()
work_dir = Path(sys.argv[2]).resolve()
model_name = sys.argv[3]
method = sys.argv[4]
judge_model = sys.argv[5]

sys.path.insert(0, str(project_dir))
from eval_acc.run_vlmeval import collect_summary  # noqa: E402

summary_path = work_dir / f"{model_name}_summary.json"
model_key = None
benchmarks = None
if summary_path.is_file():
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    model_key = payload.get("model_key")
    benchmarks = list(payload.get("benchmarks", {}).keys())

if model_key is None:
    lower = model_name.lower()
    if "deepseek" in lower:
        model_key = "deepseekvl2"
    else:
        model_key = "qwen3vlmoe"

if not benchmarks:
    benchmarks = [
        "RealWorldQA",
        "MMBench_DEV_EN_V11",
        "AI2D_TEST",
        "ScienceQA_TEST",
        "POPE",
    ]

collect_summary(
    work_dir=work_dir,
    output_model_name=model_name,
    model_key=model_key,
    method=method,
    benchmarks=benchmarks,
    metric_type="judge",
    judge_model=judge_model,
)
PY
}

update_accuracy_tables() {
  local result_root="$1"
  python3 "$SCRIPT_DIR/summarize_accuracy.py" \
    --result-root "$result_root" \
    --output-dir "$result_root/summary"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ACC_ROOT="$(resolve_abs_path "${ACC_ROOT_RAW}")"
LOG_DIR="$(resolve_abs_path "${LOG_DIR_RAW}")"
LOG_FILE="${LOG_DIR}/re_eval_${TS}.log"

mkdir -p "${LOG_DIR}"

JUDGE_ARGS_CANON="$(python3 - "${JUDGE_ARGS_RAW}" <<'PY'
import json
import sys

raw = sys.argv[1]
obj = json.loads(raw)
print(json.dumps(obj, separators=(',', ':')))
PY
)"

RESOLVED_EXP_DIRS=()
if [[ $# -eq 1 && "${1:-}" == EXP_DIR_LIST=* ]]; then
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    RESOLVED_EXP_DIRS+=("$line")
  done < <(parse_exp_dirs_raw "${1#EXP_DIR_LIST=}")
elif [[ $# -gt 0 ]]; then
  for exp_dir in "$@"; do
    RESOLVED_EXP_DIRS+=("$exp_dir")
  done
elif [[ -n "${EXP_DIR_LIST:-}" ]]; then
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    RESOLVED_EXP_DIRS+=("$line")
  done < <(parse_exp_dirs_raw "${EXP_DIR_LIST}")
elif [[ -n "${EXP_DIRS:-}" ]]; then
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    RESOLVED_EXP_DIRS+=("$line")
  done < <(parse_exp_dirs_raw "${EXP_DIRS}")
elif [[ -n "${EXP_DIRS_RAW:-}" ]]; then
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    RESOLVED_EXP_DIRS+=("$line")
  done < <(parse_exp_dirs_raw "${EXP_DIRS_RAW}")
else
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    RESOLVED_EXP_DIRS+=("$line")
  done < <(collect_default_exp_dirs)
fi

if [[ "${#RESOLVED_EXP_DIRS[@]}" -eq 0 ]]; then
  echo "[ERROR] No EXP_DIR found. Provide positional args or EXP_DIRS / EXP_DIRS_RAW." >&2
  exit 1
fi

log "Start targeted re-evaluation"
log "VLMEvalKit root: ${VLM_EVAL_ROOT}"
log "Acc root: ${ACC_ROOT}"
log "Judge: ${JUDGE_MODEL}"
log "Judge args(raw): ${JUDGE_ARGS_RAW}"
log "Judge args(canon): ${JUDGE_ARGS_CANON}"
NORMALIZED_EXP_DIRS=()
for i in "${!RESOLVED_EXP_DIRS[@]}"; do
  normalized_dir="$(normalize_exp_dir "${RESOLVED_EXP_DIRS[$i]}")"
  already_added=0
  for existing_dir in "${NORMALIZED_EXP_DIRS[@]:-}"; do
    if [[ "$existing_dir" == "$normalized_dir" ]]; then
      already_added=1
      break
    fi
  done
  if [[ "$already_added" == "0" ]]; then
    NORMALIZED_EXP_DIRS+=("$normalized_dir")
  fi
done
RESOLVED_EXP_DIRS=("${NORMALIZED_EXP_DIRS[@]}")

log "Target dirs: ${RESOLVED_EXP_DIRS[*]}"
log "Target datasets: ${TARGET_DATASETS[*]}"

cd "${VLM_EVAL_ROOT}"

UPDATED_RESULT_ROOTS=()

for EXP_DIR in "${RESOLVED_EXP_DIRS[@]}"; do
  if [[ ! -d "${EXP_DIR}" ]]; then
    log "Skip missing dir: ${EXP_DIR}"
    continue
  fi

  model_name="$(basename "${EXP_DIR}")"
  work_dir="$(dirname "${EXP_DIR}")"
  result_root="$(dirname "${work_dir}")"
  log "Processing model dir: ${EXP_DIR}"
  log "Resolved model: ${model_name}"
  log "Resolved work-dir: ${work_dir}"

  for dataset in "${TARGET_DATASETS[@]}"; do
    if ! has_prediction_for_dataset "${EXP_DIR}" "${model_name}" "${dataset}"; then
      log "  - Skip dataset ${dataset}: no historical prediction files"
      continue
    fi

    log "  - Re-eval dataset ${dataset}"
    python3 run.py \
      --model "${model_name}" \
      --data "${dataset}" \
      --work-dir "${work_dir}" \
      --mode eval \
      --reuse \
      --reuse-aux 0 \
      --judge "${JUDGE_MODEL}" \
      --judge-args "${JUDGE_ARGS_CANON}" \
      2>&1 | tee -a "${LOG_FILE}"
  done

  update_model_summary "${EXP_DIR}"

  already_added=0
  for existing_root in "${UPDATED_RESULT_ROOTS[@]:-}"; do
    if [[ "${existing_root}" == "${result_root}" ]]; then
      already_added=1
      break
    fi
  done
  if [[ "${already_added}" == "0" ]]; then
    UPDATED_RESULT_ROOTS+=("${result_root}")
  fi
done

for result_root in "${UPDATED_RESULT_ROOTS[@]:-}"; do
  log "Refresh accuracy summary tables: ${result_root}"
  update_accuracy_tables "${result_root}"
done

log "Done. Log: ${LOG_FILE}"
