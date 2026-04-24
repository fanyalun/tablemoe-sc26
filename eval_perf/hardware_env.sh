#!/usr/bin/env bash
set -euo pipefail

HARDWARE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TABLEMOE_HARDWARE_PRESET="${TABLEMOE_HARDWARE_PRESET:-}"

if [[ -n "$TABLEMOE_HARDWARE_PRESET" && -f "$TABLEMOE_HARDWARE_PRESET" ]]; then
  # shellcheck disable=SC1090
  source "$TABLEMOE_HARDWARE_PRESET"
fi

get_manual_gpu_value() {
  local gpu_index="$1"
  local key="$2"
  local names=(
    "TABLEMOE_GPU${gpu_index}_${key}"
    "GPU${gpu_index}_${key}"
  )

  local name=""
  for name in "${names[@]}"; do
    local value="${!name:-}"
    if [[ -n "$value" ]]; then
      printf '%s\n' "$value"
      return 0
    fi
  done
  return 1
}

normalize_pci_bus_id() {
  local raw="${1,,}"
  raw="${raw#00000000:}"
  if [[ "$raw" =~ ^[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]$ ]]; then
    printf '0000:%s\n' "$raw"
    return
  fi
  printf '%s\n' "$raw"
}

get_gpu_bus_id() {
  local gpu_index="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader \
    | awk -F', ' -v gpu="$gpu_index" '$1 == gpu {print $2; exit}'
}

get_gpu_numa_node() {
  local bus_id="$1"
  local sysfs_path="/sys/bus/pci/devices/$bus_id/numa_node"
  if [[ ! -f "$sysfs_path" ]]; then
    return 1
  fi
  cat "$sysfs_path"
}

get_numa_cpu_list() {
  local numa_node="$1"
  local cpu_list_file="/sys/devices/system/node/node${numa_node}/cpulist"
  if [[ ! -f "$cpu_list_file" ]]; then
    return 1
  fi
  cat "$cpu_list_file"
}

count_cpu_list() {
  local cpu_list="$1"
  awk -F',' '
    {
      total = 0
      for (i = 1; i <= NF; ++i) {
        split($i, bounds, "-")
        if (bounds[2] == "") {
          total += 1
        } else {
          total += bounds[2] - bounds[1] + 1
        }
      }
      print total
    }
  ' <<<"$cpu_list"
}

expand_cpu_list() {
  local cpu_list="$1"
  awk -F',' '
    BEGIN { first = 1 }
    {
      for (i = 1; i <= NF; ++i) {
        split($i, bounds, "-")
        start = bounds[1]
        end = (bounds[2] == "" ? bounds[1] : bounds[2])
        for (cpu = start; cpu <= end; ++cpu) {
          if (!first) {
            printf ","
          }
          printf "%d", cpu
          first = 0
        }
      }
      printf "\n"
    }
  ' <<<"$cpu_list"
}

expand_cpu_list_with_siblings() {
  local cpu_list="$1"
  local expanded
  expanded="$(expand_cpu_list "$cpu_list")"
  if [[ -z "$expanded" ]]; then
    return 1
  fi

  local sibling_values=()
  local cpu=""
  while IFS= read -r cpu; do
    [[ -z "$cpu" ]] && continue
    local sibling_file="/sys/devices/system/cpu/cpu${cpu}/topology/thread_siblings_list"
    local sibling_list="$cpu"
    if [[ -f "$sibling_file" ]]; then
      sibling_list="$(tr -d ' \n' < "$sibling_file")"
    fi

    local sibling=""
    while IFS= read -r sibling; do
      [[ -z "$sibling" ]] && continue
      sibling_values+=("$sibling")
    done < <(tr ',' '\n' <<<"$sibling_list")
  done < <(tr ',' '\n' <<<"$expanded")

  if [[ "${#sibling_values[@]}" -eq 0 ]]; then
    return 1
  fi

  printf '%s\n' "${sibling_values[@]}" \
    | sort -n -u \
    | paste -sd, -
}

select_physical_cpus() {
  local cpu_list="$1"
  local expanded
  expanded="$(expand_cpu_list "$cpu_list")"
  if [[ -z "$expanded" ]]; then
    return 1
  fi

  declare -A seen_siblings=()
  local selected=()
  local cpu=""
  while IFS= read -r cpu; do
    [[ -z "$cpu" ]] && continue
    local sibling_file="/sys/devices/system/cpu/cpu${cpu}/topology/thread_siblings_list"
    local sibling_key="$cpu"
    if [[ -f "$sibling_file" ]]; then
      sibling_key="$(tr -d ' \n' < "$sibling_file")"
    fi
    if [[ -n "${seen_siblings[$sibling_key]:-}" ]]; then
      continue
    fi
    seen_siblings["$sibling_key"]=1
    selected+=("$cpu")
  done < <(tr ',' '\n' <<<"$expanded")

  if [[ "${#selected[@]}" -eq 0 ]]; then
    return 1
  fi

  local joined=""
  local idx=""
  for idx in "${!selected[@]}"; do
    if [[ -n "$joined" ]]; then
      joined+=","
    fi
    joined+="${selected[$idx]}"
  done
  printf '%s\n' "$joined"
}

take_first_n_cpus() {
  local cpu_list="$1"
  local limit="$2"
  local expanded
  expanded="$(expand_cpu_list "$cpu_list")"
  if [[ -z "$expanded" ]]; then
    return 1
  fi

  local selected=()
  local cpu=""
  while IFS= read -r cpu; do
    [[ -z "$cpu" ]] && continue
    selected+=("$cpu")
    if (( ${#selected[@]} >= limit )); then
      break
    fi
  done < <(tr ',' '\n' <<<"$expanded")

  if [[ "${#selected[@]}" -eq 0 ]]; then
    return 1
  fi

  local joined=""
  local idx=""
  for idx in "${!selected[@]}"; do
    if [[ -n "$joined" ]]; then
      joined+=","
    fi
    joined+="${selected[$idx]}"
  done
  printf '%s\n' "$joined"
}

setup_hardware_env() {
  local gpu_index="${CUDA_DEVICE:-0}"
  local selected_cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-$gpu_index}"
  local bind_strategy="none"
  local bind_source="disabled"
  local binding_enabled="${TABLEMOE_ENABLE_HARDWARE_BINDING:-0}"
  local force_numactl="${TABLEMOE_FORCE_NUMACTL:-${FORCE_NUMACTL:-0}}"
  local use_physical_cores="${TABLEMOE_USE_PHYSICAL_CORES:-0}"
  local numa_core_limit="${TABLEMOE_NUMA_CORE_LIMIT:-}"
  local gpu_bus_id=""
  local numa_node="-1"
  local mem_node="-1"
  local cpu_list=""
  local manual_gpu_bus_id=""
  local manual_numa_node=""
  local manual_mem_node=""
  local manual_cpu_list=""

  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$selected_cuda_visible_devices"

  if [[ -n "${TABLEMOE_HARDWARE_PRESET:-}" ]]; then
    binding_enabled="1"
  fi
  if [[ "$force_numactl" == "1" ]]; then
    binding_enabled="1"
  fi

  export TABLEMOE_SELECTED_GPU="$gpu_index"
  export TABLEMOE_GPU_BUS_ID=""
  export TABLEMOE_NUMA_NODE="-1"
  export TABLEMOE_MEM_NODE="-1"
  export TABLEMOE_CPU_LIST=""
  export TABLEMOE_BIND_SOURCE="$bind_source"
  export TABLEMOE_USE_PHYSICAL_CORES="$use_physical_cores"
  export TABLEMOE_NUMA_CORE_LIMIT="$numa_core_limit"
  export TABLEMOE_HARDWARE_PRESET_PATH="${TABLEMOE_HARDWARE_PRESET:-}"
  export TABLEMOE_HARDWARE_PRESET_NAME="${TABLEMOE_HARDWARE_PRESET_NAME:-}"

  if [[ "$binding_enabled" != "1" ]]; then
    export TABLEMOE_BIND_STRATEGY="none"
    echo "=========================================="
    echo "统一硬件绑定配置"
    echo "=========================================="
    echo "CUDA_DEVICE:             $gpu_index"
    echo "CUDA_VISIBLE_DEVICES:    $CUDA_VISIBLE_DEVICES"
    echo "GPU_BUS_ID:              <unknown>"
    echo "NUMA_NODE:               -1"
    echo "MEM_NODE:                -1"
    echo "CPU_LIST:                <unbound>"
    echo "BOUND_CPU_COUNT:         <unbound>"
    echo "USE_PHYSICAL_CORES:      $use_physical_cores"
    echo "NUMA_CORE_LIMIT:         ${numa_core_limit:-<none>}"
    echo "HARDWARE_PRESET_NAME:    ${TABLEMOE_HARDWARE_PRESET_NAME:-<none>}"
    echo "HARDWARE_PRESET_PATH:    ${TABLEMOE_HARDWARE_PRESET:-<none>}"
    echo "BIND_SOURCE:             $bind_source"
    echo "BIND_STRATEGY:           none"
    return
  fi

  bind_source="auto"

  if manual_gpu_bus_id="$(get_manual_gpu_value "$gpu_index" "BUS_ID" 2>/dev/null)"; then
    gpu_bus_id="$(normalize_pci_bus_id "$manual_gpu_bus_id")"
    bind_source="manual"
  elif gpu_bus_id="$(get_gpu_bus_id "$gpu_index")"; then
    gpu_bus_id="$(normalize_pci_bus_id "$gpu_bus_id")"
  else
    gpu_bus_id=""
  fi

  if manual_numa_node="$(get_manual_gpu_value "$gpu_index" "NUMA_NODE" 2>/dev/null)"; then
    numa_node="$manual_numa_node"
    bind_source="manual"
  elif [[ -n "$gpu_bus_id" ]] && numa_node="$(get_gpu_numa_node "$gpu_bus_id" 2>/dev/null)"; then
    :
  else
    numa_node="-1"
  fi

  if manual_cpu_list="$(get_manual_gpu_value "$gpu_index" "CPU_LIST" 2>/dev/null)"; then
    cpu_list="$manual_cpu_list"
    bind_source="manual"
  elif [[ "$numa_node" =~ ^-?[0-9]+$ ]] && (( numa_node >= 0 )); then
    if cpu_list="$(get_numa_cpu_list "$numa_node" 2>/dev/null)"; then
      :
    else
      cpu_list=""
    fi
  fi

  if manual_mem_node="$(get_manual_gpu_value "$gpu_index" "MEM_NODE" 2>/dev/null)"; then
    mem_node="$manual_mem_node"
    bind_source="manual"
  elif [[ "$numa_node" =~ ^-?[0-9]+$ ]] && (( numa_node >= 0 )); then
    mem_node="$numa_node"
  fi

  if [[ "$use_physical_cores" == "1" ]] && [[ -n "$cpu_list" ]]; then
    local physical_cpu_list=""
    if physical_cpu_list="$(select_physical_cpus "$cpu_list" 2>/dev/null)" && [[ -n "$physical_cpu_list" ]]; then
      cpu_list="$physical_cpu_list"
    fi
  elif [[ -n "$cpu_list" ]]; then
    local sibling_cpu_list=""
    if sibling_cpu_list="$(expand_cpu_list_with_siblings "$cpu_list" 2>/dev/null)" && [[ -n "$sibling_cpu_list" ]]; then
      cpu_list="$sibling_cpu_list"
    fi
  fi

  if [[ -n "$numa_core_limit" ]] && [[ "$numa_core_limit" =~ ^[0-9]+$ ]] && (( numa_core_limit > 0 )) && [[ -n "$cpu_list" ]]; then
    local trimmed_cpu_list=""
    if trimmed_cpu_list="$(take_first_n_cpus "$cpu_list" "$numa_core_limit" 2>/dev/null)" && [[ -n "$trimmed_cpu_list" ]]; then
      cpu_list="$trimmed_cpu_list"
    fi
  fi

  export TABLEMOE_GPU_BUS_ID="$gpu_bus_id"
  export TABLEMOE_NUMA_NODE="$numa_node"
  export TABLEMOE_MEM_NODE="$mem_node"
  export TABLEMOE_CPU_LIST="$cpu_list"
  export TABLEMOE_BIND_SOURCE="$bind_source"
  export TABLEMOE_USE_PHYSICAL_CORES="$use_physical_cores"
  export TABLEMOE_NUMA_CORE_LIMIT="$numa_core_limit"
  export TABLEMOE_HARDWARE_PRESET_PATH="${TABLEMOE_HARDWARE_PRESET:-}"
  export TABLEMOE_HARDWARE_PRESET_NAME="${TABLEMOE_HARDWARE_PRESET_NAME:-}"

  if [[ -n "$cpu_list" ]] && command -v numactl >/dev/null 2>&1 && [[ "$mem_node" =~ ^-?[0-9]+$ ]] && (( mem_node >= 0 )); then
    bind_strategy="numactl"
  elif [[ -n "$cpu_list" ]] && command -v taskset >/dev/null 2>&1; then
    bind_strategy="taskset"
  fi

  if [[ "$force_numactl" == "1" ]]; then
    if ! command -v numactl >/dev/null 2>&1; then
      echo "[ERROR] TABLEMOE_FORCE_NUMACTL=1 but numactl is not available"
      exit 1
    fi
    if [[ -z "$cpu_list" ]]; then
      echo "[ERROR] TABLEMOE_FORCE_NUMACTL=1 but CPU_LIST is empty for GPU $gpu_index"
      echo "[ERROR] Set TABLEMOE_GPU${gpu_index}_CPU_LIST explicitly, or provide a valid NUMA mapping"
      exit 1
    fi
    if [[ ! "$mem_node" =~ ^-?[0-9]+$ ]] || (( mem_node < 0 )); then
      echo "[ERROR] TABLEMOE_FORCE_NUMACTL=1 but MEM_NODE is invalid for GPU $gpu_index"
      echo "[ERROR] Set TABLEMOE_GPU${gpu_index}_MEM_NODE explicitly, or provide a valid NUMA mapping"
      exit 1
    fi
    if [[ "$bind_strategy" != "numactl" ]]; then
      echo "[ERROR] TABLEMOE_FORCE_NUMACTL=1 but numactl binding could not be prepared"
      exit 1
    fi
  fi

  export TABLEMOE_BIND_STRATEGY="$bind_strategy"

  local bound_cpu_count="<unbound>"
  if [[ -n "$cpu_list" ]]; then
    bound_cpu_count="$(count_cpu_list "$cpu_list")"
  fi

  echo "=========================================="
  echo "统一硬件绑定配置"
  echo "=========================================="
  echo "CUDA_DEVICE:             $gpu_index"
  echo "CUDA_VISIBLE_DEVICES:    $CUDA_VISIBLE_DEVICES"
  echo "GPU_BUS_ID:              ${gpu_bus_id:-<unknown>}"
  echo "NUMA_NODE:               $numa_node"
  echo "MEM_NODE:                $mem_node"
  echo "CPU_LIST:                ${cpu_list:-<unbound>}"
  echo "BOUND_CPU_COUNT:         $bound_cpu_count"
  echo "USE_PHYSICAL_CORES:      $use_physical_cores"
  echo "NUMA_CORE_LIMIT:         ${numa_core_limit:-<none>}"
  echo "HARDWARE_PRESET_NAME:    ${TABLEMOE_HARDWARE_PRESET_NAME:-<none>}"
  echo "HARDWARE_PRESET_PATH:    ${TABLEMOE_HARDWARE_PRESET:-<none>}"
  echo "BIND_SOURCE:             $bind_source"
  echo "BIND_STRATEGY:           $bind_strategy"
}

run_with_hardware_env() {
  setup_hardware_env

  if [[ "$TABLEMOE_BIND_STRATEGY" == "numactl" ]]; then
    numactl --physcpubind="$TABLEMOE_CPU_LIST" --membind="$TABLEMOE_MEM_NODE" "$@"
    return
  fi

  if [[ "$TABLEMOE_BIND_STRATEGY" == "taskset" ]]; then
    taskset -c "$TABLEMOE_CPU_LIST" "$@"
    return
  fi

  "$@"
}
