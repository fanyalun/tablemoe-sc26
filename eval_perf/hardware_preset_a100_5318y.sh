#!/usr/bin/env bash

# Fixed hardware preset for:
# - 2 x Intel Xeon Gold 5318Y (24 physical cores per socket, SMT on)
# - 2 x NVIDIA A100 80GB PCIe
# - GPU0 <-> NUMA0, GPU1 <-> NUMA1

: "${TABLEMOE_HARDWARE_PRESET_NAME:=a100_pcie_80g_dual_5318y}"

# Default policy:
# - one isolated GPU
# - one whole local NUMA node
# - local memory binding
: "${TABLEMOE_FORCE_NUMACTL:=1}"

# GPU0 is attached to NUMA node 0.
: "${TABLEMOE_GPU0_BUS_ID:=0000:65:00.0}"
: "${TABLEMOE_GPU0_NUMA_NODE:=0}"
: "${TABLEMOE_GPU0_MEM_NODE:=0}"
: "${TABLEMOE_GPU0_CPU_LIST:=0-23}"

# GPU1 is attached to NUMA node 1.
: "${TABLEMOE_GPU1_BUS_ID:=0000:B1:00.0}"
: "${TABLEMOE_GPU1_NUMA_NODE:=1}"
: "${TABLEMOE_GPU1_MEM_NODE:=1}"
: "${TABLEMOE_GPU1_CPU_LIST:=24-47}"
