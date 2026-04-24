# Tablemoe Reproduction Guide

This document only describes how to reproduce the results reported in the paper.

## 1. Validated Software and Hardware

The repository was validated with the following stack:

| Item | Version / Hardware |
| --- | --- |
| Python | `3.10.18` |
| CUDA | `12.8` |
| PyTorch | `2.8.0` |
| torchvision | `0.23.0` |
| flash-attn | `2.7.3` |
| CPU | dual Intel Xeon Gold 5318Y |
| GPU | NVIDIA A100 80GB PCIe |

Performance numbers depend on the hardware configuration. To reproduce the paper tables as closely as possible, use the validated stack above.

## 2. Base Environment

```bash
git clone https://github.com/fanyalun/tablemoe-sc26.git
cd tablemoe-sc26

conda create -n tablemoe python=3.10.18 -y
conda activate tablemoe

pip install torch==2.8.0 torchvision==0.23.0
pip install -r requirements.txt
# Prefer a prebuilt flash-attn wheel when one is available for your local torch/CUDA stack.
pip install flash-attn==2.7.3 --no-build-isolation
```

The default environment targets `Qwen3-VL-30B-A3B-Instruct`.

If you want to run `DeepSeek-VL2`, switch `transformers` to `4.38.0` and install `xformers==0.0.32.post1`:

```bash
pip install transformers==4.38.0
pip install xformers==0.0.32.post1
```

## 3. Quick Reproduction

The quick reproduction path uses only the default configuration:

- model: `Qwen3-VL-30B-A3B-Instruct`
- dataset: `MMBench_DEV_EN_V11`
- `sample_ratio=0.01`
- published offline table downloaded from Hugging Face
- final artifact: `perf_results/quick_reproduction/summary/quick_reproduction_report.md`

### 3.1 Prepare the Default Assets

Download the default Qwen checkpoint:

```bash
bash scripts/download_model.sh
```

Download the default dataset TSV:

```bash
bash LMUData/download_datasets.sh
```

Download the published default offline table:

```bash
apt-get update && apt-get install -y 7zip
bash offline_table/download_offline_table.sh
```

Offline table construction does not support offloading. It requires loading the full model while building the table. If your GPU memory is smaller than `80GB`, we recommend downloading the published default offline table directly instead of building it locally.

### 3.2 Run the Quick Reproduction Script

```bash
CUDA_VISIBLE_DEVICES=0 bash run_quick_reproduction.sh
```

The script writes intermediate outputs under:

- `perf_results/quick_reproduction/`

The final quick reproduction artifacts are:

- `perf_results/quick_reproduction/summary/quick_reproduction_report.md`
- `perf_results/quick_reproduction/summary/quick_reproduction_manifest.json`

The final report contains:

- TTFT / TPOT comparison across AdapMoE, AdapMoE(+gating), +ALUT, +WINDOW, and TableMoE
- decoding cache hit rate comparison across AdapMoE, AdapMoE(+gating), +ALUT, +WINDOW, and TableMoE
- lightweight accuracy comparison across AdapMoE, AdapMoE(+gating), +ALUT, +WINDOW, and TableMoE

The lightweight accuracy comparison comes from the `simple_accuracy_table` already collected inside `eval_perf`. It is intended only for quick comparison under the released default configuration.

### 3.3 Optional: Run the Full Accuracy Comparison

The full five-method accuracy comparison is intentionally separated from the quick reproduction path because `MMBench_DEV_EN_V11` is significantly slower to evaluate end-to-end.

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=<model_path> \
RESULT_DIR_NAME=quick_reproduction_full_acc \
METHODS=transformers,skip,offline,online,tablemoe \
CACHE_RATIO=1.0 \
bash eval_acc/eval_acc.sh
```

This command writes:

- `acc_results/quick_reproduction_full_acc/summary/accuracy_table.{json,csv,md}`

For the final paper-quality accuracy numbers, re-run the saved outputs with the local judge model as described in Section 4.5.

## 4. Full Reproduction

### 4.1 Optional: Configure CUTLASS and Build the Two Baselines

`CUTLASS` is required only for `moe-infinity` and `pregated-moe`.

If `CUTLASS` is not available on your machine, clone the official repository first:

```bash
# Optional: only needed for moe-infinity and pregated-moe
# git clone https://github.com/NVIDIA/cutlass.git ~/cutlass
# export CUTLASS_DIR=~/cutlass
pip install --no-build-isolation --no-deps -e third_party/moe-infinity
pip install --no-build-isolation --no-deps -e third_party/pregated-moe
```

If you clone `CUTLASS` to a different location, set `CUTLASS_DIR` to that path before installing the two baselines.

`moe-infinity` and `pregated-moe` are third-party baselines. Due to upstream lock/runtime behavior, they may occasionally fail at runtime. If either baseline run exits with an error, please retry the same command.

### 4.2 Public Parameters and Default Configuration

| Variable | Default | Accepted values / meaning |
| --- | --- | --- |
| `MODEL` | `Qwen3-VL-30B-A3B-Instruct` | `Qwen3-VL-30B-A3B-Instruct`, `DeepSeek-VL2` |
| `MODEL_PATH` | optional for the default Qwen checkpoint | Path to the model checkpoint on the target server. Can be omitted if `Qwen3-VL-30B-A3B-Instruct` is stored under `models/Qwen3-VL-30B-A3B-Instruct/` |
| `DATASETS` | `MMBench_DEV_EN_V11` | Comma-separated list from `RealWorldQA`, `MMBench_DEV_EN_V11`, `AI2D_TEST`, `ScienceQA_TEST`, `POPE` |
| `METHODS` for `eval_perf` | `tablemoe` | Comma-separated list from `adapmoe`, `skip`, `offline`, `online`, `tablemoe`, `pregated-moe`, `moe-infinity` |
| `METHODS` for `eval_acc` | `tablemoe` | Comma-separated list from `transformers`, `skip`, `offline`, `online`, `tablemoe` |
| `MAX_NEW_TOKENS` for `eval_acc` | `128` | Default generation limit used by `eval_acc/eval_acc.sh` and `eval_acc/run_vlmeval.py` unless explicitly overridden |
| `CACHE_RATIO` | `0.5` | `0.1`, `0.25`, `0.5`, `0.75`, `0.9`, `1.0`; for `tablemoe`, `1.0` means all routed experts stay resident on GPU while keeping the original buffer size; for `pregated-moe` and `moe-infinity`, this is forwarded as `expert_cache_ratio` |
| `KEEP_RATE` | `0.6` | Recomputation keep rate used by `tablemoe` |
| `SAMPLE_RATIO` | `0.01` | Sampling ratio for `eval_perf` |

If `MODEL="DeepSeek-VL2"`, switch `transformers` to `4.38.0` and install `xformers==0.0.32.post1` before running the commands below.

### 4.3 Run `eval_perf` with Arbitrary Parameters

```bash
CUDA_VISIBLE_DEVICES=0 MODEL="<model-name>" MODEL_PATH="<path-to-model>" DATASETS="<dataset1,dataset2,...>" METHODS="<method1,method2,...>" CACHE_RATIO="<cache_ratio>" KEEP_RATE="<keep_rate>" SAMPLE_RATIO="<sample_ratio>" bash eval_perf/eval_perf.sh
```

`eval_perf` writes:

- `perf_results/<result_dir_name>/summary/perf_table.{json,csv,md}`
- `perf_results/<result_dir_name>/summary/simple_accuracy_table.{json,csv,md}`

### 4.4 Run `eval_acc` with Arbitrary Parameters

```bash
CUDA_VISIBLE_DEVICES=0 MODEL="<model-name>" MODEL_PATH="<path-to-model>" DATASETS="<dataset1,dataset2,...>" METHODS="<method1,method2,...>" CACHE_RATIO="<cache_ratio>" KEEP_RATE="<keep_rate>" bash eval_acc/eval_acc.sh
```

`eval_acc` writes:

- `acc_results/<result_dir_name>/summary/accuracy_table.{json,csv,md}`

### 4.5 Re-evaluate Accuracy with the Local Judge Model

For the final paper accuracy reproduction, re-evaluate the saved `eval_acc` outputs with the local judge model.

The paper uses a locally deployed LLaMA-compatible `Qwen/Qwen3-VL-30B-A3B-Instruct` endpoint as the judge model. The local API configuration is defined in:

- `third_party/VLMEvalKit/.env`

After `eval_acc` finishes, re-evaluate the saved results:

```bash
ACC_ROOT=acc_results/default bash eval_acc/eval_with_local_judge.sh
```

This command updates the saved accuracy outputs and refreshes:

- `acc_results/default/summary/accuracy_table.{json,csv,md}`

## 5. Notes

- Use the dedicated quick reproduction script for the released default configuration.
- Use `eval_perf` for arbitrary performance sweeps and lightweight accuracy comparison.
- Use `eval_acc` + local judge re-evaluation for the final paper-grade accuracy reproduction.
- Use the published Hugging Face offline tables for the released Qwen datasets whenever possible.
- Build the remaining offline tables locally when they are not published.
- `pregated-moe` exits with an explicit timeout error if layer preparation waits for more than 3 minutes. If this happens, retry the same command.
