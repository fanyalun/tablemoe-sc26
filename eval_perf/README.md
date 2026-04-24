# `eval_perf`

`eval_perf/` provides the unified performance-evaluation entrypoint for the released datasets. The public methods are:

- `adapmoe`
- `AdapMoE(+gating)`
- `+ALUT`
- `+WINDOW`
- `tablemoe`
- `pregated-moe`
- `moe-infinity`

Default values:

- `MODEL=qwen3vlmoe`
- `METHODS=tablemoe`
- `DATASETS=MMBench_DEV_EN_V11`
- `CACHE_RATIO=0.5`
- `KEEP_RATE=0.6`
- `SAMPLE_RATIO=0.01`
- `LMUDATA_DIR=<repo_root>/LMUData`

## Direct Run

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=<model_path> \
bash eval_perf/eval_perf.sh
```

Wrapper scripts:

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model_path> bash eval_perf/run_adapmoe.sh
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model_path> bash eval_perf/run_skip.sh
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model_path> bash eval_perf/run_offline.sh
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model_path> bash eval_perf/run_online.sh
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model_path> bash eval_perf/run_tablemoe.sh
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model_path> bash eval_perf/run_pregated_moe.sh
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model_path> bash eval_perf/run_moe_infinity.sh
```

## Supported Datasets

- `RealWorldQA`
- `MMBench_DEV_EN_V11`
- `AI2D_TEST`
- `ScienceQA_TEST`
- `POPE`

## Parameter Notes

- `CACHE_RATIO`
  - `adapmoe / skip / offline / online / tablemoe` use the shared preset table and support `0.1 / 0.25 / 0.5 / 0.75 / 0.9 / 1.0`
  - `pregated-moe / moe-infinity` forward this value to their own `expert_cache_ratio`
- `KEEP_RATE`
  - only `tablemoe` uses it
  - `offline / online / adapmoe / skip / pregated-moe / moe-infinity` do not use non-default keep rates

## TableMoE Offline-Table Paths

By default, the paths fall back to the model-specific presets:

- `offline_table/qwen_fp16/offline_table/<dataset>_LayerPCA_256`
- `offline_table/qwen_fp16/clustering_results/<dataset>_LayerPCA_256`
- `offline_table/ds_fp16/offline_table/<dataset>_LayerPCA_256`
- `offline_table/ds_fp16/clustering_results/<dataset>_LayerPCA_256`

You can also override them with `CACHE_ROOT / PCA_ROOT` or provide per-dataset overrides.

## Outputs

After the run finishes, the following files are generated under `perf_results/<result_dir_name>/summary/`:

- `perf_table.{json,csv,md}`
- `simple_accuracy_table.{json,csv,md}`

`simple_accuracy_table` comes from the lightweight accuracy statistic already collected inside `eval_perf`. It is only intended for quick comparison. For the full paper-quality accuracy results, use `eval_acc/` together with `VLMEvalKit`.
