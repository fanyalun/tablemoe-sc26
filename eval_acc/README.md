# `eval_acc`

`eval_acc/` provides the unified accuracy-evaluation entrypoint for the released datasets. The public methods are:

- `transformers`
- `AdapMoE(+gating)`
- `+ALUT`
- `+WINDOW`
- `tablemoe`

Default values:

- `MODEL=qwen3vlmoe`
- `METHODS=tablemoe`
- `DATASETS=MMBench_DEV_EN_V11`
- `MAX_NEW_TOKENS=128`
- `CACHE_RATIO=0.5`
- `KEEP_RATE=0.6`
- `LMUDATA_DIR=<repo_root>/LMUData`

## Direct Run

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=<model_path> \
bash eval_acc/eval_acc.sh
```

Wrapper scripts:

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model_path> bash eval_acc/run_transformers.sh
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model_path> bash eval_acc/run_skip.sh
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model_path> bash eval_acc/run_offline.sh
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model_path> bash eval_acc/run_online.sh
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model_path> bash eval_acc/run_tablemoe.sh
```

## Supported Datasets

- `RealWorldQA`
- `MMBench_DEV_EN_V11`
- `AI2D_TEST`
- `ScienceQA_TEST`
- `POPE`

## Judge-Based Re-Evaluation

The default path uses exact match or the built-in dataset metric. To enable judge-based re-evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=<model_path> \
JUDGE_MODEL=<judge_model> \
JUDGE_ARGS='{"do_sample": false}' \
bash eval_acc/eval_acc.sh
```

To rerun the judge on existing results:

```bash
bash eval_acc/eval_with_local_judge.sh
```

## Outputs

After the run finishes, the following files are generated under `acc_results/<result_dir_name>/summary/`:

- `accuracy_table.{json,csv,md}`
