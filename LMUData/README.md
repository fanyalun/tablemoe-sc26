# LMUData

This directory stores the dataset TSV files used by the reproduction scripts.

## Download TSV Files

The repository provides a wrapper that reuses the dataset registry in `third_party/VLMEvalKit` and downloads the official TSV files on demand:

```bash
bash LMUData/download_datasets.sh
```

By default, this downloads `RealWorldQA.tsv` into `LMUData/`.

To download all five paper datasets:

```bash
DATASETS="RealWorldQA MMBench_DEV_EN_V11 AI2D_TEST ScienceQA_TEST POPE" \
bash LMUData/download_datasets.sh
```

Supported dataset names:

- `RealWorldQA`
- `MMBench_DEV_EN_V11`
- `AI2D_TEST`
- `ScienceQA_TEST`
- `POPE`

## Directory Layout

The TSV files should be placed directly under `LMUData/`, for example:

- `LMUData/RealWorldQA.tsv`
- `LMUData/MMBench_DEV_EN_V11.tsv`
- `LMUData/AI2D_TEST.tsv`
- `LMUData/ScienceQA_TEST.tsv`
- `LMUData/POPE.tsv`

Image files referenced by the TSV rows should be available from the paths expected by VLMEvalKit and the local dataset loader.
