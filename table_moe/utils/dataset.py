import os
import csv

import pandas as pd
from pandas.errors import ParserError


def _read_table_file(data_path: str, sep: str) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path, sep=sep)
    except ParserError as exc:
        print(
            f"Warning: failed to parse {data_path} with pandas C engine ({exc}); "
            "retrying with the Python engine and quote handling disabled."
        )
        return pd.read_csv(
            data_path,
            sep=sep,
            engine="python",
            quoting=csv.QUOTE_NONE,
        )


def _load_table_file(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    lower = data_path.lower()
    if lower.endswith(".tsv"):
        sep = "\t"
    elif lower.endswith(".csv"):
        sep = ","
    else:
        sep = "\t"

    df = _read_table_file(data_path, sep)
    if "index" not in df.columns:
        if "id" in df.columns:
            df["index"] = df["id"]
        else:
            df["index"] = df.index

    if "id" not in df.columns:
        df["id"] = df["index"]

    return _normalize_image_references(df)


def _normalize_image_references(df: pd.DataFrame) -> pd.DataFrame:
    if "image" not in df.columns or "index" not in df.columns:
        return df

    image_map = {str(idx): str(img) for idx, img in zip(df["index"], df["image"])}
    normalized = []
    for idx in df["index"]:
        key = str(idx)
        val = image_map.get(key, "")
        if len(val) <= 64 and val in image_map and len(image_map[val]) > 64:
            val = image_map[val]
        normalized.append(val)

    df["image"] = normalized
    return df


def _load_parquet_dir(data_dir: str) -> pd.DataFrame:
    data_path = os.path.join(data_dir, "data") if os.path.isdir(os.path.join(data_dir, "data")) else data_dir
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Parquet directory not found: {data_path}")

    files = [os.path.join(data_path, name) for name in os.listdir(data_path) if name.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")

    dfs = []
    for file_path in files:
        try:
            dfs.append(pd.read_parquet(file_path))
        except Exception as exc:
            print(f"Warning: skip unreadable parquet file {file_path}: {exc}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    if "index" not in df.columns:
        if "id" in df.columns:
            df["index"] = df["id"]
        else:
            df["index"] = df.index
    if "id" not in df.columns:
        df["id"] = df["index"]
    return _normalize_image_references(df)


def load_mmbench_dataset(data_path: str) -> pd.DataFrame:
    return _load_table_file(data_path)


def load_hallusionbench_dataset(data_path: str) -> pd.DataFrame:
    return _load_table_file(data_path)


def load_ai2d_dataset(data_path: str) -> pd.DataFrame:
    return _load_table_file(data_path)


def load_mme_dataset(data_path: str) -> pd.DataFrame:
    return _load_table_file(data_path)


def load_scienceqa_dataset(data_path: str) -> pd.DataFrame:
    return _load_table_file(data_path)


def load_pope_dataset(data_path: str) -> pd.DataFrame:
    return _load_table_file(data_path)


def load_realworldqa_dataset(data_path: str) -> pd.DataFrame:
    if os.path.isfile(data_path):
        return _load_table_file(data_path)
    if os.path.isdir(data_path):
        return _load_parquet_dir(data_path)
    raise FileNotFoundError(f"RealWorldQA path not found: {data_path}")


def load_mathvision_dataset(data_path: str) -> pd.DataFrame:
    return _load_table_file(data_path)
