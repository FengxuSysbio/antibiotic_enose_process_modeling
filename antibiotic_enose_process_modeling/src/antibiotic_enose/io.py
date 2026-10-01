from __future__ import annotations
from pathlib import Path
import pandas as pd
from .constants import SENSOR_COLS, GENTAMICIN_TARGETS


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_batch_directory(data_dir, required_cols, require_time=True):
    data_dir = Path(data_dir)
    files = sorted([*data_dir.glob("*.csv"), *data_dir.glob("*.xlsx")])
    if not files:
        raise FileNotFoundError(f"No CSV/XLSX batch files found in {data_dir.resolve()}")
    frames = []
    for fp in files:
        df = read_table(fp)
        if "Batch" not in df.columns:
            df["Batch"] = fp.stem
        if require_time and "Time_h" not in df.columns:
            raise ValueError(f"{fp.name}: an explicit Time_h column is required.")
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{fp.name}: missing columns {missing}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_stage_batches(data_dir):
    return load_batch_directory(data_dir, SENSOR_COLS + ["Stage"], require_time=True)


def load_paired_regression_batches(data_dir):
    return load_batch_directory(data_dir, SENSOR_COLS + GENTAMICIN_TARGETS, require_time=True)
