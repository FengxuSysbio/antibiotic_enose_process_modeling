from __future__ import annotations
import numpy as np
from .constants import SENSOR_COLS


def baseline_log_ratio(sensor_matrix, baseline_points=10, eps=1e-8):
    x = np.asarray(sensor_matrix, float)
    n = min(int(baseline_points), len(x))
    baseline = np.nanmedian(x[:n], axis=0)
    return np.log2((x + eps) / (baseline + eps))


def relative_progress(t):
    t = np.asarray(t, float)
    return (t - t.min()) / (t.max() - t.min() + 1e-12)


def make_classification_windows(df, window=60, stride=1, baseline_points=10):
    X, y, batch_ids, end_times, abnormal = [], [], [], [], []
    for batch, g in df.groupby("Batch", sort=False):
        g = g.sort_values("Time_h").reset_index(drop=True)
        z = baseline_log_ratio(g[SENSOR_COLS].to_numpy(float), baseline_points)
        for end in range(window-1, len(g), stride):
            start = end-window+1
            X.append(z[start:end+1])
            y.append(int(g.loc[end, "Stage"]))
            batch_ids.append(batch)
            end_times.append(float(g.loc[end, "Time_h"]))
            abnormal.append(int(g.loc[end, "Is_abnormal"]) if "Is_abnormal" in g.columns else 0)
    return (np.asarray(X, np.float32), np.asarray(y, int), np.asarray(batch_ids),
            np.asarray(end_times, float), np.asarray(abnormal, int))


def make_regression_sequences(df, target, window=6):
    Xs, ys, batches, times = [], [], [], []
    for batch, g in df.groupby("Batch", sort=False):
        g = g.sort_values("Time_h").reset_index(drop=True)
        x = g[SENSOR_COLS].to_numpy(np.float32)
        y = g[target].to_numpy(np.float32)
        t = g["Time_h"].to_numpy(float)
        for i in range(window-1, len(g)):
            Xs.append(x[i-window+1:i+1])
            ys.append(y[i])
            batches.append(batch)
            times.append(t[i])
    return np.asarray(Xs), np.asarray(ys), np.asarray(batches), np.asarray(times)
