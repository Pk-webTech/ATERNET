"""
Misc helper functions shared across Phase 1 modules: chronological
splitting, min-max normalization, cyclical time encoding.
"""

import numpy as np
import pandas as pd


def chronological_split(n_rows: int, train_ratio: float, val_ratio: float):
    """
    Return (train_idx, val_idx, test_idx) as slices over a chronologically
    ordered array of length n_rows. No shuffling -- prevents temporal leakage.
    """
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n_rows)
    return train_idx, val_idx, test_idx


def minmax_fit(x: np.ndarray, axis=0, eps: float = 1e-8):
    x_min = np.nanmin(x, axis=axis, keepdims=True)
    x_max = np.nanmax(x, axis=axis, keepdims=True)
    scale = np.maximum(x_max - x_min, eps)
    return {"min": x_min, "scale": scale}


def minmax_transform(x: np.ndarray, scaler: dict) -> np.ndarray:
    return (x - scaler["min"]) / scaler["scale"]


def minmax_inverse(x: np.ndarray, scaler: dict) -> np.ndarray:
    return x * scaler["scale"] + scaler["min"]


def cyclical_encode(values: np.ndarray, period: float):
    """Encode a periodic variable (hour-of-day, day-of-year) as (sin, cos)."""
    radians = 2.0 * np.pi * values / period
    return np.sin(radians), np.cos(radians)


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    ts = pd.to_datetime(df[timestamp_col])
    hour_frac = ts.dt.hour + ts.dt.minute / 60.0
    doy = ts.dt.dayofyear

    hour_sin, hour_cos = cyclical_encode(hour_frac.values, 24.0)
    doy_sin, doy_cos = cyclical_encode(doy.values, 365.25)

    df = df.copy()
    df["hour_sin"] = hour_sin
    df["hour_cos"] = hour_cos
    df["doy_sin"] = doy_sin
    df["doy_cos"] = doy_cos
    return df


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill=0.0) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator, fill)
    return result


def count_parameters(module, trainable_only: bool = True) -> int:
    """Total parameter count of a torch module."""
    return sum(p.numel() for p in module.parameters()
               if (p.requires_grad or not trainable_only))
