"""
Phase 1 -- Sliding window generation.

Converts the flat engineered-feature table into supervised-learning
tensors:
  X_seq    (N, window_size, len(SEQUENCE_FEATURES))   temporal encoder input
  X_sigma  (N, window_size, len(SIGMA_FEATURES))       raw uncertainty features
  X_nwp    (N, window_size, len(NWP_FEATURES))         NWP ensemble-mean features
  y        (N, horizon)                                pv_norm target
  meta     (N, len(ROUTING_METADATA_FEATURES))          regime + volatility, one
                                                          row per window (last step)
  window_start_time (N,)                                for traceability / plotting

A window at index i is only kept if BOTH the lookback slice and the
horizon slice are fully contiguous (no resampled/missing timestamp gaps
introduced a discontinuity) -- this guards against silently training on
a window that spans a data outage.

Contiguity is checked with a single vectorized diff over the whole
series rather than re-diffing each window: an O(N*W) scan over ~9k
windows x 64 steps is pure waste when one pass answers it for all of them.
"""

import numpy as np
import pandas as pd

from config.model_config import DATA_CONFIG
from src.utils.feature_groups import SEQUENCE_FEATURES, SIGMA_FEATURES, NWP_FEATURES, ROUTING_METADATA_FEATURES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _is_contiguous(timestamps: pd.Series, freq_minutes: int) -> bool:
    """Kept for direct/unit-test use; build path uses the vectorized mask below."""
    diffs = timestamps.diff().dropna()
    expected = pd.Timedelta(minutes=freq_minutes)
    return bool((diffs == expected).all())


def _contiguous_run_lengths(timestamps: pd.Series, freq_minutes: int) -> np.ndarray:
    """
    run[i] = number of consecutive regularly-spaced steps ending at i
    (1 if the step before i is missing/irregular). A window [s, s+L) is
    contiguous iff run[s + L - 1] >= L.
    """
    expected = pd.Timedelta(minutes=freq_minutes)
    ok = np.asarray((timestamps.diff() == expected).to_numpy(), dtype=bool).copy()
    ok[0] = False

    run = np.ones(len(timestamps), dtype=np.int64)
    for i in range(1, len(timestamps)):
        if ok[i]:
            run[i] = run[i - 1] + 1
    return run


def create_windows(
    df: pd.DataFrame,
    window_size: int = DATA_CONFIG.window_size,
    horizon: int = DATA_CONFIG.horizon,
    stride: int = DATA_CONFIG.stride,
    freq_minutes: int = DATA_CONFIG.freq_minutes,
) -> dict:
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    seq_arr = df[SEQUENCE_FEATURES].to_numpy(dtype=np.float32)
    sigma_arr = df[SIGMA_FEATURES].to_numpy(dtype=np.float32)
    nwp_arr = df[NWP_FEATURES].to_numpy(dtype=np.float32)
    target_arr = df["pv_norm"].to_numpy(dtype=np.float32)
    meta_arr = df[ROUTING_METADATA_FEATURES].to_numpy(dtype=np.float32)
    timestamps = pd.to_datetime(df["timestamp"])

    total_span = window_size + horizon
    starts = np.arange(0, max(0, n - total_span + 1), stride)

    run = _contiguous_run_lengths(timestamps, freq_minutes)
    keep = run[starts + total_span - 1] >= total_span if len(starts) else np.zeros(0, dtype=bool)
    kept_starts = starts[keep]
    n_skipped_gap = int(len(starts) - len(kept_starts))

    if len(kept_starts):
        # Vectorized gather: (n_kept, window_size) index matrix
        hist_idx = kept_starts[:, None] + np.arange(window_size)[None, :]
        hor_idx = kept_starts[:, None] + window_size + np.arange(horizon)[None, :]

        X_seq = seq_arr[hist_idx]
        X_sigma = sigma_arr[hist_idx]
        X_nwp = nwp_arr[hist_idx]
        y = target_arr[hor_idx]
        meta = meta_arr[kept_starts + window_size - 1]  # snapshot at forecast issue time
        window_start_time = timestamps.to_numpy()[kept_starts]
    else:
        X_seq = np.empty((0, window_size, len(SEQUENCE_FEATURES)), dtype=np.float32)
        X_sigma = np.empty((0, window_size, len(SIGMA_FEATURES)), dtype=np.float32)
        X_nwp = np.empty((0, window_size, len(NWP_FEATURES)), dtype=np.float32)
        y = np.empty((0, horizon), dtype=np.float32)
        meta = np.empty((0, len(ROUTING_METADATA_FEATURES)), dtype=np.float32)
        window_start_time = np.empty((0,), dtype="datetime64[ns]")

    result = {
        "X_seq": X_seq.astype(np.float32),
        "X_sigma": X_sigma.astype(np.float32),
        "X_nwp": X_nwp.astype(np.float32),
        "y": y.astype(np.float32),
        "meta": meta.astype(np.float32),
        "window_start_time": np.asarray(window_start_time, dtype="datetime64[ns]"),
    }

    logger.info(
        f"Window generation: {len(starts)} candidate windows, "
        f"{n_skipped_gap} skipped for timestamp gaps, "
        f"{result['X_seq'].shape[0]} kept. "
        f"Shapes -> X_seq {result['X_seq'].shape}, X_sigma {result['X_sigma'].shape}, "
        f"X_nwp {result['X_nwp'].shape}, y {result['y'].shape}, meta {result['meta'].shape}."
    )
    return result
