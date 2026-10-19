"""
Phase 1 -- Sliding window generation.

Converts the flat engineered-feature table into supervised-learning
tensors:
  X_seq    (N, window_size, len(SEQUENCE_FEATURES))   temporal encoder input
  X_sigma  (N, window_size, len(SIGMA_FEATURES))       raw uncertainty features
  X_nwp    (N, window_size + horizon, len(NWP_FEATURES))
                                                        NWP ensemble-mean features over
                                                        the lookback AND the forecast
                                                        horizon (see note below)
  y        (N, horizon)                                pv_norm target
  meta     (N, len(ROUTING_METADATA_FEATURES))          regime + volatility, one
                                                          row per window (last step)
  window_start_time (N,)                                for traceability / plotting

A window at index i is only kept if BOTH the lookback slice and the
horizon slice are fully contiguous (no resampled/missing timestamp gaps
introduced a discontinuity) -- this guards against silently training on
a window that spans a data outage.

X_nwp extends over the horizon because a weather forecast for the hours
being predicted is, by construction, available at issue time -- it is
the single most informative input for day-ahead solar, and a model that
only sees past weather has to extrapolate cloud cover blind. Only NWP is
extended: pv_norm and every feature derived from measured PV stop at the
issue time, so there is no target leakage.

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
        full_idx = kept_starts[:, None] + np.arange(total_span)[None, :]

        X_seq = seq_arr[hist_idx]
        X_sigma = sigma_arr[hist_idx]
        X_nwp = nwp_arr[full_idx]
        y = target_arr[hor_idx]
        meta = meta_arr[kept_starts + window_size - 1]  # snapshot at forecast issue time
        window_start_time = timestamps.to_numpy()[kept_starts]
    else:
        X_seq = np.empty((0, window_size, len(SEQUENCE_FEATURES)), dtype=np.float32)
        X_sigma = np.empty((0, window_size, len(SIGMA_FEATURES)), dtype=np.float32)
        X_nwp = np.empty((0, total_span, len(NWP_FEATURES)), dtype=np.float32)
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


def build_inference_window(hist_df: pd.DataFrame, future_nwp: pd.DataFrame,
                           window_size: int = DATA_CONFIG.window_size,
                           horizon: int = DATA_CONFIG.horizon,
                           freq_minutes: int = DATA_CONFIG.freq_minutes) -> dict:
    """
    One unscaled window for live forecasting, laid out exactly like a
    `create_windows` window but with no target:

      hist_df    engineered features (build_features output) ending at the
                 issue time; its last `window_size` rows are the lookback.
      future_nwp NWP_FEATURES for the `horizon` steps after the issue time.
    """
    hist = hist_df.sort_values("timestamp").tail(window_size).reset_index(drop=True)
    fut = future_nwp.sort_values("timestamp").head(horizon).reset_index(drop=True)
    if len(hist) < window_size:
        raise ValueError(f"Need {window_size} history steps, got {len(hist)}.")
    if len(fut) < horizon:
        raise ValueError(f"Need {horizon} forecast steps, got {len(fut)}.")

    ts = pd.concat([pd.to_datetime(hist["timestamp"]), pd.to_datetime(fut["timestamp"])],
                   ignore_index=True)
    if not (ts.diff().iloc[1:] == pd.Timedelta(minutes=freq_minutes)).all():
        raise ValueError("History + forecast timestamps are not a contiguous "
                         f"{freq_minutes}-minute grid.")

    nwp = np.vstack([hist[NWP_FEATURES].to_numpy(np.float32),
                     fut[NWP_FEATURES].to_numpy(np.float32)])
    return {
        "X_seq": hist[SEQUENCE_FEATURES].to_numpy(np.float32)[None],
        "X_sigma": hist[SIGMA_FEATURES].to_numpy(np.float32)[None],
        "X_nwp": nwp[None],
        "meta": hist[ROUTING_METADATA_FEATURES].to_numpy(np.float32)[-1:],
        "issue_time": ts.iloc[window_size - 1],
        "target_times": pd.DatetimeIndex(fut["timestamp"]),
    }
