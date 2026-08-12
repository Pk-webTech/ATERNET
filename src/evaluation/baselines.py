"""
Reference baselines.

A probabilistic forecasting result with no baseline is uninterpretable:
"nRMSE 6.2%" means nothing until you know what a trivial model scores
on the same split. Two baselines are provided, both standard in the
solar forecasting literature:

* Persistence -- repeat the last observed pv_norm across the whole
  horizon. The classic hard-to-beat short-horizon baseline.

* Smart (clear-sky) persistence -- hold the last observed *clear-sky
  index* constant and multiply it by the clear-sky expectation over the
  horizon, so the forecast still follows the deterministic diurnal
  curve. This is the fairer baseline: plain persistence is trivially
  beaten at dawn/dusk purely by knowing what time it is, which flatters
  any model that has time features.

Both take the already-scaled X_seq window, so the caller doesn't have
to re-derive anything from the raw feature table.
"""

import numpy as np

from src.utils.feature_groups import SEQUENCE_FEATURES

_PV_IDX = SEQUENCE_FEATURES.index("pv_norm")
_CSI_IDX = SEQUENCE_FEATURES.index("clear_sky_index")


def persistence_forecast(x_seq: np.ndarray, horizon: int) -> np.ndarray:
    """
    x_seq: (N, T, F) scaled sequence windows. Returns (N, horizon).

    NOTE: X_seq was min-max scaled in Phase 1 while y was not (pv_norm
    is already in [0,1]), so the pv_norm column must be un-scaled before
    it can be compared against y. The caller passes the scaler; see
    `persistence_from_dataset` for the wired-up version.
    """
    last = x_seq[:, -1, _PV_IDX]
    return np.repeat(last[:, None], horizon, axis=1)


def persistence_from_dataset(x_seq: np.ndarray, horizon: int, scaler: dict) -> np.ndarray:
    """
    Un-scale the pv_norm column of a scaled X_seq back to physical
    pv_norm units, then persist it across the horizon.
    """
    last_scaled = x_seq[:, -1, _PV_IDX]
    pv_min = np.asarray(scaler["min"]).ravel()[_PV_IDX]
    pv_scale = np.asarray(scaler["scale"]).ravel()[_PV_IDX]
    last = last_scaled * pv_scale + pv_min
    last = np.clip(last, 0.0, 1.0)
    return np.repeat(last[:, None], horizon, axis=1)


def climatology_forecast(y_train: np.ndarray, n: int, horizon: int) -> np.ndarray:
    """Per-horizon-step training mean, broadcast to n samples."""
    mean_curve = np.asarray(y_train, dtype=np.float64).mean(axis=0)
    return np.repeat(mean_curve[None, :], n, axis=0)
