"""
Forecast metrics -- deterministic and probabilistic.

All functions take numpy arrays of shape (N, horizon) and return plain
floats (or per-step arrays where noted), so they are usable from tests,
notebooks, and the evaluation script alike without a torch dependency.

Metric choices and why:

* MAE / RMSE on P50 -- the standard deterministic pair. RMSE punishes
  large misses more, MAE is in the same units as the target and is the
  honest "typical error". Reporting both is standard in the solar
  forecasting literature and they disagree in an informative way when
  errors are heavy-tailed (as they are in the volatile regime).

* nMAE / nRMSE -- the same, divided by rated capacity. Because targets
  are already normalized to [0,1] against nameplate, this is just the
  raw value expressed as a percentage of capacity, which is the unit
  grid operators actually specify tolerances in.

* PICP (Prediction Interval Coverage Probability) -- the fraction of
  observations that land inside [P10, P90]. The nominal target is 0.80.
  This is THE calibration metric: a model whose intervals are beautiful
  but only cover 55% of outcomes is not a probabilistic forecast, it's
  a decorated point forecast.

* PINAW (Prediction Interval Normalized Average Width) -- mean interval
  width. Sharpness. Only meaningful when read *together with* PICP:
  either number alone is trivially gameable.

* Winkler score -- the proper scoring rule that combines the two, so
  there is a single number to rank models by that cannot be gamed by
  trading coverage for sharpness.

* Pinball loss -- reported per quantile, since it is the training
  objective and a per-quantile breakdown shows *which* tail is
  miscalibrated.
"""

from typing import Dict

import numpy as np


def _flat(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_flat(y_true) - _flat(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((_flat(y_true) - _flat(y_pred)) ** 2)))


def bias(y_true, y_pred) -> float:
    """Mean signed error. Positive = model under-forecasts."""
    return float(np.mean(_flat(y_true) - _flat(y_pred)))


def r2_score(y_true, y_pred) -> float:
    yt, yp = _flat(y_true), _flat(y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def skill_score(y_true, y_pred, y_baseline) -> float:
    """
    RMSE skill vs a baseline: 1 - RMSE_model / RMSE_baseline.
    Positive means the model beats the baseline; 0 means it ties.
    """
    r_b = rmse(y_true, y_baseline)
    return float(1.0 - rmse(y_true, y_pred) / r_b) if r_b > 0 else float("nan")


def pinball(y_true, y_pred, tau: float) -> float:
    err = _flat(y_true) - _flat(y_pred)
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


def picp(y_true, lower, upper) -> float:
    """Prediction Interval Coverage Probability. Nominal 0.80 for P10/P90."""
    yt, lo, up = _flat(y_true), _flat(lower), _flat(upper)
    return float(np.mean((yt >= lo) & (yt <= up)))


def pinaw(lower, upper) -> float:
    """Prediction Interval Normalized Average Width (targets are already in [0,1])."""
    return float(np.mean(_flat(upper) - _flat(lower)))


def ace(y_true, lower, upper, nominal: float = 0.80) -> float:
    """Average Coverage Error: PICP - nominal. Signed; 0 is perfect."""
    return float(picp(y_true, lower, upper) - nominal)


def winkler(y_true, lower, upper, alpha: float = 0.2) -> float:
    yt, lo, up = _flat(y_true), _flat(lower), _flat(upper)
    width = up - lo
    below = np.maximum(lo - yt, 0.0)
    above = np.maximum(yt - up, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def crps_from_quantiles(y_true, p10, p50, p90) -> float:
    """
    Approximate CRPS as the mean pinball loss over the available
    quantile levels times 2. With only three levels this is a coarse
    approximation of the true integral, so it is reported as an
    indicative number rather than a headline metric.
    """
    return float(2.0 * np.mean([
        pinball(y_true, p10, 0.1),
        pinball(y_true, p50, 0.5),
        pinball(y_true, p90, 0.9),
    ]))


def per_horizon_metrics(y_true, p50, p10=None, p90=None) -> Dict[str, list]:
    """
    Metrics computed independently at each horizon step, so error growth
    with lead time is visible (it always grows; the question is how fast).
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(p50, dtype=np.float64)
    out = {
        "mae": [float(np.mean(np.abs(yt[:, h] - yp[:, h]))) for h in range(yt.shape[1])],
        "rmse": [float(np.sqrt(np.mean((yt[:, h] - yp[:, h]) ** 2))) for h in range(yt.shape[1])],
    }
    if p10 is not None and p90 is not None:
        lo = np.asarray(p10, dtype=np.float64)
        up = np.asarray(p90, dtype=np.float64)
        out["picp"] = [float(np.mean((yt[:, h] >= lo[:, h]) & (yt[:, h] <= up[:, h])))
                       for h in range(yt.shape[1])]
        out["pinaw"] = [float(np.mean(up[:, h] - lo[:, h])) for h in range(yt.shape[1])]
    return out


def all_metrics(y_true, p10, p50, p90, y_baseline=None, nominal: float = 0.80) -> Dict[str, float]:
    m = {
        "mae": mae(y_true, p50),
        "rmse": rmse(y_true, p50),
        "bias": bias(y_true, p50),
        "r2": r2_score(y_true, p50),
        "nmae_pct": 100.0 * mae(y_true, p50),
        "nrmse_pct": 100.0 * rmse(y_true, p50),
        "pinball_p10": pinball(y_true, p10, 0.1),
        "pinball_p50": pinball(y_true, p50, 0.5),
        "pinball_p90": pinball(y_true, p90, 0.9),
        "picp": picp(y_true, p10, p90),
        "pinaw": pinaw(p10, p90),
        "ace": ace(y_true, p10, p90, nominal),
        "winkler": winkler(y_true, p10, p90, alpha=1.0 - nominal),
        "crps_approx": crps_from_quantiles(y_true, p10, p50, p90),
    }
    m["pinball_total"] = m["pinball_p10"] + m["pinball_p50"] + m["pinball_p90"]
    if y_baseline is not None:
        m["rmse_skill_vs_persistence"] = skill_score(y_true, p50, y_baseline)
        m["baseline_rmse"] = rmse(y_true, y_baseline)
        m["baseline_mae"] = mae(y_true, y_baseline)
    return m
