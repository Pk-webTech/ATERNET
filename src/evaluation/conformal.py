"""
Per-regime Conformal Quantile Regression (CQR).

Fixes the miscalibrated per-regime coverage without retraining.
The method: on the validation set, compute per-regime conformity
scores, then adjust P10/P90 on the test set by the regime-specific
quantile of those scores.

After calibration, each regime's PICP should be close to the nominal
0.80. Scores are computed per forecast step (see `_conformity_scores`);
note that a regime with few validation windows (volatile: 44) gives a
noisy adjustment, and validation and test seasons differ, so exact
nominal coverage is not guaranteed on the test split.

Usage:
    python -m src.evaluation.conformal
"""

import numpy as np
import torch

from config.paths import (
    BEST_CHECKPOINT_PATH, SCALERS_PATH, METRICS_DIR, ensure_dirs,
)
from src.models.aternet import ATERNET
from src.training.checkpoint import load_checkpoint
from src.training.dataset import ATERNETDataset
from src.evaluation.evaluate import predict
from src.evaluation.metrics import all_metrics, per_horizon_metrics
from src.evaluation.baselines import persistence_from_dataset
from src.utils.constants import ID_TO_REGIME
from src.utils.io import save_json, load_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _conformity_scores(y: np.ndarray, p10: np.ndarray, p90: np.ndarray,
                       score: str = "pointwise") -> np.ndarray:
    """
    CQR conformity score, E = max( p10 - y, y - p90 ).

    "pointwise" (default): one score per (window, horizon step), so the
        calibrated quantile targets the same per-point coverage (PICP) that
        is reported.
    "window_mean": the original per-window average over steps. It averages
        signed violations, and since most steps sit inside the interval
        (negative values) the resulting adjustment is ~0 or even negative,
        which left clear_peak under-covered and shrank the volatile
        interval. Kept only to reproduce earlier results.
    """
    if score == "pointwise":
        return np.maximum(p10 - y, y - p90).ravel()
    if score == "window_mean":
        return np.maximum((p10 - y).mean(axis=1), (y - p90).mean(axis=1))
    raise ValueError(f"Unknown score '{score}'.")


def calibrate(nominal: float = 0.80, checkpoint_path=BEST_CHECKPOINT_PATH,
              score: str = "pointwise"):
    """
    1. Run ATERNET on val and test.
    2. Compute per-regime conformity scores on val.
    3. Adjust P10/P90 on test using per-regime quantiles.
    4. Report calibrated metrics.
    """
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = ATERNETDataset("val")
    test_ds = ATERNETDataset("test")

    model, payload = load_checkpoint(checkpoint_path, model_class=ATERNET,
                                     map_location=device)
    model = model.to(device)

    val_preds = predict(model, val_ds, device)
    test_preds = predict(model, test_ds, device)

    y_val = val_ds.y.numpy()
    y_test = test_ds.y.numpy()
    val_regimes = val_ds.regime_id.numpy()
    test_regimes = test_ds.regime_id.numpy()

    alpha = 1.0 - nominal  # 0.20

    # Compute per-regime adjustment
    adjustments = {}
    for rid, rname in ID_TO_REGIME.items():
        mask = val_regimes == rid
        if mask.sum() < 5:
            adjustments[rid] = 0.0
            logger.warning(f"Regime {rname}: too few val samples ({mask.sum()}), no adjustment.")
            continue

        scores = _conformity_scores(y_val[mask], val_preds["p10"][mask],
                                    val_preds["p90"][mask], score=score)
        n = len(scores)
        # CQR quantile level: ceil((n+1)(1-alpha)) / n
        q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        q_hat = float(np.quantile(scores, q_level))
        adjustments[rid] = q_hat
        logger.info(f"Regime {rname}: n_val={n}, q_hat={q_hat:.5f}")

    # Apply adjustments to test predictions
    p10_cal = test_preds["p10"].copy()
    p90_cal = test_preds["p90"].copy()
    p50 = test_preds["p50"]

    for rid, q_hat in adjustments.items():
        mask = test_regimes == rid
        p10_cal[mask] = p10_cal[mask] - q_hat
        p90_cal[mask] = p90_cal[mask] + q_hat

    # Clip to [0, 1] and enforce non-crossing
    p10_cal = np.clip(p10_cal, 0.0, 1.0)
    p90_cal = np.clip(p90_cal, 0.0, 1.0)
    p10_cal = np.minimum(p10_cal, p50)
    p90_cal = np.maximum(p90_cal, p50)

    # Evaluate before and after
    scalers = load_pickle(SCALERS_PATH)
    baseline = persistence_from_dataset(test_ds.x_seq.numpy(), test_ds.horizon,
                                        scalers["X_seq"])

    uncal_headline = all_metrics(y_test, test_preds["p10"], p50,
                                  test_preds["p90"], y_baseline=baseline)
    cal_headline = all_metrics(y_test, p10_cal, p50, p90_cal, y_baseline=baseline)
    cal_per_h = per_horizon_metrics(y_test, p50, p10_cal, p90_cal)

    # Per-regime comparison
    comparison = {}
    for rid, rname in ID_TO_REGIME.items():
        mask = test_regimes == rid
        if mask.sum() == 0:
            continue

        m_before = all_metrics(y_test[mask], test_preds["p10"][mask], p50[mask],
                               test_preds["p90"][mask])
        m_after = all_metrics(y_test[mask], p10_cal[mask], p50[mask], p90_cal[mask])
        comparison[rname] = {
            "n": int(mask.sum()),
            "picp_before": m_before["picp"],
            "picp_after": m_after["picp"],
            "pinaw_before": m_before["pinaw"],
            "pinaw_after": m_after["pinaw"],
            "winkler_before": m_before["winkler"],
            "winkler_after": m_after["winkler"],
            "adjustment_q_hat": adjustments.get(rid, 0.0),
        }
        logger.info(f"  {rname:12s}: PICP {m_before['picp']:.3f} -> {m_after['picp']:.3f} "
                    f"| PINAW {m_before['pinaw']:.4f} -> {m_after['pinaw']:.4f}")

    results = {
        "model": "ATERNET + CQR (per-regime)",
        "nominal": nominal,
        "adjustments": {ID_TO_REGIME[rid]: q for rid, q in adjustments.items()},
        "headline_before": uncal_headline,
        "headline_after": cal_headline,
        "per_regime_comparison": comparison,
        "per_horizon_calibrated": cal_per_h,
    }

    out_path = METRICS_DIR / "conformal_calibration.json"
    save_json(results, out_path)

    logger.info(f"\nAggregate PICP: {uncal_headline['picp']:.3f} -> {cal_headline['picp']:.3f}")
    logger.info(f"Aggregate Winkler: {uncal_headline['winkler']:.4f} -> {cal_headline['winkler']:.4f}")
    logger.info(f"Results -> {out_path}")
    return results


if __name__ == "__main__":
    calibrate()