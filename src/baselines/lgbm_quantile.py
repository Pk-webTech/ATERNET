"""
Baseline 1 -- LightGBM Quantile Regression.

A strong non-deep-learning baseline. One LightGBM model per quantile
level (0.1, 0.5, 0.9), each predicting the full horizon vector.
Input: flattened last-step features from the window (no temporal
modelling — that is the point: it isolates how much the temporal
architecture buys).

Usage:
    python -m src.baselines.lgbm_quantile
"""

import numpy as np
from pathlib import Path

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from config.paths import METRICS_DIR, ensure_dirs
from src.training.dataset import ATERNETDataset
from src.evaluation.metrics import all_metrics, per_horizon_metrics
from src.evaluation.baselines import persistence_from_dataset
from src.utils.io import load_pickle, save_json
from src.utils.logger import get_logger
from config.paths import SCALERS_PATH

logger = get_logger(__name__)

QUANTILES = [0.1, 0.5, 0.9]


def _flatten_features(ds: ATERNETDataset) -> np.ndarray:
    """
    Flatten each window into a single feature vector.
    Uses last-step sequence + last-step sigma + last-step nwp + meta.
    """
    x_seq_last = ds.x_seq[:, -1, :].numpy()       # (N, 10)
    x_sigma_last = ds.x_sigma[:, -1, :].numpy()    # (N, 6)
    x_nwp_last = ds.x_nwp[:, -1, :].numpy()        # (N, 4)
    meta = ds.meta.numpy()                          # (N, 9)
    return np.hstack([x_seq_last, x_sigma_last, x_nwp_last, meta])


def train_and_evaluate():
    if not HAS_LGB:
        raise ImportError(
            "lightgbm is not installed. Run: pip install lightgbm --break-system-packages"
        )

    ensure_dirs()
    train_ds = ATERNETDataset("train")
    val_ds = ATERNETDataset("val")
    test_ds = ATERNETDataset("test")

    X_train = _flatten_features(train_ds)
    X_val = _flatten_features(val_ds)
    X_test = _flatten_features(test_ds)
    y_train = train_ds.y.numpy()    # (N, horizon)
    y_val = val_ds.y.numpy()
    y_test = test_ds.y.numpy()
    horizon = y_train.shape[1]

    predictions = {}

    for tau in QUANTILES:
        tag = f"p{int(tau*100)}"
        logger.info(f"Training LightGBM for quantile {tau} ({tag}) ...")

        preds_test = np.zeros((len(X_test), horizon), dtype=np.float64)
        preds_val = np.zeros((len(X_val), horizon), dtype=np.float64)

        for h in range(horizon):
            dtrain = lgb.Dataset(X_train, label=y_train[:, h])
            dval = lgb.Dataset(X_val, label=y_val[:, h], reference=dtrain)

            params = {
                "objective": "quantile",
                "alpha": tau,
                "metric": "quantile",
                "num_leaves": 63,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "seed": 42,
            }

            model = lgb.train(
                params, dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            preds_test[:, h] = model.predict(X_test)
            preds_val[:, h] = model.predict(X_val)

        # Clip to [0, 1] and enforce non-crossing
        predictions[tag] = np.clip(preds_test, 0.0, 1.0)

    # Enforce non-crossing: p10 <= p50 <= p90
    p10 = np.minimum(predictions["p10"], predictions["p50"])
    p90 = np.maximum(predictions["p90"], predictions["p50"])
    p50 = predictions["p50"]

    # Evaluate
    scalers = load_pickle(SCALERS_PATH)
    baseline = persistence_from_dataset(test_ds.x_seq.numpy(), horizon, scalers["X_seq"])
    headline = all_metrics(y_test, p10, p50, p90, y_baseline=baseline)
    per_h = per_horizon_metrics(y_test, p50, p10, p90)

    # Per-regime
    from src.utils.constants import ID_TO_REGIME
    regime_ids = test_ds.regime_id.numpy()
    per_regime = {}
    for rid, rname in ID_TO_REGIME.items():
        mask = regime_ids == rid
        if mask.sum() == 0:
            per_regime[rname] = {"n": 0}
            continue
        m = all_metrics(y_test[mask], p10[mask], p50[mask], p90[mask],
                        y_baseline=baseline[mask])
        m["n"] = int(mask.sum())
        per_regime[rname] = m

    results = {
        "model": "LightGBM Quantile",
        "headline": headline,
        "per_regime": per_regime,
        "per_horizon": per_h,
    }

    out_path = METRICS_DIR / "baseline_lgbm.json"
    save_json(results, out_path)
    logger.info(f"LightGBM baseline results -> {out_path}")
    logger.info(f"  nRMSE {headline['nrmse_pct']:.2f}% | PICP {headline['picp']:.3f} "
                f"| Winkler {headline['winkler']:.4f}")
    return results


if __name__ == "__main__":
    train_and_evaluate()