"""
Generate every diagnostic figure from saved training history and
evaluation predictions.

    python -m src.visualization.generate_report
"""

import numpy as np
import torch

from config.paths import (
    TRAIN_HISTORY_PATH, TEST_METRICS_PATH, TEST_PREDICTIONS_PATH,
    BEST_CHECKPOINT_PATH, ensure_dirs,
)
from src.models.aternet import ATERNET
from src.training.checkpoint import load_checkpoint
from src.training.dataset import ATERNETDataset
from src.utils.io import load_json, load_npz
from src.utils.logger import get_logger
from src.visualization import plots

logger = get_logger(__name__)


def main(split: str = "test"):
    ensure_dirs()
    written = []

    history = load_json(TRAIN_HISTORY_PATH)
    written.append(plots.plot_training_history(history))
    written.append(plots.plot_gate_share_history(history))

    results = load_json(TEST_METRICS_PATH)
    p = load_npz(TEST_PREDICTIONS_PATH)

    y, p10, p50, p90 = p["y_true"], p["p10"], p["p50"], p["p90"]
    written.append(plots.plot_forecast_examples(y, p10, p50, p90))
    written.append(plots.plot_reliability(y, p10, p50, p90))
    written.append(plots.plot_per_horizon(results["per_horizon"]))

    abs_err = np.abs(y - p50).mean(axis=1)
    width = (p90 - p10).mean(axis=1)
    written.append(plots.plot_uncertainty_vs_error(p["sigma_total"], width, abs_err))
    written.append(plots.plot_gate_by_regime(p["gate_weights"], p["regime_id"]))

    # Attention map needs a live forward pass (weights aren't cached)
    try:
        device = torch.device("cpu")
        ds = ATERNETDataset(split)
        model, _ = load_checkpoint(BEST_CHECKPOINT_PATH, model_class=ATERNET, map_location=device)
        model.eval()
        with torch.no_grad():
            out = model(ds.x_seq[:8], ds.x_nwp[:8], ds.x_sigma[:8], return_attention=True)
        written.append(plots.plot_attention_map(out["nwp_attention_weights"].numpy(), index=0))
    except Exception as e:  # plotting must never break the pipeline
        logger.warning(f"Skipped attention map: {e}")

    for path in written:
        logger.info(f"  wrote {path}")
    return written


if __name__ == "__main__":
    main()
