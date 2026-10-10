"""
Run all baselines, ablations, and conformal calibration, then build
the unified comparison table.

    python run_baselines.py [--epochs 50] [--skip-lgbm] [--skip-ablations]

Requires a trained ATERNET checkpoint and built Phase 1 data.
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.paths import BEST_CHECKPOINT_PATH, DATASET_TRAIN_PATH
from src.utils.logger import get_logger

logger = get_logger("run_baselines")


def _banner(text):
    logger.info("=" * 70)
    logger.info(text)
    logger.info("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(description="Run all baselines and ablations.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--skip-lgbm", action="store_true",
                   help="Skip LightGBM (requires lightgbm package)")
    p.add_argument("--skip-ablations", action="store_true")
    p.add_argument("--skip-conformal", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not DATASET_TRAIN_PATH.exists():
        logger.error("Phase 1 data not built. Run: python run_pipeline.py")
        sys.exit(1)
    if not BEST_CHECKPOINT_PATH.exists():
        logger.error("No trained ATERNET checkpoint. Run: python -m src.training.train")
        sys.exit(1)

    # ---- 1. LightGBM baseline ----
    if not args.skip_lgbm:
        _banner("Baseline 1/3 — LightGBM Quantile Regression")
        try:
            from src.baselines.lgbm_quantile import train_and_evaluate
            train_and_evaluate()
        except ImportError as e:
            logger.warning(f"Skipping LightGBM: {e}")
            logger.warning("Install with: pip install lightgbm --break-system-packages")

    # ---- 2. LSTM baseline ----
    _banner("Baseline 2/3 — LSTM Quantile Regression")
    from src.baselines.lstm_quantile import train_and_evaluate as lstm_eval
    lstm_eval(epochs=args.epochs)

    # ---- 3. PatchTST baseline ----
    _banner("Baseline 3/3 — Vanilla PatchTST Quantile")
    from src.baselines.patchtst_quantile import train_and_evaluate as ptst_eval
    ptst_eval(epochs=args.epochs)

    # ---- 4. Ablations ----
    if not args.skip_ablations:
        _banner("Ablation Studies (6 variants)")
        from src.baselines.ablations import run_all
        run_all(epochs=args.epochs)

    # ---- 5. Conformal calibration ----
    if not args.skip_conformal:
        _banner("Per-Regime Conformal Calibration (CQR)")
        from src.evaluation.conformal import calibrate
        calibrate()

    # ---- 6. Comparison table ----
    _banner("Building comparison table")
    from src.evaluation.comparison_table import generate
    rows = generate()

    if rows:
        logger.info("\n--- SUMMARY ---")
        for r in rows:
            logger.info(f"  {r['model']:40s} nRMSE={r.get('nrmse_pct', 0):.2f}%  "
                        f"PICP={r.get('picp', 0):.3f}  "
                        f"Winkler={r.get('winkler', 0):.4f}")

    logger.info("\nAll baselines, ablations, and calibration complete.")
    logger.info("Results in outputs/metrics/  |  Table in outputs/reports/comparison_table.md")


if __name__ == "__main__":
    main()