"""
Run the whole ATERNET pipeline end to end.

    python run_all.py [--epochs 50] [--skip-data] [--skip-train]

Stages: data pipeline -> training -> evaluation -> figures.
Each stage is skippable so you can re-evaluate or re-plot without
retraining.
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.paths import BEST_CHECKPOINT_PATH, DATASET_TRAIN_PATH
from src.utils.logger import get_logger

logger = get_logger("run_all")


def parse_args():
    p = argparse.ArgumentParser(description="Run the full ATERNET pipeline.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def _banner(text):
    logger.info("=" * 70)
    logger.info(text)
    logger.info("=" * 70)


def main():
    args = parse_args()

    # ---- 1. Data ----
    if args.skip_data and DATASET_TRAIN_PATH.exists():
        logger.info("Skipping Phase 1 (dataset already built).")
    else:
        _banner("Stage 1/4 - Phase 1 data pipeline")
        from src.phase1.build_dataset import build_dataset
        meta = build_dataset()
        logger.info(f"  {meta['n_windows_total']} windows "
                    f"({meta['n_train']}/{meta['n_val']}/{meta['n_test']})")

    # ---- 2. Train ----
    if args.skip_train and BEST_CHECKPOINT_PATH.exists():
        logger.info("Skipping training (checkpoint already exists).")
    else:
        _banner(f"Stage 2/4 - Training ({args.epochs} epochs max)")
        import torch
        from config.training_config import TRAINING_CONFIG
        from src.models.aternet import ATERNET
        from src.training.dataset import build_dataloaders
        from src.training.trainer import Trainer
        from src.utils.seed import set_seed

        set_seed(TRAINING_CONFIG.seed)
        TRAINING_CONFIG.epochs = args.epochs
        train_loader, val_loader, _, train_ds = build_dataloaders(batch_size=args.batch_size)
        model_kwargs = dict(
            n_seq_features=train_ds.n_seq_features,
            n_nwp_features=train_ds.n_nwp_features,
            n_sigma_features=train_ds.n_sigma_features,
            horizon=train_ds.horizon,
        )
        trainer = Trainer(ATERNET(**model_kwargs), train_loader, val_loader,
                          cfg=TRAINING_CONFIG, model_kwargs=model_kwargs)
        trainer.fit(epochs=args.epochs)

    # ---- 3. Evaluate ----
    _banner("Stage 3/4 - Evaluation (test split)")
    from src.evaluation.evaluate import evaluate
    results = evaluate("test")

    # ---- 4. Figures ----
    if not args.skip_plots:
        _banner("Stage 4/4 - Diagnostic figures")
        from src.visualization.generate_report import main as make_plots
        make_plots()

    h = results["headline"]
    _banner("ATERNET pipeline complete")
    logger.info(f"  nRMSE {h['nrmse_pct']:.2f}%  |  nMAE {h['nmae_pct']:.2f}%  "
                f"|  R2 {h['r2']:.4f}")
    logger.info(f"  PICP {h['picp']:.3f} (nominal 0.80)  |  PINAW {h['pinaw']:.4f}  "
                f"|  Winkler {h['winkler']:.4f}")
    logger.info(f"  Skill vs persistence: {h['rmse_skill_vs_persistence']:.3f}")
    logger.info(f"  Non-crossing: {'PASS' if results['non_crossing_ok'] else 'FAIL'}")
    logger.info("  Report -> outputs/reports/evaluation_report.md")
    return results


if __name__ == "__main__":
    main()
