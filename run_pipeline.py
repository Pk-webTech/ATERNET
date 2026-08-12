"""
Top-level entry point for the ATERNET Phase 1 data pipeline.

Usage (from the project root, ATERNET/):
    python run_pipeline.py

This will:
  1. Synthesize / load raw PV + NWP ensemble data
  2. Clean & normalize PV, aggregate NWP ensemble statistics
  3. Engineer features (clear-sky index, regimes, volatility, time features)
  4. Generate sliding windows
  5. Split chronologically into train/val/test
  6. Fit scalers on train only and save the final processed dataset

Outputs land in data/processed/ (dataset_{train,val,test}.npz, scalers.pkl,
dataset_metadata.json) and are ready to be consumed by Phase 2 onward.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path regardless of the caller's CWD.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger
from src.phase1.build_dataset import build_dataset

logger = get_logger("run_pipeline")


def main():
    logger.info("=" * 70)
    logger.info("ATERNET — Phase 1 Data Pipeline")
    logger.info("=" * 70)

    metadata = build_dataset()

    logger.info("=" * 70)
    logger.info("Phase 1 complete.")
    logger.info(f"  Total windows : {metadata['n_windows_total']}")
    logger.info(f"  Train / Val / Test : {metadata['n_train']} / {metadata['n_val']} / {metadata['n_test']}")
    logger.info(f"  Window size (steps) : {metadata['window_size']}  |  Horizon : {metadata['horizon']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
