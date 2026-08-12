"""
Centralized filesystem paths for the ATERNET project.
Every other module imports paths from here so that directory
locations only ever need to change in one place.
"""

from pathlib import Path

# Project root = two levels up from this file (config/paths.py -> ATERNET/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# ---- Data ----
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PV_DIR = RAW_DIR / "pv"
RAW_NWP_DIR = RAW_DIR / "nwp"
RAW_METADATA_DIR = RAW_DIR / "metadata"

INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# ---- Model / experiment artifacts ----
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

OUTPUTS_DIR = ROOT_DIR / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PLOTS_DIR = OUTPUTS_DIR / "plots"
ATTENTION_MAPS_DIR = OUTPUTS_DIR / "attention_maps"

LOGS_DIR = ROOT_DIR / "logs"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"

# ---- Phase 1 specific processed artifacts ----
PV_CLEAN_PATH = INTERIM_DIR / "pv_clean.parquet"
NWP_CLEAN_PATH = INTERIM_DIR / "nwp_clean.parquet"
FEATURES_PATH = INTERIM_DIR / "features.parquet"

WINDOWS_DIR = PROCESSED_DIR / "windows"
DATASET_TRAIN_PATH = PROCESSED_DIR / "dataset_train.npz"
DATASET_VAL_PATH = PROCESSED_DIR / "dataset_val.npz"
DATASET_TEST_PATH = PROCESSED_DIR / "dataset_test.npz"
SCALERS_PATH = PROCESSED_DIR / "scalers.pkl"
DATASET_METADATA_PATH = PROCESSED_DIR / "dataset_metadata.json"

# ---- Training / evaluation artifacts ----
BEST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "aternet_best.pt"
LAST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "aternet_last.pt"
TRAIN_HISTORY_PATH = METRICS_DIR / "train_history.json"
TEST_METRICS_PATH = METRICS_DIR / "test_metrics.json"
TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "test_predictions.npz"


def ensure_dirs():
    """Create every directory this project needs if it doesn't already exist."""
    dirs = [
        DATA_DIR, RAW_DIR, RAW_PV_DIR, RAW_NWP_DIR, RAW_METADATA_DIR,
        INTERIM_DIR, PROCESSED_DIR, CACHE_DIR, WINDOWS_DIR,
        CHECKPOINTS_DIR,
        OUTPUTS_DIR, PREDICTIONS_DIR, METRICS_DIR, REPORTS_DIR,
        PLOTS_DIR, ATTENTION_MAPS_DIR,
        LOGS_DIR, EXPERIMENTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs
