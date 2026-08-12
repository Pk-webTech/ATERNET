"""
Phase 1 -- Full pipeline orchestrator.

Runs, in order:
  1. download_pv        (skip if data/raw/pv/pv_raw.parquet already exists)
  2. download_nwp        (skip if data/raw/nwp/nwp_ensemble.parquet exists)
  3. pv_processor        -> data/interim/pv_clean.parquet
  4. nwp_processor        -> data/interim/nwp_clean.parquet
  5. feature_engineering  -> data/interim/features.parquet
  6. create_windows       -> in-memory tensors
  7. chronological train/val/test split
  8. fit scalers on TRAIN ONLY, transform all splits (prevents leakage)
  9. save data/processed/dataset_{train,val,test}.npz + scalers.pkl + metadata.json

Idempotent: re-running with existing raw files only regenerates the
interim/processed artifacts, so you can tweak feature engineering without
re-synthesizing the raw PV/NWP series (which are seeded and would be
identical anyway).
"""

import numpy as np

from config.paths import (
    RAW_PV_DIR, RAW_NWP_DIR, PV_CLEAN_PATH, NWP_CLEAN_PATH, FEATURES_PATH,
    DATASET_TRAIN_PATH, DATASET_VAL_PATH, DATASET_TEST_PATH,
    SCALERS_PATH, DATASET_METADATA_PATH, ensure_dirs,
)
from config.model_config import DATA_CONFIG
from config.training_config import SPLIT_CONFIG, TRAINING_CONFIG
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.io import load_parquet, save_pickle, save_npz, save_json
from src.utils.helpers import chronological_split, minmax_fit, minmax_transform
from src.utils.feature_groups import (
    SEQUENCE_FEATURES, SIGMA_FEATURES, NWP_FEATURES,
    ROUTING_METADATA_FEATURES, VOLATILITY_FEATURES,
)

from src.phase1 import download_pv, download_nwp, pv_processor, nwp_processor, feature_engineering
from src.phase1.create_windows import create_windows

logger = get_logger(__name__)


def _run_acquisition_if_needed():
    pv_raw_path = RAW_PV_DIR / "pv_raw.parquet"
    if not pv_raw_path.exists():
        logger.info("Raw PV data not found -- generating synthetic series.")
        download_pv.main()
    else:
        logger.info(f"Raw PV data already exists at {pv_raw_path}, skipping generation.")

    nwp_raw_path = RAW_NWP_DIR / "nwp_ensemble.parquet"
    if not nwp_raw_path.exists():
        logger.info("Raw NWP ensemble not found -- generating synthetic ensemble.")
        download_nwp.main()
    else:
        logger.info(f"Raw NWP data already exists at {nwp_raw_path}, skipping generation.")


def _scale_split(arr: np.ndarray, scaler: dict) -> np.ndarray:
    """Apply a per-last-axis minmax scaler to a (N, T, F) or (N, F) array."""
    original_shape = arr.shape
    flat = arr.reshape(-1, original_shape[-1])
    scaled = minmax_transform(flat, scaler)
    return scaled.reshape(original_shape).astype(np.float32)


def build_dataset():
    ensure_dirs()
    set_seed(TRAINING_CONFIG.seed)

    _run_acquisition_if_needed()

    logger.info("Running pv_processor ...")
    pv_processor.main()

    logger.info("Running nwp_processor ...")
    nwp_processor.main()

    logger.info("Running feature_engineering ...")
    feature_engineering.main()

    logger.info("Loading engineered features and generating sliding windows ...")
    features_df = load_parquet(FEATURES_PATH)
    windows = create_windows(features_df)

    n_windows = windows["X_seq"].shape[0]
    if n_windows < 10:
        raise RuntimeError(
            f"Only {n_windows} windows were generated -- check window_size/horizon "
            f"against the length of the synthesized series."
        )

    train_idx, val_idx, test_idx = chronological_split(
        n_windows, SPLIT_CONFIG.train_ratio, SPLIT_CONFIG.val_ratio
    )
    logger.info(f"Chronological split -> train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}.")

    # ---- Fit scalers on TRAIN split only ----
    scalers = {
        "X_seq": minmax_fit(windows["X_seq"][train_idx].reshape(-1, len(SEQUENCE_FEATURES))),
        "X_sigma": minmax_fit(windows["X_sigma"][train_idx].reshape(-1, len(SIGMA_FEATURES))),
        "X_nwp": minmax_fit(windows["X_nwp"][train_idx].reshape(-1, len(NWP_FEATURES))),
    }

    # Only scale the continuous volatility columns within meta; regime one-hot
    # / regime_id columns are left untouched since they are already categorical.
    vol_start = len(ROUTING_METADATA_FEATURES) - len(VOLATILITY_FEATURES)
    vol_scaler = minmax_fit(windows["meta"][train_idx][:, vol_start:])
    scalers["meta_volatility"] = vol_scaler

    def build_split(idx, name):
        meta = windows["meta"][idx].copy()
        meta[:, vol_start:] = minmax_transform(meta[:, vol_start:], vol_scaler)

        split_dict = {
            "X_seq": _scale_split(windows["X_seq"][idx], scalers["X_seq"]),
            "X_sigma": _scale_split(windows["X_sigma"][idx], scalers["X_sigma"]),
            "X_nwp": _scale_split(windows["X_nwp"][idx], scalers["X_nwp"]),
            "y": windows["y"][idx],  # pv_norm already in [0,1], no rescale needed
            "meta": meta.astype(np.float32),
            "window_start_time": windows["window_start_time"][idx],
        }
        logger.info(f"{name} split shapes: " +
                    ", ".join(f"{k}={v.shape}" for k, v in split_dict.items()))
        return split_dict

    train_split = build_split(train_idx, "train")
    val_split = build_split(val_idx, "val")
    test_split = build_split(test_idx, "test")

    save_npz(DATASET_TRAIN_PATH, **train_split)
    save_npz(DATASET_VAL_PATH, **val_split)
    save_npz(DATASET_TEST_PATH, **test_split)
    save_pickle(scalers, SCALERS_PATH)

    metadata = {
        "n_windows_total": int(n_windows),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "window_size": DATA_CONFIG.window_size,
        "horizon": DATA_CONFIG.horizon,
        "stride": DATA_CONFIG.stride,
        "freq_minutes": DATA_CONFIG.freq_minutes,
        "sequence_features": SEQUENCE_FEATURES,
        "sigma_features": SIGMA_FEATURES,
        "nwp_features": NWP_FEATURES,
        "routing_metadata_features": ROUTING_METADATA_FEATURES,
        "seed": TRAINING_CONFIG.seed,
    }
    save_json(metadata, DATASET_METADATA_PATH)

    logger.info(f"Phase 1 dataset build complete. Metadata written to {DATASET_METADATA_PATH}.")
    return metadata


def main():
    return build_dataset()


if __name__ == "__main__":
    main()
