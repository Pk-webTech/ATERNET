"""
Build an alternative processed dataset with a denser TRAIN split.

    python -m src.phase1.densify_train --stride 2 --out data/processed_stride2

Why not just rebuild with a smaller stride: the chronological split is by
window index, so changing the stride would move the validation/test
boundaries and change which windows are scored. Every comparison so far
would then be against a different test set.

Instead this keeps the existing validation and test splits and scalers
byte-for-byte, and only replaces the train split with extra (overlapping)
windows taken from the same training period -- i.e. every window that
starts before the first validation window. Existing train windows are a
subset of the new ones. Same convention as the original split: windows near
the boundary may overlap the next split's lookback.

Point training/evaluation at it with ATERNET_PROCESSED_DIR=<out>.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from config.paths import (
    DATA_DIR, FEATURES_PATH, DATASET_TRAIN_PATH, DATASET_VAL_PATH,
    DATASET_TEST_PATH, SCALERS_PATH, DATASET_METADATA_PATH,
)
from src.phase1.create_windows import create_windows
from src.utils.feature_groups import (
    ROUTING_METADATA_FEATURES, VOLATILITY_FEATURES, SEQUENCE_FEATURES,
    SIGMA_FEATURES, NWP_FEATURES,
)
from src.utils.helpers import minmax_transform
from src.utils.io import load_npz, load_parquet, load_pickle, save_json, load_json, save_npz
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Always the original dataset, regardless of ATERNET_PROCESSED_DIR.
SRC_DIR = DATA_DIR / "processed"


def _scale(arr, scaler):
    shape = arr.shape
    return minmax_transform(arr.reshape(-1, shape[-1]), scaler).reshape(shape).astype(np.float32)


def densify(stride: int, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_train = load_npz(SRC_DIR / DATASET_TRAIN_PATH.name)
    orig_val = load_npz(SRC_DIR / DATASET_VAL_PATH.name)
    scalers = load_pickle(SRC_DIR / SCALERS_PATH.name)

    windows = create_windows(load_parquet(FEATURES_PATH), stride=stride)
    cutoff = np.min(orig_val["window_start_time"])
    keep = windows["window_start_time"] < cutoff
    logger.info(f"stride {stride}: {int(keep.sum())} train windows start before the first "
                f"validation window ({cutoff}); original train has {len(orig_train['y'])}.")

    # Sanity: the original train windows must be a subset of the new ones,
    # otherwise the features file no longer matches the stored dataset.
    new_times = set(windows["window_start_time"][keep].tolist())
    missing = [t for t in orig_train["window_start_time"].tolist() if t not in new_times]
    if missing:
        raise RuntimeError(f"{len(missing)} original train windows are absent from the "
                           f"stride-{stride} set; features.parquet differs from the dataset.")

    vol_start = len(ROUTING_METADATA_FEATURES) - len(VOLATILITY_FEATURES)
    meta = windows["meta"][keep].copy()
    meta[:, vol_start:] = minmax_transform(meta[:, vol_start:], scalers["meta_volatility"])
    train = {
        "X_seq": _scale(windows["X_seq"][keep], scalers["X_seq"]),
        "X_sigma": _scale(windows["X_sigma"][keep], scalers["X_sigma"]),
        "X_nwp": _scale(windows["X_nwp"][keep], scalers["X_nwp"]),
        "y": windows["y"][keep],
        "meta": meta.astype(np.float32),
        "window_start_time": windows["window_start_time"][keep],
    }

    # Consistency check on the overlap: identical windows must scale identically.
    idx = {t: i for i, t in enumerate(windows["window_start_time"][keep].tolist())}
    probe = [idx[t] for t in orig_train["window_start_time"][:50].tolist()]
    if not np.allclose(train["X_seq"][probe], orig_train["X_seq"][:50], atol=1e-5):
        raise RuntimeError("Re-created windows do not match the stored train windows.")

    save_npz(out_dir / DATASET_TRAIN_PATH.name, **train)
    for src in (DATASET_VAL_PATH, DATASET_TEST_PATH, SCALERS_PATH):
        shutil.copy2(SRC_DIR / src.name, out_dir / src.name)

    meta_json = load_json(SRC_DIR / DATASET_METADATA_PATH.name)
    meta_json.update({"stride_train": stride, "n_train": int(len(train["y"]))})
    save_json(meta_json, out_dir / DATASET_METADATA_PATH.name)
    logger.info(f"Wrote {out_dir} (train {len(train['y'])}, val/test/scalers copied unchanged).")
    return meta_json


def main():
    p = argparse.ArgumentParser(description="Build a denser-train dataset.")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--out", default=str(DATA_DIR / "processed_stride2"))
    args = p.parse_args()
    densify(args.stride, Path(args.out))


if __name__ == "__main__":
    main()
