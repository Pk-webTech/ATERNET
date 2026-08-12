"""
Torch Dataset / DataLoader wrappers over the Phase 1 .npz splits.

The whole processed dataset is small enough (~9k windows) to hold in
memory as tensors, so this loads once and indexes tensors directly --
no per-item disk I/O, no worker processes needed to keep the GPU fed.
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config.paths import DATASET_TRAIN_PATH, DATASET_VAL_PATH, DATASET_TEST_PATH
from config.training_config import TRAINING_CONFIG
from src.utils.io import load_npz
from src.utils.feature_groups import REGIME_ID_INDEX
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SPLIT_PATHS = {
    "train": DATASET_TRAIN_PATH,
    "val": DATASET_VAL_PATH,
    "test": DATASET_TEST_PATH,
}


class ATERNETDataset(Dataset):
    """One Phase 1 split, held in memory as float32 tensors."""

    def __init__(self, split: str):
        if split not in _SPLIT_PATHS:
            raise ValueError(f"Unknown split '{split}'. Expected one of {list(_SPLIT_PATHS)}.")
        path = _SPLIT_PATHS[split]
        data = load_npz(path)

        self.split = split
        self.x_seq = torch.from_numpy(np.ascontiguousarray(data["X_seq"], dtype=np.float32))
        self.x_nwp = torch.from_numpy(np.ascontiguousarray(data["X_nwp"], dtype=np.float32))
        self.x_sigma = torch.from_numpy(np.ascontiguousarray(data["X_sigma"], dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(data["y"], dtype=np.float32))
        self.meta = torch.from_numpy(np.ascontiguousarray(data["meta"], dtype=np.float32))
        self.regime_id = self.meta[:, REGIME_ID_INDEX].long()
        self.window_start_time = data["window_start_time"]

        if not torch.isfinite(self.x_seq).all():
            raise ValueError(f"Non-finite values in X_seq of split '{split}'.")

        logger.info(f"Loaded '{split}' split: {len(self)} windows from {path.name}")

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, i: int):
        return {
            "x_seq": self.x_seq[i],
            "x_nwp": self.x_nwp[i],
            "x_sigma": self.x_sigma[i],
            "y": self.y[i],
            "regime_id": self.regime_id[i],
        }

    @property
    def n_seq_features(self) -> int:
        return self.x_seq.shape[-1]

    @property
    def n_nwp_features(self) -> int:
        return self.x_nwp.shape[-1]

    @property
    def n_sigma_features(self) -> int:
        return self.x_sigma.shape[-1]

    @property
    def horizon(self) -> int:
        return self.y.shape[-1]


def build_dataloaders(batch_size: int = TRAINING_CONFIG.batch_size,
                      num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader, ATERNETDataset]:
    """
    Returns (train_loader, val_loader, test_loader, train_dataset).

    The train dataset is returned too so callers can read feature
    dimensionality off it instead of hard-coding shapes.

    NOTE: train is shuffled. That is safe and desirable here even though
    the data is a time series -- the chronological train/val/test split
    was already done at the *window* level in Phase 1, so no future
    information can leak backwards by shuffling within the training set.

    `drop_last=True` on the training loader: Phase 4's UncertaintyFusion
    contains BatchNorm layers, and a trailing batch of size 1 makes the
    batch variance undefined.
    """
    train_ds = ATERNETDataset("train")
    val_ds = ATERNETDataset("val")
    test_ds = ATERNETDataset("test")

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader, train_ds
