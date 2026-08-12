"""Shared fixtures. Real Phase 1 data if present, synthetic otherwise."""

import numpy as np
import pytest
import torch

from config.paths import DATASET_TRAIN_PATH
from src.utils.feature_groups import SEQUENCE_FEATURES, SIGMA_FEATURES, NWP_FEATURES

WINDOW, HORIZON, BATCH = 48, 16, 8


@pytest.fixture(scope="session")
def dims():
    return dict(n_seq=len(SEQUENCE_FEATURES), n_sigma=len(SIGMA_FEATURES),
                n_nwp=len(NWP_FEATURES), window=WINDOW, horizon=HORIZON)


@pytest.fixture
def fake_batch(dims):
    g = torch.Generator().manual_seed(0)
    return {
        "x_seq": torch.rand(BATCH, dims["window"], dims["n_seq"], generator=g),
        "x_nwp": torch.rand(BATCH, dims["window"], dims["n_nwp"], generator=g),
        "x_sigma": torch.rand(BATCH, dims["window"], dims["n_sigma"], generator=g),
        "y": torch.rand(BATCH, dims["horizon"], generator=g),
        "regime_id": torch.randint(0, 4, (BATCH,), generator=g),
    }


@pytest.fixture(scope="session")
def has_real_data():
    return DATASET_TRAIN_PATH.exists()


@pytest.fixture
def model(dims):
    from src.models.aternet import ATERNET
    torch.manual_seed(0)
    return ATERNET(n_seq_features=dims["n_seq"], n_nwp_features=dims["n_nwp"],
                   n_sigma_features=dims["n_sigma"], horizon=dims["horizon"])
