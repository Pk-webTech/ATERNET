"""Phase 1: shapes, leakage, contiguity, and target sanity."""

import numpy as np
import pandas as pd
import pytest

from config.paths import DATASET_TRAIN_PATH, DATASET_VAL_PATH, DATASET_TEST_PATH, SCALERS_PATH
from src.phase1.create_windows import _contiguous_run_lengths, create_windows
from src.utils.feature_groups import SEQUENCE_FEATURES, SIGMA_FEATURES, NWP_FEATURES
from src.utils.helpers import chronological_split
from src.utils.io import load_npz, load_pickle

pytestmark = pytest.mark.skipif(
    not DATASET_TRAIN_PATH.exists(), reason="Phase 1 data not built; run run_pipeline.py"
)


def test_split_shapes_consistent():
    for path in (DATASET_TRAIN_PATH, DATASET_VAL_PATH, DATASET_TEST_PATH):
        d = load_npz(path)
        n = d["y"].shape[0]
        assert d["X_seq"].shape == (n, 48, len(SEQUENCE_FEATURES))
        assert d["X_sigma"].shape == (n, 48, len(SIGMA_FEATURES))
        assert d["X_nwp"].shape == (n, 48, len(NWP_FEATURES))
        assert d["y"].shape == (n, 16)


def test_no_nans_or_infs():
    for path in (DATASET_TRAIN_PATH, DATASET_VAL_PATH, DATASET_TEST_PATH):
        d = load_npz(path)
        for k in ("X_seq", "X_sigma", "X_nwp", "y", "meta"):
            assert np.isfinite(d[k]).all(), f"non-finite values in {k} of {path.name}"


def test_targets_in_physical_range():
    for path in (DATASET_TRAIN_PATH, DATASET_VAL_PATH, DATASET_TEST_PATH):
        y = load_npz(path)["y"]
        assert y.min() >= 0.0 and y.max() <= 1.0


def test_targets_actually_use_the_range():
    """
    Guards the PVWatts scaling fix: if pv_norm peaks at a few percent of
    nameplate, every [0,1]-calibrated constant downstream is dead.
    """
    y = load_npz(DATASET_TRAIN_PATH)["y"]
    assert y.max() > 0.5, f"train targets peak at {y.max():.3f} of capacity -- check the PV model"


def test_chronological_split_has_no_temporal_overlap():
    t_train = load_npz(DATASET_TRAIN_PATH)["window_start_time"]
    t_val = load_npz(DATASET_VAL_PATH)["window_start_time"]
    t_test = load_npz(DATASET_TEST_PATH)["window_start_time"]
    assert t_train.max() < t_val.min(), "train/val overlap in time -- leakage"
    assert t_val.max() < t_test.min(), "val/test overlap in time -- leakage"


def test_scalers_fit_on_train_only():
    """Train features must land in [0,1]; val/test may exceed it, which is correct."""
    d = load_npz(DATASET_TRAIN_PATH)
    assert d["X_seq"].min() >= -1e-6 and d["X_seq"].max() <= 1 + 1e-6
    assert SCALERS_PATH.exists()
    assert set(load_pickle(SCALERS_PATH)) >= {"X_seq", "X_sigma", "X_nwp", "meta_volatility"}


def test_all_regimes_present_in_every_split():
    """A regime with no windows in a split makes its expert unevaluable."""
    for path in (DATASET_TRAIN_PATH, DATASET_VAL_PATH, DATASET_TEST_PATH):
        ids = load_npz(path)["meta"][:, 0].astype(int)
        counts = np.bincount(ids, minlength=4)
        assert (counts > 0).all(), f"{path.name} missing regimes: {counts}"


def test_contiguity_rejects_gapped_windows():
    ts = pd.Series(pd.date_range("2023-01-01", periods=100, freq="15min"))
    ts = pd.concat([ts[:50], ts[60:]]).reset_index(drop=True)  # punch a gap
    run = _contiguous_run_lengths(ts, 15)
    assert run[49] == 50           # contiguous up to the gap
    assert run[50] == 1            # run restarts after it


def test_create_windows_skips_gaps():
    n = 200
    ts = pd.date_range("2023-01-01", periods=n, freq="15min").to_series().reset_index(drop=True)
    ts = pd.concat([ts[:100], ts[110:]]).reset_index(drop=True)
    df = pd.DataFrame({"timestamp": ts})
    for c in set(SEQUENCE_FEATURES + SIGMA_FEATURES + NWP_FEATURES):
        df[c] = np.random.rand(len(df))
    df["pv_norm"] = np.random.rand(len(df))
    from src.utils.feature_groups import ROUTING_METADATA_FEATURES
    for c in ROUTING_METADATA_FEATURES:
        df[c] = 0.0
    out = create_windows(df, window_size=48, horizon=16, stride=4, freq_minutes=15)
    assert out["X_seq"].shape[0] > 0
    assert out["X_seq"].shape[0] < len(range(0, len(df) - 64 + 1, 4))  # some were dropped
