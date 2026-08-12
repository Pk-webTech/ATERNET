"""
End-to-end integration: a real (tiny) training run on the real Phase 1
data, then checkpoint round-trip and evaluation.

These are the tests that catch the failures unit tests structurally
cannot: a loss that is finite in isolation but explodes after ten steps,
a checkpoint that saves but doesn't reload, an evaluation path that
disagrees with the training path about tensor shapes.
"""

import numpy as np
import pytest
import torch

from config.paths import DATASET_TRAIN_PATH
from config.training_config import TrainingConfig

pytestmark = pytest.mark.skipif(
    not DATASET_TRAIN_PATH.exists(), reason="Phase 1 data not built; run run_pipeline.py"
)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """
    Train briefly on a subset and return (model, history).

    Budget note: 2 epochs on 256 samples was not enough to clear even the
    climatology baseline (RMSE 0.211 vs 0.197), which made
    `test_beats_climatology_baseline` fail for want of training rather
    than for any real defect. 5 epochs on 1024 samples clears it with
    margin and still runs in a few seconds.
    """
    from torch.utils.data import DataLoader, Subset
    from src.models.aternet import ATERNET
    from src.training.dataset import ATERNETDataset
    from src.training.trainer import Trainer

    torch.manual_seed(0)
    train_ds = ATERNETDataset("train")
    val_ds = ATERNETDataset("val")
    train_loader = DataLoader(Subset(train_ds, range(1024)), batch_size=32,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(val_ds, range(128)), batch_size=32)

    kwargs = dict(n_seq_features=train_ds.n_seq_features,
                  n_nwp_features=train_ds.n_nwp_features,
                  n_sigma_features=train_ds.n_sigma_features,
                  horizon=train_ds.horizon)
    model = ATERNET(**kwargs)

    cfg = TrainingConfig(); cfg.epochs = 5
    scratch = tmp_path_factory.mktemp("tiny_run")
    trainer = Trainer(model, train_loader, val_loader, device=torch.device("cpu"),
                      cfg=cfg, model_kwargs=kwargs,
                      # never write over the real checkpoints/ artifacts
                      best_path=scratch / "best.pt", last_path=scratch / "last.pt",
                      history_path=scratch / "history.json")
    history = trainer.fit(epochs=5)
    return model, history, kwargs


def test_training_runs_and_loss_is_finite(tiny_run):
    _, history, _ = tiny_run
    assert len(history) == 5
    for record in history:
        assert np.isfinite(record["train_loss_total"])
        assert np.isfinite(record["val_pinball_total"])


def test_training_reduces_loss(tiny_run):
    _, history, _ = tiny_run
    assert history[-1]["train_loss_total"] < history[0]["train_loss_total"]


def test_gate_does_not_collapse(tiny_run):
    """Every expert must retain a non-trivial share of routing mass."""
    from src.phase3.expert_pool import EXPERT_ORDER
    _, history, _ = tiny_run
    shares = [history[-1][f"val_gate_share_{n}"] for n in EXPERT_ORDER]
    assert abs(sum(shares) - 1.0) < 1e-4
    assert min(shares) > 0.02, f"gate collapsed: {dict(zip(EXPERT_ORDER, shares))}"


def test_checkpoint_roundtrip_preserves_predictions(tiny_run, tmp_path):
    from src.models.aternet import ATERNET
    from src.training.checkpoint import save_checkpoint, load_checkpoint

    model, _, kwargs = tiny_run
    model.eval()
    x = (torch.rand(4, 48, kwargs["n_seq_features"]),
         torch.rand(4, 48, kwargs["n_nwp_features"]),
         torch.rand(4, 48, kwargs["n_sigma_features"]))
    with torch.no_grad():
        before = model(*x)["p50"]

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, model_kwargs=kwargs)
    restored, _ = load_checkpoint(path, model_class=ATERNET)
    restored.eval()
    with torch.no_grad():
        after = restored(*x)["p50"]

    assert torch.allclose(before, after, atol=1e-6)


def test_non_crossing_holds_on_real_data(tiny_run):
    from src.training.dataset import ATERNETDataset
    model, _, _ = tiny_run
    model.eval()
    ds = ATERNETDataset("test")
    with torch.no_grad():
        out = model(ds.x_seq[:512], ds.x_nwp[:512], ds.x_sigma[:512])
    assert (out["p10"] <= out["p50"] + 1e-6).all()
    assert (out["p50"] <= out["p90"] + 1e-6).all()


def test_beats_climatology_baseline(tiny_run):
    """
    2 epochs on 256 samples is not enough to beat *persistence* (a
    strong short-horizon baseline), so asserting that here would make
    the suite flaky for no diagnostic gain. Climatology -- the constant
    per-step training mean -- is the right bar at this budget: failing
    to beat it means the model has learned nothing input-dependent at
    all, which is exactly the wiring error worth catching cheaply.

    The persistence comparison is asserted against a fully-trained
    checkpoint in `test_full_model_beats_persistence` below.
    """
    from src.evaluation.baselines import climatology_forecast
    from src.evaluation.metrics import rmse
    from src.training.dataset import ATERNETDataset

    model, _, _ = tiny_run
    model.eval()
    train_ds, ds = ATERNETDataset("train"), ATERNETDataset("test")
    with torch.no_grad():
        out = model(ds.x_seq, ds.x_nwp, ds.x_sigma)
    y = ds.y.numpy()
    clim = climatology_forecast(train_ds.y.numpy(), len(y), y.shape[1])
    assert rmse(y, out["p50"].numpy()) < rmse(y, clim)


@pytest.mark.skipif(not (__import__("config.paths", fromlist=["p"]).BEST_CHECKPOINT_PATH).exists(),
                    reason="no trained checkpoint; run src.training.train first")
def test_full_model_beats_persistence():
    """A fully-trained model must beat persistence, or the whole exercise is moot."""
    from config.paths import BEST_CHECKPOINT_PATH, SCALERS_PATH
    from src.evaluation.baselines import persistence_from_dataset
    from src.evaluation.metrics import rmse
    from src.models.aternet import ATERNET
    from src.training.checkpoint import load_checkpoint
    from src.training.dataset import ATERNETDataset
    from src.utils.io import load_pickle

    model, _ = load_checkpoint(BEST_CHECKPOINT_PATH, model_class=ATERNET)
    model.eval()
    ds = ATERNETDataset("test")
    with torch.no_grad():
        out = model(ds.x_seq, ds.x_nwp, ds.x_sigma)
    y = ds.y.numpy()
    baseline = persistence_from_dataset(ds.x_seq.numpy(), y.shape[1],
                                        load_pickle(SCALERS_PATH)["X_seq"])
    assert rmse(y, out["p50"].numpy()) < rmse(y, baseline)


def test_overcast_cap_is_valid_against_real_targets():
    """
    The overcast expert's hard 0.40 ceiling is an architectural
    constraint; if real overcast targets exceed it, the expert cannot
    fit its own regime no matter how long it trains.
    """
    from src.phase3.overcast_expert import verify_cap, OVERCAST_OUTPUT_CAP
    from src.training.dataset import ATERNETDataset
    ds = ATERNETDataset("train")
    y_overcast = ds.y[ds.regime_id == 2].numpy()
    ok, q, mx = verify_cap(y_overcast)
    assert ok, (f"overcast targets exceed the {OVERCAST_OUTPUT_CAP} cap: "
                f"99.9th pct {q:.3f}, max {mx:.3f}")
