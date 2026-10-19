"""
Baseline 2 -- LSTM Quantile Regression.

A temporal deep-learning baseline without experts, routing, or
uncertainty-conditioned decoding. Two-layer LSTM on the concatenated
sequence+sigma+nwp features, with three linear heads for P10/P50/P90.
Trained with the same pinball loss as ATERNET but with no composite
terms.

Usage:
    python -m src.baselines.lstm_quantile [--epochs 50]
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from config.paths import METRICS_DIR, CHECKPOINTS_DIR, SCALERS_PATH, ensure_dirs
from config.training_config import TRAINING_CONFIG
from src.losses.pinball_loss import multi_quantile_pinball_loss
from src.training.dataset import ATERNETDataset
from src.evaluation.metrics import all_metrics, per_horizon_metrics
from src.evaluation.baselines import persistence_from_dataset
from src.utils.constants import ID_TO_REGIME
from src.utils.io import save_json, load_pickle
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LSTMQuantile(nn.Module):
    """Two-layer LSTM -> three quantile heads."""

    def __init__(self, n_features: int, horizon: int,
                 hidden_dim: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head_p10 = nn.Linear(hidden_dim, horizon)
        self.head_p50 = nn.Linear(hidden_dim, horizon)
        self.head_p90 = nn.Linear(hidden_dim, horizon)
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> dict:
        """x: (B, T, F) concatenated features. Returns dict with p10/p50/p90."""
        _, (h_n, _) = self.lstm(x)          # h_n: (n_layers, B, hidden)
        h = h_n[-1]                          # last layer: (B, hidden)
        p10 = torch.clamp(self.head_p10(h), 0.0, 1.0)
        p50 = torch.clamp(self.head_p50(h), 0.0, 1.0)
        p90 = torch.clamp(self.head_p90(h), 0.0, 1.0)

        # Enforce non-crossing
        p10 = torch.minimum(p10, p50)
        p90 = torch.maximum(p90, p50)

        return {"p10": p10, "p50": p50, "p90": p90}


def _concat_features(ds: ATERNETDataset) -> torch.Tensor:
    """Concatenate all per-step lookback features into (N, T, F_total).

    x_nwp also spans the forecast horizon; per-step concatenation needs
    equal lengths, so the horizon NWP is flattened onto every step.
    """
    T = ds.x_seq.shape[1]
    nwp_future = ds.x_nwp[:, T:, :].flatten(1).unsqueeze(1).expand(-1, T, -1)
    return torch.cat([ds.x_seq, ds.x_nwp[:, :T], nwp_future, ds.x_sigma], dim=-1)


@torch.no_grad()
def _predict(model, x: torch.Tensor, device: torch.device, batch_size: int = 256) -> dict:
    model.eval()
    preds = {"p10": [], "p50": [], "p90": []}
    for i in range(0, len(x), batch_size):
        out = model(x[i:i+batch_size].to(device))
        for k in preds:
            preds[k].append(out[k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in preds.items()}


def train_and_evaluate(epochs: int = 50):
    ensure_dirs()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ATERNETDataset("train")
    val_ds = ATERNETDataset("val")
    test_ds = ATERNETDataset("test")

    x_train = _concat_features(train_ds)
    x_val = _concat_features(val_ds)
    x_test = _concat_features(test_ds)
    n_features = x_train.shape[-1]
    horizon = train_ds.horizon

    model = LSTMQuantile(n_features, horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    batch_size = 64
    best_val_loss = float("inf")
    patience, patience_counter = 10, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        train_losses = []
        for i in range(0, len(x_train) - batch_size + 1, batch_size):
            idx = perm[i:i+batch_size]
            xb = x_train[idx].to(device)
            yb = train_ds.y[idx].to(device)
            out = model(xb)
            loss_dict = multi_quantile_pinball_loss(out, yb)
            loss = loss_dict["total"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        val_preds = _predict(model, x_val, device)
        val_q = {k: torch.from_numpy(v) for k, v in val_preds.items()}
        val_loss_dict = multi_quantile_pinball_loss(val_q, val_ds.y)
        val_loss = val_loss_dict["total"].item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "lstm_baseline_best.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter == 0:
            logger.info(f"Epoch {epoch:3d} | train_loss {np.mean(train_losses):.5f} "
                        f"| val_pinball {val_loss:.5f} {'*' if patience_counter == 0 else ''}")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    # Load best and evaluate on test
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / "lstm_baseline_best.pt",
                                     map_location=device, weights_only=True))
    test_preds = _predict(model, x_test, device)
    y_test = test_ds.y.numpy()

    scalers = load_pickle(SCALERS_PATH)
    baseline = persistence_from_dataset(test_ds.x_seq.numpy(), horizon, scalers["X_seq"])
    headline = all_metrics(y_test, test_preds["p10"], test_preds["p50"],
                           test_preds["p90"], y_baseline=baseline)
    per_h = per_horizon_metrics(y_test, test_preds["p50"],
                                test_preds["p10"], test_preds["p90"])

    regime_ids = test_ds.regime_id.numpy()
    per_regime = {}
    for rid, rname in ID_TO_REGIME.items():
        mask = regime_ids == rid
        if mask.sum() == 0:
            per_regime[rname] = {"n": 0}
            continue
        m = all_metrics(y_test[mask], test_preds["p10"][mask], test_preds["p50"][mask],
                        test_preds["p90"][mask], y_baseline=baseline[mask])
        m["n"] = int(mask.sum())
        per_regime[rname] = m

    results = {
        "model": "LSTM Quantile",
        "headline": headline,
        "per_regime": per_regime,
        "per_horizon": per_h,
    }

    out_path = METRICS_DIR / "baseline_lstm.json"
    save_json(results, out_path)
    logger.info(f"LSTM baseline results -> {out_path}")
    logger.info(f"  nRMSE {headline['nrmse_pct']:.2f}% | PICP {headline['picp']:.3f} "
                f"| Winkler {headline['winkler']:.4f}")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(epochs=args.epochs)