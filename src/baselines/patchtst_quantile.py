"""
Baseline 3 -- Vanilla PatchTST with Quantile Heads.

Uses the same PatchTST backbone configuration as ATERNET Phase 2 but
removes the expert pool (Phase 3), routing gate (Phase 4), and
uncertainty-conditioned decoder (Phase 5). Replaces them with three
direct linear heads for P10/P50/P90.

This baseline isolates the contribution of the MoE + routing + adaptive
decoder: if vanilla PatchTST scores similarly to full ATERNET, the
expert routing adds no value.

Usage:
    python -m src.baselines.patchtst_quantile [--epochs 50]
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from config.paths import METRICS_DIR, CHECKPOINTS_DIR, SCALERS_PATH, ensure_dirs
from config.model_config import EMBEDDING_CONFIG
from src.phase2.temporal_backbone import TemporalBackbone
from src.losses.pinball_loss import multi_quantile_pinball_loss
from src.training.dataset import ATERNETDataset
from src.evaluation.metrics import all_metrics, per_horizon_metrics
from src.evaluation.baselines import persistence_from_dataset
from src.utils.constants import ID_TO_REGIME
from src.utils.io import save_json, load_pickle
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PatchTSTQuantile(nn.Module):
    """PatchTST backbone + direct quantile heads. No experts, no routing."""

    def __init__(self, n_seq_features: int, n_nwp_features: int,
                 n_sigma_features: int, horizon: int,
                 h_expert_dim: int = EMBEDDING_CONFIG.h_expert_dim):
        super().__init__()
        self.backbone = TemporalBackbone(
            n_seq_features=n_seq_features,
            n_nwp_features=n_nwp_features,
            n_sigma_features=n_sigma_features,
        )
        self.head_p10 = nn.Sequential(
            nn.Linear(h_expert_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, horizon),
        )
        self.head_p50 = nn.Sequential(
            nn.Linear(h_expert_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, horizon),
        )
        self.head_p90 = nn.Sequential(
            nn.Linear(h_expert_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, horizon),
        )

    def forward(self, x_seq, x_nwp, x_sigma):
        emb = self.backbone(x_seq, x_nwp, x_sigma)
        h = emb["h_expert"]    # (B, 128)
        p50 = torch.clamp(self.head_p50(h), 0.0, 1.0)
        p10 = torch.clamp(self.head_p10(h), 0.0, 1.0)
        p90 = torch.clamp(self.head_p90(h), 0.0, 1.0)
        # Enforce non-crossing
        p10 = torch.minimum(p10, p50)
        p90 = torch.maximum(p90, p50)
        return {"p10": p10, "p50": p50, "p90": p90}


@torch.no_grad()
def _predict(model, ds, device, batch_size=256):
    model.eval()
    preds = {"p10": [], "p50": [], "p90": []}
    for i in range(0, len(ds), batch_size):
        sl = slice(i, min(i + batch_size, len(ds)))
        out = model(ds.x_seq[sl].to(device), ds.x_nwp[sl].to(device),
                     ds.x_sigma[sl].to(device))
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

    model = PatchTSTQuantile(
        n_seq_features=train_ds.n_seq_features,
        n_nwp_features=train_ds.n_nwp_features,
        n_sigma_features=train_ds.n_sigma_features,
        horizon=train_ds.horizon,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    batch_size = 64
    best_val_loss = float("inf")
    patience, patience_counter = 10, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_ds))
        train_losses = []
        for i in range(0, len(train_ds) - batch_size + 1, batch_size):
            idx = perm[i:i+batch_size]
            out = model(train_ds.x_seq[idx].to(device),
                        train_ds.x_nwp[idx].to(device),
                        train_ds.x_sigma[idx].to(device))
            yb = train_ds.y[idx].to(device)
            loss_dict = multi_quantile_pinball_loss(out, yb)
            loss = loss_dict["total"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        val_preds = _predict(model, val_ds, device)
        val_q = {k: torch.from_numpy(v) for k, v in val_preds.items()}
        val_loss_dict = multi_quantile_pinball_loss(val_q, val_ds.y)
        val_loss = val_loss_dict["total"].item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "patchtst_baseline_best.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter == 0:
            logger.info(f"Epoch {epoch:3d} | train_loss {np.mean(train_losses):.5f} "
                        f"| val_pinball {val_loss:.5f} {'*' if patience_counter == 0 else ''}")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    # Load best and evaluate
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / "patchtst_baseline_best.pt",
                                     map_location=device, weights_only=True))
    test_preds = _predict(model, test_ds, device)
    y_test = test_ds.y.numpy()

    scalers = load_pickle(SCALERS_PATH)
    baseline = persistence_from_dataset(test_ds.x_seq.numpy(), test_ds.horizon, scalers["X_seq"])
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
        "model": "PatchTST Quantile (no MoE/routing)",
        "headline": headline,
        "per_regime": per_regime,
        "per_horizon": per_h,
    }

    out_path = METRICS_DIR / "baseline_patchtst.json"
    save_json(results, out_path)
    logger.info(f"PatchTST baseline results -> {out_path}")
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