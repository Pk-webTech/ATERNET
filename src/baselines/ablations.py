"""
Ablation studies for ATERNET.

Each ablation removes or replaces one component to isolate its
contribution. All variants use the same data, seed, training budget,
and evaluation pipeline as the full model.

Ablations mapped to objectives:
  O1 (Context-Gated Routing):
    - no_gate:         uniform 1/4 weights, no routing
    - fixed_temp:      gate exists but T=1 always (no σ-conditioning)

  O2 (Uncertainty-Conditioned Dynamic Quantile Decoding):
    - fixed_width:     decoder ignores σ_total, width = softplus(base) only
    - sigma_nwp_only:  Phase 4 uses only σ_NWP, drops σ_expert
    - sigma_exp_only:  Phase 4 uses only σ_expert, drops σ_NWP

  O3 (Joint End-to-End Training):
    - no_dense_expert: remove the dense expert auxiliary loss (w=0 always)

  O4 (validated via the per-regime table, not a separate ablation)

Usage:
    python -m src.baselines.ablations --ablation no_gate [--epochs 50]
    python -m src.baselines.ablations --all [--epochs 50]
"""

import argparse
import copy
from typing import Optional

import numpy as np
import torch

from config.paths import METRICS_DIR, CHECKPOINTS_DIR, SCALERS_PATH, ensure_dirs
from config.training_config import TRAINING_CONFIG, TrainingConfig
from src.models.aternet import ATERNET
from src.training.dataset import ATERNETDataset, build_dataloaders
from src.training.trainer import Trainer
from src.evaluation.evaluate import predict
from src.evaluation.metrics import all_metrics, per_horizon_metrics
from src.evaluation.baselines import persistence_from_dataset
from src.utils.constants import ID_TO_REGIME
from src.utils.io import save_json, load_pickle
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)

ABLATION_NAMES = [
    "no_gate", "fixed_temp", "fixed_width",
    "sigma_nwp_only", "sigma_exp_only", "no_dense_expert",
]


def _patch_model(model: ATERNET, ablation: str) -> ATERNET:
    """
    Modify a model in-place to implement the ablation.
    Returns the same model for chaining.
    """
    if ablation == "no_gate":
        # Replace gate forward to always return uniform weights
        n_experts = model.n_experts
        original_router_forward = model.router.forward

        def uniform_gate_forward(h_expert, h_sigma, x_sigma_raw,
                                 sigma_expert_raw, expert_predictions_stacked):
            out = original_router_forward(h_expert, h_sigma, x_sigma_raw,
                                          sigma_expert_raw, expert_predictions_stacked)
            B = h_expert.shape[0]
            uniform = torch.ones(B, n_experts, device=h_expert.device) / n_experts
            # Re-aggregate with uniform weights
            y_hat = (uniform.unsqueeze(-1) * expert_predictions_stacked).sum(dim=1)
            out["gate_weights"] = uniform
            out["y_hat_routed"] = y_hat
            return out

        model.router.forward = uniform_gate_forward

    elif ablation == "fixed_temp":
        # Force temperature to 1.0 always
        original_gate_forward = model.router.gate.forward

        def fixed_temp_forward(h_expert, h_sigma, sigma_total):
            out = original_gate_forward(h_expert, h_sigma, sigma_total)
            # Recompute logits at T=1
            logits = out.get("_logits", None)
            if logits is not None:
                out["gate_weights"] = torch.softmax(logits, dim=-1)
                out["temperature"] = torch.ones_like(out["temperature"])
            return out

        # Patch at the router level instead
        original_rf = model.router.forward

        def fixed_temp_router(h_expert, h_sigma, x_sigma_raw,
                              sigma_expert_raw, expert_predictions_stacked):
            out = original_rf(h_expert, h_sigma, x_sigma_raw,
                              sigma_expert_raw, expert_predictions_stacked)
            # Recompute gate at T=1
            gate_logits = model.router.gate.logits_net(
                torch.cat([h_expert, h_sigma], dim=-1)
            )
            gate_weights = torch.softmax(gate_logits, dim=-1)
            y_hat = (gate_weights.unsqueeze(-1) * expert_predictions_stacked).sum(dim=1)
            out["gate_weights"] = gate_weights
            out["y_hat_routed"] = y_hat
            out["temperature"] = torch.ones(h_expert.shape[0], device=h_expert.device)
            return out

        model.router.forward = fixed_temp_router

    elif ablation == "fixed_width":
        # Zero out σ_total's contribution to interval width
        original_decoder_forward = model.decoder.forward

        def fixed_width_decoder(y_hat_routed, h_expert, h_nwp, sigma_total):
            return original_decoder_forward(
                y_hat_routed, h_expert, h_nwp,
                torch.zeros_like(sigma_total)  # σ_total = 0 -> fixed base width
            )

        model.decoder.forward = fixed_width_decoder

    elif ablation == "sigma_nwp_only":
        # Set σ_expert to zero in the fusion
        original_rf = model.router.forward

        def nwp_only_router(h_expert, h_sigma, x_sigma_raw,
                            sigma_expert_raw, expert_predictions_stacked):
            return original_rf(h_expert, h_sigma, x_sigma_raw,
                               torch.zeros_like(sigma_expert_raw),
                               expert_predictions_stacked)

        model.router.forward = nwp_only_router

    elif ablation == "sigma_exp_only":
        # Zero out σ_NWP features
        original_rf = model.router.forward

        def exp_only_router(h_expert, h_sigma, x_sigma_raw,
                            sigma_expert_raw, expert_predictions_stacked):
            return original_rf(h_expert, h_sigma,
                               torch.zeros_like(x_sigma_raw),
                               sigma_expert_raw,
                               expert_predictions_stacked)

        model.router.forward = exp_only_router

    elif ablation == "no_dense_expert":
        # Handled at training time: see trainer override below
        pass

    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    return model


def run_ablation(ablation: str, epochs: int = 50) -> dict:
    """Train one ablation variant and evaluate it."""
    logger.info(f"{'='*60}")
    logger.info(f"Running ablation: {ablation}")
    logger.info(f"{'='*60}")

    ensure_dirs()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, train_ds = build_dataloaders(batch_size=64)
    test_ds = ATERNETDataset("test")

    model_kwargs = dict(
        n_seq_features=train_ds.n_seq_features,
        n_nwp_features=train_ds.n_nwp_features,
        n_sigma_features=train_ds.n_sigma_features,
        horizon=train_ds.horizon,
    )
    model = ATERNET(**model_kwargs)
    model = _patch_model(model, ablation)

    cfg = TrainingConfig()
    cfg.epochs = epochs

    ckpt_best = CHECKPOINTS_DIR / f"ablation_{ablation}_best.pt"
    ckpt_last = CHECKPOINTS_DIR / f"ablation_{ablation}_last.pt"
    history_path = METRICS_DIR / f"ablation_{ablation}_history.json"

    # For no_dense_expert, disable regime weights
    use_regime_weights = (ablation != "no_dense_expert")
    dense_expert_w0 = 0.0 if ablation == "no_dense_expert" else None

    trainer = Trainer(
        model, train_loader, val_loader, device=device, cfg=cfg,
        model_kwargs=model_kwargs,
        use_regime_weights=use_regime_weights,
        best_path=ckpt_best, last_path=ckpt_last, history_path=history_path,
    )

    # If no_dense_expert, override the expert weight to 0
    if ablation == "no_dense_expert":
        trainer._expert_weight_fn = lambda epoch: 0.0

    trainer.fit(epochs=epochs)

    # Evaluate
    from src.training.checkpoint import load_checkpoint
    model_eval, payload = load_checkpoint(ckpt_best, model_class=ATERNET,
                                          map_location=device)
    model_eval = _patch_model(model_eval, ablation)
    model_eval = model_eval.to(device)

    preds = predict(model_eval, test_ds, device)
    y_test = test_ds.y.numpy()

    scalers = load_pickle(SCALERS_PATH)
    baseline = persistence_from_dataset(test_ds.x_seq.numpy(), test_ds.horizon,
                                        scalers["X_seq"])
    headline = all_metrics(y_test, preds["p10"], preds["p50"], preds["p90"],
                           y_baseline=baseline)
    per_h = per_horizon_metrics(y_test, preds["p50"], preds["p10"], preds["p90"])

    regime_ids = test_ds.regime_id.numpy()
    per_regime = {}
    for rid, rname in ID_TO_REGIME.items():
        mask = regime_ids == rid
        if mask.sum() == 0:
            per_regime[rname] = {"n": 0}
            continue
        m = all_metrics(y_test[mask], preds["p10"][mask], preds["p50"][mask],
                        preds["p90"][mask], y_baseline=baseline[mask])
        m["n"] = int(mask.sum())
        per_regime[rname] = m

    results = {
        "model": f"ATERNET (ablation: {ablation})",
        "ablation": ablation,
        "headline": headline,
        "per_regime": per_regime,
        "per_horizon": per_h,
    }

    out_path = METRICS_DIR / f"ablation_{ablation}.json"
    save_json(results, out_path)
    logger.info(f"Ablation {ablation} results -> {out_path}")
    logger.info(f"  nRMSE {headline['nrmse_pct']:.2f}% | PICP {headline['picp']:.3f} "
                f"| Winkler {headline['winkler']:.4f}")
    return results


def run_all(epochs: int = 50):
    """Run every ablation and collect results."""
    all_results = {}
    for name in ABLATION_NAMES:
        try:
            all_results[name] = run_ablation(name, epochs=epochs)
        except Exception as e:
            logger.error(f"Ablation {name} failed: {e}")
            all_results[name] = {"error": str(e)}

    save_json(all_results, METRICS_DIR / "ablations_all.json")
    return all_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", choices=ABLATION_NAMES, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all:
        run_all(epochs=args.epochs)
    elif args.ablation:
        run_ablation(args.ablation, epochs=args.epochs)
    else:
        print("Specify --ablation <name> or --all")