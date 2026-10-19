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
import contextlib
import copy
from typing import Optional

import numpy as np
import torch

from config.paths import METRICS_DIR, CHECKPOINTS_DIR, SCALERS_PATH, ensure_dirs
from config.training_config import TRAINING_CONFIG, TrainingConfig
from src.losses import composite_loss
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
# Runnable via run_ablation but not part of the run_all() ablation set.
EXTRA_VARIANTS = ["full", "residual_gate", "residual_gate_skip"]


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
        # Force temperature to 1.0 always. The gate already returns its raw
        # logits, so re-softmax them without the sigma-dependent divisor.
        original_gate_forward = model.router.gate.forward

        def fixed_temp_gate(h_expert, h_sigma, fusion_out):
            out = original_gate_forward(h_expert, h_sigma, fusion_out)
            logits = out["logits"]
            if model.router.gate.residual:
                logits = model.router.gate.delta * logits
            out["gate_weights"] = torch.softmax(logits, dim=-1)
            out["temperature"] = torch.ones_like(out["temperature"])
            return out

        model.router.gate.forward = fixed_temp_gate

    elif ablation in ("full", "residual_gate", "residual_gate_skip"):
        pass  # no patching; these are constructor flags (see run_ablation)

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
        # Handled at training time: see _dense_expert_disabled() in run_ablation
        pass

    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    return model


@contextlib.contextmanager
def _dense_expert_disabled():
    """
    Zero the dense-expert loss weight for the duration of a run.
    compute_composite_loss looks up expert_loss_weight in its own module
    namespace at call time, so swapping it here needs no change to the
    loss or the trainer.
    """
    original = composite_loss.expert_loss_weight
    composite_loss.expert_loss_weight = lambda *a, **k: 0.0
    try:
        yield
    finally:
        composite_loss.expert_loss_weight = original


def run_ablation(ablation: str, epochs: int = 50, seed: int = 42,
                 metrics_dir=None, checkpoints_dir=None,
                 base_model_kwargs: Optional[dict] = None) -> dict:
    """
    Train one ablation variant and evaluate it.

    metrics_dir / checkpoints_dir default to the project-wide locations;
    multi-seed runs pass their own so they don't land in outputs/metrics/
    where comparison_table.py globs ablation_*.json.
    """
    logger.info(f"{'='*60}")
    logger.info(f"Running ablation: {ablation} (seed {seed})")
    logger.info(f"{'='*60}")

    ensure_dirs()
    metrics_dir = metrics_dir or METRICS_DIR
    checkpoints_dir = checkpoints_dir or CHECKPOINTS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, train_ds = build_dataloaders(batch_size=64)
    test_ds = ATERNETDataset("test")

    model_kwargs = dict(
        n_seq_features=train_ds.n_seq_features,
        n_nwp_features=train_ds.n_nwp_features,
        n_sigma_features=train_ds.n_sigma_features,
        horizon=train_ds.horizon,
    )
    # Stored in model_kwargs so the checkpoint rebuilds the same model.
    # base_model_kwargs lets an ablation run on top of a non-default
    # architecture (e.g. the final residual-gate + skip model).
    model_kwargs.update(base_model_kwargs or {})
    if ablation in ("residual_gate", "residual_gate_skip"):
        model_kwargs["residual_gate"] = True
    if ablation == "residual_gate_skip":
        model_kwargs["last_step_skip"] = True
    model = ATERNET(**model_kwargs)
    model = _patch_model(model, ablation)

    cfg = TrainingConfig()
    cfg.epochs = epochs

    ckpt_best = checkpoints_dir / f"ablation_{ablation}_best.pt"
    ckpt_last = checkpoints_dir / f"ablation_{ablation}_last.pt"
    history_path = metrics_dir / f"ablation_{ablation}_history.json"

    # Regime weights only act through the dense expert loss, so they go
    # with it for no_dense_expert.
    use_regime_weights = (ablation != "no_dense_expert")

    trainer = Trainer(
        model, train_loader, val_loader, device=device, cfg=cfg,
        model_kwargs=model_kwargs,
        use_regime_weights=use_regime_weights,
        best_path=ckpt_best, last_path=ckpt_last, history_path=history_path,
    )

    if ablation == "no_dense_expert":
        with _dense_expert_disabled():
            trainer.fit(epochs=epochs)
    else:
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

    out_path = metrics_dir / f"ablation_{ablation}.json"
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
    p.add_argument("--ablation", choices=ABLATION_NAMES + EXTRA_VARIANTS, default=None)
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