"""
Evaluation entry point.

    python -m src.evaluation.evaluate [--split test] [--checkpoint path]

Loads the best checkpoint, runs inference over a split, and writes:
  outputs/predictions/test_predictions.npz   raw p10/p50/p90 + diagnostics
  outputs/metrics/test_metrics.json          headline + per-regime + per-horizon
  outputs/reports/evaluation_report.md       human-readable summary

Reports metrics overall, per regime, and per horizon step, plus a
non-crossing structural check -- if P10 <= P50 <= P90 ever failed, every
probabilistic metric above it would be meaningless, so it is verified
on the real predictions rather than assumed from the architecture.
"""

import argparse

import numpy as np
import torch

from config.paths import (
    BEST_CHECKPOINT_PATH, SCALERS_PATH, TEST_METRICS_PATH, TEST_PREDICTIONS_PATH,
    REPORTS_DIR, ensure_dirs,
)
from src.evaluation.baselines import persistence_from_dataset
from src.evaluation.metrics import all_metrics, per_horizon_metrics
from src.models.aternet import ATERNET
from src.phase3.expert_pool import EXPERT_ORDER
from src.training.checkpoint import load_checkpoint
from src.training.dataset import ATERNETDataset
from src.utils.constants import ID_TO_REGIME
from src.utils.io import save_json, save_npz, load_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def predict(model, dataset, device, batch_size: int = 256) -> dict:
    model.eval()
    keys = ["p10", "p50", "p90", "lower_width", "upper_width",
            "y_hat_routed", "gate_weights", "sigma_total", "sigma_nwp",
            "sigma_expert_calibrated", "alpha", "entropy", "temperature"]
    acc = {k: [] for k in keys}

    for i in range(0, len(dataset), batch_size):
        sl = slice(i, min(i + batch_size, len(dataset)))
        out = model(
            dataset.x_seq[sl].to(device),
            dataset.x_nwp[sl].to(device),
            dataset.x_sigma[sl].to(device),
        )
        for k in keys:
            acc[k].append(out[k].detach().cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def evaluate(split: str = "test", checkpoint_path=BEST_CHECKPOINT_PATH,
             device=None) -> dict:
    ensure_dirs()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ATERNETDataset(split)
    model, payload = load_checkpoint(checkpoint_path, model_class=ATERNET, map_location=device)
    model = model.to(device)

    preds = predict(model, dataset, device)
    y = dataset.y.numpy()

    # ---- Structural guarantee check on real predictions ----
    eps = 1e-6
    n_cross_lower = int(np.sum(preds["p10"] > preds["p50"] + eps))
    n_cross_upper = int(np.sum(preds["p50"] > preds["p90"] + eps))
    non_crossing_ok = (n_cross_lower == 0) and (n_cross_upper == 0)
    if not non_crossing_ok:
        logger.error(f"QUANTILE CROSSING DETECTED: {n_cross_lower} lower, "
                     f"{n_cross_upper} upper violations. Probabilistic metrics below "
                     f"are not trustworthy.")
    else:
        logger.info("Non-crossing verified: P10 <= P50 <= P90 for all samples/steps.")

    # ---- Baseline ----
    scalers = load_pickle(SCALERS_PATH)
    baseline = persistence_from_dataset(dataset.x_seq.numpy(), y.shape[1], scalers["X_seq"])

    headline = all_metrics(y, preds["p10"], preds["p50"], preds["p90"], y_baseline=baseline)

    # ---- Per regime ----
    regime_ids = dataset.regime_id.numpy()
    per_regime = {}
    for rid, rname in ID_TO_REGIME.items():
        mask = regime_ids == rid
        if mask.sum() == 0:
            per_regime[rname] = {"n": 0}
            continue
        m = all_metrics(y[mask], preds["p10"][mask], preds["p50"][mask], preds["p90"][mask],
                        y_baseline=baseline[mask])
        m["n"] = int(mask.sum())
        per_regime[rname] = m

    # ---- Per horizon step ----
    per_horizon = per_horizon_metrics(y, preds["p50"], preds["p10"], preds["p90"])

    # ---- Routing diagnostics ----
    gate = preds["gate_weights"]
    routing = {
        "mean_gate_share": {n: float(gate[:, k].mean()) for k, n in enumerate(EXPERT_ORDER)},
        "argmax_share": {
            n: float(np.mean(gate.argmax(axis=1) == k)) for k, n in enumerate(EXPERT_ORDER)
        },
        "mean_entropy": float(preds["entropy"].mean()),
        "max_entropy": float(np.log(gate.shape[1])),
        "mean_alpha": float(preds["alpha"].mean()),
        "mean_temperature": float(preds["temperature"].mean()),
    }
    # Correlation between fused uncertainty and realised absolute error --
    # the single most direct test of whether sigma_total means anything.
    abs_err = np.abs(y - preds["p50"]).mean(axis=1)
    if abs_err.std() > 0 and preds["sigma_total"].std() > 0:
        routing["corr_sigma_total_vs_abs_error"] = float(
            np.corrcoef(preds["sigma_total"], abs_err)[0, 1]
        )
    width = (preds["p90"] - preds["p10"]).mean(axis=1)
    if width.std() > 0 and abs_err.std() > 0:
        routing["corr_interval_width_vs_abs_error"] = float(np.corrcoef(width, abs_err)[0, 1])

    results = {
        "split": split,
        "n_samples": int(len(dataset)),
        "checkpoint_epoch": payload.get("epoch"),
        "non_crossing_ok": non_crossing_ok,
        "n_crossing_violations": n_cross_lower + n_cross_upper,
        "headline": headline,
        "per_regime": per_regime,
        "per_horizon": per_horizon,
        "routing": routing,
    }

    save_json(results, TEST_METRICS_PATH)
    save_npz(TEST_PREDICTIONS_PATH, y_true=y, regime_id=regime_ids,
             baseline=baseline, **preds)
    _write_report(results)

    logger.info(
        f"[{split}] nRMSE {headline['nrmse_pct']:.2f}% | nMAE {headline['nmae_pct']:.2f}% "
        f"| PICP {headline['picp']:.3f} (target 0.80) | PINAW {headline['pinaw']:.4f} "
        f"| Winkler {headline['winkler']:.4f} "
        f"| skill vs persistence {headline.get('rmse_skill_vs_persistence', float('nan')):.3f}"
    )
    return results


def _write_report(results: dict) -> None:
    h = results["headline"]
    lines = [
        f"# ATERNET evaluation -- {results['split']} split",
        "",
        f"- Samples: {results['n_samples']}",
        f"- Checkpoint epoch: {results['checkpoint_epoch']}",
        f"- Non-crossing (P10 <= P50 <= P90): "
        f"{'PASS' if results['non_crossing_ok'] else 'FAIL'} "
        f"({results['n_crossing_violations']} violations)",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| nMAE (% of capacity) | {h['nmae_pct']:.3f} |",
        f"| nRMSE (% of capacity) | {h['nrmse_pct']:.3f} |",
        f"| Bias | {h['bias']:+.5f} |",
        f"| R2 | {h['r2']:.4f} |",
        f"| PICP (nominal 0.80) | {h['picp']:.4f} |",
        f"| PINAW (mean width) | {h['pinaw']:.4f} |",
        f"| ACE (PICP - 0.80) | {h['ace']:+.4f} |",
        f"| Winkler score | {h['winkler']:.4f} |",
        f"| Pinball total | {h['pinball_total']:.5f} |",
        f"| RMSE skill vs persistence | {h.get('rmse_skill_vs_persistence', float('nan')):.4f} |",
        "",
        "## Per regime",
        "",
        "| Regime | n | nRMSE % | PICP | PINAW | Winkler |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in results["per_regime"].items():
        if m.get("n", 0) == 0:
            lines.append(f"| {name} | 0 | -- | -- | -- | -- |")
        else:
            lines.append(f"| {name} | {m['n']} | {m['nrmse_pct']:.3f} | {m['picp']:.4f} "
                         f"| {m['pinaw']:.4f} | {m['winkler']:.4f} |")

    r = results["routing"]
    lines += [
        "",
        "## Routing diagnostics",
        "",
        f"- Mean gate share: " + ", ".join(f"{k} {v:.3f}" for k, v in r["mean_gate_share"].items()),
        f"- Argmax share: " + ", ".join(f"{k} {v:.3f}" for k, v in r["argmax_share"].items()),
        f"- Mean gate entropy: {r['mean_entropy']:.4f} (max {r['max_entropy']:.4f})",
        f"- Mean alpha (NWP vs expert uncertainty mix): {r['mean_alpha']:.4f}",
        f"- corr(sigma_total, |error|): {r.get('corr_sigma_total_vs_abs_error', float('nan')):.4f}",
        f"- corr(interval width, |error|): "
        f"{r.get('corr_interval_width_vs_abs_error', float('nan')):.4f}",
        "",
        "## Per-horizon error growth",
        "",
        "| Step | MAE | RMSE | PICP |",
        "|---|---|---|---|",
    ]
    ph = results["per_horizon"]
    for i in range(len(ph["mae"])):
        picp_i = ph.get("picp", [float("nan")] * len(ph["mae"]))[i]
        lines.append(f"| {i+1} | {ph['mae'][i]:.5f} | {ph['rmse'][i]:.5f} | {picp_i:.4f} |")

    path = REPORTS_DIR / "evaluation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote evaluation report -> {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained ATERNET checkpoint.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--checkpoint", default=str(BEST_CHECKPOINT_PATH))
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else None
    return evaluate(args.split, args.checkpoint, device)


if __name__ == "__main__":
    main()
