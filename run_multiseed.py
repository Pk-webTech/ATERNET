"""
Multi-seed ablation study: is the full model actually different from its
ablations, or is the gap seed noise?

    python run_multiseed.py [--seeds 42 43 44 45 46] [--epochs 50]
                            [--variants full no_gate ...]

Results go to outputs/metrics/multiseed/ and checkpoints/multiseed/ so the
single-seed comparison table is untouched. Finished (variant, seed) runs
are skipped on re-run, so an interrupted sweep can be resumed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.paths import METRICS_DIR, CHECKPOINTS_DIR, REPORTS_DIR
from src.baselines.ablations import ABLATION_NAMES, EXTRA_VARIANTS, run_ablation
from src.utils.io import save_json, load_json
from src.utils.logger import get_logger

logger = get_logger("run_multiseed")

MULTISEED_METRICS_DIR = METRICS_DIR / "multiseed"
MULTISEED_CKPT_DIR = CHECKPOINTS_DIR / "multiseed"
METRIC_KEYS = ["nrmse_pct", "nmae_pct", "r2", "picp", "pinaw", "winkler"]


def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed ablation sweep.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--variants", nargs="+", default=["full"] + ABLATION_NAMES,
                   choices=EXTRA_VARIANTS + ABLATION_NAMES)
    p.add_argument("--final-arch", action="store_true",
                   help="Run every variant on top of the final architecture "
                        "(residual gate + last-step skip). Results go to "
                        "multiseed_final/ and multiseed_ablation_final.md.")
    p.add_argument("--tag", default="",
                   help="Suffix for the output directories/report (e.g. 'stride2'), so a run "
                        "on a different dataset doesn't reuse or overwrite existing results.")
    p.add_argument("--report-name", default=None,
                   help="Report filename under outputs/reports/ (default depends on "
                        "--final-arch). Use it so a partial variant list doesn't "
                        "overwrite the full report.")
    return p.parse_args()


def _run_one(variant, seed, epochs, metrics_root, ckpt_root, base_kwargs):
    run_dir = metrics_root / f"seed{seed}"
    result_path = run_dir / f"ablation_{variant}.json"
    if result_path.exists():
        logger.info(f"[skip] {variant} seed {seed} already done")
        return load_json(result_path)
    return run_ablation(variant, epochs=epochs, seed=seed, metrics_dir=run_dir,
                        checkpoints_dir=ckpt_root / f"seed{seed}",
                        base_model_kwargs=base_kwargs)


def _summarise(per_variant):
    rows = {}
    for variant, runs in per_variant.items():
        row = {"n_seeds": len(runs)}
        for k in METRIC_KEYS:
            vals = np.array([r["headline"][k] for r in runs if k in r["headline"]], float)
            if len(vals):
                row[k] = {"mean": float(vals.mean()),
                          "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
        rows[variant] = row
    return rows


def _write_report(summary, seeds, path):
    lines = [
        f"# Multi-seed ablation (seeds: {', '.join(map(str, seeds))})", "",
        "Mean ± std across seeds. A variant only beats `full` if the gap "
        "exceeds the seed-to-seed spread.", "",
        "| Variant | n | nRMSE % | nMAE % | R² | PICP | PINAW | Winkler |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variant, row in summary.items():
        cells = [f"{row[k]['mean']:.4f} ± {row[k]['std']:.4f}" if k in row else "—"
                 for k in METRIC_KEYS]
        lines.append(f"| {variant} | {row['n_seeds']} | " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    per_variant = {v: [] for v in args.variants}

    suffix = f"_{args.tag}" if args.tag else ""
    if args.final_arch:
        metrics_root = METRICS_DIR / f"multiseed_final{suffix}"
        ckpt_root = CHECKPOINTS_DIR / f"multiseed_final{suffix}"
        base_kwargs = {"residual_gate": True, "last_step_skip": True}
        report_name = f"multiseed_ablation_final{suffix}.md"
    else:
        metrics_root = MULTISEED_METRICS_DIR.with_name(MULTISEED_METRICS_DIR.name + suffix)
        ckpt_root = MULTISEED_CKPT_DIR.with_name(MULTISEED_CKPT_DIR.name + suffix)
        base_kwargs = None
        report_name = f"multiseed_ablation{suffix}.md"

    for seed in args.seeds:
        for variant in args.variants:
            try:
                per_variant[variant].append(
                    _run_one(variant, seed, args.epochs, metrics_root, ckpt_root, base_kwargs))
            except Exception as e:
                logger.error(f"{variant} seed {seed} failed: {e}")

    per_variant = {v: r for v, r in per_variant.items() if r}
    summary = _summarise(per_variant)
    save_json(summary, metrics_root / "summary.json")
    report_path = REPORTS_DIR / (args.report_name or report_name)
    _write_report(summary, args.seeds, report_path)
    logger.info(f"Summary -> {metrics_root / 'summary.json'}")
    logger.info(f"Report  -> {report_path}")


if __name__ == "__main__":
    main()
