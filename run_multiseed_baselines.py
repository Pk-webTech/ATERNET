"""
Run the baselines (LightGBM, LSTM, PatchTST) over several seeds so they can
be compared with the multi-seed ATERNET numbers on equal footing.

    python run_multiseed_baselines.py [--seeds 42 43 44] [--epochs 50]
                                      [--models lgbm lstm patchtst]

The baseline modules hardcode seed 42 and write to outputs/metrics/, which
comparison_table.py reads. Rather than edit them, this script patches their
seed and output directories at runtime. Results land in
outputs/metrics/multiseed_baselines/seed<N>/ and
outputs/reports/multiseed_baselines.md. Finished runs are skipped.
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.paths import METRICS_DIR, CHECKPOINTS_DIR, REPORTS_DIR
from src.utils.io import load_json, save_json
from src.utils.seed import set_seed as real_set_seed
from src.utils.logger import get_logger

logger = get_logger("run_multiseed_baselines")

OUT_DIR = METRICS_DIR / "multiseed_baselines"
CKPT_DIR = CHECKPOINTS_DIR / "multiseed_baselines"
METRIC_KEYS = ["nrmse_pct", "nmae_pct", "r2", "picp", "pinaw", "winkler"]

MODELS = {
    # name: (module, result file written by the module, takes epochs?)
    "lgbm": ("src.baselines.lgbm_quantile", "baseline_lgbm.json", False),
    "lstm": ("src.baselines.lstm_quantile", "baseline_lstm.json", True),
    "patchtst": ("src.baselines.patchtst_quantile", "baseline_patchtst.json", True),
}


def _run(name, seed, epochs):
    module_name, fname, takes_epochs = MODELS[name]
    run_dir = OUT_DIR / f"seed{seed}"
    result_path = run_dir / fname
    if result_path.exists():
        logger.info(f"[skip] {name} seed {seed} already done")
        return load_json(result_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CKPT_DIR / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mod = importlib.import_module(module_name)
    saved = {}

    def patch(obj, attr, value):
        saved[(id(obj), attr)] = (obj, attr, getattr(obj, attr))
        setattr(obj, attr, value)

    patch(mod, "METRICS_DIR", run_dir)
    if hasattr(mod, "CHECKPOINTS_DIR"):
        patch(mod, "CHECKPOINTS_DIR", ckpt_dir)
    if hasattr(mod, "set_seed"):
        patch(mod, "set_seed", lambda _s, _seed=seed: real_set_seed(_seed))
    if name == "lgbm":
        real_train = mod.lgb.train
        patch(mod.lgb, "train",
              lambda params, *a, _seed=seed, **k: real_train({**params, "seed": _seed}, *a, **k))

    try:
        logger.info(f"=== {name} seed {seed} ===")
        res = mod.train_and_evaluate(epochs=epochs) if takes_epochs else mod.train_and_evaluate()
    finally:
        for obj, attr, original in saved.values():
            setattr(obj, attr, original)
    return res


def _summarise(per_model):
    rows = {}
    for name, runs in per_model.items():
        row = {"n_seeds": len(runs)}
        for k in METRIC_KEYS:
            vals = np.array([r["headline"][k] for r in runs if k in r["headline"]], float)
            if len(vals):
                row[k] = {"mean": float(vals.mean()),
                          "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
        rows[name] = row
    return rows


def main():
    p = argparse.ArgumentParser(description="Multi-seed baselines.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = p.parse_args()

    per_model = {m: [] for m in args.models}
    for seed in args.seeds:
        for m in args.models:
            try:
                per_model[m].append(_run(m, seed, args.epochs))
            except Exception as e:
                logger.error(f"{m} seed {seed} failed: {e}")

    per_model = {m: r for m, r in per_model.items() if r}
    summary = _summarise(per_model)
    save_json(summary, OUT_DIR / "summary.json")

    lines = [f"# Multi-seed baselines (seeds: {', '.join(map(str, args.seeds))})", "",
             "| Model | n | nRMSE % | nMAE % | R² | PICP | PINAW | Winkler |",
             "|---|---|---|---|---|---|---|---|"]
    for name, row in summary.items():
        cells = [f"{row[k]['mean']:.4f} ± {row[k]['std']:.4f}" if k in row else "—"
                 for k in METRIC_KEYS]
        lines.append(f"| {name} | {row['n_seeds']} | " + " | ".join(cells) + " |")
    (REPORTS_DIR / "multiseed_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Report -> {REPORTS_DIR / 'multiseed_baselines.md'}")


if __name__ == "__main__":
    main()
