"""
Build one consistent multi-seed comparison table from the saved sweeps.

    python build_final_table.py

comparison_table.py compares single-seed (seed 42) baseline files, which
hid the LSTM's seed instability. This reads the multi-seed sweeps instead
and writes outputs/reports/final_comparison.md:

  - ATERNET variants   <- outputs/metrics/multiseed_final/seed*/ablation_<v>.json
  - baselines          <- outputs/metrics/multiseed_baselines/seed*/baseline_<m>.json
"""

import glob
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.paths import METRICS_DIR, REPORTS_DIR
from src.utils.io import load_json

KEYS = ["nrmse_pct", "nmae_pct", "r2", "picp", "pinaw", "winkler"]

ATERNET_ROWS = [
    ("ATERNET (EMA) -- main", "tune_ema"),
    ("ATERNET (no EMA)", "full"),
    ("  - no dense expert loss", "no_dense_expert"),
    ("  - no gate (uniform)", "no_gate"),
    ("  - fixed temperature", "fixed_temp"),
    ("  - fixed width", "fixed_width"),
    ("  - sigma_NWP only", "sigma_nwp_only"),
    ("  - sigma_expert only", "sigma_exp_only"),
    ("  - single expert", "single_expert"),
    ("  - no regime weights", "no_regime_weights"),
]
BASELINE_ROWS = [("LightGBM quantile", "lgbm"), ("PatchTST quantile", "patchtst"),
                 ("LSTM quantile", "lstm")]


def _collect(pattern):
    return [load_json(f)["headline"] for f in sorted(glob.glob(str(pattern)))]


def _row(label, runs):
    if not runs:
        return None
    cells = []
    for k in KEYS:
        v = np.array([r[k] for r in runs if k in r], float)
        cells.append(f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}")
    return f"| {label} | {len(runs)} | " + " | ".join(cells) + " |"


def main():
    lines = ["# Final multi-seed comparison (mean ± std over seeds)", "",
             "| Model | seeds | nRMSE % | nMAE % | R² | PICP | PINAW | Winkler |",
             "|---|---|---|---|---|---|---|---|"]
    for label, v in ATERNET_ROWS:
        row = _row(label, _collect(METRICS_DIR / f"multiseed_final/seed*/ablation_{v}.json"))
        if row:
            lines.append(row)
    for label, m in BASELINE_ROWS:
        row = _row(label, _collect(METRICS_DIR / f"multiseed_baselines/seed*/baseline_{m}.json"))
        if row:
            lines.append(row)

    lstm = _collect(METRICS_DIR / "multiseed_baselines/seed*/baseline_lstm.json")
    if lstm:
        per_seed = ", ".join(f"{r['nrmse_pct']:.2f}" for r in lstm)
        lines += ["", f"LSTM nRMSE per seed: {per_seed}. The mean is dominated by failed "
                      "runs; judge it by the per-seed values."]

    out = REPORTS_DIR / "final_comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
