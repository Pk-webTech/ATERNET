"""
Aggregate the raw experiment outputs into one small, committable file that
the dashboard reads:  docs/results_summary.json

    python build_results_summary.py

Raw results live in git-ignored directories (outputs/, workspaces/*/outputs/),
so the app cannot rely on them being present on another machine. This script
is the single place numbers are aggregated; every figure in the dashboard's
"Results & Evidence" page comes from this file.

Sites
  yorkshire   outputs/                      (PV_Live region 23, measured)
  south_west  workspaces/sw/outputs/        (PV_Live region 22, measured)
  vellore     workspaces/vit/outputs/       (SIMULATED PV, demonstration only)
"""

import datetime
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
KEYS = ["nrmse_pct", "nmae_pct", "r2", "picp", "pinaw", "winkler"]

ATERNET_LABELS = [
    ("tune_ema", "ATERNET (final: residual gate + shortcut + EMA)"),
    ("full", "ATERNET without EMA"),
    ("no_dense_expert", "  ablation: no dense expert loss"),
    ("no_gate", "  ablation: no gate (uniform average)"),
    ("fixed_temp", "  ablation: fixed temperature"),
    ("fixed_width", "  ablation: fixed interval width"),
    ("sigma_nwp_only", "  ablation: sigma_NWP only"),
    ("sigma_exp_only", "  ablation: sigma_expert only"),
    ("single_expert", "  ablation: single expert"),
    ("no_regime_weights", "  ablation: no regime weights"),
]
BASELINE_LABELS = [("lgbm", "LightGBM quantile"), ("patchtst", "PatchTST quantile"),
                   ("lstm", "LSTM quantile")]


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _headlines(pattern):
    return [_load(f)["headline"] for f in sorted(glob.glob(str(pattern)))]


def _row(label, runs, kind):
    if not runs:
        return None
    row = {"model": label, "kind": kind, "n_seeds": len(runs),
           "per_seed_nrmse": [round(r["nrmse_pct"], 3) for r in runs]}
    for k in KEYS:
        v = np.array([r[k] for r in runs if k in r], float)
        row[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return row


def _calibration(metrics_dir):
    p = Path(metrics_dir) / "conformal_calibration.json"
    if not p.exists():
        return None
    c = _load(p)
    return {
        "picp_daytime_before": c.get("picp_daytime_before"),
        "picp_daytime_after": c.get("picp_daytime_after"),
        "winkler_before": c["headline_before"]["winkler"],
        "winkler_after": c["headline_after"]["winkler"],
        "pinaw_before": c["headline_before"]["pinaw"],
        "pinaw_after": c["headline_after"]["pinaw"],
        "per_regime": {k: {"n": v["n"], "picp_daytime_before": v.get("picp_daytime_before"),
                           "picp_daytime_after": v.get("picp_daytime_after")}
                       for k, v in c["per_regime_comparison"].items()},
    }


def _site(outputs: Path, title, simulated, note, with_ablations, aternet_dir="multiseed_final"):
    m = outputs / "metrics"
    rows = []
    for key, label in ATERNET_LABELS:
        if not with_ablations and key != "tune_ema":
            continue
        r = _row(label, _headlines(m / aternet_dir / f"seed*/ablation_{key}.json"),
                 "aternet" if not label.startswith("  ") else "ablation")
        if r:
            rows.append(r)
    for key, label in BASELINE_LABELS:
        r = _row(label, _headlines(m / f"multiseed_baselines/seed*/baseline_{key}.json"), "baseline")
        if r:
            rows.append(r)
    site = {"title": title, "simulated": simulated, "note": note, "rows": rows,
            "calibration": _calibration(m)}
    return site if rows else None


def main():
    sites = {}
    y = _site(ROOT / "outputs", "Yorkshire (PV_Live region 23)", False,
              "Measured regional PV; training 2023-01 to 2026-09; seeds 42-44 (LSTM: 42-47).", True)
    if y:
        sites["yorkshire"] = y
    sw = _site(ROOT / "workspaces/sw/outputs", "South West England (PV_Live region 22)", False,
               "Second measured region; hyperparameters unchanged from Yorkshire; seeds 42-44.", False)
    if sw:
        sites["south_west"] = sw

    vit_m = ROOT / "workspaces/vit/outputs/metrics"
    vit_seeds = glob.glob(str(vit_m / "multiseed_final/seed*/ablation_tune_ema.json"))
    if len(vit_seeds) >= 2:
        v = _site(ROOT / "workspaces/vit/outputs", "VIT Vellore, India (SIMULATED PV)", True,
                  ("Generation is SIMULATED from ERA5 reanalysis with a physical PV model "
                   "(1 MWp illustrative plant); inputs are real multi-model weather forecasts. "
                   f"{len(vit_seeds)} seeds. Not a real-world accuracy result."), False)
        if v:
            sites["vellore"] = v
    elif (vit_m / "test_metrics.json").exists():
        t = _load(vit_m / "test_metrics.json")["headline"]
        rows = [{"model": "ATERNET (final configuration)", "kind": "aternet", "n_seeds": 1,
                 "per_seed_nrmse": [round(t["nrmse_pct"], 3)],
                 **{k: {"mean": float(t[k]), "std": 0.0} for k in KEYS}}]
        lg = vit_m / "multiseed_baselines/seed42/baseline_lgbm.json"
        if lg.exists():
            h = _load(lg)["headline"]
            rows.append({"model": "LightGBM quantile", "kind": "baseline", "n_seeds": 1,
                         "per_seed_nrmse": [round(h["nrmse_pct"], 3)],
                         **{k: {"mean": float(h[k]), "std": 0.0} for k in KEYS}})
        sites["vellore"] = {
            "title": "VIT Vellore, India (SIMULATED PV)", "simulated": True,
            "note": ("Generation is SIMULATED from ERA5 reanalysis with a physical PV model "
                     "(1 MWp illustrative plant); inputs are real multi-model weather forecasts. "
                     "Single seed. Not a real-world accuracy result."),
            "rows": rows, "calibration": _calibration(vit_m)}

    out = {"generated": datetime.date.today().isoformat(), "sites": sites}
    dest = ROOT / "docs" / "results_summary.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, s in sites.items():
        print(f"{name}: {len(s['rows'])} rows, calibration={'yes' if s['calibration'] else 'no'}")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
