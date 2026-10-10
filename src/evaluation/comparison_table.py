"""
Generate a unified comparison table from all model results.

Reads:
  - outputs/metrics/test_metrics.json          (ATERNET full)
  - outputs/metrics/baseline_lgbm.json         (LightGBM)
  - outputs/metrics/baseline_lstm.json         (LSTM)
  - outputs/metrics/baseline_patchtst.json     (PatchTST)
  - outputs/metrics/ablation_*.json            (ablation variants)
  - outputs/metrics/conformal_calibration.json (CQR)

Writes:
  - outputs/reports/comparison_table.md
  - outputs/reports/comparison_table.json

Usage:
    python -m src.evaluation.comparison_table
"""

import json
from pathlib import Path

from config.paths import METRICS_DIR, REPORTS_DIR, ensure_dirs
from src.utils.io import load_json, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Metrics to include in the headline comparison
HEADLINE_COLS = ["nrmse_pct", "nmae_pct", "r2", "picp", "pinaw", "winkler",
                 "rmse_skill_vs_persistence"]
COL_LABELS = {
    "nrmse_pct": "nRMSE %",
    "nmae_pct": "nMAE %",
    "r2": "R²",
    "picp": "PICP",
    "pinaw": "PINAW",
    "winkler": "Winkler",
    "rmse_skill_vs_persistence": "Skill vs Persist.",
}


def _load_if_exists(path: Path) -> dict:
    if path.exists():
        return load_json(path)
    return None


def _extract_row(result: dict, name: str) -> dict:
    h = result.get("headline", result.get("headline_after", {}))
    row = {"model": name}
    for col in HEADLINE_COLS:
        row[col] = h.get(col, float("nan"))
    return row


def generate():
    ensure_dirs()
    rows = []

    # Full ATERNET
    full = _load_if_exists(METRICS_DIR / "test_metrics.json")
    if full:
        rows.append(_extract_row(full, "ATERNET (full)"))

    # CQR
    cqr = _load_if_exists(METRICS_DIR / "conformal_calibration.json")
    if cqr:
        rows.append(_extract_row(cqr, "ATERNET + CQR"))

    # Baselines
    for bname, fname in [("LightGBM Quantile", "baseline_lgbm.json"),
                          ("LSTM Quantile", "baseline_lstm.json"),
                          ("PatchTST Quantile", "baseline_patchtst.json")]:
        b = _load_if_exists(METRICS_DIR / fname)
        if b:
            rows.append(_extract_row(b, bname))

    # Ablations
    for p in sorted(METRICS_DIR.glob("ablation_*.json")):
        if p.name in ("ablations_all.json",) or "history" in p.name:
            continue
        a = _load_if_exists(p)
        if a and "headline" in a:
            abl_name = a.get("ablation", p.stem.replace("ablation_", ""))
            rows.append(_extract_row(a, f"ATERNET (–{abl_name})"))

    # Persistence and climatology baselines from the full model results
    if full and "headline" in full:
        h = full["headline"]
        if "baseline_rmse" in h:
            rows.append({
                "model": "Persistence",
                "nrmse_pct": h["baseline_rmse"] * 100,
                "nmae_pct": h["baseline_mae"] * 100,
                "r2": float("nan"),
                "picp": float("nan"),
                "pinaw": float("nan"),
                "winkler": float("nan"),
                "rmse_skill_vs_persistence": 0.0,
            })

    if not rows:
        logger.warning("No results found. Run training and baselines first.")
        return

    # Build markdown table
    lines = ["# Model Comparison", ""]
    header = "| Model | " + " | ".join(COL_LABELS[c] for c in HEADLINE_COLS) + " |"
    sep = "|---|" + "|".join(["---"] * len(HEADLINE_COLS)) + "|"
    lines.extend([header, sep])

    for row in rows:
        cells = [row["model"]]
        for col in HEADLINE_COLS:
            v = row.get(col, float("nan"))
            if col in ("nrmse_pct", "nmae_pct"):
                cells.append(f"{v:.2f}" if v == v else "—")
            elif col == "r2":
                cells.append(f"{v:.4f}" if v == v else "—")
            elif col in ("picp", "pinaw", "winkler", "rmse_skill_vs_persistence"):
                cells.append(f"{v:.4f}" if v == v else "—")
            else:
                cells.append(f"{v}" if v == v else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    # Per-regime comparison if CQR exists
    if cqr and "per_regime_comparison" in cqr:
        lines.extend(["## Per-Regime Calibration (before → after CQR)", ""])
        lines.append("| Regime | n | PICP before | PICP after | PINAW before | PINAW after |")
        lines.append("|---|---|---|---|---|---|")
        for rname, data in cqr["per_regime_comparison"].items():
            lines.append(
                f"| {rname} | {data['n']} "
                f"| {data['picp_before']:.3f} | {data['picp_after']:.3f} "
                f"| {data['pinaw_before']:.4f} | {data['pinaw_after']:.4f} |"
            )
        lines.append("")

    md_path = REPORTS_DIR / "comparison_table.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Comparison table -> {md_path}")

    # Also save as JSON for the Streamlit app
    save_json(rows, REPORTS_DIR / "comparison_table.json")
    return rows


if __name__ == "__main__":
    generate()