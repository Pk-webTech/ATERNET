"""
Result charts for the report (Chapters 5 and 7), built from the saved experiment outputs.

    python docs/report/make_charts.py            # run from the default (Yorkshire) workspace
"""

import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_helpers import *  # noqa: F401,F403

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SUMMARY = json.loads((ROOT / "docs" / "results_summary.json").read_text(encoding="utf-8"))
M = ROOT / "outputs" / "metrics"
BLUE, GREY, GREEN, RED, AMBER = "#2E5A88", "#9AA3AD", "#2F7D4F", "#B83232", "#B7791F"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.2, "axes.spines.top": False,
                     "axes.spines.right": False})


def rows_of(site):
    return {r["model"].strip(): r for r in SUMMARY["sites"][site]["rows"]}


def seeds(pattern):
    return [json.load(open(f))["headline"] for f in sorted(glob.glob(str(pattern)))]


# ---------------------------------------------------------------- comparison bars (Fig 7.1 / 7.2)
def comparison(site, fname, title):
    rows = [r for r in SUMMARY["sites"][site]["rows"] if r["kind"] in ("aternet", "baseline")
            and r["model"] != "ATERNET without EMA"]
    names = [r["model"].replace("ATERNET (final: residual gate + shortcut + EMA)", "ATERNET").replace(" quantile", "")
             for r in rows]
    cols = [BLUE if r["kind"] == "aternet" else GREY for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.8))
    for ax, key, lab, nd in ((axs[0], "nrmse_pct", "Typical error, nRMSE (%)", 2), (axs[1], "winkler", "Winkler score", 4)):
        means = [r[key]["mean"] for r in rows]
        stds = [r[key]["std"] for r in rows]
        ax.bar(names, means, yerr=stds, color=cols, edgecolor="#333", lw=0.6, capsize=3, error_kw=dict(lw=0.9))
        for i, (m, r) in enumerate(zip(means, rows)):
            ax.text(i, m + (max(means) * 0.02) + stds[i], f"{m:.{nd}f}", ha="center", va="bottom", fontsize=7)
            if key == "nrmse_pct" and len(r["per_seed_nrmse"]) > 1 and np.ptp(r["per_seed_nrmse"]) > 1.0:
                ax.scatter([i] * len(r["per_seed_nrmse"]), r["per_seed_nrmse"], s=10, color="#222", zorder=5, marker="o")
        ax.set_ylabel(lab)
        ax.set_ylim(0, max(np.array(means) + np.array(stds)) * 1.22)
        ax.tick_params(axis="x", rotation=0)
    axs[0].set_title("Accuracy (lower is better)")
    axs[1].set_title("Quality of the forecast range (lower is better)")
    fig.suptitle(title, fontsize=8.5, y=1.0)
    fig.tight_layout()
    return save(fig, fname)


# ---------------------------------------------------------------- development of the model (Fig 7.3)
def development():
    steps = [
        ("Original\nlearned gate", seeds(M / "multiseed/seed*/ablation_full.json")),
        ("Uniform average\n(no gate)", seeds(M / "multiseed/seed*/ablation_no_gate.json")),
        ("Residual\ngate", seeds(M / "multiseed/seed*/ablation_residual_gate.json")),
        ("+ last-step\nshortcut", seeds(M / "multiseed/seed*/ablation_residual_gate_skip.json")),
        ("+ EMA weights\n(final)", seeds(M / "multiseed_final/seed*/ablation_tune_ema.json")),
    ]
    means = [np.mean([h["nrmse_pct"] for h in s]) for _, s in steps]
    stds = [np.std([h["nrmse_pct"] for h in s], ddof=1) for _, s in steps]
    lgbm = rows_of("yorkshire")["LightGBM quantile"]["nrmse_pct"]["mean"]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    xs = np.arange(len(steps))
    ax.errorbar(xs, means, yerr=stds, fmt="o-", color=BLUE, ecolor="#222", capsize=3, lw=1.5, ms=6, mfc=BLUE)
    for i, (m, sd) in enumerate(zip(means, stds)):
        ax.text(i, m + sd + 0.05, f"{m:.2f}", ha="center", fontsize=7.5)
    ax.axhline(lgbm, color=RED, ls="--", lw=1.0)
    ax.text(-0.35, lgbm - 0.04, f"LightGBM {lgbm:.2f}", color=RED, fontsize=7, ha="left", va="top")
    ax.set_xticks(xs)
    ax.set_xticklabels([t for t, _ in steps])
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_ylim(3.4, 4.85)
    ax.set_ylabel("nRMSE (%), mean ± std over 3 seeds")
    ax.grid(color="#EEEEEE")
    fig.tight_layout()
    return save(fig, "fig_7_3_model_development.png")


# ---------------------------------------------------------------- ablations (Fig 7.4)
def ablations():
    rows = rows_of("yorkshire")
    ref = rows["ATERNET without EMA"]
    abl = [r for r in SUMMARY["sites"]["yorkshire"]["rows"] if r["kind"] == "ablation"]
    names = [r["model"].replace("ablation:", "").strip() for r in abl]
    d_n = [r["nrmse_pct"]["mean"] - ref["nrmse_pct"]["mean"] for r in abl]
    e_n = [2 * max(r["nrmse_pct"]["std"], ref["nrmse_pct"]["std"]) for r in abl]
    d_w = [r["winkler"]["mean"] - ref["winkler"]["mean"] for r in abl]
    e_w = [2 * max(r["winkler"]["std"], ref["winkler"]["std"]) for r in abl]
    y = np.arange(len(abl))[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.3), sharey=True)
    for ax, d, e, lab in ((axs[0], d_n, e_n, "Change in nRMSE (points)"), (axs[1], d_w, e_w, "Change in Winkler score")):
        cols = [RED if dd > ee else GREY for dd, ee in zip(d, e)]
        ax.barh(y, d, color=cols, edgecolor="#333", lw=0.5)
        ax.errorbar(d, y, xerr=e, fmt="none", ecolor="#222", lw=0.8, capsize=2)
        ax.axvline(0, color="#222", lw=0.8)
        ax.set_xlabel(lab + "\n(right of 0 = worse without this part)")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(names)
    fig.tight_layout()
    return save(fig, "fig_7_4_ablation.png")


# ---------------------------------------------------------------- daytime coverage (Fig 7.5)
def coverage():
    sites = [("yorkshire", "Yorkshire"), ("south_west", "South West"), ("vellore", "Vellore\n(simulated)")]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    x = np.arange(len(sites))
    b = [SUMMARY["sites"][k]["calibration"]["picp_daytime_before"] * 100 for k, _ in sites]
    a = [SUMMARY["sites"][k]["calibration"]["picp_daytime_after"] * 100 for k, _ in sites]
    ax.bar(x - 0.18, b, 0.34, color=GREY, edgecolor="#333", lw=0.6, label="Raw ranges")
    ax.bar(x + 0.18, a, 0.34, color=BLUE, edgecolor="#333", lw=0.6, label="After calibration")
    for i, (bb, aa) in enumerate(zip(b, a)):
        ax.text(i - 0.18, bb + 0.8, f"{bb:.0f}", ha="center", fontsize=7.5)
        ax.text(i + 0.18, aa + 0.8, f"{aa:.0f}", ha="center", fontsize=7.5)
    ax.axhline(80, color=RED, ls="--", lw=1.0)
    ax.text(len(sites) - 0.5, 81, "target 80%", color=RED, fontsize=7, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in sites])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Real value inside the P10–P90 range (%)\ndaytime points only")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    return save(fig, "fig_7_5_daytime_coverage.png")


# ---------------------------------------------------------------- predictions-based charts
def prediction_charts():
    from src.training.dataset import ATERNETDataset
    from src.evaluation.conformal import daytime_mask
    from src.utils.constants import ID_TO_REGIME

    P = np.load(ROOT / "outputs" / "predictions" / "test_predictions.npz", allow_pickle=True)
    y, p10, p50, p90, reg = P["y_true"], P["p10"], P["p50"], P["p90"], P["regime_id"]
    day = daytime_mask(ATERNETDataset("test"))
    adj = json.load(open(M / "conformal_calibration.json"))["adjustments"]
    p10c, p90c = p10.copy(), p90.copy()
    for rid, name in ID_TO_REGIME.items():
        m = (reg == rid)[:, None] & day
        p10c = np.where(m, p10c - adj[name], p10c)
        p90c = np.where(m, p90c + adj[name], p90c)

    # --- reliability (Fig 7.6)
    levels = [0.1, 0.5, 0.9]
    emp = lambda q: [float((y[day] <= qq[day]).mean()) for qq in q]
    raw = emp([p10, p50, p90])
    cal = emp([p10c, p50, p90c])
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=0.9, label="perfect")
    ax.plot(levels, raw, "o-", color=GREY, lw=1.4, label="raw (daytime)")
    ax.plot(levels, cal, "s-", color=BLUE, lw=1.4, label="calibrated (daytime)")
    for lv, r_, c_ in zip(levels, raw, cal):
        ax.text(lv - 0.03, r_ + 0.035, f"{r_:.2f}", fontsize=6.8, color="#555", ha="right")
        ax.text(lv + 0.03, c_ - 0.06, f"{c_:.2f}", fontsize=6.8, color=BLUE, ha="left")
    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Observed share of values below the quantile")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_7_6_reliability_daytime.png")

    # --- forecast examples (Fig 7.7)
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.4), sharey=True)
    hours = np.arange(1, y.shape[1] + 1) * 0.5
    for ax, (rid, name) in zip(axs.ravel(), ID_TO_REGIME.items()):
        ids = np.where((reg == rid) & (y.max(axis=1) > 0.25))[0]
        i = ids[int(len(ids) * 0.4)]
        ax.fill_between(hours, p10[i], p90[i], color=BLUE, alpha=0.2, label="P10–P90 range")
        ax.plot(hours, p50[i], color=BLUE, lw=1.6, label="Best guess (P50)")
        ax.plot(hours, y[i], color=RED, lw=1.4, ls=":", marker="o", ms=2.5, label="What happened")
        ax.set_title({"dawn_ramp": "Morning ramp-up", "clear_peak": "Clear sunny midday",
                      "overcast": "Cloudy / overcast", "volatile": "Patchy, changing clouds"}[name])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color="#EEEEEE")
    for ax in axs[1]:
        ax.set_xlabel("Hours ahead")
    for ax in axs[:, 0]:
        ax.set_ylabel("Output (fraction of capacity)")
    axs[0, 0].legend(frameon=False, loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_7_7_forecast_examples.png")

    # --- gate usage by regime (Fig 7.8)
    G = P["gate_weights"]
    mat = np.array([G[reg == rid].mean(axis=0) for rid in ID_TO_REGIME])
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=0.6)
    names = ["Dawn ramp", "Clear peak", "Overcast", "Volatile"]
    ax.set_xticks(range(4)); ax.set_xticklabels(names, rotation=20)
    ax.set_yticks(range(4)); ax.set_yticklabels(names)
    ax.set_xlabel("Expert"); ax.set_ylabel("Actual weather situation")
    for r_ in range(4):
        for c_ in range(4):
            ax.text(c_, r_, f"{mat[r_, c_]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if mat[r_, c_] > 0.35 else "#222")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="average trust")
    fig.tight_layout()
    save(fig, "fig_7_8_gate_by_regime.png")

    # --- uncertainty vs error (Fig 7.9)
    sig = P["sigma_total"]
    err = np.abs(y - p50).mean(axis=1)
    width = (p90 - p10).mean(axis=1)
    routing = json.load(open(M / "test_metrics.json"))["routing"]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.9))
    for ax, xdata, lab, c in ((axs[0], sig, "Model's unsureness score (σ_total)", routing["corr_sigma_total_vs_abs_error"]),
                              (axs[1], width, "Width of the forecast range", routing["corr_interval_width_vs_abs_error"])):
        ax.scatter(xdata, err, s=3, alpha=0.25, color=BLUE, edgecolors="none")
        ax.set_xlabel(lab)
        ax.set_ylabel("Average error of the forecast")
        ax.text(0.04, 0.92, f"correlation = {c:.2f}", transform=ax.transAxes, fontsize=7.5,
                bbox=dict(fc="white", ec="#BBB", pad=2))
    fig.tight_layout()
    save(fig, "fig_7_9_uncertainty_vs_error.png")

    # --- per-horizon RMSE vs LightGBM (Fig 7.10)
    t = json.load(open(M / "test_metrics.json"))["per_horizon"]["rmse"]
    lg = json.load(open(M / "baseline_lgbm.json"))["per_horizon"]["rmse"]
    h = np.arange(1, len(t) + 1) * 0.5
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ax.plot(h, np.array(t) * 100, "o-", color=BLUE, lw=1.5, ms=3, label="ATERNET")
    ax.plot(h, np.array(lg) * 100, "s--", color=GREY, lw=1.3, ms=3, label="LightGBM")
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel("RMSE (% of capacity)")
    ax.legend(frameon=False)
    ax.grid(color="#EEEEEE")
    fig.tight_layout()
    save(fig, "fig_7_10_error_by_horizon.png")

    # --- training history (Fig 7.11)
    hist = json.load(open(M / "train_history.json"))
    ep = [r["epoch"] for r in hist]
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ax.plot(ep, [r["train_pinball_total"] for r in hist], color=BLUE, lw=1.4, label="Training pinball loss")
    ax.plot(ep, [r["val_pinball_total"] for r in hist], color=RED, lw=1.4, label="Validation pinball loss (EMA weights)")
    best = int(np.argmin([r["val_pinball_total"] for r in hist]))
    ax.scatter([best], [hist[best]["val_pinball_total"]], color=RED, zorder=5, s=22)
    ax.annotate(f"best epoch {hist[best]['epoch']}", (best, hist[best]["val_pinball_total"]),
                xytext=(best + 1.5, hist[best]["val_pinball_total"] + 0.004), fontsize=7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pinball loss")
    ax.legend(frameon=False)
    ax.grid(color="#EEEEEE")
    fig.tight_layout()
    save(fig, "fig_7_11_training_history.png")


# ---------------------------------------------------------------- Vellore examples (Fig 7.13)
def vellore():
    P = np.load(ROOT / "workspaces" / "vit" / "outputs" / "predictions" / "test_predictions.npz", allow_pickle=True)
    y, p10, p50, p90, base = P["y_true"], P["p10"], P["p50"], P["p90"], P["baseline"]
    ids = np.where(y.max(axis=1) > 0.4)[0]
    pick = [ids[int(len(ids) * f)] for f in (0.2, 0.5, 0.8)]
    hours = np.arange(1, y.shape[1] + 1) * 0.5
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)
    for ax, i in zip(axs, pick):
        ax.fill_between(hours, p10[i], p90[i], color=BLUE, alpha=0.2, label="P10–P90 range")
        ax.plot(hours, p50[i], color=BLUE, lw=1.5, label="ATERNET best guess")
        ax.plot(hours, y[i], color=RED, lw=1.3, ls=":", marker="o", ms=2.3, label="Simulated output")
        ax.plot(hours, base[i], color="#888", lw=1.0, ls="--", label="Naive guess")
        ax.set_xlabel("Hours ahead")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color="#EEEEEE")
    axs[0].set_ylabel("Output (fraction of 1 MW)")
    axs[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.suptitle("SIMULATED plant output, VIT Vellore", fontsize=8, color=RED, y=1.0)
    fig.tight_layout()
    return save(fig, "fig_7_13_vellore_examples.png")


# ---------------------------------------------------------------- data description (Fig 5.3)
def data_week():
    df = pd.read_parquet(ROOT / "data" / "interim" / "features.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    w = df[(df["timestamp"] >= "2026-06-08") & (df["timestamp"] < "2026-06-15")]
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 3.6), sharex=True)
    axs[0].plot(w["timestamp"], w["pv_norm"], color=BLUE, lw=1.1)
    axs[0].set_ylabel("PV output\n(fraction of capacity)")
    axs[0].grid(color="#EEEEEE")
    axs[1].plot(w["timestamp"], w["nwp_ghi_mean"], color=AMBER, lw=1.1, label="mean of 5 forecast models")
    axs[1].fill_between(w["timestamp"], w["nwp_ghi_mean"] - w["nwp_ghi_std"], w["nwp_ghi_mean"] + w["nwp_ghi_std"],
                        color=AMBER, alpha=0.25, label="± spread between models")
    axs[1].set_ylabel("Forecast sunshine\n(irradiance, W/m²)")
    axs[1].legend(frameon=False, loc="upper right", fontsize=6.8)
    axs[1].grid(color="#EEEEEE")
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return save(fig, "fig_5_3_data_week.png")


def backtest_copy():
    src = ROOT / "outputs" / "plots" / "backtest_2026-10-01_2026-10-16.png"
    shutil.copy(src, FIG_DIR / "fig_7_12_backtest.png")
    print("copied backtest")


if __name__ == "__main__":
    comparison("yorkshire", "fig_7_1_comparison_yorkshire.png", "Yorkshire (measured PV): mean over 3 training runs; dots = LSTM runs")
    comparison("south_west", "fig_7_2_comparison_south_west.png", "South West England (measured PV): mean over 3 training runs")
    development()
    ablations()
    coverage()
    prediction_charts()
    backtest_copy()
    vellore()
    data_week()
