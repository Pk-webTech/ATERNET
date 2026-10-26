"""
Architecture and flow diagrams for the report (Chapters 2, 4 and 5).

    python docs/report/make_diagrams.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_helpers import *  # noqa: F401,F403


# ---------------------------------------------------------------- Fig 4.1 system architecture
def system_architecture():
    fig, ax = canvas(6.4, 6.6, 100, 104)

    band(ax, 1, 84, 99, 103, "External data sources", "grey")
    s1 = box(ax, 20, 92.5, 29, 12, "Sheffield Solar PV_Live\nmeasured regional PV\n30-min, with capacity", "amber", 7.5)
    s2 = box(ax, 52, 92.5, 31, 12, "Open-Meteo forecast archive\n5 models: GFS, ICON, UKMO,\nMétéo-France, GEM", "amber", 7.5)
    s3 = box(ax, 83, 92.5, 26, 12, "ERA5 reanalysis\n(simulated demo site only)", "amber", 7.5)

    band(ax, 1, 60, 99, 80, "Data layer  (Phase 1)", "blue")
    d1 = box(ax, 14, 69, 22, 11, "Acquisition\ndownload_pv.py\ndownload_nwp.py", "blue", 7.5)
    d2 = box(ax, 40, 69, 24, 11, "Cleaning and\nfeature engineering\nclear-sky index, σ_NWP,\nweather regimes", "blue", 7)
    d3 = box(ax, 66, 69, 24, 11, "Windowing and scaling\n24 h in → 8 h out\nchronological 70/15/15", "blue", 7)
    d4 = box(ax, 90, 69, 15, 11, "Datasets\n(.npz) and\nscalers", "white", 7.5, ec="#2E5A88")
    arrow(ax, edge(d1, "r"), edge(d2, "l"))
    arrow(ax, edge(d2, "r"), edge(d3, "l"))
    arrow(ax, edge(d3, "r"), edge(d4, "l"))
    for s in (s1, s2):
        arrow(ax, edge(s, "b"), (edge(s, "b")[0], 80), lw=1.0)
    arrow(ax, edge(s3, "b"), (edge(s3, "b")[0], 80), lw=1.0, ls="--")
    arrow(ax, (20, 84), (14, 74.5), lw=0.01, ms=1, color="white")

    band(ax, 1, 28, 99, 56, "Model and experiment layer", "green", ha="right")
    m1 = box(ax, 18, 44, 30, 16, "ATERNET model\nbackbone → experts → router\n→ probabilistic decoder\n(P10, P50, P90)", "green", 7.5, bold=True)
    m2 = box(ax, 50, 49.5, 26, 9, "Trainer\nAdamW, cosine LR, EMA,\nearly stopping", "green", 7)
    m3 = box(ax, 50, 36.5, 26, 9, "Baselines\nLightGBM, PatchTST, LSTM", "green", 7.5)
    m4 = box(ax, 81, 49.5, 30, 9, "Evaluation\nnRMSE, nMAE, R², PICP,\nPINAW, Winkler", "green", 7)
    m5 = box(ax, 81, 36.5, 30, 9, "Conformal calibration\nper regime, daytime points", "green", 7.5)
    arrow(ax, (8, 60), (8, 52.2), ls="-")
    arrow(ax, edge(m2, "l"), (33, 49.5), text="trains", fs=6.5, text_off=(0, 1.8))
    arrow(ax, edge(m2, "r"), edge(m4, "l"))
    arrow(ax, edge(m3, "r"), (66, 36.5), lw=0.01, ms=1, color="white")
    arrow(ax, edge(m4, "b"), edge(m5, "t"))
    arrow(ax, (33, 36.5), edge(m3, "l"), ls="--", text="compared with", fs=6.5, text_off=(0, 1.8))

    band(ax, 1, 3, 99, 24, "Application layer", "violet")
    a1 = box(ax, 20, 12.5, 30, 11, "Streamlit dashboard\nOverview, Is ATERNET better?,\nIndia demo, explorer, safety checks", "violet", 7)
    a2 = box(ax, 52, 12.5, 26, 11, "predict.py\nlive forecast and\nbacktest (CLI)", "violet", 7.5)
    a3 = box(ax, 82, 12.5, 28, 11, "Results store\nmetrics JSON, reports,\nresults_summary.json", "violet", 7.5)
    arrow(ax, (50, 28), (50, 24), lw=1.2)
    arrow(ax, (20, 28), (20, 24), lw=1.2)
    arrow(ax, (82, 28), (82, 24), lw=1.2)

    arrow(ax, (50, 60), (50, 56), lw=1.2, text="windows", text_off=(6, 0))
    return save(fig, "fig_4_1_system_architecture.png")


# ---------------------------------------------------------------- Fig 4.2 model architecture
def model_architecture():
    fig, ax = canvas(6.5, 7.8, 100, 124)
    X = (16, 40, 64)

    # inputs
    box(ax, X[0], 116, 21, 9, "X_seq  (48 × 10)\npast PV, weather,\ntime features", "white", 6.8, ec="#555555")
    box(ax, X[1], 116, 21, 9, "X_nwp  (64 × 4)\nforecast weather\n(past 24 h + next 8 h)", "white", 6.8, ec="#555555")
    box(ax, X[2], 116, 21, 9, "X_sigma  (48 × 6)\nforecast-spread\nfeatures", "white", 6.8, ec="#555555")

    # backbone
    band(ax, 3, 86, 77, 106, "Temporal backbone  (Phase 2)", "blue", 7.6)
    box(ax, X[0], 94, 21, 11, "Patch embedding\n+ 3-layer Transformer\nencoder (8 heads)", "white", 6.6, ec="#2E5A88")
    box(ax, X[1], 94, 21, 11, "Cross-attention\nover forecast\nweather", "white", 6.6, ec="#2E5A88")
    box(ax, X[2], 94, 21, 11, "σ encoder\n(small MLP)", "white", 6.6, ec="#2E5A88")
    for x in X:
        arrow(ax, (x, 111.5), (x, 99.5), lw=1.0)

    # embeddings
    box(ax, X[0], 78, 21, 6.5, "h_expert  (128)", "white", 6.8, ec="#555555")
    box(ax, X[1], 78, 21, 6.5, "h_nwp  (32)", "white", 6.8, ec="#555555")
    box(ax, X[2], 78, 21, 6.5, "h_sigma  (32)", "white", 6.8, ec="#555555")
    for x in X:
        arrow(ax, (x, 88.5), (x, 81.3), lw=1.0)

    # expert pool
    band(ax, 3, 52, 77, 72, "Expert pool  (Phase 3)", "green", 7.6)
    for k, (name, xx) in enumerate([("Dawn ramp", 14), ("Clear peak", 33.5), ("Overcast", 53), ("Volatile", 69)]):
        box(ax, xx, 59.5, 16.5 if k < 3 else 13, 7, name, "white", 6.8, ec="#2F7D4F")
    arrow(ax, (X[0], 74.7), (X[0], 72.0), lw=1.0)
    ax.text(21, 73.4, "all four experts read h_expert", fontsize=5.8, ha="left", va="center")

    # router
    band(ax, 3, 22, 77, 46, "Adaptive router  (Phase 4)", "amber", 7.6)
    box(ax, 16, 32, 21, 12, "Uncertainty fusion\nσ_NWP + σ_expert\n→ σ_total, α", "white", 6.4, ec="#B7791F")
    box(ax, 40, 32, 21, 12, "Gate\nsoftmax(δ·logits / T)\nT = exp(softplus(β)·σ_total)", "white", 6.0, ec="#B7791F")
    box(ax, 64, 32, 21, 12, "Weighted sum of\nthe four experts\n→ ŷ_routed", "white", 6.4, ec="#B7791F")
    arrow(ax, (26.5, 32), (29.5, 32), lw=1.0)
    arrow(ax, (50.5, 32), (53.5, 32), lw=1.0)
    arrow(ax, (40, 52), (40, 46.0), lw=1.2)
    ax.text(41.5, 49.3, "4 × 16 forecasts\n+ expert disagreement σ_expert", fontsize=5.8, ha="left", va="center")

    # shortcut (outside the band)
    box(ax, 89, 32, 20, 18, "Last-step\nlinear shortcut\n(zero-initialised):\nreads latest PV step\nand forecast weather", "violet", 6.0)
    arrow(ax, (79, 32), (74.5, 32), lw=1.0, color="#5B4A8A", ls="--")
    ax.text(77, 35.5, "+", fontsize=9, ha="center", fontweight="bold", color="#5B4A8A")

    # decoder
    band(ax, 3, 2, 77, 17, "Probabilistic decoder  (Phase 5)", "red", 7.6)
    box(ax, 16, 8, 21, 8, "P50 = ŷ + small,\nbounded correction", "white", 6.3, ec="#B83232")
    box(ax, 40, 8, 21, 8, "Half-widths grow\nwith σ_total", "white", 6.3, ec="#B83232")
    box(ax, 64, 8, 21, 8, "P10 = P50 − lower\nP90 = P50 + upper", "white", 6.3, ec="#B83232")
    arrow(ax, (40, 22), (40, 17.2), lw=1.2)
    ax.text(41.5, 19.6, "ŷ_routed", fontsize=5.8, ha="left", va="center")
    arrow(ax, (26.5, 8), (29.5, 8), lw=1.0)
    arrow(ax, (50.5, 8), (53.5, 8), lw=1.0)

    box(ax, 89, 8, 20, 9, "Output: P10, P50, P90\nfor the next 16\nhalf-hours", "white", 6.4, ec="#333333", bold=True)
    arrow(ax, (74.5, 8), (79, 8), lw=1.2)
    return save(fig, "fig_4_2_model_architecture.png")


# ---------------------------------------------------------------- DFD level 0
def dfd0():
    fig, ax = canvas(6.4, 3.9, 100, 62)
    proc = box(ax, 50, 31, 26, 22, "0\nATERNET\nForecasting System", "blue", 9, bold=True)
    e1 = box(ax, 12, 52, 22, 8, "PV_Live service", "amber", 8, rounded=False)
    e2 = box(ax, 12, 31, 22, 8, "Open-Meteo / ERA5", "amber", 8, rounded=False)
    e3 = box(ax, 88, 52, 22, 8, "Researcher / Student", "green", 8, rounded=False)
    e4 = box(ax, 88, 31, 22, 8, "Reviewer / Examiner", "green", 8, rounded=False)
    e5 = box(ax, 88, 10, 22, 8, "Grid / energy analyst", "green", 8, rounded=False)
    arrow(ax, edge(e1, "r"), (38, 38), text="measured PV, capacity", fs=6.5, text_off=(-1, 2.6))
    arrow(ax, edge(e2, "r"), edge(proc, "l"), text="weather forecasts, reanalysis", fs=6.5, text_off=(0, 2.4))
    arrow(ax, (77, 54.5), (63, 40.5), text="configuration,\ntraining requests", fs=6.5, text_off=(2, 4.5))
    arrow(ax, (63, 37.5), (77, 49.5), text="metrics, plots", fs=6.5, text_off=(-4.5, -1.5))
    arrow(ax, (63, 31), edge(e4, "l"), text="results, comparisons", fs=6.5, text_off=(0, 2.2))
    arrow(ax, (60, 20), edge(e5, "l"), text="forecasts with P10–P90 ranges", fs=6.5, text_off=(2, -3.2), rad=0.0)
    return save(fig, "fig_4_3_dfd_level0.png")


# ---------------------------------------------------------------- DFD level 1
def dfd1():
    fig, ax = canvas(6.5, 6.2, 100, 100)

    def proc(cx, cy, num, label, w=24, h=11):
        return box(ax, cx, cy, w, h, f"{num}\n{label}", "blue", 7.2)

    def store(cx, cy, label, w=24):
        ax.plot([cx - w / 2, cx + w / 2], [cy + 3.4, cy + 3.4], color="#2E5A88", lw=1.2, zorder=2)
        ax.plot([cx - w / 2, cx + w / 2], [cy - 3.4, cy - 3.4], color="#2E5A88", lw=1.2, zorder=2)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=6.8, zorder=3)
        return (cx, cy, w, 6.8)

    pv = box(ax, 12, 93, 20, 7, "PV_Live", "amber", 7.2, rounded=False)
    om = box(ax, 42, 93, 24, 7, "Open-Meteo / ERA5", "amber", 7.2, rounded=False)
    rs = box(ax, 88, 6, 22, 8, "Researcher /\nReviewer", "green", 7.2, rounded=False)

    p1 = proc(24, 78, "1.0", "Acquire data")
    d1 = store(24, 64, "D1  Raw PV and weather")
    p2 = proc(24, 50, "2.0", "Clean, build features\nand windows")
    d2 = store(24, 36, "D2  Datasets and scalers")
    p3 = proc(58, 36, "3.0", "Train model\n(EMA, early stopping)")
    d3 = store(58, 51, "D3  Checkpoints")
    p4 = proc(58, 66, "4.0", "Evaluate and\ncalibrate")
    d4 = store(88, 66, "D4  Metrics, reports", w=22)
    p5 = proc(88, 47, "5.0", "Produce forecasts\n(live / backtest)", w=20)
    p6 = proc(88, 24, "6.0", "Show results\n(dashboard)", w=20)

    arrow(ax, (12, 89.5), (16, 83.5), text="measured PV", fs=6, text_off=(-1, 0), ha="right")
    arrow(ax, (38, 89.5), (32, 83.5), text="forecast weather", fs=6, text_off=(0.5, 0), ha="left")
    arrow(ax, edge(p1, "b"), edge(d1, "t"))
    arrow(ax, edge(d1, "b"), edge(p2, "t"))
    arrow(ax, edge(p2, "b"), edge(d2, "t"))
    arrow(ax, edge(d2, "r"), edge(p3, "l"), text="train / val / test", fs=6, text_off=(0, 2.0))
    arrow(ax, edge(p3, "t"), edge(d3, "b"), text="best weights", fs=6, text_off=(-1.5, 0), ha="right")
    arrow(ax, edge(d3, "t"), edge(p4, "b"), text="model", fs=6, text_off=(1.5, 0), ha="left")
    arrow(ax, edge(p4, "r"), edge(d4, "l"), text="metrics,\ncalibration", fs=6, text_off=(0, 3.2))
    arrow(ax, edge(d3, "r"), edge(p5, "l"), text="model", fs=6, text_off=(0, 2.0))
    arrow(ax, edge(p5, "t"), edge(d4, "b"), text="forecasts", fs=6, text_off=(-1.5, 0), ha="right")
    # D4 -> P6 along the right edge
    ax.plot([99.0, 99.0], [66, 24], color="#333333", lw=1.0, zorder=3)
    ax.plot([99.0, 98.2], [66, 66], color="#333333", lw=1.0, zorder=3)
    arrow(ax, (99.0, 24), edge(p6, "r"), text="", lw=1.0)
    ax.text(99.6, 45, "results", rotation=90, fontsize=6, va="center", ha="left")
    arrow(ax, edge(p6, "b"), edge(rs, "t"), text="views", fs=6, text_off=(2.5, 0), ha="left")
    arrow(ax, (77, 6), (56, 30), text="training requests, settings", fs=6, text_off=(-1, 3.5), ha="center")
    return save(fig, "fig_4_4_dfd_level1.png")


if __name__ == "__main__":
    system_architecture()
    model_architecture()
    dfd0()
    dfd1()
