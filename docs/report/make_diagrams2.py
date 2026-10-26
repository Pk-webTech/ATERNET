"""
UML diagrams (use case, class, sequence), project plan and methodology flowcharts.

    python docs/report/make_diagrams2.py
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_helpers import *  # noqa: F401,F403
import matplotlib.dates as mdates
from matplotlib.patches import Polygon
import numpy as np


# ---------------------------------------------------------------- helpers
def actor(ax, x, y, label, fs=7.5):
    ax.add_patch(plt.Circle((x, y + 4.2), 1.3, fc="white", ec="#222", lw=1.0, zorder=3))
    for (x0, y0, x1, y1) in [(x, y + 2.9, x, y - 1.2), (x - 2.2, y + 1.6, x + 2.2, y + 1.6),
                             (x, y - 1.2, x - 1.8, y - 4.0), (x, y - 1.2, x + 1.8, y - 4.0)]:
        ax.plot([x0, x1], [y0, y1], color="#222", lw=1.0, zorder=3)
    ax.text(x, y - 6.0, label, ha="center", va="top", fontsize=fs, linespacing=1.1)
    return (x, y)


def oval(ax, cx, cy, w, h, text, fs=7.0, fc="white"):
    ax.add_patch(Ellipse((cx, cy), w, h, fc=C.get(fc, fc), ec="#2E5A88", lw=1.0, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3)
    return (cx, cy, w, h)


def line(ax, p1, p2, ls="-", color="#444", lw=0.9):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, ls=ls, zorder=1)


# ---------------------------------------------------------------- Fig 4.5 use case
def use_case():
    fig, ax = canvas(6.4, 6.8, 100, 100)
    ax.add_patch(Rectangle((24, 3), 52, 94, fc="#FAFAFA", ec="#333", lw=1.2, zorder=0))
    ax.text(50, 94.2, "ATERNET Forecasting System", ha="center", fontsize=8.5, fontweight="bold")

    ucs = [
        "UC1  Download and prepare data",
        "UC2  Train the forecasting model",
        "UC3  Evaluate against baselines",
        "UC4  Calibrate forecast ranges",
        "UC5  Run multi-seed / ablation study",
        "UC6  Generate live forecast",
        "UC7  Replay past forecasts (backtest)",
        "UC8  Explore results in dashboard",
    ]
    ys = np.linspace(85, 12, len(ucs))
    pos = {}
    for k, (t, y) in enumerate(zip(ucs, ys)):
        pos[k + 1] = oval(ax, 50, y, 44, 7.6, t)

    res = actor(ax, 9, 66, "Researcher /\nStudent")
    rev = actor(ax, 9, 30, "Reviewer /\nExaminer")
    ana = actor(ax, 9, 10, "Grid / energy\nanalyst")
    box(ax, 90.5, 82, 15, 9, "«system»\nPV_Live", "amber", 7, rounded=False)
    box(ax, 90.5, 60, 15, 9, "«system»\nOpen-Meteo\nERA5", "amber", 7, rounded=False)

    for u in (1, 2, 3, 4, 5, 8):
        line(ax, (res[0] + 2.6, res[1] + 1), (pos[u][0] - 22, pos[u][1]))
    for u in (3, 8):
        line(ax, (rev[0] + 2.6, rev[1] + 1), (pos[u][0] - 22, pos[u][1]))
    for u in (6, 7, 8):
        line(ax, (ana[0] + 2.6, ana[1] + 1), (pos[u][0] - 22, pos[u][1]))
    for u in (1, 6, 7):
        line(ax, (83, 82 if True else 0), (pos[u][0] + 22, pos[u][1]))
    for u in (1, 6, 7):
        line(ax, (83, 60), (pos[u][0] + 22, pos[u][1]), lw=0.7)
    # include relationship
    arrow(ax, (58, pos[7][1] + 3.8), (58, pos[6][1] - 3.8), ls="--", lw=0.9, text="«include»", fs=6.2, text_off=(9, 0))
    return save(fig, "fig_4_5_use_case.png")


# ---------------------------------------------------------------- Fig 4.6 class diagram
def class_box(ax, cx, cy, w, name, attrs, methods, fc="blue", fs=5.8, abstract=False):
    lh = 2.15
    h_head = 4.2
    h_attr = max(lh * len(attrs) + 1.2, 3)
    h_meth = max(lh * len(methods) + 1.2, 3)
    h = h_head + h_attr + h_meth
    top = cy + h / 2
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="square,pad=0", fc="white",
                                ec=C[fc + "_d"], lw=1.0, zorder=2))
    ax.add_patch(Rectangle((cx - w / 2, top - h_head), w, h_head, fc=C[fc], ec=C[fc + "_d"], lw=1.0, zorder=3))
    ax.text(cx, top - h_head / 2, ("«abstract» " if abstract else "") + name, ha="center", va="center",
            fontsize=6.4, fontweight="bold", zorder=4, style="italic" if abstract else "normal")
    y = top - h_head - 0.9
    for a in attrs:
        ax.text(cx - w / 2 + 0.8, y - lh / 2, a, ha="left", va="center", fontsize=fs, zorder=4)
        y -= lh
    y = top - h_head - h_attr
    ax.plot([cx - w / 2, cx + w / 2], [y, y], color=C[fc + "_d"], lw=0.8, zorder=4)
    y -= 0.9
    for m in methods:
        ax.text(cx - w / 2 + 0.8, y - lh / 2, m, ha="left", va="center", fontsize=fs, zorder=4)
        y -= lh
    return {"cx": cx, "cy": cy, "w": w, "h": h, "top": top, "bot": cy - h / 2}


def rel(ax, p_from, p_to, kind, label=None, fs=5.8, label_off=(0, 1.0)):
    """kind: comp, agg, inherit, use"""
    x0, y0 = p_from
    x1, y1 = p_to
    if kind == "use":
        arrow(ax, p_from, p_to, ls="--", lw=0.8, ms=7)
    else:
        ax.plot([x0, x1], [y0, y1], color="#333", lw=0.9, zorder=1)
    d = np.array([x1 - x0, y1 - y0], float)
    n = np.linalg.norm(d) or 1.0
    d /= n
    p = np.array([-d[1], d[0]])
    if kind in ("comp", "agg"):
        s = 1.35
        c = np.array([x0, y0])
        pts = [c, c + d * s + p * s * 0.55, c + d * 2 * s, c + d * s - p * s * 0.55]
        ax.add_patch(Polygon(pts, closed=True, fc="#222" if kind == "comp" else "white", ec="#222", lw=0.8, zorder=5))
    elif kind == "inherit":
        s = 1.5
        c = np.array([x1, y1])
        pts = [c, c - d * s * 1.4 + p * s * 0.7, c - d * s * 1.4 - p * s * 0.7]
        ax.add_patch(Polygon(pts, closed=True, fc="white", ec="#222", lw=0.8, zorder=5))
    if label:
        ax.text((x0 + x1) / 2 + label_off[0], (y0 + y1) / 2 + label_off[1], label, fontsize=fs, ha="center",
                va="center", zorder=6, bbox=dict(fc="white", ec="none", pad=0.4))


def class_diagram():
    fig, ax = canvas(6.6, 8.9, 100, 134)
    tr = class_box(ax, 17, 117, 31, "Trainer", ["model: ATERNET", "optimizer: AdamW", "ema_state: dict",
                                                "cfg: TrainingConfig"],
                   ["fit(epochs)", "train_epoch(epoch)", "validate(epoch)"], "green")
    ds = class_box(ax, 50, 118.5, 28, "ATERNETDataset", ["x_seq, x_nwp, x_sigma", "y, meta", "regime_id"],
                   ["__getitem__(i)", "__len__()"], "amber")
    fc = class_box(ax, 84, 117, 29, "Forecaster", ["model: ATERNET", "scalers: dict", "site: dict"],
                   ["forecast_live()", "backtest(start, end)", "predict_windows(w)"], "violet")

    at = class_box(ax, 50, 90, 50, "ATERNET", ["backbone: TemporalBackbone", "expert_pool: ExpertPool",
                                             "router: AdaptiveRouter", "decoder: ProbabilisticDecoder",
                                             "last_step_skip: Linear (optional)"],
                   ["forward(x_seq, x_nwp, x_sigma) : dict"], "blue")

    bb = class_box(ax, 14, 58, 25, "TemporalBackbone", ["seq_patch_embed", "nwp_cross_attn", "transformer",
                                                       "sigma_encoder"], ["forward(...)"], "blue")
    ep = class_box(ax, 40, 58, 24, "ExpertPool", ["experts: ModuleDict (4)"],
                   ["forward(h_expert)", "compute_loss(...)"], "green")
    rt = class_box(ax, 66, 58, 25, "AdaptiveRouter", ["sigma_nwp_extractor", "sigma_expert_calibrator",
                                                    "fusion: UncertaintyFusion", "gate: GatingNetwork"],
                   ["forward(...)"], "amber")
    de = class_box(ax, 88.5, 58, 21, "Probabilistic\nDecoder", ["prediction_head", "interval_width_head",
                                                            "quantile_decoder"], ["forward(...)"], "red")

    eb = class_box(ax, 26, 29, 27, "ExpertBase", ["name, in_dim, horizon"],
                   ["forward(h_expert)", "primary_loss(...)", "auxiliary_loss(y_hat)", "total_loss(...)"],
                   "green", abstract=True)
    ex = class_box(ax, 26, 9.5, 40, "Concrete experts", ["DawnRampExpert, ClearPeakExpert,",
                                                         "OvercastExpert, VolatileExpert"], [], "green")
    gn = class_box(ax, 62, 29, 22, "GatingNetwork", ["logit_net", "raw_beta", "delta (residual gate)"],
                   ["forward(h_expert, h_sigma, fusion)"], "amber")
    uf = class_box(ax, 87.5, 29, 20, "UncertaintyFusion", ["bn_nwp, bn_expert", "alpha_head"],
                   ["forward(sigma_nwp,", " sigma_expert, h_expert)"], "amber")

    # ATERNET composition
    for tgt, x in ((bb, 14), (ep, 40), (rt, 66), (de, 88.5)):
        rel(ax, (min(max(x, 30), 70), at["bot"]), (tgt["cx"], tgt["top"]), "comp")
    # uses
    rel(ax, (tr["cx"] + 6, tr["bot"]), (35, at["top"]), "use")
    rel(ax, (ds["cx"], ds["bot"]), (ds["cx"], at["top"]), "use")
    rel(ax, (fc["cx"] - 6, fc["bot"]), (65, at["top"]), "use")
    ax.text(24, 103.2, "trains", fontsize=5.8, ha="center", bbox=dict(fc="white", ec="none", pad=0.3))
    ax.text(52, 103.2, "feeds", fontsize=5.8, ha="left", bbox=dict(fc="white", ec="none", pad=0.3))
    ax.text(76, 103.2, "serves", fontsize=5.8, ha="center", bbox=dict(fc="white", ec="none", pad=0.3))
    # pool / router internals
    rel(ax, (ep["cx"] - 6, ep["bot"]), (eb["cx"] + 6, eb["top"]), "agg", "4", label_off=(-2.2, 1.5))
    rel(ax, (rt["cx"] - 4, rt["bot"]), (gn["cx"] - 2, gn["top"]), "comp")
    rel(ax, (rt["cx"] + 6, rt["bot"]), (uf["cx"] - 3, uf["top"]), "comp")
    rel(ax, (ex["cx"], ex["top"]), (eb["cx"], eb["bot"]), "inherit")
    return save(fig, "fig_4_6_class_diagram.png")


# ---------------------------------------------------------------- sequence diagrams
def sequence(name, title, parts, msgs, frame=None, h_in=5.6):
    n = len(parts)
    top, bottom = 94, 4
    xs = np.linspace(8, 92, n)
    fig, ax = canvas(6.5, h_in, 100, 100)
    ax.text(50, 99, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
    for x, (label, fc) in zip(xs, parts):
        box(ax, x, top - 1.5, min(84 / n * 1.05, 20), 7.5, label, fc, 6.6)
        ax.plot([x, x], [top - 5.5, bottom], color="#888", lw=0.8, ls=(0, (4, 3)), zorder=1)
    y = top - 10
    step = (y - bottom - 4) / max(len(msgs), 1)
    ypos = {}
    for k, m in enumerate(msgs):
        a, b, text = m[0], m[1], m[2]
        kind = m[3] if len(m) > 3 else "call"
        yy = y - k * step
        ypos[k] = yy
        if a == b:
            x = xs[a]
            ax.plot([x, x + 6, x + 6, x + 0.3], [yy + 1.1, yy + 1.1, yy - 1.1, yy - 1.1], color="#222", lw=0.9, zorder=3)
            ax.annotate("", xy=(x + 0.1, yy - 1.1), xytext=(x + 1.5, yy - 1.1),
                        arrowprops=dict(arrowstyle="-|>", color="#222", lw=0.9))
            ax.text(x + 7, yy, text, fontsize=6.0, ha="left", va="center")
        else:
            x0, x1 = xs[a], xs[b]
            ls = "--" if kind == "return" else "-"
            ax.annotate("", xy=(x1, yy), xytext=(x0, yy),
                        arrowprops=dict(arrowstyle="-|>", color="#222", lw=0.9, ls=ls, shrinkA=0, shrinkB=0))
            ax.text((x0 + x1) / 2, yy + 1.15, f"{k + 1}. {text}", fontsize=6.0, ha="center", va="bottom",
                    bbox=dict(fc="white", ec="none", pad=0.2))
    if frame:
        s, e, label = frame
        ax.add_patch(Rectangle((3, ypos[e] - step * 0.45), 94, ypos[s] - ypos[e] + step * 0.9, fc="none",
                               ec="#2E5A88", lw=1.0, ls="-", zorder=1))
        ax.add_patch(Rectangle((3, ypos[s] + step * 0.45 - 3.2), 17, 3.2, fc="#DCE9F7", ec="#2E5A88", lw=1.0, zorder=1))
        ax.text(4, ypos[s] + step * 0.45 - 1.6, label, fontsize=6.2, va="center")
    return save(fig, name)


def sequence_live():
    parts = [("User", "green"), ("predict.py /\ndashboard", "violet"), ("Forecaster", "blue"),
             ("PV_Live\nAPI", "amber"), ("Open-Meteo\nAPI", "amber"), ("ATERNET\nmodel", "red")]
    msgs = [
        (0, 1, "request live forecast"),
        (1, 2, "forecast_live()"),
        (2, 3, "get latest measured PV"),
        (3, 2, "PV series, capacity", "return"),
        (2, 4, "get 5-model weather forecast"),
        (4, 2, "hourly weather", "return"),
        (2, 2, "align to 30-min grid, build window, scale"),
        (2, 5, "forward(x_seq, x_nwp, x_sigma)"),
        (5, 2, "P10, P50, P90 (16 steps)", "return"),
        (2, 1, "forecast table in MW", "return"),
        (1, 0, "chart, CSV and dashboard view", "return"),
    ]
    return sequence("fig_4_7_sequence_live.png", "Sequence diagram: live forecast", parts, msgs, h_in=5.8)


def sequence_training():
    parts = [("Researcher", "green"), ("train.py", "violet"), ("Dataset /\nDataLoader", "amber"),
             ("Trainer", "blue"), ("ATERNET\nmodel", "red"), ("evaluate.py /\nconformal", "violet")]
    msgs = [
        (0, 1, "run training (flags: residual gate, shortcut, EMA)"),
        (1, 2, "load train / val splits"),
        (2, 1, "batches", "return"),
        (1, 3, "fit(epochs)"),
        (3, 4, "forward(batch)"),
        (4, 3, "quantiles, gate weights, σ", "return"),
        (3, 3, "composite loss, backward, clip, AdamW step, EMA update"),
        (3, 3, "validate with EMA weights; keep best checkpoint"),
        (3, 1, "best checkpoint and history", "return"),
        (0, 5, "evaluate test split; calibrate"),
        (5, 4, "predict(test), predict(val)"),
        (5, 0, "metrics, calibrated ranges (JSON)", "return"),
    ]
    return sequence("fig_4_8_sequence_training.png", "Sequence diagram: training, evaluation and calibration",
                    parts, msgs, frame=(4, 7, "loop [each epoch]"), h_in=6.0)


# ---------------------------------------------------------------- Fig 2.1 Gantt chart
def gantt():
    tasks = [
        ("Problem study and literature review", date(2026, 7, 6), date(2026, 8, 9), "blue"),
        ("Data pipeline and model design (Phases 1–5)", date(2026, 8, 1), date(2026, 8, 31), "green"),
        ("Real-data integration (PV_Live, Open-Meteo)", date(2026, 9, 1), date(2026, 10, 10), "amber"),
        ("Baselines, ablations and accuracy improvements", date(2026, 10, 10), date(2026, 10, 24), "green"),
        ("Second region and simulated India demo", date(2026, 10, 19), date(2026, 10, 27), "amber"),
        ("Dashboard, tests and documentation", date(2026, 10, 10), date(2026, 10, 28), "violet"),
        ("Project report and demonstration", date(2026, 10, 20), date(2026, 11, 6), "red"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for i, (name, s, e, col) in enumerate(tasks[::-1]):
        ax.barh(i, (e - s).days, left=mdates.date2num(s), height=0.55, color=C[col], edgecolor=C[col + "_d"], lw=1.0)
        ax.text(mdates.date2num(e) + 0.8, i, f"{s:%d %b} – {e:%d %b}", va="center", ha="left", fontsize=6.5, color="#222")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks[::-1]], fontsize=7.2)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(mdates.date2num(date(2026, 7, 1)), mdates.date2num(date(2026, 11, 28)))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="x", color="#DDDDDD", lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig_2_1_project_plan.png")


# ---------------------------------------------------------------- Fig 5.x flowcharts
def diamond(ax, cx, cy, w, h, text, fs=6.8):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], closed=True,
                         fc=C["amber"], ec=C["amber_d"], lw=1.0, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.1)
    return (cx, cy, w, h)


def terminal(ax, cx, cy, w, h, text):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=3.5",
                                fc=C["grey"], ec=C["grey_d"], lw=1.0, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=7.2, fontweight="bold", zorder=3)
    return (cx, cy, w, h)


def flow_data():
    fig, ax = canvas(5.4, 7.9, 100, 148)
    x = 50
    steps = [
        ("t", "Start: choose region and date range"),
        ("b", "Download measured PV (PV_Live)\nhalf-hourly generation + installed capacity"),
        ("b", "Download forecast weather (5 models, 3 points)\nretry on failure; split large requests"),
        ("b", "Align to the 30-minute grid; interpolate hourly weather;\naverage over the region's points"),
        ("b", "Clean: normalise by capacity, clip to [0, 1],\nflag gaps, solar elevation"),
        ("b", "Feature engineering: clear-sky index, time encodings,\nσ_NWP spread features, weather regime labels"),
        ("b", "Windowing: 48 steps history + 16 steps ahead (stride 4)\nforecast weather over history and horizon"),
        ("d", "Any gap or\nleakage in window?"),
        ("b", "Chronological split 70 / 15 / 15\nscalers fitted on the training part only"),
        ("t", "Save datasets (.npz) and scalers"),
    ]
    ys = np.linspace(141, 7, len(steps))
    prev = None
    for (kind, text), y in zip(steps, ys):
        if kind == "t":
            cur = terminal(ax, x, y, 46, 8, text)
        elif kind == "d":
            cur = diamond(ax, x, y, 40, 11, text)
        else:
            cur = box(ax, x, y, 72, 9.5, text, "blue", 6.7)
        if prev is not None:
            arrow(ax, edge(prev, "b"), edge(cur, "t"), lw=1.0)
        prev = cur
    # decision loop: gap -> drop window
    yd = ys[7]
    box(ax, 88, yd, 20, 9, "drop the\nwindow", "red", 6.5)
    arrow(ax, (70, yd), (78, yd), text="yes", fs=6.2, text_off=(0, 1.8))
    ax.text(52.5, yd - 8.2, "no", fontsize=6.2)
    return save(fig, "fig_5_1_data_pipeline_flow.png")


def flow_train():
    fig, ax = canvas(5.8, 8.2, 100, 150)
    x = 40
    a = terminal(ax, x, 143, 50, 7, "Start: load datasets and configuration")
    b = box(ax, x, 130, 70, 9, "Build ATERNET (residual gate δ = 0, zero-initialised shortcut)\nAdamW + cosine learning-rate schedule", "green", 6.5)
    c = box(ax, x, 116, 70, 9, "For each batch: forward pass → composite loss\n(pinball + sharpness + expert + load-balance)", "blue", 6.6)
    d = box(ax, x, 103, 70, 8, "Backward pass, gradient clipping (1.0),\noptimiser step, EMA weight update", "blue", 6.6)
    e = box(ax, x, 90, 70, 8, "End of epoch: validate with the EMA weights\n(validation pinball loss)", "blue", 6.6)
    f = diamond(ax, x, 76, 44, 12, "New best\nvalidation loss?")
    g = box(ax, x, 62, 56, 7, "Save checkpoint (EMA weights) and history", "green", 6.5)
    h = diamond(ax, x, 49, 44, 12, "Patience (8) used up\nor 50 epochs done?")
    i = box(ax, x, 35, 70, 8, "Load best checkpoint; evaluate on the test split\n(nRMSE, nMAE, R², PICP, PINAW, Winkler)", "amber", 6.5)
    j = box(ax, x, 22, 70, 8, "Conformal calibration on daytime validation points\n(per weather regime), re-evaluate", "amber", 6.5)
    k = terminal(ax, x, 10, 46, 7, "Write metrics, plots and reports")
    for p, q in ((a, b), (b, c), (c, d), (d, e), (e, f), (f, g), (g, h), (h, i), (i, j), (j, k)):
        arrow(ax, edge(p, "b"), edge(q, "t"), lw=1.0)
    ax.text(x + 2, 70.0, "yes", fontsize=6.2)
    ax.text(x + 2, 43.3, "yes", fontsize=6.2)
    # no-branches loop back to next epoch
    ax.plot([x + 22, 88, 88], [76, 76, 116], color="#333", lw=1.0, zorder=1)
    arrow(ax, (88, 116), (x + 35, 116), lw=1.0)
    ax.text(x + 24, 77.4, "no", fontsize=6.2)
    ax.plot([x + 22, 94, 94], [49, 49, 116], color="#333", lw=1.0, zorder=1)
    ax.plot([94, 88], [116, 116], color="#333", lw=1.0, zorder=1)
    ax.text(x + 24, 50.4, "no", fontsize=6.2)
    ax.plot([x + 22, x + 22], [62, 62], color="#333", lw=0.1)
    ax.plot([x + 28, 88], [62, 62], color="#333", lw=1.0, zorder=1)
    ax.text(91, 90, "next epoch", rotation=90, fontsize=6.2, va="center")
    return save(fig, "fig_5_2_training_flow.png")


if __name__ == "__main__":
    use_case()
    class_diagram()
    sequence_live()
    sequence_training()
    gantt()
    flow_data()
    flow_train()
