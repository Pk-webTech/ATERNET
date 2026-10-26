"""
ATERNET -- Interactive Dashboard

    streamlit run app.py

Requires: streamlit, plotly
    pip install streamlit plotly --break-system-packages
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.paths import (
    BEST_CHECKPOINT_PATH, TEST_METRICS_PATH, TEST_PREDICTIONS_PATH,
    TRAIN_HISTORY_PATH, METRICS_DIR, REPORTS_DIR, ensure_dirs,
)
from src.models.aternet import ATERNET
from src.phase3.expert_pool import EXPERT_ORDER
from src.training.checkpoint import load_checkpoint
from src.training.dataset import ATERNETDataset
from src.utils.constants import ID_TO_REGIME
from src.utils.io import load_json, load_npz

# ------------------------------------------------------------------ #
# Config & Theme
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="ATERNET",
    page_icon="\u2600",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "amber":    "#E8A838",
    "sky":      "#4A90D9",
    "slate":    "#3B4252",
    "surface":  "#F7F5F2",
    "muted":    "#8B8B8B",
    "p50":      "#2E5A88",
    "band":     "rgba(46,90,136,0.15)",
    "observed": "#D4572A",
}
REGIME_COLORS = {
    "dawn_ramp":  "#E8853A",
    "clear_peak": "#D4A72C",
    "overcast":   "#7B8FA1",
    "volatile":   "#C23B3B",
}
PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#3B4252"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=48, r=24, t=36, b=40),
    hoverlabel=dict(font_size=12),
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', system-ui, sans-serif; }
    section[data-testid="stSidebar"] { background: #2E3440; color: #ECEFF4; }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #ECEFF4 !important; }
    section[data-testid="stSidebar"] hr { border-color: #434C5E; }
    section[data-testid="stSidebar"] .stRadio label p,
    section[data-testid="stSidebar"] .stRadio label div,
    section[data-testid="stSidebar"] .stRadio label span,
    section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] .stCaption p
    { color: #ECEFF4 !important; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #E5E5E5;
        border-radius: 8px; padding: 14px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #6B7280 !important; font-size: 0.78rem !important;
        font-weight: 500 !important; letter-spacing: 0.02em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-weight: 600 !important; color: #2E3440 !important;
    }
    .section-header {
        font-size: 1.35rem; font-weight: 600; color: #2E3440;
        margin: 0.6rem 0 0.3rem 0; padding-bottom: 6px;
        border-bottom: 2px solid #E8A838; display: inline-block;
    }
    .section-caption { font-size: 0.85rem; color: #6B7280; margin-bottom: 1rem; }
    .info-card {
        background: white; border: 1px solid #E5E5E5;
        border-radius: 8px; padding: 16px; margin-bottom: 12px;
    }
    .info-card .label {
        font-size: 0.72rem; color: #9CA3AF;
        font-weight: 500; letter-spacing: 0.03em; margin-bottom: 2px;
    }
    .info-card .value { font-size: 1.1rem; font-weight: 600; color: #2E3440; }
    .regime-tag {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.78rem; font-weight: 600; color: white;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    table.score { width: 100%; border-collapse: collapse; margin: 6px 0 14px 0; font-size: 0.92rem; }
    table.score th { text-align: left; background: #F1EFEA; color: #3B4252; padding: 8px 10px;
                     border-bottom: 2px solid #E8A838; font-weight: 600; }
    table.score td { padding: 9px 10px; border-bottom: 1px solid #ECECEC; vertical-align: top; }
    table.score small { font-weight: 400; color: #6B7280; }
    .bottom-line { background: #F3F8F4; border-left: 5px solid #1B7F4B; padding: 12px 16px;
                   border-radius: 6px; margin: 8px 0 14px 0; font-size: 1.0rem; }
    .bottom-line.neutral { background: #F6F6F2; border-left-color: #9A6B00; }
    .intro-box { background: #F4F7FB; border: 1px solid #DCE6F2; border-radius: 8px;
                 padding: 10px 14px; margin-bottom: 14px; font-size: 0.93rem; color: #2E3440; }
</style>
""", unsafe_allow_html=True)


def section_header(title, caption=None):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)

def info_card(label, value):
    st.markdown(f'<div class="info-card"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>', unsafe_allow_html=True)

def regime_tag(name):
    color = REGIME_COLORS.get(name, "#6B7280")
    return f'<span class="regime-tag" style="background:{color}">{name}</span>'

def styled_axes(fig):
    fig.update_xaxes(gridcolor="#EBEBEB", zeroline=False)
    fig.update_yaxes(gridcolor="#EBEBEB", zeroline=False)
    return fig


# ------------------------------------------------------------------ #
# Cached loaders
# ------------------------------------------------------------------ #
@st.cache_resource
def load_model():
    model, payload = load_checkpoint(BEST_CHECKPOINT_PATH, model_class=ATERNET, map_location="cpu")
    model.eval()
    return model, payload

@st.cache_data
def load_test_data():
    return ATERNETDataset("test")

@st.cache_data
def load_predictions():
    return load_npz(TEST_PREDICTIONS_PATH)

@st.cache_data
def load_metrics():
    return load_json(TEST_METRICS_PATH)

@st.cache_data
def load_history():
    return load_json(TRAIN_HISTORY_PATH)

@st.cache_data
def load_comparison():
    path = REPORTS_DIR / "comparison_table.json"
    return load_json(path) if path.exists() else None

SUMMARY_PATH = ROOT / "docs" / "results_summary.json"
VIT_DIR = ROOT / "workspaces" / "vit"


@st.cache_data
def load_summary():
    """Aggregated multi-seed results (rebuilt by build_results_summary.py)."""
    return load_json(SUMMARY_PATH) if SUMMARY_PATH.exists() else None


@st.cache_data
def load_calibration():
    p = METRICS_DIR / "conformal_calibration.json"
    return load_json(p) if p.exists() else None


def fmt_ms(d, nd=3):
    """{'mean': m, 'std': s} -> 'm +/- s'."""
    return f"{d['mean']:.{nd}f} \u00b1 {d['std']:.{nd}f}"


REGIME_PLAIN = {
    "dawn_ramp": "Morning ramp-up",
    "clear_peak": "Clear sunny midday",
    "overcast": "Cloudy / overcast",
    "volatile": "Patchy, changing clouds",
}

PAGE_LABELS = {
    "Dashboard": "Overview",
    "Results & Evidence": "Is ATERNET better?",
    "India Demo (Simulated)": "India demo (simulated)",
    "Live Forecast": "Live forecast",
    "Forecast Explorer": "Explore forecasts",
    "Routing & Experts": "How the experts work",
    "Uncertainty": "How sure is the model?",
    "Guarantee Demo": "Built-in safety checks",
    "Training": "Training history",
}

PAGE_INTRO = {
    "Dashboard": "ATERNET predicts how much electricity a solar plant will produce over the next 8 hours, "
                 "and also gives a range of likely values, so you can see how sure it is. "
                 "Start here for the headline numbers and a quick comparison with other models.",
    "Results & Evidence": "This page answers one question: <b>is ATERNET better than other forecasting models?</b> "
                          "Every model was trained several times (different random starts) so we can tell a real "
                          "difference from luck. Green = ATERNET is better, yellow = about the same, red = worse.",
    "India Demo (Simulated)": "A demonstration of ATERNET forecasting for the VIT Vellore area. "
                              "<b>The plant's output here is simulated, not measured</b>, because no real solar "
                              "data for this area is publicly available.",
    "Live Forecast": "Forecasts for the next hours using the latest weather forecasts, and a replay of past "
                     "forecasts scored against what really happened.",
    "Forecast Explorer": "Look at individual forecasts one by one and compare them with what actually happened.",
    "Routing & Experts": "ATERNET contains four small 'expert' models, each meant for a type of weather. "
                         "This page shows how much the model trusts each one. In our tests the trust does not follow "
                         "the weather type cleanly, and switching the gate off did not change accuracy.",
    "Uncertainty": "A good forecast also says how sure it is. Here we check that the model's 'unsureness' really "
                   "goes up when its errors are bigger, and that its forecast ranges contain the real value "
                   "about as often as promised.",
    "Guarantee Demo": "Three properties are built into the model's design and checked by automated tests. "
                      "Use the slider to see each one hold on a real forecast.",
    "Training": "How the model learned: the error went down over repeated rounds of training until it stopped "
                "improving.",
}

GLOSSARY_MD = """
- **Typical error (nRMSE)** – how far off the forecast usually is, as a % of the plant's full capacity.
  Lower is better. A big miss counts more than a small one.
- **Average error (nMAE)** – the average size of the miss, as a % of capacity. Lower is better.
- **Forecast range (P10–P90)** – besides one best guess, the model gives a low and a high value. The real output
  should land between them about 80% of the time.
- **Range hit-rate (coverage, PICP)** – how often the real output really landed inside that range.
  The target is 80%. We measure it in <i>daytime</i> only, because at night the output is 0 and always "inside".
- **Range quality score (Winkler)** – one number that rewards a narrow range and punishes missing the real value.
  Lower is better. It stops a model from cheating by drawing very wide ranges.
- **Runs / seeds** – each model is trained several times from different random starts. If results barely change
  between runs, the model is reliable.
- **Better / about the same / worse** – on the real regions we only say "better" or "worse" when the gap is
  bigger than the normal run-to-run variation (about 2 standard deviations); otherwise it is "about the same".
  On the simulated India demo each model was trained once, so "scored better" there is simply how that run came out.
- **Persistence** – the naive forecast "the next hours will look like right now". Beating it is the minimum bar.
"""

PILLS = {
    "better":   ("ATERNET is better", "#1B7F4B"),
    "same":     ("About the same", "#9A6B00"),
    "worse":    ("ATERNET is worse", "#B83232"),
    "lead1":    ("ATERNET scored better", "#1B7F4B"),
    "trail1":   ("ATERNET scored worse", "#B83232"),
    "unstable": ("Other model unreliable", "#6B7280"),
}


def pill(status):
    text, color = PILLS[status]
    return (f'<span style="background:{color};color:white;padding:3px 10px;border-radius:12px;'
            f'font-size:0.8rem;font-weight:600;white-space:nowrap">{text}</span>')


def compare(atern, base, key):
    """
    (status, pct): pct > 0 means ATERNET's value is lower (better) by that % of the other model's.
    A win/loss is only declared when the gap exceeds ~2 seed standard deviations, and never from one run.
    """
    spread = max(base["per_seed_nrmse"]) - min(base["per_seed_nrmse"])
    if base["n_seeds"] > 1 and spread > 1.0:
        return "unstable", 0.0
    a, b = atern[key], base[key]
    gap = b["mean"] - a["mean"]
    pct = gap / b["mean"] * 100 if b["mean"] else 0.0
    if atern["n_seeds"] < 2 or base["n_seeds"] < 2:
        if abs(pct) < 1.0:
            return "same", pct
        return ("lead1" if gap > 0 else "trail1"), pct
    tol = 2 * max(a["std"], b["std"], 1e-9)
    if gap > tol:
        return "better", pct
    if gap < -tol:
        return "worse", pct
    return "same", pct


def _words(status, pct, what):
    if status == "better":
        return f"{what} is {abs(pct):.0f}% lower (better)"
    if status == "worse":
        return f"{what} is {abs(pct):.0f}% higher (worse)"
    if status == "lead1":
        return f"{what} is {abs(pct):.0f}% lower (better)"
    if status == "trail1":
        return f"{what} is {abs(pct):.0f}% higher (worse)"
    return f"{what} is about the same"


def _site_parts(site):
    main = [r for r in site["rows"] if r["kind"] != "ablation"]
    atern = next(r for r in main if r["kind"] == "aternet")
    return main, atern, [r for r in main if r["kind"] == "baseline"]


def scoreboard_html(site):
    """Colour-coded table: ATERNET against each competitor, in plain words."""
    _, atern, bases = _site_parts(site)
    out = ['<table class="score"><tr><th>ATERNET compared with</th>'
           '<th>Forecast accuracy<br><small>typical error, lower is better</small></th>'
           '<th>Quality of the forecast range<br><small>range quality score, lower is better</small></th>'
           '<th>In plain words</th></tr>']
    for b in bases:
        sa, pa = compare(atern, b, "nrmse_pct")
        sr, pr = compare(atern, b, "winkler")
        if sa == "unstable":
            seeds = ", ".join(f"{v:.1f}" for v in b["per_seed_nrmse"])
            words = (f"{b['model']} sometimes fails badly (typical error per run: {seeds}%), so a fair "
                     "comparison is not possible. ATERNET's result barely changes between runs.")
        else:
            words = _words(sa, pa, "Typical error") + "; " + _words(sr, pr, "range quality score")
        words = words.rstrip(".")
        out.append(f"<tr><td><b>{b['model']}</b></td><td>{pill(sa)}</td><td>{pill(sr)}</td><td>{words}.</td></tr>")
    out.append("</table>")
    return "".join(out)


def bottom_line(site):
    """(is_positive, html) one-paragraph summary of where ATERNET stands on this site."""
    _, atern, bases = _site_parts(site)
    stable = [b for b in bases if compare(atern, b, "nrmse_pct")[0] != "unstable"]
    unstable = [b["model"] for b in bases if b not in stable]
    groups = {"better": [], "same": [], "worse": []}
    for b in stable:
        st_acc = compare(atern, b, "nrmse_pct")[0]
        groups["better" if st_acc in ("better", "lead1") else "worse" if st_acc in ("worse", "trail1") else "same"].append(b["model"])
    parts = []
    if groups["better"]:
        parts.append("<b>more accurate than</b> " + ", ".join(groups["better"]))
    if groups["same"]:
        parts.append("<b>about as accurate as</b> " + ", ".join(groups["same"]))
    if groups["worse"]:
        parts.append("<b>less accurate than</b> " + ", ".join(groups["worse"]))
    text = "On accuracy, ATERNET is " + "; ".join(parts) + "." if parts else "No stable competitor to compare with."
    rng_better = [b["model"] for b in stable if compare(atern, b, "winkler")[0] in ("better", "lead1")]
    if rng_better:
        text += " Its forecast ranges are scored better than " + ", ".join(rng_better) + "."
    if unstable:
        text += " " + ", ".join(unstable) + " is too unreliable to compare fairly."
    return (not groups["worse"] and bool(groups["better"] or rng_better)), text


def bar_chart(rows, key, title, nd):
    """Bars with run-to-run variation as error bars; ATERNET highlighted."""
    names = [r["model"].strip() for r in rows]
    fig = go.Figure(data=go.Bar(
        x=names, y=[r[key]["mean"] for r in rows],
        error_y=dict(type="data", array=[r[key]["std"] for r in rows]),
        marker_color=[COLORS["p50"] if r["kind"] == "aternet" else COLORS["muted"] for r in rows],
        text=[f"{r[key]['mean']:.{nd}f}" for r in rows], textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, height=330, title=dict(text=title, font=dict(size=14)),
                      yaxis_title="shorter bar = better", xaxis_tickangle=-15)
    styled_axes(fig)
    return fig


def _with_skip(model, y_routed, x_seq, x_nwp):
    """Add the optional last-step shortcut exactly as ATERNET.forward does."""
    if getattr(model, "last_step_skip", None) is None:
        return y_routed
    window = x_seq.shape[1]
    parts = [x_seq[:, -1, :], x_nwp[:, window:, :].flatten(1)]
    return y_routed + model.last_step_skip(torch.cat(parts, dim=1))


def _gate_at_sigma(model, emb, router_out, sigma):
    """Recompute the gate for an overridden sigma_total (same maths as GatingNetwork)."""
    from torch.nn.functional import softplus
    gate = model.router.gate
    gate_in = torch.cat([emb["h_expert"], emb["h_sigma"],
                         router_out["sigma_nwp_norm"].unsqueeze(1),
                         router_out["sigma_expert_norm"].unsqueeze(1),
                         router_out["alpha"].unsqueeze(1)], dim=1)
    logits = gate.logit_net(gate_in)
    temp = torch.exp(torch.clamp(softplus(gate.raw_beta) * sigma, -8, 8))
    scaled = logits / temp.unsqueeze(1)
    if getattr(gate, "residual", False):
        scaled = gate.delta * scaled
    return temp, torch.softmax(scaled, dim=-1).numpy().ravel()


# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("## ATERNET")
    st.caption("Adaptive Temporal Expert Routing Network\nfor Probabilistic Solar Power Forecasting")
    st.markdown("---")
    page = st.radio("Page", list(PAGE_LABELS), format_func=lambda k: PAGE_LABELS[k],
                    label_visibility="collapsed")
    st.markdown("---")
    st.caption("BCSE497J Capstone Project\nVIT Vellore")


st.markdown(f'<div class="intro-box">{PAGE_INTRO[page]}</div>', unsafe_allow_html=True)
with st.expander("What do these words mean? (plain-language glossary)"):
    st.markdown(GLOSSARY_MD, unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# DASHBOARD
# ------------------------------------------------------------------ #
if page == "Dashboard":
    st.markdown("# ATERNET")
    st.markdown("**Solar power forecasting that also tells you how sure it is.**")

    metrics = load_metrics()
    h = metrics["headline"]
    cal = load_calibration() or {}
    n_params = sum(p.numel() for p in load_model()[0].parameters())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Typical error", f"{h['nrmse_pct']:.2f}%",
              help="How far off the forecast usually is, as a % of the plant's full capacity. Lower is better. "
                   "(Technical name: nRMSE.)")
    c2.metric("Average error", f"{h['nmae_pct']:.2f}%",
              help="The average size of the miss, as a % of capacity. Lower is better. (Technical name: nMAE.)")
    c3.metric("Follows reality", f"{h['r2']:.3f}",
              help="1.000 would be a perfect match with what really happened. (Technical name: R².)")
    day_after = cal.get("picp_daytime_after")
    c4.metric("Range hit-rate (day)", f"{day_after * 100:.0f}%" if day_after is not None else "n/a",
              help="How often the real output landed inside the forecast range during daytime. "
                   "The target is 80%. (Technical name: PICP, measured on daytime points after calibration.)")
    skill = h.get("rmse_skill_vs_persistence", 0)
    c5.metric("Error cut vs. naive guess", f"{skill * 100:.0f}%",
              help="Compared with the naive forecast 'the next hours will look like right now', ATERNET's "
                   "error is this much lower.")
    st.caption(f"One trained model on the Yorkshire test period ({metrics['n_samples']:,} forecasts, "
               f"{n_params:,} adjustable numbers). The forecast range never crosses itself (0 violations).")

    summary = load_summary()
    st.markdown("---")
    section_header("How does ATERNET compare with other models?",
                   "Each model was trained several times; we only call it better or worse when the gap is larger "
                   "than normal run-to-run variation.")
    if summary:
        for key in ("south_west", "yorkshire"):
            site = summary["sites"].get(key)
            if not site:
                continue
            good, text = bottom_line(site)
            st.markdown(f"**{site['title']}**")
            st.markdown(f'<div class="bottom-line{"" if good else " neutral"}">{text}</div>',
                        unsafe_allow_html=True)
            st.markdown(scoreboard_html(site), unsafe_allow_html=True)
        st.caption("Open **Is ATERNET better?** in the left menu for charts, details and what each part of the "
                   "model contributes.")
    else:
        st.info("Run `python build_results_summary.py` to generate the comparison.")

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        section_header("An example forecast", "The shaded band is where the model expects the real output to land "
                                              "about 80% of the time.")
        preds = load_predictions()
        y = preds["y_true"]
        daytime = np.where(y.max(axis=1) > 0.1)[0]
        idx = daytime[len(daytime) // 3] if len(daytime) > 10 else 0
        hours = np.arange(1, y.shape[1] + 1) * 0.5
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([hours, hours[::-1]]),
            y=np.concatenate([preds["p90"][idx], preds["p10"][idx][::-1]]),
            fill="toself", fillcolor=COLORS["band"],
            line=dict(width=0), name="Likely range (80%)", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(x=hours, y=preds["p50"][idx], mode="lines",
                                  name="Best guess", line=dict(color=COLORS["p50"], width=2.5)))
        fig.add_trace(go.Scatter(x=hours, y=y[idx], mode="lines+markers",
                                  name="What actually happened", line=dict(color=COLORS["observed"], width=2, dash="dot"),
                                  marker=dict(size=4)))
        fig.update_layout(**PLOTLY_LAYOUT, height=340,
                          xaxis_title="Hours ahead", yaxis_title="Plant output (fraction of capacity)",
                          yaxis_range=[-0.02, 1.02],
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        styled_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        section_header("By weather situation")
        pr = metrics.get("per_regime", {})
        cal_pr = cal.get("per_regime_comparison", {})
        regime_data = [{"Weather": REGIME_PLAIN.get(rn, rn), "Cases": m["n"],
                        "Typical error %": round(m["nrmse_pct"], 2),
                        "Range hit-rate (day)": (f"{cal_pr[rn]['picp_daytime_after'] * 100:.0f}%"
                                                 if rn in cal_pr and cal_pr[rn].get("picp_daytime_after") is not None else "n/a")}
                       for rn, m in pr.items() if m.get("n", 0) > 0]
        if regime_data:
            st.dataframe(regime_data, width="stretch", hide_index=True)
        section_header("Built-in safety checks")
        st.markdown("Three properties are built into the design and tested automatically:\n\n"
                    "1. The low value of the range is never above the best guess, and the high value never below it\n"
                    "2. The range gets wider when the model is less sure\n"
                    "3. The experts' votes spread out when the model is less sure\n\n"
                    "See them work in **Built-in safety checks**.")


# ------------------------------------------------------------------ #
# LIVE FORECAST
# ------------------------------------------------------------------ #
elif page == "Live Forecast":
    import pandas as pd
    from config.paths import PREDICTIONS_DIR

    section_header("Live forecast",
                   "Measured generation from Sheffield Solar PV_Live + Open-Meteo multi-model "
                   "weather forecasts, fetched now. No panel or API key required.")

    @st.cache_resource
    def load_forecaster():
        from src.inference.forecaster import Forecaster
        return Forecaster()

    try:
        forecaster = load_forecaster()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    st.markdown(f"**Region:** {forecaster.region.name} (PV_Live region {forecaster.region.region_id})")
    if st.button("Fetch latest data and forecast", type="primary"):
        with st.spinner("Downloading PV_Live readings and Open-Meteo forecasts..."):
            try:
                st.session_state["live"] = forecaster.forecast_live(return_history=True)
            except Exception as e:  # network/API failures are expected in the wild
                st.error(f"Live forecast failed: {e}")

    if "live" in st.session_state:
        fc, pv_hist = st.session_state["live"]
        issue = fc["issue_time"].iloc[0]
        recent = pv_hist[pv_hist["timestamp"] > issue - pd.Timedelta(hours=24)]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Issued (UTC)", f"{issue:%d %b %H:%M}")
        c2.metric("Capacity", f"{fc['capacity_mw'].iloc[0]:.0f} MWp")
        c3.metric("Peak P50 next 8h", f"{fc['p50_mw'].max():.0f} MW")
        c4.metric("Regime at issue", fc["regime_at_issue"].iloc[0])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["target_time"], fc["target_time"][::-1]]),
            y=pd.concat([fc["p90_mw"], fc["p10_mw"][::-1]]),
            fill="toself", fillcolor=COLORS["band"], line=dict(width=0),
            name="P10–P90", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc["target_time"], y=fc["p50_mw"], mode="lines",
                                 name="P50 forecast", line=dict(color=COLORS["p50"], width=2.5)))
        fig.add_trace(go.Scatter(x=recent["timestamp"], y=recent["pv_power_kw"] / 1000.0,
                                 mode="lines+markers", name="Measured (PV_Live)",
                                 line=dict(color=COLORS["observed"], width=2), marker=dict(size=4)))
        fig.update_layout(**PLOTLY_LAYOUT, height=440, xaxis_title="Time (UTC)",
                          yaxis_title="PV generation (MW)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        styled_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fc[["target_time", "p10_mw", "p50_mw", "p90_mw"]].round(1),
                     width="stretch", hide_index=True)

    backtests = sorted(PREDICTIONS_DIR.glob("backtest_*.csv"))
    if backtests:
        st.markdown("---")
        section_header("Out-of-sample backtest",
                       "Past forecasts replayed through the live pipeline and scored against "
                       "measured PV_Live generation (python predict.py backtest)")
        choice = st.selectbox("Backtest run", [p.stem for p in backtests], index=len(backtests) - 1)
        bt = pd.read_csv(PREDICTIONS_DIR / f"{choice}.csv", parse_dates=["issue_time", "target_time"])
        metrics_path = METRICS_DIR / f"{choice}.json"
        if metrics_path.exists():
            m = load_json(metrics_path)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("nMAE", f"{m['nmae_pct']:.2f}%")
            c2.metric("Daylight MAE", f"{m['daylight_mae_mw']:.0f} MW")
            c3.metric("Skill vs persistence", f"{m['rmse_skill_vs_persistence']:.3f}")
            c4.metric("PICP (nominal 0.80)", f"{m['picp']:.3f}")
        lead = st.select_slider("Lead time (hours)", sorted(bt["lead_hours"].unique()), value=2.0)
        d = bt[bt["lead_hours"] == lead].sort_values("target_time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([d["target_time"], d["target_time"][::-1]]),
            y=pd.concat([d["p90_mw"], d["p10_mw"][::-1]]),
            fill="toself", fillcolor=COLORS["band"], line=dict(width=0),
            name="P10–P90", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d["target_time"], y=d["p50_mw"], mode="lines",
                                 name=f"P50 ({lead:g} h ahead)", line=dict(color=COLORS["p50"], width=2)))
        fig.add_trace(go.Scatter(x=d["target_time"], y=d["actual_mw"], mode="markers",
                                 name="Measured", marker=dict(color=COLORS["observed"], size=4)))
        fig.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_title="Time (UTC)",
                          yaxis_title="PV generation (MW)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        styled_axes(fig)
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ #
# FORECAST EXPLORER
# ------------------------------------------------------------------ #
elif page == "Forecast Explorer":
    section_header("Explore forecasts", "Pick any test case. The shaded band is where the model expects the real output to land about 80% of the time; the line is its best guess.")
    preds = load_predictions()
    y = preds["y_true"]
    regime_ids = preds["regime_id"]

    col_ctrl, col_chart = st.columns([1, 3])
    with col_ctrl:
        regime_filter = st.selectbox("Weather situation (regime)", ["All"] + list(ID_TO_REGIME.values()))
        valid_idx = np.arange(len(y)) if regime_filter == "All" else \
            np.where(regime_ids == [k for k, v in ID_TO_REGIME.items() if v == regime_filter][0])[0]
        daytime_idx = valid_idx[y[valid_idx].max(axis=1) > 0.05]
        if len(daytime_idx) == 0:
            st.warning("No daytime windows for this regime.")
            st.stop()
        window_pos = st.slider("Test case", 0, len(daytime_idx) - 1, 0)
        wi = daytime_idx[window_pos]
        rname = ID_TO_REGIME.get(int(regime_ids[wi]), "?")
        st.markdown(regime_tag(rname), unsafe_allow_html=True)
        st.markdown("")
        info_card("Window ID", int(wi))
        info_card("\u03c3_total", f"{float(preds['sigma_total'][wi]):.4f}")
        info_card("Temperature", f"{float(preds['temperature'][wi]):.3f}")
        info_card("\u03b1 (NWP weight)", f"{float(preds['alpha'][wi]):.3f}")

    with col_chart:
        steps = np.arange(1, y.shape[1] + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([preds["p90"][wi], preds["p10"][wi][::-1]]),
            fill="toself", fillcolor=COLORS["band"],
            line=dict(width=0), name="P10\u2013P90", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=steps, y=preds["p50"][wi], mode="lines",
                                  name="P50", line=dict(color=COLORS["p50"], width=2.5)))
        fig.add_trace(go.Scatter(x=steps, y=y[wi], mode="lines+markers",
                                  name="Observed", line=dict(color=COLORS["observed"], width=2, dash="dot"),
                                  marker=dict(size=5)))
        fig.update_layout(**PLOTLY_LAYOUT, height=460,
                          xaxis_title="Horizon step", yaxis_title="pv_norm",
                          yaxis_range=[-0.02, 1.02],
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        styled_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
        mae_val = float(np.mean(np.abs(y[wi] - preds["p50"][wi])))
        covered = float(np.mean((y[wi] >= preds["p10"][wi]) & (y[wi] <= preds["p90"][wi])))
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("MAE", f"{mae_val:.4f}")
        mc2.metric("Coverage", f"{covered:.0%}")
        mc3.metric("Width", f"{float(np.mean(preds['p90'][wi] - preds['p10'][wi])):.4f}")
        mc4.metric("Entropy", f"{float(preds['entropy'][wi]):.3f}")


# ------------------------------------------------------------------ #
# ROUTING & EXPERTS
# ------------------------------------------------------------------ #
elif page == "Routing & Experts":
    section_header("How the experts work", "ATERNET has four small expert models, one for each type of weather. A gate decides how much to trust each one. In our tests the trust does not match the weather type cleanly, "
                   "and removing the gate did not change accuracy.")
    preds = load_predictions()
    gate = preds["gate_weights"]
    regime_ids = preds["regime_id"]

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**How much each expert is trusted, by weather situation**")
        n_regimes = len(ID_TO_REGIME)
        mat = np.zeros((n_regimes, gate.shape[1]))
        regime_names = [ID_TO_REGIME[i] for i in range(n_regimes)]
        for rid in range(n_regimes):
            mask = regime_ids == rid
            if mask.sum(): mat[rid] = gate[mask].mean(axis=0)
        fig = go.Figure(data=go.Heatmap(
            z=mat, x=list(EXPERT_ORDER), y=regime_names,
            text=np.round(mat, 3).astype(str), texttemplate="%{text}",
            colorscale=[[0, "#F7F5F2"], [1, "#2E5A88"]], zmin=0, zmax=0.5, showscale=False))
        fig.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Expert", yaxis_title="Regime")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Which expert is trusted most, and how often**")
        argmax = gate.argmax(axis=1)
        shares = {EXPERT_ORDER[k]: float(np.mean(argmax == k)) for k in range(len(EXPERT_ORDER))}
        fig2 = go.Figure(data=go.Bar(
            x=list(shares.keys()), y=list(shares.values()),
            marker_color=[REGIME_COLORS.get(n, "#636EFA") for n in shares.keys()],
            text=[f"{v:.1%}" for v in shares.values()], textposition="outside"))
        fig2.update_layout(**PLOTLY_LAYOUT, height=300, yaxis_title="Fraction",
                           yaxis_range=[0, max(shares.values()) * 1.3])
        styled_axes(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    section_header("Trust in each expert for one case")
    wi2 = st.slider("Test case", 0, len(gate) - 1, 0, key="gate_slider")
    gc1, gc2 = st.columns([1, 3])
    with gc1:
        st.markdown(regime_tag(ID_TO_REGIME.get(int(regime_ids[wi2]), "?")), unsafe_allow_html=True)
        st.markdown("")
        info_card("\u03c3_total", f"{preds['sigma_total'][wi2]:.4f}")
        info_card("T", f"{preds['temperature'][wi2]:.3f}")
    with gc2:
        fig3 = go.Figure(data=go.Bar(
            x=list(EXPERT_ORDER), y=gate[wi2],
            marker_color=[REGIME_COLORS.get(n, "#636EFA") for n in EXPERT_ORDER],
            text=[f"{v:.3f}" for v in gate[wi2]], textposition="outside"))
        fig3.update_layout(**PLOTLY_LAYOUT, height=260, yaxis_range=[0, 1], yaxis_title="Gate weight")
        styled_axes(fig3)
        st.plotly_chart(fig3, use_container_width=True)


# ------------------------------------------------------------------ #
# UNCERTAINTY
# ------------------------------------------------------------------ #
elif page == "Uncertainty":
    section_header("How sure is the model?",
                   "The model gives every forecast an 'unsureness' score and draws a wider range when it is "
                   "less sure. Below we check that this score really goes up when the errors are bigger.")
    preds = load_predictions()
    metrics = load_metrics()
    y = preds["y_true"]
    abs_err = np.abs(y - preds["p50"]).mean(axis=1)
    sigma = preds["sigma_total"]
    width = (preds["p90"] - preds["p10"]).mean(axis=1)
    regime_ids = preds["regime_id"]
    rc = [REGIME_COLORS.get(ID_TO_REGIME.get(int(r), ""), "#8B8B8B") for r in regime_ids]
    r = metrics["routing"]

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Does 'unsure' predict bigger errors?", f"{r.get('corr_sigma_total_vs_abs_error', 0):.3f}")
    mc2.metric("Do wider ranges go with bigger errors?", f"{r.get('corr_interval_width_vs_abs_error', 0):.3f}")
    mc3.metric("Weather-vs-model mix (\u03b1)", f"{r.get('mean_alpha', 0):.3f}")

    col1, col2 = st.columns(2)
    for col, xdata, xlabel in [(col1, sigma, "Model's unsureness score"), (col2, width, "Width of the forecast range")]:
        with col:
            fig = go.Figure(data=go.Scatter(
                x=xdata, y=abs_err, mode="markers",
                marker=dict(size=4, opacity=0.45, color=rc),
                hovertemplate=f"{xlabel}=%{{x:.3f}}<br>|err|=%{{y:.4f}}<extra></extra>"))
            fig.update_layout(**PLOTLY_LAYOUT, height=380,
                              xaxis_title=xlabel, yaxis_title="Average error of that forecast")
            styled_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

    legend_html = " ".join(
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{REGIME_COLORS[n]};margin-right:4px"></span>{n}&nbsp;&nbsp;&nbsp;'
        for n in REGIME_COLORS)
    st.markdown(f'<div style="text-align:center;margin-top:-8px">{legend_html}</div>', unsafe_allow_html=True)

    cqr_path = METRICS_DIR / "conformal_calibration.json"
    if cqr_path.exists():
        st.markdown("---")
        section_header("Conformal calibration", "Per-regime conformal calibration fitted on daytime validation points")
        cqr = load_json(cqr_path)
        comp = cqr.get("per_regime_comparison", {})
        _f = lambda v: round(v, 3) if isinstance(v, (int, float)) else "n/a"
        st.dataframe([{"Regime": rn, "n": d["n"],
                       "Daytime PICP before": _f(d.get("picp_daytime_before")),
                       "Daytime PICP after": _f(d.get("picp_daytime_after")),
                       "PINAW before": round(d["pinaw_before"], 4), "PINAW after": round(d["pinaw_after"], 4)}
                      for rn, d in comp.items()], width="stretch", hide_index=True)
        st.caption("Coverage is measured on daytime points only (nominal 0.80). Night points have a true value "
                   "and interval of exactly 0 and are trivially covered, which would inflate the number.")


# ------------------------------------------------------------------ #
# GUARANTEE DEMO
# ------------------------------------------------------------------ #
elif page == "Guarantee Demo":
    section_header("Built-in safety checks",
                   "Move the slider to change how unsure the model is, and watch all three checks hold.")
    model, _ = load_model()
    ds = load_test_data()
    daytime = np.where(ds.y.numpy().max(axis=1) > 0.1)[0]

    wi3 = st.selectbox("Test window", daytime[:50].tolist(),
                       format_func=lambda x: f"Window {x} \u2014 {ID_TO_REGIME.get(int(ds.regime_id[x].item()), '?')}")
    sigma_override = st.slider("How unsure the model is (override)", 0.0, 2.0, 0.5, 0.05)

    x_seq, x_nwp, x_sigma = ds.x_seq[wi3:wi3+1], ds.x_nwp[wi3:wi3+1], ds.x_sigma[wi3:wi3+1]
    with torch.no_grad():
        emb = model.backbone(x_seq, x_nwp, x_sigma)
        pool_out = model.expert_pool(emb["h_expert"])
        router_real = model.router(emb["h_expert"], emb["h_sigma"], x_sigma,
                                    pool_out["sigma_expert"], pool_out["stacked"])
        y_routed = _with_skip(model, router_real["y_hat_routed"], x_seq, x_nwp)
        q_real = model.decoder(y_routed, emb["h_expert"],
                                emb["h_nwp"], router_real["sigma_total"])
        sigma_manual = torch.tensor([sigma_override], dtype=torch.float32)
        q_manual = model.decoder(y_routed, emb["h_expert"],
                                  emb["h_nwp"], sigma_manual)
        T_manual, gate_manual = _gate_at_sigma(model, emb, router_real, sigma_manual)

    y_obs = ds.y[wi3].numpy()
    steps = np.arange(1, len(y_obs) + 1)
    real_T = float(router_real["temperature"].item())
    colors_bar = [REGIME_COLORS.get(n, "#636EFA") for n in EXPERT_ORDER]

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**Check 1: the range never flips upside down**")
        p10_m, p50_m, p90_m = (q_manual[k].numpy().ravel() for k in ("p10", "p50", "p90"))
        st.metric("Times it broke", int(np.sum(p10_m > p50_m + 1e-6)) + int(np.sum(p50_m > p90_m + 1e-6)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([p90_m, p10_m[::-1]]),
            fill="toself", fillcolor=COLORS["band"],
            line=dict(width=0), name="P10\u2013P90", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=steps, y=p50_m, name="P50", line=dict(color=COLORS["p50"], width=2)))
        fig.add_trace(go.Scatter(x=steps, y=y_obs, name="Observed",
                                  line=dict(color=COLORS["observed"], width=2, dash="dot")))
        fig.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False,
                          yaxis_range=[-0.02, 1.02], xaxis_title="Step")
        styled_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        st.markdown("**Check 2: the range widens when the model is less sure**")
        sigmas_sweep = np.linspace(0, 2, 25)
        widths_sweep = []
        for s in sigmas_sweep:
            with torch.no_grad():
                q_s = model.decoder(y_routed, emb["h_expert"],
                     emb["h_nwp"], torch.tensor([s], dtype=torch.float32))
            widths_sweep.append(float((q_s["p90"] - q_s["p10"]).mean()))
        monotone = all(widths_sweep[i] <= widths_sweep[i+1] + 1e-6 for i in range(len(widths_sweep)-1))
        st.metric("Always widens?", "Yes" if monotone else "No")
        fig2 = go.Figure(data=go.Scatter(x=sigmas_sweep, y=widths_sweep, mode="lines",
                                          line=dict(color=COLORS["p50"], width=2.5)))
        fig2.add_vline(x=sigma_override, line_dash="dash", line_color=COLORS["observed"],
                       annotation_text=f"\u03c3={sigma_override:.2f}")
        fig2.update_layout(**PLOTLY_LAYOUT, height=280, xaxis_title="\u03c3_total", yaxis_title="Mean width")
        styled_axes(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with g3:
        st.markdown("**Check 3: the experts' votes even out when unsure**")
        st.caption("The final model uses a residual gate that stays close to uniform routing, so the shift is "
                   "small. The guarantee is structural (temperature rises with \u03c3), not a claim that "
                   "routing drives accuracy.")
        T_val = float(T_manual.item() if hasattr(T_manual, 'item') else T_manual)
        st.metric("Vote-spreading level", f"{T_val:.3f}")
        fig3 = make_subplots(rows=2, cols=1,
                              subplot_titles=[f"Real \u03c3 (T={real_T:.2f})", f"\u03c3={sigma_override:.2f} (T={T_val:.2f})"],
                              vertical_spacing=0.3)
        fig3.add_trace(go.Bar(x=list(EXPERT_ORDER), y=router_real["gate_weights"].numpy().ravel(),
                              marker_color=colors_bar, showlegend=False), row=1, col=1)
        fig3.add_trace(go.Bar(x=list(EXPERT_ORDER), y=gate_manual,
                              marker_color=colors_bar, showlegend=False), row=2, col=1)
        fig3.update_layout(**PLOTLY_LAYOUT, height=280)
        fig3.update_yaxes(range=[0, 1])
        st.plotly_chart(fig3, use_container_width=True)


# ------------------------------------------------------------------ #
# BASELINES & ABLATIONS
# ------------------------------------------------------------------ #
# RESULTS & EVIDENCE
# ------------------------------------------------------------------ #
elif page == "Results & Evidence":
    summary = load_summary()
    if summary is None:
        st.info("Run `python build_results_summary.py` to generate docs/results_summary.json.")
        st.stop()

    measured = {k: v for k, v in summary["sites"].items() if not v["simulated"]}
    site_key = st.radio("Which solar region?", list(measured),
                        format_func=lambda k: measured[k]["title"], horizontal=True)
    site = measured[site_key]
    main_rows, atern, baselines = _site_parts(site)
    st.caption(site["note"])

    good, text = bottom_line(site)
    section_header("The short answer")
    st.markdown(f'<div class="bottom-line{"" if good else " neutral"}">{text}</div>', unsafe_allow_html=True)
    st.markdown(scoreboard_html(site), unsafe_allow_html=True)

    section_header("The same comparison as charts", "Whiskers show how much each model's result changes between "
                                                   "training runs. Shorter bar = better.")
    c_left, c_right = st.columns(2)
    c_left.plotly_chart(bar_chart(main_rows, "nrmse_pct", "Typical error (%)", 2), use_container_width=True)
    c_right.plotly_chart(bar_chart(main_rows, "winkler", "Range quality score", 4), use_container_width=True)

    with st.expander("Show the detailed numbers"):
        st.dataframe([{
            "Model": r["model"].strip(), "Training runs": r["n_seeds"],
            "Typical error % (nRMSE)": fmt_ms(r["nrmse_pct"], 3),
            "Range quality score (Winkler)": fmt_ms(r["winkler"], 4),
            "Range width (PINAW)": fmt_ms(r["pinaw"], 4),
            "Follows reality (R²)": fmt_ms(r["r2"], 4),
            "Range hit-rate, all hours (PICP)": fmt_ms(r["picp"], 3),
            "Typical error in each run": ", ".join(f"{v:.2f}" for v in r["per_seed_nrmse"]),
        } for r in main_rows], width="stretch", hide_index=True)
        st.caption("The 'all hours' hit-rate includes night, when the output is 0 and always counts as a hit, so "
                   "it looks better than it is. Daytime hit-rate is shown below.")

    cal_s = site.get("calibration")
    if cal_s and cal_s.get("picp_daytime_before") is not None:
        st.markdown("---")
        section_header("Does the forecast range hold what it promises?",
                       "The range is meant to contain the real output 80% of the time. Checked in daytime only.")
        q1, q2, q3 = st.columns(3)
        q1.metric("Before calibration", f"{cal_s['picp_daytime_before'] * 100:.0f}%",
                  help="The model's raw ranges were a bit too narrow.")
        q2.metric("After calibration", f"{cal_s['picp_daytime_after'] * 100:.0f}%",
                  delta=f"{(cal_s['picp_daytime_after'] - cal_s['picp_daytime_before']) * 100:+.0f} points",
                  help="Calibration widens the ranges slightly, using data the model did not train on.")
        q3.metric("Target", "80%")
        st.caption("Calibration was applied to ATERNET only. The comparison tables above use everyone's raw ranges, "
                   "so they are fair.")

    abl = [r for r in site["rows"] if r["kind"] == "ablation"]
    ref = next((r for r in site["rows"] if r["model"] == "ATERNET without EMA"), None)
    if abl and ref:
        st.markdown("---")
        section_header("Which parts of ATERNET actually help?",
                       "We switched off one part at a time and checked whether results got worse.")
        ABL_PLAIN = {
            "no dense expert loss": "Without the extra training signal given to every expert",
            "no gate (uniform average)": "Without the gate (all experts always trusted equally)",
            "fixed temperature": "Without the 'spread votes when unsure' control",
            "fixed interval width": "Without letting the range width follow the model's unsureness",
            "sigma_NWP only": "Using only the weather-forecast disagreement as the unsureness signal",
            "sigma_expert only": "Using only the experts' disagreement as the unsureness signal",
            "single expert": "Using one expert instead of four",
            "no regime weights": "Without extra training weight on each expert's own weather type",
        }
        rows, mattered = [], []
        for r in abl:
            gap = r["nrmse_pct"]["mean"] - ref["nrmse_pct"]["mean"]
            tol = 2 * max(r["nrmse_pct"]["std"], ref["nrmse_pct"]["std"])
            wgap = r["winkler"]["mean"] - ref["winkler"]["mean"]
            wtol = 2 * max(r["winkler"]["std"], ref["winkler"]["std"])
            hurts = gap > tol or wgap > wtol
            helps = gap < -tol or wgap < -wtol
            name = ABL_PLAIN.get(r["model"].replace("ablation:", "").strip(), r["model"].strip())
            if hurts:
                mattered.append(name)
            rows.append({"Part removed": name,
                         "Typical error change": f"{gap:+.3f} points",
                         "Range quality change": f"{wgap:+.4f}",
                         "Verdict": ("Removing it makes things worse" if hurts else
                                     "Removing it makes things better" if helps else "No clear difference")})
        st.dataframe(rows, width="stretch", hide_index=True)
        if mattered:
            st.info("Only one change clearly mattered: **" + "; ".join(mattered) + "**. "
                    "The other parts made no measurable difference on this data, so we do not claim they improve "
                    "accuracy.")
        else:
            st.info("No single part showed a clear effect beyond run-to-run variation.")

    with st.expander("What these results do and do not show"):
        st.markdown(
            "- **They show:** ATERNET is as accurate as the best competitor tested and clearly more accurate than "
            "PatchTST; on the second region it is ahead of LightGBM by more than the run-to-run variation.\n"
            "- **They show:** ATERNET gives almost the same result every time it is trained; the LSTM does not.\n"
            "- **They do not show:** that the 'expert' structure or the gate improves accuracy; the switch-off tests "
            "above say it does not.\n"
            "- **Limits:** two regions in England; competitors were not specially tuned; LightGBM improves when "
            "given twice as many training examples, while ATERNET did not in our test.")


# ------------------------------------------------------------------ #
# INDIA DEMO (SIMULATED)
# ------------------------------------------------------------------ #
elif page == "India Demo (Simulated)":
    st.error("SIMULATED DATA, NOT MEASURED. No open measured solar data exists for the VIT Vellore area, so the "
             "'what actually happened' line on this page comes from a physical solar-panel model driven by real "
             "past weather, for an imaginary 1 MW plant. The forecasts themselves use real weather forecasts. "
             "This shows the system working for this location; it is not a real-world accuracy result.")
    summary = load_summary()
    vit = (summary or {}).get("sites", {}).get("vellore")
    section_header("VIT Vellore, India", "12.97°N, 79.16°E, simulated 1 MW plant")

    if vit:
        _, a_row, _ = _site_parts(vit)
        base = next((r for r in vit["rows"] if r["kind"] == "baseline"), None)
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("ATERNET typical error", f"{a_row['nrmse_pct']['mean']:.2f}%")
        v2.metric("LightGBM typical error", f"{base['nrmse_pct']['mean']:.2f}%" if base else "n/a")
        v3.metric("ATERNET range quality", f"{a_row['winkler']['mean']:.4f}", help="Lower is better.")
        v4.metric("LightGBM range quality", f"{base['winkler']['mean']:.4f}" if base else "n/a", help="Lower is better.")
        if base:
            st.markdown(scoreboard_html(vit), unsafe_allow_html=True)
        cal_v = vit.get("calibration")
        if cal_v and cal_v.get("picp_daytime_before") is not None:
            st.caption(f"In daytime the forecast range contained the simulated output "
                       f"{cal_v['picp_daytime_before'] * 100:.0f}% of the time "
                       f"({cal_v['picp_daytime_after'] * 100:.0f}% after calibration; target 80%): the ranges are "
                       "still somewhat too narrow here.")
    else:
        st.info("No Vellore results found. Follow the India demo steps in the README, then run "
                "`python build_results_summary.py`.")

    pred_path = VIT_DIR / "outputs" / "predictions" / "test_predictions.npz"
    if pred_path.exists():
        @st.cache_data
        def load_vit_predictions():
            return load_npz(pred_path)
        vp = load_vit_predictions()
        yv = vp["y_true"]
        day = np.where(yv.max(axis=1) > 0.1)[0]
        section_header("Pick a forecast", "Slide to choose a daytime case")
        options = day[::max(1, len(day) // 200)].tolist()
        pick = st.select_slider("Test case", options=options, value=options[len(options) // 2])
        hours = np.arange(1, yv.shape[1] + 1) * 0.5
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.concatenate([hours, hours[::-1]]),
                                 y=np.concatenate([vp["p90"][pick], vp["p10"][pick][::-1]]),
                                 fill="toself", fillcolor=COLORS["band"], line=dict(width=0),
                                 name="Likely range (80%)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hours, y=vp["p50"][pick], name="ATERNET best guess",
                                 line=dict(color=COLORS["p50"], width=2.5)))
        fig.add_trace(go.Scatter(x=hours, y=yv[pick], name="Simulated plant output",
                                 line=dict(color=COLORS["observed"], width=2, dash="dot")))
        if "baseline" in vp:
            fig.add_trace(go.Scatter(x=hours, y=vp["baseline"][pick], name="Naive guess (same as now)",
                                     line=dict(color=COLORS["muted"], width=1.5, dash="dash")))
        fig.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_title="Hours ahead",
                          yaxis_title="Output (fraction of 1 MW)", yaxis_range=[-0.02, 1.02],
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        styled_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

    plots = [("forecast_examples.png", "Example forecasts"),
             ("reliability_diagram.png", "Does the range hold what it promises?"),
             ("per_horizon_metrics.png", "Error by hours ahead")]
    shown = [(VIT_DIR / "outputs" / "plots" / f, cap) for f, cap in plots
             if (VIT_DIR / "outputs" / "plots" / f).exists()]
    if shown:
        cols = st.columns(len(shown))
        for col, (path, cap) in zip(cols, shown):
            col.image(str(path), caption=cap + " (simulated site)", width="stretch")

    with st.expander("How is the plant output simulated?"):
        st.markdown(
            "- Output = plant size × sunshine strength, reduced when panels are hot, dusty, or the inverter is "
            "at its limit, plus about 1% random noise.\n"
            "- **Why this is not circular:** the simulated output uses what the weather *actually did* (a "
            "reanalysis dataset), while the model only sees *weather forecasts* from four forecasting systems. "
            "The difference between forecast and reality is real.\n"
            "- **Limits:** no real measurements; one training run; the numbers tell you how well the model learns "
            "this simulator from forecasts.")


# ------------------------------------------------------------------ #
# TRAINING
# ------------------------------------------------------------------ #
elif page == "Training":
    history = load_history()
    section_header("Training history", f"{len(history)} rounds of learning; training stopped automatically when it "
                   "stopped improving on data it had not trained on.")
    epochs = [h["epoch"] for h in history]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Loss curves**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=[h["train_loss_total"] for h in history],
                                  name="Train", line=dict(color=COLORS["p50"])))
        fig.add_trace(go.Scatter(x=epochs, y=[h["val_loss_total"] for h in history],
                                  name="Val", line=dict(color=COLORS["observed"])))
        fig.add_trace(go.Scatter(x=epochs, y=[h["val_pinball_total"] for h in history],
                                  name="Val pinball", line=dict(color=COLORS["amber"], dash="dot")))
        fig.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="Epoch", yaxis_title="Loss",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        styled_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Expert gate share**")
        fig2 = go.Figure()
        for name in EXPERT_ORDER:
            key = f"val_gate_share_{name}"
            if key in history[0]:
                fig2.add_trace(go.Scatter(x=epochs, y=[h[key] for h in history], name=name,
                                          line=dict(color=REGIME_COLORS.get(name))))
        fig2.add_hline(y=0.25, line_dash="dash", line_color=COLORS["muted"], annotation_text="uniform")
        fig2.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="Epoch", yaxis_title="Gate weight",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        styled_axes(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Regularizers**")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=epochs, y=[h["val_sharpness"] for h in history],
                                   name="Sharpness", line=dict(color=COLORS["amber"])))
        fig3.add_trace(go.Scatter(x=epochs, y=[h["val_load_balance"] for h in history],
                                   name="Load balance", line=dict(color=COLORS["sky"])))
        fig3.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Epoch",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        styled_axes(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**Learning rate**")
        fig4 = go.Figure(data=go.Scatter(x=epochs, y=[h["lr"] for h in history],
                                          line=dict(color=COLORS["slate"])))
        fig4.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Epoch", yaxis_title="LR")
        styled_axes(fig4)
        st.plotly_chart(fig4, use_container_width=True)