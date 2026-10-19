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


# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("## ATERNET")
    st.caption("Adaptive Temporal Expert Routing Network\nfor Probabilistic Solar Power Forecasting")
    st.markdown("---")
    page = st.radio("", [
        "Dashboard", "Live Forecast", "Forecast Explorer", "Routing & Experts",
        "Uncertainty", "Guarantee Demo", "Baselines & Ablations", "Training",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("BCSE497J Capstone Project\nVIT Vellore")


# ------------------------------------------------------------------ #
# DASHBOARD
# ------------------------------------------------------------------ #
if page == "Dashboard":
    st.markdown("# ATERNET")
    st.markdown("Probabilistic solar power forecasting with uncertainty-aware expert routing. "
                "967,132 parameters, 50 passing tests, zero quantile-crossing violations.")

    metrics = load_metrics()
    h = metrics["headline"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("nRMSE", f"{h['nrmse_pct']:.2f}%")
    c2.metric("nMAE", f"{h['nmae_pct']:.2f}%")
    c3.metric("R\u00b2", f"{h['r2']:.4f}")
    c4.metric("PICP", f"{h['picp']:.3f}")
    c5.metric("Winkler", f"{h['winkler']:.4f}")
    c6.metric("Skill vs Persist.", f"{h.get('rmse_skill_vs_persistence', 0):.3f}")

    st.markdown("")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        section_header("Forecast sample")
        preds = load_predictions()
        y = preds["y_true"]
        daytime = np.where(y.max(axis=1) > 0.1)[0]
        idx = daytime[len(daytime) // 3] if len(daytime) > 10 else 0
        steps = np.arange(1, y.shape[1] + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([preds["p90"][idx], preds["p10"][idx][::-1]]),
            fill="toself", fillcolor=COLORS["band"],
            line=dict(width=0), name="P10\u2013P90", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(x=steps, y=preds["p50"][idx], mode="lines",
                                  name="P50", line=dict(color=COLORS["p50"], width=2.5)))
        fig.add_trace(go.Scatter(x=steps, y=y[idx], mode="lines+markers",
                                  name="Observed", line=dict(color=COLORS["observed"], width=2, dash="dot"),
                                  marker=dict(size=4)))
        fig.update_layout(**PLOTLY_LAYOUT, height=340,
                          xaxis_title="Horizon step", yaxis_title="pv_norm",
                          yaxis_range=[-0.02, 1.02],
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        styled_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        section_header("Per-regime coverage")
        pr = metrics.get("per_regime", {})
        regime_data = [{"Regime": rn, "n": m["n"], "nRMSE %": round(m["nrmse_pct"], 2),
                        "PICP": round(m["picp"], 3), "PINAW": round(m["pinaw"], 4)}
                       for rn, m in pr.items() if m.get("n", 0) > 0]
        if regime_data:
            st.dataframe(regime_data, use_container_width=True, hide_index=True)
        section_header("Guarantees")
        st.markdown("Three structural claims, each enforced by construction and verified by named tests:\n\n"
                    "1. P10 \u2264 P50 \u2264 P90 always (softplus width)\n"
                    "2. Interval width non-decreasing in \u03c3_total\n"
                    "3. Routing softens under higher uncertainty")


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
                     use_container_width=True, hide_index=True)

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
    section_header("Forecast explorer", "Browse test windows with P10/P50/P90 intervals")
    preds = load_predictions()
    y = preds["y_true"]
    regime_ids = preds["regime_id"]

    col_ctrl, col_chart = st.columns([1, 3])
    with col_ctrl:
        regime_filter = st.selectbox("Regime", ["All"] + list(ID_TO_REGIME.values()))
        valid_idx = np.arange(len(y)) if regime_filter == "All" else \
            np.where(regime_ids == [k for k, v in ID_TO_REGIME.items() if v == regime_filter][0])[0]
        daytime_idx = valid_idx[y[valid_idx].max(axis=1) > 0.05]
        if len(daytime_idx) == 0:
            st.warning("No daytime windows for this regime.")
            st.stop()
        window_pos = st.slider("Window", 0, len(daytime_idx) - 1, 0)
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
    section_header("Expert routing", "Gate weight distribution across regime-specialised experts")
    preds = load_predictions()
    gate = preds["gate_weights"]
    regime_ids = preds["regime_id"]

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**Gate weight matrix**")
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
        st.markdown("**Argmax share**")
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
    section_header("Per-window routing")
    wi2 = st.slider("Window", 0, len(gate) - 1, 0, key="gate_slider")
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
    section_header("Dual-uncertainty diagnostics",
                   "Core claim: the fused uncertainty signal tracks realised forecast error")
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
    mc1.metric("corr(\u03c3, |error|)", f"{r.get('corr_sigma_total_vs_abs_error', 0):.3f}")
    mc2.metric("corr(width, |error|)", f"{r.get('corr_interval_width_vs_abs_error', 0):.3f}")
    mc3.metric("Mean \u03b1", f"{r.get('mean_alpha', 0):.3f}")

    col1, col2 = st.columns(2)
    for col, xdata, xlabel in [(col1, sigma, "\u03c3_total"), (col2, width, "Interval width")]:
        with col:
            fig = go.Figure(data=go.Scatter(
                x=xdata, y=abs_err, mode="markers",
                marker=dict(size=4, opacity=0.45, color=rc),
                hovertemplate=f"{xlabel}=%{{x:.3f}}<br>|err|=%{{y:.4f}}<extra></extra>"))
            fig.update_layout(**PLOTLY_LAYOUT, height=380,
                              xaxis_title=xlabel, yaxis_title="Mean |error|")
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
        section_header("Conformal calibration", "Per-regime CQR on the validation set")
        cqr = load_json(cqr_path)
        comp = cqr.get("per_regime_comparison", {})
        st.dataframe([{"Regime": rn, "n": d["n"],
                       "PICP before": round(d["picp_before"], 3), "PICP after": round(d["picp_after"], 3),
                       "PINAW before": round(d["pinaw_before"], 4), "PINAW after": round(d["pinaw_after"], 4)}
                      for rn, d in comp.items()], use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ #
# GUARANTEE DEMO
# ------------------------------------------------------------------ #
elif page == "Guarantee Demo":
    section_header("Architectural guarantees",
                   "Slide \u03c3_total to see all three guarantees respond on a live forward pass")
    model, _ = load_model()
    ds = load_test_data()
    daytime = np.where(ds.y.numpy().max(axis=1) > 0.1)[0]

    wi3 = st.selectbox("Test window", daytime[:50].tolist(),
                       format_func=lambda x: f"Window {x} \u2014 {ID_TO_REGIME.get(int(ds.regime_id[x].item()), '?')}")
    sigma_override = st.slider("\u03c3_total override", 0.0, 2.0, 0.5, 0.05)

    x_seq, x_nwp, x_sigma = ds.x_seq[wi3:wi3+1], ds.x_nwp[wi3:wi3+1], ds.x_sigma[wi3:wi3+1]
    with torch.no_grad():
        emb = model.backbone(x_seq, x_nwp, x_sigma)
        pool_out = model.expert_pool(emb["h_expert"])
        router_real = model.router(emb["h_expert"], emb["h_sigma"], x_sigma,
                                    pool_out["sigma_expert"], pool_out["stacked"])
        q_real = model.decoder(router_real["y_hat_routed"], emb["h_expert"],
                                emb["h_nwp"], router_real["sigma_total"])
        sigma_manual = torch.tensor([sigma_override], dtype=torch.float32)
        q_manual = model.decoder(router_real["y_hat_routed"], emb["h_expert"],
                                  emb["h_nwp"], sigma_manual)
        from torch.nn.functional import softplus
        try:
            beta = model.router.gate.beta
            T_manual = torch.exp(torch.clamp(softplus(beta) * sigma_manual, -8, 8))
            logits = model.router.gate.logits_net(torch.cat([emb["h_expert"], emb["h_sigma"]], dim=-1))
            gate_manual = torch.softmax(logits / T_manual, dim=-1).numpy().ravel()
        except AttributeError:
            gate_manual = router_real["gate_weights"].numpy().ravel()
            T_manual = router_real["temperature"]

    y_obs = ds.y[wi3].numpy()
    steps = np.arange(1, len(y_obs) + 1)
    real_T = float(router_real["temperature"].item())
    colors_bar = [REGIME_COLORS.get(n, "#636EFA") for n in EXPERT_ORDER]

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**G1: Quantiles never cross**")
        p10_m, p50_m, p90_m = (q_manual[k].numpy().ravel() for k in ("p10", "p50", "p90"))
        st.metric("Violations", int(np.sum(p10_m > p50_m + 1e-6)) + int(np.sum(p50_m > p90_m + 1e-6)))
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
        st.markdown("**G2: Width grows with \u03c3**")
        sigmas_sweep = np.linspace(0, 2, 25)
        widths_sweep = []
        for s in sigmas_sweep:
            with torch.no_grad():
                q_s = model.decoder(router_real["y_hat_routed"], emb["h_expert"],
                     emb["h_nwp"], torch.tensor([s], dtype=torch.float32))
            widths_sweep.append(float((q_s["p90"] - q_s["p10"]).mean()))
        monotone = all(widths_sweep[i] <= widths_sweep[i+1] + 1e-6 for i in range(len(widths_sweep)-1))
        st.metric("Monotone", "Yes" if monotone else "No")
        fig2 = go.Figure(data=go.Scatter(x=sigmas_sweep, y=widths_sweep, mode="lines",
                                          line=dict(color=COLORS["p50"], width=2.5)))
        fig2.add_vline(x=sigma_override, line_dash="dash", line_color=COLORS["observed"],
                       annotation_text=f"\u03c3={sigma_override:.2f}")
        fig2.update_layout(**PLOTLY_LAYOUT, height=280, xaxis_title="\u03c3_total", yaxis_title="Mean width")
        styled_axes(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with g3:
        st.markdown("**G3: Routing softens**")
        T_val = float(T_manual.item() if hasattr(T_manual, 'item') else T_manual)
        st.metric("Temperature", f"{T_val:.3f}")
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
elif page == "Baselines & Ablations":
    section_header("Model comparison", "ATERNET vs baselines and ablation variants")
    comp = load_comparison()
    if comp is None:
        st.info("Run `python -m src.evaluation.comparison_table` to generate comparison data.")
        st.stop()

    labels = {"model": "Model", "nrmse_pct": "nRMSE %", "nmae_pct": "nMAE %",
              "r2": "R\u00b2", "picp": "PICP", "pinaw": "PINAW",
              "winkler": "Winkler", "rmse_skill_vs_persistence": "Skill"}
    cols = list(labels.keys())
    rows = []
    for row in comp:
        r = {}
        for c in cols:
            v = row.get(c, float("nan"))
            if isinstance(v, float) and v != v: r[labels[c]] = "\u2014"
            elif isinstance(v, float): r[labels[c]] = f"{v:.2f}" if c in ("nrmse_pct", "nmae_pct") else f"{v:.4f}"
            else: r[labels[c]] = v
        rows.append(r)
    st.dataframe(rows, use_container_width=True, hide_index=True)

    models = [r["model"] for r in comp if isinstance(r.get("nrmse_pct"), (int, float)) and r["nrmse_pct"] == r["nrmse_pct"]]
    nrmses = [r["nrmse_pct"] for r in comp if isinstance(r.get("nrmse_pct"), (int, float)) and r["nrmse_pct"] == r["nrmse_pct"]]
    if models:
        bar_c = [COLORS["p50"] if "full" in m.lower() or "CQR" in m else COLORS["muted"] for m in models]
        fig = go.Figure(data=go.Bar(x=models, y=nrmses, marker_color=bar_c,
                                     text=[f"{v:.2f}%" for v in nrmses], textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="nRMSE %", xaxis_tickangle=-30)
        styled_axes(fig)
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ #
# TRAINING
# ------------------------------------------------------------------ #
elif page == "Training":
    history = load_history()
    section_header("Training history", f"{len(history)} epochs, early-stopped on validation pinball loss")
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