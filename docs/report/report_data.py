"""Numbers and references used by the report text. Everything numeric is read from saved outputs."""

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "outputs" / "metrics"
SUMMARY = json.loads((ROOT / "docs" / "results_summary.json").read_text(encoding="utf-8"))
FIG = Path(__file__).resolve().parent / "figures"


def js(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def seeds(pattern, key="headline"):
    return [js(f)[key] for f in sorted(glob.glob(str(pattern)))]


def ms(vals):
    v = np.array(vals, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def rows(site):
    return {r["model"].strip(): r for r in SUMMARY["sites"][site]["rows"]}


def fmt(r, key, nd=3):
    return f"{r[key]['mean']:.{nd}f} ± {r[key]['std']:.{nd}f}"


class D:
    pass


import subprocess


def _count_lines():
    n = 0
    for sub in ("src", "config", "tests"):
        for f in (ROOT / sub).rglob("*.py"):
            if "__pycache__" not in f.parts:
                n += len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
    return int(round(n, -2))


def _commits():
    try:
        return int(subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True).stdout.strip())
    except Exception:
        return 0


d = D()
d.loc = _count_lines()
d.commits = _commits()
d.Y = rows("yorkshire")
d.SW = rows("south_west")
d.VIT = rows("vellore")
d.cal = {k: SUMMARY["sites"][k]["calibration"] for k in ("yorkshire", "south_west", "vellore")}
d.A = "ATERNET (final: residual gate + shortcut + EMA)"          # final model label in summary
d.test = js(M / "test_metrics.json")
d.conf = js(M / "conformal_calibration.json")
d.lgbm1 = js(M / "baseline_lgbm.json")
d.back = js(M / "backtest_2026-10-01_2026-10-16.json")
d.vit_test = js(ROOT / "workspaces/vit/outputs/metrics/test_metrics.json")

# development of the model (3 seeds each)
dev_spec = [
    ("Original learned gate", "multiseed/seed*/ablation_full.json"),
    ("Uniform average (no gate)", "multiseed/seed*/ablation_no_gate.json"),
    ("Residual gate", "multiseed/seed*/ablation_residual_gate.json"),
    ("Residual gate + last-step shortcut", "multiseed/seed*/ablation_residual_gate_skip.json"),
    ("Residual gate + shortcut + EMA (final)", "multiseed_final/seed*/ablation_tune_ema.json"),
]
d.dev = []
for name, pat in dev_spec:
    hs = seeds(M / pat)
    d.dev.append((name, len(hs), *ms([h["nrmse_pct"] for h in hs]), *ms([h["winkler"] for h in hs]),
                  *ms([h["picp"] for h in hs])))

# accuracy options (3 seeds, on the residual-gate + shortcut architecture, no EMA unless stated)
opt_spec = [
    ("Reference (no EMA)", "full"), ("EMA weight averaging (decay 0.999)", "tune_ema"),
    ("Weight decay 1e-2", "tune_wd"), ("Learning rate 5e-4", "tune_lr"),
    ("Extra squared-error term on P50", "mse_p50"), ("Routing-metadata inputs", "meta_skip"),
    ("All four combined", "combo"),
]
d.opts = []
for name, key in opt_spec:
    hs = seeds(M / f"multiseed_final/seed*/ablation_{key}.json")
    d.opts.append((name, len(hs), *ms([h["nrmse_pct"] for h in hs]), *ms([h["winkler"] for h in hs])))

# stride-2 experiment (seeds 42 and 43)
def one(pattern):
    return [js(f)["headline"]["nrmse_pct"] for f in sorted(glob.glob(str(pattern)))]

d.s4_at = one(M / "multiseed_final/seed4[23]/ablation_tune_ema.json")
d.s2_at = one(M / "multiseed_final_stride2/seed*/ablation_tune_ema.json")
d.s4_lg = one(M / "multiseed_baselines/seed4[23]/baseline_lgbm.json")
d.s2_lg = one(M / "multiseed_baselines_stride2/seed*/baseline_lgbm.json")

# per-regime and per-horizon
d.regimes = ["dawn_ramp", "clear_peak", "overcast", "volatile"]
d.REGIME_PLAIN = {"dawn_ramp": "Morning ramp-up", "clear_peak": "Clear sunny midday",
                  "overcast": "Cloudy / overcast (incl. night)", "volatile": "Patchy, changing clouds"}
d.hz_at = d.test["per_horizon"]["rmse"]
d.hz_lg = d.lgbm1["per_horizon"]["rmse"]
d.hl = d.test["headline"]

# ---------------------------------------------------------------- references (IEEE)
REFS = {
    "antonanzas2016": 'J. Antonanzas, N. Osorio, R. Escobar, R. Urraca, F. J. Martinez-de-Pison, and F. Antonanzas-Torres, "Review of photovoltaic power forecasting," Solar Energy, vol. 136, pp. 78–111, 2016.',
    "wan2015": 'C. Wan, J. Zhao, Y. Song, Z. Xu, J. Lin, and Z. Hu, "Photovoltaic and solar power forecasting for smart grid energy management," CSEE Journal of Power and Energy Systems, vol. 1, no. 4, pp. 38–46, 2015.',
    "meer2018": 'D. W. van der Meer, J. Widén, and J. Munkhammar, "Review on probabilistic forecasting of photovoltaic power production and electricity consumption," Renewable and Sustainable Energy Reviews, vol. 81, pp. 1484–1512, 2018.',
    "sweeney2020": 'C. Sweeney, R. J. Bessa, J. Browell, and P. Pinson, "The future of forecasting for renewable energy," WIREs Energy and Environment, vol. 9, no. 2, e365, 2020.',
    "hochreiter1997": 'S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.',
    "vaswani2017": 'A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems 30, 2017, pp. 5998–6008.',
    "nie2023": 'Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "A time series is worth 64 words: Long-term forecasting with transformers," in Proc. International Conference on Learning Representations (ICLR), 2023.',
    "ke2017": 'G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in Advances in Neural Information Processing Systems 30, 2017, pp. 3146–3154.',
    "koenker1978": 'R. Koenker and G. Bassett, "Regression quantiles," Econometrica, vol. 46, no. 1, pp. 33–50, 1978.',
    "gneiting2007": 'T. Gneiting and A. E. Raftery, "Strictly proper scoring rules, prediction, and estimation," Journal of the American Statistical Association, vol. 102, no. 477, pp. 359–378, 2007.',
    "gneiting2014": 'T. Gneiting and M. Katzfuss, "Probabilistic forecasting," Annual Review of Statistics and Its Application, vol. 1, pp. 125–151, 2014.',
    "winkler1972": 'R. L. Winkler, "A decision-theoretic approach to interval estimation," Journal of the American Statistical Association, vol. 67, no. 337, pp. 187–191, 1972.',
    "jacobs1991": 'R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive mixtures of local experts," Neural Computation, vol. 3, no. 1, pp. 79–87, 1991.',
    "shazeer2017": 'N. Shazeer et al., "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," in Proc. International Conference on Learning Representations (ICLR), 2017.',
    "romano2019": 'Y. Romano, E. Patterson, and E. J. Candès, "Conformalized quantile regression," in Advances in Neural Information Processing Systems 32, 2019, pp. 3543–3553.',
    "bachlechner2021": 'T. Bachlechner, B. P. Majumder, H. H. Mao, G. W. Cottrell, and J. McAuley, "ReZero is all you need: Fast convergence at large depth," in Proc. Conference on Uncertainty in Artificial Intelligence (UAI), PMLR vol. 161, 2021, pp. 1352–1361.',
    "polyak1992": 'B. T. Polyak and A. B. Juditsky, "Acceleration of stochastic approximation by averaging," SIAM Journal on Control and Optimization, vol. 30, no. 4, pp. 838–855, 1992.',
    "loshchilov2019": 'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in Proc. International Conference on Learning Representations (ICLR), 2019.',
    "hersbach2020": 'H. Hersbach et al., "The ERA5 global reanalysis," Quarterly Journal of the Royal Meteorological Society, vol. 146, no. 730, pp. 1999–2049, 2020.',
    "bauer2015": 'P. Bauer, A. Thorpe, and G. Brunet, "The quiet revolution of numerical weather prediction," Nature, vol. 525, pp. 47–55, 2015.',
    "paszke2019": 'A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in Advances in Neural Information Processing Systems 32, 2019, pp. 8024–8035.',
    "pvlive": 'Sheffield Solar, University of Sheffield, "PV_Live: near-real-time estimates of GB solar PV generation." [Online]. Available: https://www.solar.sheffield.ac.uk/pvlive/ (accessed Oct. 2026).',
    "openmeteo": 'Open-Meteo, "Open-Meteo weather forecast and historical forecast APIs." [Online]. Available: https://open-meteo.com (accessed Oct. 2026).',
    "streamlit": 'Streamlit Inc., "Streamlit documentation." [Online]. Available: https://streamlit.io (accessed Oct. 2026).',
}


# ---------------------------------------------------------------- gate usage and night/day coverage (Yorkshire, test set)
import sys
sys.path.insert(0, str(ROOT))
P = np.load(ROOT / "outputs" / "predictions" / "test_predictions.npz", allow_pickle=True)
_G, _reg = P["gate_weights"], P["regime_id"]
d.gate_mat = np.array([_G[_reg == r].mean(axis=0) for r in range(4)])
from src.training.dataset import ATERNETDataset      # noqa: E402
from src.evaluation.conformal import daytime_mask     # noqa: E402
_day = daytime_mask(ATERNETDataset("test"))
_in = (P["y_true"] >= P["p10"]) & (P["y_true"] <= P["p90"])
d.day_share = float(_day.mean())
d.picp_all, d.picp_day, d.picp_night = float(_in.mean()), float(_in[_day].mean()), float(_in[~_day].mean())
