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

    # ---- the twenty papers studied for this project (details read from the PDFs) ----
    "wang2026dgn": 'Y. Wang, Z. Zhen, F. Wang, S. Sun, Y. Cheng, and J. Yu, "A distributed PV ultra-short-term power deterministic and probabilistic forecasting based on dynamic graph network with shape-amplitude criteria," IEEE Trans. Ind. Appl., vol. 62, no. 2, pp. 1863–1873, Mar./Apr. 2026, doi: 10.1109/TIA.2025.3604364.',
    "lo2025": 'H.-Y. Lo, Y.-K. Wu, Q.-T. Phan, and W.-S. Tan, "A novel QR-based probabilistic forecasting method for solar power generation," IEEE Trans. Ind. Appl., vol. 61, no. 4, pp. 5381–5393, Jul./Aug. 2025, doi: 10.1109/TIA.2025.3546206.',
    "he2026mrdl": 'T. He, Y. Xu, and Y. Guo, "A probabilistic forecasting method for photovoltaic power under complex weather conditions based on multi-resolution deep learning," in Proc. 2026 5th Int. Conf. Smart Grid and Green Energy (ICSGGE), 2026, pp. 125–130, doi: 10.1109/ICSGGE69348.2026.11508327.',
    "wang2025cg": 'H. Wang, R. Duan, and X. Yang, "A robust ensemble framework with context-gated dynamic fusion and temporal residual correction for photovoltaic power forecasting," in Proc. 2025 9th Int. Conf. Electrical, Mechanical and Computer Engineering (ICEMCE), 2025, pp. 1022–1026, doi: 10.1109/ICEMCE68156.2025.11466687.',
    "zhao2026twostage": 'H. Zhao, Y. Zhang, W. Huo, F. Lin, and J. Wang, "A two-stage ensemble learning framework for short-term nonparametric probabilistic photovoltaic power forecasting integrating multi-source numerical weather predictions," IEEE Trans. Power Syst., vol. 41, no. 2, pp. 1231–1245, Mar. 2026, doi: 10.1109/TPWRS.2025.3613446.',
    "he2026gan": 'Y. He, X. Chen, Y. Zhang, and X. Yang, "Day-ahead PV power probability density forecasting with GAN-QRCNLSTM based on high-resolution data reconstruction," IEEE Trans. Ind. Informat., vol. 22, no. 2, pp. 1428–1439, Feb. 2026, doi: 10.1109/TII.2025.3627108.',
    "heidrich2026": 'M. Z. Heidrich, M. A. F. Reis, C. K. Yamaguchi, L. O. Seman, and S. F. Stefenon, "Patch-based transformer with mixture of experts and conformal inference for curtailment time series forecasting," IEEE Access, vol. 14, pp. 64144–64160, 2026, doi: 10.1109/ACCESS.2026.3686958.',
    "lyu2024": 'C. Lyu and S. Eftekharnejad, "Probabilistic solar generation forecasting for rapidly changing weather conditions," IEEE Access, vol. 12, pp. 79091–79103, 2024, doi: 10.1109/ACCESS.2024.3407778.',
    "lin2025": 'F. Lin, Y. Zhang, H. Zhao, W. Huo, and J. Wang, "Short-term probabilistic forecasting for regional PV power based on convolutional graph neural network and parameter transferring," IEEE Trans. Power Syst., vol. 40, no. 3, pp. 2724–2736, May 2025, doi: 10.1109/TPWRS.2024.3503288.',
    "huang2026": 'C. Huang, Y. Wang, Z. Zhen, F. Xu, F. Wang, X. Liu, X. Xie, and Q. Fang, "Spatial-temporal attention mechanism and graph convolutional networks based distributed PV ultra-short-term probabilistic power forecasting," IEEE Trans. Ind. Appl., early access, 2026, doi: 10.1109/TIA.2026.3660789.',
    "rastgoo2025": 'R. Rastgoo, N. Amjady, R. Shah, and S. M. Muyeen, "A diffusion-based probabilistic ultra-short-term solar power prediction using the sky image sequences," IEEE Access, vol. 13, pp. 149532–149553, 2025, doi: 10.1109/ACCESS.2025.3600713.',
    "seman2026": 'L. O. Seman, S. F. Stefenon, J. P. Matos-Carvalho, A. Nied, and K.-C. Yow, "End-to-end decomposition-aware neural architecture search with sparse expert transformers for thermal energy forecasting," IEEE Access, vol. 14, pp. 67765–67787, 2026, doi: 10.1109/ACCESS.2026.3689436.',
    "phan2024": 'Q.-T. Phan, Y.-K. Wu, and Q.-D. Phan, "Enhancing one-day-ahead probabilistic solar power forecast with a hybrid transformer-LUBE model and missing data imputation," IEEE Trans. Ind. Appl., vol. 60, no. 1, pp. 1396–1408, Jan./Feb. 2024, doi: 10.1109/TIA.2023.3325798.',
    "jensen2024": 'V. Jensen, F. M. Bianchi, and S. N. Anfinsen, "Ensemble conformalized quantile regression for probabilistic time series forecasting," IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 7, pp. 9014–9025, Jul. 2024, doi: 10.1109/TNNLS.2022.3217694.',
    "song2025": 'K. Song, M. Kim, and H. Kim, "Graph-based large scale probabilistic PV power forecasting insensitive to space-time missing data," IEEE Trans. Sustain. Energy, vol. 16, no. 1, pp. 160–173, Jan. 2025, doi: 10.1109/TSTE.2024.3447023.',
    "wang2026clouddiff": 'K. Wang, T. Wang, S. Shan, W. Dou, J. Zhang, H. Wei, and K. Zhang, "Multimodal ensemble photovoltaic power forecasting incorporating diffusion-based satellite image prediction," IEEE Trans. Sustain. Energy, vol. 17, no. 2, pp. 1425–1436, Apr. 2026, doi: 10.1109/TSTE.2025.3619938.',
    "zhao2024": 'P. Zhao, W. Hu, D. Cao, Z. Zhang, Y. Huang, L. Dai, and Z. Chen, "Probabilistic multienergy load forecasting based on hybrid attention-enabled transformer network and Gaussian process-aided residual learning," IEEE Trans. Ind. Informat., vol. 20, no. 6, pp. 8379–8393, Jun. 2024, doi: 10.1109/TII.2024.3366946.',
    "ye2024": 'J. Ye, B. Zhao, D. Liu, Q. Wei, and Y. Wang, "TADNet: Temporal attention decomposition networks for probabilistic energy forecasting," IEEE Trans. Power Syst., vol. 39, no. 6, pp. 7190–7202, Nov. 2024, doi: 10.1109/TPWRS.2024.3380388.',
    "wang2026joint": 'W. Wang, Y. Wang, F. Wang, F. Xu, G. Feng, and Z. Zhen, "Ultra-short-term power probabilistic forecasting method for distributed photovoltaic based on joint distribution modeling," IEEE Trans. Ind. Appl., vol. 62, no. 1, pp. 26–36, Jan./Feb. 2026, doi: 10.1109/TIA.2025.3589961.',
    "abedinia2024": 'O. Abedinia, A. Ghasemi-Marzbali, M. Shafiei, B. Sobhani, G. B. Gharehpetian, and M. Bagheri, "Wind power forecasting enhancement utilizing adaptive quantile function and CNN-LSTM: A probabilistic approach," IEEE Trans. Ind. Appl., vol. 60, no. 3, pp. 4446–4457, May/Jun. 2024, doi: 10.1109/TIA.2024.3354218.',
    "hochreiter1997": 'S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.',
    "vaswani2017": 'A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems 30, 2017, pp. 5998–6008.',
    "nie2023": 'Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "A time series is worth 64 words: Long-term forecasting with transformers," in Proc. International Conference on Learning Representations (ICLR), 2023.',
    "ke2017": 'G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in Advances in Neural Information Processing Systems 30, 2017, pp. 3146–3154.',
    "koenker1978": 'R. Koenker and G. Bassett, "Regression quantiles," Econometrica, vol. 46, no. 1, pp. 33–50, 1978.',
    "gneiting2007": 'T. Gneiting and A. E. Raftery, "Strictly proper scoring rules, prediction, and estimation," Journal of the American Statistical Association, vol. 102, no. 477, pp. 359–378, 2007.',
    "winkler1972": 'R. L. Winkler, "A decision-theoretic approach to interval estimation," Journal of the American Statistical Association, vol. 67, no. 337, pp. 187–191, 1972.',
    "jacobs1991": 'R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive mixtures of local experts," Neural Computation, vol. 3, no. 1, pp. 79–87, 1991.',
    "shazeer2017": 'N. Shazeer et al., "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," in Proc. International Conference on Learning Representations (ICLR), 2017.',
    "romano2019": 'Y. Romano, E. Patterson, and E. J. Candès, "Conformalized quantile regression," in Advances in Neural Information Processing Systems 32, 2019, pp. 3543–3553.',
    "bachlechner2021": 'T. Bachlechner, B. P. Majumder, H. H. Mao, G. W. Cottrell, and J. McAuley, "ReZero is all you need: Fast convergence at large depth," in Proc. Conference on Uncertainty in Artificial Intelligence (UAI), PMLR vol. 161, 2021, pp. 1352–1361.',
    "polyak1992": 'B. T. Polyak and A. B. Juditsky, "Acceleration of stochastic approximation by averaging," SIAM Journal on Control and Optimization, vol. 30, no. 4, pp. 838–855, 1992.',
    "loshchilov2019": 'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in Proc. International Conference on Learning Representations (ICLR), 2019.',
    "hersbach2020": 'H. Hersbach et al., "The ERA5 global reanalysis," Quarterly Journal of the Royal Meteorological Society, vol. 146, no. 730, pp. 1999–2049, 2020.',
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
