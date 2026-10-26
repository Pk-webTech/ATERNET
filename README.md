# ATERNET

**Adaptive Temporal Expert Routing Network with Dynamic Quantile Decoding
for Probabilistic Solar Power Forecasting**

Give it a weather forecast and recent solar generation, and it predicts the
next 8 hours of PV power as P10 / P50 / P90 intervals. It is trained and
tested on **real measured generation**, and it can forecast **live**
without owning a solar panel and without any API key.

- **Phase 1 — Data Pipeline** (real-data acquisition → cleaning → features → windows → splits)
- **Phase 2 — Temporal Encoding** (PatchTST backbone + NWP cross-attention over history *and* forecast horizon)
- **Phase 3 — Expert Modeling** (4 regime experts + dense expert pool)
- **Phase 4 — Dual-Uncertainty Routing** (σ_NWP + σ_Expert fusion + adaptive gating)
- **Phase 5 — Probabilistic Decoding** (P10/P50/P90, non-crossing by construction)
- **Training / Evaluation / Visualization / Tests** (55 tests, all passing)
- **Live forecasting + out-of-sample backtesting** (`predict.py`, "Live Forecast" page in the app)

`ATERNET` (`src/models/aternet.py`) wires Phases 2–5 into one model.

## Where the real data comes from

There's no access to a panel's own meter, so both inputs come from free
public sources (configured in `config/data_source.py`):

| Input | Source | What it provides |
|---|---|---|
| Solar generation | [Sheffield Solar **PV_Live**](https://www.solar.sheffield.ac.uk/pvlive/) | Measured half-hourly PV output + installed capacity for every GB region, 2013→now, updated live |
| Weather | [**Open-Meteo**](https://open-meteo.com) | Forecasts from 5 independent NWP models (GFS, ICON, UKMO, Météo-France, GEM): irradiance, cloud, temperature, wind |

The five weather models form a **multi-model ensemble**: their mean is the
weather forecast, their spread is σ_NWP (the uncertainty signal Phase 4
routes on). Training uses Open-Meteo's historical-forecast archive; live
forecasting uses the same models from the live API, so inputs match.

Default region: **NPG Yorkshire** (PV_Live region 23, ~1.17 GWp installed),
trained on 2023-01-01 → 2026-09-30 (65,713 half-hours, 16,413 windows).
Weather is averaged over three points across the region
(`src/phase1/regions.py`). Change `PVLIVE_REGION_ID` to use any GB region.

The original single-rooftop source (openclimatefix/uk_pv + ERA5 via CDS) is
still available as `DATA_SOURCE = "uk_pv_era5"`, but it has no live feed.

## Results

### Held-out test split (Mar → Sep 2026, 2,462 windows, measured PV_Live output)

Main model: residual gate + last-step shortcut + EMA weight averaging
(`python -m src.training.train --residual-gate --last-step-skip --ema-decay 0.999`),
seed 42, after per-regime conformal calibration.

| Metric | Value |
|---|---|
| nRMSE (% of capacity) | **3.69** |
| nMAE (% of capacity) | **2.00** |
| R² | **0.963** |
| PICP (nominal 0.80) | **0.816** (0.804 before calibration) |
| PINAW (mean interval width) | 0.0504 (0.0486 before) |
| Winkler score | 0.0924 (0.0929 before) |
| RMSE skill vs persistence | **0.836** |
| Quantile-crossing violations | **0** |

Per-regime coverage after calibration: overcast 0.824, dawn_ramp 0.821,
clear_peak 0.797, volatile 0.787 (before: clear_peak 0.748, volatile 0.752).
The volatile adjustment is estimated from only 44 validation windows and
the validation and test seasons differ (validation coverage was 0.88 before
calibration, test 0.80), so treat the volatile figure as approximate. Full numbers: `outputs/reports/evaluation_report.md`.

### Multi-seed comparison (seeds 42–44, mean ± std)

Single-seed comparisons were misleading on this data (one LSTM seed gave
nRMSE 9.19 where others gave 3.7), so the headline numbers are multi-seed.

| Model | nRMSE % | PICP | PINAW | Winkler |
|---|---|---|---|---|
| **ATERNET (EMA)** | **3.70 ± 0.02** | 0.809 | **0.0496** | **0.0922** |
| ATERNET (EMA), 3-model ensemble | 3.68 | 0.813 | 0.0496 | 0.0914 |
| LightGBM quantile | 3.72 ± 0.01 | 0.814 | 0.0644 | 0.0992 |
| PatchTST quantile | 4.16 ± 0.00 | 0.880 | 0.0737 | 0.0971 |
| LSTM quantile | unstable: 3 of 6 seeds failed (7.0–9.2); the other 3 reached 3.63–3.76 (Winkler 0.086–0.093) | | | |

ATERNET ties LightGBM on point accuracy (the 0.02 gap is about one seed
spread) and has the sharpest intervals and the best Winkler score of the
stable models. A well-trained LSTM seed matches or beats it, but the LSTM
cannot be trusted without seed selection; ATERNET's seed spread is 0.02.
`outputs/reports/comparison_table.md` still shows seed-42 baseline rows
(including the failed LSTM run) and ablation rows from an older gate; use
`python build_final_table.py` (-> `outputs/reports/final_comparison.md`) instead.

### Second region: South West England (PV_Live region 22, seeds 42-44)

Same pipeline, same hyperparameters and training options (chosen on the Yorkshire
experiments, nothing re-tuned for this site), isolated in its own workspace
(`ATERNET_WORKSPACE=workspaces/sw ATERNET_REGION_ID=22`). No conformal
calibration applied here.

| Model | nRMSE % | PICP | PINAW | Winkler |
|---|---|---|---|---|
| **ATERNET (EMA)** | **2.63 ± 0.01** | 0.846 | **0.0375** | **0.0634** |
| LightGBM quantile | 2.69 ± 0.00 | 0.787 | 0.0410 | 0.0670 |
| PatchTST quantile | 2.92 ± 0.03 | 0.902 | 0.0610 | 0.0732 |

ATERNET is ahead of both baselines on accuracy, interval width and Winkler score,
with a seed spread (0.01) much smaller than the gaps (0.06 to LightGBM). Its
intervals over-cover at 0.846 (nominal 0.80), so calibration would tighten them
further. The LSTM and the ablations were not run on this site, and nRMSE is not
comparable across regions (different capacity and aggregation).

### Ablations (3 seeds each; reports in `outputs/reports/`)

On the final architecture (`multiseed_ablation_final.md`, `multiseed_experts.md`,
`multiseed_accuracy.md`):

- **Dense expert loss matters.** Removing it: nRMSE 3.83 → 3.94, coverage
  0.82 → 0.78, Winkler 0.097 → 0.105.
- **Everything else is within seed noise:** no gate, fixed temperature, fixed
  interval width, σ_NWP-only, σ_expert-only, a single expert instead of four,
  and no regime weighting. On this data the adaptive routing, the
  σ-conditioned width, and regime specialisation do **not** measurably
  improve accuracy; the experts behave like a small ensemble.
- **The original learned gate hurt:** nRMSE 4.39 ± 0.20 vs 4.12 with a plain
  average (`multiseed_ablation.md`). The residual gate (starts at uniform
  routing) removes that harm; the last-step linear shortcut then took it from
  4.12 to 3.83, and EMA weight averaging to 3.70.
- **No effect:** weight decay 1e-2, learning rate 5e-4, a P50 squared-error
  term, and routing-metadata inputs.

- **Denser training windows (stride 2, 2× data) did not help ATERNET** but
  did help LightGBM (seeds 42/43; validation and test sets unchanged,
  built by `src/phase1/densify_train.py`):

  | | stride 4 | stride 2 |
  |---|---|---|
  | ATERNET nRMSE | 3.69, 3.68 | 3.75, 3.72 |
  | LightGBM nRMSE | 3.71, 3.72 | 3.68, 3.68 |

  So ATERNET's tie with LightGBM on accuracy holds only at the original data
  density; with more windows LightGBM pulls ahead on accuracy (ATERNET still
  has the sharper intervals). The schedule was not retuned for the larger set,
  so this is not proof that ATERNET cannot use more data.

Live forecasting (`predict.py`, `app.py`) works with the main checkpoint;
checkpoints trained with metadata inputs (`n_meta_features > 0`) are not
supported by the live forecaster.

### Out-of-sample backtest through the live pipeline (1–16 Oct 2026)

The model never saw these dates. Forecasts were issued 6×/day from raw API
data using only information available at each issue time, then scored
against what PV_Live measured:

| Metric | Value |
|---|---|
| Forecasts / half-hours scored | 96 / 1,536 |
| nMAE / nRMSE (% of capacity) | 2.62 / 4.08 |
| Daylight MAE | **42.2 MW** (mean output 272 MW; day-ahead persistence 147 MW) |
| RMSE skill vs persistence | 0.704 |
| PICP (nominal 0.80) | 0.749 |

(Run with the current main checkpoint. The earlier model scored nMAE/nRMSE
3.13 / 4.87, daylight MAE 50 MW, skill 0.646, PICP 0.757.)

![backtest](outputs/plots/backtest_2026-10-01_2026-10-16.png)

**Caveats.** (1) Open-Meteo's historical-forecast archive stitches together
the first hours of successive model runs, so in training and backtests the
"forecast" weather is somewhat better than a single live run's 8-hour
forecast. Expect live errors to be a little larger. (2) PV_Live is a regional
*aggregate*, which is smoother than a single site; a single rooftop system
will have larger relative errors. (3) The intervals are slightly too narrow
in clear_peak/volatile conditions and in the October backtest (0.75–0.76 vs
0.80 nominal).

## India demonstration (SIMULATED PV, not measured data)

No open measured PV record exists for the VIT Vellore area, so
`src/phase1/simulated_pv.py` generates one: a documented physical PV model
(temperature derating, soiling, inverter clipping, noise) driven by ERA5
**reanalysis** weather at Vellore for an illustrative 1 MWp plant. The model
inputs are the usual Open-Meteo **forecast** models, a different source, so the
forecast-vs-actual weather error is real. Even so, results on this site measure
how well ATERNET learns the simulator from forecast weather, **not** real-world
accuracy, and must be reported as simulated.

```powershell
$env:ATERNET_WORKSPACE = "workspaces/vit"       # isolated data/checkpoints/outputs
$env:ATERNET_REGION_ID = "900"                  # VIT Vellore (SIMULATED PV)
$env:ATERNET_START_DATE = "2024-01-01"          # all chosen models cover Vellore from here
$env:ATERNET_WEATHER_MODELS = "gfs_seamless,icon_seamless,meteofrance_seamless,gem_seamless"
python run_pipeline.py
python -m src.training.train --residual-gate --last-step-skip --ema-decay 0.999 --epochs 50
python -m src.evaluation.evaluate
```

UKMO's archive only starts around mid-January 2025 at Vellore, so it is left
out to keep the ensemble membership constant over the training period.
`predict.py live` is not available for this site (there is no live PV feed).

## Quickstart

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scriptsctivate
pip install -r requirements.txt

# 2. Download real data + build the dataset (Phase 1, ~15 min first time)
python run_pipeline.py

# 3. Train (~10-15 min on CPU). Main configuration:
python -m src.training.train --residual-gate --last-step-skip --ema-decay 0.999 --epochs 50 --batch-size 64

# 4. Evaluate on the test split
python -m src.evaluation.evaluate --split test

# 5. Forecast the next 8 hours from right now
python predict.py live

# 6. Replay past forecasts and score them against measured generation
python predict.py backtest --start 2026-10-01 --end 2026-10-16

# Dashboard (includes the Live Forecast page)
streamlit run app.py

# Tests
python -m pytest tests/ -q
```

API responses for fully-past date ranges are cached in `data/cache/http/`,
so re-running the pipeline doesn't re-download. Changing the region or
dates in `config/data_source.py` requires deleting `data/raw/*` first.

## Phase 1 outputs

```
data/raw/pv/pv_raw.parquet              measured PV (timestamp, pv_power_kw, capacity_kw, solar_elevation_deg)
data/raw/nwp/nwp_ensemble.parquet       weather per model/member, aligned to the PV half-hour grid
data/raw/metadata/resolved_site_metadata.json   region, coordinates, capacity, source
data/interim/pv_clean.parquet           cleaned, gap-filled, capacity-normalized PV
data/interim/nwp_clean.parquet          ensemble mean + std per timestamp
data/interim/features.parquet           merged + engineered feature table
data/processed/dataset_{train,val,test}.npz   X_seq, X_sigma, X_nwp, y, meta
data/processed/scalers.pkl              min-max scalers fit on train only
data/processed/dataset_metadata.json    shapes, feature lists, config snapshot
```

## Tensor shapes

| Tensor | Shape | Description |
|---|---|---|
| `X_seq` | `(N, 48, 10)` | Temporal encoder input: pv_norm, ghi, cloud, temp, wind, clear_sky_index, hour/doy sin-cos |
| `X_sigma` | `(N, 48, 6)` | Raw uncertainty features (`sigma_nwp_*` + ensemble spread; expert-disagreement column is a placeholder until Phase 3) |
| `X_nwp` | `(N, 64, 4)` | NWP ensemble-mean features over the 48-step lookback **and** the 16-step forecast horizon |
| `y` | `(N, 16)` | pv_norm target, 16-step (8h @ 30min) horizon |
| `meta` | `(N, 9)` | Regime one-hot/id + volatility features, snapshot at forecast issue time |

Window size = 48 steps (24h lookback), horizon = 16 steps (8h ahead),
stride = 4, native frequency = 30 minutes — all configurable in
`config/model_config.py` (`DataConfig`).

## Phase 2 — Temporal Encoder

`src/phase2/` implements a PatchTST-style backbone:

| File | Role |
|---|---|
| `patch_embedding.py` | Splits `(B,48,F)` into 11 overlapping patches (patch_len=8, stride=4), projects to `d_model=128` |
| `positional_encoding.py` | Learnable per-patch positional embedding |
| `feature_encoder.py` | `NWPCrossAttention` (PV patches attend to NWP patches) + `SigmaEncoder` (small MLP on raw σ features at forecast issue time — kept out of the transformer to preserve calibrated physical meaning) |
| `transformer_encoder.py` | 3-layer pre-norm transformer encoder (8 heads, d_ff=256) |
| `temporal_backbone.py` | Wires everything into one module producing 3 outputs: `h_expert (B,128)`, `h_sigma (B,32)`, `h_nwp (B,32)` |

Three separate output heads (rather than one shared embedding) because
the expert pool, uncertainty gate, and quantile decoder that consume
them downstream have different gradient scales and information needs.

`src/models/aternet.py` wires the backbone into the top-level `ATERNET`
model class that later phases attach to.

Verified: 504,064 parameters, forward + backward pass on real Phase 1
data with zero `None` gradients.

## Phase 3 — Expert Pool

`src/phase3/` implements 4 regime experts, each with capacity and loss
sized to its regime (see each file's docstring for the data-driven
justification):

| Expert | Hidden dims | Loss | Physical constraint | Params |
|---|---|---|---|---|
| `dawn_ramp_expert.py` | 256→256→128 | MSE + slope hinge | residual+cumsum ramp parameterization | 134,929 |
| `clear_peak_expert.py` | 128→64 | MSE | none (near-Gaussian regime) | 26,064 |
| `overcast_expert.py` | 128→128 | Huber (δ=0.05) | output cap ≤ 0.80 (see note) | 35,344 |
| `volatile_expert.py` | 512→256→128 | MSE | none (widest capacity, highest dropout) | 233,872 |

`expert_pool.py` runs all four densely on every sample and computes
`sigma_expert` — the epistemic-uncertainty signal from cross-expert
disagreement, as a standard deviation in pv_norm units (√ of mean
cross-expert variance) — on **detached** predictions, so the routing gate that
will consume it (Phase 4) can't backprop into the experts and distort
their training objective. `expert_base.py` defines the shared interface
(`forward`, `primary_loss`, `auxiliary_loss`, `total_loss`).

Verified: 430,209 pool parameters, forward + backward pass through the
full Phase 1 → Phase 2 → Phase 3 chain, overcast cap respected,
`sigma_expert.requires_grad == False` confirmed.

## Phase 4 — Dual-Uncertainty Routing

`src/phase4/` implements the routing gate: `Gate = f(h_expert, σ_NWP, σ_Expert)`.

| File | Role | Math |
|---|---|---|
| `sigma_nwp.py` | Extracts σ_NWP — deliberately **not learned**, stays a physically-grounded reduction of ensemble-spread features | `σ_NWP = mean_v(σ_nwp_v)` over 5 ensemble-spread columns at forecast issue time |
| `sigma_expert.py` | Calibrates Phase 3's raw σ_Expert to a comparable scale, via a single monotonic scalar (preserves disagreement ranking) | `σ_Expert_cal = softplus(scale) · σ_Expert_raw` |
| `uncertainty_fusion.py` | Combines both sources: log1p variance-stabilization → per-source BatchNorm → input-dependent convex combination | `σ_total = α·σ_NWP_norm + (1-α)·σ_Expert_norm`, `α = sigmoid(w·h_expert + b)` |
| `gating_network.py` | Produces routing weights, with temperature that **grows** with above-average combined uncertainty (softer/more-ensembled routing) and **shrinks** with below-average uncertainty (sharper routing) | `Gate = softmax(logits / T)`, `T = exp(softplus(β)·σ_total)` — always > 0, no clamping needed |
| `adaptive_router.py` | Wires the above + performs the weighted aggregation of Phase 3's expert predictions | `ŷ_routed = Σ_k Gate_k · ŷ_k` |

Key design decisions, each made so a reviewer can't dismiss it as
unjustified complexity: σ_NWP stays deterministic/physical rather than
learned; σ_Expert calibration is a single scalar (not an MLP) so it
can't distort the disagreement ranking it's supposed to represent; the
fusion weight α is predicted from temporal context (not from the sigmas
themselves), so the model learns *when* to trust which uncertainty
source rather than baking in a fixed prior; and the temperature is
exponential in σ_total so it's guaranteed positive by construction.

Verified: 10,891 router parameters, forward + backward pass through the
full Phase 1 → 2 → 3 → 4 chain, gate weights confirmed to sum to 1 per
sample, σ_Expert's detachment from Phase 3 confirmed intact.

## Phase 5 — Probabilistic Decoder

`src/phase5/` produces the final P10/P50/P90 forecast via **adaptive
interval-width decoding** rather than dynamic per-quantile movement —
the original design's critical review found dynamic quantile movement
mathematically invalid for training (pinball loss requires a *fixed*
quantile level; moving it breaks the proper-scoring-rule guarantee).

| File | Role | Math |
|---|---|---|
| `prediction_head.py` | Bounded residual correction on top of Phase 4's routed forecast, conditioned on raw NWP context + uncertainty | `p50 = clamp(ŷ_routed + tanh(MLP(h_nwp, σ_total))·0.1, 0, 1)` |
| `interval_width_head.py` | Asymmetric, strictly non-negative half-widths, monotonically non-decreasing in σ_total | `width_h = softplus(base_h + softplus(γ_h)·σ_total)` |
| `quantile_decoder.py` | Combines median + widths into P10/P90 — non-crossing **guaranteed by construction**, not by penalty | `P10=clamp(P50-lower,0,1)`, `P90=clamp(P50+upper,0,1)` |
| `probabilistic_decoder.py` | Wires the above into one module | — |

`src/losses/pinball_loss.py` (fixed-τ pinball loss, safe here precisely
*because* crossing is structurally impossible) and
`src/losses/interval_width_loss.py` (sharpness regularizer, prevents the
trivial "predict everything as very wide" solution) are the natural
companion losses — together they approximate the interval (Winkler)
score, a proper scoring rule for prediction intervals.

Verified: full Phase 1→5 forward pass, **non-crossing confirmed on real
data** (`P10 ≤ P50 ≤ P90` for every sample/step), pinball + width-penalty
backward pass with zero broken gradients across all 967,132 model
parameters.

## Training

`src/training/` implements the joint training loop. All phases train together:
the router's usefulness depends on competent experts and vice versa, so
freezing either optimizes against a target that isn't there yet.

| File | Role |
|---|---|
| `dataset.py` | In-memory `Dataset`/`DataLoader` over the Phase 1 `.npz` splits |
| `trainer.py` | Epoch loop, cosine LR, grad clipping, early stopping, checkpointing |
| `checkpoint.py` | Save/load with architecture kwargs embedded; `EarlyStopping` |
| `train.py` | CLI entry point |

The **composite objective** (`src/losses/composite_loss.py`):

```
L = 1.00 · pinball(P10,P50,P90)   ← the deliverable; proper at each fixed τ
  + 0.05 · mean(lower + upper)    ← sharpness; without it coverage is trivially gamed
  + w(t) · Σ_k expert_k loss      ← keeps all four experts learning (1.0 → 0.3 after warmup)
  + 0.01 · load_balance(gate)     ← guards against gate collapse
```

The dense expert term is the non-obvious one. The routed forecast only
backpropagates into expert *k* in proportion to its gate weight, so an expert
the gate currently distrusts gets almost no gradient and can never improve
enough to earn trust back — a self-reinforcing deadlock. Training every expert
densely against the target in parallel breaks it, then anneals down so the
routed objective dominates once the experts are competent.

Model selection is on validation **pinball** loss, not the composite total:
the composite's expert weight changes across epochs, so comparing totals
between epoch 4 and epoch 20 compares two different objectives.

## Evaluation

`src/evaluation/` reports deterministic metrics (MAE, RMSE, bias, R²),
probabilistic metrics (PICP, PINAW, ACE, Winkler score, pinball, approximate
CRPS), and two baselines (persistence and climatology) — a forecasting result
without a baseline is uninterpretable. Everything is broken out **per regime**
and **per horizon step**, and the non-crossing guarantee is re-verified on the
actual predictions rather than assumed from the architecture.

## Visualization

`python -m src.visualization.generate_report` writes to `outputs/plots/`:
training history, gate share over training (the collapse detector), forecast
fan charts, reliability diagram, per-horizon error growth, uncertainty-vs-error
scatter, and the gate-by-regime matrix — plus NWP cross-attention maps to
`outputs/attention_maps/`.

## Tests

```bash
python -m pytest tests/ -q     # 66 passed
```

| File | Covers |
|---|---|
| `test_phase1_pipeline.py` | Shapes, no NaNs, **temporal-leakage check**, train-only scaler fit, regime coverage, gap rejection |
| `test_phases_2_5.py` | The three architectural guarantees, output ranges, σ_expert detachment, gradient health |
| `test_losses_and_metrics.py` | Hand-computed loss/metric values, pinball asymmetry, Winkler trade-off behaviour |
| `test_realtime_sources.py` | Solar geometry vs longitude, Open-Meteo parsing + time alignment, per-row capacity normalization, live inference window layout |
| `test_integration.py` | Real training run, loss decrease, gate non-collapse, checkpoint round-trip, baseline comparison, cap validation |

The three guarantees are enforced structurally and each has a named test:
quantiles never cross, intervals widen monotonically with σ_total, and routing
softens under uncertainty. See `docs/architecture.md` §2.

## Project structure

See `docs/architecture.md` for the full system diagram, the three
architectural guarantees, and a frank limitations list. The directory layout mirrors the target repo structure:
`config/` for all hyperparameters and paths, `src/phase{1..5}/` for the
pipeline stages, `src/{models,losses,training,evaluation,visualization}/`
for shared components used across phases, `data/`, `checkpoints/`, `outputs/`
and `logs/` as working directories (git-ignored, see `.gitignore`).

## Requirements

Python 3.10+. See `requirements.txt` / `environment.yml`.

Verified on Python 3.12 with torch 2.13, numpy 2.4, pandas 3.0. Note that
`numpy<2.0` was removed from the original pin: nothing in the project needed
it, and it blocked installation alongside current pandas/torch builds.
