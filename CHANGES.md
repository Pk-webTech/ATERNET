# Changes

## Accuracy and ablation pass (Oct 2026)

All new options are opt-in and off by default; older checkpoints load unchanged.

### Fixed
- **Conformal calibration did almost nothing** (`src/evaluation/conformal.py`). The score
  averaged signed violations over the 16 horizon steps, giving adjustments of ~0 (and a
  negative one for volatile, which shrank that interval). Scores are now per forecast step;
  clear_peak coverage 0.748 -> 0.797, volatile 0.752 -> 0.787, aggregate PICP 0.804 -> 0.816,
  Winkler 0.0929 -> 0.0924. The old behaviour is available as `score="window_mean"`.
- `fixed_temp` ablation crashed (`logits_net` vs `logit_net`); `no_dense_expert`
  never actually disabled the dense expert loss (it set an attribute nothing read).

### Added
- `src/phase1/densify_train.py`, `ATERNET_PROCESSED_DIR`, `--tag` on the multi-seed scripts: stride-2
  training set experiment with unchanged val/test splits (negative result for ATERNET, see README).
- `residual_gate` (`GatingNetwork`, `AdaptiveRouter`, `ATERNET`): softmax(delta * logits / T)
  with delta initialised to 0, so routing starts uniform.
- `last_step_skip` (`ATERNET`): zero-initialised linear shortcut from the last
  observed step and the forecast-horizon NWP to the routed forecast.
- `n_meta_features` (shortcut also receives routing/volatility metadata),
  `w_p50_mse` loss term, and EMA weight averaging (`TrainingConfig.ema_decay`, `--ema-decay`).
- `run_multiseed.py`, `run_multiseed_baselines.py`: multi-seed ablations and
  baselines written to separate directories (`multiseed*/`).
- Ablation variants: `single_expert`, `no_regime_weights`, `full`, `residual_gate`,
  `residual_gate_skip`, `tune_*`, `mse_p50`, `meta_skip`, `combo`.

### Results
Test nRMSE 4.24 -> 3.69, Winkler 0.104 -> 0.0929 (main model, seed 42). Over three
seeds: 3.70 ± 0.02 (LightGBM 3.72 ± 0.01). See README "Results" for the
ablation conclusions, including that gating/regime specialisation did not help.
Previous artifacts are kept in `checkpoints/`, `outputs/metrics/`, `outputs/reports/`
under `legacy_gate/`, `residual_gate_only/` and `final_no_ema/`.

---

## Real-world data + live forecasting

### Fixes
1. **uk_pv units / capacity were wrong** (`download_pv.py`, `pv_processor.py`).
   uk_pv reports `generation_wh` (energy per half-hour), but it was treated as
   kW and divided by the 1000 kW placeholder capacity of the demo Chennai site
   instead of the real system's 3.36 kWp. Peaks were clipped to 1.0. Now
   converted Wh -> kW and normalized by the actual capacity. A per-row
   `capacity_kw` column takes priority, because regional capacity grows over
   time (Yorkshire: 704 -> 1165 MWp over 2023-2026).
2. **The model never saw the weather forecast for the hours it predicts**
   (`create_windows.py`). `X_nwp` covered only the lookback. It now spans
   lookback + horizon (48 + 16 steps). Measured-PV features still stop at the
   issue time, so there is no target leakage. The LSTM / LightGBM baselines
   were adapted to the longer `X_nwp`.
3. **Solar elevation ignored longitude and the equation of time**. Correct
   near Greenwich, ~5 h off for India. Moved to `src/utils/solar.py` (NOAA formulation).

### Added
- `config/data_source.py`, `src/phase1/pvlive_openmeteo.py`, `src/phase1/regions.py`:
  measured regional PV from Sheffield Solar PV_Live + 5-model Open-Meteo
  weather ensemble. Free, no API keys, historical and live. Now the default source.
- `src/inference/forecaster.py`, `predict.py`: `live` forecast and
  out-of-sample `backtest` through one shared raw-data -> forecast code path.
- "Live Forecast" page in `app.py`.
- `tests/test_realtime_sources.py` (5 offline tests). The integration fixture
  now samples its 1024 windows across the whole training period. On multi-year
  data the first 1024 windows are one winter, which trained a zero-predictor.

### Results (real data)
Test split: nRMSE 4.24%, R² 0.951, PICP 0.800. Out-of-sample backtest
(1-16 Oct 2026): daylight MAE 50 MW vs 146 MW for day-ahead persistence.
Previous uk_pv/ERA5 artifacts were moved to `archive/uk_pv_era5_2019/`.

---

## Earlier pass

Everything added and fixed in this pass, in severity order. Written so it can
be split into separate commits if you'd rather have a granular history — each
numbered fix is self-contained and touches the files listed.

## Fixes

### 1. PV power model conflated module efficiency with nameplate rating
**Files:** `src/phase1/download_pv.py`

`generate_synthetic_pv` computed power as `ghi * 0.18 * capacity/1000`, which
multiplies a module efficiency (an area-to-power conversion) by a nameplate
rating. Peak `pv_norm` came out at ~0.17, so the whole `[0,1]` target range was
dead and every constant calibrated against it was silently meaningless.

Replaced with the standard PVWatts-style DC model:

```
P = P_nameplate * (GHI / G_STC) * temp_derate * system_derate
```

Clear-sky noon now maps to ~85% of nameplate; targets peak at 0.757.

**Knock-on effect this also fixed:** the regime thresholds are defined on
`pv_norm` rolling std, so they were never triggering. Before this fix the
`volatile` regime had **3 training windows and zero in val/test** — the
233k-parameter volatile expert was training on nothing and could not be
evaluated at all. After: 518 / 84 / 36.

### 2. Overcast expert's 0.40 cap made it unable to fit its own regime
**Files:** `src/phase3/overcast_expert.py`

The regime label is assigned at forecast issue time, but the expert predicts
the following four hours. An overcast window frequently clears within that
horizon: overcast-labelled targets average 0.067 but reach 0.746. A hard 0.40
sigmoid ceiling made the expert structurally incapable of predicting the
clearing, no matter how long it trained — and the error hid inside a
merely-mediocre loss curve rather than surfacing as a bug.

Raised to 0.80 (a genuine physical ceiling; nothing in the dataset exceeds
~0.76). Added `verify_cap()`, called by the test suite against real targets, so
the constant can't drift out of agreement with the data again.

### 3. `sigma_expert` was a variance named as a standard deviation
**Files:** `src/phase3/expert_pool.py`, `src/phase4/sigma_expert.py`

Computed as `var(dim=1).mean(dim=1)` — a squared quantity, non-comparable with
`sigma_nwp` (an average of `[0,1]`-scaled spread features), and it made the
`log1p` transform in Phase 4 behave differently from what that module's
docstring described. Now takes the root. Monotone, so the sample ranking by
disagreement — the property Phase 4's calibrator exists to preserve — is
unchanged.

### 4. Gate temperature could overflow or underflow
**Files:** `src/phase4/gating_network.py`

`T = exp(softplus(beta) * sigma_total)` overflows to `inf` for a large positive
argument (softmax then silently returns uniform 1/n) and underflows to `0` for
a large negative one (making `logits / T` a division by zero → NaN). The
exponent is now clamped to ±8, bounding T to `[3.4e-4, 2981]` — far wider than
any behaviourally distinguishable temperature.

### 5. Interval-width monotonicity guarantee had a hole
**Files:** `src/phase5/interval_width_head.py`

`sigma_total` fed the base MLP *and* multiplied gamma. The softplus on gamma
guarantees a non-negative contribution, but nothing stopped the MLP from
learning a compensating negative path that cancelled it — so the advertised
"width can only grow with uncertainty" guarantee was not actually enforced end
to end. `sigma_total` is now detached on the context path: the network still
sees it as context but cannot use it to construct the cancelling gradient.

### 6. Trainer hardcoded checkpoint paths
**Files:** `src/training/trainer.py`

`Trainer` always wrote to `checkpoints/aternet_best.pt`, so running the
integration tests silently destroyed the real trained model. Output paths are
now constructor arguments; tests write to a scratch directory.

### Smaller fixes
- `src/phase1/create_windows.py` — contiguity was re-diffed per window (O(N·W)
  for something one vectorized pass answers for all windows); also fixes a
  read-only-array crash.
- `src/phase1/pv_processor.py` — ancillary columns kept their
  reindex-introduced NaNs, poisoning `clear_sky_index` and regime assignment.
- `src/phase4/uncertainty_fusion.py` — BatchNorm emitted NaN on a batch of 1.
- `requirements.txt` — dropped the `numpy<2.0` pin; nothing needed it and it
  blocks installation alongside current pandas/torch.

## Added

| Area | Contents |
|---|---|
| `src/training/` | `dataset.py`, `trainer.py`, `checkpoint.py`, `train.py` |
| `src/losses/` | `composite_loss.py`, `load_balance_loss.py` |
| `src/evaluation/` | `metrics.py`, `baselines.py`, `evaluate.py` |
| `src/visualization/` | `plots.py`, `generate_report.py` |
| `tests/` | 50 unit + integration tests |
| root / docs | `run_all.py`, `docs/architecture.md`, this file |

## Results

Test split, 1,309 held-out windows, 28 epochs (early-stopped):

| Metric | Value |
|---|---|
| nRMSE | 4.31% of capacity |
| nMAE | 1.72% |
| R² | 0.949 |
| PICP (nominal 0.80) | 0.880 |
| Winkler | 0.0924 |
| RMSE skill vs persistence | 0.708 |
| Quantile-crossing violations | 0 |

Verified reproducible: a clean checkout with no data or checkpoints regenerates
the identical 8,721 windows and retrains to consistent metrics.

## Open issues

1. **Coverage is miscalibrated in opposite directions across regimes.**
   Overcast over-covers (0.921); dawn_ramp (0.776) and volatile (0.736) are too
   narrow. The aggregate 0.880 only looks calibrated because overcast is 850 of
   1,309 windows. A single global sharpness weight cannot fix this — per-regime
   width calibration is the next real piece of work.

2. **The volatile expert never wins.** Its argmax share on test is 0.000: it
   holds mean gate mass of ~0.25 but is never the top choice for any window.
   The load-balancing penalty is keeping it nominally alive without it earning
   anything. With 36 test windows in that regime, the four-expert premise is
   validated for three experts and unproven for the fourth.

3. **All metrics are on synthetic data.** They describe a physically-grounded
   simulation, not a site. Acquisition sits behind a fixed output schema, so
   swapping in real telemetry touches only the two `download_*.py` files.
