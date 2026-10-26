# Demo guide

## Before the demo (5 minutes)

```powershell
cd C:\Users\piyus\desktop\ATERNET
python -m pytest -q                 # expect: 74 passed
python build_results_summary.py     # refresh docs/results_summary.json from raw outputs
streamlit run app.py                # opens http://localhost:8501
```

Do **not** set `ATERNET_WORKSPACE` / `ATERNET_REGION_ID` in the window you launch the
dashboard from: the Dashboard and Uncertainty pages read the default (Yorkshire) workspace.
Close other heavy programs first (earlier long runs were stopped for low memory).
The Live Forecast page needs internet; skip it if the network is unreliable.

## Suggested order (about 10 minutes)

1. **Dashboard** – one trained model on the Yorkshire test split: nRMSE 3.69 %, R² 0.963.
   Point out *PICP (daytime, calibrated)* versus *PICP (all points)*: night points are
   trivially covered, so daytime coverage is the honest figure.
2. **Results & Evidence** (the key page) – multi-seed means ± std against LightGBM,
   PatchTST and LSTM. Switch between **Yorkshire** and **South West England**.
   Read the *Verdict* column out loud; it is computed from the data.
3. **Uncertainty** – the interval widens when the model is less sure; per-regime
   daytime coverage before and after calibration.
4. **Guarantee Demo** – move the σ slider: quantiles never cross, width grows with σ,
   routing softens (structural properties, checked by tests).
5. **India Demo (Simulated)** – a red banner says the plant output is simulated. Show
   the forecast viewer. Say plainly it is a demonstration for the VIT Vellore location.

## What you can claim

- On South West England, over 3 seeds, ATERNET has lower nRMSE (2.63 vs 2.69 LightGBM,
  2.92 PatchTST) and lower Winkler score than both baselines, with a seed spread far smaller
  than the gap. Hyperparameters were chosen on Yorkshire, not tuned for this site.
- On Yorkshire it **ties** LightGBM on nRMSE (3.70 vs 3.72, within seed noise), is ahead of
  PatchTST by ~11 %, and has the better Winkler score (raw intervals).
- It is stable across seeds; the LSTM baseline is not (3 of 6 seeds failed).
- After calibration, daytime interval coverage is 0.794 (Yorkshire) and 0.798 (South West)
  against a nominal 0.80.
- Zero quantile-crossing violations; three structural guarantees are test-enforced.

## What you must not claim

- "More accurate than all existing methods." It ties LightGBM on Yorkshire, and LightGBM
  pulls ahead when both train on twice as many windows.
- That adaptive routing or regime-specialised experts improve accuracy. The ablations say
  they do not; only the dense expert loss showed a measurable contribution.
- That the Vellore numbers are real-world accuracy. The plant output is simulated.
- That calibrated-ATERNET beats raw baselines on intervals as a like-for-like result; the
  baselines were not calibrated.

## Questions reviewers are likely to ask

- **"Why not simply use LightGBM?"** On Yorkshire accuracy is equal; ATERNET gives calibrated
  P10–P90 intervals with structural guarantees, was ahead on a second region, and is stable
  across seeds. LightGBM needs a separate model per quantile and horizon step.
- **"Is the improvement real?"** Multi-seed std is shown on every row; a gap counts only if it
  exceeds about two seed standard deviations.
- **"Why is the Vellore result so good?"** The plant output is a smooth function of
  reanalysis weather, so it measures how well the model learns that simulator from
  forecasts. It is a demonstration, not a validation.
- **"Is coverage really 80 %?"** In daylight, after calibration, yes on two UK sites (0.794,
  0.798). On the simulated Vellore site it is only 0.745, i.e. somewhat too narrow.
- **"Real data for India?"** None open and measured exists for this area; with college solar
  data the same pipeline would use it (see `src/phase1/simulated_pv.py` for the data schema).
