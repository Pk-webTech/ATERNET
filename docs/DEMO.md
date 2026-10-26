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

The left menu uses plain names. Every page starts with a blue box saying what it shows, and the
"What do these words mean?" box defines every term in simple language.

1. **Overview** – five headline numbers with hover help ("Typical error", "Follows reality",
   "Range hit-rate (day)", ...). Scroll to *How does ATERNET compare with other models?*: a
   colour-coded scoreboard (green = ATERNET better, yellow = about the same, red = worse).
2. **Is ATERNET better?** (the key page) – pick a region (Yorkshire / South West England). Read the
   *short answer* box, then the scoreboard and the two charts. Open *Show the detailed numbers* only if asked.
   Scroll down for *Does the forecast range hold what it promises?* and *Which parts of ATERNET actually help?*
3. **How sure is the model?** – when the model is less sure, it draws a wider range.
4. **Built-in safety checks** – move the slider; three checks hold on a real forecast.
5. **India demo (simulated)** – a red banner says the plant output is simulated. Use the slider to browse forecasts.

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
