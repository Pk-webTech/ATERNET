# ATERNET dataset (Yorkshire, Sheffield Solar PV_Live region 23)

- Period: 2023-01-01 to 2026-10-01 (UTC), 30-minute steps, 65,713 rows.
- Sources: measured regional PV power (Sheffield Solar PV_Live) and weather forecasts from five models (Open-Meteo: GFS, ICON, UKMO, Météo-France, GEM), averaged into `nwp_*_mean` with spread in `*_std`.
- `aternet_yorkshire_sample.csv`: key columns only (easy to open in Excel).
- `aternet_yorkshire_full_features.csv`: all 43 columns used for training (clear-sky index, time encodings, ramp rates, regime labels, uncertainty features).

Main columns: `pv_power_kw` (measured power), `pv_norm` (power / capacity, the model target), `nwp_ghi_mean` (forecast sunlight), `nwp_cloud_cover_mean`, `ensemble_spread_norm` (how much the five forecasts disagree), `regime` (weather type).

Regenerated from `data/interim/features.parquet`.

## Sample-level files (what the model actually trains on)
- `aternet_sample_counts.csv`: rows, columns, windows and the 70/15/15 split with the calculation (65,713 rows -> 16,413 samples -> 11,489 / 2,462 / 2,462).
- `aternet_samples_index.csv`: one row per sample (16,413): split and the time span of its 24 h history and 8 h forecast.
- `aternet_sample_windows_example.csv`: three complete example samples (one train, one validation, one test). Each has 48 history steps (input) and 16 forecast steps (answer), with measured power and the weather-forecast columns.
