"""
Which real-world data the Phase 1 pipeline acquires.

Two sources are supported; both write the same raw schema, so nothing
downstream of `download_pv.py` / `download_nwp.py` cares which one ran.

  "pvlive_openmeteo"  (default)
      PV:      Sheffield Solar PV_Live -- measured solar generation for a
               GB region (DNO licence area), half-hourly, with installed
               capacity. Free, no API key, updated live every 30 min.
      Weather: Open-Meteo multi-model forecasts (GFS, ICON, UKMO,
               Meteo-France, GEM). Each model is treated as one ensemble
               member, so sigma_NWP is the real inter-model spread.
               Historical-forecast archive for training, live forecast
               API for inference -- the same models in both, so the
               features the model sees at inference match training.

  "uk_pv_era5"
      The original source: one rooftop system from openclimatefix/uk_pv
      (HuggingFace) + ERA5 EDA from Copernicus CDS (needs ~/.cdsapirc).
      Weather-history only; there is no live counterpart, so it cannot
      drive `predict.py live`.

Changing anything here requires deleting data/raw/* and re-running
`python run_pipeline.py` (raw files are cached and not re-downloaded).
"""

DATA_SOURCE = "pvlive_openmeteo"

# PV_Live region (PES / DNO licence area id). See src/phase1/regions.py.
# 23 = NPG Yorkshire -- the region containing the original uk_pv site.
import os

PVLIVE_REGION_ID = int(os.environ.get("ATERNET_REGION_ID", 23))  # env override for other regions

# Training period. The Open-Meteo historical-forecast archive has full
# multi-model coverage from 2023 onward. END_DATE leaves the most recent
# weeks unseen so `predict.py backtest` can run on genuinely new data.
START_DATE = "2023-01-01"
END_DATE = "2026-09-30"

# Open-Meteo model ids used as ensemble members. All five have complete
# GB coverage in the historical-forecast archive from 2023 and are also
# served by the live forecast API. (ecmwf_ifs025 is left out: its archive
# has gaps before 2024, which would make sigma_NWP jump when it appears.)
WEATHER_MODELS = [
    "gfs_seamless",
    "icon_seamless",
    "ukmo_seamless",
    "meteofrance_seamless",
    "gem_seamless",
]
