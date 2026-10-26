"""
SIMULATED PV generation for sites with no measured data (e.g. VIT Vellore).

!! Nothing produced from this module is a measurement. Any accuracy
!! number from a model trained or scored on it describes how well the model
!! learns THIS simulator from forecast weather, not real-world performance.

Design -- why the "truth" is not built from the model's own inputs:
  * model inputs : the Open-Meteo multi-model *forecast* archive (the usual
                   NWP ensemble, via download_nwp.py)
  * simulated PV : a physical PV model driven by ERA5 *reanalysis* weather
                   (archive-api.open-meteo.com), a different data source that
                   stands in for "what the weather actually did"
So the forecast-vs-actual weather error the model has to cope with is real,
instead of the task collapsing to inverting a formula applied to its inputs.

The PV model is deliberately simple and documented:
  T_cell   = T_amb + 0.03125 * GHI                      (NOCT ~ 45 C)
  P_dc     = P_peak * GHI/1000 * (1 - 0.004 (T_cell-25)) * system_losses * soiling
  P_ac     = min(P_dc, P_peak / dc_ac_ratio)            (inverter clipping)
  plus slow soiling build-up with periodic cleaning and ~1 % multiplicative noise.
GHI is used directly as plane-of-array irradiance (no tilt transposition).
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.phase1.pvlive_openmeteo import (
    fetch_range_adaptive, _payload_to_long, _utc, half_hour_grid, weather_to_pv_grid,
)
from src.phase1.regions import Region
from src.utils.logger import get_logger
from src.utils.solar import solar_elevation_deg

logger = get_logger(__name__)

OPENMETEO_ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
_ERA5_VARS = "shortwave_radiation,cloud_cover,temperature_2m,wind_speed_10m"

SYSTEM_LOSSES = 0.88      # wiring, mismatch, inverter efficiency, ...
TEMP_COEFF = -0.004       # per deg C above 25 C (crystalline silicon)
NOCT_COEFF = 0.03125      # (NOCT - 20) / 800
DC_AC_RATIO = 1.25
NOISE_STD = 0.01


def soiling_factor(n: int, steps_per_day: int = 48, seed: int = 0) -> np.ndarray:
    """Slow loss of ~0.12 %/day since the last cleaning (floor 0.90); cleaning every 20-45 days."""
    rng = np.random.default_rng(seed)
    days = np.empty(n, dtype=np.float64)
    since, next_clean = 0.0, rng.integers(20, 46)
    for i in range(n):
        days[i] = since
        since += 1.0 / steps_per_day
        if since >= next_clean:
            since, next_clean = 0.0, rng.integers(20, 46)
    return np.clip(1.0 - 0.0012 * days, 0.90, 1.0)


def pv_power_kw(ghi: np.ndarray, temp_c: np.ndarray, peak_kw: float,
                soiling: Optional[np.ndarray] = None) -> np.ndarray:
    """Physical PV model (see module docstring). Inputs in W/m^2 and deg C."""
    ghi = np.clip(np.asarray(ghi, dtype=np.float64), 0.0, None)
    t_cell = np.asarray(temp_c, dtype=np.float64) + NOCT_COEFF * ghi
    derate = 1.0 + TEMP_COEFF * (t_cell - 25.0)
    soil = 1.0 if soiling is None else soiling
    p_dc = peak_kw * (ghi / 1000.0) * derate * SYSTEM_LOSSES * soil
    return np.clip(p_dc, 0.0, peak_kw / DC_AC_RATIO)


def fetch_reanalysis(region: Region, start, end, cache: bool = True) -> pd.DataFrame:
    """Hourly ERA5 weather averaged over the region's points (single member)."""
    start, end = _utc(start), _utc(end)
    base = {
        "latitude": ",".join(f"{p[0]:.4f}" for p in region.points),
        "longitude": ",".join(f"{p[1]:.4f}" for p in region.points),
        "hourly": _ERA5_VARS, "wind_speed_unit": "ms", "timezone": "GMT",
    }
    frames = []
    for payload in fetch_range_adaptive(OPENMETEO_ERA5_URL, base, start, end, cache):
        payloads = payload if isinstance(payload, list) else [payload]
        frames.append(_payload_to_long(payloads, ["era5"]))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["timestamp", "ensemble_member"])
    return df.sort_values("timestamp").reset_index(drop=True)


def simulate_pv(region: Region, start, end, seed: int = 0, cache: bool = True) -> pd.DataFrame:
    """
    Same schema as fetch_pvlive: timestamp (period-end, UTC), pv_power_kw,
    capacity_kw, solar_elevation_deg -- generation is SIMULATED.
    """
    if not region.simulated or region.capacity_kw <= 0:
        raise ValueError(f"Region {region.region_id} is not a simulated site.")

    grid = half_hour_grid(start, end)
    hourly = fetch_reanalysis(region, _utc(start) - pd.Timedelta(days=1),
                              _utc(end) + pd.Timedelta(days=1), cache=cache)
    w = weather_to_pv_grid(hourly, grid).set_index("timestamp").reindex(grid)

    # Reanalysis gaps (if any) are filled by time-interpolation, never invented.
    ghi = w["ghi"].interpolate(limit=12).fillna(0.0).to_numpy()
    temp = w["temp_c"].interpolate(limit=12).bfill().ffill().to_numpy()

    rng = np.random.default_rng(seed)
    power = pv_power_kw(ghi, temp, region.capacity_kw, soiling_factor(len(grid), seed=seed))
    power = power * rng.normal(1.0, NOISE_STD, size=len(power))
    power = np.clip(power, 0.0, region.capacity_kw)

    out = pd.DataFrame({
        "timestamp": grid,
        "pv_power_kw": power,
        "capacity_kw": float(region.capacity_kw),
    })
    out["solar_elevation_deg"] = solar_elevation_deg(
        out["timestamp"] - pd.Timedelta(minutes=15), region.latitude, region.longitude)
    # Physically no output with the sun below the horizon.
    out.loc[out["solar_elevation_deg"] <= 0, "pv_power_kw"] = 0.0
    logger.warning(f"SIMULATED PV for {region.name}: {len(out)} half-hours from ERA5 "
                   f"reanalysis. Not measured data.")
    return out
