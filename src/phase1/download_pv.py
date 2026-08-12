"""
Phase 1 -- PV data acquisition.

No live PV telemetry / API credentials are available in this environment,
so this module generates a physically-grounded synthetic PV generation
series (clear-sky model + stochastic cloud attenuation + sensor noise).

The function signature and output schema are written so that swapping
this out for a real API / SCADA export later requires touching only
this file -- everything downstream reads data/raw/pv/pv_raw.parquet.
"""

import numpy as np
import pandas as pd

from config.paths import RAW_PV_DIR
from src.utils.site_metadata import DEFAULT_SITE
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.io import save_parquet

logger = get_logger(__name__)

# Standard-test-condition irradiance (W/m^2) -- the reference at which a
# PV module is rated at its nameplate power.
G_STC_WM2 = 1000.0


def _solar_elevation_deg(timestamps: pd.DatetimeIndex, latitude: float) -> np.ndarray:
    """
    Simplified solar elevation angle (degrees) -- enough fidelity to drive a
    synthetic clear-sky irradiance curve. Not a substitute for a real solar
    position library (e.g. pvlib) if this project moves to real deployment.
    """
    doy = timestamps.dayofyear.values.astype(float)
    hour = (timestamps.hour + timestamps.minute / 60.0).values.astype(float)

    # Solar declination (degrees)
    decl = 23.45 * np.sin(np.deg2rad(360.0 / 365.0 * (doy - 81.0)))

    # Hour angle (degrees); solar noon assumed at local hour 12
    hour_angle = 15.0 * (hour - 12.0)

    lat_rad = np.deg2rad(latitude)
    decl_rad = np.deg2rad(decl)
    ha_rad = np.deg2rad(hour_angle)

    sin_elev = (
        np.sin(lat_rad) * np.sin(decl_rad)
        + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad)
    )
    elevation = np.rad2deg(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))
    return elevation


def _clear_sky_ghi(elevation_deg: np.ndarray, solar_constant: float = 1361.0) -> np.ndarray:
    """Simplified clear-sky GHI (W/m^2) as a function of solar elevation."""
    elev_clipped = np.clip(elevation_deg, 0.0, 90.0)
    air_mass = 1.0 / np.clip(np.sin(np.deg2rad(elev_clipped + 0.001)), 1e-3, None)
    transmittance = 0.75 ** (air_mass ** 0.678)
    ghi = solar_constant * np.sin(np.deg2rad(elev_clipped)) * transmittance
    ghi = np.where(elevation_deg > 0, ghi, 0.0)
    return np.clip(ghi, 0.0, None)


def generate_synthetic_pv(
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    freq_minutes: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a full-year synthetic PV generation series with:
      - clear-sky baseline driven by solar geometry
      - correlated cloud-attenuation episodes (regime-inducing)
      - inverter/sensor noise
      - occasional missing-data gaps (realistic telemetry behaviour)
    """
    set_seed(seed)

    timestamps = pd.date_range(start=start_date, end=end_date, freq=f"{freq_minutes}min")
    n = len(timestamps)

    elevation = _solar_elevation_deg(timestamps, DEFAULT_SITE.latitude)
    clear_sky_ghi = _clear_sky_ghi(elevation)

    # Stochastic cloud attenuation process (AR(1) in [0,1], 1 = no cloud).
    # Vectorized: an explicit Python loop over ~35k steps is needlessly slow.
    shocks = np.random.normal(0, 0.03, n)
    cloud_transmittance = np.ones(n)
    rho = 0.985
    for t in range(1, n):
        cloud_transmittance[t] = np.clip(
            rho * cloud_transmittance[t - 1] + (1 - rho) * 1.0 + shocks[t], 0.05, 1.0
        )
    # Inject a handful of multi-hour deep-cloud episodes for regime diversity
    n_episodes = max(1, n // (24 * 60 // freq_minutes) // 5)
    for _ in range(n_episodes):
        center = np.random.randint(0, n)
        width = np.random.randint(8, 48)  # 2h - 12h depending on freq
        lo, hi = max(0, center - width), min(n, center + width)
        cloud_transmittance[lo:hi] *= np.random.uniform(0.2, 0.6)

    ghi_actual = clear_sky_ghi * cloud_transmittance

    # PV power from irradiance: PVWatts-style DC model.
    #   P = P_nameplate * (GHI / G_STC) * temp_derate * system_derate
    # G_STC = 1000 W/m^2 is the standard-test-condition irradiance at which
    # a panel produces its nameplate rating, so clear-sky noon maps to ~85%
    # of nameplate rather than to a few percent of it. The earlier
    # `ghi * 0.18 * capacity/1000` form conflated module efficiency (an
    # area-to-power conversion) with the nameplate rating itself, which
    # capped pv_norm at ~0.17 and left the whole [0,1] target range -- and
    # every downstream constant defined against it (the overcast expert's
    # 0.40 cap, the regime volatility thresholds) -- effectively dead.
    ambient_temp = 25 + 8 * np.sin(np.deg2rad(elevation)) + np.random.normal(0, 1.0, n)
    cell_temp = ambient_temp + ghi_actual * 0.025          # NOCT-style cell heating
    temp_derate = 1 - 0.004 * np.clip(cell_temp - 25, 0, None)
    system_derate = 0.85                                    # soiling, wiring, inverter

    pv_power_kw = (
        DEFAULT_SITE.capacity_kw * (ghi_actual / G_STC_WM2) * temp_derate * system_derate
    )
    pv_power_kw += np.random.normal(0, DEFAULT_SITE.capacity_kw * 0.002, n)  # sensor noise
    pv_power_kw = np.clip(pv_power_kw, 0, DEFAULT_SITE.capacity_kw)
    pv_power_kw = np.where(elevation > 0, pv_power_kw, 0.0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "pv_power_kw": pv_power_kw,
        "ghi_measured": ghi_actual,
        "ambient_temp_c": ambient_temp,
        "solar_elevation_deg": elevation,
    })

    # Simulate realistic missing-data gaps (~0.3% of rows, in short bursts)
    n_gaps = max(1, n // 3000)
    for _ in range(n_gaps):
        start = np.random.randint(0, n - 10)
        length = np.random.randint(1, 6)
        df.loc[start:start + length, ["pv_power_kw"]] = np.nan

    logger.info(f"Synthesized PV series: {n} rows from {start_date} to {end_date} "
                f"({df['pv_power_kw'].isna().sum()} missing points injected).")
    return df


def main():
    df = generate_synthetic_pv()
    out_path = RAW_PV_DIR / "pv_raw.parquet"
    save_parquet(df, out_path)
    logger.info(f"Saved raw PV data -> {out_path} ({len(df)} rows).")
    return out_path


if __name__ == "__main__":
    main()
