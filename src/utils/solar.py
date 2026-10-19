"""
Solar geometry helpers shared by the data sources and feature engineering.

`solar_elevation_deg` uses the NOAA low-precision formulation (fractional
year -> declination + equation of time -> true solar time). It accounts
for longitude, so it is correct anywhere, not only near the Greenwich
meridian where using the UTC clock as solar time happens to work.
"""

import numpy as np
import pandas as pd


def solar_elevation_deg(timestamps, latitude: float, longitude: float = 0.0) -> np.ndarray:
    ts = pd.DatetimeIndex(timestamps)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC")

    doy = ts.dayofyear.values.astype(float)
    hour = (ts.hour + ts.minute / 60.0 + ts.second / 3600.0).values.astype(float)

    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)
    eot_min = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )

    true_solar_min = hour * 60.0 + eot_min + 4.0 * longitude
    hour_angle = np.deg2rad(true_solar_min / 4.0 - 180.0)
    lat = np.deg2rad(latitude)

    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.rad2deg(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))


def clear_sky_ghi(elevation_deg: np.ndarray, solar_constant: float = 1361.0) -> np.ndarray:
    elev_clipped = np.clip(elevation_deg, 0.0, 90.0)
    air_mass = 1.0 / np.clip(np.sin(np.deg2rad(elev_clipped + 0.001)), 1e-3, None)
    transmittance = 0.75 ** (air_mass ** 0.678)
    ghi = solar_constant * np.sin(np.deg2rad(elev_clipped)) * transmittance
    ghi = np.where(elevation_deg > 0, ghi, 0.0)
    return np.clip(ghi, 0.0, None)
