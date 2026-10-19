"""
Offline tests for the PV_Live / Open-Meteo source and the live-inference
window builder. No network: API payloads are constructed in-test.
"""

import numpy as np
import pandas as pd
import pytest

from src.phase1.create_windows import build_inference_window
from src.phase1.pv_processor import clean_pv_series
from src.phase1.pvlive_openmeteo import _payload_to_long, weather_to_pv_grid
from src.phase1.regions import nearest_region
from src.utils.feature_groups import (
    SEQUENCE_FEATURES, SIGMA_FEATURES, NWP_FEATURES, ROUTING_METADATA_FEATURES,
)
from src.utils.solar import solar_elevation_deg


def test_solar_elevation_uses_longitude():
    # Equinox: solar noon elevation ~ 90 - latitude, reached at
    # ~12:00 UTC on the Greenwich meridian and ~6:40 UTC at 80 E.
    lat = 13.0
    day = pd.date_range("2024-03-20 00:00", periods=24 * 60, freq="1min", tz="UTC")
    greenwich = solar_elevation_deg(day, lat, 0.0)
    chennai = solar_elevation_deg(day, lat, 80.27)

    assert abs(greenwich.max() - (90 - lat)) < 1.0
    assert abs(day[greenwich.argmax()].hour + day[greenwich.argmax()].minute / 60 - 12.1) < 0.3
    chennai_noon = day[chennai.argmax()].hour + day[chennai.argmax()].minute / 60
    assert abs(chennai_noon - (12.1 - 80.27 / 15)) < 0.3


def _fake_payload(times, models, ghi_by_model):
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    for m in models:
        hourly[f"shortwave_radiation_{m}"] = list(ghi_by_model[m])
        hourly[f"cloud_cover_{m}"] = [50.0] * len(times)
        hourly[f"temperature_2m_{m}"] = [10.0] * len(times)
        hourly[f"wind_speed_10m_{m}"] = [3.0] * len(times)
    return {"hourly": hourly}


def test_openmeteo_payload_to_pv_grid():
    models = ["a", "b"]
    times = pd.date_range("2024-06-01", periods=24, freq="h")
    ghi = {"a": np.arange(24) * 10.0, "b": np.arange(24) * 20.0}
    # Two sample points with identical data -> point-average is unchanged.
    long = _payload_to_long([_fake_payload(times, models, ghi)] * 2, models)

    assert set(long["ensemble_member"]) == {0, 1}
    assert long["cloud_cover"].max() == pytest.approx(0.5)  # % -> fraction

    pv_ts = pd.date_range("2024-06-01 02:00", "2024-06-01 20:00", freq="30min", tz="UTC")
    grid = weather_to_pv_grid(long, pv_ts)
    m0 = grid[grid["ensemble_member"] == 0].set_index("timestamp")["ghi"]
    # Hourly value stamped T is the mean over (T-1h, T], centred at T-30min:
    # stamp 10:00 (100) is centred 09:30, stamp 11:00 (110) at 10:30.
    # The half-hour ending 10:00 is centred 09:45 -> 100 + 10 * 15/60.
    assert m0.loc[pd.Timestamp("2024-06-01 10:00", tz="UTC")] == pytest.approx(102.5)
    assert len(m0) == len(pv_ts)


def test_clean_pv_uses_per_row_capacity():
    ts = pd.date_range("2024-06-01", periods=10, freq="30min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "pv_power_kw": [50.0] * 10,
        "capacity_kw": [100.0] * 5 + [200.0] * 5,
        "solar_elevation_deg": 30.0,
    })
    out = clean_pv_series(df)
    assert out["pv_norm"].tolist() == [0.5] * 5 + [0.25] * 5


def test_build_inference_window_layout():
    hist_ts = pd.date_range("2024-06-01", periods=60, freq="30min", tz="UTC")
    hist = pd.DataFrame({"timestamp": hist_ts})
    for c in set(SEQUENCE_FEATURES + SIGMA_FEATURES + NWP_FEATURES + ROUTING_METADATA_FEATURES):
        hist[c] = np.arange(60, dtype=float)
    fut = pd.DataFrame({"timestamp": hist_ts[-1] + pd.Timedelta(minutes=30) * np.arange(1, 17)})
    for c in NWP_FEATURES:
        fut[c] = -1.0

    w = build_inference_window(hist, fut, window_size=48, horizon=16, freq_minutes=30)
    assert w["X_seq"].shape == (1, 48, len(SEQUENCE_FEATURES))
    assert w["X_nwp"].shape == (1, 64, len(NWP_FEATURES))
    assert (w["X_nwp"][0, 48:] == -1.0).all()
    assert w["X_seq"][0, -1, 0] == 59.0          # lookback ends at the last history row
    assert w["issue_time"] == hist_ts[-1]

    with pytest.raises(ValueError):
        build_inference_window(hist, fut.iloc[1:], window_size=48, horizon=15, freq_minutes=30)


def test_nearest_region():
    assert nearest_region(53.38, -1.47).region_id == 23   # Sheffield -> Yorkshire
    assert nearest_region(51.51, -0.13).region_id == 12   # London
