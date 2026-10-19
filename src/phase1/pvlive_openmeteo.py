"""
Phase 1 -- Real-world data source: Sheffield Solar PV_Live + Open-Meteo.

No solar panel or API key is needed:

  * PV_Live (https://www.solar.sheffield.ac.uk/pvlive/) publishes measured
    half-hourly solar generation and installed capacity for every GB
    region, historically and in near-real-time. It stands in for "the
    panel's own meter".
  * Open-Meteo (https://open-meteo.com) serves the forecasts of several
    independent NWP models. Each model is one ensemble member, so the
    ensemble mean is the point weather forecast and the inter-model std
    is sigma_NWP -- the uncertainty signal Phase 4 routes on.

Training uses the historical-forecast archive; `predict.py` uses the live
forecast API with the same models, so features match between the two.

Both fetchers produce the raw schemas the rest of Phase 1 already expects:
    PV : timestamp, pv_power_kw, capacity_kw, solar_elevation_deg
    NWP: timestamp, ensemble_member, ghi, cloud_cover, temp_c, wind_speed

TIME CONVENTION
---------------
Everything is UTC and stamped at the END of its averaging period:
PV_Live's `datetime_gmt` is the end of the half-hour, Open-Meteo's
hourly shortwave_radiation is the mean over the preceding hour. Weather
is interpolated between period centres (not stamps) onto the PV grid so
that a 10:00-10:30 PV reading is paired with the 10:15 weather value
rather than one 15-30 minutes stale.
"""

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from config.data_source import WEATHER_MODELS
from config.paths import CACHE_DIR
from src.phase1.regions import Region
from src.utils.logger import get_logger
from src.utils.solar import solar_elevation_deg

logger = get_logger(__name__)

PVLIVE_URL = "https://api.solar.sheffield.ac.uk/pvlive/api/v4/pes/{region_id}"
OPENMETEO_HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPENMETEO_LIVE_URL = "https://api.open-meteo.com/v1/forecast"

FREQ = "30min"
_HOURLY_VARS = {
    "shortwave_radiation": "ghi",       # W/m^2, mean over preceding hour
    "cloud_cover": "cloud_cover",       # %, converted to [0, 1]
    "temperature_2m": "temp_c",         # deg C
    "wind_speed_10m": "wind_speed",     # m/s (requested with wind_speed_unit=ms)
}
NWP_COLUMNS = ["timestamp", "ensemble_member", "ghi", "cloud_cover", "temp_c", "wind_speed"]


# ---------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------

def _http_get_json(url: str, params: dict, cache: bool, retries: int = 4, timeout: int = 120):
    query = urllib.parse.urlencode(params, safe=",:")
    full_url = f"{url}?{query}"

    cache_path = None
    if cache:
        cache_dir = CACHE_DIR / "http"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / (hashlib.sha1(full_url.encode()).hexdigest() + ".json")
        if cache_path.exists():
            return json.loads(cache_path.read_text())

    req = urllib.request.Request(full_url, headers={"User-Agent": "ATERNET-research/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            # 4xx other than rate limiting is a request bug -- retrying won't help.
            if e.code != 429 and 400 <= e.code < 500:
                raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
            err = f"HTTP {e.code}: {body}"
        except (urllib.error.URLError, TimeoutError) as e:
            err = str(e)
        if attempt == retries - 1:
            raise RuntimeError(f"Request to {url} failed after {retries} attempts: {err}")
        wait = 5 * 2 ** attempt
        logger.warning(f"Request to {url} failed ({err}); retrying in {wait}s.")
        time.sleep(wait)

    if cache_path is not None:
        cache_path.write_text(json.dumps(payload))
    return payload


def _date_chunks(start: pd.Timestamp, end: pd.Timestamp, days: int) -> Iterable[tuple]:
    cur = start
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=days) - pd.Timedelta(seconds=1), end)
        yield cur, nxt
        cur = nxt + pd.Timedelta(seconds=1)


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


# ---------------------------------------------------------------------
# PV_Live
# ---------------------------------------------------------------------

def fetch_pvlive(region: Region, start, end, cache: bool = True) -> pd.DataFrame:
    """
    Measured regional PV generation, one row per half-hour (period-end, UTC):
        timestamp, pv_power_kw, capacity_kw, solar_elevation_deg
    `cache=True` stores each request on disk; only use it for ranges that
    are fully in the past (PV_Live revises the most recent readings).
    """
    start, end = _utc(start), _utc(end)
    url = PVLIVE_URL.format(region_id=region.region_id)
    rows = []
    for c_start, c_end in _date_chunks(start, end, days=180):
        payload = _http_get_json(url, {
            "start": c_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end": c_end.strftime("%Y-%m-%dT%H:%M:%S"),
            "extra_fields": "installedcapacity_mwp",
        }, cache=cache)
        cols = payload["meta"]
        rows.extend(dict(zip(cols, r)) for r in payload["data"])

    if not rows:
        raise RuntimeError(f"PV_Live returned no data for region {region.region_id} "
                           f"between {start} and {end}.")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["datetime_gmt"], utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "pv_power_kw": pd.to_numeric(df["generation_mw"], errors="coerce") * 1000.0,
        "capacity_kw": pd.to_numeric(df["installedcapacity_mwp"], errors="coerce") * 1000.0,
    })
    out["capacity_kw"] = out["capacity_kw"].ffill().bfill()
    # Elevation at the middle of the half-hour the reading averages over.
    out["solar_elevation_deg"] = solar_elevation_deg(
        out["timestamp"] - pd.Timedelta(minutes=15), region.latitude, region.longitude
    )
    return out


# ---------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------

def _payload_to_long(payloads: list, models: List[str]) -> pd.DataFrame:
    """
    One payload per sample point -> hourly long frame averaged over points:
    timestamp (hour-end, UTC), ensemble_member, ghi, cloud_cover, temp_c, wind_speed.
    """
    per_point = []
    for p in payloads:
        hourly = p["hourly"]
        times = pd.to_datetime(hourly["time"]).tz_localize("UTC")
        for member, model in enumerate(models):
            frame = {"timestamp": times, "ensemble_member": member}
            for api_var, col in _HOURLY_VARS.items():
                # Open-Meteo drops the model suffix when only one model is requested.
                key = f"{api_var}_{model}" if len(models) > 1 else api_var
                frame[col] = pd.to_numeric(pd.Series(hourly.get(key, [None] * len(times))),
                                           errors="coerce").to_numpy()
            per_point.append(pd.DataFrame(frame))

    df = pd.concat(per_point, ignore_index=True)
    df = df.groupby(["timestamp", "ensemble_member"], as_index=False).mean()
    df["cloud_cover"] = df["cloud_cover"] / 100.0
    return df


def fetch_weather(region: Region, start=None, end=None, live: bool = False,
                  past_days: int = 3, forecast_days: int = 2,
                  models: Optional[List[str]] = None, cache: bool = True) -> pd.DataFrame:
    """
    Hourly multi-model weather (long format, averaged over the region's points).

    live=False: historical-forecast archive for [start, end].
    live=True : current forecast run, `past_days` back to `forecast_days` ahead.
    """
    models = models or WEATHER_MODELS
    base = {
        "latitude": ",".join(f"{p[0]:.4f}" for p in region.points),
        "longitude": ",".join(f"{p[1]:.4f}" for p in region.points),
        "hourly": ",".join(_HOURLY_VARS),
        "models": ",".join(models),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }

    payload_batches = []
    if live:
        params = dict(base, past_days=past_days, forecast_days=forecast_days)
        payload_batches.append(_http_get_json(OPENMETEO_LIVE_URL, params, cache=False))
    else:
        start, end = _utc(start), _utc(end)
        for c_start, c_end in _date_chunks(start.normalize(), end.normalize(), days=183):
            params = dict(base, start_date=c_start.strftime("%Y-%m-%d"),
                          end_date=c_end.strftime("%Y-%m-%d"))
            payload_batches.append(_http_get_json(OPENMETEO_HISTORICAL_URL, params, cache=cache))

    frames = []
    for payload in payload_batches:
        payloads = payload if isinstance(payload, list) else [payload]
        frames.append(_payload_to_long(payloads, models))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["timestamp", "ensemble_member"]).sort_values(
        ["timestamp", "ensemble_member"]).reset_index(drop=True)

    n_missing = df[["ghi", "cloud_cover", "temp_c", "wind_speed"]].isna().any(axis=1)
    if n_missing.any():
        by_member = df.loc[n_missing, "ensemble_member"].value_counts().to_dict()
        logger.info(f"Open-Meteo: {int(n_missing.sum())} member-hours with missing values "
                    f"(by member: { {models[k]: v for k, v in by_member.items()} }).")
    return df


def weather_to_pv_grid(weather_hourly: pd.DataFrame, pv_timestamps: pd.DatetimeIndex,
                       max_gap_hours: int = 6) -> pd.DataFrame:
    """
    Interpolate each member from hourly period-centres onto the half-hour
    PV period-centres, then re-stamp at PV period-ends. Gaps longer than
    `max_gap_hours` stay missing and the member-row is dropped, so the
    ensemble statistics for that step use only members that have data.
    """
    pv_timestamps = pd.DatetimeIndex(pv_timestamps)
    target_centres = pv_timestamps - pd.Timedelta(minutes=15)
    value_cols = ["ghi", "cloud_cover", "temp_c", "wind_speed"]

    out = []
    for member, g in weather_hourly.groupby("ensemble_member"):
        g = g.set_index(g["timestamp"] - pd.Timedelta(minutes=30))[value_cols].sort_index()
        g = g[~g.index.duplicated()]
        g = g.interpolate(method="time", limit=max_gap_hours, limit_area="inside")
        union = g.index.union(target_centres)
        g = g.reindex(union).interpolate(method="time", limit=2, limit_area="inside")
        g = g.reindex(target_centres)
        g.index = pv_timestamps
        g = g.dropna()
        g["ensemble_member"] = member
        out.append(g.reset_index(names="timestamp"))

    df = pd.concat(out, ignore_index=True)
    df["ghi"] = df["ghi"].clip(lower=0.0)
    df["cloud_cover"] = df["cloud_cover"].clip(0.0, 1.0)
    df["wind_speed"] = df["wind_speed"].clip(lower=0.0)
    return df[NWP_COLUMNS].sort_values(["timestamp", "ensemble_member"]).reset_index(drop=True)


def half_hour_grid(start, end) -> pd.DatetimeIndex:
    return pd.date_range(_utc(start).ceil(FREQ), _utc(end).floor(FREQ), freq=FREQ)
