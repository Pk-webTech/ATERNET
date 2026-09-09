"""
Phase 1 -- PV data acquisition (REAL DATA: openclimatefix/uk_pv).

Replaces the synthetic generator with the actual `openclimatefix/uk_pv`
HuggingFace dataset: 396 UK PV sites, half-hourly generation readings +
static site metadata (capacity, lat/lon, orientation).

Output schema is deliberately kept close to the original synthetic
version so nothing downstream except this file (and, for the two
fields uk_pv doesn't provide, `feature_engineering.py`) has to change:

    timestamp, pv_power_kw, solar_elevation_deg

WHAT'S MISSING FROM uk_pv, AND WHERE IT NOW COMES FROM
--------------------------------------------------------
uk_pv is PV generation + static metadata ONLY -- no irradiance, no
ambient temperature (this is called out explicitly in project memory).
The synthetic version populated `ghi_measured` / `ambient_temp_c`
because its PV model needed to invent them; real data has no such
column to carry forward. `feature_engineering.py` has been updated to
source the "observed weather" side of `clear_sky_index` / `ghi` /
`temp_c` from the ERA5 merge (ssrd -> ghi, 2t -> temp_c) instead of
from the PV frame. That is *not* a workaround -- it's the same join
the project was always going to need, just made structurally required
now rather than optional.

`solar_elevation_deg` IS computed here, from pure geometry (site
lat/lon + timestamp), so `clear_sky_index` in feature_engineering still
works unchanged.

SITE SELECTION
---------------
uk_pv covers 396 sites; this project's windowing/split logic is built
around a single continuous series. `SITE_ID` below pins one site
explicitly. If unset, it automatically picks the site with the fewest
missing readings in the first month to save memory, then extracts only
that site from the remaining months.
"""

from typing import Optional
import gc

import numpy as np
import pandas as pd
from huggingface_hub.utils import EntryNotFoundError

from config.paths import RAW_PV_DIR, RAW_METADATA_DIR
from src.utils.logger import get_logger
from src.utils.io import save_parquet, save_json

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Pure-geometry helpers.
# ---------------------------------------------------------------------

def _solar_elevation_deg(timestamps: pd.DatetimeIndex, latitude: float) -> np.ndarray:
    doy = timestamps.dayofyear.values.astype(float)
    hour = (timestamps.hour + timestamps.minute / 60.0).values.astype(float)

    decl = 23.45 * np.sin(np.deg2rad(360.0 / 365.0 * (doy - 81.0)))
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
    elev_clipped = np.clip(elevation_deg, 0.0, 90.0)
    air_mass = 1.0 / np.clip(np.sin(np.deg2rad(elev_clipped + 0.001)), 1e-3, None)
    transmittance = 0.75 ** (air_mass ** 0.678)
    ghi = solar_constant * np.sin(np.deg2rad(elev_clipped)) * transmittance
    ghi = np.where(elevation_deg > 0, ghi, 0.0)
    return np.clip(ghi, 0.0, None)

HF_REPO_ID = "openclimatefix/uk_pv"
HF_REPO_TYPE = "dataset"

# Pin a specific uk_pv system id (a.k.a. ss_id). None -> auto-select.
SITE_ID: Optional[int] = None

START_DATE = "2019-01-01"
END_DATE = "2019-12-31"


def _find_metadata_file():
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(HF_REPO_ID, repo_type=HF_REPO_TYPE)
    metadata_files = [f for f in files if "metadata" in f.lower() and f.endswith((".csv", ".parquet"))]
    if not metadata_files:
        raise FileNotFoundError(f"Could not locate metadata file in {HF_REPO_ID}.")
    return metadata_files[0]


def _download(path_in_repo: str) -> str:
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, filename=path_in_repo)
    logger.info(f"Fetched {HF_REPO_ID}/{path_in_repo} -> {local_path}")
    return local_path


def _load_metadata(metadata_path: str) -> pd.DataFrame:
    if metadata_path.endswith(".csv"):
        meta = pd.read_csv(metadata_path)
    else:
        meta = pd.read_parquet(metadata_path)

    rename_map = {}
    for c in meta.columns:
        cl = c.lower()
        if cl in ("ss_id", "system_id", "id"):
            rename_map[c] = "site_id"
        elif "lat" in cl:
            rename_map[c] = "latitude"
        elif "lon" in cl or "lng" in cl:
            rename_map[c] = "longitude"
        elif "kwp" in cl or "capacity" in cl:
            rename_map[c] = "capacity_kw"
    meta = meta.rename(columns=rename_map)

    required = {"site_id", "latitude", "longitude", "capacity_kw"}
    missing = required - set(meta.columns)
    if missing:
        raise KeyError(f"uk_pv metadata missing {missing}.")
    return meta.dropna(subset=["latitude", "longitude", "capacity_kw"])


def _load_generation(gen_path: str) -> pd.DataFrame:
    df = pd.read_parquet(gen_path)
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("ss_id", "system_id", "id"):
            rename_map[c] = "site_id"
        elif cl in ("timestamp", "datetime", "datetime_gmt"):
            rename_map[c] = "timestamp"
        elif "power" in cl or "generation" in cl:
            rename_map[c] = "pv_power_kw"
    df = df.rename(columns=rename_map)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def build_real_pv_series() -> pd.DataFrame:
    metadata_path_in_repo = _find_metadata_file()
    metadata_local = _download(metadata_path_in_repo)
    meta_df = _load_metadata(metadata_local)

    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    months_in_range = pd.date_range(start=start_dt.replace(day=1), end=end_dt, freq='MS')

    target_site_id = SITE_ID
    if target_site_id is None:
        logger.info("SITE_ID is None. Finding the best site using the first month of data to save memory...")
        first_month_dt = months_in_range[0]
        first_file = f"30_minutely/year={first_month_dt.year}/month={first_month_dt.month:02d}/data.parquet"
        local_path = _download(first_file)
        first_df = _load_generation(local_path)
        
        # Determine completeness just on the first month
        completeness = first_df.groupby("site_id")["pv_power_kw"].apply(lambda s: s.notna().mean())
        completeness = completeness[completeness.index.isin(meta_df["site_id"])]
        target_site_id = int(completeness.idxmax())
        logger.info(f"Auto-selected site_id={target_site_id} (first month completeness {completeness[target_site_id]:.1%}).")
        
        # Free memory immediately
        del first_df
        gc.collect()

    downloaded_dfs = []
    for dt in months_in_range:
        year = dt.year
        month = dt.month
        file_path = f"30_minutely/year={year}/month={month:02d}/data.parquet"
        
        try:
            local_path = _download(file_path)
            df_part = _load_generation(local_path)
            
            # MEMORY OPTIMIZATION: Filter for the single target site immediately
            df_part = df_part[df_part["site_id"] == target_site_id]
            # Filter exact bounds
            df_part = df_part[(df_part["timestamp"] >= START_DATE) & (df_part["timestamp"] <= END_DATE)]
            
            downloaded_dfs.append(df_part)
        except EntryNotFoundError:
            logger.warning(f"Partition not found on HuggingFace: {file_path}. Skipping.")
        except Exception as e:
            logger.warning(f"Error fetching {file_path}: {e}")

    if not downloaded_dfs:
        raise RuntimeError(f"No uk_pv generation partitions found between {START_DATE} and {END_DATE}.")

    df = pd.concat(downloaded_dfs, ignore_index=True)
    df = df[["timestamp", "pv_power_kw"]].sort_values("timestamp").reset_index(drop=True)

    print(f"Available data range for site {target_site_id}: {df['timestamp'].min()} to {df['timestamp'].max()}")

    if df.empty:
        raise RuntimeError(f"Selected site_id={target_site_id} has no rows in the requested date range.")

    site_meta = meta_df.loc[meta_df["site_id"] == target_site_id].iloc[0]
    elevation = _solar_elevation_deg(pd.DatetimeIndex(df["timestamp"]), float(site_meta["latitude"]))
    df["solar_elevation_deg"] = elevation

    resolved = {
        "site_id": int(target_site_id),
        "latitude": float(site_meta["latitude"]),
        "longitude": float(site_meta["longitude"]),
        "capacity_kw": float(site_meta["capacity_kw"]),
        "source": HF_REPO_ID,
        "date_range": [START_DATE, END_DATE],
    }
    save_json(resolved, RAW_METADATA_DIR / "resolved_site_metadata.json")
    logger.info(f"Resolved real site metadata -> {RAW_METADATA_DIR / 'resolved_site_metadata.json'}")

    n_missing = int(df["pv_power_kw"].isna().sum())
    logger.info(f"Loaded real uk_pv series for site_id={target_site_id}: {len(df)} rows "
                f"({START_DATE}..{END_DATE}), {n_missing} missing readings.")
    return df


def main():
    df = build_real_pv_series()
    out_path = RAW_PV_DIR / "pv_raw.parquet"
    save_parquet(df, out_path)
    logger.info(f"Saved raw PV data -> {out_path} ({len(df)} rows).")
    return out_path


if __name__ == "__main__":
    main()