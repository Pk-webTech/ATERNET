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
explicitly. If unset, `_select_site()` picks the site with the fewest
missing readings over the requested date range (a reasonable default,
not necessarily the "best" site for the paper -- override `SITE_ID`
once you and Piyush have picked one for the final numbers).

Multi-site training (pooling several/all 396 series) is a natural
follow-on but changes what a "window" and a "split" mean, so it's out
of scope for this swap.
"""

from typing import Optional

import numpy as np
import pandas as pd

from config.paths import RAW_PV_DIR, RAW_METADATA_DIR
from src.utils.logger import get_logger
from src.utils.io import save_parquet, save_json

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Pure-geometry helpers. Kept here (not moved) because
# feature_engineering.py imports `_clear_sky_ghi` from this module --
# relocating it would be an unrelated breaking change to a file this
# swap isn't supposed to touch.
# ---------------------------------------------------------------------

def _solar_elevation_deg(timestamps: pd.DatetimeIndex, latitude: float) -> np.ndarray:
    """Simplified solar elevation angle (degrees). Unchanged from the
    synthetic-data version -- this is pure geometry, no data dependency."""
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
    """Simplified clear-sky GHI (W/m^2). Unchanged from the synthetic-data
    version; still imported by feature_engineering.py."""
    elev_clipped = np.clip(elevation_deg, 0.0, 90.0)
    air_mass = 1.0 / np.clip(np.sin(np.deg2rad(elev_clipped + 0.001)), 1e-3, None)
    transmittance = 0.75 ** (air_mass ** 0.678)
    ghi = solar_constant * np.sin(np.deg2rad(elev_clipped)) * transmittance
    ghi = np.where(elevation_deg > 0, ghi, 0.0)
    return np.clip(ghi, 0.0, None)

HF_REPO_ID = "openclimatefix/uk_pv"
HF_REPO_TYPE = "dataset"

# Pin a specific uk_pv system id (a.k.a. ss_id) once you've chosen one for
# the paper's final numbers. None -> auto-select (see module docstring).
SITE_ID: Optional[int] = None

# uk_pv's date coverage; narrow this if you want a shorter, faster pull.
START_DATE = "2018-01-01"
END_DATE = "2021-10-31"


def _find_repo_files():
    """
    uk_pv's exact filenames have changed across dataset revisions, so we
    discover them rather than hardcoding paths that might drift.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(HF_REPO_ID, repo_type=HF_REPO_TYPE)

    metadata_files = [f for f in files if "metadata" in f.lower() and f.endswith((".csv", ".parquet"))]
    # Prefer the half-hourly ("30min") generation file over the 5-minute
    # one -- this project's native frequency is 15 min in the synthetic
    # version but uk_pv's finest granularity that's uniformly available
    # across sites is 30 min; downstream `pv_processor.py` reindexes onto
    # whatever regular grid is inferred, so 30-min input is fine.
    gen_files = [f for f in files if f.endswith(".parquet") and "metadata" not in f.lower()]
    half_hourly = [f for f in gen_files if "30" in f]
    chosen_gen = half_hourly if half_hourly else gen_files

    if not metadata_files or not chosen_gen:
        raise FileNotFoundError(
            f"Could not locate expected files in {HF_REPO_ID}. "
            f"Found {len(files)} repo files total; metadata candidates: {metadata_files}; "
            f"generation candidates: {gen_files}. Inspect the repo manually at "
            f"https://huggingface.co/datasets/{HF_REPO_ID} and set exact paths."
        )
    return metadata_files[0], chosen_gen[0]


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

    # Column names have varied across uk_pv revisions (ss_id vs system_id,
    # latitude_rounded vs latitude, kwp vs capacity_kw). Normalize the
    # handful this module actually needs.
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
        raise KeyError(
            f"uk_pv metadata is missing expected column(s) {missing} after normalization; "
            f"raw columns were {list(pd.read_parquet(metadata_path).columns) if metadata_path.endswith('.parquet') else list(pd.read_csv(metadata_path, nrows=0).columns)}. "
            f"Update the rename_map in _load_metadata()."
        )
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

    required = {"site_id", "timestamp", "pv_power_kw"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"uk_pv generation file is missing expected column(s) {missing} after "
            f"normalization; raw columns were {list(pd.read_parquet(gen_path).columns)}. "
            f"Update the rename_map in _load_generation()."
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _select_site(gen_df: pd.DataFrame, meta_df: pd.DataFrame, site_id: Optional[int]) -> int:
    if site_id is not None:
        if site_id not in set(meta_df["site_id"]):
            raise ValueError(f"SITE_ID={site_id} not present in uk_pv metadata.")
        return site_id

    in_range = gen_df[(gen_df["timestamp"] >= START_DATE) & (gen_df["timestamp"] <= END_DATE)]
    completeness = in_range.groupby("site_id")["pv_power_kw"].apply(lambda s: s.notna().mean())
    completeness = completeness[completeness.index.isin(meta_df["site_id"])]
    if completeness.empty:
        raise RuntimeError("No uk_pv site has any generation data in START_DATE..END_DATE.")
    best = completeness.idxmax()
    logger.info(f"Auto-selected site_id={best} (data completeness {completeness[best]:.1%} "
                f"over {START_DATE}..{END_DATE}). Set SITE_ID to pin this explicitly.")
    return int(best)


def build_real_pv_series() -> pd.DataFrame:
    metadata_path_in_repo, gen_path_in_repo = _find_repo_files()
    metadata_local = _download(metadata_path_in_repo)
    gen_local = _download(gen_path_in_repo)

    meta_df = _load_metadata(metadata_local)
    gen_df = _load_generation(gen_local)

    site_id = _select_site(gen_df, meta_df, SITE_ID)
    site_meta = meta_df.loc[meta_df["site_id"] == site_id].iloc[0]

    df = gen_df.loc[
        (gen_df["site_id"] == site_id)
        & (gen_df["timestamp"] >= START_DATE)
        & (gen_df["timestamp"] <= END_DATE),
        ["timestamp", "pv_power_kw"],
    ].sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"Selected site_id={site_id} has no rows in the requested date range.")

    elevation = _solar_elevation_deg(pd.DatetimeIndex(df["timestamp"]), float(site_meta["latitude"]))
    df["solar_elevation_deg"] = elevation

    # Persist the resolved real site metadata so pv_processor's capacity
    # normalization (and anything else keyed on DEFAULT_SITE) uses the
    # ACTUAL site's nameplate rather than the synthetic 1000 kW default.
    resolved = {
        "site_id": int(site_id),
        "latitude": float(site_meta["latitude"]),
        "longitude": float(site_meta["longitude"]),
        "capacity_kw": float(site_meta["capacity_kw"]),
        "source": HF_REPO_ID,
        "date_range": [START_DATE, END_DATE],
    }
    save_json(resolved, RAW_METADATA_DIR / "resolved_site_metadata.json")
    logger.info(f"Resolved real site metadata -> {RAW_METADATA_DIR / 'resolved_site_metadata.json'}: "
                f"{resolved}")

    n_missing = int(df["pv_power_kw"].isna().sum())
    logger.info(f"Loaded real uk_pv series for site_id={site_id}: {len(df)} rows "
                f"({START_DATE}..{END_DATE}), {n_missing} missing readings "
                f"(pv_processor.py handles gap-filling).")
    return df


def main():
    df = build_real_pv_series()
    out_path = RAW_PV_DIR / "pv_raw.parquet"
    save_parquet(df, out_path)
    logger.info(f"Saved raw PV data -> {out_path} ({len(df)} rows).")
    return out_path


if __name__ == "__main__":
    main()
