"""
Phase 1 -- NWP ensemble data acquisition (REAL DATA: ECMWF ERA5 via CDS).

Replaces the synthetic ensemble generator with actual ERA5 data pulled
from the Copernicus Climate Data Store (`cdsapi`), for the 6 variables
and Great Britain bounding box specified in project memory:
    ssrd, tcc, 2t, 10u, 10v, sp   |   GB bbox, 2010-2026

WHY "ensemble_members", NOT PLAIN "reanalysis"
-------------------------------------------------
`nwp_processor.py` (downstream, unchanged) computes sigma_nwp as the
STD ACROSS ENSEMBLE MEMBERS at each timestamp -- and already treats a
single-member group's std as 0 (see its `.fillna(0.0)`). Plain ERA5
"reanalysis" is a single deterministic field. Swapping that in naively
would silently zero out sigma_nwp everywhere, killing the uncertainty
signal the entire Phase 4 routing gate is built around -- a much worse
failure than it looks, because training would still run and produce
plausible-looking numbers.

CDS instead offers ERA5 in an `ensemble_members` product_type: a
10-member EDA (Ensemble of Data Assimilations), 3-hourly, ~1 degree
resolution -- coarser than the deterministic product, but it is a real
spread estimate, which is the property this architecture actually
needs. This module requests that product and reshapes it into the same
long-format (timestamp, ensemble_member, ghi, cloud_cover, temp_c,
wind_speed) that `nwp_processor.py` already expects, so nwp_processor.py
and everything downstream of it needs ZERO changes.

`ssrd` (surface solar radiation downwards) is a J/m^2 accumulated
quantity in the CDS product, not an instantaneous W/m^2 rate like the
`ghi` column downstream expects -- it's de-accumulated and divided by
the accumulation period here.

FLAGGED FOR VERIFICATION: exact CDS variable/product-type strings
(`product_type`, `variable`, `number` semantics) can drift between CDS
API versions and I can't browse cds.climate.copernicus.eu from this
sandbox to confirm against current docs. Test with a small bbox/date
range before committing to a full 2010-2026 pull.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from config.paths import RAW_NWP_DIR, RAW_PV_DIR, CACHE_DIR
from src.utils.logger import get_logger
from src.utils.io import save_parquet, load_parquet

logger = get_logger(__name__)

# Great Britain bounding box: North, West, South, East (CDS area order)
GB_AREA = [61.0, -8.5, 49.5, 2.0]

CDS_VARIABLES = [
    "surface_solar_radiation_downwards",  # ssrd
    "total_cloud_cover",                  # tcc
    "2m_temperature",                     # 2t
    "10m_u_component_of_wind",            # 10u
    "10m_v_component_of_wind",            # 10v
    "surface_pressure",                   # sp
]

# 10-member EDA spread product. See module docstring re: verification.
CDS_PRODUCT_TYPE = "ensemble_members"
N_EDA_MEMBERS = 10

START_DATE = "2018-01-01"
END_DATE = "2021-10-31"


def _fetch_era5_raw(start_date: str, end_date: str, area=GB_AREA) -> Path:
    """
    Single cdsapi retrieve call for the full date range. For a multi-year
    pull, CDS strongly prefers/queues large single requests over many
    small ones; if this times out in practice, split by year and
    concatenate -- the reshaping logic below is unaffected either way.

    Requires a working ~/.cdsapirc (CDS API key). See:
    https://cds.climate.copernicus.eu/how-to-api
    """
    import cdsapi

    out_path = CACHE_DIR / f"era5_eda_{start_date}_{end_date}.nc"
    if out_path.exists():
        logger.info(f"ERA5 EDA cache hit: {out_path}")
        return out_path

    dates = pd.date_range(start_date, end_date, freq="D")
    years = sorted(dates.year.unique().astype(str).tolist())
    months = sorted({f"{m:02d}" for m in dates.month.unique()})
    days = sorted({f"{d:02d}" for d in range(1, 32)})

    client = cdsapi.Client()
    request = {
        "product_type": CDS_PRODUCT_TYPE,
        "variable": CDS_VARIABLES,
        "year": years,
        "month": months,
        "day": days,
        "time": [f"{h:02d}:00" for h in range(0, 24, 3)],  # EDA is 3-hourly
        "area": area,
        "format": "netcdf",
    }
    logger.info(f"Submitting CDS request for {start_date}..{end_date} "
                f"({len(years)} year(s), product_type={CDS_PRODUCT_TYPE}). "
                f"This can take a long time to queue on CDS's end.")
    client.retrieve("reanalysis-era5-single-levels", request, str(out_path))
    logger.info(f"Saved raw ERA5 EDA NetCDF -> {out_path}")
    return out_path


def _nearest_site_latlon() -> tuple:
    """
    Read the site lat/lon resolved by download_pv.py so the spatial join
    targets the actual PV site rather than a hardcoded coordinate.
    """
    import json
    resolved_path = Path(RAW_PV_DIR).parent / "metadata" / "resolved_site_metadata.json"
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"{resolved_path} not found -- run download_pv.py first so the "
            f"real PV site's lat/lon is available for the ERA5 spatial join."
        )
    with open(resolved_path) as f:
        meta = json.load(f)
    return meta["latitude"], meta["longitude"]


def _reshape_to_long_format(nc_path: Path, latitude: float, longitude: float) -> pd.DataFrame:
    """
    xarray nearest-gridpoint spatial join (per project memory:
    `.sel(method='nearest')` on rounded lat/lon), then reshape the
    (time, number, [lat/lon already collapsed]) cube into the long
    (timestamp, ensemble_member, var...) format nwp_processor.py expects.
    """
    ds = xr.open_dataset(nc_path)
    ds = ds.sel(latitude=latitude, longitude=longitude, method="nearest")

    # Map CDS short names -> this project's column names. CDS netCDF
    # short names for these variables are typically ssrd/tcc/t2m/u10/v10/sp;
    # this mapping should be checked against the actual downloaded file's
    # `ds.data_vars` if CDS has renamed anything.
    var_map = {"ssrd": "ghi", "tcc": "cloud_cover", "t2m": "temp_c", "u10": "u", "v10": "v", "sp": "sp"}
    missing = [k for k in var_map if k not in ds.data_vars]
    if missing:
        raise KeyError(
            f"Expected ERA5 variables {missing} not found in downloaded file. "
            f"Available: {list(ds.data_vars)}. Update var_map in _reshape_to_long_format()."
        )

    df = ds[list(var_map)].to_dataframe().reset_index()
    df = df.rename(columns=var_map)
    df = df.rename(columns={"time": "timestamp", "number": "ensemble_member"})

    # ssrd is accumulated (J/m^2) over the 3h step in this product;
    # de-accumulate to an instantaneous-equivalent W/m^2 rate.
    df["ghi"] = np.clip(df["ghi"] / (3 * 3600.0), 0.0, None)

    # Kelvin -> Celsius
    df["temp_c"] = df["temp_c"] - 273.15

    # tcc in ERA5 is a fraction [0,1]; keep as-is (0-1), matching the
    # synthetic generator's convention.
    df["cloud_cover"] = df["cloud_cover"].clip(0.0, 1.0)

    df["wind_speed"] = np.sqrt(df["u"] ** 2 + df["v"] ** 2)

    keep = ["timestamp", "ensemble_member", "ghi", "cloud_cover", "temp_c", "wind_speed"]
    return df[keep].sort_values(["timestamp", "ensemble_member"]).reset_index(drop=True)


def _resample_to_pv_grid(nwp_long: pd.DataFrame, pv_timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    ERA5 EDA is 3-hourly; the PV series is 30-min. Interpolate each
    ensemble member's series independently onto the PV timestamp grid
    (matches the project's documented hourly->30-min interpolation
    approach, just with a 3h source cadence instead of 1h).
    """
    out = []
    for member, g in nwp_long.groupby("ensemble_member"):
        g = g.set_index("timestamp").sort_index()
        g = g.reindex(g.index.union(pv_timestamps)).interpolate(method="time").reindex(pv_timestamps)
        g["ensemble_member"] = member
        out.append(g.reset_index(names="timestamp"))
    return pd.concat(out, ignore_index=True)


def main():
    pv_raw_path = RAW_PV_DIR / "pv_raw.parquet"
    if not pv_raw_path.exists():
        raise FileNotFoundError(
            f"{pv_raw_path} not found. Run src/phase1/download_pv.py first "
            f"so NWP timestamps and the site's lat/lon can be aligned to the PV series."
        )
    pv_df = load_parquet(pv_raw_path)
    pv_timestamps = pd.DatetimeIndex(pd.to_datetime(pv_df["timestamp"]).unique()).sort_values()

    latitude, longitude = _nearest_site_latlon()
    nc_path = _fetch_era5_raw(START_DATE, END_DATE)
    nwp_long = _reshape_to_long_format(nc_path, latitude, longitude)
    nwp_resampled = _resample_to_pv_grid(nwp_long, pv_timestamps)

    out_path = RAW_NWP_DIR / "nwp_ensemble.parquet"
    save_parquet(nwp_resampled, out_path)
    logger.info(f"Saved raw NWP ensemble data -> {out_path} ({len(nwp_resampled)} rows, "
                f"{nwp_resampled['ensemble_member'].nunique()} members).")
    return out_path


if __name__ == "__main__":
    main()
