"""
Phase 1 -- PV data cleaning & normalization.

Takes data/raw/pv/pv_raw.parquet and produces a clean, gap-free,
capacity-normalized series at data/interim/pv_clean.parquet.
"""

import numpy as np
import pandas as pd

from config.paths import RAW_PV_DIR, PV_CLEAN_PATH
from src.utils.site_metadata import DEFAULT_SITE
from src.utils.constants import COL_TIMESTAMP, COL_PV_POWER, COL_PV_NORM
from src.utils.logger import get_logger
from src.utils.io import load_parquet, save_parquet

logger = get_logger(__name__)


def clean_pv_series(df: pd.DataFrame, capacity_kw: float = DEFAULT_SITE.capacity_kw) -> pd.DataFrame:
    df = df.copy()
    df[COL_TIMESTAMP] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(COL_TIMESTAMP).drop_duplicates(subset=COL_TIMESTAMP)
    df = df.set_index(COL_TIMESTAMP)

    # Reindex onto a strictly regular grid so downstream windowing never
    # silently skips a missing timestamp.
    inferred = pd.infer_freq(df.index[:20])
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=inferred or "15min")
    df = df.reindex(full_index)
    df.index.name = COL_TIMESTAMP

    # Physically invalid negative readings -> NaN, then interpolate short gaps
    df.loc[df["pv_power_kw"] < 0, "pv_power_kw"] = np.nan
    n_missing_before = int(df["pv_power_kw"].isna().sum())

    df["pv_power_kw"] = df["pv_power_kw"].interpolate(
        method="linear", limit=8, limit_direction="both"
    )
    # Anything still missing after interpolation (long gaps) is filled with 0
    # at night (elevation <= 0) and forward-filled during daytime as a
    # conservative fallback -- flagged via a boolean mask for transparency.
    df["was_imputed"] = df["pv_power_kw"].isna()
    if "solar_elevation_deg" in df.columns:
        night_mask = df["solar_elevation_deg"] <= 0
    else:
        night_mask = pd.Series(False, index=df.index)
    df.loc[df["pv_power_kw"].isna() & night_mask, "pv_power_kw"] = 0.0
    df["pv_power_kw"] = df["pv_power_kw"].ffill().fillna(0.0)

    # Ancillary columns can also carry NaNs introduced by the reindex above;
    # leaving them would poison clear_sky_index / regime assignment later.
    for col in ["ghi_measured", "ambient_temp_c", "solar_elevation_deg"]:
        if col in df.columns:
            df[col] = df[col].interpolate(limit_direction="both").ffill().bfill()

    n_missing_after = int(df["was_imputed"].sum())
    logger.info(f"PV cleaning: {n_missing_before} missing points found, "
                f"{n_missing_after} required fallback imputation.")

    # Normalize by nameplate capacity
    df[COL_PV_NORM] = np.clip(df["pv_power_kw"] / capacity_kw, 0.0, 1.0)

    df = df.reset_index()
    return df


def main():
    raw_path = RAW_PV_DIR / "pv_raw.parquet"
    df_raw = load_parquet(raw_path)
    df_clean = clean_pv_series(df_raw)
    save_parquet(df_clean, PV_CLEAN_PATH)
    logger.info(f"Saved cleaned PV data -> {PV_CLEAN_PATH} ({len(df_clean)} rows).")
    return PV_CLEAN_PATH


if __name__ == "__main__":
    main()
