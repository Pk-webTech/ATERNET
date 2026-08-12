"""
Phase 1 -- Feature engineering.

Merges cleaned PV + processed NWP series and derives every feature
required by later phases:
  - clear_sky_index                (PV physics-normalized signal)
  - cyclical time features         (hour/day-of-year sin-cos)
  - rule-based regime labels       (dawn_ramp / clear_peak / overcast / volatile)
  - volatility features            (ramp-rate & rolling std, feed expert
                                     disagreement estimation and the routing gate)
  - sigma_nwp per-variable columns (renamed/aligned to SIGMA_FEATURES schema)

Rule-based regimes exist ONLY to stratify Phase-1 windows for balanced
sampling and sanity-check plots. They are NOT the learned expert routing
mechanism (that's Phase 4) -- they are a deterministic, reproducible prior.
"""

import numpy as np
import pandas as pd

from config.paths import PV_CLEAN_PATH, NWP_CLEAN_PATH, FEATURES_PATH
from src.utils.constants import COL_TIMESTAMP, REGIME_LIST, REGIME_TO_ID
from src.utils.helpers import add_time_features
from src.utils.logger import get_logger
from src.utils.io import load_parquet, save_parquet
from src.phase1.download_pv import _clear_sky_ghi

logger = get_logger(__name__)


def merge_pv_nwp(pv_df: pd.DataFrame, nwp_df: pd.DataFrame) -> pd.DataFrame:
    pv_df = pv_df.copy()
    nwp_df = nwp_df.copy()
    pv_df[COL_TIMESTAMP] = pd.to_datetime(pv_df["timestamp"])
    nwp_df[COL_TIMESTAMP] = pd.to_datetime(nwp_df["timestamp"])

    merged = pd.merge(pv_df, nwp_df, on=COL_TIMESTAMP, how="inner", suffixes=("", "_nwp"))
    merged = merged.sort_values(COL_TIMESTAMP).reset_index(drop=True)
    return merged


def add_clear_sky_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    clear_sky_ghi = _clear_sky_ghi(df["solar_elevation_deg"].values)
    df["clear_sky_ghi"] = clear_sky_ghi
    df["clear_sky_index"] = np.where(
        clear_sky_ghi > 10.0, df["ghi_measured"] / np.clip(clear_sky_ghi, 10.0, None), 0.0
    )
    df["clear_sky_index"] = np.clip(df["clear_sky_index"], 0.0, 1.5)
    return df


def add_volatility_features(df: pd.DataFrame, freq_minutes: int = 15) -> pd.DataFrame:
    df = df.copy()
    steps_per_hour = max(1, 60 // freq_minutes)

    df["ramp_rate"] = df["pv_norm"].diff().fillna(0.0)
    df["ramp_rate_std"] = df["ramp_rate"].rolling(
        window=steps_per_hour, min_periods=1
    ).std().fillna(0.0)

    df["rolling_std_1h"] = df["pv_norm"].rolling(
        window=steps_per_hour, min_periods=1
    ).std().fillna(0.0)
    df["rolling_std_3h"] = df["pv_norm"].rolling(
        window=3 * steps_per_hour, min_periods=1
    ).std().fillna(0.0)
    df["clear_sky_index_std"] = df["clear_sky_index"].rolling(
        window=steps_per_hour, min_periods=1
    ).std().fillna(0.0)
    return df


def assign_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic rule-based regime tagging used for stratified splitting
    and sanity plots (see module docstring for scope).
    """
    df = df.copy()
    is_daylight = df["solar_elevation_deg"] > 2.0

    # Dawn/dusk ramp: low-but-rising elevation, meaningful positive ramp rate
    is_ramp = is_daylight & (df["solar_elevation_deg"] < 20.0) & (df["ramp_rate"].abs() > 0.01)

    # Clear peak: high clear-sky index, low short-term volatility
    is_clear_peak = is_daylight & (df["clear_sky_index"] > 0.75) & (df["rolling_std_1h"] < 0.05)

    # Overcast: daylight but persistently low clear-sky index
    is_overcast = is_daylight & (df["clear_sky_index"] <= 0.4) & (df["rolling_std_1h"] < 0.05)

    # Volatile: everything else with meaningful daylight and high short-term std
    is_volatile = is_daylight & (df["rolling_std_1h"] >= 0.05)

    regime = np.select(
        [is_ramp, is_clear_peak, is_overcast, is_volatile],
        ["dawn_ramp", "clear_peak", "overcast", "volatile"],
        default="overcast",  # night / low-signal steps default to the flattest regime
    )
    df["regime"] = regime
    df["regime_id"] = df["regime"].map(REGIME_TO_ID).astype(int)

    for r in REGIME_LIST:
        df[f"regime_{r}"] = (df["regime"] == r).astype(np.float32)

    return df


def build_features(pv_df: pd.DataFrame, nwp_df: pd.DataFrame, freq_minutes: int = 15) -> pd.DataFrame:
    df = merge_pv_nwp(pv_df, nwp_df)
    df = add_clear_sky_index(df)
    df = add_time_features(df, timestamp_col="timestamp")
    df = add_volatility_features(df, freq_minutes=freq_minutes)
    df = assign_regime(df)

    # Rename NWP ensemble std columns to the SIGMA_FEATURES naming convention
    df["sigma_nwp_ghi"] = df["nwp_ghi_std"]
    df["sigma_nwp_cloud"] = df["nwp_cloud_cover_std"]
    df["sigma_nwp_temp"] = df["nwp_temp_c_std"]
    df["sigma_nwp_wind"] = df["nwp_wind_speed_std"]
    # Placeholder until Phase 3 produces real expert-disagreement sigma;
    # kept at 0 here so shapes/columns are stable end-to-end from Phase 1.
    df["sigma_expert_placeholder"] = 0.0

    df["cloud_cover"] = df["nwp_cloud_cover_mean"]
    df["temp_c"] = df["nwp_temp_c_mean"]
    df["wind_speed"] = df["nwp_wind_speed_mean"]
    df["ghi"] = df["ghi_measured"]

    logger.info(f"Feature engineering complete: {len(df)} rows, {df.shape[1]} columns. "
                f"Regime distribution: {df['regime'].value_counts().to_dict()}")
    return df


def main():
    pv_df = load_parquet(PV_CLEAN_PATH)
    nwp_df = load_parquet(NWP_CLEAN_PATH)
    features_df = build_features(pv_df, nwp_df)
    save_parquet(features_df, FEATURES_PATH)
    logger.info(f"Saved engineered features -> {FEATURES_PATH}")
    return FEATURES_PATH


if __name__ == "__main__":
    main()
