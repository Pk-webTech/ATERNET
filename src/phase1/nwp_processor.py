"""
Phase 1 -- NWP ensemble processing.

Collapses the raw (timestamp, ensemble_member) long-format ensemble into
one row per timestamp containing:
  - ensemble mean for each variable (the point NWP forecast)
  - ensemble standard deviation for each variable (sigma_nwp, the raw
    ingredient for NWP forecast uncertainty used later by the
    Dual-Uncertainty Routing gate)
"""

import numpy as np
import pandas as pd

from config.paths import RAW_NWP_DIR, NWP_CLEAN_PATH
from src.utils.constants import COL_TIMESTAMP
from src.utils.logger import get_logger
from src.utils.io import load_parquet, save_parquet

logger = get_logger(__name__)

NWP_VARS = ["ghi", "cloud_cover", "temp_c", "wind_speed"]


def compute_ensemble_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[COL_TIMESTAMP] = pd.to_datetime(df["timestamp"])

    agg_dict = {}
    for var in NWP_VARS:
        agg_dict[f"nwp_{var}_mean"] = (var, "mean")
        agg_dict[f"nwp_{var}_std"] = (var, "std")

    grouped = df.groupby(COL_TIMESTAMP).agg(**agg_dict).reset_index()

    # std of a single-member group is NaN by definition; treat as 0 spread.
    for var in NWP_VARS:
        grouped[f"nwp_{var}_std"] = grouped[f"nwp_{var}_std"].fillna(0.0)

    # Normalized ensemble spread summary (used as a compact sigma_nwp signal).
    # max() over a DataFrame returns a Series indexed by column; .replace(0, 1.0)
    # guards against a degenerate all-zero-spread column producing 0/0.
    std_cols = [f"nwp_{v}_std" for v in NWP_VARS]
    max_std = grouped[std_cols].max().replace(0, 1.0)
    normalized = grouped[std_cols] / max_std
    grouped["ensemble_spread_norm"] = normalized.mean(axis=1)

    grouped = grouped.sort_values(COL_TIMESTAMP).reset_index(drop=True)
    return grouped


def main():
    raw_path = RAW_NWP_DIR / "nwp_ensemble.parquet"
    df_raw = load_parquet(raw_path)
    df_stats = compute_ensemble_stats(df_raw)
    save_parquet(df_stats, NWP_CLEAN_PATH)
    logger.info(f"Saved processed NWP ensemble stats -> {NWP_CLEAN_PATH} "
                f"({len(df_stats)} rows, columns: {list(df_stats.columns)}).")
    return NWP_CLEAN_PATH


if __name__ == "__main__":
    main()
