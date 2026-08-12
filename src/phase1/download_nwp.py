"""
Phase 1 -- NWP ensemble data acquisition.

No live ECMWF API access is available in this environment, so this
module generates a synthetic ensemble forecast (N members) whose
ensemble spread is deliberately correlated with the PV series' cloud
volatility (generated with the same seed / underlying process family
in download_pv.py) -- this is what makes sigma_nwp a meaningful,
non-trivial uncertainty signal rather than pure noise.

Output schema: long-format parquet with one row per
(timestamp, ensemble_member) so nwp_processor.py can compute
ensemble mean / spread with a simple groupby.
"""

import numpy as np
import pandas as pd

from config.paths import RAW_NWP_DIR, RAW_PV_DIR
from config.model_config import DATA_CONFIG
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.io import save_parquet, load_parquet

logger = get_logger(__name__)


def generate_synthetic_nwp_ensemble(
    reference_timestamps: pd.DatetimeIndex,
    n_members: int = DATA_CONFIG.n_ensemble_members,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build an N-member ensemble forecast for (ghi, cloud_cover, temp, wind)
    aligned to reference_timestamps. Ensemble spread grows during volatile
    periods (approximated via a rolling-derivative proxy on a synthetic
    baseline), which is exactly the behaviour real ECMWF ENS spread exhibits
    around frontal passages / convective cloud.
    """
    set_seed(seed + 1)  # offset from PV seed so series aren't identical

    n = len(reference_timestamps)
    hour = (reference_timestamps.hour + reference_timestamps.minute / 60.0).values.astype(float)
    doy = reference_timestamps.dayofyear.values.astype(float)

    # Deterministic ensemble-mean baseline (smooth diurnal + seasonal shape)
    ghi_base = np.clip(
        900 * np.sin(np.pi * np.clip((hour - 6) / 12.0, 0, 1)) *
        (0.8 + 0.2 * np.sin(2 * np.pi * doy / 365.0)),
        0, None,
    )
    cloud_base = 0.4 + 0.2 * np.sin(2 * np.pi * doy / 365.0 + 1.0)
    temp_base = 24 + 6 * np.sin(np.pi * np.clip((hour - 6) / 12.0, 0, 1))
    wind_base = 3.0 + 1.5 * np.abs(np.sin(2 * np.pi * doy / 200.0))

    # Volatility proxy: local variability drives ensemble spread magnitude
    volatility_proxy = np.abs(np.gradient(cloud_base + 0.3 * np.sin(np.arange(n) / 17.0)))
    volatility_proxy = (volatility_proxy - volatility_proxy.min()) / (
        volatility_proxy.max() - volatility_proxy.min() + 1e-8
    )
    spread_scale = 0.15 + 0.85 * volatility_proxy  # in [0.15, 1.0]

    rows = []
    for member in range(n_members):
        member_noise = np.random.normal(0, 1, n)
        ghi_m = ghi_base * (1 + spread_scale * 0.25 * member_noise)
        cloud_m = np.clip(cloud_base + spread_scale * 0.20 * member_noise, 0, 1)
        temp_m = temp_base + spread_scale * 1.5 * np.random.normal(0, 1, n)
        wind_m = np.clip(wind_base + spread_scale * 1.0 * np.random.normal(0, 1, n), 0, None)

        rows.append(pd.DataFrame({
            "timestamp": reference_timestamps,
            "ensemble_member": member,
            "ghi": np.clip(ghi_m, 0, None),
            "cloud_cover": cloud_m,
            "temp_c": temp_m,
            "wind_speed": wind_m,
        }))

    df = pd.concat(rows, ignore_index=True)
    logger.info(f"Synthesized NWP ensemble: {n_members} members x {n} timestamps "
                f"= {len(df)} rows.")
    return df


def main():
    pv_raw_path = RAW_PV_DIR / "pv_raw.parquet"
    if not pv_raw_path.exists():
        raise FileNotFoundError(
            f"{pv_raw_path} not found. Run src/phase1/download_pv.py first "
            f"so NWP timestamps can be aligned to the PV series."
        )
    pv_df = load_parquet(pv_raw_path)
    timestamps = pd.DatetimeIndex(pd.to_datetime(pv_df["timestamp"]).unique()).sort_values()

    df = generate_synthetic_nwp_ensemble(timestamps)
    out_path = RAW_NWP_DIR / "nwp_ensemble.parquet"
    save_parquet(df, out_path)
    logger.info(f"Saved raw NWP ensemble data -> {out_path} ({len(df)} rows).")
    return out_path


if __name__ == "__main__":
    main()
