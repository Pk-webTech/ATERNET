"""
Named feature groups produced by Phase 1. Downstream phases index
into the processed tensors using these lists so that column order
is never hard-coded in more than one place.
"""

# Sequence features fed to the temporal encoder (per time step, in the window)
SEQUENCE_FEATURES = [
    "pv_norm",
    "ghi",
    "cloud_cover",
    "temp_c",
    "wind_speed",
    "clear_sky_index",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

# Raw sigma / uncertainty features (per time step) -- consumed by Phase 4 gate
SIGMA_FEATURES = [
    "sigma_nwp_ghi",
    "sigma_nwp_cloud",
    "sigma_nwp_temp",
    "sigma_nwp_wind",
    "sigma_expert_placeholder",  # overwritten with real sigma_expert after Phase 3
    "ensemble_spread_norm",
]

# NWP-only features (per time step) -- ensemble mean forecast values
NWP_FEATURES = [
    "nwp_ghi_mean",
    "nwp_cloud_cover_mean",
    "nwp_temp_c_mean",
    "nwp_wind_speed_mean",
]

# Static / per-window regime & volatility features
REGIME_FEATURES = [
    "regime_id",
    "regime_dawn_ramp",
    "regime_clear_peak",
    "regime_overcast",
    "regime_volatile",
]

VOLATILITY_FEATURES = [
    "ramp_rate_std",
    "rolling_std_1h",
    "rolling_std_3h",
    "clear_sky_index_std",
]

# Everything needed by the Dual-Uncertainty Routing gate as metadata
ROUTING_METADATA_FEATURES = REGIME_FEATURES + VOLATILITY_FEATURES

# Index of regime_id within a meta row -- used by evaluation to slice
# per-regime metrics without re-deriving column order.
REGIME_ID_INDEX = ROUTING_METADATA_FEATURES.index("regime_id")

ALL_FEATURE_GROUPS = {
    "sequence": SEQUENCE_FEATURES,
    "sigma": SIGMA_FEATURES,
    "nwp": NWP_FEATURES,
    "regime": REGIME_FEATURES,
    "volatility": VOLATILITY_FEATURES,
}
