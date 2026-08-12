"""
Project-wide constants that are not user-tunable hyperparameters
(physical constants, fixed column names, etc).
"""

# Solar physics
SOLAR_CONSTANT_WM2 = 1361.0          # extraterrestrial irradiance
STEFAN_BOLTZMANN = 5.670374419e-8

# Clip PV output to [0, 1] normalized capacity
PV_NORM_MIN = 0.0
PV_NORM_MAX = 1.0

# Column name conventions used across all Phase 1 modules
COL_TIMESTAMP = "timestamp"
COL_PV_POWER = "pv_power_kw"
COL_PV_NORM = "pv_norm"
COL_GHI = "ghi"
COL_CLOUD_COVER = "cloud_cover"
COL_TEMP = "temp_c"
COL_WIND = "wind_speed"

COL_SIGMA_NWP = "sigma_nwp"
COL_REGIME = "regime"
COL_VOLATILITY = "volatility"

# Regime labels (rule-based, used before any learned expert gating exists)
REGIME_DAWN_RAMP = "dawn_ramp"
REGIME_CLEAR_PEAK = "clear_peak"
REGIME_OVERCAST = "overcast"
REGIME_VOLATILE = "volatile"

REGIME_LIST = [REGIME_DAWN_RAMP, REGIME_CLEAR_PEAK, REGIME_OVERCAST, REGIME_VOLATILE]
REGIME_TO_ID = {r: i for i, r in enumerate(REGIME_LIST)}
ID_TO_REGIME = {i: r for r, i in REGIME_TO_ID.items()}

# Random seed default
DEFAULT_SEED = 42
