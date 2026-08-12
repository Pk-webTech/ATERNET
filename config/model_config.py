"""
Model architecture hyperparameters for ATERNET.
Phase 1 only uses the WINDOW / HORIZON / FREQ constants; the rest
is here so later phases (2-5) share a single source of truth.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    # Native sampling frequency of the PV / NWP series
    freq_minutes: int = 15

    # Lookback window length (in time steps) fed to the temporal encoder
    window_size: int = 48          # 48 * 15min = 12 hours of history

    # Forecast horizon (in time steps)
    horizon: int = 16              # 16 * 15min = 4 hours ahead

    # Stride between consecutive sliding windows
    stride: int = 4

    # Number of NWP ensemble members simulated / ingested
    n_ensemble_members: int = 20

    # Regimes used for expert routing / stratification
    regimes: List[str] = field(default_factory=lambda: [
        "dawn_ramp", "clear_peak", "overcast", "volatile"
    ])


@dataclass
class PatchTSTConfig:
    patch_len: int = 8
    stride: int = 4
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1


@dataclass
class ExpertConfig:
    n_experts: int = 4
    hidden_dims: dict = field(default_factory=lambda: {
        "dawn_ramp": [256, 256, 128],
        "clear_peak": [128, 64],
        "overcast": [128, 128],
        "volatile": [512, 256, 128],
    })


@dataclass
class QuantileConfig:
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


@dataclass
class EmbeddingConfig:
    """Dimensions of the three Phase 2 backbone output heads."""
    h_expert_dim: int = 128
    h_sigma_dim: int = 32
    h_nwp_dim: int = 32


DATA_CONFIG = DataConfig()
PATCHTST_CONFIG = PatchTSTConfig()
EXPERT_CONFIG = ExpertConfig()
QUANTILE_CONFIG = QuantileConfig()
EMBEDDING_CONFIG = EmbeddingConfig()
