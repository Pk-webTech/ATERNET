"""
Phase 3 -- Overcast expert.

Regime characteristics: near-flat, low power output, but the *highest*
skew of any regime (occasional thin-cloud brightening spikes
punctuating an otherwise flat, low baseline). This regime also
dominates the raw sample count (>60% of all daylight+night steps in
the Phase 1 feature table), so it needs to resolve small values well
without being swamped by the rare spikes.

Architecture choices:
  - Huber loss (delta=0.05) instead of MSE: robust to the occasional
    brightening spike so those outliers don't dominate the gradient
    and drag the flat baseline prediction off target.
  - Output cap at OVERCAST_OUTPUT_CAP (pv_norm), applied via a scaled
    sigmoid.

    IMPORTANT -- the cap is 0.80, not the 0.40 the regime's *current*
    conditions would suggest. The regime label is assigned at the
    forecast issue time (the last step of the lookback window), but the
    expert predicts the following four hours. A window that is overcast
    at issue time frequently clears within that horizon: empirically,
    overcast-labelled training targets average 0.067 but reach 0.746.
    A 0.40 ceiling would therefore make this expert structurally
    incapable of fitting its own regime's targets -- it could never
    predict the clearing, no matter how long it trained, and the error
    would hide inside a merely-mediocre loss curve rather than
    surfacing as a bug.

    The cap that remains is a genuine physical ceiling (nothing in this
    dataset exceeds ~0.76 of nameplate, so 0.80 bounds the output
    without binding on real data) plus a guarantee that the expert
    cannot emit a nonsensical near-nameplate forecast from an overcast
    starting state. `verify_cap()` below is called by the test suite
    against the real Phase 1 targets so this constant can never again
    drift out of agreement with the data it constrains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.phase3.expert_base import ExpertBase, _weighted_reduce

# See the module docstring: this bounds the expert's output at a true
# physical ceiling, and is deliberately NOT set to the typical overcast
# generation level, because the 4h horizon routinely leaves the regime.
OVERCAST_OUTPUT_CAP = 0.80


class OvercastExpert(ExpertBase):
    def __init__(self, in_dim: int, horizon: int, dropout: float = 0.1, huber_delta: float = 0.05):
        super().__init__("overcast", in_dim, horizon)
        self.huber_delta = huber_delta

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, horizon),
        )

    def forward(self, h_expert: torch.Tensor) -> torch.Tensor:
        raw = self.net(h_expert)
        return torch.sigmoid(raw) * OVERCAST_OUTPUT_CAP

    def primary_loss(self, y_hat: torch.Tensor, y_true: torch.Tensor,
                     sample_weight: torch.Tensor = None) -> torch.Tensor:
        return _weighted_reduce(
            F.huber_loss(y_hat, y_true, delta=self.huber_delta, reduction="none"),
            sample_weight,
        )


def verify_cap(y_overcast, cap: float = OVERCAST_OUTPUT_CAP, quantile: float = 0.999):
    """
    Check the hard cap against real overcast-regime targets.

    Returns (ok, observed_quantile, observed_max). Uses a high quantile
    rather than the raw max so a single mislabeled window doesn't fail
    the check, but reports both.
    """
    import numpy as np
    y = np.asarray(y_overcast).ravel()
    if y.size == 0:
        return True, 0.0, 0.0
    q = float(np.quantile(y, quantile))
    return bool(q <= cap), q, float(y.max())
