"""
Phase 4 -- sigma_NWP extraction.

sigma_NWP must stay a physically-grounded measurement of ECMWF ensemble
spread, not a free-floating learned quantity -- otherwise "NWP forecast
uncertainty" in the routing gate would just be an arbitrary embedding
with no defensible physical meaning, which is exactly the kind of
un-interpretable complexity a reviewer would flag.

So this module is deliberately NOT a neural network: it deterministically
reduces the five ensemble-spread-derived columns of X_sigma (everything
in SIGMA_FEATURES except the expert-disagreement placeholder) at the
forecast issue time (last window step) into a single non-negative scalar
per sample.

    sigma_nwp = mean_v ( sigma_nwp_v )   for v in {ghi, cloud, temp, wind, ensemble_spread_norm}

All five inputs are already min-max scaled to a comparable [0,1]-ish
range by the Phase 1 scaler (fit on train only), so a plain mean is a
defensible aggregation -- no additional learned weighting is introduced
here. If per-variable weighting is later justified empirically, it
belongs in uncertainty_fusion.py where it can be validated against
calibration metrics, not silently baked into "the" NWP uncertainty.
"""

import torch
import torch.nn as nn

from src.utils.feature_groups import SIGMA_FEATURES

# All sigma feature columns except the expert-disagreement placeholder
# (that placeholder is overwritten with real sigma_expert by the Phase 3
# expert pool at training time, not consumed by this module).
_NWP_SIGMA_COLS = [c for c in SIGMA_FEATURES if c != "sigma_expert_placeholder"]
_NWP_SIGMA_IDX = [SIGMA_FEATURES.index(c) for c in _NWP_SIGMA_COLS]


class SigmaNWPExtractor(nn.Module):
    """Stateless (no learned parameters) -- kept as an nn.Module only so it
    composes cleanly inside AdaptiveRouter's forward graph."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_idx", torch.tensor(_NWP_SIGMA_IDX, dtype=torch.long), persistent=False)

    def forward(self, x_sigma: torch.Tensor) -> torch.Tensor:
        """
        x_sigma: (B, T, len(SIGMA_FEATURES)) raw window of sigma features.
        Returns: (B,) non-negative scalar sigma_nwp.
        """
        last_step = x_sigma[:, -1, :]                      # (B, F)
        selected = last_step.index_select(dim=1, index=self._idx)  # (B, 5)
        sigma_nwp = selected.mean(dim=1)
        return torch.clamp(sigma_nwp, min=0.0)
