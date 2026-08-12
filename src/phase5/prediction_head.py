"""
Phase 5 -- P50 (median) prediction head.

Phase 4's y_hat_routed (the gate-weighted aggregation of expert
predictions) is already a reasonable median estimate: every expert was
trained with a symmetric loss (MSE / Huber), whose optimum is the
conditional mean -- which, for the near-symmetric residual distributions
seen in this dataset outside the overcast regime, closely approximates
the conditional median.

This head applies a small, BOUNDED residual correction on top of
y_hat_routed, conditioned on h_nwp (raw weather context the experts
never see directly) and sigma_total (uncertainty context). Bounding the
correction via tanh means a poorly-initialized or noisy correction can't
swamp the already-reasonable expert aggregation; it can only nudge it.
"""

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(self, h_nwp_dim: int = 32, horizon: int = 16,
                 hidden_dim: int = 64, max_correction: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_nwp_dim + 1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, horizon),
        )
        self.max_correction = max_correction

    def forward(self, y_hat_routed: torch.Tensor, h_nwp: torch.Tensor,
                sigma_total: torch.Tensor) -> torch.Tensor:
        """
        y_hat_routed: (B, horizon), h_nwp: (B, h_nwp_dim), sigma_total: (B,)
        Returns p50: (B, horizon), clamped to [0, 1] (pv_norm range).
        """
        ctx = torch.cat([h_nwp, sigma_total.unsqueeze(1)], dim=1)
        residual = torch.tanh(self.net(ctx)) * self.max_correction
        p50 = torch.clamp(y_hat_routed + residual, 0.0, 1.0)
        return p50
