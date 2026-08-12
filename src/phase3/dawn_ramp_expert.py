"""
Phase 3 -- Dawn/dusk ramp expert.

Regime characteristics (from Phase 1 feature analysis): rising power
curve, positive mean slope, right-skewed distribution -- the hardest
regime because ramp *timing* (exactly when the sun clears the horizon
haze / local obstructions) varies day to day.

Architecture choices:
  - Deepest capacity (256->256->128) of the four experts -- ramp timing
    needs more representational power than a flat regime.
  - Residual + cumsum parameterization: the network predicts a base
    level and a sequence of (signed) increments, and the forecast is
    the cumulative sum of base + increments. This builds the physical
    prior "power should evolve smoothly, not jump" directly into the
    architecture rather than hoping the network learns it from data.
  - Slope auxiliary loss: a soft hinge that penalizes the predicted
    window from having a non-positive mean increment, matching the
    defining physical property of a ramp regime (soft constraint, not
    a hard clamp -- plateaus near the end of the ramp are still allowed).
"""

import torch
import torch.nn as nn

from src.phase3.expert_base import ExpertBase


class DawnRampExpert(ExpertBase):
    def __init__(self, in_dim: int, horizon: int, dropout: float = 0.1):
        super().__init__("dawn_ramp", in_dim, horizon)

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.base_head = nn.Linear(128, 1)
        self.increment_head = nn.Linear(128, horizon)

        # kept for interface consistency; forward() bypasses self.net directly
        self.net = None

    def forward(self, h_expert: torch.Tensor) -> torch.Tensor:
        features = self.trunk(h_expert)
        base = self.base_head(features)                       # (B, 1)
        increments = torch.tanh(self.increment_head(features)) * 0.15  # bounded step size
        y_hat = base + torch.cumsum(increments, dim=1)
        return torch.clamp(y_hat, 0.0, 1.0)

    def auxiliary_loss(self, y_hat: torch.Tensor) -> torch.Tensor:
        diffs = y_hat[:, 1:] - y_hat[:, :-1]
        mean_slope = diffs.mean(dim=1)
        margin = 0.005
        hinge = torch.relu(margin - mean_slope)
        return hinge.mean()
