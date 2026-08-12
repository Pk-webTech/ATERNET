"""
Phase 3 -- Clear-sky peak expert.

Regime characteristics: smooth, near-Gaussian, falling-or-flat power
curve around solar noon under stable clear-sky conditions -- the most
predictable regime in the dataset (lowest pre-training validation loss
of all four experts).

Architecture choice: intentionally the *shallowest* expert (128->64).
A regime this well-behaved doesn't need extra capacity; giving it a
deep network would only invite overfitting to noise, and would waste
the parameter budget that the harder regimes need more.
"""

import torch.nn as nn

from src.phase3.expert_base import ExpertBase


class ClearPeakExpert(ExpertBase):
    def __init__(self, in_dim: int, horizon: int, dropout: float = 0.05):
        super().__init__("clear_peak", in_dim, horizon)

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, horizon), nn.Sigmoid(),  # smooth regime -> bounded [0,1] output
        )

    # MSE (inherited primary_loss) and no auxiliary constraint: the regime
    # is well-behaved enough that a physical prior would add nothing.
