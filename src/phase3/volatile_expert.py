"""
Phase 3 -- Volatile expert.

Regime characteristics: the most heterogeneous regime in the dataset --
rapid cloud-edge transitions, mixed ramp/drop patterns, highest
short-term rolling std. No single physical prior applies (unlike
dawn_ramp's monotonic rise or overcast's low ceiling), so this expert
gets the *widest* capacity (512->256->128) of the four: it has to cover
the largest, least-structured slice of pattern-space.

Architecture choices:
  - MSE loss, no auxiliary constraint -- deliberately unconstrained
    since imposing a shape prior on a regime defined by having no
    consistent shape would hurt more than help.
  - Higher dropout than the other experts, as the largest-capacity
    network is the most prone to overfitting the volatile regime's
    comparatively smaller, noisier sample count.
"""

import torch.nn as nn

from src.phase3.expert_base import ExpertBase


class VolatileExpert(ExpertBase):
    def __init__(self, in_dim: int, horizon: int, dropout: float = 0.2):
        super().__init__("volatile", in_dim, horizon)

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, horizon), nn.Sigmoid(),
        )
