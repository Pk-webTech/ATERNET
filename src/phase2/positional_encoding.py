"""
Phase 2 -- Positional encoding for patch tokens.

Uses a learnable positional embedding rather than fixed sinusoidal
encoding: with only ~11 patches per window, a learnable table has
plenty of data to fit and can capture the fact that patch position
correlates with time-of-day-relative-to-forecast-issue in a way a
generic sinusoid can't specialize to this dataset's regularity.
"""

import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, n_patches: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_patches, d_model)"""
        assert x.shape[1] == self.pos_embedding.shape[1], (
            f"Positional encoding built for {self.pos_embedding.shape[1]} patches, "
            f"got input with {x.shape[1]} patches."
        )
        x = x + self.pos_embedding
        return self.dropout(x)
