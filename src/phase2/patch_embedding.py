"""
Phase 2 -- Patch embedding.

Splits a (B, T, F) time series into overlapping patches along the time
axis and linearly projects each flattened patch into d_model. This is
the PatchTST-style tokenization: each patch becomes one "token" for the
transformer encoder, which is what lets the model reason over 4-hour
chunks of history instead of individual 15-minute steps.

n_patches = floor((T - patch_len) / stride) + 1
With T=48, patch_len=8, stride=4 -> n_patches = 11 (each patch spans 2h
of history, consecutive patches overlap by 1h -- enough overlap that a
cloud edge crossing a patch boundary is still fully visible in a
neighbouring patch).
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, n_features: int, patch_len: int, stride: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features

        self.proj = nn.Linear(patch_len * n_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def n_patches(self, seq_len: int) -> int:
        return (seq_len - self.patch_len) // self.stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F) -> (B, n_patches, d_model)
        """
        B, T, F = x.shape
        assert F == self.n_features, f"Expected {self.n_features} features, got {F}"

        # unfold along the time dimension: (B, n_patches, F, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # x.unfold puts the unfolded window as the LAST dim -> (B, n_patches, F, patch_len)
        patches = patches.permute(0, 1, 3, 2).contiguous()  # (B, n_patches, patch_len, F)
        patches = patches.view(B, patches.shape[1], self.patch_len * F)

        out = self.proj(patches)
        out = self.norm(out)
        out = self.dropout(out)
        return out
