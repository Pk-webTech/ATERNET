"""
Phase 2 -- Auxiliary feature encoders.

Two distinct encoders, each justified by what they must preserve:

1. NWPCrossAttention -- the PV/clear-sky patch tokens attend to a
   separately-patchified NWP token sequence. Cross-attention (not
   concatenation) lets each PV patch independently decide how much
   weather context it needs; a cloud-heavy patch should weight NWP
   context differently than a stable clear-sky patch.

2. SigmaEncoder -- a small MLP applied to the raw sigma feature vector
   at the forecast issue time (last window step), NOT routed through
   the transformer. sigma_nwp values are already calibrated physical
   ensemble-spread statistics; passing them through several transformer
   layers would let the model arbitrarily rescale/distort a quantity
   that needs to stay interpretable and well-behaved for the Phase 4
   uncertainty gate.
"""

import torch
import torch.nn as nn

from src.phase2.patch_embedding import PatchEmbedding


class NWPCrossAttention(nn.Module):
    def __init__(self, n_nwp_features: int, patch_len: int, stride: int,
                 d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.nwp_patch_embed = PatchEmbedding(n_nwp_features, patch_len, stride, d_model, dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_patches: torch.Tensor, x_nwp: torch.Tensor):
        """
        seq_patches: (B, n_patches, d_model)  -- query (PV/clear-sky patches)
        x_nwp:       (B, T, n_nwp_features)    -- raw NWP sequence, patchified here
        returns: (fused (B, n_patches, d_model), nwp_patches, attn_weights)
        """
        nwp_patches = self.nwp_patch_embed(x_nwp)
        attn_out, attn_weights = self.cross_attn(
            query=seq_patches, key=nwp_patches, value=nwp_patches, need_weights=True
        )
        fused = self.norm(seq_patches + self.dropout(attn_out))
        return fused, nwp_patches, attn_weights


class SigmaEncoder(nn.Module):
    def __init__(self, n_sigma_features: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_sigma_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_sigma: torch.Tensor) -> torch.Tensor:
        """
        x_sigma: (B, T, n_sigma_features) -- uses the LAST time step only
        (the forecast issue time -- the only point at which "current"
        uncertainty is meaningfully defined). Returns (B, out_dim).
        """
        last_step = x_sigma[:, -1, :]
        return self.norm(self.mlp(last_step))
