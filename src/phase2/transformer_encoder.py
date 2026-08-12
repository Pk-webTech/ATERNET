"""
Phase 2 -- Transformer encoder stack.

Standard pre-norm transformer encoder (multi-head self-attention + GELU
feed-forward), applied over the patch-token sequence produced by
PatchEmbedding. Pre-norm is used instead of post-norm because it trains
more stably at this relatively small scale / short training-run regime
(no need for a learning-rate warmup schedule as strict as post-norm
would require).
"""

import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-block (pre-norm + residual)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.dropout(attn_out)

        # Feed-forward sub-block (pre-norm + residual)
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + self.dropout(ff_out)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
