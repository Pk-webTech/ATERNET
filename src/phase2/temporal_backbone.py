"""
Phase 2 -- Temporal encoder backbone (PatchTST + NWP fusion).

Full forward path:
  X_seq   (B, 48, 10) --patchify--> (B, 11, d_model) --+pos_enc--
  X_nwp   (B, 48, 4)  --cross-attn--> fused patches
  fused patches --TransformerEncoder--> encoded (B, 11, d_model)
  encoded --mean-pool + Linear--> h_expert (B, 128)   -> feeds Phase 3 experts
  nwp_patches --mean-pool + Linear--> h_nwp (B, 32)    -> feeds Phase 5 decoder
  X_sigma[:, -1, :] --MLP--> h_sigma (B, 32)            -> feeds Phase 4 gate

Three separate output heads (h_expert, h_sigma, h_nwp) rather than one
shared embedding: the three downstream consumers (expert pool, routing
gate, quantile decoder) have different gradient scales and different
information needs, and conflating them into a single vector caused
training instability in earlier iterations of this architecture
(see docs/architecture.md).
"""

import torch
import torch.nn as nn

from config.model_config import PATCHTST_CONFIG, DATA_CONFIG, EMBEDDING_CONFIG
from src.phase2.patch_embedding import PatchEmbedding
from src.phase2.positional_encoding import LearnablePositionalEncoding
from src.phase2.transformer_encoder import TransformerEncoder
from src.phase2.feature_encoder import NWPCrossAttention, SigmaEncoder


class TemporalBackbone(nn.Module):
    def __init__(
        self,
        n_seq_features: int,
        n_nwp_features: int,
        n_sigma_features: int,
        window_size: int = DATA_CONFIG.window_size,
        patch_len: int = PATCHTST_CONFIG.patch_len,
        stride: int = PATCHTST_CONFIG.stride,
        d_model: int = PATCHTST_CONFIG.d_model,
        n_heads: int = PATCHTST_CONFIG.n_heads,
        n_layers: int = PATCHTST_CONFIG.n_layers,
        d_ff: int = PATCHTST_CONFIG.d_ff,
        dropout: float = PATCHTST_CONFIG.dropout,
        h_expert_dim: int = EMBEDDING_CONFIG.h_expert_dim,
        h_sigma_dim: int = EMBEDDING_CONFIG.h_sigma_dim,
        h_nwp_dim: int = EMBEDDING_CONFIG.h_nwp_dim,
    ):
        super().__init__()

        self.seq_patch_embed = PatchEmbedding(n_seq_features, patch_len, stride, d_model, dropout)
        n_patches = self.seq_patch_embed.n_patches(window_size)

        self.pos_encoding = LearnablePositionalEncoding(n_patches, d_model, dropout)
        self.nwp_cross_attn = NWPCrossAttention(n_nwp_features, patch_len, stride, d_model, n_heads, dropout)
        self.transformer = TransformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.sigma_encoder = SigmaEncoder(n_sigma_features, hidden_dim=64, out_dim=h_sigma_dim, dropout=dropout)

        self.expert_head = nn.Sequential(
            nn.Linear(d_model, h_expert_dim),
            nn.GELU(),
            nn.LayerNorm(h_expert_dim),
        )
        self.nwp_head = nn.Sequential(
            nn.Linear(d_model, h_nwp_dim),
            nn.GELU(),
            nn.LayerNorm(h_nwp_dim),
        )

        self.n_patches = n_patches
        self.d_model = d_model

    def forward(self, x_seq: torch.Tensor, x_nwp: torch.Tensor, x_sigma: torch.Tensor,
                return_attention: bool = False):
        """
        x_seq:   (B, window_size, n_seq_features)
        x_nwp:   (B, window_size, n_nwp_features)
        x_sigma: (B, window_size, n_sigma_features)

        Returns dict with h_expert (B, h_expert_dim), h_sigma (B, h_sigma_dim),
        h_nwp (B, h_nwp_dim), and (optionally) the NWP cross-attention weights
        for interpretability plots (attention_maps output).
        """
        seq_patches = self.seq_patch_embed(x_seq)
        seq_patches = self.pos_encoding(seq_patches)

        fused_patches, nwp_patches, attn_weights = self.nwp_cross_attn(seq_patches, x_nwp)

        encoded = self.transformer(fused_patches)

        pooled_expert = encoded.mean(dim=1)
        pooled_nwp = nwp_patches.mean(dim=1)

        h_expert = self.expert_head(pooled_expert)
        h_nwp = self.nwp_head(pooled_nwp)
        h_sigma = self.sigma_encoder(x_sigma)

        out = {"h_expert": h_expert, "h_sigma": h_sigma, "h_nwp": h_nwp}
        if return_attention:
            out["nwp_attention_weights"] = attn_weights
        return out
