"""
Top-level ATERNET model.

Wires together every phase:
  Phase 2 TemporalBackbone -> Phase 3 ExpertPool -> Phase 4 AdaptiveRouter
  -> Phase 5 ProbabilisticDecoder
into the single class trainers/evaluators import.
"""

import torch
import torch.nn as nn

from config.model_config import EMBEDDING_CONFIG, DATA_CONFIG, EXPERT_CONFIG
from src.phase2.temporal_backbone import TemporalBackbone
from src.phase3.expert_pool import ExpertPool
from src.phase4.adaptive_router import AdaptiveRouter
from src.phase5.probabilistic_decoder import ProbabilisticDecoder


class ATERNET(nn.Module):
    def __init__(self, n_seq_features: int, n_nwp_features: int, n_sigma_features: int,
                 horizon: int = DATA_CONFIG.horizon,
                 h_expert_dim: int = EMBEDDING_CONFIG.h_expert_dim,
                 h_sigma_dim: int = EMBEDDING_CONFIG.h_sigma_dim,
                 h_nwp_dim: int = EMBEDDING_CONFIG.h_nwp_dim,
                 n_experts: int = EXPERT_CONFIG.n_experts):
        super().__init__()
        self.backbone = TemporalBackbone(
            n_seq_features=n_seq_features,
            n_nwp_features=n_nwp_features,
            n_sigma_features=n_sigma_features,
            h_expert_dim=h_expert_dim, h_sigma_dim=h_sigma_dim, h_nwp_dim=h_nwp_dim,
        )
        self.expert_pool = ExpertPool(in_dim=h_expert_dim, horizon=horizon)
        self.router = AdaptiveRouter(h_expert_dim=h_expert_dim, h_sigma_dim=h_sigma_dim,
                                     n_experts=n_experts)
        self.decoder = ProbabilisticDecoder(h_expert_dim=h_expert_dim, h_nwp_dim=h_nwp_dim,
                                            horizon=horizon)
        self.horizon = horizon
        self.n_experts = n_experts

    def forward(self, x_seq: torch.Tensor, x_nwp: torch.Tensor, x_sigma: torch.Tensor,
                return_attention: bool = False) -> dict:
        emb = self.backbone(x_seq, x_nwp, x_sigma, return_attention=return_attention)
        pool_out = self.expert_pool(emb["h_expert"])
        router_out = self.router(
            emb["h_expert"], emb["h_sigma"], x_sigma,
            pool_out["sigma_expert"], pool_out["stacked"],
        )
        quantiles = self.decoder(
            router_out["y_hat_routed"], emb["h_expert"], emb["h_nwp"], router_out["sigma_total"],
        )

        out = {
            **quantiles,
            "y_hat_routed": router_out["y_hat_routed"],
            "gate_weights": router_out["gate_weights"],
            "temperature": router_out["temperature"],
            "entropy": router_out["entropy"],
            "alpha": router_out["alpha"],
            "sigma_nwp": router_out["sigma_nwp"],
            "sigma_expert_calibrated": router_out["sigma_expert_calibrated"],
            "sigma_total": router_out["sigma_total"],
            "expert_predictions": pool_out["predictions"],
            "expert_stacked": pool_out["stacked"],
            # h_expert is needed by the trainer to compute the dense
            # per-expert auxiliary loss without a second backbone pass.
            "h_expert": emb["h_expert"],
            "pool_out": pool_out,
        }
        if return_attention:
            out["nwp_attention_weights"] = emb["nwp_attention_weights"]
        return out
