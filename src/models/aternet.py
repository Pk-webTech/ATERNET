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
                 n_experts: int = EXPERT_CONFIG.n_experts,
                 residual_gate: bool = False,
                 last_step_skip: bool = False,
                 n_meta_features: int = 0):
        super().__init__()
        self.backbone = TemporalBackbone(
            n_seq_features=n_seq_features,
            n_nwp_features=n_nwp_features,
            n_sigma_features=n_sigma_features,
            h_expert_dim=h_expert_dim, h_sigma_dim=h_sigma_dim, h_nwp_dim=h_nwp_dim,
        )
        self.expert_pool = ExpertPool(in_dim=h_expert_dim, horizon=horizon)
        self.router = AdaptiveRouter(h_expert_dim=h_expert_dim, h_sigma_dim=h_sigma_dim,
                                     n_experts=n_experts, residual_gate=residual_gate)
        self.decoder = ProbabilisticDecoder(h_expert_dim=h_expert_dim, h_nwp_dim=h_nwp_dim,
                                            horizon=horizon)
        self.horizon = horizon
        self.n_experts = n_experts

        # Optional zero-initialised linear shortcut from the latest observed
        # step and the forecast-horizon NWP to the routed forecast. The deep
        # path then only has to learn the nonlinear correction on top of a
        # linear nowcast. Zero init => identical to the plain model at step 0.
        # n_meta_features > 0 additionally feeds the routing/volatility
        # metadata (ramp rate, rolling std, ...) into that shortcut -- the
        # same signals the LightGBM baseline gets as direct inputs.
        if n_meta_features and not last_step_skip:
            raise ValueError("n_meta_features requires last_step_skip=True.")
        self.n_meta_features = n_meta_features
        self.last_step_skip = None
        if last_step_skip:
            self.last_step_skip = nn.Linear(
                n_seq_features + horizon * n_nwp_features + n_meta_features, horizon)
            nn.init.zeros_(self.last_step_skip.weight)
            nn.init.zeros_(self.last_step_skip.bias)

    def forward(self, x_seq: torch.Tensor, x_nwp: torch.Tensor, x_sigma: torch.Tensor,
                return_attention: bool = False, meta: torch.Tensor = None) -> dict:
        emb = self.backbone(x_seq, x_nwp, x_sigma, return_attention=return_attention)
        pool_out = self.expert_pool(emb["h_expert"])
        router_out = self.router(
            emb["h_expert"], emb["h_sigma"], x_sigma,
            pool_out["sigma_expert"], pool_out["stacked"],
        )
        y_hat_routed = router_out["y_hat_routed"]
        if self.last_step_skip is not None:
            window = x_seq.shape[1]
            parts = [x_seq[:, -1, :], x_nwp[:, window:, :].flatten(1)]
            if self.n_meta_features:
                if meta is None:
                    raise ValueError("This model was built with n_meta_features>0; "
                                     "pass meta=<(B, n_meta_features) tensor>.")
                parts.append(meta)
            skip_in = torch.cat(parts, dim=1)
            y_hat_routed = y_hat_routed + self.last_step_skip(skip_in)

        quantiles = self.decoder(
            y_hat_routed, emb["h_expert"], emb["h_nwp"], router_out["sigma_total"],
        )

        out = {
            **quantiles,
            "y_hat_routed": y_hat_routed,
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
