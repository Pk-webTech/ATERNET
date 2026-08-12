"""
Phase 4 -- Adaptive Dual-Uncertainty Router.

Top-level module wiring together everything in this phase:

    X_sigma (raw)          --SigmaNWPExtractor-->      sigma_nwp        (B,)
    sigma_expert (Phase 3) --SigmaExpertCalibrator-->  sigma_expert_cal (B,)
    (sigma_nwp, sigma_expert_cal, h_expert) --UncertaintyFusion--> sigma_total, alpha, ...
    (h_expert, h_sigma, fusion_out) --GatingNetwork--> gate_weights (B, n_experts)
    gate_weights, expert predictions (stacked) --weighted sum--> y_hat_routed (B, horizon)

This module does NOT own the experts themselves (they stay in
src/phase3/expert_pool.py) -- it only consumes the pool's stacked
predictions and detached sigma_expert, keeping Phase 3 and Phase 4
cleanly separated: Phase 3 answers "what does each expert predict and
how much do they disagree", Phase 4 answers "how much should we trust
each expert's opinion right now, given both sources of uncertainty".
"""

import torch
import torch.nn as nn

from src.phase4.sigma_nwp import SigmaNWPExtractor
from src.phase4.sigma_expert import SigmaExpertCalibrator
from src.phase4.uncertainty_fusion import UncertaintyFusion
from src.phase4.gating_network import GatingNetwork


class AdaptiveRouter(nn.Module):
    def __init__(self, h_expert_dim: int = 128, h_sigma_dim: int = 32, n_experts: int = 4):
        super().__init__()
        self.sigma_nwp_extractor = SigmaNWPExtractor()
        self.sigma_expert_calibrator = SigmaExpertCalibrator()
        self.fusion = UncertaintyFusion(h_expert_dim=h_expert_dim)
        self.gate = GatingNetwork(h_expert_dim=h_expert_dim, h_sigma_dim=h_sigma_dim, n_experts=n_experts)
        self.n_experts = n_experts

    def forward(self, h_expert: torch.Tensor, h_sigma: torch.Tensor,
                x_sigma_raw: torch.Tensor, sigma_expert_raw: torch.Tensor,
                expert_predictions_stacked: torch.Tensor) -> dict:
        """
        h_expert: (B, h_expert_dim)                          Phase 2 output
        h_sigma:  (B, h_sigma_dim)                            Phase 2 output
        x_sigma_raw: (B, T, len(SIGMA_FEATURES))              raw Phase 1 window
        sigma_expert_raw: (B,)                                Phase 3 (detached)
        expert_predictions_stacked: (B, n_experts, horizon)   Phase 3 stacked

        Returns dict with y_hat_routed (B, horizon), gate_weights
        (B, n_experts), and every intermediate uncertainty signal for
        diagnostics / loss terms / interpretability plots.
        """
        sigma_nwp = self.sigma_nwp_extractor(x_sigma_raw)
        sigma_expert_cal = self.sigma_expert_calibrator(sigma_expert_raw)

        fusion_out = self.fusion(sigma_nwp, sigma_expert_cal, h_expert)
        gate_out = self.gate(h_expert, h_sigma, fusion_out)

        gate_weights = gate_out["gate_weights"]  # (B, n_experts)
        # weighted aggregation over experts -> (B, horizon)
        y_hat_routed = torch.einsum("be,beh->bh", gate_weights, expert_predictions_stacked)

        return {
            "y_hat_routed": y_hat_routed,
            "gate_weights": gate_weights,
            "temperature": gate_out["temperature"],
            "entropy": gate_out["entropy"],
            "sigma_nwp": sigma_nwp,
            "sigma_expert_calibrated": sigma_expert_cal,
            "sigma_nwp_norm": fusion_out["sigma_nwp_norm"],
            "sigma_expert_norm": fusion_out["sigma_expert_norm"],
            "alpha": fusion_out["alpha"],
            "sigma_total": fusion_out["sigma_total"],
        }
