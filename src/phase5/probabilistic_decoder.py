"""
Phase 5 -- Probabilistic decoder (top-level).

Wires PredictionHead + IntervalWidthHead + QuantileDecoder into a
single module consuming Phase 4's outputs and producing the final
P10/P50/P90 probabilistic forecast.

    y_hat_routed, h_nwp, sigma_total  --PredictionHead-->      p50
    h_expert, sigma_total             --IntervalWidthHead-->   lower_width, upper_width
    p50, lower_width, upper_width     --QuantileDecoder-->     {p10, p50, p90}
"""

import torch
import torch.nn as nn

from src.phase5.prediction_head import PredictionHead
from src.phase5.interval_width_head import IntervalWidthHead
from src.phase5.quantile_decoder import QuantileDecoder


class ProbabilisticDecoder(nn.Module):
    def __init__(self, h_expert_dim: int = 128, h_nwp_dim: int = 32, horizon: int = 16):
        super().__init__()
        self.prediction_head = PredictionHead(h_nwp_dim=h_nwp_dim, horizon=horizon)
        self.interval_width_head = IntervalWidthHead(h_expert_dim=h_expert_dim, horizon=horizon)
        self.quantile_decoder = QuantileDecoder()
        self.horizon = horizon

    def forward(self, y_hat_routed: torch.Tensor, h_expert: torch.Tensor,
                h_nwp: torch.Tensor, sigma_total: torch.Tensor) -> dict:
        """
        Returns dict: p10, p50, p90 (each (B, horizon)), plus
        lower_width/upper_width for loss terms and interpretability plots.
        """
        p50 = self.prediction_head(y_hat_routed, h_nwp, sigma_total)
        lower_width, upper_width = self.interval_width_head(h_expert, sigma_total)
        quantiles = self.quantile_decoder(p50, lower_width, upper_width)
        quantiles["lower_width"] = lower_width
        quantiles["upper_width"] = upper_width
        return quantiles
