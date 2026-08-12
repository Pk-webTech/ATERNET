"""
Phase 5 -- Quantile decoder.

Combines P50 and the two non-negative half-widths into the final
P10/P50/P90 forecast, with quantile ordering guaranteed BY
CONSTRUCTION:

    P10 = clamp(P50 - lower_width, 0, 1)
    P50 = P50
    P90 = clamp(P50 + upper_width, 0, 1)

Since lower_width, upper_width >= 0 (enforced by softplus in
IntervalWidthHead), P10 <= P50 <= P90 holds for every sample and every
horizon step *before* clamping. Clamping to the physical [0, 1] pv_norm
range can compress an interval near the boundary (e.g. deep night, true
value pinned at 0 -- the lower bound can't go negative); since P50 is
itself already clamped into [0, 1] by PredictionHead, clamping the
bounds to the same interval cannot reorder them. That's the correct
physical behaviour, not a bug: a forecast interval for a non-negative
quantity must itself stay non-negative.
"""

import torch
import torch.nn as nn


class QuantileDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p50: torch.Tensor, lower_width: torch.Tensor,
                upper_width: torch.Tensor) -> dict:
        p10 = torch.clamp(p50 - lower_width, 0.0, 1.0)
        p90 = torch.clamp(p50 + upper_width, 0.0, 1.0)
        return {"p10": p10, "p50": p50, "p90": p90}
