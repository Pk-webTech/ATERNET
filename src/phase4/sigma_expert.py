"""
Phase 4 -- sigma_Expert calibration.

Phase 3's ExpertPool already computes the raw epistemic-uncertainty
signal (the standard deviation of cross-expert predictions, detached --
see src/phase3/expert_pool.py). That raw signal is in pv_norm units,
while sigma_nwp is an average of min-max-scaled ensemble-spread
features; the two are dimensionally comparable but not guaranteed to
share a numeric range.

This module applies a single learned, strictly positive, zero-preserving
rescaling:

    sigma_expert_calibrated = softplus(scale_param) * sigma_expert_raw

This is intentionally the simplest possible calibration -- a scalar
multiplicative factor, not an MLP -- because anything richer would risk
distorting the *ranking* of samples by expert disagreement, which is
the property that actually matters for routing (a sample where experts
disagree more should always get a higher sigma_expert than one where
they agree more, regardless of calibration). A monotonic scalar rescale
preserves that ranking exactly; an MLP would not be guaranteed to.
"""

import torch
import torch.nn as nn


class SigmaExpertCalibrator(nn.Module):
    def __init__(self, init_scale: float = 1.0):
        super().__init__()
        # softplus(raw_scale) = init_scale  =>  raw_scale = log(exp(init_scale) - 1)
        raw_init = torch.log(torch.expm1(torch.tensor(float(init_scale))))
        self.raw_scale = nn.Parameter(raw_init)

    def forward(self, sigma_expert_raw: torch.Tensor) -> torch.Tensor:
        """sigma_expert_raw: (B,) non-negative. Returns calibrated (B,)."""
        scale = nn.functional.softplus(self.raw_scale)
        return scale * sigma_expert_raw
