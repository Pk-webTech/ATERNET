"""
Phase 5 -- Adaptive interval-width head.

Predicts asymmetric, strictly non-negative half-widths for the P10/P90
bounds around P50, as a function of the temporal embedding (h_expert)
and sigma_total (Phase 4's fused dual-uncertainty signal). Widths are
asymmetric because solar power residual distributions are NOT
symmetric -- e.g. in the overcast regime, output is floored near 0 but
occasional thin-cloud brightening can push well above the median, so
the upper tail is heavier than the lower tail.

Per horizon step h:
    width_h = softplus( base_h + softplus(gamma_h) * sigma_total )

gamma_h is passed through softplus so it is strictly >= 0, which
guarantees d(width_h)/d(sigma_total) >= 0 everywhere: width can only
GROW as combined uncertainty rises above the batch average (sigma_total
is 0-centered by Phase 4's BatchNorm) and can only SHRINK as it falls
below average. This is the direct architectural encoding of "the
interval should widen when the model is less certain" -- enforced by
construction, not left for training to discover on its own.

Why interval-width (not per-quantile dynamic movement): predicting P10
and P90 as two independent heads (or worse, moving quantile levels
dynamically) gives no structural guarantee against quantile crossing.
Parameterizing as median +/- a non-negative width makes crossing
mathematically impossible rather than merely discouraged by a penalty
-- this is the concrete fix for the flaw identified in the original
Dual-Uncertainty Routing critical review.

NOTE on the context input: sigma_total is fed to the base MLP as well
as multiplying gamma. The monotonicity guarantee above is a statement
about the *gamma path only*; the MLP path is free to move base_h in
either direction with sigma_total. To keep the architectural guarantee
literally true end-to-end, sigma_total is detached on the MLP path --
the network still sees it as context, but cannot use it to construct a
compensating negative gradient that cancels the gamma term.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntervalWidthHead(nn.Module):
    def __init__(self, h_expert_dim: int = 128, horizon: int = 16, hidden_dim: int = 64):
        super().__init__()
        ctx_dim = h_expert_dim + 1

        self.lower_net = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, horizon)
        )
        self.upper_net = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, horizon)
        )
        # Per-horizon-step sensitivity to sigma_total, constrained >= 0 via softplus.
        self.raw_gamma_lower = nn.Parameter(torch.ones(horizon))
        self.raw_gamma_upper = nn.Parameter(torch.ones(horizon))

    def forward(self, h_expert: torch.Tensor, sigma_total: torch.Tensor):
        """
        h_expert: (B, h_expert_dim), sigma_total: (B,)
        Returns (lower_width, upper_width), each (B, horizon), both >= 0.
        """
        # Detached on the context path so the monotonicity guarantee holds
        # end-to-end (see module docstring).
        ctx = torch.cat([h_expert, sigma_total.detach().unsqueeze(1)], dim=1)
        base_lower = self.lower_net(ctx)   # (B, horizon)
        base_upper = self.upper_net(ctx)   # (B, horizon)

        gamma_lower = F.softplus(self.raw_gamma_lower).unsqueeze(0)  # (1, horizon)
        gamma_upper = F.softplus(self.raw_gamma_upper).unsqueeze(0)

        sigma_col = sigma_total.unsqueeze(1)  # (B, 1)

        lower_width = F.softplus(base_lower + gamma_lower * sigma_col)
        upper_width = F.softplus(base_upper + gamma_upper * sigma_col)
        return lower_width, upper_width
