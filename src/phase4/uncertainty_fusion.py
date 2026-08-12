"""
Phase 4 -- Dual-uncertainty fusion.

Combines sigma_nwp (weather forecast uncertainty) and sigma_expert
(model/epistemic uncertainty) into a single fused uncertainty signal
for the gating network, in three mathematically justified steps:

1. Variance-stabilizing transform: log1p(sigma) for each source. Both
   raw signals are non-negative and right-skewed (most samples have
   low uncertainty, a minority have much higher); log1p compresses the
   long tail so neither source can dominate purely because of scale.

2. Batch normalization (per source, independently): sigma_nwp and
   sigma_expert are measured in genuinely different units (ensemble
   spread vs. cross-expert prediction spread) -- there's no a priori
   reason their log1p-transformed values should sit on comparable
   scales. BatchNorm1d gives each source a learned, data-driven
   standardization so the combination step below isn't dominated by
   whichever source happens to have larger raw magnitude.

   IMPORTANT consequence: sigma_total is 0-centered only in train mode
   and only in expectation over a batch. At eval time BatchNorm uses its
   running statistics, so the "T = 1 at average uncertainty" reading of
   the gating temperature holds against the *training* average, not the
   current batch. That is the intended behaviour (eval must not depend
   on batch composition) but it is why sigma_total must never be
   interpreted as an absolute physical uncertainty -- it is a relative,
   standardized signal.

3. Input-dependent adaptive convex combination:
       alpha = sigmoid( w^T h_expert + b )                    in (0, 1)
       sigma_total = alpha * sigma_nwp_norm + (1-alpha) * sigma_expert_norm
   alpha is predicted from the temporal embedding, NOT from the sigmas
   themselves -- the model learns, from the local weather/generation
   pattern, whether NWP uncertainty or expert disagreement is the more
   trustworthy uncertainty signal in that context. Making alpha a
   convex-combination weight (rather than an unconstrained sum)
   guarantees sigma_total stays bounded by the two inputs and can never
   explode from a runaway learned weight.
"""

import torch
import torch.nn as nn


class UncertaintyFusion(nn.Module):
    def __init__(self, h_expert_dim: int = 128):
        super().__init__()
        self.bn_nwp = nn.BatchNorm1d(1)
        self.bn_expert = nn.BatchNorm1d(1)
        self.alpha_head = nn.Linear(h_expert_dim, 1)

    def forward(self, sigma_nwp: torch.Tensor, sigma_expert: torch.Tensor,
                h_expert: torch.Tensor) -> dict:
        """
        sigma_nwp, sigma_expert: (B,) non-negative scalars
        h_expert: (B, h_expert_dim) temporal embedding from Phase 2

        Returns dict with sigma_nwp_norm, sigma_expert_norm, alpha,
        sigma_total (all (B,)), each usable directly as gate input /
        interpretability output.
        """
        log_nwp = torch.log1p(sigma_nwp).unsqueeze(1)        # (B,1)
        log_expert = torch.log1p(sigma_expert).unsqueeze(1)  # (B,1)

        # BatchNorm1d over a batch of size 1 in train mode has zero variance
        # and would emit NaN. Fall back to the running stats in that case
        # rather than letting a stray odd-sized final batch poison training.
        if self.training and log_nwp.shape[0] < 2:
            was_training = True
            self.bn_nwp.eval()
            self.bn_expert.eval()
        else:
            was_training = False

        sigma_nwp_norm = self.bn_nwp(log_nwp).squeeze(1)
        sigma_expert_norm = self.bn_expert(log_expert).squeeze(1)

        if was_training:
            self.bn_nwp.train()
            self.bn_expert.train()

        alpha = torch.sigmoid(self.alpha_head(h_expert)).squeeze(1)  # (B,)
        sigma_total = alpha * sigma_nwp_norm + (1.0 - alpha) * sigma_expert_norm

        return {
            "sigma_nwp_norm": sigma_nwp_norm,
            "sigma_expert_norm": sigma_expert_norm,
            "alpha": alpha,
            "sigma_total": sigma_total,
        }
