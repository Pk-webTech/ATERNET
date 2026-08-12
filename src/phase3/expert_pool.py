"""
Phase 3 -- Expert pool.

Wraps the four regime experts, runs all of them on every sample
(dense mixture -- routing/sparsification happens in Phase 4, not here),
and computes sigma_expert: the epistemic uncertainty signal derived
from cross-expert disagreement.

sigma_expert is computed on DETACHED predictions. This is a deliberate
design choice, not an oversight: sigma_expert feeds the Phase 4 routing
gate, and if gradients from the gate's use of sigma_expert flowed back
into the experts, the experts would be incentivized to change their
predictions to manipulate their own uncertainty estimate (e.g. collapse
toward agreement to look confident) rather than to minimize their
actual forecast error. Detaching breaks that feedback loop: experts are
shaped only by their own primary_loss / auxiliary_loss, and
sigma_expert is a pure (if noisy, early in training) measurement of
their disagreement.

UNITS: sigma_expert is a standard deviation (pv_norm units), i.e. the
square root of the mean per-step cross-expert variance -- not the
variance itself. A quantity named "sigma" that is actually a variance
lives on a squared scale, which (a) makes it non-comparable with
sigma_nwp, an average of [0,1]-scaled spread features, and (b) makes
the log1p transform in Phase 4's fusion behave differently than the
docstring there describes. Taking the root is a monotone transform, so
the sample *ranking* by disagreement -- the property Phase 4's
calibrator is designed to preserve -- is unchanged.
"""

from typing import Dict

import torch
import torch.nn as nn

from src.phase3.dawn_ramp_expert import DawnRampExpert
from src.phase3.clear_peak_expert import ClearPeakExpert
from src.phase3.overcast_expert import OvercastExpert
from src.phase3.volatile_expert import VolatileExpert

EXPERT_ORDER = ["dawn_ramp", "clear_peak", "overcast", "volatile"]


class ExpertPool(nn.Module):
    def __init__(self, in_dim: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.experts = nn.ModuleDict({
            "dawn_ramp": DawnRampExpert(in_dim, horizon),
            "clear_peak": ClearPeakExpert(in_dim, horizon),
            "overcast": OvercastExpert(in_dim, horizon),
            "volatile": VolatileExpert(in_dim, horizon),
        })

    def forward(self, h_expert: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        h_expert: (B, in_dim) -- from Phase 2 TemporalBackbone.

        Returns:
          predictions  : dict[name -> (B, horizon)] per-expert forecast
          stacked      : (B, n_experts, horizon), ordered by EXPERT_ORDER
          sigma_expert : (B,) epistemic-uncertainty signal per sample --
                          sqrt of the horizon-mean cross-expert variance,
                          computed on detached predictions (see docstring)
        """
        predictions = {name: expert(h_expert) for name, expert in self.experts.items()}

        stacked = torch.stack([predictions[name] for name in EXPERT_ORDER], dim=1)  # (B,4,H)

        with torch.no_grad():
            detached_stack = stacked.detach()
            per_step_variance = detached_stack.var(dim=1, unbiased=False)  # (B, horizon)
            sigma_expert = per_step_variance.mean(dim=1).clamp_min(0.0).sqrt()  # (B,)

        return {
            "predictions": predictions,
            "stacked": stacked,
            "sigma_expert": sigma_expert,
        }

    def compute_loss(self, h_expert: torch.Tensor, y_true: torch.Tensor,
                     aux_weight: float = 0.1,
                     sample_weights: Dict[str, torch.Tensor] = None,
                     out: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Independent per-expert loss (dense training: every expert is
        trained on every sample at this stage; Phase 4 handles sparse,
        routing-weighted aggregation for the final forecast).

        `sample_weights` optionally maps expert name -> (B,) non-negative
        weights, letting the trainer emphasise each expert's own regime
        without ever fully starving it of gradient.

        `out` lets a caller that has already run forward() pass the result
        back in, so the pool isn't re-run (and dropout re-sampled) just to
        compute a loss over predictions it already holds.
        """
        if out is None:
            out = self.forward(h_expert)

        losses = {}
        for name in EXPERT_ORDER:
            w = None if sample_weights is None else sample_weights.get(name)
            losses[name] = self.experts[name].total_loss(
                out["predictions"][name], y_true, aux_weight, sample_weight=w
            )
        total = torch.stack(list(losses.values())).sum()
        return {"total": total, "per_expert": losses, **out}
