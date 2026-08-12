"""
Phase 4 -- Uncertainty-aware gating network.

    Gate = softmax( logits(h_expert, sigma_features, h_sigma) / T(sigma_total) )

Two components:

1. Routing logits: an MLP over [h_expert, h_sigma, sigma_nwp_norm,
   sigma_expert_norm, alpha] produces one logit per expert. This is the
   standard "which regime does this look like" signal, now additionally
   informed by both uncertainty sources and their fused mixing weight --
   a sample where sigma_expert is high but sigma_nwp is low, for
   instance, should route differently than one where both are low.

2. Uncertainty-adaptive temperature:
       T = exp( clamp(softplus(beta) * sigma_total, +/- MAX_LOG_T) )
   sigma_total is 0-centered (see UncertaintyFusion), so T = 1 when a
   sample's combined uncertainty sits at the average, T > 1 (softer,
   more uniform routing -- hedge across experts) when uncertainty is
   above average, and T < 1 (sharper, more confident routing) when below.
   softplus(beta) keeps the learned sensitivity strictly positive so
   temperature can only move in the direction increasing uncertainty ->
   increasing entropy, never the reverse (which would be physically
   backwards).

   The clamp on the *exponent* is a numerical guard, not a change of
   behaviour: exp() of a large positive argument overflows to inf
   (softmax then returns a uniform 1/n, silently), and of a large
   negative argument underflows to 0, making logits/T a division by
   zero -> NaN. Clamping the exponent to +/-8 bounds T to [3.4e-4, 2981],
   far wider than any temperature that is behaviourally distinguishable,
   while guaranteeing the softmax is always well-defined.
"""

import torch
import torch.nn as nn

# Bound on |log T|. See module docstring.
MAX_LOG_TEMPERATURE = 8.0


class GatingNetwork(nn.Module):
    def __init__(self, h_expert_dim: int = 128, h_sigma_dim: int = 32,
                 n_experts: int = 4, hidden_dim: int = 64, dropout: float = 0.1,
                 init_beta: float = 1.0):
        super().__init__()
        gate_input_dim = h_expert_dim + h_sigma_dim + 3  # + [sigma_nwp_norm, sigma_expert_norm, alpha]

        self.logit_net = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_experts),
        )
        self.raw_beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.n_experts = n_experts

    def forward(self, h_expert: torch.Tensor, h_sigma: torch.Tensor,
                fusion_out: dict) -> dict:
        """
        h_expert: (B, h_expert_dim), h_sigma: (B, h_sigma_dim)
        fusion_out: dict from UncertaintyFusion.forward()

        Returns dict with gate_weights (B, n_experts), temperature (B,),
        entropy (B,) -- entropy is a useful diagnostic exposed for
        evaluation/interpretability plots.
        """
        sigma_nwp_norm = fusion_out["sigma_nwp_norm"]
        sigma_expert_norm = fusion_out["sigma_expert_norm"]
        alpha = fusion_out["alpha"]
        sigma_total = fusion_out["sigma_total"]

        gate_input = torch.cat([
            h_expert, h_sigma,
            sigma_nwp_norm.unsqueeze(1), sigma_expert_norm.unsqueeze(1), alpha.unsqueeze(1),
        ], dim=1)

        logits = self.logit_net(gate_input)  # (B, n_experts)

        beta = nn.functional.softplus(self.raw_beta)
        log_t = torch.clamp(beta * sigma_total, -MAX_LOG_TEMPERATURE, MAX_LOG_TEMPERATURE)
        temperature = torch.exp(log_t)  # (B,), strictly positive and finite

        gate_weights = torch.softmax(logits / temperature.unsqueeze(1), dim=1)

        with torch.no_grad():
            p = gate_weights.clamp_min(1e-8)
            entropy = -(p * p.log()).sum(dim=1)

        return {
            "gate_weights": gate_weights,
            "temperature": temperature,
            "entropy": entropy,
            "logits": logits,
        }
