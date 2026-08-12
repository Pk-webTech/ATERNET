"""
Gate load-balancing penalty.

A softmax gate over experts has a well-known failure mode: it collapses
onto whichever expert is best early in training, that expert then gets
all the gradient and improves fastest, and the collapse becomes
self-reinforcing. The other three experts -- and with them the entire
justification for a mixture architecture -- end up dead weight.

The standard fix (Shazeer et al., 2017) is to penalize the *dispersion*
of the batch-mean gate weight across experts. Here:

    importance_k = mean over batch of gate_weights[:, k]      (sums to 1)
    L_balance    = n_experts * sum_k importance_k^2 - 1        (>= 0)

This is the squared coefficient of variation of the importance vector
up to an affine transform: it is exactly 0 when every expert receives
an equal share of the batch's routing mass, and grows as the gate
concentrates. Subtracting 1 makes the perfectly-balanced case read as
0 rather than as an arbitrary constant, which keeps the logged value
interpretable.

Deliberately weighted very low (see TrainingConfig.w_load_balance):
this is a tie-breaker against degenerate collapse, not a target.
Perfectly uniform routing would defeat the point of having regime
specialists -- the aim is only to keep every expert alive long enough
for the gate's routing decision to be based on genuine competence
differences rather than initialization luck.
"""

import torch


def load_balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """gate_weights: (B, n_experts), rows sum to 1."""
    n_experts = gate_weights.shape[1]
    importance = gate_weights.mean(dim=0)              # (n_experts,), sums to 1
    return n_experts * (importance ** 2).sum() - 1.0
