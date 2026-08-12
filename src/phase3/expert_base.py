"""
Phase 3 -- Expert base class.

Common interface every regime expert implements. Each subclass owns its
own MLP capacity (depth/width chosen per-regime -- see individual files
for the data-driven justification) and may override `auxiliary_loss` to
encode a physical prior specific to its regime. `primary_loss` is also
overridable since regimes with different residual distributions call
for different loss functions (e.g. Huber for the near-zero, spiky
overcast regime vs plain MSE for the well-behaved clear-peak regime).
"""

import torch
import torch.nn as nn


class ExpertBase(nn.Module):
    def __init__(self, name: str, in_dim: int, horizon: int):
        super().__init__()
        self.name = name
        self.in_dim = in_dim
        self.horizon = horizon
        self.net = None  # built by subclass __init__

    def forward(self, h_expert: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            raise NotImplementedError(f"{self.__class__.__name__} must build self.net")
        return self.net(h_expert)

    def auxiliary_loss(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Optional physical-prior regularizer. Default: no-op."""
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)

    def primary_loss(self, y_hat: torch.Tensor, y_true: torch.Tensor,
                     sample_weight: torch.Tensor = None) -> torch.Tensor:
        return _weighted_reduce(
            nn.functional.mse_loss(y_hat, y_true, reduction="none"), sample_weight
        )

    def total_loss(self, y_hat: torch.Tensor, y_true: torch.Tensor,
                   aux_weight: float = 0.1, sample_weight: torch.Tensor = None) -> torch.Tensor:
        return (self.primary_loss(y_hat, y_true, sample_weight)
                + aux_weight * self.auxiliary_loss(y_hat))


def _weighted_reduce(per_element: torch.Tensor, sample_weight: torch.Tensor = None) -> torch.Tensor:
    """
    Reduce a (B, horizon) per-element loss to a scalar.

    `sample_weight` is (B,) and non-negative. It exists so the trainer can
    optionally emphasise samples belonging to an expert's own regime
    without changing the loss's scale: the result is a weighted *mean*
    (normalized by the weight sum), not a weighted sum, so the gradient
    magnitude doesn't silently depend on how many samples happened to
    match the regime in a given batch.
    """
    if sample_weight is None:
        return per_element.mean()
    w = sample_weight.view(-1, 1)
    return (per_element * w).sum() / w.sum().clamp_min(1e-8) / per_element.shape[1]
