"""
Pinball (quantile) loss -- the natural companion loss for Phase 5's
quantile decoder.

Critically, tau here is FIXED per quantile head (0.1 / 0.5 / 0.9),
decided before training, never moved sample-by-sample. Pinball loss is
only a proper scoring rule for a quantile level fixed in advance -- this
was precisely the mathematical flaw in the original "dynamic quantile
movement" idea (moving tau at inference/training time invalidates the
pinball loss's proper-scoring-rule guarantee). Because Phase 5's
quantile decoder guarantees P10 <= P50 <= P90 by construction (via the
non-negative interval-width parameterization), applying a standard
fixed-tau pinball loss to each of the three outputs is safe and
standard -- no risk of the loss rewarding crossed quantiles.
"""

import torch

DEFAULT_TAUS = (0.1, 0.5, 0.9)
QUANTILE_NAMES = ("p10", "p50", "p90")


def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor, tau: float) -> torch.Tensor:
    """
    y_pred, y_true: (B, horizon). tau: fixed scalar in (0, 1).
    """
    error = y_true - y_pred
    return torch.maximum(tau * error, (tau - 1.0) * error).mean()


def multi_quantile_pinball_loss(quantiles: dict, y_true: torch.Tensor,
                                taus=DEFAULT_TAUS) -> dict:
    """
    quantiles: dict with keys 'p10', 'p50', 'p90' (from ProbabilisticDecoder).
    Returns dict with per-quantile losses and 'total' (their sum).
    """
    per_quantile = {
        name: pinball_loss(quantiles[name], y_true, tau)
        for name, tau in zip(QUANTILE_NAMES, taus)
    }
    total = torch.stack(list(per_quantile.values())).sum()
    return {"total": total, **per_quantile}
