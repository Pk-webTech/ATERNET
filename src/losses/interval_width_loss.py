"""
Interval sharpness regularizer -- the companion term to pinball loss on
P10/P90.

Pinball loss on P10/P90 alone pushes toward correct *coverage* but has
no pressure toward *narrow* intervals: a model could satisfy coverage
trivially by predicting arbitrarily wide bounds. This term penalizes
total interval width directly, so training balances coverage (pinball
loss on p10/p90) against sharpness (this term) -- together they
approximate the interval (Winkler) score, a proper scoring rule for
prediction intervals, without having to switch away from the
already-standard pinball loss used for P50.
"""

import torch


def interval_width_penalty(lower_width: torch.Tensor, upper_width: torch.Tensor) -> torch.Tensor:
    """lower_width, upper_width: (B, horizon), both >= 0 by construction."""
    return (lower_width + upper_width).mean()


def winkler_score(y_true: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
                  alpha: float = 0.2) -> torch.Tensor:
    """
    Winkler / interval score for a central (1-alpha) interval.

        W = (u - l) + (2/alpha)*(l - y)*1[y < l] + (2/alpha)*(y - u)*1[y > u]

    Reported at evaluation time as the single proper scoring rule that
    jointly grades coverage and sharpness -- the thing the pinball +
    width-penalty combination is approximating during training. Lower is
    better. alpha=0.2 corresponds to the P10/P90 (80% central) interval.
    """
    width = upper - lower
    below = torch.clamp(lower - y_true, min=0.0)
    above = torch.clamp(y_true - upper, min=0.0)
    return (width + (2.0 / alpha) * (below + above)).mean()
