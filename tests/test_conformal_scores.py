"""Point-wise CQR scores: the calibrated interval must actually cover."""

import numpy as np

from src.evaluation.conformal import _conformity_scores


def _case(n=400, h=16, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0.5, 0.1, size=(n, h))
    p50 = np.full((n, h), 0.5)
    return y, p50 - 0.05, p50 + 0.05  # intervals narrower than the noise


def test_pointwise_adjustment_restores_coverage_where_window_mean_does_not():
    y, p10, p90 = _case(seed=0)
    y_new, _, _ = _case(seed=1)

    def adjusted_coverage(score):
        s = _conformity_scores(y, p10, p90, score=score)
        n = len(s)
        q = np.quantile(s, min(np.ceil((n + 1) * 0.8) / n, 1.0))
        return float(((y_new >= p10 - q) & (y_new <= p90 + q)).mean())

    assert abs(adjusted_coverage("pointwise") - 0.80) < 0.03
    # the window-mean score under-adjusts on the same data
    assert adjusted_coverage("window_mean") < 0.75


def test_unknown_score_rejected():
    y, p10, p90 = _case(n=5)
    try:
        _conformity_scores(y, p10, p90, score="nope")
        assert False
    except ValueError:
        pass
