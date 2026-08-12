"""Losses and metrics: correctness against hand-computable cases."""

import numpy as np
import torch

from src.evaluation.metrics import (
    mae, rmse, picp, pinaw, winkler, pinball, all_metrics, skill_score,
)
from src.losses.interval_width_loss import interval_width_penalty
from src.losses.load_balance_loss import load_balance_loss
from src.losses.pinball_loss import pinball_loss, multi_quantile_pinball_loss
from src.losses.composite_loss import expert_loss_weight


def test_pinball_is_zero_on_perfect_prediction():
    y = torch.rand(8, 16)
    for tau in (0.1, 0.5, 0.9):
        assert pinball_loss(y, y, tau).item() == 0.0


def test_pinball_asymmetry_matches_tau():
    """tau=0.9 must punish under-prediction more than over-prediction."""
    y = torch.ones(4, 4)
    under = pinball_loss(y - 0.1, y, 0.9)
    over = pinball_loss(y + 0.1, y, 0.9)
    assert under > over
    # and tau=0.1 the other way around
    assert pinball_loss(y - 0.1, y, 0.1) < pinball_loss(y + 0.1, y, 0.1)


def test_multi_quantile_total_is_the_sum():
    q = {k: torch.rand(4, 8) for k in ("p10", "p50", "p90")}
    y = torch.rand(4, 8)
    out = multi_quantile_pinball_loss(q, y)
    assert torch.allclose(out["total"], out["p10"] + out["p50"] + out["p90"])


def test_width_penalty_rewards_narrower():
    narrow = torch.full((4, 8), 0.05)
    wide = torch.full((4, 8), 0.5)
    assert interval_width_penalty(narrow, narrow) < interval_width_penalty(wide, wide)


def test_load_balance_zero_when_uniform():
    uniform = torch.full((32, 4), 0.25)
    assert abs(load_balance_loss(uniform).item()) < 1e-6


def test_load_balance_positive_when_collapsed():
    collapsed = torch.zeros(32, 4); collapsed[:, 0] = 1.0
    assert load_balance_loss(collapsed).item() > 1.0


def test_expert_weight_schedule_decays_to_target():
    from config.training_config import TRAINING_CONFIG
    assert expert_loss_weight(0) == 1.0
    assert expert_loss_weight(TRAINING_CONFIG.expert_warmup_epochs - 1) == 1.0
    late = expert_loss_weight(1000)
    assert abs(late - TRAINING_CONFIG.w_expert) < 1e-9


def test_picp_hand_computed():
    y = np.array([[0.5, 0.5, 0.5, 0.5]])
    lo = np.array([[0.0, 0.0, 0.6, 0.6]])   # last two exclude y
    up = np.array([[1.0, 1.0, 0.9, 0.9]])
    assert picp(y, lo, up) == 0.5


def test_pinaw_is_mean_width():
    lo = np.zeros((3, 4)); up = np.full((3, 4), 0.4)
    assert abs(pinaw(lo, up) - 0.4) < 1e-12


def test_winkler_penalizes_misses():
    y = np.array([[0.5]])
    covered = winkler(y, np.array([[0.4]]), np.array([[0.6]]), alpha=0.2)
    missed = winkler(y, np.array([[0.7]]), np.array([[0.9]]), alpha=0.2)
    assert missed > covered


def test_winkler_prefers_sharp_when_both_cover():
    y = np.array([[0.5]])
    sharp = winkler(y, np.array([[0.45]]), np.array([[0.55]]), alpha=0.2)
    loose = winkler(y, np.array([[0.0]]), np.array([[1.0]]), alpha=0.2)
    assert sharp < loose


def test_mae_rmse_hand_computed():
    y = np.array([[0.0, 0.0]]); p = np.array([[0.1, 0.3]])
    assert abs(mae(y, p) - 0.2) < 1e-12
    assert abs(rmse(y, p) - np.sqrt(0.05)) < 1e-12


def test_skill_score_signs():
    y = np.zeros((4, 4))
    good = np.full((4, 4), 0.01)
    bad = np.full((4, 4), 0.5)
    assert skill_score(y, good, bad) > 0
    assert skill_score(y, bad, good) < 0


def test_all_metrics_keys_present():
    y = np.random.rand(20, 16)
    p50 = y + np.random.randn(20, 16) * 0.01
    m = all_metrics(y, p50 - 0.1, p50, p50 + 0.1, y_baseline=np.zeros_like(y))
    for k in ("mae", "rmse", "picp", "pinaw", "winkler", "ace",
              "pinball_total", "rmse_skill_vs_persistence"):
        assert k in m and np.isfinite(m[k])
