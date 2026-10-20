"""Opt-in accuracy options: meta inputs, p50 MSE term, EMA weights."""

import pytest
import torch

from config.training_config import TrainingConfig
from src.losses.composite_loss import compute_composite_loss
from src.models.aternet import ATERNET

KW = dict(n_seq_features=10, n_nwp_features=4, n_sigma_features=6, horizon=16)


def _batch(B=4):
    return torch.randn(B, 48, 10), torch.randn(B, 64, 4), torch.randn(B, 48, 6)


def test_meta_requires_skip():
    with pytest.raises(ValueError):
        ATERNET(**KW, n_meta_features=8)


def test_meta_model_needs_meta_and_is_noop_at_init():
    torch.manual_seed(0)
    plain = ATERNET(**KW, last_step_skip=True).eval()
    meta_model = ATERNET(**KW, last_step_skip=True, n_meta_features=8).eval()
    meta_model.load_state_dict(
        {k: v for k, v in plain.state_dict().items() if not k.startswith("last_step_skip")},
        strict=False)
    x = _batch()
    with pytest.raises(ValueError):
        meta_model(*x)
    with torch.no_grad():
        a = plain(*x)
        b = meta_model(*x, meta=torch.randn(4, 8))
    assert torch.allclose(a["p50"], b["p50"], atol=1e-6)  # zero-init skip


def test_p50_mse_term_changes_loss_only_when_enabled():
    m = ATERNET(**KW)
    out = m(*_batch())
    y = torch.rand(4, 16)
    off = compute_composite_loss(out, y, m.expert_pool, cfg=TrainingConfig())["total"]
    cfg = TrainingConfig()
    cfg.w_p50_mse = 5.0
    on = compute_composite_loss(out, y, m.expert_pool, cfg=cfg)["total"]
    assert on.item() > off.item()
