"""Architectural guarantees for Phases 2-5 and the assembled model."""

import numpy as np
import pytest
import torch

from src.models.aternet import ATERNET
from src.phase2.temporal_backbone import TemporalBackbone
from src.phase3.expert_pool import ExpertPool, EXPERT_ORDER
from src.phase3.overcast_expert import OVERCAST_OUTPUT_CAP
from src.phase4.gating_network import GatingNetwork, MAX_LOG_TEMPERATURE
from src.phase4.uncertainty_fusion import UncertaintyFusion
from src.phase5.interval_width_head import IntervalWidthHead
from src.phase5.probabilistic_decoder import ProbabilisticDecoder


# ---------------- Phase 2 ----------------

def test_backbone_output_shapes(fake_batch, dims):
    bb = TemporalBackbone(dims["n_seq"], dims["n_nwp"], dims["n_sigma"])
    out = bb(fake_batch["x_seq"], fake_batch["x_nwp"], fake_batch["x_sigma"])
    assert out["h_expert"].shape == (8, 128)
    assert out["h_sigma"].shape == (8, 32)
    assert out["h_nwp"].shape == (8, 32)


def test_patch_count_matches_formula(dims):
    bb = TemporalBackbone(dims["n_seq"], dims["n_nwp"], dims["n_sigma"])
    assert bb.n_patches == (48 - 8) // 4 + 1 == 11


# ---------------- Phase 3 ----------------

def test_overcast_expert_respects_cap():
    e = ExpertPool(in_dim=128, horizon=16).experts["overcast"]
    y = e(torch.randn(64, 128) * 10)  # deliberately extreme input
    assert y.max().item() <= OVERCAST_OUTPUT_CAP + 1e-6


def test_all_expert_outputs_in_physical_range():
    pool = ExpertPool(in_dim=128, horizon=16)
    out = pool(torch.randn(32, 128) * 5)
    for name, pred in out["predictions"].items():
        assert pred.min() >= -1e-6, name
        assert pred.max() <= 1 + 1e-6, name


def test_sigma_expert_is_detached():
    """The gate must not be able to backprop into the experts via sigma_expert."""
    pool = ExpertPool(in_dim=128, horizon=16)
    out = pool(torch.randn(16, 128))
    assert not out["sigma_expert"].requires_grad


def test_sigma_expert_is_a_std_not_a_variance():
    """
    sigma_expert must be the sqrt of mean cross-expert variance. Checked
    directly against a hand-computed value on the pool's own stack.
    """
    pool = ExpertPool(in_dim=128, horizon=16)
    out = pool(torch.randn(16, 128))
    expected = out["stacked"].detach().var(dim=1, unbiased=False).mean(dim=1).sqrt()
    assert torch.allclose(out["sigma_expert"], expected, atol=1e-6)


def test_dawn_ramp_aux_loss_penalizes_falling_ramp():
    e = ExpertPool(in_dim=128, horizon=16).experts["dawn_ramp"]
    rising = torch.linspace(0, 0.8, 16).repeat(4, 1)
    falling = torch.linspace(0.8, 0, 16).repeat(4, 1)
    assert e.auxiliary_loss(falling) > e.auxiliary_loss(rising)


# ---------------- Phase 4 ----------------

def test_gate_weights_sum_to_one(fake_batch):
    gate = GatingNetwork()
    fusion_out = {k: torch.randn(8) for k in
                  ["sigma_nwp_norm", "sigma_expert_norm", "alpha", "sigma_total"]}
    out = gate(torch.randn(8, 128), torch.randn(8, 32), fusion_out)
    assert torch.allclose(out["gate_weights"].sum(dim=1), torch.ones(8), atol=1e-5)


def test_temperature_is_finite_under_extreme_sigma():
    """The clamp guard: exp() must not overflow to inf or underflow to 0."""
    gate = GatingNetwork()
    fusion_out = {
        "sigma_nwp_norm": torch.zeros(4), "sigma_expert_norm": torch.zeros(4),
        "alpha": torch.zeros(4),
        "sigma_total": torch.tensor([-1e6, -50.0, 50.0, 1e6]),
    }
    out = gate(torch.randn(4, 128), torch.randn(4, 32), fusion_out)
    assert torch.isfinite(out["temperature"]).all()
    assert (out["temperature"] > 0).all()
    assert torch.isfinite(out["gate_weights"]).all()
    assert torch.allclose(out["gate_weights"].sum(dim=1), torch.ones(4), atol=1e-5)


def test_temperature_increases_with_uncertainty():
    gate = GatingNetwork()
    def temp(s):
        fo = {"sigma_nwp_norm": torch.zeros(2), "sigma_expert_norm": torch.zeros(2),
              "alpha": torch.zeros(2), "sigma_total": torch.tensor([s, s])}
        return gate(torch.randn(2, 128), torch.randn(2, 32), fo)["temperature"][0].item()
    assert temp(1.0) > temp(0.0) > temp(-1.0)


def test_alpha_is_a_convex_weight():
    f = UncertaintyFusion()
    out = f(torch.rand(16), torch.rand(16), torch.randn(16, 128))
    assert (out["alpha"] >= 0).all() and (out["alpha"] <= 1).all()


def test_fusion_survives_batch_of_one():
    """BatchNorm over a singleton batch would otherwise emit NaN."""
    f = UncertaintyFusion()
    f.train()
    out = f(torch.rand(1), torch.rand(1), torch.randn(1, 128))
    assert torch.isfinite(out["sigma_total"]).all()


# ---------------- Phase 5 ----------------

def test_widths_are_non_negative():
    head = IntervalWidthHead()
    lo, up = head(torch.randn(32, 128) * 10, torch.randn(32) * 10)
    assert (lo >= 0).all() and (up >= 0).all()


def test_width_is_monotone_non_decreasing_in_sigma():
    """The core architectural claim of Phase 5's width head."""
    head = IntervalWidthHead()
    h = torch.randn(16, 128)
    prev_lo = prev_up = None
    for s in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        lo, up = head(h, torch.full((16,), s))
        if prev_lo is not None:
            assert (lo >= prev_lo - 1e-6).all(), f"lower width shrank as sigma rose to {s}"
            assert (up >= prev_up - 1e-6).all(), f"upper width shrank as sigma rose to {s}"
        prev_lo, prev_up = lo, up


def test_non_crossing_guaranteed():
    dec = ProbabilisticDecoder()
    q = dec(torch.rand(64, 16), torch.randn(64, 128) * 5,
            torch.randn(64, 32) * 5, torch.randn(64) * 5)
    assert (q["p10"] <= q["p50"] + 1e-6).all()
    assert (q["p50"] <= q["p90"] + 1e-6).all()


def test_quantiles_stay_in_physical_range():
    dec = ProbabilisticDecoder()
    q = dec(torch.rand(32, 16), torch.randn(32, 128) * 5,
            torch.randn(32, 32) * 5, torch.randn(32) * 5)
    for k in ("p10", "p50", "p90"):
        assert q[k].min() >= 0 and q[k].max() <= 1


# ---------------- Whole model ----------------

def test_full_forward_shapes(model, fake_batch, dims):
    out = model(fake_batch["x_seq"], fake_batch["x_nwp"], fake_batch["x_sigma"])
    for k in ("p10", "p50", "p90", "y_hat_routed"):
        assert out[k].shape == (8, dims["horizon"])
    assert out["gate_weights"].shape == (8, 4)


def test_full_backward_has_no_broken_gradients(model, fake_batch):
    from src.losses.composite_loss import compute_composite_loss
    out = model(fake_batch["x_seq"], fake_batch["x_nwp"], fake_batch["x_sigma"])
    loss = compute_composite_loss(out, fake_batch["y"], model.expert_pool, epoch=0)
    loss["total"].backward()

    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"parameters received no gradient: {missing[:10]}"
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients in: {bad[:10]}"


def test_eval_mode_is_deterministic(model, fake_batch):
    model.eval()
    with torch.no_grad():
        a = model(fake_batch["x_seq"], fake_batch["x_nwp"], fake_batch["x_sigma"])["p50"]
        b = model(fake_batch["x_seq"], fake_batch["x_nwp"], fake_batch["x_sigma"])["p50"]
    assert torch.allclose(a, b)
