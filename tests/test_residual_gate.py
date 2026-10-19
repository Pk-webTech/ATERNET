"""Residual (ReZero-style) gate: opt-in, starts uniform, default unchanged."""

import torch

from src.phase4.gating_network import GatingNetwork


def _fusion_out(B=6):
    return {
        "sigma_nwp_norm": torch.randn(B), "sigma_expert_norm": torch.randn(B),
        "alpha": torch.rand(B), "sigma_total": torch.randn(B),
    }


def test_residual_gate_starts_exactly_uniform():
    gate = GatingNetwork(residual=True).eval()
    out = gate(torch.randn(6, 128), torch.randn(6, 32), _fusion_out())
    assert torch.allclose(out["gate_weights"], torch.full((6, 4), 0.25))


def test_residual_gate_delta_gets_gradient_and_can_leave_uniform():
    gate = GatingNetwork(residual=True)
    out = gate(torch.randn(6, 128), torch.randn(6, 32), _fusion_out())
    # Loss that prefers expert 0 -> nonzero gradient on delta at delta=0.
    out["gate_weights"][:, 0].sum().backward()
    assert gate.delta.grad is not None and gate.delta.grad.abs() > 0

    with torch.no_grad():
        gate.delta.fill_(1.0)
    out = gate.eval()(torch.randn(6, 128), torch.randn(6, 32), _fusion_out())
    assert not torch.allclose(out["gate_weights"], torch.full((6, 4), 0.25))
    assert torch.allclose(out["gate_weights"].sum(dim=1), torch.ones(6), atol=1e-5)


def test_default_gate_has_no_delta_so_old_checkpoints_load():
    gate = GatingNetwork()
    assert "delta" not in gate.state_dict()
