"""Last-step linear shortcut: opt-in, zero-init no-op, trains."""

import torch

from src.models.aternet import ATERNET

KW = dict(n_seq_features=10, n_nwp_features=4, n_sigma_features=6, horizon=16)


def _batch(B=4):
    return torch.randn(B, 48, 10), torch.randn(B, 64, 4), torch.randn(B, 48, 6)


def test_default_has_no_skip_params():
    assert ATERNET(**KW).last_step_skip is None


def test_zero_init_skip_matches_plain_model():
    torch.manual_seed(0)
    plain = ATERNET(**KW).eval()
    skip = ATERNET(**KW, last_step_skip=True).eval()
    missing = skip.load_state_dict(plain.state_dict(), strict=False)
    assert all(k.startswith("last_step_skip") for k in missing.missing_keys)
    x = _batch()
    with torch.no_grad():
        a, b = plain(*x), skip(*x)
    assert torch.allclose(a["p50"], b["p50"], atol=1e-6)


def test_skip_receives_gradient():
    m = ATERNET(**KW, last_step_skip=True)
    out = m(*_batch())
    out["p50"].sum().backward()
    assert m.last_step_skip.weight.grad.abs().sum() > 0
