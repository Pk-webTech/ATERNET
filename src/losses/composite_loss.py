"""
The single composite training objective for ATERNET.

    L = w_pinball  * L_pinball(p10, p50, p90)
      + w_width    * L_sharpness(lower_width, upper_width)
      + w_expert   * L_dense_expert(each expert vs y)
      + w_balance  * L_load_balance(gate_weights)

Why each term exists:

* L_pinball is the primary objective. It is the only term that grades
  the model's actual deliverable (a calibrated P10/P50/P90 forecast),
  and it is a proper scoring rule at each fixed tau.

* L_sharpness stops the trivial solution. Pinball on P10/P90 alone is
  minimized by wide-enough intervals in the coverage sense; without a
  width penalty the model has no incentive to be sharp. The two
  together approximate the Winkler interval score.

* L_dense_expert keeps all four experts learning. The routed forecast
  only backpropagates into an expert in proportion to its gate weight,
  so an expert the gate currently distrusts receives almost no gradient
  and can never improve enough to earn trust back. Training each expert
  densely against the target, in parallel with the routed objective,
  breaks that deadlock. Its weight is annealed down after a warmup
  period (see `expert_loss_weight`) -- early on the experts need to be
  competent in their own right; later the routed objective should
  dominate so the model optimizes the forecast it actually emits.

* L_load_balance is a small guard against gate collapse (see
  load_balance_loss.py).

Every term is returned separately as a float in `components` so the
training loop can log them independently -- when a run goes wrong, the
first question is always *which* term moved.
"""

from typing import Dict

import torch

from config.training_config import TRAINING_CONFIG
from src.losses.pinball_loss import multi_quantile_pinball_loss
from src.losses.interval_width_loss import interval_width_penalty
from src.losses.load_balance_loss import load_balance_loss


def expert_loss_weight(epoch: int, cfg=TRAINING_CONFIG) -> float:
    """
    Dense-expert loss weight schedule: held at 1.0 for the first
    `expert_warmup_epochs`, then decayed linearly to `w_expert` over an
    equal-length window and held there.
    """
    warm = max(1, cfg.expert_warmup_epochs)
    if epoch < warm:
        return 1.0
    progress = min(1.0, (epoch - warm) / warm)
    return 1.0 + progress * (cfg.w_expert - 1.0)


def compute_composite_loss(
    model_out: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    expert_pool,
    epoch: int = 0,
    sample_weights: Dict[str, torch.Tensor] = None,
    cfg=TRAINING_CONFIG,
) -> Dict[str, object]:
    """
    model_out: the dict returned by ATERNET.forward()
    y_true:    (B, horizon)
    expert_pool: the model's ExpertPool (for the dense per-expert loss)

    Returns {"total": tensor, "components": {name: float}}.
    """
    pinball = multi_quantile_pinball_loss(model_out, y_true)
    sharpness = interval_width_penalty(model_out["lower_width"], model_out["upper_width"])
    balance = load_balance_loss(model_out["gate_weights"])

    # Reuse the pool output already computed in the forward pass rather
    # than re-running the experts (which would also re-sample dropout and
    # make the two paths inconsistent).
    expert_losses = expert_pool.compute_loss(
        model_out["h_expert"], y_true,
        aux_weight=cfg.aux_weight,
        sample_weights=sample_weights,
        out=model_out["pool_out"],
    )

    w_exp = expert_loss_weight(epoch, cfg)

    total = (
        cfg.w_pinball * pinball["total"]
        + cfg.w_width * sharpness
        + w_exp * expert_losses["total"]
        + cfg.w_load_balance * balance
    )

    components = {
        "loss_total": float(total.detach()),
        "pinball_total": float(pinball["total"].detach()),
        "pinball_p10": float(pinball["p10"].detach()),
        "pinball_p50": float(pinball["p50"].detach()),
        "pinball_p90": float(pinball["p90"].detach()),
        "sharpness": float(sharpness.detach()),
        "load_balance": float(balance.detach()),
        "expert_total": float(expert_losses["total"].detach()),
        "expert_loss_weight": w_exp,
    }
    for name, l in expert_losses["per_expert"].items():
        components[f"expert_{name}"] = float(l.detach())

    return {"total": total, "components": components}
