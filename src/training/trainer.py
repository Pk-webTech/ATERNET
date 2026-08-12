"""
Training loop for ATERNET.

Design notes:

* One optimizer over the whole model. The phases are trained jointly,
  not stage-wise: the router's usefulness depends on the experts being
  competent and vice versa, so freezing either while training the other
  optimizes against a moving target that isn't there yet. The dense
  expert loss (see composite_loss.py) is what makes joint training
  stable without a separate pretraining stage.

* Model selection is on validation *pinball* loss, not the full
  composite loss. The composite total includes the dense-expert term
  whose weight changes across epochs, so comparing composite totals
  between epoch 4 and epoch 12 compares two different objectives.
  Pinball on P10/P50/P90 is fixed for the whole run and is the metric
  that reflects the deliverable.

* Gradient clipping is on by default. The gate temperature is
  exponential in sigma_total, so a large sigma_total early in training
  (before BatchNorm's running stats settle) can produce a large
  gradient through beta; clipping bounds the damage from those spikes.
"""

import math
import time
from typing import Dict

import torch

from config.paths import (
    BEST_CHECKPOINT_PATH, LAST_CHECKPOINT_PATH, TRAIN_HISTORY_PATH, ensure_dirs,
)
from config.training_config import TRAINING_CONFIG
from src.losses.composite_loss import compute_composite_loss
from src.losses.pinball_loss import multi_quantile_pinball_loss
from src.phase3.expert_pool import EXPERT_ORDER
from src.training.checkpoint import save_checkpoint, EarlyStopping
from src.utils.io import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def regime_sample_weights(regime_id: torch.Tensor, n_experts: int = 4,
                          own_regime_weight: float = 3.0,
                          other_weight: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Per-expert (B,) weights that up-weight each expert's own regime.

    Every expert still sees every sample (weight `other_weight` > 0) --
    a hard mask would leave an expert with zero gradient on any batch
    that happens to contain none of its regime, and would make the
    experts useless on exactly the ambiguous samples where the gate
    most needs a second opinion.
    """
    weights = {}
    for k, name in enumerate(EXPERT_ORDER[:n_experts]):
        w = torch.full_like(regime_id, other_weight, dtype=torch.float32)
        w[regime_id == k] = own_regime_weight
        weights[name] = w
    return weights


class Trainer:
    def __init__(self, model, train_loader, val_loader, device=None,
                 cfg=TRAINING_CONFIG, model_kwargs: dict = None,
                 use_regime_weights: bool = True,
                 best_path=BEST_CHECKPOINT_PATH, last_path=LAST_CHECKPOINT_PATH,
                 history_path=TRAIN_HISTORY_PATH):
        # Output paths are injectable so a test (or a hyperparameter sweep)
        # can run a throwaway fit without overwriting the real trained
        # artifact sitting in checkpoints/.
        self.best_path = best_path
        self.last_path = last_path
        self.history_path = history_path
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_kwargs = model_kwargs or {}
        self.use_regime_weights = use_regime_weights

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr * cfg.min_lr_factor
        )
        self.early_stopping = EarlyStopping(patience=cfg.early_stopping_patience)
        self.history = []
        ensure_dirs()

    # ---------------------------------------------------------------- #

    def _batch_to_device(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _forward_loss(self, batch, epoch: int):
        out = self.model(batch["x_seq"], batch["x_nwp"], batch["x_sigma"])
        sw = (regime_sample_weights(batch["regime_id"], self.model.n_experts)
              if self.use_regime_weights else None)
        loss = compute_composite_loss(
            out, batch["y"], self.model.expert_pool,
            epoch=epoch, sample_weights=sw, cfg=self.cfg,
        )
        return out, loss

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        agg, n_batches = {}, 0

        for batch in self.train_loader:
            batch = self._batch_to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            _, loss = self._forward_loss(batch, epoch)
            total = loss["total"]

            if not torch.isfinite(total):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}: {loss['components']}"
                )

            total.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()

            for k, v in loss["components"].items():
                agg[k] = agg.get(k, 0.0) + v
            n_batches += 1

        return {f"train_{k}": v / max(1, n_batches) for k, v in agg.items()}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        agg, n_batches = {}, 0
        gate_mass = torch.zeros(self.model.n_experts, device=self.device)

        for batch in self.val_loader:
            batch = self._batch_to_device(batch)
            out = self.model(batch["x_seq"], batch["x_nwp"], batch["x_sigma"])

            pinball = multi_quantile_pinball_loss(out, batch["y"])
            _, loss = self._forward_loss(batch, epoch)

            comps = dict(loss["components"])
            comps["pinball_total"] = float(pinball["total"])
            for k, v in comps.items():
                agg[k] = agg.get(k, 0.0) + v
            gate_mass += out["gate_weights"].mean(dim=0)
            n_batches += 1

        metrics = {f"val_{k}": v / max(1, n_batches) for k, v in agg.items()}
        share = (gate_mass / max(1, n_batches)).tolist()
        for name, s in zip(EXPERT_ORDER, share):
            metrics[f"val_gate_share_{name}"] = s
        return metrics

    # ---------------------------------------------------------------- #

    def fit(self, epochs: int = None) -> list:
        epochs = epochs or self.cfg.epochs
        logger.info(f"Training on {self.device} for up to {epochs} epochs "
                    f"({sum(p.numel() for p in self.model.parameters()):,} parameters).")

        for epoch in range(epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            self.scheduler.step()

            record = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"],
                      "seconds": round(time.time() - t0, 2),
                      **train_metrics, **val_metrics}
            self.history.append(record)

            val_pinball = val_metrics["val_pinball_total"]
            is_best = self.early_stopping.step(val_pinball)

            gate_str = " ".join(
                f"{n[:4]}={val_metrics[f'val_gate_share_{n}']:.2f}" for n in EXPERT_ORDER
            )
            logger.info(
                f"epoch {epoch:3d} | train {train_metrics['train_loss_total']:.5f} "
                f"| val pinball {val_pinball:.5f} "
                f"| sharp {val_metrics['val_sharpness']:.4f} "
                f"| gate {gate_str} "
                f"| {record['seconds']}s{'  <- best' if is_best else ''}"
            )

            if is_best:
                save_checkpoint(self.best_path, self.model, self.optimizer,
                                self.scheduler, epoch, val_pinball,
                                self.model_kwargs, self.history)

            save_checkpoint(self.last_path, self.model, self.optimizer,
                            self.scheduler, epoch, self.early_stopping.best,
                            self.model_kwargs, self.history)

            if self.early_stopping.should_stop:
                logger.info(f"Early stopping at epoch {epoch} "
                            f"(no val improvement for {self.cfg.early_stopping_patience} epochs).")
                break

        save_json(self.history, self.history_path)
        logger.info(f"Training complete. Best val pinball: {self.early_stopping.best:.6f}. "
                    f"History -> {self.history_path}")
        return self.history
