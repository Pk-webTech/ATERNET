"""
Checkpoint save/load.

Stores model + optimizer + scheduler state alongside the epoch, the
best validation score, and the feature dimensions the model was built
with -- so `load_checkpoint` can reconstruct the architecture without
the caller having to remember how it was configured.
"""

from pathlib import Path

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch: int = 0,
                    best_metric: float = float("inf"), model_kwargs: dict = None,
                    history: list = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "model_kwargs": model_kwargs or {},
        "history": history or [],
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path, model_class=None, map_location="cpu", model=None):
    """
    If `model` is given, load weights into it. Otherwise build a fresh
    instance of `model_class` from the stored model_kwargs.
    Returns (model, payload).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is None:
        if model_class is None:
            raise ValueError("Provide either `model` or `model_class`.")
        model = model_class(**payload["model_kwargs"])

    model.load_state_dict(payload["model_state"])
    logger.info(f"Loaded checkpoint from {path} (epoch {payload.get('epoch')}, "
                f"best_metric {payload.get('best_metric'):.6f}).")
    return model, payload


class EarlyStopping:
    """Standard patience-based early stopping on a minimized metric."""

    def __init__(self, patience: int = 8, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.n_bad = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Returns True if this metric is a new best."""
        if metric < self.best - self.min_delta:
            self.best = metric
            self.n_bad = 0
            return True
        self.n_bad += 1
        if self.n_bad >= self.patience:
            self.should_stop = True
        return False
