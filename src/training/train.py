"""
Training entry point.

    python -m src.training.train --epochs 30 --batch-size 64
"""

import argparse

import torch

from config.training_config import TRAINING_CONFIG
from config.model_config import DATA_CONFIG
from src.models.aternet import ATERNET
from src.training.dataset import build_dataloaders
from src.training.trainer import Trainer
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train ATERNET.")
    p.add_argument("--epochs", type=int, default=TRAINING_CONFIG.epochs)
    p.add_argument("--batch-size", type=int, default=TRAINING_CONFIG.batch_size)
    p.add_argument("--lr", type=float, default=TRAINING_CONFIG.lr)
    p.add_argument("--seed", type=int, default=TRAINING_CONFIG.seed)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-regime-weights", action="store_true",
                   help="Disable per-expert regime up-weighting in the dense expert loss.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = TRAINING_CONFIG
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr

    train_loader, val_loader, _, train_ds = build_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model_kwargs = dict(
        n_seq_features=train_ds.n_seq_features,
        n_nwp_features=train_ds.n_nwp_features,
        n_sigma_features=train_ds.n_sigma_features,
        horizon=train_ds.horizon,
    )
    model = ATERNET(**model_kwargs)

    device = torch.device(args.device) if args.device else None
    trainer = Trainer(model, train_loader, val_loader, device=device, cfg=cfg,
                      model_kwargs=model_kwargs,
                      use_regime_weights=not args.no_regime_weights)
    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()
