"""
Training / dataset-split hyperparameters.
Phase 1 uses the split ratios and random seed; the rest supports
later phases.
"""

from dataclasses import dataclass


@dataclass
class SplitConfig:
    # Chronological split (no shuffling across time to avoid leakage)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    num_workers: int = 2

    # Gradient clipping (max global grad norm). None disables.
    grad_clip_norm: float = 1.0

    # Cosine LR schedule floor as a fraction of the initial lr
    min_lr_factor: float = 0.01

    # ---- Composite loss weights (see src/losses/composite_loss.py) ----
    # Pinball loss over P10/P50/P90 is the primary objective.
    w_pinball: float = 1.0
    # Interval sharpness regularizer: counterweight to pinball's
    # "wider is safer" pull. Small -- coverage must dominate sharpness.
    w_width: float = 0.05
    # Auxiliary dense per-expert loss, so every expert keeps receiving
    # gradient even when the gate routes away from it.
    w_expert: float = 0.3
    # Weight on each expert's own physical-prior auxiliary loss
    aux_weight: float = 0.1
    # Gate load-balancing penalty: discourages collapse onto one expert.
    w_load_balance: float = 0.01

    # Number of warmup epochs during which the expert loss weight is held
    # at 1.0 so experts learn something before routing starts to matter.
    expert_warmup_epochs: int = 5


SPLIT_CONFIG = SplitConfig()
TRAINING_CONFIG = TrainingConfig()
