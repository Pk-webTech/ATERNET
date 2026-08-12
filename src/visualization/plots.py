"""
Diagnostic plots written to outputs/plots/ and outputs/attention_maps/.

Matplotlib only, one figure per function, no seaborn and no explicit
colours -- defaults keep the figures readable under any style sheet and
avoid encoding meaning in colour alone.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: these are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np

from config.paths import PLOTS_DIR, ATTENTION_MAPS_DIR
from src.phase3.expert_pool import EXPERT_ORDER
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_training_history(history: list, out_dir=PLOTS_DIR) -> Path:
    """Train vs val loss curves + the component breakdown."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [h["train_loss_total"] for h in history], label="train total")
    axes[0].plot(epochs, [h["val_loss_total"] for h in history], label="val total")
    axes[0].set_title("Composite loss")
    axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(epochs, [h["train_pinball_total"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_pinball_total"] for h in history], label="val")
    axes[1].set_title("Pinball loss (model-selection metric)")
    axes[1].set_xlabel("epoch"); axes[1].legend()

    axes[2].plot(epochs, [h["val_sharpness"] for h in history], label="sharpness")
    axes[2].plot(epochs, [h["val_load_balance"] for h in history], label="load balance")
    axes[2].set_title("Regularizer terms (val)")
    axes[2].set_xlabel("epoch"); axes[2].legend()

    return _save(fig, Path(out_dir) / "training_history.png")


def plot_gate_share_history(history: list, out_dir=PLOTS_DIR) -> Path:
    """Gate mass per expert over training -- the collapse detector."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in EXPERT_ORDER:
        key = f"val_gate_share_{name}"
        if key in history[0]:
            ax.plot(epochs, [h[key] for h in history], label=name)
    ax.axhline(1.0 / len(EXPERT_ORDER), linestyle="--", linewidth=1, label="uniform")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean gate weight")
    ax.set_title("Expert routing share over training")
    ax.legend()
    return _save(fig, Path(out_dir) / "gate_share_history.png")


def plot_forecast_examples(y_true, p10, p50, p90, n_examples: int = 6,
                           indices=None, out_dir=PLOTS_DIR) -> Path:
    """Fan charts: observed vs P50 with the P10-P90 band."""
    n = len(y_true)
    if indices is None:
        # Prefer windows with meaningful generation. Evenly-spaced sampling
        # over the test split lands mostly on night windows (target pinned
        # at 0), which produce six identical flat-line panels that show
        # nothing about forecast quality. Sample across the daytime windows
        # instead, spanning the range of generation levels.
        daytime = np.where(y_true.max(axis=1) > 0.05)[0]
        pool = daytime if len(daytime) >= n_examples else np.arange(n)
        order = pool[np.argsort(y_true[pool].max(axis=1))]
        indices = order[np.linspace(0, len(order) - 1, min(n_examples, len(order))).astype(int)]

    rows = int(np.ceil(len(indices) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3.2 * rows), squeeze=False)
    steps = np.arange(y_true.shape[1]) + 1

    for ax, i in zip(axes.ravel(), indices):
        ax.fill_between(steps, p10[i], p90[i], alpha=0.3, label="P10-P90")
        ax.plot(steps, p50[i], label="P50")
        ax.plot(steps, y_true[i], linestyle="--", label="observed")
        ax.set_title(f"window {i}")
        ax.set_xlabel("horizon step")
    for ax in axes.ravel()[len(indices):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=8)

    return _save(fig, Path(out_dir) / "forecast_examples.png")


def plot_reliability(y_true, p10, p50, p90, out_dir=PLOTS_DIR) -> Path:
    """
    Empirical vs nominal coverage at the three predicted quantile levels.
    A perfectly calibrated model sits on the diagonal.
    """
    nominal = [0.1, 0.5, 0.9]
    empirical = [float(np.mean(y_true <= q)) for q in (p10, p50, p90)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="perfect calibration")
    ax.plot(nominal, empirical, marker="o", label="ATERNET")
    for nq, eq in zip(nominal, empirical):
        ax.annotate(f"{eq:.3f}", (nq, eq), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("nominal quantile level"); ax.set_ylabel("empirical frequency below")
    ax.set_title("Reliability diagram")
    ax.legend()
    return _save(fig, Path(out_dir) / "reliability_diagram.png")


def plot_per_horizon(per_horizon: dict, out_dir=PLOTS_DIR) -> Path:
    """Error growth and coverage stability with lead time."""
    steps = np.arange(len(per_horizon["mae"])) + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(steps, per_horizon["mae"], marker="o", label="MAE")
    axes[0].plot(steps, per_horizon["rmse"], marker="s", label="RMSE")
    axes[0].set_xlabel("horizon step (15 min each)"); axes[0].set_title("Error vs lead time")
    axes[0].legend()

    if "picp" in per_horizon:
        axes[1].plot(steps, per_horizon["picp"], marker="o", label="PICP")
        axes[1].axhline(0.80, linestyle="--", linewidth=1, label="nominal 0.80")
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("horizon step"); axes[1].set_title("Coverage vs lead time")
        axes[1].legend()

    return _save(fig, Path(out_dir) / "per_horizon_metrics.png")


def plot_uncertainty_vs_error(sigma_total, interval_width, abs_error, out_dir=PLOTS_DIR) -> Path:
    """
    The claim the whole dual-uncertainty design rests on: samples the
    model flags as uncertain should be the ones it actually gets wrong.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(sigma_total, abs_error, s=4, alpha=0.3)
    axes[0].set_xlabel("sigma_total"); axes[0].set_ylabel("mean |error|")
    axes[0].set_title("Fused uncertainty vs realised error")

    axes[1].scatter(interval_width, abs_error, s=4, alpha=0.3)
    axes[1].set_xlabel("mean P10-P90 width"); axes[1].set_ylabel("mean |error|")
    axes[1].set_title("Interval width vs realised error")

    return _save(fig, Path(out_dir) / "uncertainty_vs_error.png")


def plot_gate_by_regime(gate_weights, regime_ids, out_dir=PLOTS_DIR) -> Path:
    """
    Mean routing weight per (true regime, expert). If the architecture's
    premise holds, this matrix should be strongest on the diagonal.
    """
    from src.utils.constants import ID_TO_REGIME
    n_regimes = len(ID_TO_REGIME)
    mat = np.full((n_regimes, gate_weights.shape[1]), np.nan)
    for rid in range(n_regimes):
        mask = regime_ids == rid
        if mask.sum():
            mat[rid] = gate_weights[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(EXPERT_ORDER)), EXPERT_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(n_regimes), [ID_TO_REGIME[i] for i in range(n_regimes)])
    ax.set_xlabel("routed to expert"); ax.set_ylabel("true regime label")
    ax.set_title("Mean gate weight by regime")
    for i in range(n_regimes):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    return _save(fig, Path(out_dir) / "gate_by_regime.png")


def plot_attention_map(attn_weights, out_dir=ATTENTION_MAPS_DIR, index: int = 0) -> Path:
    """NWP cross-attention: which NWP patch each PV patch attends to."""
    a = np.asarray(attn_weights)
    if a.ndim == 3:
        a = a[index]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(a, aspect="auto")
    ax.set_xlabel("NWP patch (key)"); ax.set_ylabel("PV patch (query)")
    ax.set_title(f"NWP cross-attention -- sample {index}")
    fig.colorbar(im, ax=ax)
    return _save(fig, Path(out_dir) / f"nwp_cross_attention_{index}.png")
