"""Shared drawing helpers for the report figures (matplotlib only, no external tools)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Times New Roman" if "Times New Roman" in _fonts else "DejaVu Serif"
plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False

# calm, print-friendly palette (works in greyscale too)
C = {
    "blue": "#DCE9F7", "blue_d": "#2E5A88",
    "green": "#DDEFE0", "green_d": "#2F7D4F",
    "amber": "#FBEBC8", "amber_d": "#B7791F",
    "grey": "#EEEEEE", "grey_d": "#555555",
    "red": "#F8DCDC", "red_d": "#B83232",
    "violet": "#E8E0F3", "violet_d": "#5B4A8A",
    "white": "#FFFFFF", "ink": "#222222",
}


def canvas(w_in, h_in, xmax=100, ymax=100):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    ax.set_aspect("auto")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig, ax


def box(ax, cx, cy, w, h, text, fc="blue", fs=8, bold=False, rounded=True, lw=1.0, ec=None, ls="-", color=None):
    ecol = ec or C[fc + "_d"] if (fc + "_d") in C else (ec or C["grey_d"])
    face = C.get(fc, fc)
    style = "round,pad=0.0,rounding_size=1.2" if rounded else "square,pad=0.0"
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=style, fc=face, ec=ecol, lw=lw, ls=ls, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color=color or C["ink"], zorder=3, linespacing=1.15)
    return (cx, cy, w, h)


def band(ax, x0, y0, x1, y1, label, fc="grey", fs=8.5, ha="left"):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=C.get(fc, fc), ec="#999999", lw=0.8, ls="--", zorder=0, alpha=0.55))
    ax.text((x0 + 0.8) if ha == "left" else (x1 - 0.8), y1 - 0.8, label, ha=ha, va="top", fontsize=fs, fontweight="bold", color="#444444", zorder=1)


def arrow(ax, p1, p2, text=None, fs=7, rad=0.0, ls="-", color="#333333", lw=1.0, style="-|>", text_off=(0, 1.2), ha="center", ms=9):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, lw=lw, color=color, ls=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2 + text_off[0], (p1[1] + p2[1]) / 2 + text_off[1]
        ax.text(mx, my, text, ha=ha, va="center", fontsize=fs, color=color, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


def edge(b, side):
    """Anchor point on a box tuple (cx, cy, w, h): 'l','r','t','b'."""
    cx, cy, w, h = b
    return {"l": (cx - w / 2, cy), "r": (cx + w / 2, cy), "t": (cx, cy + h / 2), "b": (cx, cy - h / 2)}[side]


def save(fig, name, dpi=300):
    out = FIG_DIR / name
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    print("wrote", out.name)
    return out
