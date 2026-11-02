"""Consistent paper-quality matplotlib theme for all ICAIF figures."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Palette ──────────────────────────────────────────────────────────────────
C_REF    = "#4878CF"   # blue  — reference regime
C_TACO   = "#2CA02C"   # green — TACO output
C_TRUTH  = "#D62728"   # red   — ground-truth / stressed regime
C_PRE    = "#9467BD"   # purple — pretrained-only baseline
C_OTCFM  = "#E67E22"   # orange — OT-CFM† (cost-aware baseline)
C_NEUTRAL= "#7F7F7F"   # grey  — neutral annotations

PALETTE = {
    "reference":    C_REF,
    "taco":         C_TACO,
    "truth":        C_TRUTH,
    "pretrained":   C_PRE,
    "otcfm":        C_OTCFM,
}


def apply_theme() -> None:
    """Apply a clean, publication-ready rcParam theme."""
    mpl.rcParams.update({
        # Font
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "#cccccc",

        # Lines
        "lines.linewidth":    2.0,
        "lines.markersize":   6,

        # Axes
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.6,

        # Figure
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "figure.facecolor":   "white",
        "axes.facecolor":     "#fafafa",
    })


def iteration_cmap(n: int):
    """Return n colors from blue→green for iteration evolution plots."""
    return plt.cm.viridis(np.linspace(0.15, 0.85, n))


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.06) -> None:
    """Add bold panel label (a), (b), … to axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")
