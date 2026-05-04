"""
paper_theme.py
--------------
Shared visual style and colour palette for all DualWCFM paper figures.

Matches the palette defined in misc/tikz_transposed.tex and the theorem
graph colours used in the paper appendix.

Usage
-----
    import paper_theme
    paper_theme.setup_style()          # apply rcParams
    ax.scatter(x, y, color=paper_theme.BLUE)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Primary palette  (hex values from tikz_transposed.tex)
# ---------------------------------------------------------------------------

BLUE   = "#3468DE"   # mBlue   — reference source µ
PINK   = "#CD4164"   # mPink   — reference target ν
TEAL   = "#00A898"   # mTeal   — new source µ'
PURPLE = "#744CC0"   # mPurple — new target ν' / adapted prediction
LBLUE  = "#84B6F2"   # mLBlue  — light blue / intermediate trajectories
LPINK  = "#F894B2"   # mLPink  — light pink
SLATE  = "#606C7C"   # mSlate  — coupling lines, neutral elements
DARK   = "#282C38"   # mDark   — text, axes labels
BG     = "#F7FAFE"   # mBg     — figure background
BORDER = "#D0D8E8"   # mBorder — spine / frame colour

# Darker shades for edge/shadow effects
BLUE_DARK   = "#1E3E9A"
PINK_DARK   = "#8B2340"
TEAL_DARK   = "#006E64"
PURPLE_DARK = "#4A2E80"

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

# Blue density (combined source+target KDE background)
CMAP_DENSITY = LinearSegmentedColormap.from_list(
    "paper_density",
    [BG, LBLUE, BLUE, BLUE_DARK],
    N=256,
)

# Teal density (source-only KDE)
CMAP_TEAL = LinearSegmentedColormap.from_list(
    "paper_teal",
    [BG, "#A0DDD8", TEAL, TEAL_DARK],
    N=256,
)

# Purple density (target/prediction KDE)
CMAP_PURPLE = LinearSegmentedColormap.from_list(
    "paper_purple",
    [BG, "#C8B8E8", PURPLE, PURPLE_DARK],
    N=256,
)

# Single-cell density (dark → purple → pink, replaces viridis)
CMAP_CELL = LinearSegmentedColormap.from_list(
    "paper_cell",
    [BG, LBLUE, BLUE, PURPLE, PINK],
    N=256,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def setup_style(serif: bool = True) -> None:
    """Apply paper-consistent matplotlib rcParams.

    Call once at the top of each figure script, before any plot is created.
    """
    family = "serif" if serif else "sans-serif"
    plt.rcParams.update({
        "font.family":          family,
        "font.serif":           ["Times New Roman", "DejaVu Serif", "serif"],
        "font.sans-serif":      ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "mathtext.fontset":     "cm",
        "font.size":            9,
        "axes.titlesize":       11,
        "axes.labelsize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.fontsize":      8,
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "savefig.facecolor":    "white",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.7,
        "axes.edgecolor":       BORDER,
        "grid.linewidth":       0.4,
        "grid.alpha":           0.30,
        "grid.color":           BORDER,
        "lines.linewidth":      1.5,
        "patch.linewidth":      0.5,
        "xtick.color":          DARK,
        "ytick.color":          DARK,
        "text.color":           DARK,
        "axes.labelcolor":      DARK,
        "axes.titlecolor":      DARK,
    })


def style_axes(ax) -> None:
    """Apply consistent spine/tick styling to a single Axes."""
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.7)
    ax.tick_params(colors=DARK, labelcolor=DARK)


def legend_kw(**overrides) -> dict:
    """Return default legend kwargs that match the paper style."""
    kw = dict(
        framealpha=0.88,
        facecolor="white",
        edgecolor=BORDER,
        labelcolor=DARK,
        handletextpad=0.4,
        borderpad=0.5,
    )
    kw.update(overrides)
    return kw
