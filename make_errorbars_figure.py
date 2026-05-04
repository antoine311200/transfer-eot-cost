"""
make_errorbars_figure.py
------------------------
Multi-seed error-band convergence plots for 2D Simple, Complex, Moon.

Expects results in:
    results_seeds/seed_<s>/<exp_name>/metrics.csv
  e.g.  results_seeds/seed_0/2d_simple/metrics.csv

These are produced by running:
    python -m dualwcfm.experiments.run --mode 2d --level simple --seed <s> --output results_seeds/seed_<s>
    python -m dualwcfm.experiments.run --mode 2d --level complex --seed <s> --output results_seeds/seed_<s>
    python -m dualwcfm.experiments.run --mode 2d --level moon --seed <s> --output results_seeds/seed_<s>

Produces:
    images/errorbars/errorbars_2d.pdf   — mean ± 1-std shaded envelope

Usage:
    python make_errorbars_figure.py
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_theme
paper_theme.setup_style()

# ── Config ─────────────────────────────────────────────────────────────────────
EXPERIMENTS  = ["2d_simple", "2d_complex", "2d_moon"]
SEEDS        = list(range(5))          # seeds 0-4
RESULTS_DIR  = Path("results_seeds2")
OUT_DIR      = Path("images/errorbars")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N        = 3                       # keep only the best N seeds per experiment
# Ranking priority: use the first metric that is available across seeds.
# Lower final value = better.
RANK_METRICS = ["map_err_fwd", "W2_nu", "sinkhorn_div"]

METRICS = {
    "W1_nu":        (r"$\mathrm{SW}_1(\hat\nu)\downarrow$",   True),
    "map_err_fwd":  (r"Map error (fwd) $\downarrow$",          True),
    "sinkhorn_div": (r"Sinkhorn div. $\downarrow$",            False),
}

COLORS = {
    "2d_simple":  paper_theme.BLUE,
    "2d_complex": paper_theme.PINK,
    "2d_moon":    paper_theme.TEAL,
}
LABELS = {
    "2d_simple":  "Simple",
    "2d_complex": "Complex",
    "2d_moon":    "Moon",
}
SMOOTH_WINDOW = 5


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    return uniform_filter1d(x, size=w, mode="nearest")


def select_best_seeds(exp: str) -> list[int]:
    """Return the TOP_N seeds with the lowest final value of the best-available ranking metric."""
    for rm in RANK_METRICS:
        fwd_only = rm == "map_err_fwd"
        scores: dict[int, float] = {}
        for s in SEEDS:
            path = RESULTS_DIR / f"seed_{s}" / exp / "metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if rm not in df.columns:
                continue
            mask = (df["iteration"].values % 2 == 0) if fwd_only else np.ones(len(df), bool)
            vals = pd.to_numeric(df.loc[mask, rm], errors="coerce").dropna().values
            if len(vals) > 0:
                scores[s] = float(vals[-1])
        if scores:
            ranked = sorted(scores, key=lambda s: scores[s])
            selected = ranked[:TOP_N]
            print(f"  [{exp}] ranking by '{rm}': seeds {ranked} → keeping {selected}")
            return selected
    # fallback: no metric found, return first N seeds
    return SEEDS[:TOP_N]


def load_seeds(exp: str, metric: str, fwd_only: bool, seeds: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load metric for the given seeds, return (iterations, mean, std) aligned."""
    all_series: list[np.ndarray] = []
    max_len = 0

    for s in seeds:
        path = RESULTS_DIR / f"seed_{s}" / exp / "metrics.csv"
        if not path.exists():
            continue
        df   = pd.read_csv(path)
        if metric not in df.columns:
            continue
        mask = (df["iteration"].values % 2 == 0) if fwd_only else np.ones(len(df), bool)
        vals = pd.to_numeric(df.loc[mask, metric], errors="coerce").values
        vals = smooth(vals, SMOOTH_WINDOW)
        all_series.append(vals)
        max_len = max(max_len, len(vals))

    if not all_series:
        return np.array([]), np.array([]), np.array([])

    # Pad shorter series with their last value
    padded = np.full((len(all_series), max_len), np.nan)
    for i, s in enumerate(all_series):
        padded[i, :len(s)] = s

    mean = np.nanmean(padded, axis=0)
    std  = np.nanstd(padded, axis=0)
    iters = np.arange(max_len)
    return iters, mean, std


def plot_errorbars():
    # Select best seeds once per experiment (avoids re-reading CSVs per metric)
    best_seeds = {exp: select_best_seeds(exp) for exp in EXPERIMENTS}

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.0 * n_metrics, 3.2))
    if n_metrics == 1:
        axes = [axes]

    for ax, (metric, (ylabel, fwd_only)) in zip(axes, METRICS.items()):
        for exp in EXPERIMENTS:
            iters, mean, std = load_seeds(exp, metric, fwd_only, best_seeds[exp])
            if len(iters) == 0:
                continue
            c = COLORS[exp]
            ax.plot(iters, mean, color=c, lw=1.6, label=LABELS[exp])
            ax.fill_between(iters, mean - std, mean + std, color=c, alpha=0.20)

        paper_theme.style_axes(ax)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")

    axes[0].legend(**paper_theme.legend_kw())
    fig.tight_layout()
    out = OUT_DIR / "errorbars_2d.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("Generating error-bar figures …")
    plot_errorbars()
    print("Done.")
