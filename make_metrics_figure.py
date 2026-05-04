"""
make_metrics_figure.py
----------------------
Per-experiment convergence overview with mean ± 1-std envelope across best N seeds.

Two output figures:
  images/metrics/<exp>_transport.pdf   — sinkhorn div, MMD, energy, map errors, coupling errors
  images/metrics/<exp>_wasserstein.pdf — W1, W2, BW2

Seed selection: best TOP_N seeds ranked by final map_err_fwd → W2_nu → sinkhorn_div.

Usage:
    python make_metrics_figure.py --exp 2d_simple
    python make_metrics_figure.py --exp 2d_simple --results results_seeds2 --seeds 5 --top_n 3
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_theme
paper_theme.setup_style()

# ── Defaults ──────────────────────────────────────────────────────────────────
RESULTS_DIR_DEFAULT = "results_seeds2"
N_SEEDS_DEFAULT     = 5
TOP_N_DEFAULT       = 3
RANK_METRICS        = ["map_err_fwd", "W2_nu", "sinkhorn_div"]
SMOOTH_WINDOW       = 3
NCOLS               = 3

# ── Metric panel definitions ──────────────────────────────────────────────────
# (csv_column, display_title, mask_mode)
# mask_mode: "fwd" = even iters only, "bwd" = odd iters only, "all" = every iter
TRANSPORT_PANELS = [
    ("sinkhorn_div",         "Sinkhorn divergence",    "all"),
    ("mmd_nu",               "MMD (target)",           "all"),
    ("energy_dist_nu",       "Energy distance",        "all"),
    ("map_err_fwd",          "Map error →",            "fwd"),
    ("map_err_bwd",          "Map error ←",            "bwd"),
    ("coupling_map_err_fwd", "Coupling map err. →",    "fwd"),
    ("coupling_map_err_bwd", "Coupling map err. ←",    "bwd"),
]

WASSERSTEIN_PANELS = [
    ("W1_nu", r"$W_1$ (target)",            "fwd"),
    ("W2_nu", r"$W_2$ (target)",            "fwd"),
    ("bw2",   r"$\mathrm{BW}_2$",          "all"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < 2:
        return x
    return uniform_filter1d(x.astype(float), size=w, mode="nearest")


def load_single(
    path: Path, metric: str, mask_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load (iteration, values) for one CSV, applying fwd/bwd/all mask."""
    if not path.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(path)
    if metric not in df.columns:
        return np.array([]), np.array([])
    iters = df["iteration"].values.astype(int)
    if mask_mode == "fwd":
        mask = iters % 2 == 0
    elif mask_mode == "bwd":
        mask = iters % 2 == 1
    else:
        mask = np.ones(len(iters), dtype=bool)
    iters = iters[mask]
    vals = pd.to_numeric(df.loc[mask, metric], errors="coerce").values
    valid = ~np.isnan(vals)
    return iters[valid], vals[valid]


def select_best_seeds(
    exp: str, all_seeds: list[int], top_n: int, results_dir: Path
) -> list[int]:
    """Pick the TOP_N seeds with the lowest final value of the best-available ranking metric."""
    for rm in RANK_METRICS:
        mask_mode = "fwd" if rm == "map_err_fwd" else "all"
        scores: dict[int, float] = {}
        for s in all_seeds:
            path = results_dir / f"seed_{s}" / exp / "metrics.csv"
            _, vals = load_single(path, rm, mask_mode)
            if len(vals) > 0:
                scores[s] = float(vals[-1])
        if scores:
            ranked = sorted(scores, key=lambda s: scores[s])
            selected = ranked[:top_n]
            print(f"  [{exp}] ranking by '{rm}': {ranked} → keeping {selected}")
            return selected
    return all_seeds[:top_n]


def compute_envelope(
    exp: str,
    metric: str,
    mask_mode: str,
    seeds: list[int],
    results_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate across seeds at each iteration.
    Returns (iters, mean, std) with smoothing applied per seed before aggregation.
    """
    grouped: dict[int, list[float]] = {}
    for s in seeds:
        path = results_dir / f"seed_{s}" / exp / "metrics.csv"
        iters, vals = load_single(path, metric, mask_mode)
        if len(vals) == 0:
            continue
        vals_s = smooth(vals, SMOOTH_WINDOW)
        for it, v in zip(iters, vals_s):
            grouped.setdefault(int(it), []).append(float(v))

    if not grouped:
        return np.array([]), np.array([]), np.array([])

    sorted_iters = sorted(grouped)
    means = np.array([np.mean(grouped[it]) for it in sorted_iters])
    stds  = np.array([np.std(grouped[it])  for it in sorted_iters])
    return np.array(sorted_iters, dtype=float), means, stds


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_panels(
    exp: str,
    panels: list[tuple[str, str, str]],
    seeds: list[int],
    results_dir: Path,
    out_path: Path,
) -> None:
    n = len(panels)
    nrows = (n + NCOLS - 1) // NCOLS

    fig, axes = plt.subplots(
        nrows, NCOLS,
        figsize=(3.7 * NCOLS, 2.7 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    color  = paper_theme.BLUE
    ecolor = paper_theme.LBLUE

    for idx, (col, title, mask_mode) in enumerate(panels):
        ax = axes_flat[idx]
        iters, mean, std = compute_envelope(exp, col, mask_mode, seeds, results_dir)

        if len(iters) == 0:
            ax.set_visible(False)
            continue

        lo = np.clip(mean - std, 1e-12, None)
        hi = mean + std

        ax.plot(iters, mean, color=color, lw=1.6, zorder=3)
        ax.fill_between(iters, lo, hi, color=ecolor, alpha=0.45, zorder=2)

        paper_theme.style_axes(ax)

        # Spine / border colour
        for spine in ax.spines.values():
            spine.set_edgecolor(paper_theme.BORDER)
            spine.set_linewidth(0.8)

        ax.set_title(title, fontsize=8.5, pad=4, color=paper_theme.DARK,
                     fontweight="semibold")
        ax.set_xlabel("Iteration", fontsize=7.5, color=paper_theme.SLATE)
        ax.tick_params(labelsize=7, colors=paper_theme.SLATE, length=3, width=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

        # Log scale, fall back to linear if values include zeros/negatives
        if np.all(mean > 0):
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Hide unused subplots
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout(h_pad=1.8, w_pad=1.2)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convergence metrics figure (mean ± std envelope across best seeds)"
    )
    parser.add_argument("--exp",     required=True,
                        help="Experiment name, e.g. 2d_simple")
    parser.add_argument("--results", default=RESULTS_DIR_DEFAULT,
                        help=f"Root results directory (default: {RESULTS_DIR_DEFAULT})")
    parser.add_argument("--seeds",   type=int, default=N_SEEDS_DEFAULT,
                        help="Total number of seeds available (default: 5)")
    parser.add_argument("--top_n",   type=int, default=TOP_N_DEFAULT,
                        help="Keep only the best N seeds (default: 3)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    all_seeds   = list(range(args.seeds))
    out_dir     = Path("images/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selecting best {args.top_n} seeds for '{args.exp}' …")
    seeds = select_best_seeds(args.exp, all_seeds, args.top_n, results_dir)

    print("Plotting transport quality metrics …")
    plot_panels(
        args.exp, TRANSPORT_PANELS, seeds, results_dir,
        out_dir / f"{args.exp}_transport.pdf",
    )

    print("Plotting Wasserstein distances …")
    plot_panels(
        args.exp, WASSERSTEIN_PANELS, seeds, results_dir,
        out_dir / f"{args.exp}_wasserstein.pdf",
    )

    print("Done.")


if __name__ == "__main__":
    main()
