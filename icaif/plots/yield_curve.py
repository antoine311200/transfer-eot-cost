"""Paper-quality figures for Experiment E2: Yield Curve Dynamics Transfer.

Figure types
------------
plot_curve_comparison       — overlay of mean curves (ref / TACO / truth)
plot_term_structure_bands   — shaded ±1σ bands per tenor
plot_pca_distribution       — 2D PCA scatter of joint coupling (today, tomorrow)
plot_factor_distributions   — per-tenor marginal histograms
plot_correlation_heatmaps   — daily-change correlation matrices compared
plot_yc_convergence         — W2_nu + factor RMSE vs iteration
"""

from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

from .theme import apply_theme, C_REF, C_TACO, C_TRUTH, C_PRE, add_panel_label, iteration_cmap
from ..data.yield_curve import TENORS, TENOR_LABELS, D_YC

apply_theme()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mean_std(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return data.mean(0), data.std(0)


# ── Main figures ─────────────────────────────────────────────────────────────

def plot_curve_comparison(
    ref_curves:   np.ndarray,   # (N, D_YC)
    taco_curves:  np.ndarray,
    truth_curves: np.ndarray,
    pre_curves:   Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Left: mean curves with ±1σ bands.  Right: difference from truth."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Left: mean curve comparison ──
    ax = axes[0]
    for data, color, label, ls in [
        (ref_curves,   C_REF,   "Reference (ZIRP)",  ":"),
        (truth_curves, C_TRUTH, "True (Hiking)",     "--"),
        (taco_curves,  C_TACO,  "TACO (ours)",       "-"),
    ] + ([(pre_curves, C_PRE, "Pretrained-only", "-.")] if pre_curves is not None else []):
        m, s = _mean_std(data)
        ax.plot(TENORS, m, color=color, lw=2.2, ls=ls, label=label)
        ax.fill_between(TENORS, m - s, m + s, color=color, alpha=0.12)

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (%)")
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    ax.set_xticklabels(TENOR_LABELS, rotation=30, ha="right")
    ax.set_title("Yield Curve Comparison — Mean ± 1σ", fontweight="bold")
    ax.legend()
    add_panel_label(ax, "(a)")

    # ── Right: absolute error vs truth ──
    ax = axes[1]
    tru_m, _ = _mean_std(truth_curves)
    for data, color, label in [
        (ref_curves,  C_REF,  "Reference"),
        (taco_curves, C_TACO, "TACO (ours)"),
    ] + ([(pre_curves, C_PRE, "Pretrained-only")] if pre_curves is not None else []):
        m, _ = _mean_std(data)
        ax.plot(TENORS, np.abs(m - tru_m) * 100, color=color, lw=2.2, label=label)

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("|Mean error| (basis points)")
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    ax.set_xticklabels(TENOR_LABELS, rotation=30, ha="right")
    ax.set_title("Absolute Error vs True Hiking Curve", fontweight="bold")
    ax.legend()
    add_panel_label(ax, "(b)")

    fig.suptitle("Yield Curve Regime Transfer", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_pca_distribution(
    ref_xy:   tuple[np.ndarray, np.ndarray],    # (x_today, y_tomorrow) reference
    taco_xy:  tuple[np.ndarray, np.ndarray],
    truth_xy: tuple[np.ndarray, np.ndarray],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """2D PCA scatter of joint coupling — shows how the coupling structure transfers."""
    # Stack x and y horizontally to form the joint coupling
    all_xy = np.vstack([
        np.hstack([ref_xy[0],   ref_xy[1]]),
        np.hstack([truth_xy[0], truth_xy[1]]),
        np.hstack([taco_xy[0],  taco_xy[1]]),
    ])
    pca = PCA(n_components=2)
    pca.fit(all_xy[:min(len(all_xy), 3000)])

    def proj(x, y):
        return pca.transform(np.hstack([x, y]))[:, :2]

    ref_pc   = proj(*ref_xy)
    tru_pc   = proj(*truth_xy)
    tco_pc   = proj(*taco_xy)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    for ax, data, color, title in [
        (axes[0], ref_pc,  C_REF,   "Reference (ZIRP)"),
        (axes[1], tru_pc,  C_TRUTH, "True Hiking"),
        (axes[2], tco_pc,  C_TACO,  "TACO Transfer"),
    ]:
        n = min(len(data), 800)
        ax.scatter(data[:n, 0], data[:n, 1], c=color, s=8, alpha=0.4, rasterized=True)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(title, fontweight="bold")

    var_exp = pca.explained_variance_ratio_ * 100
    fig.suptitle(
        f"Joint Coupling in PCA Space  (PC1 {var_exp[0]:.0f}% + PC2 {var_exp[1]:.0f}% variance)",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_correlation_heatmaps(
    ref_data:   np.ndarray,   # (N, D_YC) daily changes
    taco_data:  np.ndarray,
    truth_data: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Correlation matrix of daily yield changes: reference vs TACO vs truth."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    def corr(data):
        # Daily differences ≈ data itself if data = (today, tomorrow) difference
        return np.corrcoef(data.T)

    datasets = [
        (ref_data,                    "Reference (ZIRP)", "Blues"),
        (truth_data,                  "True Hiking",       "Reds"),
        (taco_data,                   "TACO",              "Greens"),
        (np.abs(corr(taco_data) - corr(truth_data)), "|TACO – Truth|", "Oranges"),
    ]

    for ax, (data, title, cmap) in zip(axes, datasets):
        is_err = title.startswith("|")
        mat = data if is_err else corr(data)
        vmin, vmax = (0, 0.5) if is_err else (-1, 1)
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.85)
        ticks = range(D_YC)
        ax.set_xticks(ticks); ax.set_xticklabels(TENOR_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(ticks); ax.set_yticklabels(TENOR_LABELS, fontsize=7)
        ax.set_title(title, fontweight="bold")

    fig.suptitle("Yield Curve Correlation Structure", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_yc_coupling_evolution(
    history:    list[np.ndarray],   # list of (N, D_YC) target samples
    truth:      np.ndarray,          # (N, D_YC)
    iterations: Sequence[int],
    n_sel: int = 5,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Show how the generated yield curve distribution converges to truth."""
    sel = np.round(np.linspace(0, len(history) - 1, n_sel)).astype(int)
    colors = iteration_cmap(n_sel)

    # Focus on 10Y tenor (index 7)
    tenor_idx = 7

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: 10Y yield marginal evolution ──
    ax = axes[0]
    ax.hist(truth[:, tenor_idx], bins=35, density=True, alpha=0.45, color=C_TRUTH,
            label="True (Hiking)", zorder=4)
    for k, (ci, color) in enumerate(zip(sel, colors)):
        label = f"TACO iter {iterations[ci]}" if k in (0, n_sel - 1) else "_no_legend_"
        ax.hist(history[ci][:, tenor_idx], bins=35, density=True,
                alpha=0.3, color=color, label=label)
    ax.set_xlabel("10Y Yield (%)")
    ax.set_ylabel("Density")
    ax.set_title("10Y Yield Marginal — Evolution")
    ax.legend(fontsize=8)
    add_panel_label(ax, "(a)")

    # ── Right: mean curve evolution ──
    ax = axes[1]
    tru_m, _ = _mean_std(truth)
    ax.plot(TENORS, tru_m, color=C_TRUTH, lw=2.5, ls="--", label="True (Hiking)", zorder=10)
    for k, (ci, color) in enumerate(zip(sel, colors)):
        m, _ = _mean_std(history[ci])
        label = f"Iter {iterations[ci]}" if k in (0, n_sel - 1) else "_no_legend_"
        ax.plot(TENORS, m, color=color, lw=1.6, alpha=0.8, label=label)

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (%)")
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    ax.set_xticklabels(TENOR_LABELS, rotation=30, ha="right")
    ax.set_title("Mean Yield Curve — TACO Evolution")
    ax.legend(fontsize=8)
    add_panel_label(ax, "(b)")

    fig.suptitle("Yield Curve Transfer — Coupling Evolution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def compute_yc_rmse(generated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-tenor RMSE (in %) between generated and truth mean curves."""
    return np.abs(generated.mean(0) - truth.mean(0))
