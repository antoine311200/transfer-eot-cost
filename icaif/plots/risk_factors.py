"""Paper-quality figures for Experiment E3: Cross-Regime Risk Factor Scenario Generation.

Figure types
------------
plot_joint_scatter          — 2D scatter of equity vs vol, comparing regimes
plot_correlation_comparison — 5×5 correlation heatmaps (ref / TACO / truth / error)
plot_marginal_distributions — per-factor histograms, 5 panels
plot_tail_dependence        — upper/lower tail dependence coefficients
plot_portfolio_var          — portfolio VaR/CVaR comparison bar chart
plot_rf_convergence         — W2_nu + correlation-matrix Frobenius error vs iteration
"""

from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

from .theme import (apply_theme, C_REF, C_TACO, C_TRUTH, C_PRE,
                    add_panel_label, iteration_cmap, PALETTE)
from ..data.risk_factors import FACTOR_NAMES, D_RF

apply_theme()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _corr_matrix(data: np.ndarray) -> np.ndarray:
    return np.corrcoef(data.T)


def _portfolio_var(returns: np.ndarray, weights: Optional[np.ndarray] = None,
                   level: float = 0.05) -> tuple[float, float]:
    """Equal-weighted portfolio VaR and CVaR at confidence level (1 - level)."""
    if weights is None:
        weights = np.ones(returns.shape[1]) / returns.shape[1]
    port = returns @ weights
    var = float(np.percentile(port, level * 100))
    cvar = float(port[port <= var].mean())
    return var, cvar


def _tail_dependence(x: np.ndarray, y: np.ndarray,
                     q: float = 0.10) -> tuple[float, float]:
    """Empirical lower and upper tail dependence coefficients."""
    lo = np.quantile(x, q), np.quantile(y, q)
    hi = np.quantile(x, 1 - q), np.quantile(y, 1 - q)
    λ_lo = np.mean((x <= lo[0]) & (y <= lo[1])) / q
    λ_hi = np.mean((x >= hi[0]) & (y >= hi[1])) / q
    return float(λ_lo), float(λ_hi)


# ── Main figures ─────────────────────────────────────────────────────────────

def plot_joint_scatter(
    ref_data:   np.ndarray,   # (N, D_RF) reference regime daily changes
    taco_data:  np.ndarray,   # (N, D_RF) TACO-generated stressed
    truth_data: np.ndarray,   # (N, D_RF) true stressed
    xi: int = 0,              # factor index for x-axis (default: Equity)
    yi: int = 1,              # factor index for y-axis (default: Vol)
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """3-panel scatter: reference / TACO / truth for two risk factors.

    Overlaid density contours highlight the joint distribution shape.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    datasets = [
        (ref_data,   C_REF,   "Reference (Normal)"),
        (truth_data, C_TRUTH, "True Stressed"),
        (taco_data,  C_TACO,  "TACO Transfer"),
    ]
    for ax, (data, color, title) in zip(axes, datasets):
        x, y = data[:, xi], data[:, yi]
        n = min(len(x), 500)
        ax.scatter(x[:n] * 100, y[:n] * 100, color=color, s=10, alpha=0.4, rasterized=True)

        # KDE contours
        try:
            xg = np.linspace(x.min() * 100, x.max() * 100, 60)
            yg = np.linspace(y.min() * 100, y.max() * 100, 60)
            Xg, Yg = np.meshgrid(xg, yg)
            kde = gaussian_kde(np.vstack([x * 100, y * 100]))
            Zg = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
            ax.contour(Xg, Yg, Zg, levels=5, colors=[color], alpha=0.5, linewidths=1.2)
        except Exception:
            pass

        ax.axhline(0, color="#999", lw=0.8, ls=":")
        ax.axvline(0, color="#999", lw=0.8, ls=":")
        ax.set_xlabel(f"{FACTOR_NAMES[xi]} return (%)")
        ax.set_ylabel(f"{FACTOR_NAMES[yi]} change (%)")
        ax.set_title(title, fontweight="bold")
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.05, 0.94, f"ρ = {corr:.2f}", transform=ax.transAxes,
                fontsize=9, va="top", color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8))

    fig.suptitle(f"Joint Distribution: {FACTOR_NAMES[xi]} vs {FACTOR_NAMES[yi]}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_correlation_comparison(
    ref_data:   np.ndarray,
    taco_data:  np.ndarray,
    truth_data: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """4-panel correlation matrix comparison."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    datasets = [
        (_corr_matrix(ref_data),   "Reference (Normal)", "Blues"),
        (_corr_matrix(truth_data), "True Stressed",       "Reds"),
        (_corr_matrix(taco_data),  "TACO",                "Greens"),
        (np.abs(_corr_matrix(taco_data) - _corr_matrix(truth_data)),
                                   "|TACO – Truth|",      "Oranges"),
    ]
    for ax, (mat, title, cmap) in zip(axes, datasets):
        is_err = title.startswith("|")
        vmin, vmax = (0, 0.4) if is_err else (-1, 1)
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.ax.tick_params(labelsize=8)
        n = D_RF
        ax.set_xticks(range(n)); ax.set_xticklabels(FACTOR_NAMES, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(FACTOR_NAMES, fontsize=8)
        ax.set_title(title, fontweight="bold")

        # Annotate cells with correlation value
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.suptitle("Risk Factor Correlation Matrix Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_marginal_distributions(
    ref_data:   np.ndarray,
    taco_data:  np.ndarray,
    truth_data: np.ndarray,
    pre_data:   Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """5-panel histogram: one per risk factor, all methods overlaid."""
    fig, axes = plt.subplots(1, D_RF, figsize=(15, 3.8))

    for i, (ax, name) in enumerate(zip(axes, FACTOR_NAMES)):
        bins = 40
        kw = dict(bins=bins, density=True, histtype="step", lw=2.0)
        ax.hist(ref_data[:, i] * 100,   **kw, color=C_REF,   label="Reference")
        ax.hist(truth_data[:, i] * 100, **kw, color=C_TRUTH,  label="True stressed", ls="--")
        if pre_data is not None:
            ax.hist(pre_data[:, i] * 100, **kw, color=C_PRE, label="Pretrained", ls="-.")
        ax.hist(taco_data[:, i] * 100,  **kw, color=C_TACO,   label="TACO")

        ax.set_xlabel(f"{name} (%)" if i < 2 else f"{name} (bp)")
        ax.set_ylabel("Density" if i == 0 else "")
        ax.set_title(name, fontweight="bold")
        if i == D_RF - 1:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Risk Factor Marginal Distributions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_portfolio_var(
    ref_data:   np.ndarray,
    taco_data:  np.ndarray,
    truth_data: np.ndarray,
    pre_data:   Optional[np.ndarray] = None,
    levels: tuple = (0.01, 0.05),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart comparing equal-weighted portfolio VaR and CVaR across methods."""
    methods = ["Reference", "True stressed", "TACO"]
    colors  = [C_REF,       C_TRUTH,          C_TACO]
    dsets   = [ref_data,    truth_data,        taco_data]
    if pre_data is not None:
        methods.append("Pretrained")
        colors.append(C_PRE)
        dsets.append(pre_data)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, level in zip(axes, levels):
        var_vals  = [_portfolio_var(d, level=level)[0] * 100 for d in dsets]
        cvar_vals = [_portfolio_var(d, level=level)[1] * 100 for d in dsets]

        x = np.arange(len(methods))
        w = 0.35
        ax.bar(x - w / 2, [-v for v in var_vals],  width=w, color=colors, alpha=0.75,
               label="VaR",  edgecolor="white")
        ax.bar(x + w / 2, [-v for v in cvar_vals], width=w, color=colors, alpha=0.45,
               label="CVaR", edgecolor="white", hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("Loss (%)")
        ax.set_title(f"{int((1-level)*100)}% VaR / CVaR", fontweight="bold")
        ax.legend()
        add_panel_label(ax, "ab"[list(levels).index(level)])

    fig.suptitle("Equal-Weighted Portfolio Risk Measures",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_rf_coupling_evolution(
    history:    list[np.ndarray],   # list of (N, D_RF) target samples
    truth:      np.ndarray,
    iterations: Sequence[int],
    n_sel: int = 5,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Equity–Vol scatter evolution + correlation Frobenius error."""
    sel = np.round(np.linspace(0, len(history) - 1, n_sel)).astype(int)
    colors_iter = iteration_cmap(n_sel)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: Equity–Vol scatter evolution ──
    ax = axes[0]
    n_truth = min(len(truth), 250)
    ax.scatter(truth[:n_truth, 0] * 100, truth[:n_truth, 1] * 100, c=C_TRUTH, s=12, alpha=0.5,
               label="True stressed", zorder=10)
    for k, (ci, c) in enumerate(zip(sel, colors_iter)):
        lbl = f"TACO iter {iterations[ci]}" if k in (0, n_sel - 1) else "_no_legend_"
        d = history[ci]
        n_d = min(len(d), 250)
        ax.scatter(d[:n_d, 0] * 100, d[:n_d, 1] * 100, c=[c] * n_d, s=8, alpha=0.35, label=lbl)
    ax.set_xlabel("Equity return (%)")
    ax.set_ylabel("Vol change (%)")
    ax.set_title("Equity–Vol Joint Distribution Evolution")
    ax.legend(fontsize=8)
    add_panel_label(ax, "(a)")

    # ── Right: Frobenius error of correlation matrix vs iteration ──
    ax = axes[1]
    truth_corr = _corr_matrix(truth)
    frob = [
        float(np.linalg.norm(_corr_matrix(h) - truth_corr, "fro"))
        for h in history
    ]
    ax.plot(iterations, frob, color=C_TACO, lw=2.2, label="TACO")
    ax.set_xlabel("TACO Iteration")
    ax.set_ylabel("Correlation Matrix Error (Frobenius)")
    ax.set_title("Correlation Structure Convergence")
    ax.legend()
    add_panel_label(ax, "(b)")

    fig.suptitle("Risk Factor Scenario Transfer — Coupling Evolution",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def compute_frob_error(generated: np.ndarray, truth: np.ndarray) -> float:
    """Frobenius error of correlation matrices."""
    return float(np.linalg.norm(_corr_matrix(generated) - _corr_matrix(truth), "fro"))
