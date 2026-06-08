"""Paper-quality figures for the cross-maturity options EMOT experiment.

Figure catalogue
----------------
plot_joint_coupling          — centrepiece: marginal-density jointplot (T1 × T2)
                               with price histograms on each axis.
plot_strike_comparison       — per-strike call price distributions (T1 and T2).
plot_martingale_verification — conditional mean E[Y|X=x] vs x for EOT / EMOT.
plot_convergence             — W2_ν and marginal distances vs iteration.
plot_calendar_spread         — calendar-spread C(T2)−C(T1) distribution per strike.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde

from .theme import apply_theme, C_REF, C_TACO, C_TRUTH, C_PRE, add_panel_label
from ..data.options import STRIKE_LABELS, MATURITY_LABELS, ATM_IDX, MONEYNESS_GRID

apply_theme()

# Extra colour for EMOT variant
C_EMOT = "#E67E22"   # orange


# ── Internal helpers ──────────────────────────────────────────────────────────

def _kde_density(data: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """1-D Gaussian KDE evaluated on grid."""
    if len(data) < 2:
        return np.zeros_like(grid)
    kde = gaussian_kde(data, bw_method="scott")
    return kde(grid)


def _hex_grid(ax, x: np.ndarray, y: np.ndarray, color: str,
              alpha: float = 0.7, gridsize: int = 30) -> None:
    """Hexbin with single-colour mapping (no rainbow clutter)."""
    ax.hexbin(x, y, gridsize=gridsize, alpha=alpha,
              cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
                  "mono", ["white", color]),
              linewidths=0.2)


def _side_hist(ax, data: np.ndarray, color: str, label: str,
               orientation: str = "vertical", bins: int = 35,
               alpha: float = 0.65) -> None:
    """Filled histogram on a side axis."""
    ax.hist(data, bins=bins, orientation=orientation,
            color=color, alpha=alpha, density=True, label=label,
            edgecolor="white", linewidth=0.4)


# ── Main jointplot ────────────────────────────────────────────────────────────

def plot_joint_coupling(
    ref_x:    np.ndarray,   # (N, 5) — calm T1 call prices
    ref_y:    np.ndarray,   # (N, 5) — calm T2 call prices
    eot_x:    np.ndarray,   # (N, 5) — TACO EOT T1 (should ≈ stress T1)
    eot_y:    np.ndarray,   # (N, 5) — TACO EOT T2
    emot_x:   np.ndarray,   # (N, 5) — TACO EMOT T1
    emot_y:   np.ndarray,   # (N, 5) — TACO EMOT T2
    truth_x:  np.ndarray,   # (N, 5) — true stress T1
    truth_y:  np.ndarray,   # (N, 5) — true stress T2
    strike_idx: int = ATM_IDX,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Centrepiece jointplot: ATM T1 vs ATM T2 call price with marginals on sides.

    Layout (GridSpec 3×3):
    ┌────────────┬───────────────────────────────┬─────┐
    │  (empty)   │   top: T1 marginal density    │     │
    ├────────────┼───────────────────────────────┤     │
    │            │                               │ T2  │
    │  legend    │   central: joint scatter      │marg │
    │  / info    │   + hexbin density            │inal │
    │            │                               │     │
    └────────────┴───────────────────────────────┴─────┘

    The four datasets are overlaid with different colours.
    """
    fig = plt.figure(figsize=(11, 9))
    gs = gridspec.GridSpec(
        3, 3,
        width_ratios=[0.18, 1.0, 0.25],
        height_ratios=[0.25, 1.0, 0.12],
        hspace=0.04,
        wspace=0.04,
    )

    ax_main  = fig.add_subplot(gs[1, 1])
    ax_top   = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    ax_info  = fig.add_subplot(gs[1, 0])
    ax_info.axis("off")

    # ── Data extraction: ATM column ──────────────────────────────────────────
    ki = strike_idx
    r_x, r_y  = ref_x[:, ki],   ref_y[:, ki]
    e_x, e_y  = eot_x[:, ki],   eot_y[:, ki]
    m_x, m_y  = emot_x[:, ki],  emot_y[:, ki]
    t_x, t_y  = truth_x[:, ki], truth_y[:, ki]

    # ── Central scatter ──────────────────────────────────────────────────────
    ax_main.scatter(r_x, r_y, color=C_REF,   label="Reference (calm)",  s=6,  alpha=0.30, linewidths=0)
    ax_main.scatter(t_x, t_y, color=C_TRUTH, label="True stressed",     s=8,  alpha=0.45, linewidths=0)
    ax_main.scatter(e_x, e_y, color=C_TACO,  label="TACO-EOT",          s=7,  alpha=0.50, linewidths=0)
    ax_main.scatter(m_x, m_y, color=C_EMOT,  label="TACO-EMOT",         s=7,  alpha=0.50, linewidths=0)

    # Conditional-mean lines  E[Y | X ≈ x]  for EOT and EMOT
    for (xx, yy, col, ls) in [
        (e_x, e_y, C_TACO, "--"),
        (m_x, m_y, C_EMOT, "-"),
        (t_x, t_y, C_TRUTH, ":"),
    ]:
        order = np.argsort(xx)
        xs_s = xx[order]
        # Smooth conditional mean via rolling window
        w = max(1, len(xs_s) // 20)
        ys_sm = np.convolve(yy[order], np.ones(w) / w, mode="valid")
        xs_sm = xs_s[w // 2: w // 2 + len(ys_sm)]
        ax_main.plot(xs_sm, ys_sm, color=col, lw=1.8, ls=ls, alpha=0.85)

    # Diagonal (martingale reference E[Y]=X)
    all_v = np.concatenate([r_x, t_x, e_x, m_x])
    vmin, vmax = all_v.min() * 0.97, all_v.max() * 1.03
    ax_main.plot([vmin, vmax], [vmin, vmax], color="#888888",
                 lw=1.2, ls=":", alpha=0.6, label="E[Y]=X (martingale)")

    ax_main.set_xlabel(f"T1 ATM Call Price  [K={STRIKE_LABELS[ki]}]", fontsize=11)
    ax_main.set_ylabel(f"T2 ATM Call Price  [K={STRIKE_LABELS[ki]}]", fontsize=11)
    ax_main.legend(loc="upper left", fontsize=8, markerscale=2.5,
                   framealpha=0.92, edgecolor="#cccccc")

    # ── Top histogram: T1 marginal ────────────────────────────────────────────
    bins_x = np.linspace(all_v.min(), all_v.max(), 50)
    for data, col in [(r_x, C_REF), (t_x, C_TRUTH), (e_x, C_TACO), (m_x, C_EMOT)]:
        ax_top.hist(data, bins=bins_x, density=True,
                    alpha=0.55, color=col, edgecolor="white", linewidth=0.3)
    ax_top.set_ylabel("Density", fontsize=9)
    ax_top.tick_params(labelbottom=False, labelsize=8)
    ax_top.set_title(
        f"Cross-Maturity Option Price Coupling — ATM Strike  (K={STRIKE_LABELS[ki]})",
        fontsize=13, fontweight="bold", pad=8,
    )
    ax_top.spines[["top", "right"]].set_visible(False)

    # ── Right histogram: T2 marginal ─────────────────────────────────────────
    all_v2 = np.concatenate([r_y, t_y, e_y, m_y])
    bins_y = np.linspace(all_v2.min(), all_v2.max(), 50)
    for data, col in [(r_y, C_REF), (t_y, C_TRUTH), (e_y, C_TACO), (m_y, C_EMOT)]:
        ax_right.hist(data, bins=bins_y, orientation="horizontal",
                      density=True, alpha=0.55, color=col,
                      edgecolor="white", linewidth=0.3)
    ax_right.set_xlabel("Density", fontsize=9)
    ax_right.tick_params(labelleft=False, labelsize=8)
    ax_right.spines[["top", "right"]].set_visible(False)

    # ── Info panel: summary stats ─────────────────────────────────────────────
    def _fmt(data: np.ndarray) -> str:
        return f"µ={data.mean():.4f}\nσ={data.std():.4f}"

    info_text = (
        "T1 ATM call price\n"
        "─────────────────\n"
        f"Calm (ref):  {_fmt(r_x)}\n\n"
        f"Stress:      {_fmt(t_x)}\n\n"
        f"TACO-EOT:    {_fmt(e_x)}\n\n"
        f"TACO-EMOT:   {_fmt(m_x)}"
    )
    ax_info.text(0.05, 0.95, info_text,
                 transform=ax_info.transAxes,
                 fontsize=7.5, va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4f8",
                           edgecolor="#aab4c0", alpha=0.95))

    fig.patch.set_facecolor("white")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ── Per-strike comparison ─────────────────────────────────────────────────────

def plot_strike_comparison(
    ref_x:    np.ndarray,
    ref_y:    np.ndarray,
    eot_x:    np.ndarray,
    eot_y:    np.ndarray,
    emot_x:   np.ndarray,
    emot_y:   np.ndarray,
    truth_x:  np.ndarray,
    truth_y:  np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Per-strike mean call price: T1 (top row) and T2 (bottom row).

    5 columns (one per strike), 2 rows (T1 / T2).
    Each cell is a violin/box summary across simulated samples.
    """
    n_strikes = ref_x.shape[1]
    fig, axes = plt.subplots(2, n_strikes, figsize=(14, 7), sharey="row")

    methods = [
        (ref_x,   ref_y,   C_REF,   "Reference"),
        (truth_x, truth_y, C_TRUTH, "True stress"),
        (eot_x,   eot_y,   C_TACO,  "TACO-EOT"),
        (emot_x,  emot_y,  C_EMOT,  "TACO-EMOT"),
    ]

    for row, (mat_label, getter) in enumerate(
        [("T1 (1-Month)", lambda mx, my: mx),
         ("T2 (3-Month)", lambda mx, my: my)]
    ):
        for ki in range(n_strikes):
            ax = axes[row, ki]
            data_list = [getter(mx, my)[:, ki] for mx, my, _, _ in methods]
            colors    = [c for _, _, c, _ in methods]
            labels    = [l for _, _, _, l in methods]

            parts = ax.violinplot(
                data_list,
                positions=range(len(methods)),
                showmedians=True,
                showextrema=False,
                widths=0.6,
            )
            for pc, col in zip(parts["bodies"], colors):
                pc.set_facecolor(col)
                pc.set_alpha(0.55)
            parts["cmedians"].set_colors(colors)
            parts["cmedians"].set_linewidth(2.5)

            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(
                ["Ref", "Truth", "EOT", "EMOT"],
                rotation=30, ha="right", fontsize=7,
            )
            if ki == 0:
                ax.set_ylabel(f"{mat_label}\nCall price", fontsize=9)
            ax.set_title(STRIKE_LABELS[ki], fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            add_panel_label(ax, f"{'ab'[row]}{ki+1}", x=-0.18, y=1.10)

    # Single legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, alpha=0.7, label=l) for _, _, c, l in methods]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Call Price Distributions by Strike — T1 vs T2",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ── Martingale constraint verification ───────────────────────────────────────

def plot_martingale_verification(
    eot_x:   np.ndarray,   # (N, 5) — EOT T1 prices
    eot_y:   np.ndarray,   # (N, 5) — EOT T2 prices
    emot_x:  np.ndarray,   # (N, 5) — EMOT T1 prices
    emot_y:  np.ndarray,   # (N, 5) — EMOT T2 prices
    truth_x: np.ndarray,
    truth_y: np.ndarray,
    n_bins: int = 12,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Verify E[Y | X ≈ x] for each method at the ATM strike.

    For EOT: E[Y|X=x] - x should be non-zero (no martingale constraint).
    For EMOT: E[Y|X=x] - x should be ≈ 0 (martingale enforced).

    Layout: 1×2
        (a) Scatter of conditional mean: E[Y|X∈bin] vs bin centre
        (b) Conditional bias E[Y|X=x] − x vs X for each method
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ki = ATM_IDX

    def _conditional_means(xx, yy, n_bins):
        """Bin X into n_bins quantile bins; return (bin_centres, E[Y|bin])."""
        edges   = np.quantile(xx, np.linspace(0, 1, n_bins + 1))
        centres = (edges[:-1] + edges[1:]) / 2
        means   = np.array([
            yy[(xx >= edges[i]) & (xx < edges[i + 1])].mean()
            if np.any((xx >= edges[i]) & (xx < edges[i + 1])) else np.nan
            for i in range(n_bins)
        ])
        return centres, means

    datasets = [
        (eot_x[:, ki],  eot_y[:, ki],  C_TACO,  "TACO-EOT",  "--"),
        (emot_x[:, ki], emot_y[:, ki], C_EMOT,  "TACO-EMOT", "-"),
        (truth_x[:, ki],truth_y[:, ki],C_TRUTH, "True stress","-."),
    ]

    # ── (a) Conditional-mean scatter ─────────────────────────────────────────
    ax = axes[0]
    for xx, yy, col, lbl, ls in datasets:
        centres, means = _conditional_means(xx, yy, n_bins)
        mask = ~np.isnan(means)
        ax.plot(centres[mask], means[mask], color=col, lw=2.2, ls=ls, label=lbl,
                marker="o", markersize=5, zorder=3)

    all_x = np.concatenate([d[0] for d in datasets])
    lo, hi = all_x.min() * 0.97, all_x.max() * 1.03
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=1.2, ls=":",
            label="E[Y|X]=X (ideal martingale)")
    ax.set_xlabel(f"T1 ATM call price X  ({STRIKE_LABELS[ki]})")
    ax.set_ylabel(f"E[T2 ATM call price Y | X]  ({STRIKE_LABELS[ki]})")
    ax.set_title("Conditional Mean E[Y | X] vs X", fontweight="bold")
    ax.legend(fontsize=8)
    add_panel_label(ax, "(a)")

    # ── (b) Conditional bias E[Y|X] − X ──────────────────────────────────────
    ax = axes[1]
    for xx, yy, col, lbl, ls in datasets:
        centres, means = _conditional_means(xx, yy, n_bins)
        mask = ~np.isnan(means)
        bias = means[mask] - centres[mask]
        ax.plot(centres[mask], bias, color=col, lw=2.2, ls=ls, label=lbl,
                marker="o", markersize=5, zorder=3)

    ax.axhline(0, color="#888888", lw=1.2, ls=":", label="Zero bias (martingale)")
    ax.set_xlabel(f"T1 ATM call price X  ({STRIKE_LABELS[ki]})")
    ax.set_ylabel("Conditional bias  E[Y|X] − X")
    ax.set_title("Martingale Constraint Verification\nE[Y|X=x] − x  (should be 0 for EMOT)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    add_panel_label(ax, "(b)")

    fig.suptitle(
        "Martingale Constraint Verification — EOT vs EMOT",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ── Calendar-spread distribution ──────────────────────────────────────────────

def plot_calendar_spread(
    ref_x:    np.ndarray,
    ref_y:    np.ndarray,
    eot_x:    np.ndarray,
    eot_y:    np.ndarray,
    emot_x:   np.ndarray,
    emot_y:   np.ndarray,
    truth_x:  np.ndarray,
    truth_y:  np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Distribution of calendar spread C(T2) − C(T1) per strike.

    No-arbitrage requires C(T2) ≥ C(T1) for all strikes (more time = more value).
    The EMOT martingale constraint should better preserve this property.

    Layout: 1×5 (one column per strike).
    """
    n_strikes = ref_x.shape[1]
    fig, axes = plt.subplots(1, n_strikes, figsize=(15, 4.5), sharey=False)

    methods = [
        (ref_x,   ref_y,   C_REF,   "Reference"),
        (truth_x, truth_y, C_TRUTH, "True stress"),
        (eot_x,   eot_y,   C_TACO,  "TACO-EOT"),
        (emot_x,  emot_y,  C_EMOT,  "TACO-EMOT"),
    ]

    for ki, ax in enumerate(axes):
        spreads = [my[:, ki] - mx[:, ki] for mx, my, _, _ in methods]
        colors  = [c for _, _, c, _ in methods]
        labels  = [l for _, _, _, l in methods]

        all_s = np.concatenate(spreads)
        bins  = np.linspace(min(all_s.min(), 0) * 1.05,
                            all_s.max() * 1.05, 40)

        for sp, col, lbl in zip(spreads, colors, labels):
            ax.hist(sp, bins=bins, density=True, alpha=0.55,
                    color=col, edgecolor="white", linewidth=0.3, label=lbl)

        ax.axvline(0, color="#333333", lw=1.5, ls="--",
                   label="No-arb floor (=0)")
        violation = [100 * (sp < 0).mean() for sp in spreads]
        ax.text(0.97, 0.97,
                "Violations:\n" + "\n".join(
                    f"{l[:5]}: {v:.1f}%" for l, v in zip(labels, violation)
                ),
                transform=ax.transAxes, fontsize=7, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        ax.set_xlabel("Calendar spread\nC(T2) − C(T1)")
        if ki == 0:
            ax.set_ylabel("Density")
        ax.set_title(STRIKE_LABELS[ki], fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        add_panel_label(ax, f"({chr(97 + ki)})")

    # Legend from first axis
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles[:4], lbls[:4], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Calendar Spread C(T2) − C(T1) — No-Arbitrage Check",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ── All-in-one summary figure ─────────────────────────────────────────────────

def plot_options_summary(
    ref_x:    np.ndarray,
    ref_y:    np.ndarray,
    eot_x:    np.ndarray,
    eot_y:    np.ndarray,
    emot_x:   np.ndarray,
    emot_y:   np.ndarray,
    truth_x:  np.ndarray,
    truth_y:  np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Comprehensive 3×3 paper figure combining the key diagnostics.

    Layout:
    ┌──────────────────────┬──────────────────────┬────────────────────┐
    │ (a) Joint ATM scatter│ (b) T1 marginal      │ (c) T2 marginal    │
    │     + density contour│     ATM distribution │     ATM distribut. │
    ├──────────────────────┼──────────────────────┼────────────────────┤
    │ (d) Cond. mean E[Y|X]│ (e) Calendar spread  │ (f) Per-strike     │
    │     (martingale check│     ATM strike       │     mean prices    │
    ├──────────────────────┼──────────────────────┼────────────────────┤
    │ (g) ATM T1-T2 scatter│ (h) Spread violin    │ (i) Cond. bias     │
    │     coloured by meth.│     all 5 strikes    │     EMOT vs EOT    │
    └──────────────────────┴──────────────────────┴────────────────────┘
    """
    fig = plt.figure(figsize=(16, 13))
    gs  = gridspec.GridSpec(3, 3, hspace=0.40, wspace=0.35)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]

    ki = ATM_IDX
    methods_xy = [
        (ref_x,   ref_y,   C_REF,   "Reference (calm)"),
        (truth_x, truth_y, C_TRUTH, "True stressed"),
        (eot_x,   eot_y,   C_TACO,  "TACO-EOT"),
        (emot_x,  emot_y,  C_EMOT,  "TACO-EMOT"),
    ]

    # ── (a) Joint ATM scatter ─────────────────────────────────────────────────
    ax = axes[0][0]
    for mx, my, col, lbl in methods_xy:
        ax.scatter(mx[:, ki], my[:, ki], s=5, alpha=0.30,
                   color=col, label=lbl, linewidths=0)
    xv = np.concatenate([m[0][:, ki] for m in methods_xy])
    lo, hi = xv.min() * 0.97, xv.max() * 1.03
    ax.plot([lo, hi], [lo, hi], color="#888", lw=1.0, ls=":",
            label="Martingale (E[Y]=X)")
    ax.set_xlabel("T1 ATM call price"); ax.set_ylabel("T2 ATM call price")
    ax.set_title("(a) Joint ATM Coupling", fontweight="bold")
    ax.legend(fontsize=7, markerscale=2.5, loc="upper left")

    # ── (b) T1 ATM marginal ───────────────────────────────────────────────────
    ax = axes[0][1]
    bins = np.linspace(xv.min(), xv.max(), 45)
    for mx, my, col, lbl in methods_xy:
        ax.hist(mx[:, ki], bins=bins, density=True, alpha=0.55,
                color=col, label=lbl, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("T1 ATM call price"); ax.set_ylabel("Density")
    ax.set_title("(b) T1 ATM Marginal", fontweight="bold")

    # ── (c) T2 ATM marginal ───────────────────────────────────────────────────
    ax = axes[0][2]
    yv = np.concatenate([m[1][:, ki] for m in methods_xy])
    bins2 = np.linspace(yv.min(), yv.max(), 45)
    for mx, my, col, lbl in methods_xy:
        ax.hist(my[:, ki], bins=bins2, density=True, alpha=0.55,
                color=col, label=lbl, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("T2 ATM call price"); ax.set_ylabel("Density")
    ax.set_title("(c) T2 ATM Marginal", fontweight="bold")

    # ── (d) Conditional mean E[Y | X] ─────────────────────────────────────────
    ax = axes[1][0]
    n_bins = 10

    def _cond_mean(xx, yy, n_bins):
        edges = np.quantile(xx, np.linspace(0, 1, n_bins + 1))
        centres = (edges[:-1] + edges[1:]) / 2
        means = np.array([
            yy[(xx >= edges[i]) & (xx < edges[i + 1])].mean()
            if np.any((xx >= edges[i]) & (xx < edges[i + 1])) else np.nan
            for i in range(n_bins)
        ])
        return centres, means

    for mx, my, col, lbl in methods_xy[1:]:   # skip reference
        c, m = _cond_mean(mx[:, ki], my[:, ki], n_bins)
        mask = ~np.isnan(m)
        ax.plot(c[mask], m[mask], color=col, lw=2.2, label=lbl, marker="o", ms=5)
    lo2 = min(mx[:, ki].min() for mx, _, _, _ in methods_xy)
    hi2 = max(mx[:, ki].max() for mx, _, _, _ in methods_xy)
    ax.plot([lo2, hi2], [lo2, hi2], color="#888", lw=1.2, ls=":", label="E[Y]=X")
    ax.set_xlabel("T1 ATM call price"); ax.set_ylabel("E[T2 ATM | T1]")
    ax.set_title("(d) Conditional Mean E[Y | X]", fontweight="bold")
    ax.legend(fontsize=7)

    # ── (e) ATM calendar spread distribution ─────────────────────────────────
    ax = axes[1][1]
    all_sp = np.concatenate([my[:, ki] - mx[:, ki] for mx, my, _, _ in methods_xy])
    bins_s = np.linspace(all_sp.min() * 1.05, all_sp.max() * 1.05, 45)
    for mx, my, col, lbl in methods_xy:
        ax.hist(my[:, ki] - mx[:, ki], bins=bins_s, density=True, alpha=0.55,
                color=col, label=lbl, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="#333", lw=1.5, ls="--", label="Floor=0")
    ax.set_xlabel("C(T2) − C(T1)  ATM"); ax.set_ylabel("Density")
    ax.set_title("(e) Calendar Spread (ATM)", fontweight="bold")
    ax.legend(fontsize=7)

    # ── (f) Mean call price per strike ────────────────────────────────────────
    ax = axes[1][2]
    n_strikes = ref_x.shape[1]
    x_pos = np.arange(n_strikes)
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, len(methods_xy)) * width
    for i, (mx, my, col, lbl) in enumerate(methods_xy):
        means_t1 = mx.mean(axis=0)
        ax.bar(x_pos + offsets[i], means_t1, width, color=col, alpha=0.80, label=lbl)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(STRIKE_LABELS, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean T1 call price")
    ax.set_title("(f) Mean T1 Price per Strike", fontweight="bold")
    ax.legend(fontsize=7)

    # ── (g) All-strike joint scatter (ATM vs OTM -10%) ────────────────────────
    ax = axes[2][0]
    ki2 = 1   # k = -10% OTM put
    for mx, my, col, lbl in methods_xy:
        ax.scatter(mx[:, ki], mx[:, ki2], s=5, alpha=0.30, color=col, label=lbl, linewidths=0)
    ax.set_xlabel(f"T1 ATM ({STRIKE_LABELS[ki]})"); ax.set_ylabel(f"T1 OTM ({STRIKE_LABELS[ki2]})")
    ax.set_title("(g) T1 ATM vs OTM Coupling", fontweight="bold")
    ax.legend(fontsize=7, markerscale=2.5)

    # ── (h) Spread violin all strikes ────────────────────────────────────────
    ax = axes[2][1]
    positions_base = np.arange(n_strikes) * (len(methods_xy) + 1)
    for i, (mx, my, col, lbl) in enumerate(methods_xy):
        positions = positions_base + i
        parts = ax.violinplot(
            [my[:, ki_] - mx[:, ki_] for ki_ in range(n_strikes)],
            positions=positions,
            showmedians=True, showextrema=False, widths=0.7,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(col); pc.set_alpha(0.5)
        parts["cmedians"].set_color(col); parts["cmedians"].set_linewidth(2)
    ax.axhline(0, color="#444", lw=1.2, ls="--")
    ax.set_xticks(positions_base + 1.5)
    ax.set_xticklabels(STRIKE_LABELS, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("C(T2) − C(T1)")
    ax.set_title("(h) Calendar Spread by Strike", fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=c, alpha=0.7, label=l) for _, _, c, l in methods_xy],
        fontsize=7, loc="upper right",
    )

    # ── (i) Conditional bias EMOT vs EOT ─────────────────────────────────────
    ax = axes[2][2]
    for mx, my, col, lbl in [(eot_x, eot_y, C_TACO, "TACO-EOT"),
                              (emot_x, emot_y, C_EMOT, "TACO-EMOT"),
                              (truth_x, truth_y, C_TRUTH, "True stress")]:
        c, cm = _cond_mean(mx[:, ki], my[:, ki], n_bins)
        mask  = ~np.isnan(cm)
        bias  = cm[mask] - c[mask]
        ax.plot(c[mask], bias, color=col, lw=2.2, label=lbl, marker="o", ms=5)
    ax.axhline(0, color="#888", lw=1.2, ls=":", label="Zero bias (martingale)")
    ax.set_xlabel("T1 ATM call price")
    ax.set_ylabel("Conditional bias  E[Y|X] − X")
    ax.set_title("(i) Martingale Constraint Bias", fontweight="bold")
    ax.legend(fontsize=7)

    fig.suptitle(
        "Cross-Maturity Options Coupling — EOT vs EMOT (Martingale Constraint)",
        fontsize=14, fontweight="bold", y=1.005,
    )
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
