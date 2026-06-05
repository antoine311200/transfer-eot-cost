"""Paper-quality figures for Experiment E1: IV Surface Regime Transfer.

Figure types
------------
plot_smile_comparison    — 4-panel: ref smile / TACO / truth / all overlaid
plot_surface_heatmap     — 2D colour grid (moneyness × maturity) for each method
plot_coupling_evolution  — how the (σ_atm, skew) joint coupling evolves over TACO iterations
plot_iv_convergence      — W2_nu + RMSE vs iteration
plot_3d_surface          — 3D surface (K, T) → σ for ref / TACO / truth
plot_skew_slope          — ATM-normalised skew slope per maturity (key practitioner metric)
plot_term_structure      — ATM vol at 1M vs 3M scatter (shows term-structure transfer)
plot_arbitrage_checks    — butterfly + calendar spread violation rates vs TACO iteration
plot_daily_moves         — distribution of daily surface changes (the transferred dynamics)
"""

from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .theme import apply_theme, C_REF, C_TACO, C_TRUTH, C_PRE, C_OTCFM, add_panel_label, iteration_cmap
from ..data.heston import MONEYNESS, MATURITIES, D_IV


apply_theme()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_smile(iv_surface: np.ndarray, maturity_idx: int) -> np.ndarray:
    """Extract the smile (5-point) for a given maturity index from a D_IV surface."""
    n_m = len(MONEYNESS)
    return iv_surface[maturity_idx * n_m: (maturity_idx + 1) * n_m]


def _mean_smile(surfaces: np.ndarray, maturity_idx: int) -> np.ndarray:
    """Mean smile over many surface samples."""
    smiles = np.array([_to_smile(s, maturity_idx) for s in surfaces])
    return smiles.mean(axis=0), smiles.std(axis=0)


# ── Main figures ─────────────────────────────────────────────────────────────

def plot_smile_comparison(
    ref_surfaces:   np.ndarray,   # (N, D_IV) — reference regime samples
    taco_surfaces:  np.ndarray,   # (N, D_IV) — TACO-generated stressed
    truth_surfaces: np.ndarray,   # (N, D_IV) — true stressed regime samples
    pre_surfaces:   Optional[np.ndarray] = None,   # pretrained-only
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """4-panel figure: IV smile for each method at 1M and 3M maturities.

    Layout: 2 rows (1M / 3M) × 2 columns (single-method vs all overlaid).
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
    mat_names = ["1-Month", "3-Month"]

    for row, (mat_idx, mat_name) in enumerate(zip(range(len(MATURITIES)), mat_names)):
        ref_m,  ref_s  = _mean_smile(ref_surfaces,   mat_idx)
        tco_m,  tco_s  = _mean_smile(taco_surfaces,  mat_idx)
        tru_m,  tru_s  = _mean_smile(truth_surfaces, mat_idx)

        # ── Left: reference vs truth (the problem) ──
        ax = axes[row, 0]
        ax.plot(MONEYNESS * 100, ref_m * 100, color=C_REF, lw=2.2, label="Reference (calm)")
        ax.fill_between(MONEYNESS * 100, (ref_m - ref_s) * 100, (ref_m + ref_s) * 100,
                        color=C_REF, alpha=0.15)
        ax.plot(MONEYNESS * 100, tru_m * 100, color=C_TRUTH, lw=2.2, ls="--",
                label="True stressed target")
        ax.fill_between(MONEYNESS * 100, (tru_m - tru_s) * 100, (tru_m + tru_s) * 100,
                        color=C_TRUTH, alpha=0.15)
        ax.set_xlabel("Log-moneyness (%)")
        ax.set_ylabel("Implied vol (%)")
        ax.set_title(f"{mat_name} — The Transfer Problem")
        ax.legend()
        add_panel_label(ax, "abcd"[row * 2])

        # ── Right: all methods overlaid ──
        ax = axes[row, 1]
        ax.plot(MONEYNESS * 100, ref_m * 100, color=C_REF,   lw=1.6, ls=":",  alpha=0.7, label="Reference (calm)")
        ax.plot(MONEYNESS * 100, tru_m * 100, color=C_TRUTH, lw=2.4, ls="--", label="True stressed")
        if pre_surfaces is not None:
            pre_m, _ = _mean_smile(pre_surfaces, mat_idx)
            ax.plot(MONEYNESS * 100, pre_m * 100, color=C_PRE, lw=1.8, ls="-.", label="Pretrained-only")
        ax.plot(MONEYNESS * 100, tco_m * 100, color=C_TACO, lw=2.4, label="TACO (ours)")
        ax.fill_between(MONEYNESS * 100, (tco_m - tco_s) * 100, (tco_m + tco_s) * 100,
                        color=C_TACO, alpha=0.18)
        ax.set_xlabel("Log-moneyness (%)")
        ax.set_ylabel("Implied vol (%)")
        ax.set_title(f"{mat_name} — All Methods")
        ax.legend()
        add_panel_label(ax, "abcd"[row * 2 + 1])

    fig.suptitle("IV Surface Regime Transfer — Implied Volatility Smile Comparison",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_surface_heatmap(
    ref_mean:   np.ndarray,   # (D_IV,)
    taco_mean:  np.ndarray,   # (D_IV,)
    truth_mean: np.ndarray,   # (D_IV,)
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """2D heatmap (moneyness × maturity) for reference, TACO, truth, and error."""
    n_m, n_T = len(MONEYNESS), len(MATURITIES)

    def reshape(v):
        # v is ordered [T0 × M0..M4, T1 × M0..M4] → shape (n_T, n_m)
        return v.reshape(n_T, n_m) * 100  # to percent

    ref_g   = reshape(ref_mean)
    taco_g  = reshape(taco_mean)
    truth_g = reshape(truth_mean)
    err_g   = np.abs(taco_g - truth_g)

    mat_labels = [f"{int(round(T * 12))}M" for T in MATURITIES]
    mon_labels = [f"{int(m * 100):+d}%" for m in MONEYNESS]

    vmin = min(ref_g.min(), truth_g.min())
    vmax = max(ref_g.max(), truth_g.max())

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    datasets = [
        (ref_g,   "Reference (calm)",   "Blues",       vmin, vmax),
        (truth_g, "True stressed",       "Reds",        vmin, vmax),
        (taco_g,  "TACO (ours)",         "Greens",      vmin, vmax),
        (err_g,   "|TACO – Truth| (pp)", "Oranges",     0,    err_g.max() + 0.05),
    ]
    for i, (data, title, cmap, lo, hi) in enumerate(datasets):
        ax = axes[i]
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=lo, vmax=hi,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.85, label="IV (%)")
        ax.set_xticks(range(n_m))
        ax.set_xticklabels(mon_labels, rotation=30, ha="right")
        ax.set_yticks(range(n_T))
        ax.set_yticklabels(mat_labels)
        ax.set_xlabel("Log-moneyness")
        ax.set_ylabel("Maturity")
        ax.set_title(title, fontweight="bold")
        add_panel_label(ax, "abcd"[i])

    fig.suptitle("IV Surface Heatmap Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_coupling_evolution(
    coupling_history: list[np.ndarray],   # list of (N, D_IV) arrays, one per logged iteration
    truth_surfaces:   np.ndarray,          # (N, D_IV) — true stressed samples
    iterations:       Sequence[int],
    mat_idx: int = 0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Show how the TACO-generated IV smile evolves across iterations.

    Plots ATM vol distribution + smile at 1M for selected checkpoints.
    """
    n_panels = min(len(coupling_history), 6)
    sel = np.round(np.linspace(0, len(coupling_history) - 1, n_panels)).astype(int)
    colors = iteration_cmap(n_panels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: ATM vol marginal distribution evolution ──
    ax = axes[0]
    atm_idx = mat_idx * len(MONEYNESS) + 2  # ATM = moneyness index 2 (m=0)
    truth_atm = truth_surfaces[:, atm_idx] * 100

    ax.hist(truth_atm, bins=30, density=True, alpha=0.4, color=C_TRUTH,
            label="True stressed", zorder=3)
    for k, (ci, color) in enumerate(zip(sel, colors)):
        atm = coupling_history[ci][:, atm_idx] * 100
        ax.hist(atm, bins=30, density=True, alpha=0.35, color=color,
                label=f"TACO iter {iterations[ci]}" if k in (0, n_panels-1) else "_no_legend_")

    ax.set_xlabel("ATM Implied Vol (%)")
    ax.set_ylabel("Density")
    ax.set_title("ATM Vol Marginal — Evolution")
    ax.legend(fontsize=8)
    add_panel_label(ax, "(a)")

    # ── Right: smile evolution ──
    ax = axes[1]
    tru_m, _ = _mean_smile(truth_surfaces, mat_idx)
    ax.plot(MONEYNESS * 100, tru_m * 100, color=C_TRUTH, lw=2.5, ls="--",
            label="True stressed", zorder=10)
    for k, (ci, color) in enumerate(zip(sel, colors)):
        sm_m, _ = _mean_smile(coupling_history[ci], mat_idx)
        label = f"Iter {iterations[ci]}" if k in (0, n_panels-1) else "_no_legend_"
        ax.plot(MONEYNESS * 100, sm_m * 100, color=color, lw=1.6, alpha=0.8, label=label)

    ax.set_xlabel("Log-moneyness (%)")
    ax.set_ylabel("Implied Vol (%)")
    ax.set_title("1M IV Smile — TACO Iteration Evolution")
    ax.legend(fontsize=8)
    add_panel_label(ax, "(b)")

    fig.suptitle("IV Surface Coupling Evolution Across TACO Iterations",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_iv_convergence(
    iterations:   np.ndarray,
    w2_nu:        np.ndarray,
    rmse:         Optional[np.ndarray] = None,
    pre_w2:       Optional[float] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Convergence plot: W2_ν and surface RMSE vs TACO iteration."""
    has_rmse = rmse is not None
    fig, axes = plt.subplots(1, 1 + int(has_rmse), figsize=(9 if has_rmse else 5, 4))
    if not has_rmse:
        axes = [axes]

    ax = axes[0]
    ax.plot(iterations, w2_nu, color=C_TACO, lw=2.2, label="TACO")
    if pre_w2 is not None:
        ax.axhline(pre_w2, color=C_PRE, ls="--", lw=1.8, label="Pretrained-only")
    ax.set_xlabel("TACO Iteration")
    ax.set_ylabel("SW₂ (target marginal)")
    ax.set_title("Target Distribution Distance")
    ax.legend()
    add_panel_label(ax, "(a)")

    if has_rmse:
        ax2 = axes[1]
        ax2.plot(iterations, rmse * 100, color=C_TACO, lw=2.2, label="TACO")
        ax2.set_xlabel("TACO Iteration")
        ax2.set_ylabel("IV Surface RMSE (pp)")
        ax2.set_title("Implied Vol Surface RMSE")
        ax2.legend()
        add_panel_label(ax2, "(b)")

    fig.suptitle("IV Surface Transfer — Convergence", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def compute_iv_rmse(generated: np.ndarray, truth: np.ndarray) -> float:
    """RMSE between mean generated and mean true IV surface (in vol units)."""
    return float(np.sqrt(np.mean((generated.mean(0) - truth.mean(0)) ** 2)))


# ── Convexity projection ──────────────────────────────────────────────────────

def _project_smile_convex(smile: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Project a 5-point smile to satisfy convexity (H2 ≥ ε).

    Fits σ(k) = a + b·k + c·k²/2 by constrained least squares, then clips
    c = max(c, epsilon).  With c > 0 the quadratic is convex everywhere and
    the second finite-difference Δ²σ = c·h² > 0 at every interior point.
    """
    k = MONEYNESS   # shape (5,)
    A = np.column_stack([np.ones_like(k), k, k ** 2 / 2])

    beta, _, _, _ = np.linalg.lstsq(A, smile, rcond=None)

    if beta[2] >= epsilon:
        return np.clip(A @ beta, 1e-3, 2.0)

    # Fix curvature at floor, refit intercept + slope
    beta[2] = epsilon
    residual = smile - epsilon * k ** 2 / 2
    beta[:2], _, _, _ = np.linalg.lstsq(A[:, :2], residual, rcond=None)
    return np.clip(A @ beta, 1e-3, 2.0)


def project_surfaces_convex(surfaces: np.ndarray) -> np.ndarray:
    """Project every surface's smiles to be convex. Returns same shape array."""
    n_m = len(MONEYNESS)
    out = surfaces.copy()
    for i, surf in enumerate(out):
        for T in range(len(MATURITIES)):
            base = T * n_m
            out[i, base: base + n_m] = _project_smile_convex(surf[base: base + n_m])
    return out


# ── Derived surface analytics ─────────────────────────────────────────────────

def _atm_col(surfaces: np.ndarray) -> np.ndarray:
    """Extract ATM (k=0) vol for each maturity. Returns (N, n_T)."""
    n_m = len(MONEYNESS)
    atm_idx = np.where(MONEYNESS == 0.0)[0][0]
    return np.stack([surfaces[:, T * n_m + atm_idx] for T in range(len(MATURITIES))], axis=1)


def _skew_slope(surfaces: np.ndarray) -> np.ndarray:
    """25-delta put skew slope ≈ σ(k=-0.10) - σ(k=0) per maturity. Returns (N, n_T)."""
    n_m = len(MONEYNESS)
    k_minus = np.where(np.isclose(MONEYNESS, -0.10))[0][0]   # k = -10% put
    k_atm   = np.where(MONEYNESS == 0.0)[0][0]
    slopes = []
    for T in range(len(MATURITIES)):
        base = T * n_m
        slopes.append(surfaces[:, base + k_minus] - surfaces[:, base + k_atm])
    return np.stack(slopes, axis=1)   # (N, n_T)


def _check_butterfly(surfaces: np.ndarray) -> np.ndarray:
    """Butterfly arbitrage: smile must be convex in k. Returns (N, n_T) bool (True = violated)."""
    n_m = len(MONEYNESS)
    violations = []
    for T in range(len(MATURITIES)):
        base = T * n_m
        smile = surfaces[:, base: base + n_m]  # (N, n_m)
        # Second finite difference < 0 at any interior point → non-convex
        d2 = np.diff(smile, n=2, axis=1)   # (N, n_m-2)
        viol = (d2 < 0).any(axis=1)         # (N,)
        violations.append(viol)
    return np.stack(violations, axis=1)


def _check_calendar(surfaces: np.ndarray) -> np.ndarray:
    """Calendar spread arbitrage: total implied variance must increase with T.
    Returns (N,) bool (True = violated)."""
    n_m = len(MONEYNESS)
    # Total variance = σ²·T; check σ²(T=3M)·(3/12) >= σ²(T=1M)·(1/12) at each k
    violations = np.zeros(len(surfaces), dtype=bool)
    for ki in range(n_m):
        var_1m = surfaces[:, ki]         ** 2 * MATURITIES[0]
        var_3m = surfaces[:, n_m + ki]   ** 2 * MATURITIES[1]
        violations |= (var_3m < var_1m)
    return violations


def compute_arbitrage_violations(surfaces: np.ndarray) -> dict:
    """Return fraction of surfaces with butterfly or calendar violations."""
    bf = _check_butterfly(surfaces)   # (N, n_T)
    cal = _check_calendar(surfaces)   # (N,)
    return {
        "butterfly_1m": float(bf[:, 0].mean()),
        "butterfly_3m": float(bf[:, 1].mean()),
        "calendar":     float(cal.mean()),
    }


def compute_skew_rmse(generated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-maturity RMSE of the 25-delta skew slope (in pp)."""
    gen_skew  = _skew_slope(generated).mean(0)
    tru_skew  = _skew_slope(truth).mean(0)
    return np.abs(gen_skew - tru_skew) * 100   # (n_T,)


# ── New practitioner-oriented figures ────────────────────────────────────────

def plot_3d_surface(
    ref_surfaces:   np.ndarray,
    taco_surfaces:  np.ndarray,
    truth_surfaces: np.ndarray,
    pre_surfaces:   Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """3D bar / wireframe of the mean IV surface for ref, TACO, and truth.

    Shows the practitioner's most intuitive view: the actual surface shape.
    """
    n_m, n_T = len(MONEYNESS), len(MATURITIES)
    K_pct = MONEYNESS * 100
    T_mo  = np.array([int(round(T * 12)) for T in MATURITIES])

    def mean_surface(surfaces):
        return surfaces.mean(0).reshape(n_T, n_m) * 100   # (n_T, n_m) in %

    datasets = [
        (mean_surface(ref_surfaces),  "Reference (calm)",  C_REF),
        (mean_surface(truth_surfaces),"True stressed",      C_TRUTH),
    ]
    if pre_surfaces is not None:
        datasets.insert(2, (mean_surface(pre_surfaces), "Pretrained-only", C_PRE))
    datasets.append((mean_surface(taco_surfaces), "TACO (ours)", C_TACO))

    ncols = len(datasets)
    fig = plt.figure(figsize=(4 * ncols, 4.5))
    K_mesh, T_mesh = np.meshgrid(K_pct, T_mo, indexing="ij")  # (n_m, n_T)

    vmin = min(d[0].min() for d in datasets)
    vmax = max(d[0].max() for d in datasets)

    for i, (surf, title, color) in enumerate(datasets):
        ax = fig.add_subplot(1, ncols, i + 1, projection="3d")
        # surf is (n_T, n_m); transpose for (n_m, n_T) to align with meshes
        ax.plot_surface(K_mesh, T_mesh, surf.T,
                        alpha=0.85, cmap="RdYlGn_r",
                        vmin=vmin, vmax=vmax, edgecolor="none")
        ax.set_xlabel("Log-moneyness (%)", fontsize=8, labelpad=4)
        ax.set_ylabel("Maturity (M)", fontsize=8, labelpad=4)
        ax.set_zlabel("IV (%)", fontsize=8, labelpad=4)
        ax.set_zlim(vmin, vmax)
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.tick_params(labelsize=7)

    fig.suptitle("IV Surface Regime Transfer — 3D View", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_skew_slope(
    ref_surfaces:   np.ndarray,
    taco_surfaces:  np.ndarray,
    truth_surfaces: np.ndarray,
    pre_surfaces:   Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """25-delta put skew slope per maturity: ref / pretrained / TACO / truth.

    This is the key diagnostic for skew transfer.  A method that merely
    rescales the ATM level will get this wrong; TACO should recover it.
    The skew slope = σ(k=−10%) − σ(k=0%), shown as a bar chart per maturity.
    """
    mat_names = [f"{int(round(T * 12))}M" for T in MATURITIES]
    x = np.arange(len(MATURITIES))
    width = 0.18

    methods = [
        (ref_surfaces,   "Reference (calm)",   C_REF,   -1.5),
        (truth_surfaces, "True stressed",       C_TRUTH, -0.5),
    ]
    if pre_surfaces is not None:
        methods.append((pre_surfaces, "Pretrained-only", C_PRE, 0.5))
        taco_offset = 1.5
    else:
        taco_offset = 0.5
    methods.append((taco_surfaces, "TACO (ours)", C_TACO, taco_offset))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Left: mean skew slope per maturity ──
    ax = axes[0]
    for surfaces, label, color, offset in methods:
        slopes = _skew_slope(surfaces).mean(0) * 100  # pp
        ax.bar(x + offset * width, slopes, width, label=label, color=color, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(mat_names)
    ax.set_xlabel("Maturity"); ax.set_ylabel("25Δ Skew slope (pp)")
    ax.set_title("Mean 25-Delta Put Skew  [σ(k=−10%) − σ(k=0%)]", fontweight="bold")
    ax.legend(fontsize=8)
    add_panel_label(ax, "(a)")

    # ── Right: skew slope scatter (1M vs 3M), shows the joint structure ──
    ax = axes[1]
    for surfaces, label, color, _ in methods:
        s = _skew_slope(surfaces) * 100
        ax.scatter(s[:, 0], s[:, 1], alpha=0.25, s=8, color=color, label=label)
    ax.set_xlabel("1M skew slope (pp)"); ax.set_ylabel("3M skew slope (pp)")
    ax.set_title("1M vs 3M Skew Slope Distribution", fontweight="bold")
    ax.legend(fontsize=8, markerscale=3)
    add_panel_label(ax, "(b)")

    fig.suptitle("IV Skew Slope Comparison — Key Diagnostic for Cost Transfer",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_term_structure(
    ref_surfaces:   np.ndarray,
    taco_surfaces:  np.ndarray,
    truth_surfaces: np.ndarray,
    pre_surfaces:   Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """ATM vol at 1M vs 3M scatter — shows the term-structure joint distribution.

    A key view for options traders who manage term-structure risk (vega by expiry).
    TACO should reproduce the stressed joint (1M ATM, 3M ATM) distribution;
    simple marginal methods only match each maturity independently.
    """
    methods = [
        (ref_surfaces,   "Reference (calm)", C_REF,   10),
        (truth_surfaces, "True stressed",    C_TRUTH,  10),
    ]
    if pre_surfaces is not None:
        methods.insert(2, (pre_surfaces, "Pretrained-only", C_PRE, 10))
    methods.append((taco_surfaces, "TACO (ours)", C_TACO, 10))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: scatter (1M ATM, 3M ATM) ──
    ax = axes[0]
    for surfaces, label, color, size in methods:
        atm = _atm_col(surfaces) * 100  # (N, 2)
        ax.scatter(atm[:, 0], atm[:, 1], alpha=0.3, s=size, color=color, label=label)
    lim_lo = min(m[0]._atm_col_cache if hasattr(m[0], '_atm_col_cache') else
                 _atm_col(m[0]).min() * 100 for m in [(ref_surfaces,), (truth_surfaces,)])
    ax.set_xlabel("1M ATM vol (%)"); ax.set_ylabel("3M ATM vol (%)")
    ax.set_title("Term Structure: 1M vs 3M ATM Vol", fontweight="bold")
    ax.legend(fontsize=8, markerscale=3)
    add_panel_label(ax, "(a)")

    # ── Right: term-structure ratio = 3M/1M (vol term structure slope) ──
    ax = axes[1]
    mat_names = [f"{int(round(T * 12))}M" for T in MATURITIES]
    for surfaces, label, color, _ in methods:
        atm = _atm_col(surfaces) * 100
        # Violin-style: show distribution of each tenor's ATM
        parts = ax.violinplot(
            [atm[:, i] for i in range(len(MATURITIES))],
            positions=range(len(MATURITIES)),
            showmedians=True, showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_alpha(0.35)
        parts["cmedians"].set_color(color)
        parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(MATURITIES))); ax.set_xticklabels(mat_names)
    ax.set_xlabel("Maturity"); ax.set_ylabel("ATM vol (%)")
    ax.set_title("ATM Vol Distribution by Maturity", fontweight="bold")
    add_panel_label(ax, "(b)")

    # Manual legend for violin plot
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=c, alpha=0.7, label=l)
                    for _, l, c, _ in methods]
    ax.legend(handles=legend_items, fontsize=8)

    fig.suptitle("IV Term Structure Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_arbitrage_checks(
    coupling_history: list[np.ndarray],
    truth_surfaces:   np.ndarray,
    iterations:       Sequence[int],
    ref_surfaces:     Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Butterfly and calendar-spread arbitrage violation rate vs TACO iteration.

    Shows both raw flow-model outputs (no convexity constraint) and the
    post-processed surfaces after quadratic smile projection, which brings
    butterfly violations to 0%.  Calendar arbitrage is eliminated by TACO
    because it jointly matches the term-structure marginals.
    """
    iters = list(iterations)

    # Raw violations
    bf_1m_raw = [_check_butterfly(h)[:, 0].mean() * 100 for h in coupling_history]
    bf_3m_raw = [_check_butterfly(h)[:, 1].mean() * 100 for h in coupling_history]
    cal_raw   = [_check_calendar(h).mean()          * 100 for h in coupling_history]

    # Projected violations (apply quadratic projection → should be 0%)
    bf_1m_proj = [_check_butterfly(project_surfaces_convex(h))[:, 0].mean() * 100 for h in coupling_history]
    bf_3m_proj = [_check_butterfly(project_surfaces_convex(h))[:, 1].mean() * 100 for h in coupling_history]
    cal_proj   = [_check_calendar(project_surfaces_convex(h)).mean()          * 100 for h in coupling_history]

    truth_bf1 = _check_butterfly(truth_surfaces)[:, 0].mean() * 100
    truth_bf3 = _check_butterfly(truth_surfaces)[:, 1].mean() * 100
    truth_cal = _check_calendar(truth_surfaces).mean() * 100

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    specs = [
        (axes[0], bf_1m_raw, bf_1m_proj, truth_bf1,
         "1M Butterfly Arb", "(a)",
         "Flow model outputs violate\nconvexity (no constraint);\nquadratic projection → 0%"),
        (axes[1], bf_3m_raw, bf_3m_proj, truth_bf3,
         "3M Butterfly Arb", "(b)", None),
        (axes[2], cal_raw,   cal_proj,   truth_cal,
         "Calendar Spread Arb", "(c)",
         "Calendar arb eliminated by\nTACO (joint term-structure\nmarginal matching)"),
    ]
    for ax, vals_raw, vals_proj, truth_val, title, label, note in specs:
        ax.plot(iters, vals_raw,  color=C_TACO, lw=2.2, label="TACO (raw)")
        ax.plot(iters, vals_proj, color=C_TACO, lw=1.6, ls="--", alpha=0.6,
                label="TACO (after projection)")
        ax.axhline(truth_val, color=C_TRUTH, ls="--", lw=1.8, label="True stressed")
        if ref_surfaces is not None:
            if "Calendar" in title:
                ref_val = _check_calendar(ref_surfaces).mean() * 100
            elif "1M" in title:
                ref_val = _check_butterfly(ref_surfaces)[:, 0].mean() * 100
            else:
                ref_val = _check_butterfly(ref_surfaces)[:, 1].mean() * 100
            ax.axhline(ref_val, color=C_REF, ls=":", lw=1.5, label="Reference (calm)")
        if note:
            ax.text(0.05, 0.97, note, transform=ax.transAxes, fontsize=6.5,
                    va="top", bbox=dict(boxstyle="round,pad=0.25", facecolor="wheat", alpha=0.5))
        ax.set_xlabel("TACO Iteration")
        ax.set_ylabel("Violation rate (%)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7)
        add_panel_label(ax, label)

    fig.suptitle("Static Arbitrage Violation Rates vs TACO Iteration",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_daily_moves(
    ref_x:        np.ndarray, ref_y:        np.ndarray,
    taco_x:       np.ndarray, taco_y:       np.ndarray,
    truth_x:      np.ndarray, truth_y:      np.ndarray,
    transported_x: Optional[np.ndarray] = None,
    transported_y: Optional[np.ndarray] = None,
    save_path:    Optional[Path] = None,
) -> plt.Figure:
    """Distribution of daily IV surface moves (Δσ) — dynamics transfer diagnostic.

    TACO is an EOT solver between source μ and target ν.  When μ ≈ ν (both
    are the stationary stressed-regime distribution), the optimal EOT coupling
    degenerates to near-identity and the coupling's (x, y) daily moves collapse
    to ≈ 0.  This is CORRECT EOT behaviour, not a bug.

    To visualise what TACO *does* learn about dynamics, we include
    `transported_x/y`: the reference calm pairs (x_calm, y_calm) pushed
    through the marginal bridges μ_bridge and ν_bridge.  This shows how
    calm day-to-day moves are *transformed* under the regime change — the
    actually meaningful dynamics diagnostic.

    Panels:
      (a) 1M ATM daily move distribution
      (b) 3M ATM daily move distribution
      (c) Joint 1M × 3M move scatter  — shows cross-maturity coupling structure
      (d) TACO coupling Δ — exposes the μ≈ν degeneracy directly
    """
    atm_idx_1m = np.where(MONEYNESS == 0.0)[0][0]
    atm_idx_3m = len(MONEYNESS) + atm_idx_1m

    def _atm_delta(x, y, idx):
        return (y - x)[:, idx] * 100

    d_ref_1m   = _atm_delta(ref_x,   ref_y,   atm_idx_1m)
    d_truth_1m = _atm_delta(truth_x, truth_y, atm_idx_1m)
    d_ref_3m   = _atm_delta(ref_x,   ref_y,   atm_idx_3m)
    d_truth_3m = _atm_delta(truth_x, truth_y, atm_idx_3m)

    has_transported = (transported_x is not None) and (transported_y is not None)
    if has_transported:
        d_trans_1m = _atm_delta(transported_x, transported_y, atm_idx_1m)
        d_trans_3m = _atm_delta(transported_x, transported_y, atm_idx_3m)

    d_taco_1m  = _atm_delta(taco_x,  taco_y,  atm_idx_1m)
    d_taco_3m  = _atm_delta(taco_x,  taco_y,  atm_idx_3m)

    ncols = 4 if has_transported else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5))

    C_TRANS = "#e67e22"   # orange for transported pairs

    for ax, d_ref, d_truth, d_trans_opt, title, label in [
        (axes[0], d_ref_1m, d_truth_1m,
         d_trans_1m if has_transported else None, "1M ATM Daily Move", "(a)"),
        (axes[1], d_ref_3m, d_truth_3m,
         d_trans_3m if has_transported else None, "3M ATM Daily Move", "(b)"),
    ]:
        all_vals = [d_ref, d_truth]
        if d_trans_opt is not None: all_vals.append(d_trans_opt)
        lo = min(v.min() for v in all_vals)
        hi = max(v.max() for v in all_vals)
        bins = np.linspace(lo, hi, 35)

        ax.hist(d_ref,   bins=bins, density=True, alpha=0.45, color=C_REF,   label="Reference (calm)")
        ax.hist(d_truth, bins=bins, density=True, alpha=0.45, color=C_TRUTH, label="True stressed")
        if d_trans_opt is not None:
            ax.hist(d_trans_opt, bins=bins, density=True, alpha=0.55, color=C_TRANS,
                    label="TACO transported\n(μ/ν bridges on ref)")
        ax.set_xlabel("Daily Δ(ATM vol) (pp)")
        ax.set_ylabel("Density")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7)
        add_panel_label(ax, label)

    # ── Joint scatter ──
    ax = axes[2]
    ax.scatter(d_ref_1m,   d_ref_3m,   s=6, alpha=0.3, color=C_REF,   label="Reference (calm)")
    ax.scatter(d_truth_1m, d_truth_3m, s=6, alpha=0.4, color=C_TRUTH, label="True stressed")
    if has_transported:
        ax.scatter(d_trans_1m, d_trans_3m, s=6, alpha=0.4, color=C_TRANS, label="TACO transported")
    ax.set_xlabel("Δ 1M ATM (pp)")
    ax.set_ylabel("Δ 3M ATM (pp)")
    ax.set_title("Joint Daily Moves: 1M × 3M", fontweight="bold")
    ax.legend(fontsize=7, markerscale=3)
    add_panel_label(ax, "(c)")

    if has_transported:
        # ── TACO coupling Δ — shows degeneracy explicitly ──
        ax = axes[3]
        all_d = np.concatenate([d_ref_1m, d_taco_1m, d_truth_1m])
        bins2 = np.linspace(all_d.min(), all_d.max(), 40)
        ax.hist(d_ref_1m,   bins=bins2, density=True, alpha=0.35, color=C_REF,   label="Reference (calm)")
        ax.hist(d_truth_1m, bins=bins2, density=True, alpha=0.35, color=C_TRUTH, label="True stressed")
        ax.hist(d_taco_1m,  bins=bins2, density=True, alpha=0.60, color=C_TACO,
                label="TACO coupling (x,y)", histtype="step", lw=2.2)
        ax.set_xlabel("Daily Δ(1M ATM vol) (pp)")
        ax.set_ylabel("Density")
        ax.set_title("TACO Coupling Daily Moves\n(degenerates to ≈0 when μ≈ν)", fontweight="bold")
        ax.legend(fontsize=7)
        ax.text(0.05, 0.95,
                "EOT(μ,ν) → identity\nwhen μ≈ν: correct but\ndoes not capture dynamics",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
        add_panel_label(ax, "(d)")

    fig.suptitle("Daily IV Surface Move Distribution — Dynamics Transfer Diagnostic",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
