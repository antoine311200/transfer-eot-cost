#!/usr/bin/env python
"""
plot_wandb_results.py  —  Publication-quality ablation visualiser for DualWCFM
===============================================================================
Fetches W&B runs and produces a comprehensive suite of publication-quality
figures for analysing the effect of sigma, particle_ratio, iterations, and
other hyperparameters on entropic optimal transport quality.

Plot suite
----------
  C  — Convergence curves (one line per hparam value, ±std band)
  S  — Final-metric scatter (hparam vs metric, per-run dots + mean)
  V  — Violin / box plots of final metric distributions
  H  — Annotated heatmap (sigma × particle_ratio joint effect)
  P  — Parallel-coordinates plot (multi-param overview)
  E  — ESS + weight-statistic health over iterations
  X  — Cross-experiment comparison (best run per experiment)
  T  — LaTeX table (mean ± std, bold best per column, booktabs)

Usage
-----
  python plot_wandb_results.py --project exp_2d_simple --entity antoine311200
  python plot_wandb_results.py --project exp_2d_simple --no-heatmap --no-parallel
  python plot_wandb_results.py --project exp_2d_simple --outlier-abs 0
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants — metrics, labels, layout
# ---------------------------------------------------------------------------

# Okabe-Ito palette (accessible, no pure yellow)
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # mauve
    "#56B4E9",  # sky
    "#E69F00",  # amber
    "#6B4C9A",  # purple
    "#555555",  # grey
]

# Figure widths targeting NeurIPS / ICML column sizes
COL_W  = 3.3    # single column (in)
COL_2W = 6.75   # double column (in)
DPI    = 300

MLABEL: Dict[str, str] = {
    "W1_nu":          r"$\mathrm{SW}_1(\hat\nu,\nu)$",
    "W2_nu":          r"$\mathrm{SW}_2(\hat\nu,\nu)$",
    "W1_mu":          r"$\mathrm{SW}_1(\hat\mu,\mu)$",
    "W2_mu":          r"$\mathrm{SW}_2(\hat\mu,\mu)$",
    "mmd_nu":         r"$\mathrm{MMD}(\hat\nu,\nu)$",
    "energy_dist_nu": r"Energy distance",
    "sinkhorn_div":   r"Sinkhorn divergence",
    "map_err_fwd":    r"Map error $\|\hat T - T^*\|$",
    "map_err_bwd":    r"Map error $\|\hat T^{-1} - {T^*}^{-1}\|$",
    "bw2":            r"$\mathrm{BW}_2$ (joint)",
    "fid":            r"FID",
    "kid":            r"KID",
    "weight_ess":     r"ESS (normalised)",
    "weight_mean":    r"Weight mean $\mathbb{E}[w]$",
    "weight_std":     r"Weight std $\sigma(w)$",
}

MLABEL_SHORT: Dict[str, str] = {
    "W1_nu":          r"$\mathrm{SW}_1(\hat\nu)$",
    "W2_nu":          r"$\mathrm{SW}_2(\hat\nu)$",
    "W1_mu":          r"$\mathrm{SW}_1(\hat\mu)$",
    "W2_mu":          r"$\mathrm{SW}_2(\hat\mu)$",
    "mmd_nu":         r"MMD",
    "energy_dist_nu": r"Energy dist.",
    "sinkhorn_div":   r"Sinkhorn div.",
    "map_err_fwd":    r"Map err (fwd)",
    "map_err_bwd":    r"Map err (bwd)",
    "bw2":            r"$\mathrm{BW}_2$",
    "fid":            "FID",
    "kid":            "KID",
    "weight_ess":     "ESS",
    "weight_mean":    r"$\mathbb{E}[w]$",
    "weight_std":     r"$\sigma(w)$",
}

HPARAM_LABEL: Dict[str, str] = {
    "sigma":          r"$\sigma$",
    "particle_ratio": r"$\rho$ (particle ratio)",
    "iterations":     r"Iterations",
    "cfm_epochs":     r"CFM epochs",
    "dual_epochs":    r"Dual epochs",
    "advance_steps":  r"Advance steps",
    "batch_size":     r"Batch size",
    "scale_factor":   r"Scale factor",
}

HPARAM_LABEL_SHORT: Dict[str, str] = {
    "sigma":          r"$\sigma$",
    "particle_ratio": r"$\rho$",
    "iterations":     "iter.",
    "cfm_epochs":     r"$E_\mathrm{cfm}$",
    "dual_epochs":    r"$E_\mathrm{dual}$",
    "advance_steps":  "adv.",
    "batch_size":     "BS",
    "scale_factor":   "scale",
}

LOWER_IS_BETTER: Set[str] = {
    "W1_nu", "W2_nu", "W1_mu", "W2_mu",
    "mmd_nu", "energy_dist_nu", "sinkhorn_div",
    "map_err_fwd", "map_err_bwd", "bw2", "fid", "kid",
    "weight_std",
}

MUST_CONVERGE: Set[str] = {
    "W1_nu", "W2_nu", "W1_mu", "W2_mu",
    "mmd_nu", "energy_dist_nu", "sinkhorn_div",
    "map_err_fwd", "map_err_bwd", "bw2",
}

RESCALABLE: Set[str] = {
    "W1_mu", "W2_mu", "W1_nu", "W2_nu",
    "mmd_nu", "energy_dist_nu", "sinkhorn_div",
    "map_err_fwd", "map_err_bwd", "bw2",
}

# Ordered priority list for auto-selecting which metrics to show
PRIMARY_METRICS  = ["W2_nu", "map_err_fwd", "sinkhorn_div", "mmd_nu",
                    "energy_dist_nu", "bw2", "W1_nu",
                    "W2_mu", "map_err_bwd", "W1_mu"]
TARGET_MARGINAL  = ["W2_nu", "W1_nu"]
SOURCE_MARGINAL  = ["W2_mu", "W1_mu"]
WEIGHT_METRICS   = ["weight_ess", "weight_mean", "weight_std"]

# Hyperparameters to treat as ablation axes (in priority order)
ABLATION_PARAMS  = ["sigma", "particle_ratio", "iterations",
                    "cfm_epochs", "dual_epochs", "advance_steps", "batch_size"]

# Log-scale axes for these hparams when plotting hparam on x-axis
LOG_SCALE_PARAMS: Set[str] = {"sigma"}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        # Typography
        "text.usetex":          False,
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica Neue", "Arial", "Liberation Sans",
                                 "DejaVu Sans"],
        "mathtext.fontset":     "cm",
        # Backgrounds
        "figure.facecolor":     "white",
        "axes.facecolor":       "#F8F9FA",
        "savefig.facecolor":    "white",
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.06,
        "figure.dpi":           130,
        # Spines — only left + bottom, clean look
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.spines.left":     True,
        "axes.spines.bottom":   True,
        "axes.linewidth":       0.8,
        # Text sizes — slightly larger for readability
        "axes.labelsize":       10,
        "axes.titlesize":       10.5,
        "axes.titleweight":     "semibold",
        "axes.titlepad":        7,
        # Colors
        "axes.prop_cycle":      mpl.cycler("color", PALETTE),
        # Grid — subtle horizontal only
        "axes.grid":            True,
        "axes.grid.axis":       "y",
        "grid.linewidth":       0.35,
        "grid.alpha":           0.5,
        "grid.color":           "#CACACA",
        # Ticks
        "xtick.labelsize":      8.5,
        "ytick.labelsize":      8.5,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.size":     4,
        "ytick.major.size":     4,
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "xtick.minor.visible":  False,
        # Lines / markers
        "lines.linewidth":      2.2,
        "lines.markersize":     5.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle":"round",
        # Legend
        "legend.fontsize":      8.5,
        "legend.framealpha":    0.93,
        "legend.edgecolor":     "#CCCCCC",
        "legend.handlelength":  2.0,
        "legend.labelspacing":  0.45,
        "legend.borderpad":     0.55,
        "legend.handletextpad": 0.55,
        "legend.fancybox":      False,
        # Figure spacing
        "figure.constrained_layout.use": False,
    })


def _get_cmap(name: str):
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except Exception:
        import matplotlib.pyplot as plt
        return plt.cm.get_cmap(name)  # fallback for old matplotlib


def _palette_for(vals: List, cmap_name: str = "plasma") -> Dict[Any, str]:
    """Assign sequential colormap colours to sorted unique values."""
    cmap = _get_cmap(cmap_name)
    n = len(vals)
    return {v: cmap(0.12 + 0.76 * i / max(n - 1, 1))
            for i, v in enumerate(sorted(vals))}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _wandb_api():
    try:
        import wandb
        return wandb.Api()
    except ImportError:
        sys.exit("wandb is required:  pip install wandb")


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _exp_key(rd: Dict) -> str:
    cfg = rd["config"]
    return cfg.get("base_experiment") or cfg.get("experiment") or "unknown"


def fetch_runs(project: str, entity: Optional[str], exp_filter: Optional[str],
               max_runs: Optional[int]) -> List:
    api  = _wandb_api()
    path = f"{entity}/{project}" if entity else project
    filt: Dict = {}
    if exp_filter:
        filt["$or"] = [
            {"config.experiment":      {"$regex": exp_filter}},
            {"config.base_experiment": {"$regex": exp_filter}},
        ]
    print(f"Fetching runs from  {path} ...", flush=True)
    runs = list(api.runs(path, filters=filt or None))
    if max_runs:
        runs = runs[:max_runs]
    print(f"  {len(runs)} run(s) found.", flush=True)
    return runs


def load_run(run, rescale: bool = False) -> Optional[Dict]:
    cfg          = dict(run.config)
    scale_factor = _safe_float(cfg.get("scale_factor", 1.0)) or 1.0

    try:
        rows = run.history(samples=1000, x_axis="_step", pandas=False)
    except Exception:
        rows = []
    if not rows:
        return None

    raw: Dict[str, Dict[int, float]] = defaultdict(dict)
    for row in rows:
        step = int(row.get("_step", row.get("iteration", 0)))
        for k, v in row.items():
            if k.startswith("_"):
                continue
            fv = _safe_float(v)
            if fv is not None:
                raw[k][step] = fv

    if rescale and scale_factor != 1.0:
        for m in RESCALABLE:
            if m in raw:
                raw[m] = {s: v * scale_factor for s, v in raw[m].items()}

    history: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for m, sv in raw.items():
        if not sv:
            continue
        steps = np.array(sorted(sv), dtype=int)
        vals  = np.array([sv[s] for s in steps], dtype=float)
        history[m] = (steps, vals)

    final   = {m: float(v[-1]) for m, (_, v) in history.items() if len(v)}
    best    = {m: float(np.min(v) if m in LOWER_IS_BETTER else np.max(v))
               for m, (_, v) in history.items() if len(v)}
    n_steps = max((int(s[-1]) for (s, _) in history.values()), default=0)

    return dict(
        name=run.name or run.id, id=run.id, state=run.state,
        config=cfg, scale_factor=scale_factor,
        history=history, final=final, best=best, n_steps=n_steps,
    )


def group_by_experiment(runs: List[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = defaultdict(list)
    for rd in runs:
        out[_exp_key(rd)].append(rd)
    return dict(out)


# ---------------------------------------------------------------------------
# Outlier / divergence detection
# ---------------------------------------------------------------------------

def detect_diverged(
    runs: List[Dict],
    iqr_threshold: float = 3.5,
    abs_threshold: float = 1.0,
    rebound_ratio: float = 3.0,
) -> Dict[str, str]:
    reasons: Dict[str, str] = {}

    def flag(rid, reason):
        if rid not in reasons:
            reasons[rid] = reason

    for rd in runs:
        if rd.get("state") in ("crashed", "failed"):
            flag(rd["id"], f"state={rd['state']}")

    if abs_threshold > 0:
        for rd in runs:
            for m in MUST_CONVERGE:
                fv = rd["final"].get(m)
                if fv is not None and fv >= abs_threshold:
                    flag(rd["id"], f"final {m}={fv:.3g} >= {abs_threshold}")
                    break

    by_exp: Dict[str, List[Dict]] = defaultdict(list)
    for rd in runs:
        by_exp[_exp_key(rd)].append(rd)

    for exp_runs in by_exp.values():
        for m in MUST_CONVERGE:
            pairs = [(rd["id"], rd["final"][m])
                     for rd in exp_runs if m in rd["final"]]
            if len(pairs) < 4:
                continue
            vals = np.array([v for _, v in pairs])
            q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
            iqr = q3 - q1
            if iqr < 1e-12:
                continue
            cutoff = q3 + iqr_threshold * iqr
            for rid, v in pairs:
                if v > cutoff:
                    flag(rid, f"{m}={v:.3g} > IQR cutoff {cutoff:.3g}")

    if rebound_ratio > 0:
        for rd in runs:
            for m in MUST_CONVERGE:
                if m not in rd["history"]:
                    continue
                _, vals = rd["history"][m]
                if len(vals) < 6:
                    continue
                best_v = float(np.min(vals))
                last_v = float(vals[-1])
                if best_v < 0.5 and last_v > best_v * rebound_ratio:
                    flag(rd["id"],
                         f"{m} rebounded: {best_v:.3g} → {last_v:.3g} "
                         f"(×{last_v/best_v:.1f})")
                    break

    return reasons


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _avail(runs: List[Dict], pool: List[str]) -> List[str]:
    have = set().union(*[set(rd["history"]) for rd in runs])
    return [m for m in pool if m in have]


def _varying_params(runs: List[Dict]) -> List[str]:
    return [k for k in ABLATION_PARAMS
            if len({rd["config"].get(k) for rd in runs
                    if k in rd["config"]}) > 1]


def aggregate_by_param(
    runs: List[Dict],
    param: str,
    metrics: List[str],
    n_grid: int = 300,
) -> Dict[Any, Dict[str, Any]]:
    """Interpolate each run's history onto a common step grid and
    return per-group mean/std for every requested metric."""
    unique_vals = sorted(
        {rd["config"].get(param) for rd in runs
         if rd["config"].get(param) is not None}
    )
    result: Dict[Any, Dict[str, Any]] = {}

    for val in unique_vals:
        group = [rd for rd in runs if rd["config"].get(param) == val]
        if not group:
            continue
        max_step = max(rd["n_steps"] for rd in group)
        if max_step == 0:
            continue
        grid = np.linspace(0, max_step, n_grid)

        mean_d: Dict[str, np.ndarray] = {}
        std_d:  Dict[str, np.ndarray] = {}
        for m in metrics:
            series = []
            for rd in group:
                if m not in rd["history"]:
                    continue
                steps, vals = rd["history"][m]
                if len(steps) < 2:
                    continue
                series.append(np.interp(grid, steps, vals))
            if not series:
                continue
            arr = np.stack(series, axis=0)
            mean_d[m] = arr.mean(axis=0)
            std_d[m]  = arr.std(axis=0)

        result[val] = {
            "steps": grid,
            "mean":  mean_d,
            "std":   std_d,
            "n":     len(group),
        }

    return result


# ---------------------------------------------------------------------------
# Shared drawing utilities
# ---------------------------------------------------------------------------

def _save(fig, out: Path, name: str, tex: List[str]) -> None:
    import matplotlib.pyplot as plt
    png = out / f"{name}.png"
    pdf = out / f"{name}.pdf"
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    plt.close(fig)
    label   = name.replace("_", "-").replace("/", "-")
    caption = (name.replace("__", " — ").replace("_", " ").capitalize())
    tex.append(
        f"\\begin{{figure}}[htbp]\n  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{figures/{name}.pdf}}\n"
        f"  \\caption{{{caption}}}\n  \\label{{fig:{label}}}\n"
        f"\\end{{figure}}\n"
    )
    print(f"  {png.name}")


def _val_label(v: Any) -> str:
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e4:
            return str(int(v))
        return f"{v:.3g}"
    return str(v)


def _legend_outside(fig, axes_flat, ncol: int = 5) -> None:
    """Deduplicate and place a single legend below the figure."""
    seen: Dict[str, Any] = {}
    for ax in axes_flat:
        if not ax.get_visible():
            continue
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and not l.startswith("_") and l not in seen:
                seen[l] = h
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()),
                   loc="upper center",
                   bbox_to_anchor=(0.5, 0.0),
                   ncol=min(ncol, len(seen)),
                   fontsize=8.5,
                   framealpha=0.93,
                   edgecolor="#D0D0D0",
                   columnspacing=1.0)


def _finalize_ax(ax, ylabel: str, xlabel: str = "Iteration",
                 logy: bool = False) -> None:
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(left=0)
    if logy:
        try:
            ax.set_yscale("log")
        except Exception:
            pass
    else:
        ylo, yhi = ax.get_ylim()
        if ylo < 0:
            ax.set_ylim(bottom=0)


# ---------------------------------------------------------------------------
# C — Convergence curves
# ---------------------------------------------------------------------------

def plot_convergence(
    runs: List[Dict],
    exp: str,
    out: Path,
    tex: List[str],
    param: str,
    metrics: Optional[List[str]] = None,
    cmap: str = "plasma",
    logy: bool = False,
) -> None:
    """Convergence over iterations, one line per hparam value, ±std band."""
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = _avail(runs, PRIMARY_METRICS)[:5]
    metrics = [m for m in metrics if any(m in rd["history"] for rd in runs)]
    if not metrics:
        return

    agg = aggregate_by_param(runs, param, metrics)
    if not agg:
        return

    unique_vals = sorted(agg.keys())
    colors      = _palette_for(unique_vals, cmap)
    plabel      = HPARAM_LABEL_SHORT.get(param, param)

    n  = len(metrics)
    nc = min(n, 3)
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc,
                             figsize=(nc * COL_W + 0.1, nr * 2.5 + 0.3),
                             squeeze=False)
    ax_flat = [axes[r][c] for r in range(nr) for c in range(nc)]

    for ax, m in zip(ax_flat, metrics):
        for val in unique_vals:
            entry = agg[val]
            if m not in entry["mean"]:
                continue
            steps = entry["steps"]
            mean  = entry["mean"][m]
            std   = entry["std"][m]
            n_r   = entry["n"]
            col   = colors[val]
            lbl   = f"{plabel}={_val_label(val)}"
            if n_r > 1:
                lbl += f"  (n={n_r})"
            ax.plot(steps, mean, color=col, lw=2.0, label=lbl, zorder=3)
            ax.fill_between(steps, np.maximum(mean - std, 0), mean + std,
                            color=col, alpha=0.14, zorder=2, linewidth=0)
        _finalize_ax(ax, MLABEL_SHORT.get(m, m), logy=logy)
        ax.set_title(MLABEL.get(m, m), fontsize=9.5)

    for ax in ax_flat[len(metrics):]:
        ax.set_visible(False)

    _legend_outside(fig, ax_flat)
    safe  = exp.replace("/", "_")
    title = (f"Convergence — ablation {HPARAM_LABEL.get(param, param)}"
             f"\n{exp.replace('_', ' ')}")
    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, out, f"{safe}__conv_{param}", tex)


# ---------------------------------------------------------------------------
# S — Final-metric scatter (hparam on x, metric on y)
# ---------------------------------------------------------------------------

def plot_final_scatter(
    runs: List[Dict],
    exp: str,
    out: Path,
    tex: List[str],
    param: str,
    metrics: Optional[List[str]] = None,
    cmap: str = "plasma",
) -> None:
    """Dot plot: hparam value on x, final metric on y.
    Each dot = one run; overlaid mean ± std error bar; optional trend line."""
    import matplotlib.pyplot as plt
    from scipy.stats import linregress  # optional

    if metrics is None:
        metrics = _avail(runs, PRIMARY_METRICS)[:4]
    metrics = [m for m in metrics if any(m in rd["final"] for rd in runs)]
    if not metrics:
        return

    unique_vals = sorted(
        {rd["config"].get(param) for rd in runs
         if rd["config"].get(param) is not None}
    )
    if len(unique_vals) < 2:
        return

    colors = _palette_for(unique_vals, cmap)
    log_x  = param in LOG_SCALE_PARAMS and all(v > 0 for v in unique_vals)

    n  = len(metrics)
    nc = min(n, 2)
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc,
                             figsize=(nc * 3.1, nr * 2.7),
                             squeeze=False)
    ax_flat = [axes[r][c] for r in range(nr) for c in range(nc)]

    rng = np.random.default_rng(0)

    for ax, m in zip(ax_flat, metrics):
        xs_mean, ys_mean, ys_std = [], [], []

        for xi, val in enumerate(unique_vals):
            group = [rd for rd in runs
                     if rd["config"].get(param) == val and m in rd["final"]]
            if not group:
                continue
            fvals = np.array([rd["final"][m] for rd in group])
            col   = colors[val]
            jitter = rng.uniform(-0.06, 0.06, len(fvals))
            x_plot = float(val)
            ax.scatter(x_plot + jitter, fvals,
                       color=col, s=30, alpha=0.65,
                       edgecolors="white", linewidths=0.5, zorder=4)
            mean_v = float(fvals.mean())
            std_v  = float(fvals.std()) if len(fvals) > 1 else 0.0
            ax.errorbar(x_plot, mean_v, yerr=std_v,
                        fmt="D", color=col, markersize=7,
                        capsize=4, capthick=1.5, elinewidth=1.5, zorder=5)
            xs_mean.append(x_plot)
            ys_mean.append(mean_v)
            ys_std.append(std_v)

        # Trend line (log or linear x)
        if len(xs_mean) >= 3:
            try:
                x_arr = np.log10(xs_mean) if log_x else np.array(xs_mean)
                slope, intercept, r, *_ = linregress(x_arr, ys_mean)
                x_line = np.linspace(min(x_arr), max(x_arr), 100)
                y_line = slope * x_line + intercept
                x_plot_line = 10 ** x_line if log_x else x_line
                ax.plot(x_plot_line, y_line, "--", color="#777", lw=1.2,
                        alpha=0.7, label=f"trend  $r={r:.2f}$", zorder=2)
                ax.legend(fontsize=7.5, framealpha=0.9)
            except Exception:
                pass

        ax.set_xlabel(HPARAM_LABEL.get(param, param), fontsize=9)
        ax.set_ylabel(MLABEL_SHORT.get(m, m), fontsize=9)
        ax.set_title(MLABEL.get(m, m), fontsize=9.5)
        ax.set_xticks(unique_vals)
        ax.set_xticklabels([_val_label(v) for v in unique_vals], fontsize=8)
        if log_x:
            ax.set_xscale("log")
        ylo, yhi = ax.get_ylim()
        if ylo < 0:
            ax.set_ylim(bottom=0)

    for ax in ax_flat[len(metrics):]:
        ax.set_visible(False)

    safe  = exp.replace("/", "_")
    title = (f"Final {HPARAM_LABEL.get(param, param)} sensitivity"
             f"\n{exp.replace('_', ' ')}")
    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    _save(fig, out, f"{safe}__scatter_{param}", tex)


# ---------------------------------------------------------------------------
# V — Violin / box plots
# ---------------------------------------------------------------------------

def plot_violin(
    runs: List[Dict],
    exp: str,
    out: Path,
    tex: List[str],
    param: str,
    metrics: Optional[List[str]] = None,
    cmap: str = "plasma",
) -> None:
    """Violin plots of the final metric distribution grouped by hparam value."""
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = _avail(runs, PRIMARY_METRICS)[:4]
    metrics = [m for m in metrics if any(m in rd["final"] for rd in runs)]
    if not metrics:
        return

    unique_vals = sorted(
        {rd["config"].get(param) for rd in runs
         if rd["config"].get(param) is not None}
    )
    if len(unique_vals) < 2:
        return

    colors = _palette_for(unique_vals, cmap)
    n  = len(metrics)
    nc = min(n, 2)
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc,
                             figsize=(nc * 3.2, nr * 2.7),
                             squeeze=False)
    ax_flat = [axes[r][c] for r in range(nr) for c in range(nc)]

    rng = np.random.default_rng(1)

    for ax, m in zip(ax_flat, metrics):
        data = []
        positions = []
        tick_labels = []
        for xi, val in enumerate(unique_vals):
            group = [rd["final"][m] for rd in runs
                     if rd["config"].get(param) == val and m in rd["final"]]
            if not group:
                continue
            data.append(group)
            positions.append(xi)
            tick_labels.append(_val_label(val))

        if not data:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(data, positions=positions,
                              showmedians=True, showextrema=False,
                              widths=0.6)
        for i, (pc, val) in enumerate(zip(parts["bodies"], unique_vals)):
            col = colors[val]
            pc.set_facecolor(col)
            pc.set_edgecolor(col)
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("#111")
        parts["cmedians"].set_linewidth(1.5)

        # Jitter strip
        for i, (vals_group, xi, val) in enumerate(
                zip(data, positions, unique_vals)):
            jitter = rng.uniform(-0.1, 0.1, len(vals_group))
            ax.scatter(np.full(len(vals_group), xi) + jitter,
                       vals_group, s=18, color=colors[val],
                       alpha=0.75, edgecolors="white", linewidths=0.4,
                       zorder=5)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_xlabel(HPARAM_LABEL.get(param, param), fontsize=9)
        ax.set_ylabel(MLABEL_SHORT.get(m, m), fontsize=9)
        ax.set_title(MLABEL.get(m, m), fontsize=9.5)
        ylo, yhi = ax.get_ylim()
        if ylo < 0:
            ax.set_ylim(bottom=0)

    for ax in ax_flat[len(metrics):]:
        ax.set_visible(False)

    safe  = exp.replace("/", "_")
    title = (f"Distribution of final metrics by {HPARAM_LABEL.get(param, param)}"
             f"\n{exp.replace('_', ' ')}")
    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    _save(fig, out, f"{safe}__violin_{param}", tex)


# ---------------------------------------------------------------------------
# H — Annotated heatmap (two hparams × one metric)
# ---------------------------------------------------------------------------

def plot_heatmap(
    runs: List[Dict],
    exp: str,
    out: Path,
    tex: List[str],
    x_key: str = "sigma",
    y_key: str = "particle_ratio",
    metrics: Optional[List[str]] = None,
) -> None:
    """Clean annotated heatmap for each metric (x=sigma, y=particle_ratio)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if metrics is None:
        metrics = _avail(runs, PRIMARY_METRICS)[:3]
    if not metrics:
        return

    x_vals = sorted({rd["config"].get(x_key) for rd in runs
                     if rd["config"].get(x_key) is not None})
    y_vals = sorted({rd["config"].get(y_key) for rd in runs
                     if rd["config"].get(y_key) is not None})
    if len(x_vals) < 2 or len(y_vals) < 2:
        return

    # Good → green, Bad → red
    cmap_good = LinearSegmentedColormap.from_list(
        "grn_red", ["#1a7431", "#a8ddb5", "#fffde7", "#f97316", "#7f1d1d"], N=256)

    n  = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.2, 3.0), squeeze=False)
    axes = axes[0]

    for ax, m in zip(axes, metrics):
        grid   = np.full((len(y_vals), len(x_vals)), np.nan)
        grid_n = np.zeros_like(grid, dtype=int)
        grid_s = np.full_like(grid, np.nan)   # std

        for rd in runs:
            xv = rd["config"].get(x_key)
            yv = rd["config"].get(y_key)
            fv = rd["final"].get(m)
            if None in (xv, yv, fv):
                continue
            if xv not in x_vals or yv not in y_vals:
                continue
            xi, yi = x_vals.index(xv), y_vals.index(yv)
            if np.isnan(grid[yi, xi]):
                grid[yi, xi] = fv
                grid_n[yi, xi] = 1
            else:
                # Online mean update
                n_prev = grid_n[yi, xi]
                grid[yi, xi] = (grid[yi, xi] * n_prev + fv) / (n_prev + 1)
                grid_n[yi, xi] += 1

        if np.all(np.isnan(grid)):
            ax.set_visible(False)
            continue

        vmin = np.nanmin(grid)
        vmax = np.nanmax(grid)
        rng  = max(vmax - vmin, 1e-12)

        # Flip so green=low for lower-is-better
        cmap = cmap_good
        im   = ax.imshow(grid, aspect="auto", origin="lower",
                         cmap=cmap, interpolation="nearest",
                         vmin=vmin, vmax=vmax)
        ax.grid(False)

        # Axis labels
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([_val_label(v) for v in x_vals], fontsize=8.5)
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([_val_label(v) for v in y_vals], fontsize=8.5)
        ax.set_xlabel(HPARAM_LABEL.get(x_key, x_key), fontsize=9.5)
        ax.set_ylabel(HPARAM_LABEL.get(y_key, y_key), fontsize=9.5)
        ax.set_title(MLABEL.get(m, m), fontsize=9.5)

        # Cell annotations
        for yi in range(len(y_vals)):
            for xi in range(len(x_vals)):
                v = grid[yi, xi]
                if np.isnan(v):
                    ax.text(xi, yi, "—", ha="center", va="center",
                            fontsize=8, color="#888")
                    continue
                brightness = (v - vmin) / rng
                txt_col = "white" if brightness < 0.38 or brightness > 0.78 else "#111"
                n_v = grid_n[yi, xi]
                txt = f"{v:.3g}"
                if n_v > 1:
                    txt += f"\n±(n={n_v})"
                # Bold the best cell
                is_best = (v == vmin) if m in LOWER_IS_BETTER else (v == vmax)
                ax.text(xi, yi, txt, ha="center", va="center",
                        fontsize=7.5, color=txt_col,
                        fontweight="bold" if is_best else "normal")

        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(MLABEL_SHORT.get(m, m), fontsize=8)

    safe = exp.replace("/", "_")
    x_lbl = HPARAM_LABEL_SHORT.get(x_key, x_key)
    y_lbl = HPARAM_LABEL_SHORT.get(y_key, y_key)
    fig.suptitle(
        f"Final metric — {x_lbl} × {y_lbl}"
        f"  |  {exp.replace('_', ' ')}",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    _save(fig, out, f"{safe}__heatmap_{x_key}_x_{y_key}", tex)


# ---------------------------------------------------------------------------
# P — Parallel coordinates (curved Catmull-Rom splines, all metrics)
# ---------------------------------------------------------------------------

def _catmull_rom(p0, p1, p2, p3, n: int = 60):
    """Return n points along a Catmull-Rom spline between p1 and p2."""
    t  = np.linspace(0, 1, n)
    t2 = t * t
    t3 = t2 * t
    q  = (
        0.5 * (
            (2 * p1)
            + (-p0 + p2) * t[:, None]
            + (2*p0 - 5*p1 + 4*p2 - p3) * t2[:, None]
            + (-p0 + 3*p1 - 3*p2 + p3) * t3[:, None]
        )
    )
    return q


def plot_parallel_coords(
    runs: List[Dict],
    exp: str,
    out: Path,
    tex: List[str],
    target_metric: str = "W2_nu",
    params: Optional[List[str]] = None,
) -> None:
    """Parallel-coordinates with curved (Catmull-Rom) lines.
    Axes = hyperparams + ALL available final metrics; colour = target_metric."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    # Build axes list: varying hparams first, then available metrics
    if params is None:
        params = _varying_params(runs)
    params = [p for p in params if any(p in rd["config"] for rd in runs)]

    avail_metrics = _avail(runs, PRIMARY_METRICS)
    if not avail_metrics:
        return

    # Ensure target_metric exists, fall back otherwise
    target = target_metric
    if not any(target in rd["final"] for rd in runs):
        target = avail_metrics[0]

    axes_keys = params + avail_metrics   # hparams then metrics

    # Collect records — keep runs that have at least the target metric
    records = []
    for rd in runs:
        row: Dict[str, Any] = {}
        for p in params:
            v = rd["config"].get(p)
            if v is None:
                break
            row[p] = float(v)
        else:
            for m in avail_metrics:
                row[m] = rd["final"].get(m)   # may be None → drawn as NaN
            records.append(row)

    if len(records) < 2:
        return

    # Per-axis normalisation (NaN-safe)
    def _axis_stats(key):
        vals = np.array([r.get(key) for r in records], dtype=float)
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        return lo, hi, max(hi - lo, 1e-12), vals

    ax_lo, ax_hi, ax_rng, ax_raw = {}, {}, {}, {}
    for k in axes_keys:
        ax_lo[k], ax_hi[k], ax_rng[k], ax_raw[k] = _axis_stats(k)

    def _norm(key, val):
        if val is None or np.isnan(float(val)):
            return np.nan
        return (float(val) - ax_lo[key]) / ax_rng[key]

    target_vals = ax_raw[target]
    finite_mask = np.isfinite(target_vals)
    if not finite_mask.any():
        return

    cmap_name = "viridis_r" if target in LOWER_IS_BETTER else "viridis"
    cmap      = _get_cmap(cmap_name)
    t_min     = np.nanmin(target_vals)
    t_max     = np.nanmax(target_vals)
    cnorm     = mcolors.Normalize(vmin=t_min, vmax=t_max)

    n_axes = len(axes_keys)
    fig_w  = max(COL_2W + 1, n_axes * 1.05)
    fig, host = plt.subplots(figsize=(fig_w, 4.0))
    host.set_xlim(-0.5, n_axes - 0.5)
    host.set_ylim(-0.08, 1.08)
    host.set_facecolor("#F7F7F7")
    host.axis("off")

    # ── Draw vertical axis lines ────────────────────────────────────────────
    sep_x = n_axes - 1 - len(avail_metrics)   # position of hparam/metric separator

    for i, key in enumerate(axes_keys):
        lo, hi = ax_lo[key], ax_hi[key]
        x = float(i)

        # Shaded background for metric axes
        if i > sep_x:
            host.axvspan(x - 0.5, x + 0.5, color="#E8EEF4", alpha=0.45, zorder=0)

        host.axvline(x, color="#AAAAAA", lw=1.2, zorder=1)

        # Tick marks + labels on the axis
        n_ticks = 5
        for ti in range(n_ticks):
            y_norm = ti / (n_ticks - 1)
            raw_v  = lo + y_norm * ax_rng[key]
            host.plot([x - 0.04, x + 0.04], [y_norm, y_norm],
                      color="#888", lw=0.8, zorder=2)
            host.text(x - 0.07, y_norm, _val_label(raw_v),
                      ha="right", va="center", fontsize=6.5, color="#555")

        # Axis label at top
        if key in HPARAM_LABEL_SHORT:
            lbl = HPARAM_LABEL_SHORT[key]
            style = "normal"
        elif key in MLABEL_SHORT:
            lbl = MLABEL_SHORT[key]
            style = "italic"
        else:
            lbl = key
            style = "normal"
        host.text(x, 1.055, lbl,
                  ha="center", va="bottom", fontsize=8,
                  fontstyle=style, fontweight="semibold", color="#222")

    # Divider label
    if params and avail_metrics:
        mid = (sep_x + sep_x + 1) / 2
        host.text(mid + 0.5, -0.065, "← params   metrics →",
                  ha="center", va="top", fontsize=7, color="#888", style="italic")

    # ── Draw curved lines ────────────────────────────────────────────────────
    x_pos = np.arange(n_axes, dtype=float)

    # Sort by target metric so good runs are drawn on top
    order = np.argsort(target_vals)[::-1] if target in LOWER_IS_BETTER \
        else np.argsort(target_vals)

    for ri in order:
        rec = records[ri]
        y_norm = np.array([_norm(k, rec.get(k)) for k in axes_keys])
        if np.isnan(target_vals[ri]):
            continue

        col   = cmap(cnorm(target_vals[ri]))
        alpha = 0.55 + 0.35 * (1 - cnorm(target_vals[ri]))  # better runs more opaque

        # Build curved path via Catmull-Rom between consecutive axes
        xs_all, ys_all = [], []
        for i in range(n_axes - 1):
            if np.isnan(y_norm[i]) or np.isnan(y_norm[i + 1]):
                continue
            # Ghost points for spline continuity
            p0 = np.array([x_pos[max(i-1, 0)],   y_norm[max(i-1, 0)]])
            p1 = np.array([x_pos[i],               y_norm[i]])
            p2 = np.array([x_pos[i+1],             y_norm[i+1]])
            p3 = np.array([x_pos[min(i+2, n_axes-1)], y_norm[min(i+2, n_axes-1)]])
            seg = _catmull_rom(p0, p1, p2, p3, n=40)
            xs_all.extend(seg[:, 0])
            ys_all.extend(seg[:, 1])

        if xs_all:
            host.plot(xs_all, ys_all, color=col, alpha=alpha,
                      lw=1.0, solid_capstyle="round", zorder=3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=host, shrink=0.72, pad=0.02, aspect=22)
    cb.set_label(MLABEL_SHORT.get(target, target), fontsize=9)
    cb.ax.tick_params(labelsize=7.5)

    safe = exp.replace("/", "_")
    fig.suptitle(
        f"Parallel coordinates  |  {exp.replace('_', ' ')}",
        fontsize=11, y=0.98, fontweight="semibold",
    )
    plt.tight_layout(rect=[0, 0.0, 0.92, 0.96])
    _save(fig, out, f"{safe}__parallel_coords", tex)


# ---------------------------------------------------------------------------
# M — Map error at fixed iteration counts
# ---------------------------------------------------------------------------

def plot_map_err_at_iterations(
    runs: List[Dict],
    exp: str,
    out: Path,
    tex: List[str],
    iteration_counts: List[int],
    param: str,
    cmap: str = "plasma",
) -> None:
    """For each iteration count in iteration_counts: a 2-subplot figure
    (map_err_fwd left, map_err_bwd right) showing values across param values."""
    import matplotlib.pyplot as plt

    has_fwd = any("map_err_fwd" in rd["history"] for rd in runs)
    has_bwd = any("map_err_bwd" in rd["history"] for rd in runs)
    if not has_fwd and not has_bwd:
        return

    unique_vals = sorted(
        {rd["config"].get(param) for rd in runs
         if rd["config"].get(param) is not None}
    )
    if len(unique_vals) < 1:
        return

    colors = _palette_for(unique_vals, cmap)
    rng    = np.random.default_rng(7)

    for it_count in iteration_counts:
        fig, axes = plt.subplots(1, 2, figsize=(COL_2W, 2.9),
                                 sharey=False, squeeze=True)

        for ax, (metric, m_label) in zip(
            axes,
            [("map_err_fwd", "Forward  " + r"$\|\hat T - T^*\|$"),
             ("map_err_bwd", "Backward  " + r"$\|\hat T^{-1} - {T^*}^{-1}\|$")],
        ):
            ax.set_facecolor("#F7F7F7")
            vals_per_group: Dict[Any, List[float]] = {}

            for rd in runs:
                val = rd["config"].get(param)
                if val is None or metric not in rd["history"]:
                    continue
                steps, series = rd["history"][metric]
                # Find closest logged step to it_count
                idx = int(np.argmin(np.abs(steps - it_count)))
                if abs(int(steps[idx]) - it_count) > max(2, it_count * 0.15):
                    continue   # too far from the target step
                vals_per_group.setdefault(val, []).append(float(series[idx]))

            if not vals_per_group:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="#999")
                ax.set_visible(True)
            else:
                positions = list(range(len(unique_vals)))
                for xi, gval in enumerate(unique_vals):
                    fvals = np.array(vals_per_group.get(gval, []))
                    if len(fvals) == 0:
                        continue
                    col  = colors[gval]
                    mean = float(fvals.mean())
                    std  = float(fvals.std()) if len(fvals) > 1 else 0.0

                    # Bar
                    ax.bar(xi, mean, color=col, alpha=0.75, width=0.6,
                           zorder=2, linewidth=0)
                    # Error cap
                    ax.errorbar(xi, mean, yerr=std, fmt="none",
                                color=col, capsize=5, capthick=1.8,
                                elinewidth=1.5, zorder=4)
                    # Jitter strip
                    if len(fvals) > 1:
                        jitter = rng.uniform(-0.18, 0.18, len(fvals))
                        ax.scatter(xi + jitter, fvals, color=col,
                                   s=22, edgecolors="white",
                                   linewidths=0.5, zorder=5, alpha=0.85)
                    # Value annotation
                    ax.text(xi, mean + std + ax.get_ylim()[1] * 0.02 + 1e-8,
                            f"{mean:.3g}", ha="center", va="bottom",
                            fontsize=7, color="#333")

                ax.set_xticks(range(len(unique_vals)))
                ax.set_xticklabels([_val_label(v) for v in unique_vals],
                                   fontsize=8)
                ax.set_xlabel(HPARAM_LABEL.get(param, param), fontsize=9)
                ax.set_ylabel("Error", fontsize=9)

            ax.set_title(m_label, fontsize=9.5)
            ax.grid(True, axis="y", linewidth=0.3, alpha=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ylo, _ = ax.get_ylim()
            if ylo < 0:
                ax.set_ylim(bottom=0)

        safe = exp.replace("/", "_")
        fig.suptitle(
            f"Map error at iteration {it_count}"
            f"  |  {exp.replace('_', ' ')}",
            fontsize=10.5, y=1.01, fontweight="semibold",
        )
        plt.tight_layout()
        _save(fig, out, f"{safe}__map_err_iter{it_count:04d}_{param}", tex)


# ---------------------------------------------------------------------------
# X — Cross-experiment comparison
# ---------------------------------------------------------------------------

def plot_cross_experiment(
    groups: Dict[str, List[Dict]],
    out: Path,
    tex: List[str],
) -> None:
    """Best run per experiment: convergence curves + final bar chart."""
    import matplotlib.pyplot as plt

    metrics = [m for m in PRIMARY_METRICS
               if any(any(m in rd["history"] for rd in rs)
                      for rs in groups.values())][:4]
    if not metrics:
        return

    exp_list = sorted(groups.keys())
    colors   = {e: PALETTE[i % len(PALETTE)] for i, e in enumerate(exp_list)}

    # --- Panel A: convergence (best run per experiment) ---
    nc  = len(metrics)
    fig, axes = plt.subplots(1, nc, figsize=(nc * 2.7, 2.7), squeeze=False)
    axes = axes[0]

    for ax, m in zip(axes, metrics):
        for exp in exp_list:
            cands = [rd for rd in groups[exp] if m in rd["history"]]
            if not cands:
                continue
            best = min(cands, key=lambda r: r["final"].get(m, np.inf))
            steps, vals = best["history"][m]
            ax.plot(steps, vals, color=colors[exp], lw=1.8,
                    alpha=0.88, label=exp.replace("_", " "), zorder=3)
        _finalize_ax(ax, MLABEL_SHORT.get(m, m))
        ax.set_title(MLABEL.get(m, m), fontsize=9.5)

    _legend_outside(fig, list(axes))
    fig.suptitle("Best run per experiment — convergence comparison",
                 fontsize=10, y=1.01)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    _save(fig, out, "cross__convergence", tex)

    # --- Panel B: final-metric bar chart ---
    fig2, axes2 = plt.subplots(1, nc, figsize=(nc * 2.7, 2.8), squeeze=False)
    axes2 = axes2[0]
    rng   = np.random.default_rng(2)

    for ax, m in zip(axes2, metrics):
        for xi, exp in enumerate(exp_list):
            vals = [rd["final"][m] for rd in groups[exp] if m in rd["final"]]
            if not vals:
                continue
            mean = np.mean(vals)
            std  = np.std(vals) if len(vals) > 1 else 0.0
            col  = colors[exp]
            ax.bar(xi, mean, yerr=std, color=col, alpha=0.80, capsize=4,
                   error_kw={"lw": 1.2, "capthick": 1.2}, width=0.6, zorder=3)
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(xi + jitter, vals, color=col, s=22,
                       edgecolors="white", linewidths=0.5, zorder=4, alpha=0.85)
            ax.text(xi, mean + std * 1.25 + 1e-8, f"{mean:.3g}",
                    ha="center", va="bottom", fontsize=7, color="#333")

        short_labels = [e.replace("2d_", "").replace("3d_", "").replace("_", "\n")
                        for e in exp_list]
        ax.set_xticks(range(len(exp_list)))
        ax.set_xticklabels(short_labels, fontsize=7.5)
        ax.set_ylabel(MLABEL_SHORT.get(m, m), fontsize=9)
        ax.set_title(MLABEL.get(m, m), fontsize=9.5)

    fig2.suptitle("Final metric across experiments", fontsize=10, y=1.01)
    plt.tight_layout()
    _save(fig2, out, "cross__final_bar", tex)


# ---------------------------------------------------------------------------
# T — LaTeX table (booktabs, mean ± std, bold best)
# ---------------------------------------------------------------------------

def write_latex_table(
    groups: Dict[str, List[Dict]],
    out: Path,
    tex: List[str],
    param: Optional[str] = None,
    metrics: Optional[List[str]] = None,
) -> None:
    """Write a booktabs LaTeX table: rows = (experiment [× hparam value]),
    columns = metrics (mean ± std), best per column bolded."""

    if metrics is None:
        all_runs = [rd for rs in groups.values() for rd in rs]
        metrics  = _avail(all_runs, PRIMARY_METRICS)[:6]
    if not metrics:
        return

    # Build table rows
    Row = Dict[str, Any]
    rows: List[Row] = []

    for exp, runs in sorted(groups.items()):
        if param and len({rd["config"].get(param) for rd in runs
                          if rd["config"].get(param) is not None}) > 1:
            unique_vals = sorted(
                {rd["config"].get(param) for rd in runs
                 if rd["config"].get(param) is not None}
            )
            for val in unique_vals:
                grp = [rd for rd in runs if rd["config"].get(param) == val]
                row: Row = {
                    "Experiment": exp.replace("_", " "),
                    HPARAM_LABEL_SHORT.get(param, param): _val_label(val),
                }
                for m in metrics:
                    fvals = [rd["final"][m] for rd in grp if m in rd["final"]]
                    if fvals:
                        row[m] = (float(np.mean(fvals)),
                                  float(np.std(fvals)) if len(fvals) > 1 else 0.0,
                                  len(fvals))
                    else:
                        row[m] = None
                rows.append(row)
        else:
            row = {"Experiment": exp.replace("_", " ")}
            for m in metrics:
                fvals = [rd["final"][m] for rd in runs if m in rd["final"]]
                if fvals:
                    row[m] = (float(np.mean(fvals)),
                              float(np.std(fvals)) if len(fvals) > 1 else 0.0,
                              len(fvals))
                else:
                    row[m] = None
            rows.append(row)

    if not rows:
        return

    # Find best value per metric column (for bolding)
    best_val: Dict[str, float] = {}
    for m in metrics:
        col_vals = [r[m][0] for r in rows if r.get(m) is not None]
        if not col_vals:
            continue
        best_val[m] = min(col_vals) if m in LOWER_IS_BETTER else max(col_vals)

    # Header columns
    extra_cols = [k for k in rows[0] if k not in metrics and k != "Experiment"]
    id_cols    = ["Experiment"] + extra_cols
    all_cols   = id_cols + metrics

    # Column spec: l for id cols, r for metrics
    col_spec = "l" * len(id_cols) + ("r" * len(metrics))

    lines = [
        "% Auto-generated — DualWCFM ablation table",
        "% Requires: \\usepackage{booktabs,multirow,amsmath}",
        "",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Ablation study results (mean $\\pm$ std). "
        "\\textbf{Bold} = best per column.}",
        "  \\label{tab:ablation}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
    ]

    # Header row
    header_id  = " & ".join(c for c in id_cols)
    header_met = " & ".join(
        f"\\multicolumn{{1}}{{r}}{{{MLABEL_SHORT.get(m, m)}}}"
        for m in metrics
    )
    lines.append(f"    {header_id} & {header_met} \\\\")
    lines.append("    \\midrule")

    for row in rows:
        id_part = " & ".join(str(row.get(c, "")) for c in id_cols)
        metric_parts = []
        for m in metrics:
            v = row.get(m)
            if v is None:
                metric_parts.append("—")
            else:
                mean, std, n_r = v
                cell = f"{mean:.4f}"
                if n_r > 1:
                    cell += f" $\\pm$ {std:.4f}"
                if m in best_val and abs(mean - best_val[m]) < 1e-9:
                    cell = f"\\textbf{{{cell}}}"
                metric_parts.append(cell)
        lines.append(f"    {id_part} & {' & '.join(metric_parts)} \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]

    table_str = "\n".join(lines) + "\n"
    tbl_path  = out / "ablation_table.tex"
    tbl_path.write_text(table_str, encoding="utf-8")
    print(f"  LaTeX table -> {tbl_path}")
    tex.append(f"% Table: \\input{{{tbl_path.name}}}\n")


# ---------------------------------------------------------------------------
# LaTeX wrapper document
# ---------------------------------------------------------------------------

def write_latex_wrapper(tex: List[str], out_dir: Path, project: str) -> None:
    p = out_dir / "paper_figures.tex"
    header = (
        "% Auto-generated by plot_wandb_results.py\n"
        f"% Project: {project}\n"
        "% Include with: \\input{paper_figures.tex}\n"
        "%\n"
        "% Preamble requirements:\n"
        "%   \\usepackage{graphicx,subcaption,booktabs,multirow,amsmath}\n"
        "%   \\graphicspath{{./plots/}}\n\n"
    )
    p.write_text(header + "\n".join(tex), encoding="utf-8")
    print(f"\nLaTeX wrapper → {p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python plot_wandb_results.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    fetch = p.add_argument_group("W&B fetch")
    fetch.add_argument("--project",    required=True,
                       help="W&B project name.")
    fetch.add_argument("--entity",     default=None,
                       help="W&B entity (username or team).")
    fetch.add_argument("--experiment", default=None,
                       help="Filter runs by experiment name substring.")
    fetch.add_argument("--max-runs",   type=int, default=None,
                       help="Cap the number of runs fetched.")
    fetch.add_argument("--min-steps",  type=int, default=3,
                       help="Discard runs with fewer than N logged steps.")

    out = p.add_argument_group("output")
    out.add_argument("--output-dir",  default="plots",
                     help="Root directory for figures (default: plots/).")
    out.add_argument("--rescale",     action="store_true",
                     dest="rescale_metrics",
                     help="Multiply metric values by scale_factor.")

    filt = p.add_argument_group("outlier filtering")
    filt.add_argument("--outlier-abs",     type=float, default=1.0,
                      help="Flag runs with final MUST_CONVERGE metric >= this (0 = off).")
    filt.add_argument("--outlier-iqr",     type=float, default=3.5,
                      help="Tukey IQR multiplier for per-experiment filtering.")
    filt.add_argument("--outlier-rebound", type=float, default=3.0,
                      help="Flag runs where final > best × ratio (0 = off).")
    filt.add_argument("--include-diverged", action="store_true",
                      help="Keep diverged/crashed runs in all plots.")
    filt.add_argument("--exclude-param", action="append", default=[],
                      metavar="PARAM=VAL",
                      help="Drop runs where config[PARAM] == VAL. "
                           "Repeat to exclude multiple values. "
                           "E.g. --exclude-param sigma=1.0 --exclude-param sigma2=1.0")

    plots = p.add_argument_group("plot selection")
    plots.add_argument("--no-convergence", action="store_true",
                       help="Skip convergence-curve plots.")
    plots.add_argument("--no-scatter",    action="store_true",
                       help="Skip final-metric scatter plots.")
    plots.add_argument("--no-violin",     action="store_true",
                       help="Skip violin plots.")
    plots.add_argument("--no-heatmap",    action="store_true",
                       help="Skip heatmap plots.")
    plots.add_argument("--no-parallel",   action="store_true",
                       help="Skip parallel-coordinates plot.")
    plots.add_argument("--no-cross",      action="store_true",
                       help="Skip cross-experiment comparison.")
    plots.add_argument("--no-table",      action="store_true",
                       help="Skip LaTeX table generation.")
    plots.add_argument("--logy",          action="store_true",
                       help="Use log y-scale for convergence plots.")

    plots.add_argument("--map-err-iterations", default="10,50,100,250",
                       dest="map_err_iterations",
                       help="Comma-separated iteration counts for map-error snapshots "
                            "(default: 10,50,100,250).")
    plots.add_argument("--target-metric", default="W2_nu",
                       help="Metric to colour parallel-coordinates lines by "
                            "(default: W2_nu).")
    plots.add_argument("--heatmap-x",    default="sigma",
                       help="X-axis parameter for heatmap (default: sigma).")
    plots.add_argument("--heatmap-y",    default="particle_ratio",
                       help="Y-axis parameter for heatmap (default: particle_ratio).")
    plots.add_argument("--metrics",      default=None,
                       help="Comma-separated list of metrics to plot "
                            "(default: auto-select from available).")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)
    args = _parser().parse_args()

    import matplotlib
    matplotlib.use("Agg")
    _setup_style()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out.resolve()}\n")

    # ── Fetch ────────────────────────────────────────────────────────────────
    raw = fetch_runs(args.project, args.entity, args.experiment, args.max_runs)
    if not raw:
        sys.exit("No runs found.  Check --project / --entity / --experiment.")

    # ── Load histories ───────────────────────────────────────────────────────
    print("Loading run histories ...")
    all_runs: List[Dict] = []
    for r in raw:
        rd = load_run(r, rescale=args.rescale_metrics)
        if rd is None or rd["n_steps"] < args.min_steps:
            continue
        all_runs.append(rd)
    print(f"  {len(all_runs)} usable run(s).\n")

    if not all_runs:
        sys.exit("No usable runs after filtering.")

    # ── Config-value exclusions ──────────────────────────────────────────────
    if args.exclude_param:
        exclusions: List[Tuple[str, Any]] = []
        for spec in args.exclude_param:
            if "=" not in spec:
                sys.exit(f"--exclude-param must be PARAM=VAL, got: {spec!r}")
            key, raw_val = spec.split("=", 1)
            # Try to coerce to float so 1.0 matches config values stored as float
            try:
                val: Any = float(raw_val)
            except ValueError:
                val = raw_val
            exclusions.append((key.strip(), val))

        before = len(all_runs)
        def _excluded(rd: Dict) -> bool:
            for k, v in exclusions:
                cfg_v = rd["config"].get(k)
                if cfg_v is None:
                    continue
                try:
                    if float(cfg_v) == float(v):
                        return True
                except (TypeError, ValueError):
                    if str(cfg_v) == str(v):
                        return True
            return False

        all_runs = [rd for rd in all_runs if not _excluded(rd)]
        dropped  = before - len(all_runs)
        if dropped:
            print(f"  --exclude-param: dropped {dropped} run(s) "
                  f"({', '.join(f'{k}={v}' for k, v in exclusions)})")
            print(f"  {len(all_runs)} run(s) remaining.\n")

    # ── Outlier detection ────────────────────────────────────────────────────
    if not args.include_diverged:
        divg = detect_diverged(
            all_runs,
            iqr_threshold  = args.outlier_iqr,
            abs_threshold  = args.outlier_abs,
            rebound_ratio  = args.outlier_rebound,
        )
        if divg:
            print(f"  Filtering {len(divg)} diverged run(s):")
            for rd in all_runs:
                if rd["id"] in divg:
                    print(f"    [{_exp_key(rd)}]  {rd['name']}"
                          f"  →  {divg[rd['id']]}")
            all_runs = [rd for rd in all_runs if rd["id"] not in divg]
            print(f"  {len(all_runs)} run(s) remaining.\n")
        else:
            print("  No diverged runs detected.\n")

    if not all_runs:
        sys.exit(
            "All runs removed by outlier filter.\n"
            "Try --outlier-abs 0 / --outlier-iqr 10 / --include-diverged."
        )

    # ── Metric selection ─────────────────────────────────────────────────────
    user_metrics: Optional[List[str]] = None
    if args.metrics:
        user_metrics = [m.strip() for m in args.metrics.split(",")]

    groups = group_by_experiment(all_runs)
    print(f"Experiments found: {sorted(groups.keys())}\n")

    tex: List[str] = []

    # Parse map-error iteration list once
    map_err_iters = [int(x.strip())
                     for x in args.map_err_iterations.split(",")
                     if x.strip()]

    # ── Per-experiment plots ─────────────────────────────────────────────────
    for exp, runs in sorted(groups.items()):
        if not runs:
            continue
        safe    = exp.replace("/", "_")
        exp_dir = out / safe
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{exp}]  {len(runs)} run(s)")

        vp = _varying_params(runs)
        print(f"  Varying params: {vp or '(none)'}")

        metrics = user_metrics or _avail(runs, PRIMARY_METRICS)

        param_cmap = {
            "sigma":          "plasma",
            "particle_ratio": "viridis",
            "iterations":     "cividis",
            "cfm_epochs":     "magma",
            "dual_epochs":    "magma",
        }

        # Primary ablation param for map-err snapshots (first varying param)
        primary_param = vp[0] if vp else None

        for param in vp:
            cmap = param_cmap.get(param, "plasma")

            if not args.no_convergence:
                plot_convergence(runs, exp, exp_dir, tex, param=param,
                                 metrics=metrics[:5], cmap=cmap,
                                 logy=args.logy)

            if not args.no_scatter:
                plot_final_scatter(runs, exp, exp_dir, tex, param=param,
                                   metrics=metrics[:4], cmap=cmap)

            if not args.no_violin:
                plot_violin(runs, exp, exp_dir, tex, param=param,
                            metrics=metrics[:4], cmap=cmap)

            # Map-error snapshots at fixed iteration counts
            plot_map_err_at_iterations(
                runs, exp, exp_dir, tex,
                iteration_counts=map_err_iters,
                param=param,
                cmap=cmap,
            )

        # Heatmap: needs two varying params
        hx, hy = args.heatmap_x, args.heatmap_y
        if not args.no_heatmap and hx in vp and hy in vp:
            plot_heatmap(runs, exp, exp_dir, tex,
                         x_key=hx, y_key=hy,
                         metrics=metrics[:3])

        # Parallel coordinates (all metrics, curved lines)
        if not args.no_parallel and len(vp) >= 1:
            plot_parallel_coords(runs, exp, exp_dir, tex,
                                 target_metric=args.target_metric,
                                 params=vp)

        # ── Single-config fallback: plain convergence ────────────────────────
        if not vp and not args.no_convergence:
            import matplotlib.pyplot as plt
            avail = _avail(runs, PRIMARY_METRICS)[:4]
            if avail:
                nc  = len(avail)
                fig, axes = plt.subplots(1, nc,
                                         figsize=(nc * COL_W, 2.6),
                                         squeeze=False)
                col = PALETTE[0]
                for ax, m in zip(axes[0], avail):
                    series = []
                    for rd in runs:
                        if m not in rd["history"]:
                            continue
                        s, v = rd["history"][m]
                        if len(s) >= 2:
                            series.append((s, v))
                    if not series:
                        ax.set_visible(False)
                        continue
                    max_s = max(s[-1] for s, _ in series)
                    grid  = np.linspace(0, max_s, 200)
                    arr   = np.stack([np.interp(grid, s, v)
                                      for s, v in series], axis=0)
                    mean, std = arr.mean(0), arr.std(0)
                    ax.plot(grid, mean, color=col, lw=2.0, zorder=3)
                    ax.fill_between(grid, np.maximum(mean - std, 0),
                                    mean + std, color=col,
                                    alpha=0.14, zorder=2, linewidth=0)
                    _finalize_ax(ax, MLABEL_SHORT.get(m, m))
                    ax.set_title(MLABEL.get(m, m), fontsize=9.5)
                fig.suptitle(f"Convergence  —  {exp.replace('_', ' ')}",
                             fontsize=10, y=1.01)
                plt.tight_layout()
                _save(fig, exp_dir, f"{safe}__convergence", tex)

        print()  # blank line between experiments

    # ── Cross-experiment ─────────────────────────────────────────────────────
    if not args.no_cross and len(groups) >= 2:
        cross = out / "_cross"
        cross.mkdir(parents=True, exist_ok=True)
        print("[Cross-experiment]")
        plot_cross_experiment(groups, cross, tex)

    # ── LaTeX table ──────────────────────────────────────────────────────────
    if not args.no_table:
        # Determine dominant varying param across all runs (for table rows)
        all_runs_flat = [rd for rs in groups.values() for rd in rs]
        vp_global = _varying_params(all_runs_flat)
        tbl_param = vp_global[0] if vp_global else None
        write_latex_table(groups, out, tex,
                          param=tbl_param,
                          metrics=user_metrics or _avail(all_runs_flat,
                                                         PRIMARY_METRICS)[:6])

    write_latex_wrapper(tex, out, args.project)
    print(f"\nDone — {len(tex)} figure(s)/table(s) written to {out.resolve()}")


if __name__ == "__main__":
    main()
