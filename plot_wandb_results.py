#!/usr/bin/env python
"""
plot_wandb_results.py  --  Ablation-study visualiser for DualWCFM / EOT paper
==============================================================================
Fetches W&B runs and produces focused, publication-quality figures that tell
a clear ablation story for a NeurIPS submission.

Philosophy
----------
* One metric per panel.  Never mix metrics on the same axes.
* One line per unique hyperparameter value (sigma or particle_ratio).
  Lines are mean across all runs sharing that value; shaded band = +/-std.
* mu and nu quality are always paired side-by-side (they are conceptually
  linked: source-marginal fidelity and target-marginal fidelity).
* Outlier detection is conservative: crashed runs + final metric >= 1.0 +
  per-group IQR extreme outliers.

Usage
-----
  python plot_wandb_results.py --project "Data-Based EOT" --entity antoine311200
  python plot_wandb_results.py --project "Data-Based EOT" --rescale-metrics
  python plot_wandb_results.py --project "Data-Based EOT" --outlier-abs 0
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
# Constants
# ---------------------------------------------------------------------------

# Okabe-Ito colorblind-safe (8 colours, no yellow-on-white)
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # teal
    "#CC79A7",  # mauve
    "#56B4E9",  # sky blue
    "#E69F00",  # amber
    "#6B4C9A",  # purple
    "#555555",  # dark grey
]

COL_W  = 3.3    # NeurIPS single-column (inches)
COL_2W = 6.75   # NeurIPS double-column (inches)
DPI    = 300

# Human-readable metric labels (LaTeX math)
MLABEL: Dict[str, str] = {
    "map_err_fwd":    r"Map error $\|\hat T - T\|_2$",
    "map_err_bwd":    r"Map error $\|\hat T^{-1} - T^{-1}\|_2$",
    "sinkhorn_div":   r"Sinkhorn divergence",
    "mmd_nu":         r"MMD $(\hat\nu, \nu)$",
    "energy_dist_nu": r"Energy distance $(\hat\nu, \nu)$",
    "W1_nu":          r"$\mathrm{SW}_1(\hat\nu, \nu)$",
    "W2_nu":          r"$\mathrm{SW}_2(\hat\nu, \nu)$",
    "W1_mu":          r"$\mathrm{SW}_1(\hat\mu, \mu)$",
    "W2_mu":          r"$\mathrm{SW}_2(\hat\mu, \mu)$",
    "bw2":            r"$\mathrm{BW}_2$ (joint coupling)",
    "fid":            r"FID",
    "kid":            r"KID",
    "weight_ess":     r"Effective sample size (ESS)",
    "weight_mean":    r"Weight mean $\mathbb{E}[w]$",
    "weight_std":     r"Weight std $\sigma(w)$",
}

# Which metrics are "lower is better"
LOWER_IS_BETTER: Set[str] = {
    "map_err_fwd", "map_err_bwd",
    "sinkhorn_div", "mmd_nu", "energy_dist_nu",
    "W1_nu", "W2_nu", "W1_mu", "W2_mu",
    "bw2", "fid", "kid", "weight_std",
}

# Physically must converge < 1 in a well-scaled run
MUST_CONVERGE: Set[str] = {
    "map_err_fwd", "map_err_bwd",
    "sinkhorn_div", "mmd_nu", "energy_dist_nu",
    "W1_nu", "W2_nu", "W1_mu", "W2_mu", "bw2",
}

# Ordered list used when auto-selecting metrics to plot
PRIMARY_METRICS  = ["map_err_fwd", "map_err_bwd", "sinkhorn_div",
                    "W2_nu", "mmd_nu", "energy_dist_nu", "bw2"]
TARGET_MARGINAL  = ["W2_nu", "W1_nu"]
SOURCE_MARGINAL  = ["W2_mu", "W1_mu"]
WEIGHT_METRICS   = ["weight_ess", "weight_mean", "weight_std"]

HPARAM_LABEL: Dict[str, str] = {
    "sigma":          r"$\sigma$",
    "particle_ratio": r"$\rho$",
    "dual_epochs":    r"$E_\mathrm{dual}$",
    "cfm_epochs":     r"$E_\mathrm{cfm}$",
    "advance_steps":  r"adv. steps",
    "batch_size":     r"batch size",
    "scale_factor":   r"scale",
}

RESCALABLE: Set[str] = {
    "W1_mu", "W2_mu", "W1_nu", "W2_nu",
    "mmd_nu", "energy_dist_nu", "sinkhorn_div",
    "map_err_fwd", "map_err_bwd", "bw2",
}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "text.usetex":          False,
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "mathtext.fontset":     "cm",
        "figure.facecolor":     "white",
        "savefig.facecolor":    "white",
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.04,
        "figure.dpi":           110,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.8,
        "axes.labelsize":       9,
        "axes.titlesize":       9.5,
        "axes.titleweight":     "semibold",
        "axes.prop_cycle":      mpl.cycler("color", PALETTE),
        "axes.grid":            True,
        "axes.grid.axis":       "y",
        "grid.linewidth":       0.35,
        "grid.alpha":           0.5,
        "grid.color":           "#CCCCCC",
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.size":     3,
        "ytick.major.size":     3,
        "lines.linewidth":      2.0,
        "lines.markersize":     5,
        "legend.fontsize":      8,
        "legend.framealpha":    0.93,
        "legend.edgecolor":     "#DDDDDD",
        "legend.handlelength":  1.6,
        "legend.labelspacing":  0.35,
        "legend.borderpad":     0.45,
    })

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _wandb_api():
    try:
        import wandb
        return wandb.Api()
    except ImportError:
        sys.exit("wandb required: pip install wandb")


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _exp_key(rd: Dict) -> str:
    cfg = rd["config"]
    return cfg.get("base_experiment") or cfg.get("experiment") or "unknown"


def fetch_runs(project, entity=None, exp_filter=None, max_runs=None):
    api  = _wandb_api()
    path = f"{entity}/{project}" if entity else project
    filt: Dict = {}
    if exp_filter:
        filt["$or"] = [
            {"config.experiment":      {"$regex": exp_filter}},
            {"config.base_experiment": {"$regex": exp_filter}},
        ]
    print(f"Fetching from {path} ...", flush=True)
    runs = list(api.runs(path, filters=filt or None))
    if max_runs:
        runs = runs[:max_runs]
    print(f"  {len(runs)} runs found.", flush=True)
    return runs


def load_run(run, rescale: bool = False) -> Optional[Dict]:
    cfg          = dict(run.config)
    scale_factor = _safe_float(cfg.get("scale_factor", 1.0)) or 1.0

    try:
        rows = run.history(samples=500, x_axis="_step", pandas=False)
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
    n_steps = max((int(s[-1]) for (s, _) in history.values()), default=0)

    return dict(
        name=run.name or run.id, id=run.id, state=run.state,
        config=cfg, scale_factor=scale_factor,
        history=history, final=final, n_steps=n_steps,
    )


def group_by_experiment(runs: List[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = defaultdict(list)
    for rd in runs:
        out[_exp_key(rd)].append(rd)
    return dict(out)

# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_diverged(
    runs: List[Dict],
    iqr_threshold: float = 3.5,
    abs_threshold: float = 1.0,
    rebound_ratio: float = 3.0,
) -> Dict[str, str]:
    """Return {run_id: reason_string} for diverged runs.

    C1. W&B state crashed/failed.
    C2. Final value of a MUST_CONVERGE metric >= abs_threshold (default 1.0).
    C3. Final value is a Tukey outer-fence outlier within its experiment group.
    C4. Run rebounded: final > best_ever * rebound_ratio, only when best < 0.5.
    """
    reasons: Dict[str, str] = {}

    def flag(rid, reason):
        if rid not in reasons:
            reasons[rid] = reason

    # C1
    for rd in runs:
        if rd.get("state") in ("crashed", "failed"):
            flag(rd["id"], f"state={rd['state']}")

    # C2
    if abs_threshold > 0:
        for rd in runs:
            for m in MUST_CONVERGE:
                fv = rd["final"].get(m)
                if fv is not None and fv >= abs_threshold:
                    flag(rd["id"], f"final {m}={fv:.3g} >= {abs_threshold}")
                    break

    # C3 — per group
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

    # C4
    if rebound_ratio > 0:
        for rd in runs:
            for m in MUST_CONVERGE:
                if m not in rd["history"]:
                    continue
                _, vals = rd["history"][m]
                if len(vals) < 6:
                    continue
                best = float(np.min(vals))
                last = float(vals[-1])
                if best < 0.5 and last > best * rebound_ratio:
                    flag(rd["id"],
                         f"{m} rebounded: best={best:.3g} -> final={last:.3g} "
                         f"(x{last/best:.1f})")
                    break

    return reasons

# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _avail(runs: List[Dict], pool: List[str]) -> List[str]:
    have = set().union(*[set(rd["history"]) for rd in runs])
    return [m for m in pool if m in have]


def _varying_params(runs: List[Dict]) -> List[str]:
    candidates = ["sigma", "particle_ratio", "dual_epochs",
                  "cfm_epochs", "advance_steps", "batch_size"]
    return [k for k in candidates
            if len({rd["config"].get(k) for rd in runs
                    if k in rd["config"]}) > 1]


def aggregate_by_param(
    runs: List[Dict],
    param: str,
    metrics: List[str],
    n_grid: int = 200,
) -> Dict[Any, Dict[str, Any]]:
    """Group runs by unique values of `param`, interpolate histories to a
    common step grid, return mean and std for each metric.

    Returns:
        {
          param_value: {
            "steps":  np.ndarray,          # common grid
            "mean":   {metric: np.ndarray},
            "std":    {metric: np.ndarray},
            "n":      int,                 # number of runs in group
          }
        }
    """
    # Collect unique values
    unique_vals = sorted(
        {rd["config"].get(param) for rd in runs if rd["config"].get(param) is not None}
    )
    result: Dict[Any, Dict[str, Any]] = {}

    for val in unique_vals:
        group = [rd for rd in runs if rd["config"].get(param) == val]
        if not group:
            continue

        # Common step grid: span from 0 to max steps across this group
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
                interp = np.interp(grid, steps, vals)
                series.append(interp)
            if not series:
                continue
            arr = np.stack(series, axis=0)   # (n_runs, n_grid)
            mean_d[m] = arr.mean(axis=0)
            std_d[m]  = arr.std(axis=0)

        result[val] = {
            "steps": grid,
            "mean":  mean_d,
            "std":   std_d,
            "n":     len(group),
        }

    return result


def param_colors(unique_vals: List, cmap_name: str = "plasma") -> Dict[Any, Any]:
    """Assign a color from a sequential colormap to each unique parameter value."""
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap(cmap_name)
    n = len(unique_vals)
    return {v: cmap(0.15 + 0.7 * i / max(n - 1, 1))
            for i, v in enumerate(sorted(unique_vals))}

# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, out: Path, name: str, tex: List[str]) -> None:
    import matplotlib.pyplot as plt
    png = out / f"{name}.png"
    pdf = out / f"{name}.pdf"
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    plt.close(fig)
    label   = name.replace("_", "-").replace("/", "-")
    caption = name.replace("_", " ").replace("-", " ").replace("__", " -- ").capitalize()
    tex.append(
        f"\\begin{{figure}}[htbp]\n  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{figures/{name}.pdf}}\n"
        f"  \\caption{{{caption}}}\n  \\label{{fig:{label}}}\n"
        f"\\end{{figure}}\n"
    )
    print(f"  {png.name}")

# ---------------------------------------------------------------------------
# Core drawing primitive
# ---------------------------------------------------------------------------

def _draw_convergence_line(
    ax,
    agg: Dict[Any, Dict[str, Any]],
    metric: str,
    colors: Dict,
    label_prefix: str = "",
) -> bool:
    """Draw one line per param value on `ax` for `metric`.
    Returns True if any data was drawn."""
    drawn = False
    for val in sorted(agg.keys()):
        entry = agg[val]
        if metric not in entry["mean"]:
            continue
        steps = entry["steps"]
        mean  = entry["mean"][metric]
        std   = entry["std"][metric]
        n     = entry["n"]
        col   = colors.get(val, PALETTE[0])
        lbl   = f"{label_prefix}{val:.3g}" if isinstance(val, float) else f"{label_prefix}{val}"
        if n > 1:
            lbl += f" (n={n})"
        ax.plot(steps, mean, color=col, lw=2.0, label=lbl, zorder=3)
        ax.fill_between(steps, mean - std, mean + std,
                        color=col, alpha=0.15, zorder=2)
        drawn = True
    return drawn


def _finalize_ax(ax, ylabel: str, xlabel: str = "Iteration") -> None:
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(left=0)
    # Never let y-axis go below 0 for positive metrics
    ylo, yhi = ax.get_ylim()
    if ylo < 0:
        ax.set_ylim(bottom=0)

# ---------------------------------------------------------------------------
# P1 — Ablation: one hyperparameter, multiple metrics as rows
# ---------------------------------------------------------------------------

def plot_ablation(
    runs: List[Dict],
    exp: str,
    out: Path,
    tex: List[str],
    param: str,                         # "sigma" or "particle_ratio"
    metrics: Optional[List[str]] = None,
    cmap: str = "plasma",
) -> None:
    """One column of panels (one per metric), lines coloured by param value."""
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = _avail(runs, PRIMARY_METRICS)[:4]
    metrics = [m for m in metrics if any(m in rd["history"] for rd in runs)]
    if not metrics:
        return

    agg    = aggregate_by_param(runs, param, metrics)
    if not agg:
        return
    colors = param_colors(list(agg.keys()), cmap)
    plabel = HPARAM_LABEL.get(param, param) + " = "

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(min(COL_2W, n * 2.5), 2.6),
                             squeeze=False)
    axes = axes[0]

    any_drawn = False
    for ax, m in zip(axes, metrics):
        drawn = _draw_convergence_line(ax, agg, m, colors, plabel)
        _finalize_ax(ax, MLABEL.get(m, m))
        ax.set_title(MLABEL.get(m, m), fontsize=9, pad=5)
        if drawn:
            any_drawn = True

    if not any_drawn:
        plt.close(fig)
        return

    # Shared legend in last panel or outside
    handles, labels = axes[-1].get_legend_handles_labels()
    if not handles:
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    if handles:
        n_vals = len(handles)
        if n_vals <= 5:
            axes[-1].legend(handles, labels, fontsize=8, loc="best",
                            framealpha=0.93, edgecolor="#DDD")
        else:
            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(0.5, -0.04),
                       ncol=n_vals, fontsize=8,
                       framealpha=0.93, edgecolor="#DDD")

    safe  = exp.replace("/", "_")
    title = (f"Ablation: {HPARAM_LABEL.get(param, param)}"
             f"  —  {exp.replace('_', ' ')}")
    fig.suptitle(title, fontsize=9.5, y=1.02)
    plt.tight_layout()
    _save(fig, out, f"{safe}__ablation_{param}", tex)


# ---------------------------------------------------------------------------
# P2 — Marginal quality pair: mu and nu side-by-side
# ---------------------------------------------------------------------------

def plot_marginal_pair(
    runs: List[Dict],
    exp: str,
    out: Path,
    tex: List[str],
    param: str = "sigma",               # colour lines by this param
    cmap: str = "plasma",
) -> None:
    """Side-by-side: source marginal quality (mu) and target (nu)."""
    import matplotlib.pyplot as plt

    nu_metrics = _avail(runs, TARGET_MARGINAL)
    mu_metrics = _avail(runs, SOURCE_MARGINAL)
    if not nu_metrics and not mu_metrics:
        return

    all_metrics = mu_metrics[:2] + nu_metrics[:2]
    agg    = aggregate_by_param(runs, param, all_metrics)
    if not agg:
        return
    colors = param_colors(list(agg.keys()), cmap)
    plabel = HPARAM_LABEL.get(param, param) + " = "

    panels = []
    if mu_metrics:
        panels.append(("Source marginal  " + r"$\hat\mu \to \mu$",
                       mu_metrics[:2]))
    if nu_metrics:
        panels.append(("Target marginal  " + r"$\hat\nu \to \nu$",
                       nu_metrics[:2]))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(n_panels * 3.0, 2.8), squeeze=False)
    axes = axes[0]

    for ax, (title, mlist) in zip(axes, panels):
        ls_cycle = ["-", "--"]
        for li, m in enumerate(mlist):
            for val in sorted(agg.keys()):
                entry = agg[val]
                if m not in entry["mean"]:
                    continue
                col   = colors[val]
                steps = entry["steps"]
                mean  = entry["mean"][m]
                std   = entry["std"][m]
                lbl   = (f"{plabel}{val:.3g}"
                         if isinstance(val, float)
                         else f"{plabel}{val}")
                suffix = f"  [{MLABEL.get(m, m).split('(')[0].strip()}]"
                ax.plot(steps, mean, color=col, lw=2.0, ls=ls_cycle[li],
                        label=(lbl + suffix) if li == 0 else "_",
                        zorder=3)
                ax.fill_between(steps, mean - std, mean + std,
                                color=col, alpha=0.12, zorder=2)
        _finalize_ax(ax, "SW distance")
        ax.set_title(title, fontsize=9, pad=5)

    # Legend — unique by label
    seen: Dict[str, Any] = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen and not l.startswith("_"):
                seen[l] = h

    if seen:
        fig.legend(list(seen.values()), list(seen.keys()),
                   loc="upper center", bbox_to_anchor=(0.5, -0.04),
                   ncol=min(len(seen), 5), fontsize=8,
                   framealpha=0.93, edgecolor="#DDD")

    safe = exp.replace("/", "_")
    fig.suptitle(f"Marginal quality  —  {exp.replace('_', ' ')}",
                 fontsize=9.5, y=1.02)
    plt.tight_layout()
    _save(fig, out, f"{safe}__marginal_pair_{param}", tex)


# ---------------------------------------------------------------------------
# P3 — Convergence grid: all available metrics, one panel each
# ---------------------------------------------------------------------------

def plot_convergence_grid(
    runs: List[Dict],
    exp: str,
    out: Path,
    tex: List[str],
    param: str,
    cmap: str = "plasma",
) -> None:
    """All available metrics in a grid; lines by param value."""
    import matplotlib.pyplot as plt

    all_metrics = _avail(runs, PRIMARY_METRICS + WEIGHT_METRICS)
    if not all_metrics:
        return

    agg = aggregate_by_param(runs, param, all_metrics)
    if not agg:
        return
    colors = param_colors(list(agg.keys()), cmap)
    plabel = HPARAM_LABEL.get(param, param) + " = "

    n  = len(all_metrics)
    nc = min(n, 4)
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc,
                             figsize=(nc * 2.6, nr * 2.4),
                             squeeze=False)
    ax_flat = [axes[r][c] for r in range(nr) for c in range(nc)]

    for ax, m in zip(ax_flat, all_metrics):
        drawn = _draw_convergence_line(ax, agg, m, colors, plabel)
        if m == "weight_ess":
            ax.axhline(0.8, color="#009E73", lw=0.9, ls="--", alpha=0.6)
            ax.axhline(0.5, color="#D55E00", lw=0.9, ls="--", alpha=0.6)
            ax.set_ylim(0, 1.1)
        _finalize_ax(ax, MLABEL.get(m, m))
        ax.set_title(MLABEL.get(m, m), fontsize=8.5, pad=4)
        if not drawn:
            ax.set_visible(False)

    # Hide unused panels
    for ax in ax_flat[len(all_metrics):]:
        ax.set_visible(False)

    # Single legend, shared
    handles, labels = [], []
    for ax in ax_flat:
        if not ax.get_visible():
            continue
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.02),
                   ncol=min(len(handles), 6), fontsize=8,
                   framealpha=0.93, edgecolor="#DDD")

    safe  = exp.replace("/", "_")
    title = (f"All metrics — ablation {HPARAM_LABEL.get(param, param)}"
             f"  |  {exp.replace('_', ' ')}")
    fig.suptitle(title, fontsize=9.5, y=1.01)
    plt.tight_layout()
    _save(fig, out, f"{safe}__metric_grid_{param}", tex)


# ---------------------------------------------------------------------------
# P4 — Final-value summary: dot + error-bar per hyperparameter value
# ---------------------------------------------------------------------------

def plot_final_summary(
    runs: List[Dict],
    exp: str,
    out: Path,
    tex: List[str],
    param: str,
    cmap: str = "plasma",
) -> None:
    """Horizontal dot-plot of final metric vs param value."""
    import matplotlib.pyplot as plt

    metrics = _avail(runs, PRIMARY_METRICS)[:4]
    if not metrics:
        return

    unique_vals = sorted({rd["config"].get(param)
                          for rd in runs if rd["config"].get(param) is not None})
    if len(unique_vals) < 2:
        return

    colors = param_colors(unique_vals, cmap)

    n  = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(min(COL_2W, n * 2.5), 2.8),
                             squeeze=False)
    axes = axes[0]

    for ax, m in zip(axes, metrics):
        positions = range(len(unique_vals))
        for xi, val in enumerate(unique_vals):
            group = [rd for rd in runs if rd["config"].get(param) == val
                     and m in rd["final"]]
            if not group:
                continue
            final_vals = [rd["final"][m] for rd in group]
            mean = np.mean(final_vals)
            std  = np.std(final_vals) if len(final_vals) > 1 else 0.0
            col  = colors[val]

            # Error bar
            ax.errorbar(xi, mean, yerr=std, fmt="o", color=col,
                        markersize=7, capsize=4, capthick=1.5, elinewidth=1.2,
                        zorder=4)
            # Individual points (jitter)
            jitter = np.random.uniform(-0.12, 0.12, len(final_vals))
            ax.scatter(xi + jitter, final_vals, color=col,
                       s=20, alpha=0.6, edgecolors="white",
                       linewidths=0.4, zorder=3)

        ax.set_xticks(range(len(unique_vals)))
        ax.set_xticklabels(
            [f"{v:.3g}" if isinstance(v, float) else str(v)
             for v in unique_vals],
            fontsize=8,
        )
        ax.set_xlabel(HPARAM_LABEL.get(param, param), fontsize=9)
        ax.set_ylabel(MLABEL.get(m, m), fontsize=9)
        ax.set_title(MLABEL.get(m, m), fontsize=9, pad=5)
        ax.grid(True, axis="y", linewidth=0.35, alpha=0.5)

    safe  = exp.replace("/", "_")
    title = (f"Final metric vs {HPARAM_LABEL.get(param, param)}"
             f"  —  {exp.replace('_', ' ')}")
    fig.suptitle(title, fontsize=9.5, y=1.02)
    plt.tight_layout()
    _save(fig, out, f"{safe}__final_vs_{param}", tex)


# ---------------------------------------------------------------------------
# P5 — ESS health over iterations (separate, always useful)
# ---------------------------------------------------------------------------

def plot_ess(
    runs: List[Dict],
    exp: str,
    out: Path,
    tex: List[str],
    param: str,
    cmap: str = "plasma",
) -> None:
    if not any("weight_ess" in rd["history"] for rd in runs):
        return
    import matplotlib.pyplot as plt

    agg    = aggregate_by_param(runs, param, ["weight_ess"])
    if not agg:
        return
    colors = param_colors(list(agg.keys()), cmap)
    plabel = HPARAM_LABEL.get(param, param) + " = "

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for val in sorted(agg.keys()):
        entry = agg[val]
        if "weight_ess" not in entry["mean"]:
            continue
        col   = colors[val]
        steps = entry["steps"]
        mean  = entry["mean"]["weight_ess"]
        std   = entry["std"]["weight_ess"]
        lbl   = (f"{plabel}{val:.3g}" if isinstance(val, float)
                 else f"{plabel}{val}")
        ax.plot(steps, mean, color=col, lw=2.0, label=lbl, zorder=3)
        ax.fill_between(steps, mean - std, mean + std,
                        color=col, alpha=0.15, zorder=2)

    ax.axhline(0.8, color="#009E73", lw=1.0, ls="--", alpha=0.6, label="ESS = 0.8")
    ax.axhline(0.5, color="#D55E00", lw=1.0, ls="--", alpha=0.6, label="ESS = 0.5")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Effective sample size (ESS)")
    ax.set_title(f"Weight ESS  —  {exp.replace('_', ' ')}", fontsize=9)

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 7:
        ax.legend(fontsize=8, loc="lower right", framealpha=0.93, edgecolor="#DDD")
    else:
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.04),
                   ncol=min(len(handles), 6), fontsize=8)

    safe = exp.replace("/", "_")
    plt.tight_layout()
    _save(fig, out, f"{safe}__ess_{param}", tex)


# ---------------------------------------------------------------------------
# P6 — Cross-experiment: best run per experiment, convergence comparison
# ---------------------------------------------------------------------------

def plot_cross_convergence(
    groups: Dict[str, List[Dict]],
    out: Path,
    tex: List[str],
) -> None:
    """For each key metric: one line per experiment (best run in each)."""
    import matplotlib.pyplot as plt

    metrics = []
    for m in PRIMARY_METRICS:
        if any(any(m in rd["history"] for rd in rs) for rs in groups.values()):
            metrics.append(m)
    metrics = metrics[:4]
    if not metrics:
        return

    exp_list = sorted(groups.keys())
    colors   = {exp: PALETTE[i % len(PALETTE)] for i, exp in enumerate(exp_list)}

    n  = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(min(COL_2W, n * 2.6), 2.8),
                             squeeze=False)
    axes = axes[0]

    for ax, m in zip(axes, metrics):
        for exp in exp_list:
            rs = groups[exp]
            cands = [rd for rd in rs if m in rd["history"]]
            if not cands:
                continue
            best = min(cands, key=lambda r: r["final"].get(m, np.inf))
            steps, vals = best["history"][m]
            col = colors[exp]
            ax.plot(steps, vals, color=col, lw=1.8, alpha=0.85,
                    label=exp.replace("_", " "), zorder=3)

        _finalize_ax(ax, MLABEL.get(m, m))
        ax.set_title(MLABEL.get(m, m), fontsize=9, pad=5)

    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)

    if handles:
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.04),
                   ncol=min(len(handles), 5), fontsize=8,
                   framealpha=0.93, edgecolor="#DDD")

    fig.suptitle("Best run per experiment  —  convergence comparison",
                 fontsize=9.5, y=1.02)
    plt.tight_layout()
    _save(fig, out, "cross__convergence_comparison", tex)


# ---------------------------------------------------------------------------
# P7 — Cross-experiment: final metric bar chart (all experiments)
# ---------------------------------------------------------------------------

def plot_cross_final_bar(
    groups: Dict[str, List[Dict]],
    out: Path,
    tex: List[str],
) -> None:
    """Bar chart: final metric per experiment, one panel per metric."""
    import matplotlib.pyplot as plt

    # Metrics available in at least 2 experiments
    metric_counts: Dict[str, int] = defaultdict(int)
    for rs in groups.values():
        for rd in rs:
            for m in rd["final"]:
                metric_counts[m] += 1
    metrics = [m for m in PRIMARY_METRICS
               if metric_counts[m] >= 2][:4]
    if not metrics:
        return

    exp_list = sorted(groups.keys())
    colors   = {exp: PALETTE[i % len(PALETTE)] for i, exp in enumerate(exp_list)}

    n  = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(min(COL_2W, n * 2.6), 2.8),
                             squeeze=False, sharey=False)
    axes = axes[0]

    for ax, m in zip(axes, metrics):
        for xi, exp in enumerate(exp_list):
            vals = [rd["final"][m] for rd in groups[exp] if m in rd["final"]]
            if not vals:
                continue
            mean = np.mean(vals)
            std  = np.std(vals) if len(vals) > 1 else 0.0
            col  = colors[exp]

            ax.bar(xi, mean, yerr=std, color=col, alpha=0.80,
                   capsize=4, error_kw={"lw": 1.2, "capthick": 1.2},
                   width=0.65, zorder=3)
            # Individual points
            jitter = np.random.uniform(-0.18, 0.18, len(vals))
            ax.scatter(xi + jitter, vals, color=col, s=22,
                       edgecolors="white", linewidths=0.5,
                       zorder=4, alpha=0.85)
            ax.text(xi, mean + std * 1.15 + 1e-6,
                    f"{mean:.3g}", ha="center", va="bottom",
                    fontsize=7, color="#333")

        ax.set_xticks(range(len(exp_list)))
        ax.set_xticklabels(
            [e.replace("_", "\n").replace("perturbed", "pert.")
             for e in exp_list],
            fontsize=7,
        )
        ax.set_ylabel(MLABEL.get(m, m), fontsize=9)
        ax.set_title(MLABEL.get(m, m), fontsize=9, pad=5)
        ax.grid(True, axis="y", linewidth=0.35, alpha=0.5)

    fig.suptitle("Final metric across experiments", fontsize=9.5, y=1.02)
    plt.tight_layout()
    _save(fig, out, "cross__final_bar", tex)


# ---------------------------------------------------------------------------
# P8 — Sigma × particle_ratio heatmap of final metric (one per metric)
# ---------------------------------------------------------------------------

def plot_heatmap(
    runs: List[Dict],
    exp: str,
    out: Path,
    tex: List[str],
    x_key: str = "sigma",
    y_key: str = "particle_ratio",
) -> None:
    """Clean annotated heatmap for each available metric."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    metrics = _avail(runs, PRIMARY_METRICS)[:3]
    if not metrics:
        return

    x_vals = sorted({rd["config"].get(x_key) for rd in runs
                     if rd["config"].get(x_key) is not None})
    y_vals = sorted({rd["config"].get(y_key) for rd in runs
                     if rd["config"].get(y_key) is not None})
    if len(x_vals) < 2 or len(y_vals) < 2:
        return

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(min(COL_2W, n * 2.8), 2.6),
                             squeeze=False)
    axes = axes[0]

    # green=low=good, red=high=bad
    cmap_lb = LinearSegmentedColormap.from_list(
        "lb", ["#1a7431", "#a8ddb5", "#fff9db", "#f97316", "#7f1d1d"], N=256)

    for ax, m in zip(axes, metrics):
        cell: Dict = defaultdict(list)
        for rd in runs:
            xv = rd["config"].get(x_key)
            yv = rd["config"].get(y_key)
            fv = rd["final"].get(m)
            if None in (xv, yv, fv) or xv not in x_vals or yv not in y_vals:
                continue
            cell[(x_vals.index(xv), y_vals.index(yv))].append(fv)

        grid = np.full((len(y_vals), len(x_vals)), np.nan)
        grid_n = np.zeros_like(grid, dtype=int)
        for (xi, yi), cv in cell.items():
            grid[yi, xi] = float(np.mean(cv))
            grid_n[yi, xi] = len(cv)

        if np.all(np.isnan(grid)):
            ax.set_visible(False)
            continue

        cmap = cmap_lb if m in LOWER_IS_BETTER else cmap_lb
        im   = ax.imshow(grid, aspect="auto", origin="lower",
                         cmap=cmap, interpolation="nearest")
        ax.grid(False)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([f"{v:.3g}" for v in x_vals], fontsize=8)
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([f"{v:.3g}" for v in y_vals], fontsize=8)
        ax.set_xlabel(HPARAM_LABEL.get(x_key, x_key), fontsize=9)
        ax.set_ylabel(HPARAM_LABEL.get(y_key, y_key), fontsize=9)
        ax.set_title(MLABEL.get(m, m), fontsize=9, pad=5)

        vmin, vmax = np.nanmin(grid), np.nanmax(grid)
        rng = max(vmax - vmin, 1e-12)
        for yi in range(len(y_vals)):
            for xi in range(len(x_vals)):
                v = grid[yi, xi]
                if np.isnan(v):
                    continue
                brightness = (v - vmin) / rng
                txt_col    = "white" if brightness < 0.45 else "#111"
                txt = f"{v:.3g}"
                if grid_n[yi, xi] > 1:
                    txt += f"\n(n={grid_n[yi, xi]})"
                ax.text(xi, yi, txt, ha="center", va="center",
                        fontsize=7, color=txt_col, fontweight="semibold")

        cb = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(MLABEL.get(m, m), fontsize=7.5)

    safe  = exp.replace("/", "_")
    fig.suptitle(
        f"Final metric — {HPARAM_LABEL.get(x_key, x_key)} x "
        f"{HPARAM_LABEL.get(y_key, y_key)}  |  {exp.replace('_', ' ')}",
        fontsize=9.5, y=1.02,
    )
    plt.tight_layout()
    _save(fig, out, f"{safe}__heatmap", tex)


# ---------------------------------------------------------------------------
# LaTeX index
# ---------------------------------------------------------------------------

def write_latex(tex: List[str], out_dir: Path, project: str) -> None:
    p = out_dir / "paper_figures.tex"
    p.write_text(
        "% Auto-generated\n"
        f"% Project: {project}\n"
        "% \\input{paper_figures.tex}\n"
        "%\n% Preamble: \\usepackage{graphicx,subcaption}\n"
        "% \\graphicspath{{./plots/}}\n\n"
        + "\n".join(tex),
        encoding="utf-8",
    )
    print(f"\nLaTeX: {p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python plot_wandb_results.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project",    required=True)
    p.add_argument("--entity",     default=None)
    p.add_argument("--output-dir", default="plots")
    p.add_argument("--experiment", default=None,
                   help="Filter runs by experiment name substring.")
    p.add_argument("--max-runs",   type=int, default=None)
    p.add_argument("--min-steps",  type=int, default=3)
    p.add_argument("--rescale-metrics", action="store_true",
                   help="Multiply W1_*, W2_*, map_err_*, mmd_nu, sinkhorn_div, bw2 "
                        "by scale_factor (pre-fix runs).")
    p.add_argument("--outlier-abs",     type=float, default=1.0,
                   help="Flag runs with any final MUST_CONVERGE metric >= this (default 1.0).")
    p.add_argument("--outlier-iqr",     type=float, default=3.5,
                   help="Tukey IQR multiplier per experiment group (default 3.5).")
    p.add_argument("--outlier-rebound", type=float, default=3.0,
                   help="Flag rebound: final > best*ratio when best < 0.5 (default 3.0).")
    p.add_argument("--include-diverged", action="store_true")
    p.add_argument("--no-cross",   action="store_true")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)
    args = _parser().parse_args()

    _setup_style()
    import matplotlib
    matplotlib.use("Agg")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out.resolve()}\n")

    # Fetch
    raw = fetch_runs(args.project, args.entity, args.experiment, args.max_runs)
    if not raw:
        sys.exit("No runs found.")

    # Load
    print("Loading histories ...")
    all_runs: List[Dict] = []
    for r in raw:
        rd = load_run(r, rescale=args.rescale_metrics)
        if rd is None or rd["n_steps"] < args.min_steps:
            continue
        all_runs.append(rd)
    print(f"  {len(all_runs)} usable runs.")

    # Filter diverged
    if not args.include_diverged:
        divg = detect_diverged(
            all_runs,
            iqr_threshold=args.outlier_iqr,
            abs_threshold=args.outlier_abs,
            rebound_ratio=args.outlier_rebound,
        )
        if divg:
            print(f"\n  Filtered {len(divg)} diverged run(s):")
            for rd in all_runs:
                if rd["id"] in divg:
                    print(f"    [{_exp_key(rd)}] {rd['name']}"
                          f"  --  {divg[rd['id']]}")
            all_runs = [rd for rd in all_runs if rd["id"] not in divg]
            print(f"  {len(all_runs)} runs remaining.\n")
        else:
            print("  No diverged runs.\n")

    if not all_runs:
        sys.exit(
            "All runs filtered out.\n"
            "Use --outlier-abs 0 / --outlier-iqr 10 / --include-diverged."
        )

    groups = group_by_experiment(all_runs)
    print(f"Experiments: {sorted(groups.keys())}\n")

    tex: List[str] = []

    # Per-experiment
    for exp, runs in sorted(groups.items()):
        if not runs:
            continue
        safe    = exp.replace("/", "_")
        exp_dir = out / safe
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{exp}]  {len(runs)} run(s)")

        vp = _varying_params(runs)

        # For each axis that actually varies, make ablation plots
        for param, cmap in [("sigma", "plasma"), ("particle_ratio", "viridis")]:
            if param not in vp:
                continue
            # Core ablation: focused metrics
            plot_ablation(runs, exp, exp_dir, tex, param=param, cmap=cmap)
            # Marginal quality pair
            plot_marginal_pair(runs, exp, exp_dir, tex, param=param, cmap=cmap)
            # Full metric grid
            plot_convergence_grid(runs, exp, exp_dir, tex, param=param, cmap=cmap)
            # Final-value summary
            plot_final_summary(runs, exp, exp_dir, tex, param=param, cmap=cmap)
            # ESS health
            plot_ess(runs, exp, exp_dir, tex, param=param, cmap=cmap)

        # If both vary: heatmap
        if "sigma" in vp and "particle_ratio" in vp:
            plot_heatmap(runs, exp, exp_dir, tex)

        # If only one param varies (or none), still make a convergence grid
        # using the first varying param (or sigma by default)
        if not vp:
            # All runs have same config — show one convergence panel per metric
            agg = {"all": {
                "steps": None, "mean": {}, "std": {}, "n": len(runs),
            }}
            # Rebuild aggregation manually for single-group case
            all_metrics = _avail(runs, PRIMARY_METRICS + WEIGHT_METRICS)
            for m in all_metrics:
                series = []
                for rd in runs:
                    if m not in rd["history"]:
                        continue
                    steps, vals = rd["history"][m]
                    if len(steps) < 2:
                        continue
                    series.append((steps, vals))
                if not series:
                    continue
                max_s = max(s[-1] for s, _ in series)
                grid  = np.linspace(0, max_s, 200)
                interp_arr = np.stack(
                    [np.interp(grid, s, v) for s, v in series], axis=0)
                agg["all"]["steps"]      = grid
                agg["all"]["mean"][m]    = interp_arr.mean(0)
                agg["all"]["std"][m]     = interp_arr.std(0)

            # Use the single agg to make a plain convergence figure
            import matplotlib.pyplot as plt
            avail_m = [m for m in PRIMARY_METRICS if m in agg["all"]["mean"]][:4]
            if avail_m:
                nc = len(avail_m)
                fig, axes = plt.subplots(1, nc, figsize=(nc * 2.6, 2.6),
                                         squeeze=False)
                col = PALETTE[0]
                for ax, m in zip(axes[0], avail_m):
                    steps = agg["all"]["steps"]
                    mean  = agg["all"]["mean"][m]
                    std   = agg["all"]["std"][m]
                    ax.plot(steps, mean, color=col, lw=2.0, zorder=3)
                    ax.fill_between(steps, mean - std, mean + std,
                                    color=col, alpha=0.15, zorder=2)
                    _finalize_ax(ax, MLABEL.get(m, m))
                    ax.set_title(MLABEL.get(m, m), fontsize=9, pad=4)
                fig.suptitle(f"Convergence  —  {exp.replace('_', ' ')}",
                             fontsize=9.5, y=1.02)
                plt.tight_layout()
                _save(fig, exp_dir, f"{safe}__convergence", tex)

    # Cross-experiment
    if not args.no_cross and len(groups) >= 2:
        cross = out / "_cross"
        cross.mkdir(parents=True, exist_ok=True)
        print("\n[Cross-experiment]")
        plot_cross_convergence(groups, cross, tex)
        plot_cross_final_bar(groups, cross, tex)

    write_latex(tex, out, args.project)
    print(f"\nDone  --  {len(tex)} figures in {out.resolve()}")


if __name__ == "__main__":
    main()
