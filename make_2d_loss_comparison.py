#!/usr/bin/env python
"""
make_2d_loss_comparison.py
==========================
Fetch 2-D experiment results from W&B (project: dualwcfm-2d, entity: antoine311200)
and produce:

  1. A LaTeX table (2d_comparison_table.tex) — mean ± std across seeds for each
     (experiment, loss) pair; "--" for losses not yet run (e.g. weighted).

  2. Convergence figures (PDF + PNG) showing iteration curves grouped by loss type
     for the following metric panels:
       - map_errors    : map_err_fwd + map_err_bwd  (2 subplots)
       - sinkhorn      : sinkhorn_div
       - wasserstein   : W2_nu, W1_nu, W2_mu, W1_mu  (compound)
       - distribution  : mmd_nu, energy_dist_nu

  Each figure has rows = metric subplots, columns = experiment variants
  (simple / medium / complex  ×  standard / perturbed).
  Lines within each panel = loss types (implicit / explicit / weighted).

  Weighted runs are OPTIONAL: if absent the curves are simply omitted and the
  table cells show "--".  Re-running the script after adding weighted runs fills
  in everything automatically.

Usage
-----
    python make_2d_loss_comparison.py
    python make_2d_loss_comparison.py --out plots/2d_comparison
    python make_2d_loss_comparison.py --entity antoine311200 --project dualwcfm-2d
    python make_2d_loss_comparison.py --no-figures      # table only
    python make_2d_loss_comparison.py --no-table        # figures only
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------------

DEFAULT_ENTITY  = "antoine311200"
DEFAULT_PROJECT = "dualwcfm-2d"

# Experiment base names (moon excluded)
BASE_EXPERIMENTS = ["2d_simple", "2d_complex"]
VARIANTS         = [False, True]   # perturbed flag

# All loss types we care about — weighted may be absent
LOSSES = ["implicit", "explicit", "weighted"]

# Pretty display labels
EXP_LABEL: Dict[str, str] = {
    "2d_simple":            "Simple",
    "2d_simple_perturbed":  "Simple (pert.)",
    "2d_complex":           "Complex",
    "2d_complex_perturbed": "Complex (pert.)",
}

LOSS_LABEL: Dict[str, str] = {
    "implicit": "Implicit",
    "explicit": "Explicit",
    "weighted": "Weighted",
}

# Okabe-Ito accessible palette for loss types
LOSS_COLOR: Dict[str, str] = {
    "implicit": "#0072B2",   # blue
    "explicit": "#D55E00",   # vermillion
    "weighted": "#009E73",   # green
}

LOSS_LINESTYLE: Dict[str, str] = {
    "implicit": "-",
    "explicit": "--",
    "weighted": "-.",
}

LOSS_MARKER: Dict[str, str] = {
    "implicit": "o",
    "explicit": "s",
    "weighted": "^",
}

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# Groups used for separate convergence figures
METRIC_GROUPS: Dict[str, List[str]] = {
    "map_errors":   ["map_err_fwd", "map_err_bwd"],
    "sinkhorn":     ["sinkhorn_div"],
    "wasserstein":  ["W2_nu", "W1_nu", "W2_mu", "W1_mu"],
    "distribution": ["mmd_nu", "energy_dist_nu"],
}

# Metrics included in the LaTeX table
TABLE_METRICS = [
    "map_err_fwd", "map_err_bwd",
    "sinkhorn_div",
    "W2_nu", "W1_nu",
    "mmd_nu",
]

LOWER_IS_BETTER = {
    "W1_nu", "W2_nu", "W1_mu", "W2_mu",
    "mmd_nu", "energy_dist_nu", "sinkhorn_div",
    "map_err_fwd", "map_err_bwd",
}

# Human-readable axis labels (matplotlib mathtext)
MLABEL: Dict[str, str] = {
    "W1_nu":          r"$\mathrm{SW}_1(\hat\nu,\nu)$",
    "W2_nu":          r"$\mathrm{SW}_2(\hat\nu,\nu)$",
    "W1_mu":          r"$\mathrm{SW}_1(\hat\mu,\mu)$",
    "W2_mu":          r"$\mathrm{SW}_2(\hat\mu,\mu)$",
    "mmd_nu":         r"$\mathrm{MMD}(\hat\nu,\nu)$",
    "energy_dist_nu": r"Energy distance",
    "sinkhorn_div":   r"Sinkhorn div.",
    "map_err_fwd":    r"Map err. (fwd)",
    "map_err_bwd":    r"Map err. (bwd)",
}

# LaTeX-safe labels for the table
MLABEL_TEX: Dict[str, str] = {
    "W1_nu":          r"$\mathrm{SW}_1(\hat\nu,\nu)$",
    "W2_nu":          r"$\mathrm{SW}_2(\hat\nu,\nu)$",
    "W1_mu":          r"$\mathrm{SW}_1(\hat\mu,\mu)$",
    "W2_mu":          r"$\mathrm{SW}_2(\hat\mu,\mu)$",
    "mmd_nu":         r"MMD",
    "energy_dist_nu": r"Energy dist.",
    "sinkhorn_div":   r"Sinkhorn div.",
    "map_err_fwd":    r"Map err.\ (fwd)",
    "map_err_bwd":    r"Map err.\ (bwd)",
}

# Figure sizes (targeting NeurIPS / ICML column widths)
COL_2W = 6.75   # double-column width in inches
DPI    = 250


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

_PALETTE = [
    "#0072B2", "#D55E00", "#009E73",
    "#CC79A7", "#56B4E9", "#E69F00", "#555555",
]


def _setup_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "text.usetex":                    False,
        "font.family":                    "sans-serif",
        "font.sans-serif":                ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "mathtext.fontset":               "cm",
        "figure.facecolor":               "white",
        "axes.facecolor":                 "#F8F9FA",
        "savefig.facecolor":              "white",
        "savefig.bbox":                   "tight",
        "savefig.pad_inches":             0.06,
        "figure.dpi":                     130,
        "axes.spines.top":                False,
        "axes.spines.right":              False,
        "axes.spines.left":               True,
        "axes.spines.bottom":             True,
        "axes.linewidth":                 0.8,
        "axes.labelsize":                 9,
        "axes.titlesize":                 9.5,
        "axes.titleweight":               "semibold",
        "axes.titlepad":                  6,
        "axes.prop_cycle":                mpl.cycler("color", _PALETTE),
        "axes.grid":                      True,
        "axes.grid.axis":                 "y",
        "grid.linewidth":                 0.35,
        "grid.alpha":                     0.50,
        "grid.color":                     "#CACACA",
        "xtick.labelsize":                7.5,
        "ytick.labelsize":                7.5,
        "xtick.direction":                "out",
        "ytick.direction":                "out",
        "xtick.major.size":               3.5,
        "ytick.major.size":               3.5,
        "lines.linewidth":                2.0,
        "lines.markersize":               4.0,
        "lines.solid_capstyle":           "round",
        "lines.solid_joinstyle":          "round",
        "legend.fontsize":                7.5,
        "legend.framealpha":              0.93,
        "legend.edgecolor":               "#CCCCCC",
        "legend.handlelength":            2.2,
        "legend.labelspacing":            0.40,
        "legend.borderpad":               0.50,
        "legend.fancybox":                False,
        "figure.constrained_layout.use":  False,
    })


# ---------------------------------------------------------------------------
# W&B data loading
# ---------------------------------------------------------------------------

def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def fetch_runs(entity: str, project: str) -> List[Dict]:
    """Fetch and parse all 2-D (non-moon) runs from the W&B project."""
    try:
        import wandb
    except ImportError:
        sys.exit("wandb is required:  pip install wandb")

    api  = wandb.Api()
    path = f"{entity}/{project}"
    print(f"Fetching runs from {path} ...", flush=True)

    try:
        raw_runs = list(api.runs(path))
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return []

    print(f"  {len(raw_runs)} run(s) fetched from W&B.", flush=True)

    processed = []
    for run in raw_runs:
        cfg      = dict(run.config)
        exp_name = cfg.get("experiment") or cfg.get("base_experiment") or ""

        # Keep only the 6 target experiment variants (no moon, no other modes)
        if "moon" in exp_name:
            continue
        if not any(exp_name == e or exp_name == f"{e}_perturbed"
                   for e in BASE_EXPERIMENTS):
            continue

        loss = cfg.get("loss", "unknown")

        # Pull full history (up to 2 000 rows — enough for 200 iterations)
        try:
            rows = run.history(samples=2000, x_axis="_step", pandas=False)
        except Exception:
            rows = []

        if not rows:
            print(f"  SKIP (empty history): {run.name!r}  exp={exp_name}  loss={loss}")
            continue

        # Build per-metric step -> value dicts
        raw: Dict[str, Dict[int, float]] = defaultdict(dict)
        for row in rows:
            step = int(row.get("_step", row.get("iteration", 0)))
            for k, v in row.items():
                if k.startswith("_"):
                    continue
                fv = _safe_float(v)
                if fv is not None:
                    raw[k][step] = fv

        # Convert to sorted (steps, values) arrays
        history: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for m, sv in raw.items():
            if not sv:
                continue
            steps = np.array(sorted(sv), dtype=int)
            vals  = np.array([sv[s] for s in steps], dtype=float)
            history[m] = (steps, vals)

        if not history:
            continue

        final    = {m: float(v[-1])  for m, (_, v) in history.items() if len(v)}
        n_steps  = max((int(s[-1])   for s, _ in history.values()), default=0)

        processed.append({
            "name":       run.name or run.id,
            "id":         run.id,
            "state":      run.state,
            "config":     cfg,
            "experiment": exp_name,
            "loss":       loss,
            "history":    history,
            "final":      final,
            "n_steps":    n_steps,
        })
        print(f"  Loaded: {run.name!r:<40}  exp={exp_name:<25}  loss={loss:<10}  iters={n_steps}")

    return processed


def group_runs(runs: List[Dict]) -> Dict[Tuple[str, str], List[Dict]]:
    """Group runs by (experiment_name, loss_type)."""
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for rd in runs:
        groups[(rd["experiment"], rd["loss"])].append(rd)
    return dict(groups)


# ---------------------------------------------------------------------------
# Multi-seed history aggregation
# ---------------------------------------------------------------------------

def _aggregate_histories(
    run_list: List[Dict],
    metric: str,
    n_points: int = 150,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (grid_steps, mean, std) interpolated to a uniform grid, or None."""
    curves = []
    for rd in run_list:
        hist = rd["history"]
        if metric not in hist:
            continue
        steps, vals = hist[metric]
        if len(steps) < 2:
            continue
        curves.append((steps.astype(float), vals))

    if not curves:
        return None

    max_step = min(float(s[-1]) for s, _ in curves)
    if max_step <= 0:
        return None

    grid = np.linspace(0, max_step, n_points)
    interp = np.stack(
        [np.interp(grid, s, v) for s, v in curves], axis=0
    )
    return grid, interp.mean(axis=0), interp.std(axis=0)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def make_latex_table(
    groups:   Dict[Tuple[str, str], List[Dict]],
    metrics:  List[str],
    out_path: Path,
) -> None:
    """Write a booktabs table: rows = experiment variants, cols = loss × metric."""

    exp_names = [
        f"{base}{'_perturbed' if p else ''}"
        for base in BASE_EXPERIMENTS
        for p in VARIANTS
    ]

    # Aggregate: data[exp][loss][metric] = (mean, std) | None
    data: Dict = {}
    for exp in exp_names:
        data[exp] = {}
        for loss in LOSSES:
            data[exp][loss] = {}
            run_list = groups.get((exp, loss), [])
            for metric in metrics:
                vals = [rd["final"][metric] for rd in run_list
                        if metric in rd["final"]]
                if vals:
                    data[exp][loss][metric] = (float(np.mean(vals)), float(np.std(vals)))
                else:
                    data[exp][loss][metric] = None

    def is_best(exp: str, loss: str, metric: str) -> bool:
        cell = data[exp][loss][metric]
        if cell is None:
            return False
        all_vals = [
            data[exp][l][metric][0]
            for l in LOSSES
            if data[exp][l][metric] is not None
        ]
        if len(all_vals) < 2:
            return False
        best = min(all_vals) if metric in LOWER_IS_BETTER else max(all_vals)
        return abs(cell[0] - best) < 1e-9

    def fmt_cell(cell: Optional[Tuple[float, float]], bold: bool) -> str:
        if cell is None:
            return r"\text{--}"
        v, s = cell
        num = f"{v:.3f}"
        if s > 1e-4:
            num += rf" \pm {s:.3f}"
        return rf"\mathbf{{{num}}}" if bold else num

    n_m = len(metrics)
    n_l = len(LOSSES)
    col_spec = "l" + ("r" * n_l) * n_m

    lines: List[str] = []
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Final metrics for 2-D generative transport experiments across three "
         r"loss functions (Implicit / Explicit / Weighted). "
         r"\textbf{Bold}: best value per cell.  \textrm{--}: runs not yet completed.}"),
        r"\label{tab:2d_loss_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # ── Header row 1: metric spans ──────────────────────────────────────────
    span_parts = " & ".join(
        rf"\multicolumn{{{n_l}}}{{c}}{{{MLABEL_TEX.get(m, m)}}}"
        for m in metrics
    )
    lines.append(rf" & {span_parts} \\")

    # cmidrules
    cmidrules = []
    for i in range(n_m):
        c0 = 2 + i * n_l
        c1 = c0 + n_l - 1
        cmidrules.append(rf"\cmidrule(lr){{{c0}-{c1}}}")
    lines.append("  ".join(cmidrules))

    # ── Header row 2: loss sub-headers ──────────────────────────────────────
    loss_hdr = " & ".join([LOSS_LABEL[l] for l in LOSSES] * n_m)
    lines.append(rf"Experiment & {loss_hdr} \\")
    lines.append(r"\midrule")

    # ── Data rows ───────────────────────────────────────────────────────────
    prev_base = None
    for exp in exp_names:
        base = exp.replace("_perturbed", "")
        if prev_base is not None and base != prev_base:
            lines.append(r"\addlinespace[3pt]")
        prev_base = base

        cells = [EXP_LABEL.get(exp, exp)]
        for metric in metrics:
            for loss in LOSSES:
                cell = data[exp][loss][metric]
                bold = is_best(exp, loss, metric)
                cells.append(f"${fmt_cell(cell, bold)}$")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",   # end resizebox
        r"\end{table}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  LaTeX table  ->  {out_path}")


# ---------------------------------------------------------------------------
# Convergence figures
# ---------------------------------------------------------------------------

def _all_exp_names() -> List[str]:
    return [
        f"{base}{'_perturbed' if p else ''}"
        for base in BASE_EXPERIMENTS
        for p in VARIANTS
    ]


def make_convergence_figures(
    groups:  Dict[Tuple[str, str], List[Dict]],
    out_dir: Path,
) -> None:
    """
    One PDF/PNG per metric group.

    Layout:
      rows = metrics within the group
      cols = experiment variants  (simple | simple (pert.) | medium | … | complex (pert.))
      lines = loss types
    """
    import matplotlib.pyplot as plt

    _setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_names = _all_exp_names()
    n_cols    = len(exp_names)

    for group_name, metrics in METRIC_GROUPS.items():
        n_rows  = len(metrics)
        fig_w   = max(COL_2W, n_cols * 2.1)
        fig_h   = n_rows * 2.3

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            squeeze=False,
        )

        for c, exp in enumerate(exp_names):
            for r, metric in enumerate(metrics):
                ax = axes[r][c]

                plotted_losses = []
                for loss in LOSSES:
                    run_list = groups.get((exp, loss), [])
                    agg = _aggregate_histories(run_list, metric)
                    if agg is None:
                        continue
                    steps, mean, std = agg
                    color = LOSS_COLOR[loss]
                    ls    = LOSS_LINESTYLE[loss]
                    label = LOSS_LABEL[loss]

                    # Add small marker every N steps for line differentiation in BW print
                    n = len(steps)
                    markevery = max(1, n // 8)
                    ax.plot(steps, mean,
                            color=color, ls=ls, lw=2.0,
                            marker=LOSS_MARKER[loss], markevery=markevery,
                            markersize=3.5, label=label, zorder=3)
                    if len(run_list) > 1 and std.max() > 1e-9:
                        ax.fill_between(steps, mean - std, mean + std,
                                        color=color, alpha=0.13, zorder=2)
                    plotted_losses.append(loss)

                if not plotted_losses:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, color="#999999", fontsize=7.5,
                            style="italic")

                # Titles and axis labels
                if r == 0:
                    ax.set_title(EXP_LABEL.get(exp, exp), fontsize=8.5, pad=5)
                if c == 0:
                    ax.set_ylabel(MLABEL.get(metric, metric), fontsize=8.5)
                else:
                    ax.set_yticklabels([])
                if r == n_rows - 1:
                    ax.set_xlabel("Iteration", fontsize=8)
                ax.tick_params(labelsize=7)

        # Shared legend below the figure
        handles, lbls = axes[0][0].get_legend_handles_labels()
        # Collect from all axes in case first had no data
        for ax_row in axes:
            for ax in ax_row:
                h, l = ax.get_legend_handles_labels()
                for hh, ll in zip(h, l):
                    if ll not in lbls:
                        handles.append(hh)
                        lbls.append(ll)
        if handles:
            fig.legend(
                handles, lbls,
                loc="lower center",
                ncol=len(LOSSES),
                bbox_to_anchor=(0.5, -0.03),
                fontsize=8.5,
                framealpha=0.93,
                handlelength=2.4,
            )

        title_map = {
            "map_errors":   "Map errors (fwd & bwd)",
            "sinkhorn":     "Sinkhorn divergence",
            "wasserstein":  "Wasserstein distances (source & target)",
            "distribution": "Distribution distances (MMD & energy)",
        }
        fig.suptitle(
            title_map.get(group_name, group_name.replace("_", " ").title()),
            fontsize=11, y=1.02, fontweight="semibold",
        )
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        stem = out_dir / f"2d_convergence_{group_name}"
        fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
        fig.savefig(str(stem) + ".png", bbox_inches="tight", dpi=DPI)
        plt.close(fig)
        print(f"  Convergence  ->  {stem}.pdf / .png")


# ---------------------------------------------------------------------------
# Summary bar-chart figure (final values)
# ---------------------------------------------------------------------------

def make_summary_bar_figures(
    groups:  Dict[Tuple[str, str], List[Dict]],
    metrics: List[str],
    out_dir: Path,
) -> None:
    """
    For each metric: one figure with a bar per loss type, grouped by experiment.
    Rows = 2 (standard / perturbed)  ×  cols = 3 (simple / medium / complex).
    """
    import matplotlib.pyplot as plt

    _setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    x_pos = np.arange(len(LOSSES))

    for metric in metrics:
        fig, axes = plt.subplots(
            2, len(BASE_EXPERIMENTS),
            figsize=(len(BASE_EXPERIMENTS) * 2.2, 4.4),
            squeeze=False,
        )

        for c, base in enumerate(BASE_EXPERIMENTS):
            for r, perturbed in enumerate(VARIANTS):
                exp = f"{base}{'_perturbed' if perturbed else ''}"
                ax  = axes[r][c]

                heights = []
                errors  = []
                colors  = []
                for loss in LOSSES:
                    run_list = groups.get((exp, loss), [])
                    vals = [rd["final"][metric]
                            for rd in run_list if metric in rd["final"]]
                    if vals:
                        heights.append(float(np.mean(vals)))
                        errors.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
                    else:
                        heights.append(np.nan)
                        errors.append(0.0)
                    colors.append(LOSS_COLOR[loss])

                valid = [not np.isnan(h) for h in heights]
                ax.bar(
                    x_pos[valid], [h for h, v in zip(heights, valid) if v],
                    color=[colors[i] for i, v in enumerate(valid) if v],
                    width=0.6, zorder=3,
                )
                ax.errorbar(
                    x_pos[valid],
                    [h for h, v in zip(heights, valid) if v],
                    yerr=[e for e, v in zip(errors, valid) if v],
                    fmt="none", color="black", capsize=3, lw=1.0, zorder=4,
                )

                ax.set_xticks(x_pos)
                ax.set_xticklabels(
                    [LOSS_LABEL[l] for l in LOSSES],
                    fontsize=7.5, rotation=20, ha="right",
                )
                if r == 0:
                    ax.set_title(base.replace("2d_", "").capitalize(),
                                 fontsize=9, pad=5)
                if c == 0:
                    plab = "Perturbed" if perturbed else "Standard"
                    ax.set_ylabel(f"{plab}\n{MLABEL.get(metric, metric)}",
                                  fontsize=8)
                ax.tick_params(axis="y", labelsize=7)

        fig.suptitle(
            f"Final {MLABEL.get(metric, metric)} — by experiment & loss",
            fontsize=10, y=1.02, fontweight="semibold",
        )
        fig.tight_layout()

        stem = out_dir / f"2d_bar_{metric}"
        fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
        fig.savefig(str(stem) + ".png", bbox_inches="tight", dpi=DPI)
        plt.close(fig)
        print(f"  Bar chart    ->  {stem}.pdf / .png")


# ---------------------------------------------------------------------------
# Combined overview figure  (one panel per experiment, all key metrics)
# ---------------------------------------------------------------------------

OVERVIEW_METRICS = ["map_err_fwd", "map_err_bwd", "sinkhorn_div", "W2_nu", "mmd_nu"]


def make_overview_figure(
    groups:  Dict[Tuple[str, str], List[Dict]],
    out_dir: Path,
) -> None:
    """
    A compact overview figure suitable for the paper body:
      rows = 2 (standard / perturbed)
      cols = 3 (simple / medium / complex)
      Within each panel: 5 sub-rows of metric strips sharing the x-axis.

    Uses a tight twinx / broken-axis style so all metrics fit in one panel.
    Each metric is shown as a small convergence strip.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    _setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_metrics = len(OVERVIEW_METRICS)
    n_rows    = len(VARIANTS)
    n_cols    = len(BASE_EXPERIMENTS)

    # Each "panel" is a vertical stack of n_metrics mini-axes
    fig_w = n_cols * 2.4
    fig_h = n_rows * n_metrics * 1.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.45, wspace=0.35,
    )

    for r, perturbed in enumerate(VARIANTS):
        for c, base in enumerate(BASE_EXPERIMENTS):
            exp  = f"{base}{'_perturbed' if perturbed else ''}"
            inner = gridspec.GridSpecFromSubplotSpec(
                n_metrics, 1,
                subplot_spec=outer[r, c],
                hspace=0.05,
            )

            for m_idx, metric in enumerate(OVERVIEW_METRICS):
                ax = fig.add_subplot(inner[m_idx])

                plotted = False
                for loss in LOSSES:
                    run_list = groups.get((exp, loss), [])
                    agg = _aggregate_histories(run_list, metric)
                    if agg is None:
                        continue
                    steps, mean, std = agg
                    ax.plot(steps, mean,
                            color=LOSS_COLOR[loss],
                            ls=LOSS_LINESTYLE[loss],
                            lw=1.6, label=LOSS_LABEL[loss], zorder=3)
                    if len(run_list) > 1 and std.max() > 1e-9:
                        ax.fill_between(steps, mean - std, mean + std,
                                        color=LOSS_COLOR[loss], alpha=0.12, zorder=2)
                    plotted = True

                if not plotted:
                    ax.text(0.5, 0.5, "–", ha="center", va="center",
                            transform=ax.transAxes, color="#AAAAAA", fontsize=7)

                # Minimal decoration
                ax.set_ylabel(MLABEL.get(metric, metric), fontsize=6.5,
                              labelpad=2, rotation=0, ha="right", va="center")
                ax.yaxis.set_label_coords(-0.22, 0.5)
                ax.tick_params(labelsize=6, pad=1)
                if m_idx < n_metrics - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Iter.", fontsize=7)

                if m_idx == 0:
                    plab = " (pert.)" if perturbed else ""
                    title = base.replace("2d_", "").capitalize() + plab
                    ax.set_title(title, fontsize=8.5, pad=4)

    # Shared legend
    handles, lbls = [], []
    for loss in LOSSES:
        import matplotlib.lines as mlines
        h = mlines.Line2D([], [],
                          color=LOSS_COLOR[loss],
                          ls=LOSS_LINESTYLE[loss],
                          lw=2.0, label=LOSS_LABEL[loss])
        handles.append(h)
        lbls.append(LOSS_LABEL[loss])
    fig.legend(
        handles, lbls,
        loc="lower center",
        ncol=len(LOSSES),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8.5,
        framealpha=0.93,
    )
    fig.suptitle(
        "2-D experiments — convergence overview by loss type",
        fontsize=10.5, y=1.01, fontweight="semibold",
    )

    stem = out_dir / "2d_overview"
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"  Overview     ->  {stem}.pdf / .png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--entity",  default=DEFAULT_ENTITY,
                        help=f"W&B entity (default: {DEFAULT_ENTITY})")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help=f"W&B project (default: {DEFAULT_PROJECT})")
    parser.add_argument("--out", default="plots/2d_comparison",
                        metavar="DIR",
                        help="Output directory (default: plots/2d_comparison)")
    parser.add_argument("--no-figures", action="store_true", dest="no_figures",
                        help="Skip all figure generation (table only).")
    parser.add_argument("--no-table",   action="store_true", dest="no_table",
                        help="Skip LaTeX table generation (figures only).")
    parser.add_argument("--no-bars",    action="store_true", dest="no_bars",
                        help="Skip bar-chart summary figures.")
    parser.add_argument("--no-overview", action="store_true", dest="no_overview",
                        help="Skip the compact overview figure.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Fetch ────────────────────────────────────────────────────────────────
    runs   = fetch_runs(args.entity, args.project)
    groups = group_runs(runs)

    if not runs:
        print("\nNo matching runs found. Exiting.")
        print("(Check that --entity and --project are correct, and that you are logged in to W&B.)")
        return

    print(f"\nRun groups loaded ({len(groups)} unique (experiment, loss) pairs):")
    for (exp, loss), run_list in sorted(groups.items()):
        print(f"  {exp:<30}  loss={loss:<12}  n_runs={len(run_list)}")

    # ── LaTeX table ──────────────────────────────────────────────────────────
    if not args.no_table:
        print("\nGenerating LaTeX table ...")
        make_latex_table(groups, TABLE_METRICS, out_dir / "2d_comparison_table.tex")

    if args.no_figures:
        print("\nDone (figures skipped).")
        return

    # ── Convergence figures ──────────────────────────────────────────────────
    print("\nGenerating convergence figures ...")
    make_convergence_figures(groups, out_dir)

    # ── Summary bar charts ───────────────────────────────────────────────────
    if not args.no_bars:
        print("\nGenerating bar-chart summaries ...")
        make_summary_bar_figures(groups, TABLE_METRICS, out_dir)

    # ── Compact overview figure ──────────────────────────────────────────────
    if not args.no_overview:
        print("\nGenerating overview figure ...")
        make_overview_figure(groups, out_dir)

    print(f"\nAll outputs written to  {out_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
