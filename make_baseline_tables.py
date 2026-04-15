#!/usr/bin/env python
"""
make_baseline_tables.py
=======================
Read every  results_baseline/{exp}/baselines/{baseline}/result.csv
and produce:

  plots/baseline_tables/
    per_experiment/
      {exp}_table.tex          -- per-experiment booktabs table (all baselines x all metrics)
    full_comparison.csv        -- wide-format CSV (all experiments x baselines x metrics)
    full_comparison.tex        -- one big booktabs table grouped by experiment
    full_comparison_transposed.tex  -- metrics as row groups, experiments as columns

Usage
-----
    python make_baseline_tables.py
    python make_baseline_tables.py --results results_baseline --out plots/baseline_tables
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR          = Path(r"C:\Antoine\Study\Thèse\Barclays\Codes\schrodinger-dualgan\dualwcfm\results_baseline")
RESULTS_DIR_COUPLING = Path(r"C:\Antoine\Study\Thèse\Barclays\Codes\schrodinger-dualgan\dualwcfm\results_baseline_coupling")
OUT_DIR              = Path(r"C:\Antoine\Study\Thèse\Barclays\Codes\schrodinger-dualgan\dualwcfm\plots\baseline_tables")

# Directories for the three variants of our method.
# Each contains {exp}/metrics.csv with one row per iteration.
# The best iteration (lowest W2_nu) is used.
OUR_WEIGHT_DIR = Path(r"C:\Antoine\Study\Thèse\Barclays\Codes\schrodinger-dualgan\dualwcfm\results_final_2d")
OUR_IMPL_DIR   = Path(r"C:\Antoine\Study\Thèse\Barclays\Codes\schrodinger-dualgan\dualwcfm\results_final_2d_impl")
OUR_EXPL_DIR   = Path(r"C:\Antoine\Study\Thèse\Barclays\Codes\schrodinger-dualgan\dualwcfm\results_final_2d_expl")

# Canonical metric order and display labels
METRICS: List[str] = [
    "W2_nu", "W1_nu",
    "W2_mu", "W1_mu",
    "map_err_fwd",
    "sinkhorn_div",
    "mmd_nu",
    "energy_dist_nu",
]

METRIC_LABEL: Dict[str, str] = {
    "W1_mu":          r"$\mathrm{SW}_1(\hat\mu)$",
    "W2_mu":          r"$\mathrm{SW}_2(\hat\mu)$",
    "W1_nu":          r"$\mathrm{SW}_1(\hat\nu)$",
    "W2_nu":          r"$\mathrm{SW}_2(\hat\nu)$",
    "mmd_nu":         r"MMD",
    "energy_dist_nu": r"Energy dist.",
    "sinkhorn_div":   r"Sinkhorn div.",
    "map_err_fwd":    r"Map err.\ (fwd)",
}

# Metric groups used for the column-span headers
METRIC_GROUPS: List[Tuple[str, List[str]]] = [
    ("Target marginal",  ["W2_nu", "W1_nu"]),
    ("Source marginal",  ["W2_mu", "W1_mu"]),
    ("Map quality",      ["map_err_fwd"]),
    ("Transport quality",["sinkhorn_div", "mmd_nu", "energy_dist_nu"]),
]

# All metrics are lower-is-better
LOWER_IS_BETTER = set(METRICS)

# Preferred display order for baselines  (ours goes last — conventional for comparison tables)
# Indep. baselines first, then coupled variants, then our rows.
BASELINE_ORDER = [
    "cfm", "otcfm", "lightsb", "dsb", "genot", "metaot",
    "cfm_coupled", "otcfm_coupled", "lightsb_coupled", "dsb_coupled", "genot_coupled",
    "pretrained", "our_weight", "our_impl", "our_expl",
]

BASELINE_LABEL: Dict[str, str] = {
    "cfm":             "CFM (indep.)",
    "otcfm":           "OT-CFM (indep.)",
    "lightsb":         "LightSB (indep.)",
    "dsb":             "DSB (indep.)",
    "genot":           "GENOT (indep.)",
    "metaot":          "MetaOT (indep.)",
    "cfm_coupled":     "CFM (coupl.)",
    "otcfm_coupled":   "OT-CFM (coupl.)",
    "lightsb_coupled": "LightSB (coupl.)",
    "dsb_coupled":     "DSB (coupl.)",
    "genot_coupled":   "GENOT (coupl.)",
    "pretrained":      r"\textit{Pretrained}",
    "our_weight":      r"\textbf{Ours (weight.)}",
    "our_impl":        r"\textbf{Ours (impl.)}",
    "our_expl":        r"\textbf{Ours (exp.)}",
}

# Rows that belong to our method — excluded from the bold-best competition
# and visually separated from the external baselines with a dashed rule.
_OUR_ROWS = {"pretrained", "our_weight", "our_impl", "our_expl"}

# Coupled baseline names — visually separated from indep. rows with a thin rule.
_COUPLED_ROWS = {"cfm_coupled", "otcfm_coupled", "lightsb_coupled", "dsb_coupled", "genot_coupled"}

# Sentinel: stored in the data dict to signal a "to be filled" cell.
_TBF = "TBF"

# Step-0 metrics (pretrained model applied without any adaptation).
OUR_PRETRAINED_RESULTS: Dict[str, Dict[str, float]] = {
    "2d_simple": {
        "W2_nu":          0.02823,
        "W1_nu":          0.02187,
        "W2_mu":          0.02647,
        "W1_mu":          0.01890,
        "map_err_fwd":    7.50311,
        "sinkhorn_div":   195.572,
        "mmd_nu":         1.01438,
        "energy_dist_nu": 26.2408,
    },
    "2d_complex": {
        "W2_nu":          0.08902,
        "W1_nu":          0.03873,
        "W2_mu":          0.05958,
        "W1_mu":          0.03303,
        "map_err_fwd":    1.14249,
        "sinkhorn_div":   49.1493,
        "mmd_nu":         0.41064,
        "energy_dist_nu": 4.03350,
    },
}

EXP_LABEL: Dict[str, str] = {
    "2d_simple":            r"Simple",
    "2d_simple_perturbed":  r"Simple (pert.)",
    "2d_medium":            r"Medium",
    "2d_medium_perturbed":  r"Medium (pert.)",
    "2d_complex":           r"Complex",
    "2d_complex_perturbed": r"Complex (pert.)",
    "2d_moon":              r"Moon",
    "2d_moon_perturbed":    r"Moon (pert.)",
}

# Preferred experiment display order
EXP_ORDER = [
    "2d_simple", "2d_simple_perturbed",
    "2d_medium", "2d_medium_perturbed",
    "2d_complex", "2d_complex_perturbed",
    "2d_moon",   "2d_moon_perturbed",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Row type: {metric: float | None, "baseline": str, "experiment": str}
Row = Dict[str, object]


def _safe_float(v: str) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def load_our_results_wandb(
    loss_type: str,
    entity:    str = "antoine311200",
    project:   str = "dualwcfm-2d",
) -> Dict[str, Dict[str, Optional[float]]]:
    """Load our-method results from W&B for a given loss variant.

    For each experiment, picks the most-recently-created *finished* run whose
    ``config.loss == loss_type``, then reads all metrics from its run summary.
    Using the summary avoids the alternating-metric problem in local CSVs
    (map_err_fwd / map_err_bwd are logged on alternate steps, so local CSVs
    have gaps; the W&B summary accumulates the last logged value per key,
    giving a complete row).

    Parameters
    ----------
    loss_type : one of "weighted", "implicit", "explicit"
    entity    : W&B entity (default: "antoine311200")
    project   : W&B project (default: "dualwcfm-2d")
    """
    try:
        import wandb
        api = wandb.Api()
    except ImportError:
        print("  [warn] wandb not installed — skipping W&B result loading.")
        return {}

    out: Dict[str, Dict[str, Optional[float]]] = {}

    runs = api.runs(
        f"{entity}/{project}",
        filters={"config.loss": loss_type, "state": "finished"},
    )

    # Group by experiment, keep only the most recently created run per experiment
    latest: Dict[str, object] = {}
    for run in runs:
        exp = run.config.get("experiment")
        if exp is None:
            continue
        prev = latest.get(exp)
        if prev is None or run.created_at > prev.created_at:
            latest[exp] = run

    for exp, run in sorted(latest.items()):
        summary = dict(run.summary)
        out[exp] = {m: _safe_float(str(summary.get(m, ""))) for m in METRICS}

    return out


def load_all_results(results_dir: Path, only_coupled: bool = False) -> List[Row]:
    """Scan results_dir/{exp}/baselines/{baseline}/result.csv and return all rows.

    If *only_coupled* is True, only baseline directories whose name ends with
    ``_coupled`` are loaded (used for the coupling results directory).
    """
    rows: List[Row] = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        baselines_dir = exp_dir / "baselines"
        if not baselines_dir.is_dir():
            continue
        for bl_dir in sorted(baselines_dir.iterdir()):
            if not bl_dir.is_dir():
                continue
            baseline_name = bl_dir.name
            if only_coupled and not baseline_name.endswith("_coupled"):
                continue
            csv_path = bl_dir / "result.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for raw in reader:
                    row: Row = {"experiment": exp_name, "baseline": baseline_name}
                    for m in METRICS:
                        row[m] = _safe_float(raw.get(m, ""))
                    rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], decimals: int = 4) -> str:
    """Format a metric value for CSV output."""
    if v is None:
        return "--"
    if v is _TBF or v == _TBF:
        return _TBF
    return f"{v:.{decimals}f}"


def _fmt_bold(v, bold: bool, decimals: int = 4) -> str:
    """Format a metric value for LaTeX output.  TBF sentinel renders as italic placeholder."""
    if v is _TBF or v == _TBF:
        return r"\textit{TBF}"
    if v is None:
        return "--"
    s = f"{v:.{decimals}f}"
    if bold:
        return r"\textbf{" + s + "}"
    return s


def _best_per_metric(
    metric_vals: Dict[str, Optional[float]],
) -> Optional[str]:
    """Return the baseline key with the smallest finite value, or None.
    TBF sentinels, None values, and our own rows are excluded from the competition."""
    best_bl, best_v = None, float("inf")
    for bl, v in metric_vals.items():
        if bl in _OUR_ROWS:
            continue  # our method rows compete separately
        if v is None or v is _TBF or v == _TBF:
            continue
        try:
            fv = float(v)
            if fv < best_v:
                best_v, best_bl = fv, bl
        except (TypeError, ValueError):
            pass
    return best_bl


def _inject_our_rows(
    rows:        List[Row],
    experiments: List[str],
    our_weight:  Dict[str, Dict[str, Optional[float]]],
    our_impl:    Dict[str, Dict[str, Optional[float]]],
    our_expl:    Dict[str, Dict[str, Optional[float]]],
) -> List[Row]:
    """Append pretrained + three Ours rows per experiment.

    - pretrained  : step-0 state (hardcoded from OUR_PRETRAINED_RESULTS)
    - our_weight  : loaded from results_final_2d       (best iter by W2_nu)
    - our_impl    : loaded from results_final_2d_impl  (best iter by W2_nu)
    - our_expl    : loaded from results_final_2d_expl  (best iter by W2_nu)
    """
    extra: List[Row] = []
    for exp in experiments:
        # Pretrained row
        pretrain_row: Row = {"experiment": exp, "baseline": "pretrained"}
        known_pre = OUR_PRETRAINED_RESULTS.get(exp, {})
        for m in METRICS:
            pretrain_row[m] = known_pre[m] if m in known_pre else _TBF
        extra.append(pretrain_row)

        # Three Ours variants
        for bl_key, result_dict in [
            ("our_weight", our_weight),
            ("our_impl",   our_impl),
            ("our_expl",   our_expl),
        ]:
            row: Row = {"experiment": exp, "baseline": bl_key}
            known = result_dict.get(exp, {})
            for m in METRICS:
                v = known.get(m)
                row[m] = v if v is not None else _TBF
            extra.append(row)

    return rows + extra


def _ordered_baselines(available: List[str]) -> List[str]:
    ordered = [b for b in BASELINE_ORDER if b in available]
    for b in sorted(available):
        if b not in ordered:
            ordered.append(b)
    return ordered


def _ordered_experiments(available: List[str]) -> List[str]:
    ordered = [e for e in EXP_ORDER if e in available]
    for e in sorted(available):
        if e not in ordered:
            ordered.append(e)
    return ordered


# ---------------------------------------------------------------------------
# Per-experiment LaTeX tables
# ---------------------------------------------------------------------------

def make_per_experiment_tables(
    rows: List[Row],
    out_dir: Path,
) -> None:
    """One .tex file per experiment: baselines as rows, metrics as columns."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group data by experiment
    by_exp: Dict[str, Dict[str, Row]] = {}
    for r in rows:
        exp = str(r["experiment"])
        bl  = str(r["baseline"])
        by_exp.setdefault(exp, {})[bl] = r

    for exp in _ordered_experiments(list(by_exp.keys())):
        bl_data = by_exp[exp]
        baselines = _ordered_baselines(list(bl_data.keys()))

        # Determine which metrics are present for this experiment
        present_metrics = [
            m for m in METRICS
            if any(bl_data[bl].get(m) is not None for bl in baselines)
        ]

        # Find best per metric
        best: Dict[str, Optional[str]] = {}
        for m in present_metrics:
            vals = {bl: bl_data[bl].get(m) for bl in baselines}
            best[m] = _best_per_metric(vals)

        col_spec = "l" + "r" * len(present_metrics)
        header_cells = ["Method"] + [METRIC_LABEL.get(m, m) for m in present_metrics]

        lines: List[str] = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Baselines on \textbf{" + EXP_LABEL.get(exp, exp) + r"}"
            r". \textbf{Bold}: best per metric.  \textrm{--}: not available.}",
            r"\label{tab:baselines_" + exp + r"}",
            r"\resizebox{\linewidth}{!}{%",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join(header_cells) + r" \\",
            r"\midrule",
        ]

        for bl in baselines:
            row_data = bl_data[bl]
            cells = [BASELINE_LABEL.get(bl, bl)]
            for m in present_metrics:
                v    = row_data.get(m)
                bold = best[m] == bl
                cells.append(_fmt_bold(v, bold))
            lines.append(" & ".join(cells) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",  # resizebox
            r"\end{table}",
        ]

        tex = "\n".join(lines) + "\n"
        path = out_dir / f"{exp}_table.tex"
        path.write_text(tex, encoding="utf-8")
        print(f"  Per-exp table -> {path.name}")


# ---------------------------------------------------------------------------
# Full comparison CSV
# ---------------------------------------------------------------------------

def make_full_csv(rows: List[Row], out_path: Path,
                  our_weight=None, our_impl=None, our_expl=None) -> None:
    """Wide-format CSV: one row per (experiment, baseline), all metrics as columns."""
    our_weight = our_weight or {}
    our_impl   = our_impl   or {}
    our_expl   = our_expl   or {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    experiments = _ordered_experiments(sorted({str(r["experiment"]) for r in rows}))
    rows_with_our = _inject_our_rows(rows, experiments, our_weight, our_impl, our_expl)

    by_exp_bl: Dict[Tuple[str, str], Row] = {}
    for r in rows_with_our:
        by_exp_bl[(str(r["experiment"]), str(r["baseline"]))] = r

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "baseline"] + METRICS)
        for exp in experiments:
            baselines = _ordered_baselines(
                sorted({str(r["baseline"]) for r in rows_with_our if r["experiment"] == exp})
            )
            for bl in baselines:
                key = (exp, bl)
                if key not in by_exp_bl:
                    continue
                row_data = by_exp_bl[key]
                writer.writerow(
                    [exp, bl] + [_fmt(row_data.get(m)) for m in METRICS]
                )
    print(f"  Full CSV      -> {out_path.name}")


# ---------------------------------------------------------------------------
# Full comparison LaTeX  (experiments as row-groups, baselines as sub-rows)
# ---------------------------------------------------------------------------

def make_full_latex_table(rows: List[Row], out_path: Path,
                          our_weight=None, our_impl=None, our_expl=None) -> None:
    """
    Large booktabs table:
      - Column groups = metric groups
      - Row groups    = experiments (separated by \\midrule)
      - Sub-rows      = baselines  (Ours last, separated by a thin rule)
      - Bold best per (experiment, metric)  — TBF cells excluded from competition
    """
    our_weight = our_weight or {}
    our_impl   = our_impl   or {}
    our_expl   = our_expl   or {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    experiments = _ordered_experiments(sorted({str(r["experiment"]) for r in rows}))
    rows = _inject_our_rows(rows, experiments, our_weight, our_impl, our_expl)

    # Collect all baselines across the whole dataset (now includes 'our')
    all_baselines = _ordered_baselines(sorted({str(r["baseline"]) for r in rows}))

    # Index: data[(exp, bl)][metric] = float | None
    data: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    for r in rows:
        key = (str(r["experiment"]), str(r["baseline"]))
        data[key] = {m: r.get(m) for m in METRICS}

    # Determine which metrics are actually present (at least one non-None value)
    present_metrics = [
        m for m in METRICS
        if any(v is not None for d in data.values() for k, v in d.items() if k == m)
    ]

    # Best per (exp, metric)
    best: Dict[Tuple[str, str], Optional[str]] = {}
    for exp in experiments:
        for m in present_metrics:
            vals = {bl: data.get((exp, bl), {}).get(m) for bl in all_baselines}
            best[(exp, m)] = _best_per_metric(vals)

    # Build the flat list of metrics we'll actually show in the table
    shown_groups: List[Tuple[str, List[str]]] = []
    for grp_name, grp_metrics in METRIC_GROUPS:
        grp_present = [m for m in grp_metrics if m in present_metrics]
        if grp_present:
            shown_groups.append((grp_name, grp_present))
    flat_metrics = [m for _, mlist in shown_groups for m in mlist]

    n_metrics = len(flat_metrics)
    col_spec  = "ll" + "r" * n_metrics  # exp col + baseline col + metric cols

    lines: List[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full baseline comparison across all 2-D experiments and metrics. "
        r"\textbf{Bold}: best value per experiment and metric. "
        r"\textrm{--}: not available (no ground-truth map or failed run).}",
        r"\label{tab:full_baseline_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # ── Header row 1: metric group spans ────────────────────────────────────
    group_spans = []
    col_offset = 3  # 1-indexed: col1=exp, col2=baseline
    for grp_name, grp_m in shown_groups:
        n = len(grp_m)
        group_spans.append(rf"\multicolumn{{{n}}}{{c}}{{{grp_name}}}")
        col_offset += n
    lines.append(r"\multicolumn{2}{l}{} & " + " & ".join(group_spans) + r" \\")

    # cmidrules under each group
    cmidrules = []
    c = 3
    for _, grp_m in shown_groups:
        n = len(grp_m)
        cmidrules.append(rf"\cmidrule(lr){{{c}-{c + n - 1}}}")
        c += n
    lines.append("  ".join(cmidrules))

    # ── Header row 2: individual metric names ────────────────────────────────
    metric_headers = " & ".join(METRIC_LABEL.get(m, m) for m in flat_metrics)
    lines.append(r"Experiment & Method & " + metric_headers + r" \\")
    lines.append(r"\midrule")

    # ── Data rows ────────────────────────────────────────────────────────────
    for exp_idx, exp in enumerate(experiments):
        exp_baselines = _ordered_baselines(
            [bl for bl in all_baselines if (exp, bl) in data]
        )
        n_bl      = len(exp_baselines)
        exp_label = EXP_LABEL.get(exp, exp.replace("_", r"\_"))

        for bl_idx, bl in enumerate(exp_baselines):
            # Thin dashed separator before the first coupled row
            if bl in _COUPLED_ROWS and bl_idx > 0 and exp_baselines[bl_idx - 1] not in _COUPLED_ROWS:
                lines.append(r"\addlinespace[2pt]"
                              r"\cdashline{2-" + str(2 + len(flat_metrics)) + r"}[0.4pt/2pt]"
                              r"\addlinespace[2pt]")
            # Thin dashed separator before the first of our own rows
            if bl in _OUR_ROWS and bl_idx > 0 and exp_baselines[bl_idx - 1] not in _OUR_ROWS:
                lines.append(r"\addlinespace[2pt]"
                              r"\cdashline{2-" + str(2 + len(flat_metrics)) + r"}[0.4pt/2pt]"
                              r"\addlinespace[2pt]")

            # Experiment name only on first sub-row, spanning all baseline rows
            if bl_idx == 0:
                exp_cell = rf"\multirow{{{n_bl}}}{{*}}{{{exp_label}}}"
            else:
                exp_cell = ""

            bl_label  = BASELINE_LABEL.get(bl, bl)
            row_data  = data.get((exp, bl), {})
            cells     = [exp_cell, bl_label]
            for m in flat_metrics:
                v    = row_data.get(m)
                bold = best.get((exp, m)) == bl
                cells.append(_fmt_bold(v, bold))
            lines.append(" & ".join(cells) + r" \\")

        # Separator between experiment groups (but not after the last)
        if exp_idx < len(experiments) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",   # resizebox
        r"\end{table*}",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Full LaTeX    -> {out_path.name}")


# ---------------------------------------------------------------------------
# Transposed LaTeX table  (metric groups as row blocks, baselines as cols)
# Useful when there are few baselines but many experiments
# ---------------------------------------------------------------------------

def make_transposed_latex_table(rows: List[Row], out_path: Path,
                                our_weight=None, our_impl=None, our_expl=None) -> None:
    """
    Alternative layout:
      Rows = (metric-group, metric) pairs
      Cols = (experiment, baseline) pairs  — Ours last per experiment group
      Cell groups = experiments (cmidrule per experiment)
    """
    our_weight = our_weight or {}
    our_impl   = our_impl   or {}
    our_expl   = our_expl   or {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    experiments = _ordered_experiments(sorted({str(r["experiment"]) for r in rows}))
    rows = _inject_our_rows(rows, experiments, our_weight, our_impl, our_expl)
    all_baselines = _ordered_baselines(sorted({str(r["baseline"]) for r in rows}))

    data: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    for r in rows:
        key = (str(r["experiment"]), str(r["baseline"]))
        data[key] = {m: r.get(m) for m in METRICS}

    # Per-experiment baselines (some experiments lack genot, etc.)
    exp_bls: Dict[str, List[str]] = {
        exp: _ordered_baselines([bl for bl in all_baselines if (exp, bl) in data])
        for exp in experiments
    }

    # Best per (exp, metric)
    best: Dict[Tuple[str, str], Optional[str]] = {}
    for exp in experiments:
        for m in METRICS:
            vals = {bl: data.get((exp, bl), {}).get(m) for bl in exp_bls[exp]}
            best[(exp, m)] = _best_per_metric(vals)

    # Build column list: for each experiment, one col per baseline
    col_spec = "ll" + "r" * sum(len(bls) for bls in exp_bls.values())

    lines: List[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full baseline comparison (transposed view): "
        r"metric groups as row blocks, experiments as column groups. "
        r"\textbf{Bold}: best per experiment-metric pair.  \textrm{--}: n/a.}",
        r"\label{tab:full_baseline_comparison_T}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header row 1: experiment group spans
    exp_spans = []
    c = 3
    for exp in experiments:
        n = len(exp_bls[exp])
        exp_spans.append(rf"\multicolumn{{{n}}}{{c}}{{{EXP_LABEL.get(exp, exp)}}}")
        c += n
    lines.append(r"\multicolumn{2}{l}{} & " + " & ".join(exp_spans) + r" \\")

    # cmidrules
    cmidrules2 = []
    c = 3
    for exp in experiments:
        n = len(exp_bls[exp])
        cmidrules2.append(rf"\cmidrule(lr){{{c}-{c+n-1}}}")
        c += n
    lines.append("  ".join(cmidrules2))

    # Header row 2: baseline names per experiment
    bl_headers = " & ".join(
        BASELINE_LABEL.get(bl, bl) for exp in experiments for bl in exp_bls[exp]
    )
    lines.append(r"Group & Metric & " + bl_headers + r" \\")
    lines.append(r"\midrule")

    # Data rows: one per metric
    present_metrics = [
        m for m in METRICS
        if any(v is not None for d in data.values() for k, v in d.items() if k == m)
    ]

    for grp_name, grp_metrics in METRIC_GROUPS:
        grp_present = [m for m in grp_metrics if m in present_metrics]
        if not grp_present:
            continue
        for m_idx, m in enumerate(grp_present):
            grp_cell = (
                rf"\multirow{{{len(grp_present)}}}{{*}}{{\rotatebox{{90}}{{{grp_name}}}}}"
                if m_idx == 0 else ""
            )
            metric_cell = METRIC_LABEL.get(m, m)
            cells = [grp_cell, metric_cell]
            for exp in experiments:
                for bl in exp_bls[exp]:
                    v    = data.get((exp, bl), {}).get(m)
                    bold = best.get((exp, m)) == bl
                    cells.append(_fmt_bold(v, bold))
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\addlinespace[3pt]")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Transposed    -> {out_path.name}")


# ---------------------------------------------------------------------------
# Summary statistics print
# ---------------------------------------------------------------------------

def print_summary(rows: List[Row]) -> None:
    experiments = _ordered_experiments(sorted({str(r["experiment"]) for r in rows}))
    all_baselines = sorted({str(r["baseline"]) for r in rows})

    print(f"\n{'Experiment':<30} {'Baselines found'}")
    print("-" * 60)
    by_exp: Dict[str, List[str]] = {}
    for r in rows:
        by_exp.setdefault(str(r["experiment"]), [])
        bl = str(r["baseline"])
        if bl not in by_exp[str(r["experiment"])]:
            by_exp[str(r["experiment"])].append(bl)

    for exp in experiments:
        bls = _ordered_baselines(by_exp.get(exp, []))
        print(f"  {exp:<28} {', '.join(bls)}")
    print(f"\nTotal: {len(rows)} result rows, "
          f"{len(experiments)} experiments, {len(all_baselines)} baselines")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results", default=str(RESULTS_DIR),
        help=f"Path to results_baseline directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--results-coupling", default=str(RESULTS_DIR_COUPLING),
        help=f"Path to results_baseline_coupling directory (default: {RESULTS_DIR_COUPLING})",
    )
    parser.add_argument(
        "--out", default=str(OUT_DIR),
        help=f"Output directory (default: {OUT_DIR})",
    )
    args = parser.parse_args()

    results_dir          = Path(args.results)
    results_dir_coupling = Path(args.results_coupling)
    out_dir              = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading indep. results from   {results_dir} ...")
    rows = load_all_results(results_dir)
    if not rows:
        print("No result.csv files found. Check --results path.")
        return

    if results_dir_coupling.exists():
        print(f"Loading coupled results from   {results_dir_coupling} ...")
        coupled_rows = load_all_results(results_dir_coupling, only_coupled=True)
        print(f"  Found {len(coupled_rows)} coupled rows.")
        rows = rows + coupled_rows
    else:
        print(f"Coupling results dir not found, skipping: {results_dir_coupling}")

    # Load our-method results from W&B (entity=antoine311200, project=dualwcfm-2d)
    # Discriminated by config.loss: weighted / implicit / explicit
    print("Loading Ours (weight.) from W&B [loss=weighted] ...")
    our_weight = load_our_results_wandb("weighted")
    print(f"  Found {len(our_weight)} experiments.")

    print("Loading Ours (impl.)   from W&B [loss=implicit] ...")
    our_impl = load_our_results_wandb("implicit")
    print(f"  Found {len(our_impl)} experiments.")

    print("Loading Ours (exp.)    from W&B [loss=explicit] ...")
    our_expl = load_our_results_wandb("explicit")
    print(f"  Found {len(our_expl)} experiments.")

    print_summary(rows)

    print(f"\nWriting outputs to  {out_dir} ...")

    # 1. Per-experiment tables
    make_per_experiment_tables(rows, out_dir / "per_experiment")

    # 2. Full CSV
    make_full_csv(rows, out_dir / "full_comparison.csv",
                  our_weight=our_weight, our_impl=our_impl, our_expl=our_expl)

    # 3. Full LaTeX (grouped by experiment, metrics as columns)
    make_full_latex_table(rows, out_dir / "full_comparison.tex",
                          our_weight=our_weight, our_impl=our_impl, our_expl=our_expl)

    # 4. Transposed LaTeX (metrics as rows, experiments as column groups)
    make_transposed_latex_table(rows, out_dir / "full_comparison_transposed.tex",
                                our_weight=our_weight, our_impl=our_impl, our_expl=our_expl)

    print(f"\nDone.  All outputs in  {out_dir.resolve()}")


if __name__ == "__main__":
    main()
