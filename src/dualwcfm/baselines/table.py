"""dualwcfm.baselines.table
===========================
Aggregate baseline results into LaTeX tables and wide-format CSVs.

Reads two sources:
  - ``{output_root}/baselines_summary.csv`` — one row per (experiment, baseline)
  - ``{output_root}/{exp}/metrics.csv``     — final row per DualWCFM experiment

Produces a pivot table: rows = experiments, columns = methods, cells = metric value.

Usage
-----
    from dualwcfm.baselines.table import build_comparison_table
    build_comparison_table("results", metric="W2_nu")

    # Or from the CLI:
    python -m dualwcfm.baselines.run --table --output results --metric W2_nu
"""
from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("dualwcfm")

# All possible metric columns (not every mode has all of them — sparse is fine).
_ALL_METRICS = [
    "W1_mu", "W2_mu", "W1_nu", "W2_nu",
    "mmd_nu", "energy_dist_nu", "sinkhorn_div",
    "map_err_fwd",
    "bw2", "mean_err_nu", "cov_frob_nu",
    "fid", "kid",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_baseline_csv(output_root: str) -> List[Dict]:
    path = os.path.join(output_root, "baselines_summary.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_main_metrics(output_root: str) -> List[Dict]:
    """Read the *final* row from each experiment's metrics.csv as a DualWCFM row."""
    rows = []
    for exp_name in sorted(os.listdir(output_root)):
        metrics_path = os.path.join(output_root, exp_name, "metrics.csv")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, newline="") as f:
            data = list(csv.DictReader(f))
        if data:
            last = data[-1]
            row = {"experiment": exp_name, "baseline": "DualWCFM"}
            for m in _ALL_METRICS:
                row[m] = last.get(m, "")
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Pivot & format
# ---------------------------------------------------------------------------

def _pivot(rows: List[Dict], metric: str) -> Dict[str, Dict[str, str]]:
    """Pivot rows into {experiment: {baseline: metric_value}}."""
    table: Dict[str, Dict[str, str]] = {}
    for row in rows:
        exp = row.get("experiment", "")
        bl  = row.get("baseline",   "")
        val = row.get(metric, "")
        table.setdefault(exp, {})[bl] = val
    return table


def _fmt(v: str, bold: bool = False) -> str:
    try:
        s = f"{float(v):.4f}"
    except (ValueError, TypeError):
        s = str(v) if v else "--"
    return r"\textbf{" + s + "}" if bold else s


def _best_key(row_vals: Dict[str, str]) -> Optional[str]:
    """Return the method name with the smallest numeric value (lower = better)."""
    best_k, best_v = None, float("inf")
    for k, v in row_vals.items():
        try:
            fv = float(v)
            if fv < best_v:
                best_v, best_k = fv, k
        except (ValueError, TypeError):
            pass
    return best_k


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_comparison_table(
    output_root: str,
    metric:      str  = "W2_nu",
    save_latex:  bool = True,
    save_csv:    bool = True,
) -> str:
    """Build a comparison table for *metric* across all methods and experiments.

    Parameters
    ----------
    output_root:
        Root results directory.
    metric:
        Which metric to display (e.g. ``"W2_nu"``, ``"fid"``, ``"bw2"``).
    save_latex:
        Write ``comparison_{metric}.tex``.
    save_csv:
        Write ``comparison_{metric}.csv``.

    Returns
    -------
    LaTeX table string (also printed to stdout / saved to disk).
    """
    all_rows = _load_baseline_csv(output_root) + _load_main_metrics(output_root)
    if not all_rows:
        logger.warning("No data found under %s", output_root)
        return ""

    table       = _pivot(all_rows, metric)
    experiments = sorted(table.keys())

    # Column order: DualWCFM first, then alphabetical baselines
    bl_set: List[str] = []
    for vals in table.values():
        for k in vals:
            if k not in bl_set:
                bl_set.append(k)
    if "DualWCFM" in bl_set:
        bl_set.remove("DualWCFM")
    all_baselines = ["DualWCFM"] + sorted(bl_set)

    # ------------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------------
    col_spec = "l" + "c" * len(all_baselines)
    header   = " & ".join(
        ["Experiment"] + [bl.replace("_", r"\_") for bl in all_baselines]
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + f"Comparison on metric \\texttt{{{metric}}}" + r"}",
        r"\label{tab:comparison_" + metric + r"}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for exp in experiments:
        row_vals = {bl: table[exp].get(bl, "") for bl in all_baselines}
        best     = _best_key(row_vals)
        cells    = [exp.replace("_", r"\_")]
        for bl in all_baselines:
            cells.append(_fmt(row_vals[bl], bold=(bl == best)))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(lines)

    # ------------------------------------------------------------------
    # Wide CSV
    # ------------------------------------------------------------------
    if save_csv:
        csv_path = os.path.join(output_root, f"comparison_{metric}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment"] + all_baselines)
            for exp in experiments:
                writer.writerow(
                    [exp] + [table[exp].get(bl, "") for bl in all_baselines]
                )
        logger.info("Comparison CSV  -> %s", csv_path)

    if save_latex:
        tex_path = os.path.join(output_root, f"comparison_{metric}.tex")
        with open(tex_path, "w") as f:
            f.write(latex)
        logger.info("LaTeX table     -> %s", tex_path)

    return latex


def build_all_tables(output_root: str) -> None:
    """Build one comparison table per available metric."""
    all_rows = _load_baseline_csv(output_root) + _load_main_metrics(output_root)
    present  = {m for row in all_rows for m in _ALL_METRICS if row.get(m, "")}
    for metric in _ALL_METRICS:
        if metric in present:
            build_comparison_table(output_root, metric=metric)
    logger.info("All comparison tables written to %s/", output_root)
