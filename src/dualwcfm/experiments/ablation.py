"""dualwcfm.experiments.ablation
=================================
Ablation study infrastructure for paper-quality experiment comparisons.

Overview
--------
An *ablation study* runs the same base experiment under multiple configurations
(different methods, hyper-parameters, or random seeds) and then aggregates
results into publication-ready figures and tables.

Core classes
~~~~~~~~~~~~
``AblationVariant``
    One variant configuration: a name, a :class:`~dualwcfm.methods.base.MethodConfig`
    override, optional :class:`~dualwcfm.config.TrainingConfig` overrides, and a
    random seed.

``AblationSuite``
    A base :class:`~dualwcfm.experiments.base.ExperimentSpec` paired with a list
    of :class:`AblationVariant` objects to run against it.

``AblationRunner``
    Orchestrates training, metrics loading, checkpoint visualisation, and figure /
    table generation.

Typical workflow
~~~~~~~~~~~~~~~~
::

    suite = AblationSuite(
        name        = "2d_simple_methods",
        base_spec   = make_2d_experiments()[0],  # 2d_simple
        variants    = [
            AblationVariant("standard",  StandardConfig(sigma=0.01), seed=0),
            AblationVariant("mixture",   MixtureConfig(sigma=0.01, particle_ratio=0.25), seed=0),
            AblationVariant("score",     ScoreConfig(sigma=0.01), seed=0),
        ],
    )

    runner = AblationRunner(suite, output_root="ablation_results")
    runner.run()                        # train all variants
    runner.plot_convergence()           # SW distances vs iteration
    runner.plot_final_distributions()   # scatter plots of final coupling
    runner.plot_weight_evolution()      # importance weight comparison
    runner.plot_checkpoint_evolution(   # per-iteration coupling snapshots
        variant_names=["mixture"],
        iterations=[0, 25, 50, 75, 99],
    )
    runner.export_latex_table()         # LaTeX table for paper
    runner.export_summary_csv()         # machine-readable summary

Output layout
~~~~~~~~~~~~~
::

    {output_root}/{suite.name}/
        {variant_name}/              ← full per-variant experiment output
            models/
            checkpoints/
            metrics.csv
            ...
        figures/
            convergence.pdf
            final_distributions.pdf
            weight_evolution.pdf
            checkpoint_evolution_{variant}.pdf
        tables/
            summary.tex
            summary.csv
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from dualwcfm.config import TrainingConfig
from dualwcfm.methods.base import MethodConfig
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.experiments.base import ExperimentSpec, run_experiment, setup_logging

logger = logging.getLogger("dualwcfm")


# ---------------------------------------------------------------------------
# Plot style - consistent across all figures
# ---------------------------------------------------------------------------

_STYLE = {
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "lines.linewidth":    1.8,
}

#: Colour cycle for variants (up to 8 colours).
_COLORS = [
    "#2B7BB9",   # blue
    "#E05C5C",   # red
    "#4CAF50",   # green
    "#FF8C00",   # orange
    "#9C27B0",   # purple
    "#009688",   # teal
    "#795548",   # brown
    "#607D8B",   # blue-grey
]

#: Marker cycle
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


# ---------------------------------------------------------------------------
# AblationVariant
# ---------------------------------------------------------------------------

@dataclass
class AblationVariant:
    """One configuration variant for an ablation study.

    Parameters
    ----------
    name:
        Short human-readable identifier (used in plot legends and output paths).
    method_cfg:
        Method configuration that overrides the base spec's ``method_cfg``.
    train_overrides:
        Mapping of :class:`~dualwcfm.config.TrainingConfig` field names to
        values that override the base spec's ``train_cfg``.  Useful for sweeping
        ``iterations`` or ``batch_size`` etc.
    seed:
        Random seed applied before training.  Set to *None* to skip seeding.
    label:
        Optional longer label for figure legends (falls back to *name*).
    """
    name:            str
    method_cfg:      MethodConfig        = field(default_factory=MixtureConfig)
    train_overrides: Dict[str, Any]      = field(default_factory=dict)
    seed:            Optional[int]       = 0
    label:           Optional[str]       = None

    @property
    def display_label(self) -> str:
        return self.label if self.label is not None else self.name


# ---------------------------------------------------------------------------
# AblationSuite
# ---------------------------------------------------------------------------

@dataclass
class AblationSuite:
    """A base experiment paired with a list of variants to run.

    Parameters
    ----------
    name:
        Unique identifier for this ablation suite (used as the output sub-directory).
    base_spec:
        Template :class:`~dualwcfm.experiments.base.ExperimentSpec`.  Each
        variant derives its spec from this template by overriding ``method_cfg``
        and selected ``train_cfg`` fields.
    variants:
        Ordered list of :class:`AblationVariant` objects to run.
    """
    name:      str
    base_spec: ExperimentSpec
    variants:  List[AblationVariant] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AblationRunner
# ---------------------------------------------------------------------------

class AblationRunner:
    """Orchestrates training, visualisation, and reporting for an ablation suite.

    Parameters
    ----------
    suite:
        The :class:`AblationSuite` to run.
    output_root:
        Root directory;  all outputs land under ``{output_root}/{suite.name}/``.
    retrain:
        If *True* (default) run the full training loop.  Pass *False* to skip
        training and only regenerate plots from existing ``metrics.csv`` files.
    save_models:
        Whether to write model checkpoints to disk during training.
    """

    def __init__(
        self,
        suite:         AblationSuite,
        output_root:   str  = "ablation_results",
        retrain:       bool = True,
        save_models:   bool = True,
        wandb_project: Optional[str] = None,
        wandb_entity:  Optional[str] = None,
    ):
        self.suite         = suite
        self.output_root   = output_root
        self.retrain       = retrain
        self.save_models   = save_models
        self.wandb_project = wandb_project
        self.wandb_entity  = wandb_entity

        self.suite_dir  = os.path.join(output_root, suite.name)
        self.fig_dir    = os.path.join(self.suite_dir, "figures")
        self.table_dir  = os.path.join(self.suite_dir, "tables")

        for d in (self.suite_dir, self.fig_dir, self.table_dir):
            os.makedirs(d, exist_ok=True)

        setup_logging(self.suite_dir)

    # ------------------------------------------------------------------
    # Spec derivation
    # ------------------------------------------------------------------

    def _make_spec(self, variant: AblationVariant) -> ExperimentSpec:
        """Build a concrete :class:`ExperimentSpec` for *variant*."""
        spec = copy.deepcopy(self.suite.base_spec)

        # Variant-specific output directory
        spec.name        = variant.name
        spec.output_root = self.suite_dir

        # Override method config
        spec.method_cfg = copy.deepcopy(variant.method_cfg)

        # Override selected train_cfg fields
        for field_name, value in variant.train_overrides.items():
            if not hasattr(spec.train_cfg, field_name):
                raise ValueError(
                    f"TrainingConfig has no field '{field_name}' "
                    f"(requested by variant '{variant.name}')."
                )
            setattr(spec.train_cfg, field_name, value)

        return spec

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Train all variants sequentially."""
        logger.info("\n%s", "=" * 70)
        logger.info("  Ablation suite : %s  (%d variants)", self.suite.name, len(self.suite.variants))
        logger.info("%s", "=" * 70)

        for i, variant in enumerate(self.suite.variants):
            logger.info(
                "\n[%d/%d] Variant: %s", i + 1, len(self.suite.variants), variant.name
            )
            if variant.seed is not None:
                torch.manual_seed(variant.seed)
                np.random.seed(variant.seed)
                logger.info("  Seed: %d", variant.seed)

            spec = self._make_spec(variant)
            run_experiment(
                spec,
                retrain       = self.retrain,
                save_models   = self.save_models,
                wandb_project = self.wandb_project,
                wandb_entity  = self.wandb_entity,
                wandb_group   = self.suite.name,   # cluster all variants under the suite name
            )

        logger.info("\n  All variants complete.  Generating figures and tables ...")
        self.plot_convergence()
        self.plot_weight_evolution()
        self.export_summary_csv()
        self.export_latex_table()
        logger.info("  Done -> %s", self.suite_dir)

    # ------------------------------------------------------------------
    # Metrics loading
    # ------------------------------------------------------------------

    def _variant_dir(self, variant: AblationVariant) -> str:
        return os.path.join(self.suite_dir, variant.name)

    def load_metrics(self, variant: AblationVariant) -> List[Dict[str, str]]:
        """Read ``metrics.csv`` for *variant* and return it as a list of row dicts."""
        path = os.path.join(self._variant_dir(variant), "metrics.csv")
        if not os.path.exists(path):
            logger.warning("metrics.csv not found for variant '%s': %s", variant.name, path)
            return []
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def _final_metrics(self, variant: AblationVariant) -> Optional[Dict[str, float]]:
        """Return the *last* row of metrics.csv as a dict of floats, or None."""
        rows = self.load_metrics(variant)
        if not rows:
            return None
        last = rows[-1]
        return {k: float(v) for k, v in last.items() if v not in ("", "iteration")}

    # ------------------------------------------------------------------
    # Convergence plot
    # ------------------------------------------------------------------

    def plot_convergence(
        self,
        keys: Sequence[str] = ("W1_mu", "W2_mu", "W1_nu", "W2_nu"),
        save_pdf: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        """Plot SW-distance convergence curves for all variants.

        Creates a 2×2 grid of sub-plots (one per SW metric).

        Parameters
        ----------
        keys:
            Which metrics from ``metrics.csv`` to plot.
        save_pdf:
            Save the figure as ``figures/convergence.pdf``.
        show:
            Call ``plt.show()`` interactively.

        Returns
        -------
        The matplotlib :class:`~matplotlib.figure.Figure`.
        """
        key_labels = {
            "W1_mu": "SW₁ - source (μ)",
            "W2_mu": "SW₂ - source (μ)",
            "W1_nu": "SW₁ - target (ν)",
            "W2_nu": "SW₂ - target (ν)",
        }
        n_keys = len(keys)
        ncols  = min(n_keys, 2)
        nrows  = (n_keys + ncols - 1) // ncols

        with plt.rc_context(_STYLE):
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
            axes_flat = axes.ravel()

            for ax, key in zip(axes_flat, keys):
                for idx, variant in enumerate(self.suite.variants):
                    rows   = self.load_metrics(variant)
                    if not rows:
                        continue
                    iters  = [int(r["iteration"]) for r in rows]
                    vals   = [float(r.get(key, 0)) for r in rows]
                    ax.plot(
                        iters, vals,
                        label  = variant.display_label,
                        color  = _COLORS[idx % len(_COLORS)],
                        marker = _MARKERS[idx % len(_MARKERS)],
                        markevery = max(1, len(iters) // 10),
                        markersize = 5,
                    )
                ax.set_title(key_labels.get(key, key))
                ax.set_xlabel("Iteration")
                ax.set_ylabel("SW distance")
                ax.legend(frameon=False)

            # Hide any spare axes
            for ax in axes_flat[n_keys:]:
                ax.set_visible(False)

            fig.suptitle(
                f"Convergence - {self.suite.name}",
                fontsize=13, fontweight="bold", y=1.01,
            )
            fig.tight_layout()

        if save_pdf:
            path = os.path.join(self.fig_dir, "convergence.pdf")
            fig.savefig(path, bbox_inches="tight")
            logger.info("  Saved convergence figure -> %s", path)
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Weight evolution plot
    # ------------------------------------------------------------------

    def plot_weight_evolution(
        self,
        save_pdf: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        """Plot importance-weight mean ± std for all variants.

        Returns
        -------
        The matplotlib :class:`~matplotlib.figure.Figure`.
        """
        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(7, 4))

            for idx, variant in enumerate(self.suite.variants):
                rows = self.load_metrics(variant)
                if not rows:
                    continue
                iters  = [int(r["iteration"]) for r in rows]
                w_mean = [float(r.get("weight_mean", 1.0)) for r in rows]
                w_std  = [float(r.get("weight_std",  0.0)) for r in rows]
                color  = _COLORS[idx % len(_COLORS)]

                ax.plot(iters, w_mean, label=variant.display_label, color=color)
                ax.fill_between(
                    iters,
                    [m - s for m, s in zip(w_mean, w_std)],
                    [m + s for m, s in zip(w_mean, w_std)],
                    alpha=0.15, color=color,
                )

            ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="ideal (=1)")
            ax.set_title(f"Importance weight evolution - {self.suite.name}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("mean(w)")
            ax.legend(frameon=False)
            fig.tight_layout()

        if save_pdf:
            path = os.path.join(self.fig_dir, "weight_evolution.pdf")
            fig.savefig(path, bbox_inches="tight")
            logger.info("  Saved weight evolution figure -> %s", path)
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Final-distribution scatter (2D only)
    # ------------------------------------------------------------------

    def plot_final_distributions(
        self,
        n_samples: int = 2_000,
        save_pdf: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """Scatter-plot the final coupling (x, y) produced by each variant.

        Only supported for 2-D experiments.  Returns *None* for other modes.

        Parameters
        ----------
        n_samples:
            Number of coupling pairs to sample from the final checkpoint.
        save_pdf:
            Save as ``figures/final_distributions.pdf``.
        show:
            Call ``plt.show()`` interactively.
        """
        if self.suite.base_spec.mode != "2d":
            logger.info(
                "  plot_final_distributions: skipping (mode='%s', only supported for '2d').",
                self.suite.base_spec.mode,
            )
            return None

        n_variants = len(self.suite.variants)
        with plt.rc_context(_STYLE):
            fig, axes = plt.subplots(1, n_variants, figsize=(4 * n_variants, 4), squeeze=False)

            for ax, variant in zip(axes[0], self.suite.variants):
                ckpt_path = os.path.join(
                    self._variant_dir(variant), "models", "checkpoints",
                    "coupling", "iter_final.pt",
                )
                # Prefer the last available coupling checkpoint
                coup_dir = os.path.join(
                    self._variant_dir(variant), "models", "checkpoints", "coupling"
                )
                xy = None
                if os.path.isdir(coup_dir):
                    pts = sorted(
                        [f for f in os.listdir(coup_dir) if f.endswith(".pt")],
                        key=lambda f: int(f.split("_")[1].split(".")[0])
                    )
                    if pts:
                        data = torch.load(os.path.join(coup_dir, pts[-1]), map_location="cpu")
                        x_t  = data["x"][:n_samples]
                        y_t  = data["y"][:n_samples]
                        xy   = (x_t.numpy(), y_t.numpy())

                if xy is None:
                    ax.text(0.5, 0.5, "no checkpoint", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="grey")
                else:
                    x_np, y_np = xy
                    ax.scatter(x_np[:, 0], x_np[:, 1], s=3, alpha=0.4,
                               color=_COLORS[0], label="source x")
                    ax.scatter(y_np[:, 0], y_np[:, 1], s=3, alpha=0.4,
                               color=_COLORS[1], label="target y")
                    # Draw transport arrows for a subset
                    step = max(1, len(x_np) // 40)
                    for xi, yi in zip(x_np[::step], y_np[::step]):
                        ax.annotate(
                            "", xy=yi[:2], xytext=xi[:2],
                            arrowprops=dict(arrowstyle="-|>", color="grey", lw=0.6, alpha=0.4),
                        )

                ax.set_title(variant.display_label)
                ax.set_aspect("equal")
                ax.legend(frameon=False, markerscale=3, fontsize=8)

            fig.suptitle(
                f"Final coupling - {self.suite.name}",
                fontsize=13, fontweight="bold",
            )
            fig.tight_layout()

        if save_pdf:
            path = os.path.join(self.fig_dir, "final_distributions.pdf")
            fig.savefig(path, bbox_inches="tight")
            logger.info("  Saved final distributions figure -> %s", path)
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Checkpoint evolution plot
    # ------------------------------------------------------------------

    def plot_checkpoint_evolution(
        self,
        variant_names: Optional[List[str]] = None,
        iterations: Optional[Sequence[int]] = None,
        n_samples: int = 1_000,
        save_pdf: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """Visualise how the coupling evolves across saved iteration checkpoints.

        Only supported for 2-D experiments.  Plots columns = iterations,
        rows = selected variants.

        Parameters
        ----------
        variant_names:
            Subset of variant names to plot.  Defaults to all variants.
        iterations:
            Which iteration indices to load (e.g. ``[0, 25, 50, 75, 99]``).
            Defaults to 5 evenly-spaced checkpoints.
        n_samples:
            Number of samples to draw from each coupling checkpoint.
        save_pdf:
            Save as ``figures/checkpoint_evolution.pdf``.
        show:
            Call ``plt.show()`` interactively.
        """
        if self.suite.base_spec.mode != "2d":
            logger.info(
                "  plot_checkpoint_evolution: skipping (mode='%s', only supported for '2d').",
                self.suite.base_spec.mode,
            )
            return None

        # Filter variants
        selected = (
            [v for v in self.suite.variants if v.name in variant_names]
            if variant_names
            else self.suite.variants
        )

        # Discover available checkpoints for the first variant to determine iteration range
        def _available_iterations(variant: AblationVariant) -> List[int]:
            coup_dir = os.path.join(
                self._variant_dir(variant), "models", "checkpoints", "coupling"
            )
            if not os.path.isdir(coup_dir):
                return []
            files = sorted(
                [f for f in os.listdir(coup_dir) if f.startswith("iter_") and f.endswith(".pt")],
                key=lambda f: int(f.split("_")[1].split(".")[0]),
            )
            return [int(f.split("_")[1].split(".")[0]) for f in files]

        # Determine iteration list
        if iterations is None:
            all_iters = _available_iterations(selected[0]) if selected else []
            if not all_iters:
                logger.warning("  No coupling checkpoints found - skipping checkpoint evolution.")
                return None
            step       = max(1, len(all_iters) // 5)
            iterations = all_iters[::step][:5]

        n_rows = len(selected)
        n_cols = len(iterations)
        with plt.rc_context(_STYLE):
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize = (3.5 * n_cols, 3.5 * n_rows),
                squeeze = False,
            )

            def _load_coupling(variant, it) -> Optional[Tuple[np.ndarray, np.ndarray]]:
                coup_dir = os.path.join(
                    self._variant_dir(variant), "models", "checkpoints", "coupling"
                )
                path = os.path.join(coup_dir, f"iter_{it:04d}.pt")
                if not os.path.exists(path):
                    return None
                data = torch.load(path, map_location="cpu")
                x    = data["x"][:n_samples].numpy()
                y    = data["y"][:n_samples].numpy()
                return x, y

            for row, variant in enumerate(selected):
                for col, it in enumerate(iterations):
                    ax  = axes[row][col]
                    xy  = _load_coupling(variant, it)
                    if xy is None:
                        ax.text(0.5, 0.5, f"no ckpt\niter={it}", ha="center", va="center",
                                transform=ax.transAxes, fontsize=8, color="grey")
                    else:
                        x_np, y_np = xy
                        ax.scatter(x_np[:, 0], x_np[:, 1], s=2, alpha=0.4, color=_COLORS[0])
                        ax.scatter(y_np[:, 0], y_np[:, 1], s=2, alpha=0.4, color=_COLORS[1])
                        # Arrows for a subset
                        step = max(1, len(x_np) // 25)
                        for xi, yi in zip(x_np[::step], y_np[::step]):
                            ax.annotate(
                                "", xy=yi[:2], xytext=xi[:2],
                                arrowprops=dict(arrowstyle="-|>", color="grey", lw=0.5, alpha=0.35),
                            )
                    if col == 0:
                        ax.set_ylabel(variant.display_label, fontsize=10)
                    if row == 0:
                        ax.set_title(f"iter {it}", fontsize=10)
                    ax.set_aspect("equal")
                    ax.tick_params(left=False, bottom=False,
                                   labelleft=False, labelbottom=False)

            fig.suptitle(
                f"Coupling evolution - {self.suite.name}",
                fontsize=13, fontweight="bold",
            )
            fig.tight_layout()

        if save_pdf:
            path = os.path.join(self.fig_dir, "checkpoint_evolution.pdf")
            fig.savefig(path, bbox_inches="tight")
            logger.info("  Saved checkpoint evolution figure -> %s", path)
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Multi-metric bar chart (final values)
    # ------------------------------------------------------------------

    def plot_final_metrics_bar(
        self,
        keys: Sequence[str] = ("W1_mu", "W2_mu", "W1_nu", "W2_nu"),
        save_pdf: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        """Grouped bar chart of final SW-distance values for all variants.

        Useful for a quick side-by-side comparison in a paper.

        Returns
        -------
        The matplotlib :class:`~matplotlib.figure.Figure`.
        """
        key_labels = {
            "W1_mu": "SW₁(μ)", "W2_mu": "SW₂(μ)",
            "W1_nu": "SW₁(ν)", "W2_nu": "SW₂(ν)",
        }
        n_variants = len(self.suite.variants)
        n_keys     = len(keys)
        x          = np.arange(n_keys)
        bar_width  = 0.8 / n_variants

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(max(6, 2 * n_keys), 4))

            for idx, variant in enumerate(self.suite.variants):
                fm = self._final_metrics(variant)
                if fm is None:
                    continue
                vals   = [fm.get(k, 0.0) for k in keys]
                offset = (idx - n_variants / 2 + 0.5) * bar_width
                bars   = ax.bar(
                    x + offset, vals,
                    width  = bar_width * 0.9,
                    color  = _COLORS[idx % len(_COLORS)],
                    label  = variant.display_label,
                    alpha  = 0.85,
                )
                # Value labels on top of bars
                for bar, val in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels([key_labels.get(k, k) for k in keys])
            ax.set_ylabel("SW distance (final iteration)")
            ax.set_title(f"Final metrics - {self.suite.name}")
            ax.legend(frameon=False)
            fig.tight_layout()

        if save_pdf:
            path = os.path.join(self.fig_dir, "final_metrics_bar.pdf")
            fig.savefig(path, bbox_inches="tight")
            logger.info("  Saved final metrics bar chart -> %s", path)
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # LaTeX table export
    # ------------------------------------------------------------------

    def export_latex_table(
        self,
        keys: Sequence[str] = ("W1_mu", "W2_mu", "W1_nu", "W2_nu"),
        caption: str = "",
        label: str   = "tab:ablation",
    ) -> str:
        """Export a LaTeX ``booktabs`` table of final metrics.

        The best (minimum) value in each column is rendered in bold.

        Parameters
        ----------
        keys:
            Metric columns to include.
        caption:
            LaTeX caption string.
        label:
            LaTeX label string for ``\\label{...}``.

        Returns
        -------
        The LaTeX source as a string (also written to ``tables/summary.tex``).
        """
        key_labels = {
            "W1_mu": r"$\mathrm{SW}_1(\mu)$",
            "W2_mu": r"$\mathrm{SW}_2(\mu)$",
            "W1_nu": r"$\mathrm{SW}_1(\nu)$",
            "W2_nu": r"$\mathrm{SW}_2(\nu)$",
        }

        # Collect final metric values
        data: List[Tuple[str, Dict[str, Optional[float]]]] = []
        for variant in self.suite.variants:
            fm = self._final_metrics(variant)
            data.append((variant.display_label, fm or {}))

        # Find best (min) per column
        best: Dict[str, float] = {}
        for key in keys:
            vals = [d[key] for _, d in data if key in d and d[key] is not None]
            if vals:
                best[key] = min(vals)

        # Build LaTeX
        col_fmt   = "l" + "r" * len(keys)
        header    = " & ".join(["Method"] + [key_labels.get(k, k) for k in keys])
        sep       = r"\midrule"
        rows      = []
        for name, fm in data:
            cells = [name]
            for key in keys:
                val = fm.get(key)
                if val is None:
                    cells.append("—")
                elif key in best and abs(val - best[key]) < 1e-9:
                    cells.append(r"\textbf{" + f"{val:.4f}" + "}")
                else:
                    cells.append(f"{val:.4f}")
            rows.append(" & ".join(cells) + r" \\")

        cap_str   = caption or f"Final SW distances for the {self.suite.name} ablation."
        latex = "\n".join([
            r"\begin{table}[ht]",
            r"  \centering",
            r"  \caption{" + cap_str + "}",
            r"  \label{" + label + "}",
            r"  \begin{tabular}{" + col_fmt + "}",
            r"    \toprule",
            f"    {header} \\\\",
            f"    {sep}",
        ] + [
            f"    {row}" for row in rows
        ] + [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ])

        path = os.path.join(self.table_dir, "summary.tex")
        with open(path, "w") as f:
            f.write(latex + "\n")
        logger.info("  Saved LaTeX table -> %s", path)
        return latex

    # ------------------------------------------------------------------
    # CSV summary export
    # ------------------------------------------------------------------

    def export_summary_csv(
        self,
        keys: Sequence[str] = ("W1_mu", "W2_mu", "W1_nu", "W2_nu"),
    ) -> str:
        """Write a machine-readable CSV with one row per variant.

        Returns the path to the written file.
        """
        path = os.path.join(self.table_dir, "summary.csv")
        fieldnames = ["variant"] + list(keys)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for variant in self.suite.variants:
                fm  = self._final_metrics(variant) or {}
                row = {"variant": variant.display_label}
                row.update({k: fm.get(k, "") for k in keys})
                writer.writerow(row)

        logger.info("  Saved summary CSV -> %s", path)
        return path

    # ------------------------------------------------------------------
    # Per-metric seed-aggregation plot (multiple seeds)
    # ------------------------------------------------------------------

    def plot_seed_aggregate(
        self,
        metric: str = "W2_nu",
        save_pdf: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        """Plot mean ± std over seeds, grouping variants by *name prefix*.

        Variants that share the same ``name`` prefix (everything before the last
        ``_seed<N>`` suffix) are treated as seeds of the same configuration.

        Example
        -------
        If the suite contains variants named ``mixture_seed0``, ``mixture_seed1``,
        ``mixture_seed2``, they are aggregated into a single shaded curve.

        Returns
        -------
        The matplotlib :class:`~matplotlib.figure.Figure`.
        """
        import re

        def _group_key(name: str) -> str:
            return re.sub(r"_seed\d+$", "", name)

        groups: Dict[str, List[AblationVariant]] = {}
        for v in self.suite.variants:
            key = _group_key(v.name)
            groups.setdefault(key, []).append(v)

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(7, 4))

            for idx, (group_name, variants) in enumerate(groups.items()):
                # Collect curves, pad to equal length
                all_curves = []
                for v in variants:
                    rows = self.load_metrics(v)
                    if rows:
                        all_curves.append([float(r.get(metric, 0)) for r in rows])

                if not all_curves:
                    continue

                min_len = min(len(c) for c in all_curves)
                arr     = np.array([c[:min_len] for c in all_curves])  # (n_seeds, T)
                iters   = list(range(min_len))
                mean    = arr.mean(axis=0)
                std     = arr.std(axis=0)
                color   = _COLORS[idx % len(_COLORS)]

                # Pick display label from first variant (strip seed suffix)
                label = variants[0].label or group_name

                ax.plot(iters, mean, label=label, color=color)
                ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=color)

            ax.set_title(f"Seed aggregate - {metric} - {self.suite.name}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(metric)
            ax.legend(frameon=False)
            fig.tight_layout()

        if save_pdf:
            path = os.path.join(self.fig_dir, f"seed_aggregate_{metric}.pdf")
            fig.savefig(path, bbox_inches="tight")
            logger.info("  Saved seed aggregate figure -> %s", path)
        if show:
            plt.show()
        return fig
