"""Finance-specific logging visualizer.

Inherits from LoggingVisualizer (2-D) for standard W2/MMD/ESS metrics
and CSV/GIF machinery, then adds domain-specific plot callbacks.

Each experiment mode (iv_surface, yield_curve, risk_factors) passes its
ground-truth test data via `extra_kwargs` so the visualizer can compute
domain metrics at every checkpoint.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import numpy as np
import torch

from dualwcfm.config import VisualizeConfig
from dualwcfm.logging_visualizer import LoggingVisualizer

if TYPE_CHECKING:
    from dualwcfm.trainer import Trainer

logger = logging.getLogger("dualwcfm")


class FinanceVisualizer(LoggingVisualizer):
    """Generic finance visualizer — records domain metrics in a separate CSV.

    Sub-classes (IV, YC, RF) override `_make_domain_plots` and
    `_compute_domain_metrics`.

    Args:
        vis_cfg:         Standard VisualizeConfig.
        output_dir:      Where to save plots and CSVs.
        truth_x:         Ground-truth target *source* samples (N, D) tensor.
        truth_y:         Ground-truth target *target* samples (N, D) tensor.
        wandb_run:       Optional W&B run for logging.
    """

    def __init__(
        self,
        vis_cfg: VisualizeConfig,
        output_dir: str,
        truth_x: np.ndarray,
        truth_y: np.ndarray,
        wandb_run=None,
        **kwargs,          # mode-specific data passed by experiment factories
    ):
        super().__init__(vis_cfg, transport_map=None, output_dir=output_dir,
                         wandb_run=wandb_run)
        self.truth_x = truth_x
        self.truth_y = truth_y

        self._domain_csv = Path(output_dir) / "finance_metrics.csv"
        self._domain_fields: list[str] = []   # set by subclasses
        self._domain_rows: list[dict] = []

        # Store coupling history for evolution plots
        self._coupling_history: list[np.ndarray] = []
        self._iteration_log: list[int] = []

        # Extra mode-specific data (e.g. HestonParams, regime labels)
        self._extra = kwargs

        # Create domain-specific sub-directories
        (Path(output_dir) / "finance").mkdir(exist_ok=True)

    # ── Hooks ────────────────────────────────────────────────────────────────

    def display(self, iteration: int, trainer: "Trainer") -> None:
        """Standard display + domain plots at checkpoint iterations."""
        # Run standard 2-D scatter / weight visualisation
        super().display(iteration, trainer)

        if iteration % self.config.show_each != 0:
            return

        # Snapshot current coupling (y samples only — the "target")
        try:
            n_snap = min(2000, trainer.config.dataset_size)
            x_gen, y_gen = trainer.current_coupling.coupling_sample(n_snap)
            # Rescale back to original data space
            y_np = y_gen.detach().cpu().numpy() * float(trainer.scale)
            self._coupling_history.append(y_np)
            self._iteration_log.append(iteration)

            fig = self._make_domain_plots(iteration, y_np)
            if fig is not None:
                import matplotlib.pyplot as plt
                path = Path(self.output_dir) / "finance" / f"finance_{iteration:04d}.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                if self._wandb_run is not None:
                    try:
                        import wandb
                        self._wandb_run.log(
                            {"finance/plot": wandb.Image(str(path))}, step=iteration
                        )
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("Finance display error at iter %d: %s", iteration, exc)

    def analyse(self, iteration: int, trainer: "Trainer") -> dict:
        """Standard metrics + domain-specific metrics appended to CSV."""
        base = super().analyse(iteration, trainer)

        # Domain metrics
        try:
            n_snap = min(2000, trainer.config.dataset_size)
            x_gen, y_gen = trainer.current_coupling.coupling_sample(n_snap)
            y_np = y_gen.detach().cpu().numpy() * float(trainer.scale)
            dom  = self._compute_domain_metrics(y_np)
        except Exception as exc:
            logger.debug("Domain metrics error at iter %d: %s", iteration, exc)
            dom = {}

        # Write to domain CSV (lazy init — always fresh per visualizer instance)
        if dom:
            if not self._domain_fields:
                # Initialise on first non-empty dom dict and (re)write the header,
                # overwriting any stale file from a previous run that may have
                # different columns (e.g. old 2-column vs new 7-column schema).
                self._domain_fields = ["iteration"] + sorted(dom.keys())
                with open(self._domain_csv, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=self._domain_fields).writeheader()

            row = {"iteration": iteration, **dom}
            with open(self._domain_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self._domain_fields).writerow(row)

            # Log to W&B
            if self._wandb_run is not None:
                try:
                    self._wandb_run.log(
                        {f"finance/{k}": v for k, v in dom.items()}, step=iteration
                    )
                except Exception:
                    pass

        return {**base, **dom}

    # ── Override in subclasses ────────────────────────────────────────────────

    def _make_domain_plots(self, iteration: int, y_gen: np.ndarray):
        """Return a matplotlib Figure or None.  Called every `show_each` iters."""
        return None

    def _compute_domain_metrics(self, y_gen: np.ndarray) -> dict:
        """Return dict of domain-specific scalar metrics."""
        return {}

    def make_final_plots(self, trainer: "Trainer") -> None:
        """Snapshot the final trained coupling state and append to history.

        Always called before subclass make_final_plots so that
        self._coupling_history[-1] is the true end-of-training distribution,
        even when the last iteration index was not a show_each multiple.
        """
        try:
            n_snap = min(2000, trainer.config.dataset_size)
            _, y_final = trainer.current_coupling.coupling_sample(n_snap)
            y_final_np = y_final.detach().cpu().numpy() * float(trainer.scale)
            # Append as the terminal entry — label it as iteration N (total)
            self._coupling_history.append(y_final_np)
            self._iteration_log.append(trainer.config.iterations)
            logger.info("Snapshotted final coupling at iteration %d.",
                        trainer.config.iterations)
        except Exception as exc:
            logger.warning("Could not snapshot final coupling: %s", exc)


# ── IV Surface Visualizer ────────────────────────────────────────────────────

class IVSurfaceVisualizer(FinanceVisualizer):
    """Visualizer for E1: IV Surface Regime Transfer."""

    def __init__(self, vis_cfg, output_dir, truth_x, truth_y,
                 ref_x, ref_y, wandb_run=None, **kwargs):
        super().__init__(vis_cfg, output_dir, truth_x, truth_y,
                         wandb_run=wandb_run, **kwargs)
        self._ref_x = ref_x
        self._ref_y = ref_y

    def _make_domain_plots(self, iteration, y_gen):
        from .plots.iv_surface import plot_smile_comparison
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return plot_smile_comparison(
                    ref_surfaces=self._ref_y,
                    taco_surfaces=y_gen,
                    truth_surfaces=self.truth_y,
                )
        except Exception:
            return None

    def _compute_domain_metrics(self, y_gen) -> dict:
        from .plots.iv_surface import (
            compute_iv_rmse, compute_skew_rmse,
            compute_arbitrage_violations, project_surfaces_convex,
        )
        try:
            y_proj = project_surfaces_convex(y_gen)
            rmse      = compute_iv_rmse(y_proj, self.truth_y)
            skew_err  = compute_skew_rmse(y_proj, self.truth_y)
            arb_raw   = compute_arbitrage_violations(y_gen)    # before projection
            arb_proj  = compute_arbitrage_violations(y_proj)   # after projection
            return {
                "iv_rmse_pp":          float(rmse * 100),
                "skew_rmse_1m_pp":     float(skew_err[0]),
                "skew_rmse_3m_pp":     float(skew_err[1]),
                "arb_bf_1m_raw_pct":   float(arb_raw["butterfly_1m"] * 100),
                "arb_bf_1m_proj_pct":  float(arb_proj["butterfly_1m"] * 100),
                "arb_cal_pct":         float(arb_proj["calendar"] * 100),
            }
        except Exception:
            return {}

    def make_final_plots(self, trainer) -> None:
        super().make_final_plots(trainer)   # snapshot final state into history
        out = Path(self.output_dir) / "finance"
        y_gen = self._coupling_history[-1]  # always the final trained state
        y_pre = self._coupling_history[0] if self._coupling_history else None

        # ── Apply convexity projection to TACO outputs ─────────────────────
        # The flow model has no convexity constraint, so its outputs may violate
        # butterfly no-arbitrage.  Project each smile to a convex quadratic.
        from .plots.iv_surface import project_surfaces_convex
        y_gen_proj = project_surfaces_convex(y_gen)

        # ── Get current coupling (x, y) pairs ──────────────────────────────
        try:
            n_snap = min(2000, trainer.config.dataset_size)
            x_gen_t, y_gen_t = trainer.current_coupling.coupling_sample(n_snap)
            x_gen_np = x_gen_t.detach().cpu().numpy() * float(trainer.scale)
            y_gen_np = y_gen_t.detach().cpu().numpy() * float(trainer.scale)
        except Exception:
            x_gen_np, y_gen_np = None, None

        # ── Compute "transported" reference pairs for daily-moves ──────────
        # The TACO coupling degenerates to near-identity when μ≈ν (both are the
        # stressed stationary distribution), so coupling daily moves are ≈ 0.
        # The informative diagnostic is: push the REFERENCE consecutive-day
        # pairs (x_calm, y_calm) through the μ/ν marginal bridges, then compute
        # transported_y - transported_x to show how calm dynamics are mapped to
        # the stressed regime.
        transported_x, transported_y = None, None
        try:
            import torch
            device = trainer.device
            (mu_model, _), (nu_model, _) = trainer.strategy.get_bridges(device)
            mu_model.eval()
            nu_model.eval()

            n_pairs = min(500, len(self._ref_x))
            x_r = torch.tensor(self._ref_x[:n_pairs], dtype=torch.float32, device=device)
            y_r = torch.tensor(self._ref_y[:n_pairs], dtype=torch.float32, device=device)
            scale = float(trainer.scale)

            with torch.no_grad():
                x_r_n = x_r / scale
                y_r_n = y_r / scale
                # extend=False: bridge models take plain (t, xt), not doubled state
                x_stress_n = trainer.flow_matching.evolve(
                    x_r_n, mu_model, num_steps=50, extend=False
                )
                y_stress_n = trainer.flow_matching.evolve(
                    y_r_n, nu_model, num_steps=50, extend=False
                )
            transported_x = x_stress_n.cpu().numpy() * scale
            transported_y = y_stress_n.cpu().numpy() * scale
        except Exception as exc:
            logger.warning("Could not compute transported pairs: %s", exc)

        from .plots.iv_surface import (
            plot_smile_comparison, plot_surface_heatmap,
            plot_coupling_evolution, plot_3d_surface,
            plot_skew_slope, plot_term_structure,
            plot_arbitrage_checks, plot_daily_moves,
        )
        import matplotlib.pyplot as plt

        figs = [
            ("smile_comparison.pdf",
             plot_smile_comparison(self._ref_y, y_gen_proj, self.truth_y, y_pre)),
            ("surface_heatmap.pdf",
             plot_surface_heatmap(self._ref_y.mean(0), y_gen_proj.mean(0), self.truth_y.mean(0))),
            ("surface_3d.pdf",
             plot_3d_surface(self._ref_y, y_gen_proj, self.truth_y, y_pre)),
            ("skew_slope.pdf",
             plot_skew_slope(self._ref_y, y_gen_proj, self.truth_y, y_pre)),
            ("term_structure.pdf",
             plot_term_structure(self._ref_y, y_gen_proj, self.truth_y, y_pre)),
        ]
        if len(self._coupling_history) > 1:
            figs.append(("coupling_evolution.pdf",
                          plot_coupling_evolution(
                              self._coupling_history, self.truth_y, self._iteration_log)))
            figs.append(("arbitrage_checks.pdf",
                          plot_arbitrage_checks(
                              self._coupling_history, self.truth_y, self._iteration_log,
                              ref_surfaces=self._ref_y)))
        if x_gen_np is not None:
            figs.append(("daily_moves.pdf",
                          plot_daily_moves(
                              self._ref_x, self._ref_y,
                              x_gen_np, y_gen_np,
                              self.truth_x, self.truth_y,
                              transported_x=transported_x,
                              transported_y=transported_y)))

        for name, fig in figs:
            if fig is not None:
                fig.savefig(out / name, dpi=300, bbox_inches="tight")
                if self._wandb_run is not None:
                    try:
                        import wandb as _wb
                        self._wandb_run.log({
                            f"finance/final/{name.replace('.pdf', '')}":
                            _wb.Image(str(out / name))
                        })
                    except Exception:
                        pass
                plt.close(fig)


# ── Yield Curve Visualizer ───────────────────────────────────────────────────

class YieldCurveVisualizer(FinanceVisualizer):
    """Visualizer for E2: Yield Curve Dynamics Transfer."""

    def __init__(self, vis_cfg, output_dir, truth_x, truth_y,
                 ref_x, ref_y, wandb_run=None, **kwargs):
        super().__init__(vis_cfg, output_dir, truth_x, truth_y,
                         wandb_run=wandb_run, **kwargs)
        self._ref_x = ref_x
        self._ref_y = ref_y

    def _make_domain_plots(self, iteration, y_gen):
        from .plots.yield_curve import plot_curve_comparison
        try:
            return plot_curve_comparison(self._ref_y, y_gen, self.truth_y)
        except Exception:
            return None

    def _compute_domain_metrics(self, y_gen) -> dict:
        from .plots.yield_curve import compute_yc_rmse
        try:
            rmse = compute_yc_rmse(y_gen, self.truth_y)
            return {
                "yc_rmse_mean_bp": float(rmse.mean() * 100),
                "yc_rmse_10y_bp":  float(rmse[7] * 100),   # 10Y tenor
            }
        except Exception:
            return {}

    def make_final_plots(self, trainer) -> None:
        super().make_final_plots(trainer)   # snapshot final state into history
        out = Path(self.output_dir) / "finance"
        y_gen = self._coupling_history[-1]  # always the final trained state
        y_pre = self._coupling_history[0] if self._coupling_history else None

        from .plots.yield_curve import (
            plot_curve_comparison, plot_pca_distribution,
            plot_correlation_heatmaps, plot_yc_coupling_evolution,
        )
        import matplotlib.pyplot as plt

        figs = [
            ("curve_comparison.pdf",
             plot_curve_comparison(self._ref_y, y_gen, self.truth_y, y_pre)),
            ("pca_distribution.pdf",
             plot_pca_distribution(
                 (self._ref_x, self._ref_y),
                 (y_gen, y_gen),
                 (self.truth_x, self.truth_y))),
            ("correlation_heatmaps.pdf",
             plot_correlation_heatmaps(
                 self._ref_y - self._ref_x,
                 y_gen - y_gen,
                 self.truth_y - self.truth_x)),
        ]
        if len(self._coupling_history) > 1:
            figs.append(("coupling_evolution.pdf",
                          plot_yc_coupling_evolution(
                              self._coupling_history, self.truth_y, self._iteration_log)))
        for name, fig in figs:
            if fig is not None:
                fig.savefig(out / name, dpi=300, bbox_inches="tight")
                plt.close(fig)


# ── Risk Factor Visualizer ───────────────────────────────────────────────────

class RiskFactorVisualizer(FinanceVisualizer):
    """Visualizer for E3: Cross-Regime Risk Factor Scenario Generation."""

    def __init__(self, vis_cfg, output_dir, truth_x, truth_y,
                 ref_x, ref_y, wandb_run=None, **kwargs):
        super().__init__(vis_cfg, output_dir, truth_x, truth_y,
                         wandb_run=wandb_run, **kwargs)
        self._ref_x = ref_x
        self._ref_y = ref_y

    def _make_domain_plots(self, iteration, y_gen):
        from .plots.risk_factors import plot_joint_scatter
        try:
            return plot_joint_scatter(self._ref_y, y_gen, self.truth_y)
        except Exception:
            return None

    def _compute_domain_metrics(self, y_gen) -> dict:
        from .plots.risk_factors import compute_frob_error, _portfolio_var
        try:
            frob = compute_frob_error(y_gen, self.truth_y)
            var95, cvar95 = _portfolio_var(y_gen, level=0.05)
            var99, cvar99 = _portfolio_var(y_gen, level=0.01)
            return {
                "corr_frob_err": float(frob),
                "var95": float(-var95 * 100),
                "cvar95": float(-cvar95 * 100),
                "var99": float(-var99 * 100),
            }
        except Exception:
            return {}

    def make_final_plots(self, trainer) -> None:
        super().make_final_plots(trainer)   # snapshot final state into history
        out = Path(self.output_dir) / "finance"
        y_gen = self._coupling_history[-1]  # always the final trained state
        y_pre = self._coupling_history[0] if self._coupling_history else None

        from .plots.risk_factors import (
            plot_joint_scatter, plot_correlation_comparison,
            plot_marginal_distributions, plot_portfolio_var,
            plot_rf_coupling_evolution,
        )
        import matplotlib.pyplot as plt

        figs = [
            ("joint_scatter.pdf",
             plot_joint_scatter(self._ref_y, y_gen, self.truth_y)),
            ("correlation_comparison.pdf",
             plot_correlation_comparison(self._ref_y, y_gen, self.truth_y)),
            ("marginal_distributions.pdf",
             plot_marginal_distributions(self._ref_y, y_gen, self.truth_y, y_pre)),
            ("portfolio_var.pdf",
             plot_portfolio_var(self._ref_y, y_gen, self.truth_y, y_pre)),
        ]
        if len(self._coupling_history) > 1:
            figs.append(("coupling_evolution.pdf",
                          plot_rf_coupling_evolution(
                              self._coupling_history, self.truth_y, self._iteration_log)))
        for name, fig in figs:
            if fig is not None:
                fig.savefig(out / name, dpi=300, bbox_inches="tight")
                plt.close(fig)
