"""Logging visualiser for high-dimensional single-cell experiments.

Extends the standard _LoggingMixin / Visualizer stack with:
  - PCA projection  (fitted lazily on the first display call, reused throughout)
  - Cell-type colouring  (points coloured by their nearest cluster centre, if provided)
  - 50-D-appropriate coupling plots  (no 2-D quiver field)

Saved outputs follow the same directory layout as LoggingVisualizer:
  coupling/    — per-iteration PCA-projected coupling scatter
  marginals/   — source / target marginals + coupling arrows
  eval/        — fixed eval-set comparison vs true target
  weights/     — importance-weight histogram  (inherited)
  metrics.csv  — all scalar metrics           (inherited)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import logging

from dualwcfm.config import VisualizeConfig
from dualwcfm.logging_visualizer import _LoggingMixin
from dualwcfm.visualization import Visualizer

logger = logging.getLogger("dualwcfm")

if TYPE_CHECKING:
    from dualwcfm.trainer import Trainer
    from dualwcfm.distributions.distribution import Distribution


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_CELL_COLORS = [
    "#4878d0",   # type 0 — blue
    "#ee854a",   # type 1 — orange
    "#6acc65",   # type 2 — green
    "#d65f5f",   # type 3 — red
    "#956cb4",   # type 4 — purple
    "#8c613c",   # type 5 — brown
    "#dc7ec0",   # type 6 — pink
    "#797979",   # type 7 — grey
]

_COLOR_SRC  = "#4878d0"   # control / source
_COLOR_TGT  = "#d65f5f"   # treated / target
_COLOR_PRED = "#6acc65"   # model prediction


# ---------------------------------------------------------------------------
# LoggingVisualizerSingleCell
# ---------------------------------------------------------------------------

class LoggingVisualizerSingleCell(_LoggingMixin, Visualizer):
    """Logging visualiser for single-cell / high-dimensional experiments.

    Parameters
    ----------
    config:
        Standard VisualizeConfig.
    transport_map:
        Optional ground-truth T(x) for map-error diagnostics.
        Pass None when using real data (no analytical ground truth).
    output_dir:
        Directory for saved plots and metrics.csv.
    wandb_run:
        Optional W&B run handle.
    inverse_transport_map:
        Optional T⁻¹(y).  Pass None when using real data.
    ref_cell_line:
        Name of the reference cell line (for plot titles).
    new_cell_line:
        Name of the transfer cell line (for plot titles).
    drug_name:
        Drug name shown in plot titles.
    ref_drug_emb:
        Distribution of true drug-treated cells for the reference condition.
        Used to overlay the true target in eval plots.
    new_tgt_dist:
        Distribution of true drug-treated cells for the new cell line.
        Used to overlay the true new target in eval plots.
    n_pcs:
        Number of PCA dimensions in the data (informational).
    cluster_centers:
        (K, d) array.  Kept for back-compat with synthetic experiments; unused
        when real data is loaded.
    cell_type_names:
        List of K names.  Same back-compat note as cluster_centers.
    """

    def __init__(
        self,
        config:                VisualizeConfig,
        transport_map,
        output_dir:            str,
        wandb_run              = None,
        inverse_transport_map  = None,
        # Real-data kwargs (Sci-Plex3)
        ref_cell_line:         Optional[str] = None,
        new_cell_line:         Optional[str] = None,
        drug_name:             Optional[str] = None,
        ref_drug_emb           = None,
        new_tgt_dist           = None,
        n_pcs:                 int = 50,
        # Back-compat synthetic kwargs (ignored when ref/new_cell_line are set)
        cluster_centers:       Optional[np.ndarray] = None,
        cell_type_names:       Optional[List[str]]  = None,
    ):
        Visualizer.__init__(self, config, transport_map)
        self._init_logging(
            output_dir,
            wandb_run             = wandb_run,
            transport_map         = transport_map,
            inverse_transport_map = inverse_transport_map,
        )

        # Real-data metadata
        self._ref_cell_line = ref_cell_line or "Reference"
        self._new_cell_line = new_cell_line or "Transfer"
        self._drug_name     = drug_name     or "Drug"
        self._ref_drug_emb  = ref_drug_emb   # Distribution | None
        self._new_tgt_dist  = new_tgt_dist   # Distribution | None
        self._n_pcs         = n_pcs

        # Synthetic back-compat
        self._cluster_centers = cluster_centers   # (K, d) | None
        self._cell_type_names = list(cell_type_names) if cell_type_names else []

        # PCA state — fitted lazily on the first call to _fit_pca()
        self._pca_comps: Optional[np.ndarray] = None   # (2, d)
        self._pca_mean:  Optional[np.ndarray] = None   # (d,)

    # ── PCA helpers ──────────────────────────────────────────────────────────

    def _fit_pca(self, X: np.ndarray) -> None:
        """Fit a 2-component PCA on *X* (N, d).  No-op after the first call."""
        if self._pca_comps is not None:
            return
        if X.shape[0] < 3 or X.shape[1] < 2:
            return
        mean = X.mean(0)
        Xc   = X - mean
        try:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            self._pca_comps = Vt[:2]
            self._pca_mean  = mean
        except np.linalg.LinAlgError:
            pass  # leave un-fitted; fallback to first-2-dims

    def _project(self, X: np.ndarray) -> np.ndarray:
        """Project (N, d) → (N, 2).  Falls back to dims (0,1) if PCA not ready."""
        if self._pca_comps is None or X.ndim != 2 or X.shape[1] < 2:
            return X[:, :2]
        return (X - self._pca_mean) @ self._pca_comps.T

    def _cell_types(self, X: np.ndarray) -> np.ndarray:
        """Integer cell-type label per sample (nearest cluster centre, or all-zero for real data)."""
        if self._cluster_centers is None or len(self._cluster_centers) == 0:
            return np.zeros(len(X), dtype=int)
        diff  = X[:, None, :] - self._cluster_centers[None, :, :]   # (N, K, d)
        dists = (diff ** 2).sum(-1)                                  # (N, K)
        return dists.argmin(-1)

    def _type_label(self, k: int) -> str:
        if k < len(self._cell_type_names):
            return self._cell_type_names[k]
        return f"type{k}"

    def _type_color(self, k: int) -> str:
        return _CELL_COLORS[k % len(_CELL_COLORS)]

    def _scatter_by_type(
        self,
        ax,
        X2d:    np.ndarray,
        ct:     np.ndarray,
        alpha:  float = 0.40,
        s:      int   = 5,
        prefix: str   = "",
    ) -> None:
        """Scatter coloured by cell type (one call per type for legend entries)."""
        n_types = max(len(self._cell_type_names),
                      int(ct.max()) + 1) if len(ct) else 0
        for k in range(n_types):
            mask = ct == k
            if not mask.any():
                continue
            ax.scatter(
                X2d[mask, 0], X2d[mask, 1],
                c = self._type_color(k),
                s = s, alpha = alpha, linewidths = 0,
                label = f"{prefix}{self._type_label(k)}",
                zorder = 2,
            )

    # ── Coupling display (per-iteration) ─────────────────────────────────────

    def display(self, epoch: int, trainer: "Trainer") -> None:
        if epoch % self.config.show_each != 0:
            return
        backward = self._is_backward(epoch, trainer)
        n  = min(self.config.n_trajectories, 600)
        sc = trainer.scale

        x, y = trainer.current_coupling.coupling_sample(n)
        x, y = x.to(trainer.device), y.to(trainer.device)
        start   = y if backward else x
        ref_tgt = (trainer.new_source if backward else trainer.new_target).sample(n).to(trainer.device)

        traj = trainer.flow_matching.evolve_trajectory(
            start,
            trainer.strategy.get_flow(epoch, trainer.device)[0],
            num_steps = 25, sigma = trainer._sigma, extend = True,
        )
        d_data = start.shape[1]
        traj   = traj[:, :, :d_data]
        end    = traj[-1]

        start_raw = (start   * sc).cpu().numpy()
        end_raw   = (end     * sc).cpu().numpy()
        ref_raw   = (ref_tgt * sc).cpu().numpy()
        traj_raw  = (traj    * sc).cpu().numpy()   # (T, n, d)

        self._fit_pca(np.vstack([start_raw, end_raw, ref_raw]))

        start_2d = self._project(start_raw)
        end_2d   = self._project(end_raw)
        ref_2d   = self._project(ref_raw)
        ct_s     = self._cell_types(start_raw)
        ct_e     = self._cell_types(end_raw)
        ct_r     = self._cell_types(ref_raw)

        self._lock_axis_lim(
            torch.tensor(start_2d), torch.tensor(end_2d), torch.tensor(ref_2d),
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        self._scatter_by_type(axes[0], start_2d, ct_s, alpha=0.5, s=5)
        axes[0].set_title(f"{'ν' if backward else 'μ'} coupling (iter {epoch})")

        # Trajectory lines in PCA space (subsample to avoid overplotting)
        stride   = max(1, self.config.trajectory_stride)
        n_tlines = min(n, 60)
        for j in range(0, n, max(1, n // n_tlines)):
            pts = self._project(traj_raw[:, j, :])
            axes[1].plot(
                pts[::stride, 0], pts[::stride, 1],
                color="#aaaaaa", alpha=0.15, lw=0.5, zorder=1,
            )
        self._scatter_by_type(axes[1], end_2d, ct_e, alpha=0.5, s=5)
        axes[1].set_title(f"Evolved (iter {epoch})")

        self._scatter_by_type(axes[2], ref_2d, ct_r, alpha=0.5, s=5)
        axes[2].set_title(f"{'μ' if backward else 'ν'} target (iter {epoch})")

        for ax in axes:
            ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
            ax.legend(fontsize=5, ncol=2, loc="upper right")
            ax.set_aspect("equal")
            self._apply_axis_lim(ax)

        direction = "Backward ν→μ" if backward else "Forward μ→ν"
        fig.suptitle(f"Single-Cell Transport — Iter {epoch} ({direction})", fontsize=10)
        fig.tight_layout()
        self._accumulate(fig, "coupling")
        self._save_fig(fig, "coupling", f"coupling_{epoch:04d}.png")

    # ── Postprocess (coupling-arrow plot) ────────────────────────────────────

    def at_postprocess_end(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        if (
            (iteration + 1) % self.config.show_each != 0
            and iteration != trainer.config.iterations - 1
        ):
            return

        sc     = trainer.scale
        device = trainer.device
        n_coup = min(200, getattr(trainer.current_coupling, "size", 200))
        n_samp = 500
        t_val  = (iteration + 1) / trainer.config.iterations
        t_tens = torch.full((n_samp // 10, 1), t_val, device=device)

        with torch.no_grad():
            mu_s = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_init(n_samp // 10).to(device),
                trainer.new_source.sample(n_samp // 10).to(device),
                t_tens, 0.0,
            )
            nu_s = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_target(n_samp // 10).to(device),
                trainer.new_target.sample(n_samp // 10).to(device),
                t_tens, 0.0,
            )

        xc, yc   = trainer.current_coupling.coupling_sample(n_coup)
        xc_raw   = (xc * sc).cpu().numpy()
        yc_raw   = (yc * sc).cpu().numpy()
        mu_raw   = (mu_s * sc).cpu().numpy()
        nu_raw   = (nu_s * sc).cpu().numpy()

        all_raw  = np.vstack([xc_raw, yc_raw, mu_raw, nu_raw])
        self._fit_pca(all_raw)

        xc_2d  = self._project(xc_raw)
        yc_2d  = self._project(yc_raw)
        mu_2d  = self._project(mu_raw)
        nu_2d  = self._project(nu_raw)
        ct_xc  = self._cell_types(xc_raw)
        ct_yc  = self._cell_types(yc_raw)

        fig, ax = plt.subplots(figsize=(7, 6))

        ax.scatter(mu_2d[:, 0], mu_2d[:, 1], c="#4878d0", s=3, alpha=0.18,
                   linewidths=0, label=r"$\mu_t$ (theory)", zorder=1)
        ax.scatter(nu_2d[:, 0], nu_2d[:, 1], c="#c44e52", s=3, alpha=0.18,
                   linewidths=0, label=r"$\nu_t$ (theory)", zorder=1)

        # Coupling arrows coloured by source cell type
        for j in range(n_coup):
            col = self._type_color(int(ct_xc[j]))
            ax.annotate(
                "", xy=yc_2d[j], xytext=xc_2d[j],
                arrowprops=dict(
                    arrowstyle="-|>", color=col, alpha=0.45, lw=0.7,
                    mutation_scale=6,
                ),
                zorder=3,
            )

        self._scatter_by_type(ax, xc_2d, ct_xc, alpha=0.85, s=9, prefix="src:")
        self._scatter_by_type(ax, yc_2d, ct_yc, alpha=0.85, s=9, prefix="tgt:")

        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.legend(fontsize=6, ncol=2, loc="upper right")
        ax.set_aspect("equal")
        ax.set_title(f"Coupling step {iteration + 1}")
        fig.tight_layout()
        self.save_fig(fig, "marginals", f"distributions_{iteration:04d}.png")

    # ── Marginals plot ────────────────────────────────────────────────────────

    def _save_marginals_plot(
        self,
        time_step: int,
        mu_t:   torch.Tensor,
        nu_t:   torch.Tensor,
        x_coup: torch.Tensor,
        y_coup: torch.Tensor,
    ) -> None:
        if mu_t.dim() != 2 or mu_t.shape[0] == 0:
            return

        mu_raw  = mu_t.numpy()
        nu_raw  = nu_t.numpy()
        xc_raw  = x_coup.numpy()
        yc_raw  = y_coup.numpy()

        self._fit_pca(np.vstack([mu_raw, nu_raw, xc_raw, yc_raw]))

        mu_2d  = self._project(mu_raw);  ct_mu = self._cell_types(mu_raw)
        nu_2d  = self._project(nu_raw);  ct_nu = self._cell_types(nu_raw)
        xc_2d  = self._project(xc_raw);  ct_xc = self._cell_types(xc_raw)
        yc_2d  = self._project(yc_raw);  ct_yc = self._cell_types(yc_raw)

        self._lock_axis_lim(
            torch.tensor(mu_2d), torch.tensor(nu_2d),
            torch.tensor(xc_2d), torch.tensor(yc_2d),
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        kw = dict(s=4, alpha=0.35)

        # Source
        self._scatter_by_type(axes[0], mu_2d, ct_mu, **kw)
        axes[0].scatter(xc_2d[:, 0], xc_2d[:, 1], c="darkorange", s=4, alpha=0.35,
                        linewidths=0, label=r"$\pi_t^{(1)}$ coupling", zorder=3)
        axes[0].set_title(f"Source marginal — iter {time_step}")
        axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2")
        axes[0].legend(fontsize=6, ncol=2); axes[0].set_aspect("equal")
        self._apply_axis_lim(axes[0])

        # Target
        self._scatter_by_type(axes[1], nu_2d, ct_nu, **kw)
        axes[1].scatter(yc_2d[:, 0], yc_2d[:, 1], c="limegreen", s=4, alpha=0.35,
                        linewidths=0, label=r"$\pi_t^{(2)}$ coupling", zorder=3)
        axes[1].set_title(f"Target marginal — iter {time_step}")
        axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
        axes[1].legend(fontsize=6, ncol=2); axes[1].set_aspect("equal")
        self._apply_axis_lim(axes[1])

        fig.tight_layout()
        self._accumulate(fig, "marginals")
        self._save_fig(fig, "marginals", f"marginals_{time_step:04d}.png")

    # ── Eval scatter ─────────────────────────────────────────────────────────

    def _save_eval_scatter(
        self,
        time_step: int,
        src:    torch.Tensor,
        evol:   torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        if src.dim() != 2 or src.shape[0] == 0:
            return

        src_raw    = src.numpy()
        evol_raw   = evol.numpy()
        target_raw = target.numpy()

        # Optionally overlay the true new target from the real distribution
        true_tgt_raw: Optional[np.ndarray] = None
        if self._new_tgt_dist is not None:
            try:
                n_true = min(len(src_raw), 500)
                true_tgt_raw = self._new_tgt_dist.sample(n_true).cpu().numpy()
            except Exception:
                pass

        stacks = [src_raw, evol_raw, target_raw]
        if true_tgt_raw is not None:
            stacks.append(true_tgt_raw)
        self._fit_pca(np.vstack(stacks))

        src_2d    = self._project(src_raw)
        evol_2d   = self._project(evol_raw)
        target_2d = self._project(target_raw)

        n_panels = 3 if true_tgt_raw is not None else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        kw = dict(s=5, alpha=0.55, linewidths=0)

        # Panel 0: source → evolved
        axes[0].scatter(src_2d[:, 0],  src_2d[:, 1],  c=_COLOR_SRC,  label=f"{self._new_cell_line} control", **kw)
        axes[0].scatter(evol_2d[:, 0], evol_2d[:, 1], c=_COLOR_PRED, label="Predicted treated", **kw)
        axes[0].set_title(f"{self._new_cell_line}: control → predicted (iter {time_step})")
        axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2")
        axes[0].legend(fontsize=7); axes[0].set_aspect("equal")
        self._apply_axis_lim(axes[0])

        # Panel 1: predicted vs coupling target
        axes[1].scatter(evol_2d[:, 0],   evol_2d[:, 1],   c=_COLOR_PRED, label="Predicted", **kw)
        axes[1].scatter(target_2d[:, 0], target_2d[:, 1], c=_COLOR_TGT,  label="Coupling target", alpha=0.25, s=5, linewidths=0)
        axes[1].set_title(f"Predicted vs coupling target (iter {time_step})")
        axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
        axes[1].legend(fontsize=7); axes[1].set_aspect("equal")
        self._apply_axis_lim(axes[1])

        # Panel 2 (optional): predicted vs true marginal
        if true_tgt_raw is not None:
            true_2d = self._project(true_tgt_raw)
            axes[2].scatter(evol_2d[:, 0],  evol_2d[:, 1],  c=_COLOR_PRED, label="Predicted", **kw)
            axes[2].scatter(true_2d[:, 0],  true_2d[:, 1],  c=_COLOR_TGT,  label=f"True {self._new_cell_line} treated", alpha=0.4, s=5, linewidths=0)
            axes[2].set_title(f"Predicted vs true target (iter {time_step})")
            axes[2].set_xlabel("PC 1"); axes[2].set_ylabel("PC 2")
            axes[2].legend(fontsize=7); axes[2].set_aspect("equal")
            self._apply_axis_lim(axes[2])

        fig.suptitle(
            f"{self._drug_name}: {self._ref_cell_line}→{self._new_cell_line} transfer — iter {time_step}",
            fontsize=10,
        )
        fig.tight_layout()
        self._accumulate(fig, "eval")
        self._save_fig(fig, "eval", f"eval_{time_step:04d}.png")

        if self._wandb_run:
            try:
                import PIL.Image
                png_path = Path(self.output_dir) / "eval" / f"eval_{time_step:04d}.png"
                self._wandb_run.log_image(
                    "eval/transport", [PIL.Image.open(str(png_path))], step=time_step
                )
            except Exception:
                pass

    # ── Duals end — PCA scatter coloured by log-weight ───────────────────────

    def at_duals_end(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        if ctx.get("is_image", False):
            return
        x0 = ctx.get("x0")
        x1 = ctx.get("x1")
        if x0 is None or x0.dim() < 2 or x0.shape[0] == 0:
            return

        sc     = trainer.scale
        log_w  = ctx["log_w"].detach().cpu().numpy()
        x0_raw = (x0 * sc).cpu().numpy()
        x1_raw = (x1 * sc).cpu().numpy()

        self._fit_pca(np.vstack([x0_raw, x1_raw]))
        x0_2d = self._project(x0_raw)
        x1_2d = self._project(x1_raw)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        sc_obj = axes[0].scatter(x0_2d[:, 0], x0_2d[:, 1], c=log_w,
                                  cmap="viridis", edgecolors="none", alpha=0.7, s=6)
        axes[0].set_title(r"Source $x_0$ — $(a+b)/\varepsilon$")
        axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2")
        axes[0].set_aspect("equal")

        axes[1].scatter(x1_2d[:, 0], x1_2d[:, 1], c=log_w,
                       cmap="viridis", edgecolors="none", alpha=0.7, s=6)
        axes[1].set_title(r"Target $x_1$ — $(a+b)/\varepsilon$")
        axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
        axes[1].set_aspect("equal")

        fig.colorbar(sc_obj, ax=axes[1], fraction=0.046, pad=0.04).set_label(
            r"$(a+b)/\varepsilon$"
        )
        plt.tight_layout()
        self.save_fig(fig, "marginals", f"duals_{iteration:04d}.png")

    # ── Skip CFM quiver (not meaningful in 10-D) ─────────────────────────────

    def at_flow_matching_start(self, iteration: int, trainer: "Trainer", ctx: dict) -> dict:
        return {}

    def at_flow_matching_end(
        self, iteration: int, trainer: "Trainer", ctx: dict, start_ctx: dict
    ) -> None:
        pass
