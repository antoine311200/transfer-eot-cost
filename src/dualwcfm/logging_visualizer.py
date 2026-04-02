"""Logging visualizers that save per-iteration plots and metrics to disk.

Each iteration's key diagnostics are saved as PNG files under `output_dir/`:
  coupling/coupling_{iter}.png   — flow trajectories
  cfm/cfm_{iter}.png            — CFM vector field quiver (2D only)
  marginals/marginals_{iter}.png — mu_t / nu_t vs coupling marginals
  marginals/duals_{iter}.png    — coupling colored by importance weight
  weights/weights_{iter}.png    — importance weight histogram
  metrics.csv                   — scalar metrics per iteration

After training, timelapse_{key}.gif is saved for each plot type.
All plots of the same type use locked axis limits so frames are consistent.
"""
from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import logging

from dualwcfm.config import VisualizeConfig
from dualwcfm.strategy import BidirectionalStrategy
from dualwcfm.utils import is_image_tensor
from dualwcfm.visualization import Visualizer, Visualizer3D, VisualizerImage

logger = logging.getLogger("dualwcfm")

if TYPE_CHECKING:
    from dualwcfm.trainer import Trainer


# ---------------------------------------------------------------------------
# Shared mixin
# ---------------------------------------------------------------------------

class _LoggingMixin:
    """Adds disk-saving, CSV logging, per-type GIF timelapse, and locked axis
    limits (so all frames of a given plot type have identical boundaries)."""

    def _init_logging(self, output_dir: str, wandb_run=None,
                      transport_map=None, inverse_transport_map=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("coupling", "cfm", "weights", "marginals", "pretrain", "eval"):
            (self.output_dir / sub).mkdir(exist_ok=True)

        self._wandb_run = wandb_run  # may be None or a WandbRun instance
        self._transport_map         = transport_map          # T(x) ≈ y
        self._inverse_transport_map = inverse_transport_map  # T⁻¹(y) ≈ x

        # Fixed eval set — lazily initialised on first analyse() call
        self._eval_source: Optional[torch.Tensor] = None
        self._eval_target: Optional[torch.Tensor] = None

        self._csv_path = self.output_dir / "metrics.csv"
        self._csv_fields = [
            "iteration",
            # Sliced-Wasserstein marginal distances
            "W1_mu", "W2_mu", "W1_nu", "W2_nu",
            # Non-parametric target-marginal divergences
            "mmd_nu",            # Maximum Mean Discrepancy (RBF kernel, median BW)
            "energy_dist_nu",    # Energy distance (kernel-free)
            "sinkhorn_div",      # Debiased Sinkhorn divergence (W2 proxy, distribution-only)
            # Ground-truth transport map errors (present only when T is known)
            "map_err_fwd",       # sqrt(E[||T(x) - y||²])   forward direction
            "map_err_bwd",       # sqrt(E[||T⁻¹(y) - x||²]) backward direction
            # Gushchin coupling quality
            "bw2",               # BW₂ between learned joint coupling and reference joint coupling
            # Image-specific quality (empty for non-image experiments)
            "fid", "kid",
            # Importance-weight diagnostics
            "weight_mean", "weight_std", "weight_min", "weight_max",
            "weight_ess",        # Effective Sample Size (normalised to [0,1])
        ]
        with open(self._csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writeheader()

        self._frames: dict[str, List[np.ndarray]] = defaultdict(list)
        self._axis_lim: Optional[Tuple[float, float, float, float]] = None

    # ------------------------------------------------------------------
    # Axis-limit helpers (2-D only)
    # ------------------------------------------------------------------

    def _lock_axis_lim(self, *tensors: torch.Tensor, padding: float = 0.18):
        if self._axis_lim is not None:
            return
        valid = [t for t in tensors if isinstance(t, torch.Tensor)
                 and t.dim() == 2 and t.shape[1] >= 2]
        if not valid:
            return
        pts = torch.cat([t[:, :2] for t in valid], dim=0)
        xmin, xmax = pts[:, 0].min().item(), pts[:, 0].max().item()
        ymin, ymax = pts[:, 1].min().item(), pts[:, 1].max().item()
        dx = max((xmax - xmin) * padding, 0.5)
        dy = max((ymax - ymin) * padding, 0.5)
        self._axis_lim = (xmin - dx, xmax + dx, ymin - dy, ymax + dy)

    def _apply_axis_lim(self, ax):
        if self._axis_lim is None:
            return
        ax.set_xlim(self._axis_lim[0], self._axis_lim[1])
        ax.set_ylim(self._axis_lim[2], self._axis_lim[3])

    # ------------------------------------------------------------------
    # Frame accumulation
    # ------------------------------------------------------------------

    def _accumulate(self, fig: plt.Figure, key: str):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=60, bbox_inches="tight")
        buf.seek(0)
        try:
            import PIL.Image
            arr = np.array(PIL.Image.open(buf).convert("RGB"))
            self._frames[key].append(arr)
        except ImportError:
            pass
        buf.close()

    # ------------------------------------------------------------------
    # Override save_fig: accumulate then save
    # ------------------------------------------------------------------

    def save_fig(self, fig: plt.Figure, subfolder: str, name: str):
        key = "duals" if (subfolder == "marginals" and name.startswith("duals")) else subfolder
        self._accumulate(fig, key)
        self._save_fig(fig, subfolder, name)

    # ------------------------------------------------------------------
    # Override analyse(): compute weights + save marginals + write CSV
    # ------------------------------------------------------------------

    def analyse(self, time_step: int, trainer: "Trainer",
                n_samples: int = 2000, n_proj: int = 2000) -> dict:
        distances = super().analyse(time_step, trainer, n_samples, n_proj)  # type: ignore[misc]

        device = trainer.device
        n = min(n_samples, trainer.config.batch_size)
        t = time_step / trainer.config.iterations
        t_tensor = torch.full((n, 1), t, device=device)
        s = trainer.scale

        # Determine training direction (for BidirectionalStrategy only)
        is_bidir = isinstance(trainer.strategy, BidirectionalStrategy)
        is_bwd = is_bidir and self._is_backward(time_step, trainer)

        # Lazy-init fixed eval set (same samples across all iterations)
        if self._eval_source is None:
            n_eval = min(n_samples, 512)
            with torch.no_grad():
                self._eval_source = trainer.new_source.sample(n_eval).cpu()
                self._eval_target = trainer.new_target.sample(n_eval).cpu()

        with torch.no_grad():
            mu_t = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_init(n).to(device),
                trainer.new_source.sample(n).to(device),
                t_tensor, sigma=0.0)
            nu_t = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_target(n).to(device),
                trainer.new_target.sample(n).to(device),
                t_tensor, sigma=0.0)
            x, y = trainer.current_coupling.coupling_sample(n)
            x, y = x.to(device), y.to(device)
            log_w = trainer.strategy.log_weight(x, y, trainer.config.iterations)
            w = torch.exp(log_w)

        w_np = w.cpu().float().numpy().ravel()

        # Effective Sample Size (Kish, 1965): ESS = (Σw)² / Σ(w²), normalised by N
        w_sum  = float(w_np.sum())
        w_sum2 = float((w_np ** 2).sum())
        ess_norm = (w_sum ** 2 / (w_sum2 * len(w_np))) if w_sum2 > 0 else 0.0

        # Non-parametric distributional metrics on the target marginal
        from dualwcfm.metrics import (
            mmd as _mmd, energy_distance as _ed,
            sinkhorn_divergence as _sd, map_error as _map_err,
            joint_bw2 as _joint_bw2,
        )
        nu_ref = trainer.new_target.sample(n).to(device)
        if is_image_tensor(y):
            nu_ref = nu_ref.view_as(y)
        # Rescale to original data space for cross-run comparability
        y_scaled      = y      * s
        nu_ref_scaled = nu_ref * s
        mmd_nu         = _mmd(y_scaled, nu_ref_scaled)
        energy_dist_nu = _ed(y_scaled,  nu_ref_scaled)
        sinkhorn_div   = _sd(y_scaled,  nu_ref_scaled)

        # Ground-truth map errors — only compute for the direction trained this step.
        # For BidirectionalStrategy: skip fwd error on backward steps and vice versa
        # to avoid flat segments in W&B graphs from an untrained model.
        n_map = min(n, 2_000)
        fwd_model = getattr(trainer.strategy, "forward_model",  None)
        bwd_model = getattr(trainer.strategy, "backward_model", None)

        compute_fwd = (not is_bidir) or (not is_bwd)
        compute_bwd = (not is_bidir) or is_bwd

        if compute_fwd and self._transport_map is not None and fwd_model is not None:
            x_eval = trainer.new_source.sample(n_map).to(device)
            with torch.no_grad():
                evolved = trainer.flow_matching.evolve(
                    x_eval, fwd_model, num_steps=200, sigma=0.0, extend=True)
                y_hat = (evolved[:, : x_eval.shape[1], :, :]
                         if is_image_tensor(x_eval)
                         else evolved[:, : trainer.strategy.config.dimension])
            map_err_fwd = _map_err((x_eval * s).cpu(), (y_hat * s).cpu(), self._transport_map)
        else:
            map_err_fwd = ""

        if compute_bwd and self._inverse_transport_map is not None and bwd_model is not None:
            y_eval = trainer.new_target.sample(n_map).to(device)
            with torch.no_grad():
                evolved = trainer.flow_matching.evolve(
                    y_eval, bwd_model, num_steps=200, sigma=0.0, extend=True)
                x_hat = (evolved[:, : y_eval.shape[1], :, :]
                         if is_image_tensor(y_eval)
                         else evolved[:, : trainer.strategy.config.dimension])
            map_err_bwd = _map_err((y_eval * s).cpu(), (x_hat * s).cpu(), self._inverse_transport_map)
        else:
            map_err_bwd = ""

        # Gushchin BW₂: joint Bures-Wasserstein distance between the learned
        # coupling and the ground-truth coupling π*(μ', ν') = {(x, T(x)) : x~μ'}.
        # Requires a known transport map; skipped for images or when unavailable.
        bw2_val: "str | float" = ""
        if not is_image_tensor(x) and self._transport_map is not None:
            n_bw = min(n, 2_000)
            with torch.no_grad():
                x_bw, y_bw = trainer.current_coupling.coupling_sample(n_bw)
                x_gt = trainer.new_source.sample(n_bw).to(device)
                y_gt = self._transport_map((x_gt * s).cpu()).to(x_gt.device) / s
            bw2_val = _joint_bw2(
                (x_bw * s).float().cpu(), (y_bw * s).float().cpu(),
                (x_gt * s).float().cpu(), (y_gt * s).float().cpu(),
            )

        # Image-specific metrics (FID / KID) — provided by subclass hook
        img_metrics = self._compute_image_metrics(time_step, trainer)
        fid_val = img_metrics.get("fid", "")
        kid_val = img_metrics.get("kid", "")

        row = {
            "iteration":      time_step,
            **distances,
            "mmd_nu":         mmd_nu,
            "energy_dist_nu": energy_dist_nu,
            "sinkhorn_div":   sinkhorn_div,
            "map_err_fwd":    map_err_fwd,
            "map_err_bwd":    map_err_bwd,
            "bw2":            bw2_val,
            "fid":            fid_val,
            "kid":            kid_val,
            "weight_mean":    float(w_np.mean()),
            "weight_std":     float(w_np.std()),
            "weight_min":     float(w_np.min()),
            "weight_max":     float(w_np.max()),
            "weight_ess":     ess_norm,
        }
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writerow(row)

        # W&B logging — direction-aware: only log metrics for the direction that
        # was actually trained this step to avoid flat "no-op" segments on graphs.
        if self._wandb_run:
            wb: dict = {}

            # Direction-independent metrics
            wb["mmd_nu"]         = mmd_nu
            wb["energy_dist_nu"] = energy_dist_nu
            wb["sinkhorn_div"]   = sinkhorn_div
            wb["weight_mean"]    = float(w_np.mean())
            wb["weight_std"]     = float(w_np.std())
            wb["weight_ess"]     = ess_norm
            if bw2_val != "":
                wb["bw2"] = bw2_val
            if fid_val != "":
                wb["fid"] = fid_val
            if kid_val != "":
                wb["kid"] = kid_val

            # Forward metrics: only log on forward steps
            if compute_fwd:
                wb["W1_nu"] = distances["W1_nu"]
                wb["W2_nu"] = distances["W2_nu"]
                if map_err_fwd != "":
                    wb["map_err_fwd"] = map_err_fwd

            # Backward metrics: only log on backward steps
            if compute_bwd:
                wb["W1_mu"] = distances["W1_mu"]
                wb["W2_mu"] = distances["W2_mu"]
                if map_err_bwd != "":
                    wb["map_err_bwd"] = map_err_bwd

            self._wandb_run.log(wb, step=time_step)

        self._save_weight_histogram(time_step, w_np)
        self._save_marginals_plot(
            time_step,
            (mu_t * s).cpu(), (nu_t * s).cpu(),
            (x * s).cpu(),    (y * s).cpu(),
        )

        # Eval-set plots: show fixed source samples evolved through current fwd model
        if time_step % self.config.show_each == 0 or time_step == trainer.config.iterations - 1:
            self._save_eval_plots(time_step, trainer)

        return distances

    # ------------------------------------------------------------------
    # Hook for subclasses to provide image-specific metrics
    # ------------------------------------------------------------------

    def _compute_image_metrics(self, time_step: int, trainer: "Trainer") -> dict:
        """Return dict of image metrics (e.g. ``{"fid": ..., "kid": ...}``).

        Default implementation returns empty dict.  Override in
        :class:`LoggingVisualizerImage` to compute FID / KID periodically.
        """
        return {}

    # ------------------------------------------------------------------
    # Eval-set plots
    # ------------------------------------------------------------------

    def _save_eval_plots(self, time_step: int, trainer: "Trainer") -> None:
        """Evolve the fixed eval source samples and save comparison plots.

        The eval set is created once (first call) and reused across iterations
        so plots are directly comparable frame-to-frame.
        """
        if self._eval_source is None:
            return
        device = trainer.device
        fwd_model = trainer.strategy.get_flow(0, device)[0]
        with torch.no_grad():
            x_in = self._eval_source.to(device)
            evolved = trainer.flow_matching.evolve(
                x_in, fwd_model, num_steps=100, sigma=0.0, extend=True,
            )
            if is_image_tensor(evolved):
                C = x_in.shape[1]
                y_hat = evolved[:, :C, :, :]
            else:
                d = self._eval_source.shape[1]
                y_hat = evolved[:, :d]

        s = trainer.scale
        src    = (x_in   * s).cpu()
        evol   = (y_hat  * s).cpu()
        target = (self._eval_target * s).cpu()

        if is_image_tensor(y_hat):
            self._save_eval_image_grid(time_step, src, evol, target)
        else:
            self._save_eval_scatter(time_step, src, evol, target)

    def _save_eval_scatter(self, time_step: int,
                           src: torch.Tensor,
                           evol: torch.Tensor,
                           target: torch.Tensor) -> None:
        """2-D / projected scatter plot for fixed eval samples."""
        if src.dim() != 2 or src.shape[1] < 2:
            return
        kw = dict(s=5, alpha=0.55, linewidths=0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(src[:, 0],  src[:, 1],  c="royalblue", label="Eval source",   **kw)
        axes[0].scatter(evol[:, 0], evol[:, 1], c="limegreen", label="Evolved (fwd)", **kw)
        axes[0].set_title(f"Source → Evolved (iter {time_step})")
        axes[0].legend(fontsize=7); axes[0].set_aspect("equal")
        self._apply_axis_lim(axes[0])

        axes[1].scatter(evol[:, 0],   evol[:, 1],   c="limegreen", label="Evolved (fwd)", **kw)
        axes[1].scatter(target[:, 0], target[:, 1], c="firebrick",  label="Eval target",  **kw)
        axes[1].set_title(f"Evolved vs Target (iter {time_step})")
        axes[1].legend(fontsize=7); axes[1].set_aspect("equal")
        self._apply_axis_lim(axes[1])

        fig.suptitle(f"Eval samples — fixed set, iter {time_step}", fontsize=10)
        fig.tight_layout()
        self._accumulate(fig, "eval")
        self._save_fig(fig, "eval", f"eval_{time_step:04d}.png")

        if self._wandb_run:
            try:
                import PIL.Image
                png_path = self.output_dir / "eval" / f"eval_{time_step:04d}.png"
                self._wandb_run.log_image("eval/transport", [PIL.Image.open(str(png_path))], step=time_step)
            except Exception:
                pass

    def _save_eval_image_grid(self, time_step: int,
                              src: torch.Tensor,
                              evol: torch.Tensor,
                              target: torch.Tensor) -> None:
        """3-row image grid (source / evolved / target) for fixed eval samples."""
        n = min(8, src.shape[0])
        C = src.shape[1]
        H, W = src.shape[2], src.shape[3]
        cmap = "gray" if C == 1 else None

        def _to_hw(t, i):
            img = t[i].float().numpy()
            return img.squeeze(0) if C == 1 else img.transpose(1, 2, 0)

        fig, axes = plt.subplots(3, n, figsize=(2 * n, 7))
        if n == 1:
            axes = axes.reshape(3, 1)

        for row_idx, (label, tensor) in enumerate(
                [("Source", src), ("Evolved (fwd)", evol), ("Target", target)]):
            for col in range(n):
                axes[row_idx, col].imshow(_to_hw(tensor, col), cmap=cmap, vmin=-1, vmax=1)
                axes[row_idx, col].axis("off")
            axes[row_idx, 0].set_ylabel(label, fontsize=8)

        fig.suptitle(f"Eval samples — fixed set, iter {time_step}", fontsize=9)
        plt.tight_layout()
        self._accumulate(fig, "eval")
        self._save_fig(fig, "eval", f"eval_{time_step:04d}.png")

        if self._wandb_run:
            try:
                import PIL.Image
                png_path = self.output_dir / "eval" / f"eval_{time_step:04d}.png"
                self._wandb_run.log_image("eval/transport", [PIL.Image.open(str(png_path))], step=time_step)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Duals lifecycle hook
    # ------------------------------------------------------------------

    def at_duals_end(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        """2-D scatter plot coloured by log-weight."""
        if ctx.get("is_image", False):
            return
        x0 = ctx["x0"]
        x1 = ctx["x1"]
        log_w = ctx["log_w"]
        x0_np = x0.cpu().numpy()
        x1_np = x1.cpu().numpy()
        sm_np = log_w.detach().cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        cmap = "viridis"
        axes[0].scatter(x0_np[:, 0], x0_np[:, 1], c=sm_np, cmap=cmap, edgecolors="k", alpha=0.7)
        axes[0].set_title(r"Source $x_0$ coloured by $(a+b)/\epsilon$")
        axes[0].set_xlabel("Dim 1"); axes[0].set_ylabel("Dim 2")
        axes[0].set_aspect("equal", adjustable="box")
        sc = axes[1].scatter(x1_np[:, 0], x1_np[:, 1], c=sm_np, cmap=cmap, edgecolors="k", alpha=0.7)
        axes[1].set_title(r"Target $x_1$ coloured by $(a+b)/\epsilon$")
        axes[1].set_xlabel("Dim 1"); axes[1].set_ylabel("Dim 2")
        axes[1].set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04).set_label(r"$(a+b)/\epsilon$")
        plt.tight_layout()
        self.save_fig(fig, "marginals", f"duals_{iteration:04d}.png")

    # ------------------------------------------------------------------
    # Flow matching lifecycle hooks
    # ------------------------------------------------------------------

    def at_flow_matching_start(self, iteration: int, trainer: "Trainer", ctx: dict) -> dict:
        """Capture vecs_before for the quiver comparison plot."""
        if ctx.get("is_image", False) or ctx.get("x0_quiver") is None:
            return {}
        model = ctx["model"]
        x0_quiver = ctx["x0_quiver"]
        n = x0_quiver.shape[0]
        t_zero = torch.zeros((n, 1), device=x0_quiver.device)
        with torch.no_grad():
            vecs_before = model(t_zero, x0_quiver, x0_quiver).cpu().numpy()
        return {"vecs_before": vecs_before}

    def at_flow_matching_end(self, iteration: int, trainer: "Trainer", ctx: dict, start_ctx: dict) -> None:
        """Quiver before/after comparison plot (2-D only)."""
        if ctx.get("is_image", False) or ctx.get("x0_quiver") is None:
            return
        vecs_before = start_ctx.get("vecs_before")
        if vecs_before is None:
            return
        model = ctx["model"]
        x0_quiver = ctx["x0_quiver"]
        x0_samp = ctx["x0_samp"]
        x1_samp = ctx["x1_samp"]
        x1_target = ctx["x1_target"]
        n = x0_quiver.shape[0]
        t_zero = torch.zeros((n, 1), device=x0_quiver.device)
        with torch.no_grad():
            vecs_after = model(t_zero, x0_quiver, x0_quiver).cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, vecs, title in [
            (axes[0], vecs_before, f"t=0 vector field — before training (step {iteration})"),
            (axes[1], vecs_after,  f"t=0 vector field — after training (step {iteration})"),
        ]:
            qx = x0_quiver[:, 0].cpu().numpy()
            qy = x0_quiver[:, 1].cpu().numpy()
            mag = np.sqrt(vecs[:, 0]**2 + vecs[:, 1]**2) + 1e-8
            ax.scatter(x1_target[:, 0].cpu(), x1_target[:, 1].cpu(),
                       s=6, alpha=0.8, c="greenyellow", label=r"$\nu_{t,\mathrm{target}}$", zorder=2)
            ax.scatter(x0_samp[:, 0].cpu(), x0_samp[:, 1].cpu(),
                       s=6, alpha=0.3, c="steelblue", label=r"$\mu_t$", zorder=2)
            ax.scatter(x1_samp[:, 0].cpu(), x1_samp[:, 1].cpu(),
                       s=6, alpha=0.3, c="tomato", label=r"$\nu_t$", zorder=2)
            q = ax.quiver(qx, qy, vecs[:, 0], vecs[:, 1], mag, cmap="plasma", alpha=0.9, zorder=3)
            plt.colorbar(q, ax=ax, label="|v|", shrink=0.8)
            ax.set_title(title, fontsize=10)
            ax.legend(loc="upper right", markerscale=2, fontsize=8)
            ax.set_aspect("equal")
        fig.suptitle(rf"CFM vector field $\mu_t \to \nu_t$ at $t=0$ (step {iteration})", fontsize=12)
        plt.tight_layout()
        self.save_fig(fig, "cfm", f"cfm_{iteration:04d}.png")

    # ------------------------------------------------------------------
    # Per-iteration plot helpers
    # ------------------------------------------------------------------

    def _save_marginals_plot(self, time_step: int,
                             mu_t: torch.Tensor, nu_t: torch.Tensor,
                             x_coup: torch.Tensor, y_coup: torch.Tensor):
        if mu_t.dim() != 2:
            return
        self._lock_axis_lim(mu_t, nu_t, x_coup, y_coup)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        kw = dict(s=4, alpha=0.35, linewidths=0)
        axes[0].scatter(mu_t[:, 0],   mu_t[:, 1],   c="royalblue",  label=r"$\mu_t$ (theory)",        **kw)
        axes[0].scatter(x_coup[:, 0], x_coup[:, 1], c="darkorange", label=r"$\pi_t^{(1)}$ (coupling)", **kw)
        axes[0].legend(fontsize=7); axes[0].set_aspect("equal")
        axes[0].set_title(f"Source marginal — iter {time_step}")
        self._apply_axis_lim(axes[0])
        axes[1].scatter(nu_t[:, 0],   nu_t[:, 1],   c="firebrick",  label=r"$\nu_t$ (theory)",        **kw)
        axes[1].scatter(y_coup[:, 0], y_coup[:, 1], c="limegreen",  label=r"$\pi_t^{(2)}$ (coupling)", **kw)
        axes[1].legend(fontsize=7); axes[1].set_aspect("equal")
        axes[1].set_title(f"Target marginal — iter {time_step}")
        self._apply_axis_lim(axes[1])
        fig.tight_layout()
        self._accumulate(fig, "marginals")
        self._save_fig(fig, "marginals", f"marginals_{time_step:04d}.png")

    def _save_weight_histogram(self, time_step: int, w: np.ndarray):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(w, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="w=1")
        ax.set_xlabel("Weight $w_k$"); ax.set_ylabel("Count")
        ax.set_title(f"Weight distribution — iter {time_step}")
        ax.legend()
        fig.tight_layout()
        self._accumulate(fig, "weights")
        self._save_fig(fig, "weights", f"weights_{time_step:04d}.png")

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _save_fig(self, fig: plt.Figure, subfolder: str, name: str):
        path = self.output_dir / subfolder / name
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  [plot] saved -> {path}")

    @staticmethod
    def _is_backward(epoch: int, trainer: "Trainer") -> bool:
        return isinstance(trainer.strategy, BidirectionalStrategy) and epoch % 2 != 0

    # ------------------------------------------------------------------
    # Timelapse
    # ------------------------------------------------------------------

    def make_timelapse_all(self, fps: int = 4):
        for key, frames in self._frames.items():
            if not frames:
                continue
            self._write_gif(frames, self.output_dir / f"timelapse_{key}.gif", fps)

    def _write_gif(self, frames: List[np.ndarray], path: Path, fps: int):
        try:
            import PIL.Image
            h0, w0 = frames[0].shape[:2]
            pil_frames = []
            for arr in frames:
                img = PIL.Image.fromarray(arr).convert("RGB")
                if img.size != (w0, h0):
                    img = img.resize((w0, h0), PIL.Image.LANCZOS)
                pil_frames.append(img)
            pil_frames[0].save(
                path, save_all=True, append_images=pil_frames[1:],
                loop=0, duration=int(1000 / fps))
            logger.info(f"  Timelapse [{path.stem}] -> {path}")
        except ImportError:
            logger.warning("Pillow not installed; skipping GIF timelapse.")


# ---------------------------------------------------------------------------
# 2-D logging visualizer
# ---------------------------------------------------------------------------

class LoggingVisualizer(_LoggingMixin, Visualizer):
    def __init__(self, config: VisualizeConfig, transport_map: Callable,
                 output_dir: str, wandb_run=None, inverse_transport_map=None):
        Visualizer.__init__(self, config, transport_map)
        self._init_logging(output_dir, wandb_run=wandb_run,
                           transport_map=transport_map,
                           inverse_transport_map=inverse_transport_map)

    def display(self, epoch: int, trainer: "Trainer"):
        if epoch % self.config.show_each != 0:
            return
        backward = self._is_backward(epoch, trainer)
        n = self.config.n_trajectories
        x, y = trainer.current_coupling.coupling_sample(n)
        x, y = x.to(trainer.device), y.to(trainer.device)
        start = y if backward else x
        ref_target = (trainer.new_source if backward else trainer.new_target).sample(n).to(trainer.device)
        traj = trainer.flow_matching.evolve_trajectory(
            start, trainer.strategy.get_flow(epoch, trainer.device)[0],
            num_steps=25, sigma=trainer._sigma, extend=True,
        )
        data_dim = start.shape[1]
        traj = traj[:, :, :data_dim]
        end = traj[-1]
        s = trainer.scale
        stride = self.config.trajectory_stride
        traj_2d = (traj * s).cpu() if stride > 0 else None
        self._lock_axis_lim((start * s).cpu(), (end * s).cpu(), (ref_target * s).cpu())
        self._display_evolution(
            (start * s).cpu(), (end * s).cpu(), (ref_target * s).cpu(),
            trajectories=traj_2d, trajectory_stride=stride,
        )
        fig = plt.gcf()
        for ax in fig.axes:
            self._apply_axis_lim(ax)
        direction = "Backward ν->μ" if backward else "Forward μ->ν"
        fig.suptitle(f"Iter {epoch} ({direction})")
        self._accumulate(fig, "coupling")
        self._save_fig(fig, "coupling", f"coupling_{epoch:04d}.png")

    def at_postprocess_end(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        if (iteration + 1) % self.config.show_each != 0 and iteration != trainer.config.iterations - 1:
            return
        device = trainer.device
        n_samples = 2500
        n_coup = 250
        t_val = (iteration + 1) / trainer.config.iterations
        t_tensor = torch.full((n_samples // 10, 1), t_val, device=device)

        with torch.no_grad():
            mu_samples = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_init(n_samples // 10).to(device),
                trainer.new_source.sample(n_samples // 10).to(device),
                t_tensor, 0.0,
            )
            nu_samples = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_target(n_samples // 10).to(device),
                trainer.new_target.sample(n_samples // 10).to(device),
                t_tensor, 0.0,
            )
        mu_coup, nu_coup = trainer.current_coupling.coupling_sample(n_coup)
        mu_cpu, nu_cpu = mu_coup.cpu(), nu_coup.cpu()
        mu_s = mu_samples.cpu()
        nu_s = nu_samples.cpu()

        fig = plt.figure(figsize=(6, 6))
        plt.plot(mu_s[:, 0], mu_s[:, 1], ".", label="Mu Distribution", alpha=0.5)
        plt.plot(nu_s[:, 0], nu_s[:, 1], ".", label="Nu Distribution", alpha=0.5)
        plt.plot(mu_cpu[:, 0], mu_cpu[:, 1], "x", label="Mu Coupling Samples", alpha=0.5)
        plt.plot(nu_cpu[:, 0], nu_cpu[:, 1], "x", label="Nu Coupling Samples", alpha=0.5)
        for j in range(mu_cpu.shape[0]):
            plt.plot([mu_cpu[j, 0], nu_cpu[j, 0]], [mu_cpu[j, 1], nu_cpu[j, 1]], "k-", alpha=0.1)
        plt.legend()
        plt.title(f"Distributions after Advancing Step {iteration+1}")
        self.save_fig(fig, "marginals", f"distributions_{iteration:04d}.png")


# ---------------------------------------------------------------------------
# 3-D logging visualizer
# ---------------------------------------------------------------------------

class LoggingVisualizer3D(_LoggingMixin, Visualizer3D):
    def __init__(self, config: VisualizeConfig, transport_map: Callable,
                 output_dir: str, wandb_run=None, inverse_transport_map=None):
        Visualizer3D.__init__(self, config, transport_map)
        self._init_logging(output_dir, wandb_run=wandb_run,
                           transport_map=transport_map,
                           inverse_transport_map=inverse_transport_map)

    def display(self, epoch: int, trainer: "Trainer"):
        if epoch % self.config.show_each != 0:
            return
        backward = self._is_backward(epoch, trainer)
        n = self.config.n_trajectories
        x, y = trainer.current_coupling.coupling_sample(n)
        x, y = x.to(trainer.device), y.to(trainer.device)
        start = y if backward else x
        ref_target = (trainer.new_source if backward else trainer.new_target).sample(n).to(trainer.device)
        traj = trainer.flow_matching.evolve_trajectory(
            start, trainer.strategy.get_flow(epoch, trainer.device)[0],
            num_steps=25, sigma=trainer._sigma, extend=True,
        )
        traj = traj[:, :, :self.dim]
        end = traj[-1]
        s = trainer.scale
        stride = self.config.trajectory_stride
        traj_3d = (traj * s).cpu() if stride > 0 else None
        self._display_evolution(
            (start * s).cpu(), (end * s).cpu(), (ref_target * s).cpu(),
            trajectories=traj_3d, trajectory_stride=stride,
        )
        fig = plt.gcf()
        direction = "Backward ν->μ" if backward else "Forward μ->ν"
        fig.suptitle(f"Iter {epoch} ({direction})")
        self._accumulate(fig, "coupling")
        self._save_fig(fig, "coupling", f"coupling_{epoch:04d}.png")

    def at_postprocess_end(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        if (iteration + 1) % self.config.show_each != 0 and iteration != trainer.config.iterations - 1:
            return
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        device = trainer.device
        n_samples = 2500
        n_coup = 250
        t_val = (iteration + 1) / trainer.config.iterations
        t_tensor = torch.full((n_samples // 10, 1), t_val, device=device)

        with torch.no_grad():
            mu_samples = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_init(n_samples // 10).to(device),
                trainer.new_source.sample(n_samples // 10).to(device),
                t_tensor, 0.0,
            )
            nu_samples = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_target(n_samples // 10).to(device),
                trainer.new_target.sample(n_samples // 10).to(device),
                t_tensor, 0.0,
            )
        mu_coup, nu_coup = trainer.current_coupling.coupling_sample(n_coup)
        mu_cpu, nu_cpu = mu_coup.cpu(), nu_coup.cpu()
        mu_s = mu_samples.cpu()
        nu_s = nu_samples.cpu()

        with torch.no_grad():
            nu_flow = trainer.flow_matching.evolve(
                mu_cpu.to(device), trainer.strategy.get_flow(0, device)[0],
                num_steps=200, sigma=trainer._sigma, extend=True,
            )[:, :trainer.strategy.config.dimension]
        nu_flow_cpu = nu_flow.cpu()

        s = trainer.scale
        # Scale all point clouds consistently with display()
        mu_s        = mu_s        * s
        nu_s        = nu_s        * s
        mu_cpu      = mu_cpu      * s
        nu_cpu      = nu_cpu      * s
        nu_flow_cpu = nu_flow_cpu * s

        # Data-adaptive axis limits with 10 % padding
        all_pts = torch.cat([mu_s, nu_s, mu_cpu, nu_cpu, nu_flow_cpu], dim=0)
        lo = all_pts.min().item()
        hi = all_pts.max().item()
        pad = (hi - lo) * 0.10 + 0.1
        L_lo, L_hi = lo - pad, hi + pad

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(mu_s[:, 0], mu_s[:, 1], mu_s[:, 2], c="blue", alpha=0.1, depthshade=True, s=3, label=r"$\mu_t$")
        ax.scatter(nu_s[:, 0], nu_s[:, 1], nu_s[:, 2], c="red", alpha=0.1, depthshade=True, s=3, label=r"$\nu_t$")
        ax.scatter(mu_cpu[:, 0], mu_cpu[:, 1], mu_cpu[:, 2], c="purple", marker="x", alpha=0.75, depthshade=True, label=r"$\pi_t^{(1)}$")
        ax.scatter(nu_cpu[:, 0], nu_cpu[:, 1], nu_cpu[:, 2], c="orange", marker="x", alpha=0.75, depthshade=True, label=r"$\pi_t^{(2)}$")
        ax.scatter(nu_flow_cpu[:, 0], nu_flow_cpu[:, 1], nu_flow_cpu[:, 2],
                   c="green", marker="o", alpha=0.75, depthshade=True, s=50, label="Flow evolved")
        dist = torch.norm(nu_cpu - mu_cpu, dim=1)
        for j in range(mu_cpu.shape[0]):
            ax.plot([mu_cpu[j, 0], nu_cpu[j, 0]], [mu_cpu[j, 1], nu_cpu[j, 1]],
                    [mu_cpu[j, 2], nu_cpu[j, 2]], color="gray", alpha=0.8)
        ax.set_xlim([L_lo, L_hi]); ax.set_ylim([L_lo, L_hi]); ax.set_zlim([L_lo, L_hi])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(fontsize=7, loc="upper left")
        sm_cb = plt.cm.ScalarMappable(cmap="viridis")
        sm_cb.set_array(dist.numpy())
        plt.colorbar(sm_cb, ax=ax, shrink=0.6, label="Transport length")
        ax.set_title(f"Distributions after Advancing Step {iteration+1}")
        self.save_fig(fig, "marginals", f"distributions_{iteration:04d}.png")

    def _save_marginals_plot(self, time_step, mu_t, nu_t, x_coup, y_coup):
        if mu_t.dim() != 2 or mu_t.shape[1] < 2:
            return
        self._lock_axis_lim(mu_t, nu_t, x_coup, y_coup)
        # 3D experiments: show XY, XZ, YZ projections (3 columns × 2 rows)
        # 2D experiments: show 1 column × 2 rows
        dim = mu_t.shape[1]
        projections = [(0, 1, "X", "Y")]
        if dim >= 3:
            projections += [(0, 2, "X", "Z"), (1, 2, "Y", "Z")]

        ncols = len(projections)
        fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 9))
        if ncols == 1:
            axes = axes[:, None]   # make 2D

        kw = dict(s=5, alpha=0.40, linewidths=0)
        for col, (xi, yi, xl, yl) in enumerate(projections):
            ax_src, ax_tgt = axes[0, col], axes[1, col]
            proj = f"{xl}{yl}"

            ax_src.scatter(mu_t[:, xi],   mu_t[:, yi],   c="#4878d0", label=r"$\mu_t$",         **kw)
            ax_src.scatter(x_coup[:, xi], x_coup[:, yi], c="#ee854a", label=r"$\pi_t^{(1)}$",   **kw)
            ax_src.set_title(f"Source — {proj}  (iter {time_step})", fontsize=8, fontweight="bold")
            ax_src.set_xlabel(xl, fontsize=7); ax_src.set_ylabel(yl, fontsize=7)
            ax_src.legend(fontsize=6); ax_src.set_aspect("equal")
            ax_src.grid(True, alpha=0.2); ax_src.tick_params(labelsize=6)
            ax_src.spines[["top", "right"]].set_visible(False)
            if col == 0:
                self._apply_axis_lim(ax_src)

            ax_tgt.scatter(nu_t[:, xi],   nu_t[:, yi],   c="#c44e52", label=r"$\nu_t$",         **kw)
            ax_tgt.scatter(y_coup[:, xi], y_coup[:, yi], c="#55a868", label=r"$\pi_t^{(2)}$",   **kw)
            ax_tgt.set_title(f"Target — {proj}  (iter {time_step})", fontsize=8, fontweight="bold")
            ax_tgt.set_xlabel(xl, fontsize=7); ax_tgt.set_ylabel(yl, fontsize=7)
            ax_tgt.legend(fontsize=6); ax_tgt.set_aspect("equal")
            ax_tgt.grid(True, alpha=0.2); ax_tgt.tick_params(labelsize=6)
            ax_tgt.spines[["top", "right"]].set_visible(False)
            if col == 0:
                self._apply_axis_lim(ax_tgt)

        fig.tight_layout()
        self._accumulate(fig, "marginals")
        self._save_fig(fig, "marginals", f"marginals_{time_step:04d}.png")


# ---------------------------------------------------------------------------
# Image logging visualizer
# ---------------------------------------------------------------------------

class LoggingVisualizerImage(_LoggingMixin, VisualizerImage):
    def __init__(self, config: VisualizeConfig, img_shape: Tuple[int, int, int],
                 output_dir: str, wandb_run=None,
                 transport_map=None, inverse_transport_map=None):
        VisualizerImage.__init__(self, config, img_shape, transport_map=transport_map)
        self._init_logging(output_dir, wandb_run=wandb_run,
                           transport_map=transport_map,
                           inverse_transport_map=inverse_transport_map)

    def _compute_image_metrics(self, time_step: int, trainer: "Trainer") -> dict:
        """Compute FID and KID every ``show_each`` iterations (and at the final step)."""
        is_final = (time_step == trainer.config.iterations - 1)
        if not is_final and time_step % self.config.show_each != 0:
            return {}
        result = {}
        fid_val = self._compute_fid(trainer)
        if fid_val is not None:
            result["fid"] = fid_val
            logger.info("  FID (iter %d) = %.4f", time_step, fid_val)
        kid_val = self._compute_kid(trainer)
        if kid_val is not None:
            result["kid"] = kid_val
            logger.info("  KID (iter %d) = %.4f", time_step, kid_val)
        return result

    def _compute_fid(self, trainer: "Trainer", n_gen: int = 128) -> Optional[float]:
        """Compute FID between generated and true target images.

        Uses ``torchmetrics`` if available, falls back to a lightweight
        statistics-based estimate using scipy otherwise.
        """
        try:
            import scipy.linalg
        except ImportError:
            logger.debug("scipy not installed — skipping FID computation.")
            return None

        device = trainer.device
        C, H, W = self.img_shape

        # --- Generate images by evolving new source through the learned flow ---
        generated, real = [], []
        batch = min(64, n_gen)
        n_batches = (n_gen + batch - 1) // batch

        fwd_model = trainer.strategy.get_flow(0, device)[0]
        fwd_model.eval()
        s = trainer.scale

        with torch.no_grad():
            for _ in range(n_batches):
                x_src = trainer.new_source.sample(batch).to(device)
                x_gen = trainer.flow_matching.evolve(
                    x_src, fwd_model, num_steps=200,
                    sigma=trainer._sigma, extend=True,
                )
                if is_image_tensor(x_gen):
                    x_gen = x_gen[:, :C, :, :]
                generated.append((x_gen * s).cpu())

                x_tgt = trainer.new_target.sample(batch).to(device)
                real.append((x_tgt * s).cpu())

        gen_imgs  = torch.cat(generated, dim=0)[:n_gen]
        real_imgs = torch.cat(real,      dim=0)[:n_gen]

        # Try torchmetrics FID first (preferred, handles InceptionV3 features)
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to("cpu")
            # torchmetrics expects uint8 [0, 255] or float [0, 1] with normalize=True
            gen_01  = (gen_imgs.clamp(-1, 1) + 1) / 2   # [-1,1] → [0,1]
            real_01 = (real_imgs.clamp(-1, 1) + 1) / 2
            # FID requires 3-channel; expand grayscale if needed
            if gen_01.shape[1] == 1:
                gen_01  = gen_01.expand(-1, 3, -1, -1)
                real_01 = real_01.expand(-1, 3, -1, -1)
            fid_metric.update(real_01, real=True)
            fid_metric.update(gen_01,  real=False)
            return float(fid_metric.compute().item())
        except Exception:
            pass

        # Fallback: raw pixel statistics (cheap but less accurate)
        gen_flat  = gen_imgs.view(n_gen, -1).numpy()
        real_flat = real_imgs.view(n_gen, -1).numpy()
        mu_g, mu_r = gen_flat.mean(0), real_flat.mean(0)
        cov_g = np.cov(gen_flat,  rowvar=False)
        cov_r = np.cov(real_flat, rowvar=False)
        diff  = mu_g - mu_r
        # Bures metric (matrix square root)
        try:
            covmean = scipy.linalg.sqrtm(cov_g @ cov_r)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fid_approx = float(diff @ diff + np.trace(cov_g + cov_r - 2 * covmean))
        except Exception:
            fid_approx = None
        return fid_approx

    def _compute_kid(self, trainer: "Trainer", n_gen: int = 128) -> Optional[float]:
        """Compute KID between generated and true target images."""
        device = trainer.device
        C, H, W = self.img_shape
        fwd_model = trainer.strategy.get_flow(0, device)[0]
        fwd_model.eval()
        s = trainer.scale
        generated, real = [], []
        batch = min(64, n_gen)
        n_batches = (n_gen + batch - 1) // batch
        with torch.no_grad():
            for _ in range(n_batches):
                x_src = trainer.new_source.sample(batch).to(device)
                x_gen = trainer.flow_matching.evolve(
                    x_src, fwd_model, num_steps=200,
                    sigma=trainer._sigma, extend=True,
                )
                if is_image_tensor(x_gen):
                    x_gen = x_gen[:, :C, :, :]
                generated.append((x_gen * s).cpu())
                x_tgt = trainer.new_target.sample(batch).to(device)
                real.append((x_tgt * s).cpu())
        gen_imgs  = torch.cat(generated, dim=0)[:n_gen]
        real_imgs = torch.cat(real,      dim=0)[:n_gen]
        gen_01  = (gen_imgs.clamp(-1, 1) + 1) / 2
        real_01 = (real_imgs.clamp(-1, 1) + 1) / 2
        if gen_01.shape[1] == 1:
            gen_01  = gen_01.expand(-1, 3, -1, -1)
            real_01 = real_01.expand(-1, 3, -1, -1)
        try:
            from torchmetrics.image.kid import KernelInceptionDistance
            n_sub = min(1_000, len(gen_01), len(real_01))
            kid_metric = KernelInceptionDistance(normalize=True, subset_size=n_sub).to("cpu")
            kid_metric.update(real_01, real=True)
            kid_metric.update(gen_01,  real=False)
            kid_mean, _ = kid_metric.compute()
            return float(kid_mean.item())
        except Exception:
            pass
        # Fallback: MMD on flattened pixels
        from dualwcfm.metrics import mmd as _mmd
        return _mmd(gen_01.flatten(1), real_01.flatten(1))

    def display(self, epoch: int, trainer: "Trainer"):
        if epoch % self.config.show_each != 0:
            return
        backward = self._is_backward(epoch, trainer)
        n = min(6, self.config.n_trajectories)
        x, y = trainer.current_coupling.coupling_sample(n)
        x, y = x.to(trainer.device), y.to(trainer.device)
        start = y if backward else x
        end = x if backward else y
        # ref_target = (trainer.new_source if backward else trainer.new_target).sample(n).to(trainer.device)
        evolved = trainer.flow_matching.evolve(
            start, trainer.strategy.get_flow(epoch, trainer.device)[0],
            num_steps=200, sigma=trainer._sigma, extend=True,
        )
        if is_image_tensor(evolved):
            evolved = evolved[:, :self.img_shape[0], :, :]
        s = trainer.scale
        self._display_evolution((start * s).cpu(), (evolved * s).cpu(), (end * s).cpu())
        fig = plt.gcf()
        direction = "Backward ν->μ" if backward else "Forward μ->ν"
        fig.suptitle(f"Iter {epoch} ({direction})")
        self._accumulate(fig, "coupling")
        self._save_fig(fig, "coupling", f"coupling_{epoch:04d}.png")

    def at_postprocess_end(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        n_samples = 4
        mu_coup, nu_coup = trainer.current_coupling.coupling_sample(n_samples)
        mu_coup, nu_coup = mu_coup.cpu(), nu_coup.cpu()

        fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))
        C, H, W = self.img_shape
        cmap = "gray" if C == 1 else None

        def _to_imshow(tensor, i):
            img = tensor[i] if tensor.dim() == 4 else tensor[i].reshape(C, H, W)
            img = img.float().numpy()
            if C == 1:
                return img.squeeze(0)  # (H, W)
            else:
                return img.transpose(1, 2, 0)  # (H, W, C)

        for i in range(n_samples):
            axes[0, i].imshow(_to_imshow(mu_coup, i), cmap=cmap, vmin=-1, vmax=1)
            axes[0, i].set_title(f"Source {i+1}"); axes[0, i].axis("off")
            axes[1, i].imshow(_to_imshow(nu_coup, i), cmap=cmap, vmin=-1, vmax=1)
            axes[1, i].set_title(f"Target {i+1}"); axes[1, i].axis("off")

        plt.suptitle(f"Coupling after Step {iteration + 1}")
        plt.tight_layout()
        self.save_fig(fig, "coupling", f"coupling_img_{iteration:04d}.png")

    def _save_marginals_plot(self, time_step, mu_t, nu_t, x_coup, y_coup):
        n = min(8, x_coup.shape[0])
        C, H, W = self.img_shape
        fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
        if n == 1:
            axes = axes.reshape(2, 1)

        cmap = "gray" if C == 1 else None

        def _show(ax, tensor, i, title):
            img = tensor[i] if tensor[i].dim() == 3 else tensor[i].reshape(C, H, W)
            img = img.float().numpy()
            if C == 1:
                img = img.squeeze(0)  # (H, W)
            else:
                img = img.transpose(1, 2, 0)  # (H, W, C)
            ax.imshow(img, cmap=cmap, vmin=-1, vmax=1)
            ax.set_title(title, fontsize=6); ax.axis("off")

        for i in range(n):
            _show(axes[0, i], x_coup, i, f"src {i}")
            _show(axes[1, i], y_coup, i, f"tgt {i}")
        fig.suptitle(f"Coupling samples — iter {time_step}", fontsize=9)
        plt.tight_layout()
        self._accumulate(fig, "marginals")
        self._save_fig(fig, "marginals", f"marginals_{time_step:04d}.png")
