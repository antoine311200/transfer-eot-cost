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

    def _init_logging(self, output_dir: str, wandb_run=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("coupling", "cfm", "weights", "marginals", "pretrain"):
            (self.output_dir / sub).mkdir(exist_ok=True)

        self._wandb_run = wandb_run  # may be None or a WandbRun instance

        self._csv_path = self.output_dir / "metrics.csv"
        self._csv_fields = [
            "iteration",
            # Sliced-Wasserstein marginal distances
            "W1_mu", "W2_mu", "W1_nu", "W2_nu",
            # Non-parametric target-marginal divergences
            "mmd_nu",            # Maximum Mean Discrepancy (RBF kernel, median BW)
            "energy_dist_nu",    # Energy distance (kernel-free)
            "sinkhorn_div",      # Debiased Sinkhorn divergence (W2 proxy, distribution-only)
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
        pts = torch.cat(valid, dim=0)[:, :2]
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
        from dualwcfm.metrics import mmd as _mmd, energy_distance as _ed, sinkhorn_divergence as _sd
        nu_ref = trainer.new_target.sample(n).to(device)
        if is_image_tensor(y):
            nu_ref = nu_ref.view_as(y)
        mmd_nu         = _mmd(y, nu_ref)
        energy_dist_nu = _ed(y, nu_ref)
        sinkhorn_div   = _sd(y, nu_ref)

        row = {
            "iteration":      time_step,
            **distances,
            "mmd_nu":         mmd_nu,
            "energy_dist_nu": energy_dist_nu,
            "sinkhorn_div":   sinkhorn_div,
            "weight_mean":    float(w_np.mean()),
            "weight_std":     float(w_np.std()),
            "weight_min":     float(w_np.min()),
            "weight_max":     float(w_np.max()),
            "weight_ess":     ess_norm,
        }
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writerow(row)

        # W&B logging (no-op when wandb_run is None)
        if self._wandb_run:
            self._wandb_run.log(
                {
                    **{k: v for k, v in row.items() if k != "iteration"},
                },
                step=time_step,
            )

        self._save_weight_histogram(time_step, w_np)
        self._save_marginals_plot(
            time_step,
            (mu_t * s).cpu(), (nu_t * s).cpu(),
            (x * s).cpu(),    (y * s).cpu(),
        )
        return distances

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
                 output_dir: str, wandb_run=None):
        Visualizer.__init__(self, config, transport_map)
        self._init_logging(output_dir, wandb_run=wandb_run)

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
                 output_dir: str, wandb_run=None):
        Visualizer3D.__init__(self, config, transport_map)
        self._init_logging(output_dir, wandb_run=wandb_run)

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
                num_steps=100, sigma=trainer._sigma, extend=True,
            )[:, :trainer.strategy.config.dimension]
        nu_flow_cpu = nu_flow.cpu()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(mu_s[:, 0], mu_s[:, 1], mu_s[:, 2], c="blue", alpha=0.1, depthshade=True, s=3)
        ax.scatter(nu_s[:, 0], nu_s[:, 1], nu_s[:, 2], c="red", alpha=0.1, depthshade=True, s=3)
        ax.scatter(mu_cpu[:, 0], mu_cpu[:, 1], mu_cpu[:, 2], c="purple", marker="x", alpha=0.75, depthshade=True)
        ax.scatter(nu_cpu[:, 0], nu_cpu[:, 1], nu_cpu[:, 2], c="orange", marker="x", alpha=0.75, depthshade=True)
        ax.scatter(nu_flow_cpu[:, 0], nu_flow_cpu[:, 1], nu_flow_cpu[:, 2],
                   c="green", marker="o", alpha=0.75, depthshade=True, s=50, label="Flow Evolved")
        dist = torch.norm(nu_cpu - mu_cpu, dim=1)
        for j in range(mu_cpu.shape[0]):
            ax.plot([mu_cpu[j, 0], nu_cpu[j, 0]], [mu_cpu[j, 1], nu_cpu[j, 1]],
                    [mu_cpu[j, 2], nu_cpu[j, 2]], color="gray", alpha=0.8)
        L = 1.2
        ax.plot([-L, L], [0, 0], [0, 0], color="k", linewidth=1)
        ax.plot([0, 0], [-L, L], [0, 0], color="k", linewidth=1)
        ax.plot([0, 0], [0, 0], [-L, L], color="k", linewidth=1)
        ax.set_xlim([-L, L]); ax.set_ylim([-L, L]); ax.set_zlim([-L, L])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        sm_cb = plt.cm.ScalarMappable(cmap="viridis")
        sm_cb.set_array(dist.numpy())
        plt.colorbar(sm_cb, ax=ax, shrink=0.6, label="Transport length")
        ax.set_title(f"Distributions after Advancing Step {iteration+1}")
        self.save_fig(fig, "marginals", f"distributions_{iteration:04d}.png")

    def _save_marginals_plot(self, time_step, mu_t, nu_t, x_coup, y_coup):
        if mu_t.dim() != 2 or mu_t.shape[1] < 2:
            return
        self._lock_axis_lim(mu_t, nu_t, x_coup, y_coup)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        kw = dict(s=4, alpha=0.35, linewidths=0)
        axes[0].scatter(mu_t[:, 0],   mu_t[:, 1],   c="royalblue",  label=r"$\mu_t$ theory",  **kw)
        axes[0].scatter(x_coup[:, 0], x_coup[:, 1], c="darkorange", label=r"$\pi_t^{(1)}$",   **kw)
        axes[0].legend(fontsize=7); axes[0].set_aspect("equal")
        axes[0].set_title(f"Source marginal (xy) — iter {time_step}")
        self._apply_axis_lim(axes[0])
        axes[1].scatter(nu_t[:, 0],   nu_t[:, 1],   c="firebrick",  label=r"$\nu_t$ theory",  **kw)
        axes[1].scatter(y_coup[:, 0], y_coup[:, 1], c="limegreen",  label=r"$\pi_t^{(2)}$",   **kw)
        axes[1].legend(fontsize=7); axes[1].set_aspect("equal")
        axes[1].set_title(f"Target marginal (xy) — iter {time_step}")
        self._apply_axis_lim(axes[1])
        fig.tight_layout()
        self._accumulate(fig, "marginals")
        self._save_fig(fig, "marginals", f"marginals_{time_step:04d}.png")


# ---------------------------------------------------------------------------
# Image logging visualizer
# ---------------------------------------------------------------------------

class LoggingVisualizerImage(_LoggingMixin, VisualizerImage):
    def __init__(self, config: VisualizeConfig, img_shape: Tuple[int, int, int],
                 output_dir: str, wandb_run=None):
        VisualizerImage.__init__(self, config, img_shape, transport_map=None)
        self._init_logging(output_dir, wandb_run=wandb_run)
        self._fid_computed = False  # compute at most once (final iteration)

    def analyse(self, time_step: int, trainer: "Trainer",
                n_samples: int = 2000, n_proj: int = 2000) -> dict:
        """Extend mixin analyse() with optional FID at the last iteration."""
        distances = super().analyse(time_step, trainer, n_samples, n_proj)

        is_final = (time_step == trainer.config.iterations - 1)
        if is_final and not self._fid_computed:
            fid_val = self._compute_fid(trainer)
            if fid_val is not None:
                logger.info("  FID (final iteration) = %.4f", fid_val)
                # Append FID to the last CSV row as a separate summary entry
                summary_path = self.output_dir / "fid_summary.txt"
                summary_path.write_text(f"FID={fid_val:.4f}\n")
                if self._wandb_run:
                    self._wandb_run.summary({"FID": fid_val})
            self._fid_computed = True

        return distances

    def _compute_fid(self, trainer: "Trainer", n_gen: int = 2048) -> Optional[float]:
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
                    x_src, fwd_model, num_steps=50,
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
            num_steps=25, sigma=trainer._sigma, extend=True,
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
        for i in range(n_samples):
            img_mu = mu_coup[i].squeeze(0).numpy() if mu_coup.dim() == 4 else \
                mu_coup[i].reshape(H, W).numpy()
            axes[0, i].imshow(img_mu, cmap="gray", vmin=-1, vmax=1)
            axes[0, i].set_title(f"Source {i+1}"); axes[0, i].axis("off")
            img_nu = nu_coup[i].squeeze(0).numpy() if nu_coup.dim() == 4 else \
                nu_coup[i].reshape(H, W).numpy()
            axes[1, i].imshow(img_nu, cmap="gray", vmin=-1, vmax=1)
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

        def _show(ax, tensor, i, title):
            img = tensor[i]
            img = img.squeeze(0) if img.dim() == 3 else img.reshape(H, W)
            ax.imshow(img.float().numpy(), cmap="gray", vmin=-1, vmax=1)
            ax.set_title(title, fontsize=6); ax.axis("off")

        for i in range(n):
            _show(axes[0, i], x_coup, i, f"src {i}")
            _show(axes[1, i], y_coup, i, f"tgt {i}")
        fig.suptitle(f"Coupling samples — iter {time_step}", fontsize=9)
        plt.tight_layout()
        self._accumulate(fig, "marginals")
        self._save_fig(fig, "marginals", f"marginals_{time_step:04d}.png")
