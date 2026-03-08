"""Logging visualizers that save per-iteration plots and metrics to disk.

Each iteration's key diagnostics are saved as PNG files under `output_dir/`:
  coupling/coupling_{iter}.png   — flow trajectories
  cfm/cfm_{iter}.png            — vector field quiver (2D/3D only)
  marginals/marginals_{iter}.png — mu_t / nu_t vs coupling marginals
  marginals/duals_{iter}.png    — dual potentials scatter (2D/3D only)
  weights/weights_{iter}.png    — importance weight histogram
  metrics.csv                   — scalar metrics per iteration

At the end of training a timelapse GIF is assembled from coupling snapshots.
"""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
import torch

from dualwcfm.config import VisualizeConfig
from dualwcfm.utils import is_image_tensor
from dualwcfm.visualization import Visualizer, Visualizer3D, VisualizerImage

if TYPE_CHECKING:
    from dualwcfm.trainer import Trainer


# ---------------------------------------------------------------------------
# Shared mixin – saving, CSV logging, GIF assembly
# ---------------------------------------------------------------------------

class _LoggingMixin:
    """Mixin that adds disk-saving, CSV logging and GIF timelapse to Visualizer."""

    def _init_logging(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "coupling").mkdir(exist_ok=True)
        (self.output_dir / "cfm").mkdir(exist_ok=True)
        (self.output_dir / "weights").mkdir(exist_ok=True)
        (self.output_dir / "marginals").mkdir(exist_ok=True)

        self._csv_path = self.output_dir / "metrics.csv"
        self._csv_fields = ["iteration", "W1_mu", "W2_mu", "W1_nu", "W2_nu",
                            "weight_mean", "weight_std", "weight_min", "weight_max"]
        with open(self._csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writeheader()

        self._coupling_frames: list[np.ndarray] = []   # for GIF

    # ------------------------------------------------------------------
    # Override save_fig so trainer plt calls go to disk
    # ------------------------------------------------------------------

    def save_fig(self, fig: plt.Figure, subfolder: str, name: str):
        self._save_fig(fig, subfolder, name)

    # ------------------------------------------------------------------
    # Override analyse() to save weight stats, marginals and write CSV
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
            # Theoretical marginals at time t
            mu_t = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_init(n).to(device),
                trainer.new_source.sample(n).to(device),
                t_tensor, sigma=0.0)
            nu_t = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_target(n).to(device),
                trainer.new_target.sample(n).to(device),
                t_tensor, sigma=0.0)

            # Current coupling samples
            x, y = trainer.current_coupling.coupling_sample(n)
            x, y = x.to(device), y.to(device)

            # Importance weights
            f_dual, g_dual, _ = trainer.strategy.get_duals(device)
            h_dual, _ = trainer.strategy.get_dual_mismatch(device)
            a, b = f_dual(x), g_dual(y)
            if trainer.config.dual_mismatch_reg > 0:
                xf = x.view(x.shape[0], -1)
                yf = y.view(y.shape[0], -1)
                d = h_dual(torch.cat([xf, yf], dim=1))
            else:
                d = torch.zeros_like(a)
            log_w = (a + b + d) / trainer.config.iterations
            w = torch.exp(log_w)

        w_np = w.cpu().float().numpy().ravel()
        row = {
            "iteration": time_step,
            **distances,
            "weight_mean": float(w_np.mean()),
            "weight_std":  float(w_np.std()),
            "weight_min":  float(w_np.min()),
            "weight_max":  float(w_np.max()),
        }
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writerow(row)

        self._save_weight_histogram(time_step, w_np)
        self._save_marginals_plot(
            time_step,
            (mu_t * s).cpu(), (nu_t * s).cpu(),
            (x * s).cpu(), (y * s).cpu(),
        )

        return distances

    # ------------------------------------------------------------------
    # Per-iteration plots
    # ------------------------------------------------------------------

    def _save_marginals_plot(self, time_step: int,
                             mu_t: torch.Tensor, nu_t: torch.Tensor,
                             x_coup: torch.Tensor, y_coup: torch.Tensor):
        """2D scatter: theoretical marginals vs coupling marginals.

        Subclasses override this for image or 3D data.
        """
        if mu_t.dim() != 2:
            return   # images handled by subclass
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        kw = dict(s=4, alpha=0.35)
        axes[0].scatter(mu_t[:, 0], mu_t[:, 1], c="royalblue", label=r"$\mu_t$ theory", **kw)
        axes[0].scatter(x_coup[:, 0], x_coup[:, 1], c="darkorange", label=r"$\pi_t^{(1)}$ coupling", **kw)
        axes[0].legend(fontsize=7)
        axes[0].set_aspect("equal")
        axes[0].set_title(f"Source marginal — iter {time_step}")

        axes[1].scatter(nu_t[:, 0], nu_t[:, 1], c="firebrick", label=r"$\nu_t$ theory", **kw)
        axes[1].scatter(y_coup[:, 0], y_coup[:, 1], c="limegreen", label=r"$\pi_t^{(2)}$ coupling", **kw)
        axes[1].legend(fontsize=7)
        axes[1].set_aspect("equal")
        axes[1].set_title(f"Target marginal — iter {time_step}")

        fig.tight_layout()
        self._save_fig(fig, "marginals", f"marginals_{time_step:04d}.png")

    def _save_weight_histogram(self, time_step: int, w: np.ndarray):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(w, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="w=1")
        ax.set_xlabel("Weight $w_k$")
        ax.set_ylabel("Count")
        ax.set_title(f"Weight distribution — iter {time_step}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "weights" / f"weights_{time_step:04d}.png", dpi=80)
        plt.close(fig)

    def _save_fig(self, fig: plt.Figure, subfolder: str, name: str):
        path = self.output_dir / subfolder / name
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)

    def _fig_to_rgb_array(self, fig: plt.Figure) -> np.ndarray:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=60)
        buf.seek(0)
        import PIL.Image
        img = PIL.Image.open(buf)
        arr = np.array(img.convert("RGB"))
        buf.close()
        return arr

    def make_timelapse(self, fps: int = 4):
        """Save a GIF from accumulated coupling frames."""
        if not self._coupling_frames:
            return
        try:
            import PIL.Image
            frames = [PIL.Image.fromarray(f) for f in self._coupling_frames]
            out = self.output_dir / "timelapse.gif"
            frames[0].save(out, save_all=True, append_images=frames[1:],
                           loop=0, duration=int(1000 / fps))
            print(f"Timelapse saved → {out}")
        except ImportError:
            print("Pillow not installed; skipping GIF timelapse.")

    def _accumulate_fig(self, fig: plt.Figure):
        """Render fig to an RGB array and store for timelapse."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=60)
        buf.seek(0)
        try:
            import PIL.Image
            arr = np.array(PIL.Image.open(buf).convert("RGB"))
            self._coupling_frames.append(arr)
        except ImportError:
            pass
        buf.close()


# ---------------------------------------------------------------------------
# 2-D logging visualizer
# ---------------------------------------------------------------------------

class LoggingVisualizer(_LoggingMixin, Visualizer):
    def __init__(self, config: VisualizeConfig, transport_map: Callable,
                 output_dir: str):
        Visualizer.__init__(self, config, transport_map)
        self._init_logging(output_dir)

    def display(self, epoch: int, trainer: "Trainer"):
        if epoch % self.config.show_each != 0:
            return
        x, _ = trainer.current_coupling.coupling_sample(self.config.n_trajectories)
        y_target = trainer.new_target.sample(self.config.n_trajectories)
        x, y_target = x.to(trainer.device), y_target.to(trainer.device)
        traj = trainer.flow_matching.evolve_trajectory(
            x, trainer.strategy.get_flow(epoch, trainer.device)[0],
            num_steps=25, sigma=trainer.config.sigma, extend=True,
        )
        traj = traj[:, :, :self.dim]
        y = traj[-1]
        s = trainer.scale
        stride = self.config.trajectory_stride
        traj_2d = (traj * s).cpu() if stride > 0 else None
        self._display_evolution(
            (x * s).cpu(), (y * s).cpu(), (y_target * s).cpu(),
            trajectories=traj_2d, trajectory_stride=stride,
        )
        fig = plt.gcf()
        fig.suptitle(f"Iter {epoch}")
        self._accumulate_fig(fig)
        self._save_fig(fig, "coupling", f"coupling_{epoch:04d}.png")

    def plot_distributions_after_advance_log(self, time_step: int, trainer: "Trainer"):
        """Called from trainer after advance; saves the advance plot."""
        n_samples = 500
        t_tensor = torch.full((n_samples, 1), (time_step + 1) / trainer.config.iterations, device=trainer.device)
        mu_s = trainer.flow_matching.interpolate(
            trainer.initial_coupling.sample_init(n_samples).to(trainer.device),
            trainer.new_source.sample(n_samples).to(trainer.device),
            t_tensor, 0.0).cpu()
        nu_s = trainer.flow_matching.interpolate(
            trainer.initial_coupling.sample_target(n_samples).to(trainer.device),
            trainer.new_target.sample(n_samples).to(trainer.device),
            t_tensor, 0.0).cpu()
        x_c, y_c = trainer.current_coupling.coupling_sample(250)
        x_c, y_c = x_c.cpu(), y_c.cpu()
        s = trainer.scale
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(mu_s[:, 0] * s, mu_s[:, 1] * s, s=4, alpha=0.3, c="blue", label=r"$\mu_t$")
        ax.scatter(nu_s[:, 0] * s, nu_s[:, 1] * s, s=4, alpha=0.3, c="red",  label=r"$\nu_t$")
        ax.scatter(x_c[:, 0] * s, x_c[:, 1] * s, s=8, alpha=0.7, c="purple", marker="x", label="x coupling")
        ax.scatter(y_c[:, 0] * s, y_c[:, 1] * s, s=8, alpha=0.7, c="orange", marker="x", label="y coupling")
        for j in range(x_c.shape[0]):
            ax.plot([x_c[j, 0] * s, y_c[j, 0] * s], [x_c[j, 1] * s, y_c[j, 1] * s],
                    "k-", alpha=0.1, linewidth=0.5)
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
        ax.set_title(f"Coupling after advance — iter {time_step}")
        self._accumulate_fig(fig)
        self._save_fig(fig, "coupling", f"advance_{time_step:04d}.png")


# ---------------------------------------------------------------------------
# 3-D logging visualizer
# ---------------------------------------------------------------------------

class LoggingVisualizer3D(_LoggingMixin, Visualizer3D):
    def __init__(self, config: VisualizeConfig, transport_map: Callable,
                 output_dir: str):
        Visualizer3D.__init__(self, config, transport_map)
        self._init_logging(output_dir)

    def display(self, epoch: int, trainer: "Trainer"):
        if epoch % self.config.show_each != 0:
            return
        x, _ = trainer.current_coupling.coupling_sample(self.config.n_trajectories)
        y_target = trainer.new_target.sample(self.config.n_trajectories)
        x, y_target = x.to(trainer.device), y_target.to(trainer.device)
        traj = trainer.flow_matching.evolve_trajectory(
            x, trainer.strategy.get_flow(epoch, trainer.device)[0],
            num_steps=25, sigma=trainer.config.sigma, extend=True,
        )
        traj = traj[:, :, :self.dim]
        y = traj[-1]
        s = trainer.scale
        stride = self.config.trajectory_stride
        traj_3d = (traj * s).cpu() if stride > 0 else None
        self._display_evolution(
            (x * s).cpu(), (y * s).cpu(), (y_target * s).cpu(),
            trajectories=traj_3d, trajectory_stride=stride,
        )
        fig = plt.gcf()
        fig.suptitle(f"Iter {epoch}")
        self._accumulate_fig(fig)
        self._save_fig(fig, "coupling", f"coupling_{epoch:04d}.png")

    def _save_marginals_plot(self, time_step, mu_t, nu_t, x_coup, y_coup):
        """3D marginals scatter (project onto first 2 dims for readability)."""
        if mu_t.dim() != 2 or mu_t.shape[1] < 2:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        kw = dict(s=4, alpha=0.35)
        axes[0].scatter(mu_t[:, 0], mu_t[:, 1], c="royalblue", label=r"$\mu_t$ theory", **kw)
        axes[0].scatter(x_coup[:, 0], x_coup[:, 1], c="darkorange", label=r"$\pi_t^{(1)}$", **kw)
        axes[0].legend(fontsize=7); axes[0].set_aspect("equal")
        axes[0].set_title(f"Source marginal (xy) — iter {time_step}")

        axes[1].scatter(nu_t[:, 0], nu_t[:, 1], c="firebrick", label=r"$\nu_t$ theory", **kw)
        axes[1].scatter(y_coup[:, 0], y_coup[:, 1], c="limegreen", label=r"$\pi_t^{(2)}$", **kw)
        axes[1].legend(fontsize=7); axes[1].set_aspect("equal")
        axes[1].set_title(f"Target marginal (xy) — iter {time_step}")

        fig.tight_layout()
        self._save_fig(fig, "marginals", f"marginals_{time_step:04d}.png")


# ---------------------------------------------------------------------------
# Image logging visualizer
# ---------------------------------------------------------------------------

class LoggingVisualizerImage(_LoggingMixin, VisualizerImage):
    def __init__(self, config: VisualizeConfig, img_shape: Tuple[int, int, int],
                 output_dir: str):
        VisualizerImage.__init__(self, config, img_shape)
        self._init_logging(output_dir)

    def display(self, epoch: int, trainer: "Trainer"):
        if epoch % self.config.show_each != 0:
            return
        n = min(6, self.config.n_trajectories)
        x, y_target = trainer.current_coupling.coupling_sample(n)
        x, y_target = x.to(trainer.device), y_target.to(trainer.device)
        y = trainer.flow_matching.evolve(
            x, trainer.strategy.get_flow(epoch, trainer.device)[0],
            num_steps=25, sigma=trainer.config.sigma, extend=True,
        )
        if is_image_tensor(y):
            y = y[:, :self.img_shape[0], :, :]
        s = trainer.scale
        self._display_evolution((x * s).cpu(), (y * s).cpu(), (y_target * s).cpu())
        fig = plt.gcf()
        fig.suptitle(f"Iter {epoch}")
        self._accumulate_fig(fig)
        self._save_fig(fig, "coupling", f"coupling_{epoch:04d}.png")

    def _display_evolution(self, start, end, target=None, **kwargs):
        """Save image grid to file and show inline."""
        n = min(start.shape[0], 6)
        rows = 3 if target is not None else 2
        fig, axes = plt.subplots(rows, n, figsize=(2 * n, 2 * rows))
        if n == 1:
            axes = axes.reshape(rows, 1)

        def _show(ax, tensor, i, title):
            if tensor.dim() == 4:
                img = tensor[i].squeeze(0).numpy()
            else:
                img = tensor[i].reshape(self.img_shape[1], self.img_shape[2]).numpy()
            ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
            ax.set_title(title, fontsize=7)
            ax.axis("off")

        for i in range(n):
            _show(axes[0, i], start, i, "Source")
            _show(axes[1, i], end, i, "Transported")
            if target is not None:
                _show(axes[2, i], target, i, "Target")
        plt.tight_layout()

    def _save_marginals_plot(self, time_step, mu_t, nu_t, x_coup, y_coup):
        """Image marginals: save a grid of source/target coupling samples."""
        n = min(8, x_coup.shape[0])
        C, H, W = self.img_shape
        fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
        if n == 1:
            axes = axes.reshape(2, 1)

        def _show(ax, tensor, i, title):
            img = tensor[i]
            if img.dim() == 3:
                img = img.squeeze(0)
            elif img.dim() == 1:
                img = img.reshape(H, W)
            ax.imshow(img.float().numpy(), cmap="gray", vmin=-1, vmax=1)
            ax.set_title(title, fontsize=6)
            ax.axis("off")

        for i in range(n):
            _show(axes[0, i], x_coup, i, f"src {i}")
            _show(axes[1, i], y_coup, i, f"tgt {i}")
        fig.suptitle(f"Coupling samples — iter {time_step}", fontsize=9)
        plt.tight_layout()
        self._save_fig(fig, "marginals", f"marginals_{time_step:04d}.png")
