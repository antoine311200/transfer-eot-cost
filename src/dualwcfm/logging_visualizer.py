"""Logging visualizers that save per-iteration plots and metrics to disk.

Each iteration's key diagnostics are saved as PNG files under `output_dir/`:
  coupling/coupling_{iter}.png   — flow trajectories
  cfm/cfm_{iter}.png            — CFM vector field quiver (2D/3D)
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

from dualwcfm.config import VisualizeConfig
from dualwcfm.strategy import BidirectionalStrategy
from dualwcfm.utils import is_image_tensor
from dualwcfm.visualization import Visualizer, Visualizer3D, VisualizerImage

if TYPE_CHECKING:
    from dualwcfm.trainer import Trainer


# ---------------------------------------------------------------------------
# Shared mixin
# ---------------------------------------------------------------------------

class _LoggingMixin:
    """Adds disk-saving, CSV logging, per-type GIF timelapse, and locked axis
    limits (so all frames of a given plot type have identical boundaries)."""

    def _init_logging(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("coupling", "cfm", "weights", "marginals", "pretrain"):
            (self.output_dir / sub).mkdir(exist_ok=True)

        self._csv_path = self.output_dir / "metrics.csv"
        self._csv_fields = ["iteration", "W1_mu", "W2_mu", "W1_nu", "W2_nu",
                            "weight_mean", "weight_std", "weight_min", "weight_max"]
        with open(self._csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writeheader()

        # One frame list per plot type  →  timelapse_{key}.gif
        self._frames: dict[str, List[np.ndarray]] = defaultdict(list)
        # Locked axis limits for 2-D scatter plots (set from first iteration)
        self._axis_lim: Optional[Tuple[float, float, float, float]] = None

    # ------------------------------------------------------------------
    # Axis-limit helpers (2-D only)
    # ------------------------------------------------------------------

    def _lock_axis_lim(self, *tensors: torch.Tensor, padding: float = 0.18):
        """Compute and lock axis limits from the union of 2-D data tensors.
        Called once; subsequent calls are no-ops."""
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
        """Render fig to RGB array and store in the given timelapse bucket."""
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
        """Intercept trainer plt calls: accumulate frame then save to disk."""
        # Route duals plots to their own timelapse key
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
            (x * s).cpu(),    (y * s).cpu(),
        )
        return distances

    # ------------------------------------------------------------------
    # Per-iteration plot helpers
    # ------------------------------------------------------------------

    def _save_marginals_plot(self, time_step: int,
                             mu_t: torch.Tensor, nu_t: torch.Tensor,
                             x_coup: torch.Tensor, y_coup: torch.Tensor):
        """2-D scatter: theoretical marginals vs coupling marginals.
        Subclasses override for image / 3-D data."""
        if mu_t.dim() != 2:
            return
        self._lock_axis_lim(mu_t, nu_t, x_coup, y_coup)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        kw = dict(s=4, alpha=0.35, linewidths=0)
        axes[0].scatter(mu_t[:, 0],    mu_t[:, 1],    c="royalblue",  label=r"$\mu_t$ (theory)",   **kw)
        axes[0].scatter(x_coup[:, 0],  x_coup[:, 1],  c="darkorange", label=r"$\pi_t^{(1)}$ (coupling)", **kw)
        axes[0].legend(fontsize=7); axes[0].set_aspect("equal")
        axes[0].set_title(f"Source marginal — iter {time_step}")
        self._apply_axis_lim(axes[0])

        axes[1].scatter(nu_t[:, 0],   nu_t[:, 1],   c="firebrick",  label=r"$\nu_t$ (theory)",   **kw)
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
        print(f"  [plot] saved → {path}")

    # ------------------------------------------------------------------
    # Bidirectional helper
    # ------------------------------------------------------------------

    @staticmethod
    def _is_backward(epoch: int, trainer: "Trainer") -> bool:
        """True when the current iteration uses the backward flow model."""
        return isinstance(trainer.strategy, BidirectionalStrategy) and epoch % 2 != 0

    # ------------------------------------------------------------------
    # Timelapse
    # ------------------------------------------------------------------

    def make_timelapse_all(self, fps: int = 4):
        """Save one GIF per plot type (coupling, marginals, duals, cfm, weights)."""
        for key, frames in self._frames.items():
            if not frames:
                continue
            self._write_gif(frames, self.output_dir / f"timelapse_{key}.gif", fps)

    def _write_gif(self, frames: List[np.ndarray], path: Path, fps: int):
        try:
            import PIL.Image
            # Make all frames same size (use first frame as reference)
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
            print(f"  Timelapse [{path.stem}] → {path}")
        except ImportError:
            print("Pillow not installed; skipping GIF timelapse.")


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
        backward = self._is_backward(epoch, trainer)
        n = self.config.n_trajectories
        x, y = trainer.current_coupling.coupling_sample(n)
        x, y = x.to(trainer.device), y.to(trainer.device)

        if backward:
            start = y
            ref_target = trainer.new_source.sample(n).to(trainer.device)
        else:
            start = x
            ref_target = trainer.new_target.sample(n).to(trainer.device)

        traj = trainer.flow_matching.evolve_trajectory(
            start, trainer.strategy.get_flow(epoch, trainer.device)[0],
            num_steps=25, sigma=trainer.config.sigma, extend=True,
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
        direction = "Backward ν→μ" if backward else "Forward μ→ν"
        fig.suptitle(f"Iter {epoch} ({direction})")
        self._accumulate(fig, "coupling")
        self._save_fig(fig, "coupling", f"coupling_{epoch:04d}.png")


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
        backward = self._is_backward(epoch, trainer)
        n = self.config.n_trajectories
        x, y = trainer.current_coupling.coupling_sample(n)
        x, y = x.to(trainer.device), y.to(trainer.device)

        if backward:
            start = y
            ref_target = trainer.new_source.sample(n).to(trainer.device)
        else:
            start = x
            ref_target = trainer.new_target.sample(n).to(trainer.device)

        traj = trainer.flow_matching.evolve_trajectory(
            start, trainer.strategy.get_flow(epoch, trainer.device)[0],
            num_steps=25, sigma=trainer.config.sigma, extend=True,
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
        direction = "Backward ν→μ" if backward else "Forward μ→ν"
        fig.suptitle(f"Iter {epoch} ({direction})")
        self._accumulate(fig, "coupling")
        self._save_fig(fig, "coupling", f"coupling_{epoch:04d}.png")

    def _save_marginals_plot(self, time_step, mu_t, nu_t, x_coup, y_coup):
        """3-D: project onto xy plane for 2-D scatter."""
        if mu_t.dim() != 2 or mu_t.shape[1] < 2:
            return
        self._lock_axis_lim(mu_t, nu_t, x_coup, y_coup)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        kw = dict(s=4, alpha=0.35, linewidths=0)
        axes[0].scatter(mu_t[:, 0],   mu_t[:, 1],   c="royalblue",  label=r"$\mu_t$ theory",     **kw)
        axes[0].scatter(x_coup[:, 0], x_coup[:, 1], c="darkorange", label=r"$\pi_t^{(1)}$",      **kw)
        axes[0].legend(fontsize=7); axes[0].set_aspect("equal")
        axes[0].set_title(f"Source marginal (xy) — iter {time_step}")
        self._apply_axis_lim(axes[0])

        axes[1].scatter(nu_t[:, 0],   nu_t[:, 1],   c="firebrick",  label=r"$\nu_t$ theory",     **kw)
        axes[1].scatter(y_coup[:, 0], y_coup[:, 1], c="limegreen",  label=r"$\pi_t^{(2)}$",      **kw)
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
                 output_dir: str):
        VisualizerImage.__init__(self, config, img_shape, transport_map=None)
        self._init_logging(output_dir)

    def display(self, epoch: int, trainer: "Trainer"):
        if epoch % self.config.show_each != 0:
            return
        backward = self._is_backward(epoch, trainer)
        n = min(6, self.config.n_trajectories)
        x, y = trainer.current_coupling.coupling_sample(n)
        x, y = x.to(trainer.device), y.to(trainer.device)

        if backward:
            start = y
            ref_target = trainer.new_source.sample(n).to(trainer.device)
        else:
            start = x
            ref_target = trainer.new_target.sample(n).to(trainer.device)

        evolved = trainer.flow_matching.evolve(
            start, trainer.strategy.get_flow(epoch, trainer.device)[0],
            num_steps=25, sigma=trainer.config.sigma, extend=True,
        )
        if is_image_tensor(evolved):
            evolved = evolved[:, :self.img_shape[0], :, :]
        s = trainer.scale
        self._display_evolution((start * s).cpu(), (evolved * s).cpu(), (ref_target * s).cpu())
        fig = plt.gcf()
        direction = "Backward ν→μ" if backward else "Forward μ→ν"
        fig.suptitle(f"Iter {epoch} ({direction})")
        self._accumulate(fig, "coupling")
        self._save_fig(fig, "coupling", f"coupling_{epoch:04d}.png")

    def _display_evolution(self, start, end, target=None, **kwargs):
        n = min(start.shape[0], 6)
        rows = 3 if target is not None else 2
        fig, axes = plt.subplots(rows, n, figsize=(2 * n, 2 * rows))
        if n == 1:
            axes = axes.reshape(rows, 1)
        C, H, W = self.img_shape

        def _show(ax, tensor, i, title):
            img = tensor[i]
            img = img.squeeze(0) if img.dim() == 3 else img.reshape(H, W)
            ax.imshow(img.float().numpy(), cmap="gray", vmin=-1, vmax=1)
            ax.set_title(title, fontsize=7); ax.axis("off")

        for i in range(n):
            _show(axes[0, i], start,  i, "Source")
            _show(axes[1, i], end,    i, "Transported")
            if target is not None:
                _show(axes[2, i], target, i, "Target")
        plt.tight_layout()

    def _save_marginals_plot(self, time_step, mu_t, nu_t, x_coup, y_coup):
        """Image marginals: grid of coupling source vs target samples."""
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
