from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING, Callable

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator

from dualwcfm.config import VisualizeConfig
from dualwcfm.utils import is_image_tensor

if TYPE_CHECKING:
    from dualwcfm.trainer import Trainer


class Visualizer:
    def __init__(self, config: VisualizeConfig, transport_map: Callable):
        self.config = config
        self.transport_map = transport_map
        self.dim = 2

    def display(self, epoch: int, trainer: Trainer):
        if epoch % self.config.show_each == 0:
            x, _ = trainer.current_coupling.coupling_sample(self.config.n_trajectories)
            y_target = trainer.new_target.sample(self.config.n_trajectories)
            x, y_target = x.to(trainer.device), y_target.to(trainer.device)
            y = trainer.flow_matching.evolve(
                x,
                trainer.strategy.get_flow(epoch, trainer.device)[0],
                num_steps=25,
                sigma=trainer.config.sigma,
                extend=True,
            )
            y = y[:, : self.dim]
            s = trainer.scale
            self._display_evolution(
                (x * s).cpu(), (y * s).cpu(), (y_target * s).cpu()
            )
            plt.show()

    def transport_cost(self, x: torch.Tensor, y: torch.Tensor):
        if self.transport_map is None:
            return None
        return y - self.transport_map(x)

    # ------------------------------------------------------------------
    # Diagnostic: sliced Wasserstein distances between marginals
    # ------------------------------------------------------------------

    def _sliced_wasserstein(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        n_proj: int = 200,
        p: int = 1,
    ) -> float:
        """Sliced Wasserstein-p distance between two point clouds.

        Projects X and Y onto *n_proj* random unit directions, computes the
        1-D Wasserstein-p for each projection, and returns the average (for
        p=1) or the sqrt-of-average-squared (for p=2), matching the standard
        sliced Wasserstein definition.

        Args:
            X: (n, d) float tensor.
            Y: (m, d) float tensor.
            n_proj: Number of random 1-D projections.
            p: Wasserstein order — 1 or 2.

        Returns:
            Scalar sliced Wasserstein-p estimate.
        """
        d = X.shape[1]
        device = X.device
        dtype = X.dtype

        # Random unit directions: (d, n_proj)
        dirs = torch.randn(d, n_proj, device=device, dtype=dtype)
        dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp(min=1e-8)

        # Scalar projections: (n, n_proj) and (m, n_proj)
        X_proj = X @ dirs
        Y_proj = Y @ dirs

        # Sort along the sample axis
        X_sorted = X_proj.sort(dim=0).values
        Y_sorted = Y_proj.sort(dim=0).values

        # Align sizes via uniform quantile sub-indexing
        n, m = X_sorted.shape[0], Y_sorted.shape[0]
        if n != m:
            n_q = min(n, m)
            idx_X = torch.linspace(0, n - 1, n_q, device=device).long()
            idx_Y = torch.linspace(0, m - 1, n_q, device=device).long()
            X_sorted = X_sorted[idx_X]
            Y_sorted = Y_sorted[idx_Y]

        diff = (X_sorted - Y_sorted).abs()  # (n_q, n_proj)
        if p == 1:
            return diff.mean().item()
        else:  # p == 2 — SW_2 = sqrt(E[W_2^2])
            return diff.pow(2).mean().sqrt().item()

    def analyse(
        self,
        time_step: int,
        trainer,
        n_samples: int = 2000,
        n_proj: int = 2000,
    ) -> dict:
        """Measure drift between theoretical marginals and current coupling.

        Computes sliced Wasserstein-1 and -2 between:
          * mu_t  (noiseless linear interpolation of mu_0 → mu_1 at global
                   time t = time_step / iterations)
            and the *first* marginal of ``trainer.current_coupling``.
          * nu_t  (noiseless linear interpolation of nu_0 → nu_1 at time t)
            and the *second* marginal of ``trainer.current_coupling``.

        This quantifies how much error has accumulated in the coupling
        relative to the ideal linear-interpolation path.

        Args:
            time_step: Current training iteration index.
            trainer: The :class:`~dualwcfm.trainer.Trainer` instance.
            n_samples: Number of samples used for the distance estimates.
            n_proj: Number of random projections for sliced Wasserstein.

        Returns:
            dict with keys ``'W1_mu'``, ``'W2_mu'``, ``'W1_nu'``, ``'W2_nu'``.
        """
        device = trainer.device
        config = trainer.config
        t = time_step / config.iterations
        t_tensor = torch.full((n_samples, 1), t, device=device)

        with torch.no_grad():
            # Theoretical mu_t: noiseless linear interpolation at time t
            mu_t = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_init(n_samples).to(device),
                trainer.new_source.sample(n_samples).to(device),
                t_tensor,
                sigma=0.0,
            )
            # Theoretical nu_t: noiseless linear interpolation at time t
            nu_t = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_target(n_samples).to(device),
                trainer.new_target.sample(n_samples).to(device),
                t_tensor,
                sigma=0.0,
            )
            # First and second marginals of the current coupling
            x_coupling, y_coupling = trainer.current_coupling.coupling_sample(n_samples)
            x_coupling = x_coupling.to(device)
            y_coupling = y_coupling.to(device)

        # Flatten to (N, d) — works for both vector and image data
        n_x = x_coupling.shape[0]
        n_y = y_coupling.shape[0]
        mu_t_flat = mu_t.view(n_samples, -1).float()
        nu_t_flat = nu_t.view(n_samples, -1).float()
        x_flat = x_coupling.view(n_x, -1).float()
        y_flat = y_coupling.view(n_y, -1).float()

        results = {
            "W1_mu": self._sliced_wasserstein(mu_t_flat, x_flat, n_proj=n_proj, p=1),
            "W2_mu": self._sliced_wasserstein(mu_t_flat, x_flat, n_proj=n_proj, p=2),
            "W1_nu": self._sliced_wasserstein(nu_t_flat, y_flat, n_proj=n_proj, p=1),
            "W2_nu": self._sliced_wasserstein(nu_t_flat, y_flat, n_proj=n_proj, p=2),
        }
        return results

    def _display_evolution(
        self, start_points: torch.Tensor, end_points: torch.Tensor, target_points: Optional[torch.Tensor] = None
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        segments = []
        for j in range(min(self.config.n_trajectories, start_points.shape[0])):
            points = torch.stack([start_points[j], end_points[j]], dim=0).numpy()
            segs = np.stack([points[:-1], points[1:]], axis=1)
            segments.extend(segs)

        if target_points is not None:
            ax.scatter(target_points[:, 0], target_points[:, 1], c="orange", s=5, alpha=0.5, zorder=4, label="Target")
        ax.scatter(start_points[:, 0], start_points[:, 1], c="red", s=5, alpha=0.5, zorder=4, label="Start")
        ax.scatter(end_points[:, 0], end_points[:, 1], c="green", s=5, alpha=0.5, zorder=4, label="End")
        mid_points = (start_points + end_points) / 2
        ax.scatter(mid_points[:, 0], mid_points[:, 1], c="blue", s=5, alpha=0.5, zorder=4, label="Mid")
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.grid(True, alpha=0.2)

        lc = LineCollection(segments, colors="black", alpha=0.75, linewidths=0.8, zorder=10)
        ax.add_collection(lc)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.set_aspect("equal")

    def display_marginal(self, epoch: int, trainer: Trainer, model: nn.Module,
                         source_dist, target_dist, title: str = "Marginal Flow"):
        """Display marginal flow evolution during training.

        Args:
            epoch: Current epoch number
            trainer: The trainer instance
            model: The marginal flow model (mu_bridge or nu_bridge)
            source_dist: Source distribution to sample from
            target_dist: Target distribution to compare against
            title: Title for the plot
        """
        if self.config.show_each_marginal <= 0 or epoch % self.config.show_each_marginal != 0:
            return

        n_samples = min(self.config.n_trajectories, 500)
        start_points = source_dist.sample(n_samples).to(trainer.device)

        with torch.no_grad():
            end_points = trainer.flow_matching.evolve(
                start_points, model, num_steps=50, sigma=trainer.config.sigma, extend=False
            )

        target_points = target_dist.sample(n_samples).to(trainer.device)

        s = trainer.scale
        self._display_evolution(
            (start_points * s).cpu(), (end_points * s).cpu(), (target_points * s).cpu()
        )
        plt.suptitle(f"{title} - Epoch {epoch}")
        plt.show()

    def display_pretrain(self, epoch: int, trainer: Trainer, model: nn.Module, backward: bool = False):
        """Display pretraining flow evolution during training.

        Args:
            epoch: Current epoch number
            trainer: The trainer instance
            model: The CFM model being pretrained
            backward: Whether this is backward flow (swaps source/target)
        """
        if self.config.show_each_pretrain <= 0 or epoch % self.config.show_each_pretrain != 0:
            return

        n_samples = min(self.config.n_trajectories, 500)
        x0, x1 = trainer.initial_coupling.coupling_sample(n_samples)
        x0, x1 = x0.to(trainer.device), x1.to(trainer.device)

        if backward:
            x0, x1 = x1, x0

        with torch.no_grad():
            end_points = trainer.flow_matching.evolve(
                x0, model, num_steps=50, sigma=trainer.config.sigma, extend=True
            )
            # Extract first half of output (evolved part)
            end_points = end_points[:, :self.dim]

        s = trainer.scale
        direction = "Backward" if backward else "Forward"
        self._display_evolution(
            (x0 * s).cpu(), (end_points * s).cpu(), (x1 * s).cpu()
        )
        plt.suptitle(f"Pretrain {direction} Flow - Epoch {epoch}")
        plt.show()


class Visualizer3D(Visualizer):
    def __init__(self, config: VisualizeConfig):
        super().__init__(config)
        self.dim = 3

    def _display_evolution(
        self, start_points: torch.Tensor, end_points: torch.Tensor, target_points: Optional[torch.Tensor] = None
    ):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        for j in range(self.config.n_trajectories):
            points = torch.stack([start_points[j], end_points[j]], dim=0).numpy()
            ax.plot(points[:, 0], points[:, 1], points[:, 2], c="black", alpha=0.25, linewidth=0.8, zorder=10)

        if target_points is not None:
            ax.scatter(
                target_points[:, 0],
                target_points[:, 1],
                target_points[:, 2],
                c="orange",
                s=3,
                alpha=0.5,
                zorder=4,
                label="Target",
            )
        ax.scatter(
            start_points[:, 0],
            start_points[:, 1],
            start_points[:, 2],
            c="red",
            s=3,
            alpha=0.5,
            zorder=4,
            label="Start",
        )
        ax.scatter(
            end_points[:, 0], end_points[:, 1], end_points[:, 2], c="green", s=3, alpha=0.5, zorder=4, label="End"
        )
        mid_points = (start_points + end_points) / 2
        ax.scatter(
            mid_points[:, 0], mid_points[:, 1], mid_points[:, 2], c="blue", s=3, alpha=0.5, zorder=4, label="Mid"
        )
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.zaxis.set_major_locator(MultipleLocator(0.25))
        ax.grid(True, alpha=0.2)

        L = 3  # length of the axis lines

        ax.plot([-L, L], [0, 0], [0, 0], color="k", linewidth=1)  # X-axis
        ax.plot([0, 0], [-L, L], [0, 0], color="k", linewidth=1)  # Y-axis
        ax.plot([0, 0], [0, 0], [-L, L], color="k", linewidth=1)  # Z-axis

        # Set limits
        ax.set_xlim([-L, L])
        ax.set_ylim([-L, L])
        ax.set_zlim([-L, L])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore
        ax.legend()
        ax.set_box_aspect([1, 1, 1])  # type: ignore


class VisualizerImage(Visualizer):
    def __init__(self, config: VisualizeConfig, img_shape: Tuple[int, int, int]):
        """
        Args:
            config: Visualization configuration
            img_shape: Image shape as (C, H, W)
        """
        super().__init__(config)
        self.img_shape = img_shape  # (C, H, W)
        self.dim = img_shape[0] * img_shape[1] * img_shape[2]

    def display(self, epoch: int, trainer: Trainer):
        """Override to handle image tensors properly."""
        if epoch % self.config.show_each == 0:
            n_samples = min(self.config.n_trajectories, 6)  # Show max 6 images
            x, y_target = trainer.current_coupling.coupling_sample(n_samples)
            # y_target = trainer.new_target.sample(n_samples)
            x, y_target = x.to(trainer.device), y_target.to(trainer.device)
            y = trainer.flow_matching.evolve(
                x,
                trainer.strategy.get_flow(epoch, trainer.device)[0],
                num_steps=25,
                sigma=trainer.config.sigma,
                extend=True,
            )
            # For images, extract first half of channels (the evolved part)
            if is_image_tensor(y):
                y = y[:, :self.img_shape[0], :, :]
            s = trainer.scale
            self._display_evolution(
                (x * s).cpu(), (y * s).cpu(), (y_target * s).cpu()
            )
            plt.show()

    def _display_evolution(
        self, start_points: torch.Tensor, end_points: torch.Tensor, target_points: Optional[torch.Tensor] = None
    ):
        """Display image evolution.

        Args:
            start_points: Tensor of shape (N, C, H, W) or (N, C*H*W)
            end_points: Tensor of shape (N, C, H, W) or (N, C*H*W)
            target_points: Optional tensor of shape (N, C, H, W) or (N, C*H*W)
        """
        n = min(start_points.shape[0], 6)  # Show max 6 images
        fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))

        # Handle case where n=1 (axes won't be 2D)
        if n == 1:
            axes = axes.reshape(3, 1)

        for i in range(n):
            # Handle both 4D (N, C, H, W) and 2D (N, flat) tensors
            if start_points.dim() == 4:
                start_img = start_points[i].squeeze(0).cpu().numpy()  # Remove channel dim for grayscale
                end_img = end_points[i].squeeze(0).cpu().numpy()
            else:
                start_img = start_points[i].reshape(self.img_shape[1], self.img_shape[2]).cpu().numpy()
                end_img = end_points[i].reshape(self.img_shape[1], self.img_shape[2]).cpu().numpy()

            axes[0, i].imshow(start_img, cmap="gray", vmin=-1, vmax=1)
            axes[0, i].set_title("Start")
            axes[0, i].axis("off")

            axes[1, i].imshow(end_img, cmap="gray", vmin=-1, vmax=1)
            axes[1, i].set_title("End")
            axes[1, i].axis("off")

            if target_points is not None:
                if target_points.dim() == 4:
                    target_img = target_points[i].squeeze(0).cpu().numpy()
                else:
                    target_img = target_points[i].reshape(self.img_shape[1], self.img_shape[2]).cpu().numpy()
                axes[2, i].imshow(target_img, cmap="gray", vmin=-1, vmax=1)
                axes[2, i].set_title("Target")
                axes[2, i].axis("off")
            else:
                axes[2, i].axis("off")

        plt.tight_layout()

    def display_marginal(self, epoch: int, trainer: Trainer, model: nn.Module,
                         source_dist, target_dist, title: str = "Marginal Flow"):
        """Display marginal flow evolution during training for images."""
        if self.config.show_each_marginal <= 0 or epoch % self.config.show_each_marginal != 0:
            return

        n_samples = min(self.config.n_trajectories, 6)
        start_points = source_dist.sample(n_samples).to(trainer.device)

        with torch.no_grad():
            end_points = trainer.flow_matching.evolve(
                start_points, model, num_steps=50, sigma=trainer.config.sigma, extend=False
            )

        target_points = target_dist.sample(n_samples).to(trainer.device)

        s = trainer.scale
        self._display_evolution(
            (start_points * s).cpu(), (end_points * s).cpu(), (target_points * s).cpu()
        )
        plt.suptitle(f"{title} - Epoch {epoch}")
        plt.show()

    def display_pretrain(self, epoch: int, trainer: Trainer, model: nn.Module, backward: bool = False):
        """Display pretraining flow evolution during training for images."""
        if self.config.show_each_pretrain <= 0 or epoch % self.config.show_each_pretrain != 0:
            return

        n_samples = min(self.config.n_trajectories, 6)
        x0, x1 = trainer.initial_coupling.coupling_sample(n_samples)
        x0, x1 = x0.to(trainer.device), x1.to(trainer.device)

        if backward:
            x0, x1 = x1, x0

        with torch.no_grad():
            end_points = trainer.flow_matching.evolve(
                x0, model, num_steps=50, sigma=trainer.config.sigma, extend=True
            )
            # Extract first half of channels (evolved part)
            if is_image_tensor(end_points):
                end_points = end_points[:, :self.img_shape[0], :, :]

        s = trainer.scale
        direction = "Backward" if backward else "Forward"
        self._display_evolution(
            (x0 * s).cpu(), (end_points * s).cpu(), (x1 * s).cpu()
        )
        plt.suptitle(f"Pretrain {direction} Flow - Epoch {epoch}")
        plt.show()



def plot_2d_flow_trajectories(
    trajectories: torch.Tensor,
    t: torch.Tensor,
    n_trajectories: int = 50,
    target_points: Optional[torch.Tensor] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Flow Trajectories",
):
    """Plot sample flow trajectories"""
    fig, ax = plt.subplots(figsize=(8, 8))
    trajectories_np = trajectories.cpu().numpy()

    n_trajectories = min(n_trajectories, trajectories.shape[1])

    from matplotlib.collections import LineCollection

    segments = []
    for j in range(n_trajectories):
        points = trajectories_np[:, j, :2]
        segs = np.stack([points[:-1], points[1:]], axis=1)
        segments.extend(segs)

    start_points = trajectories_np[0, :, :2]
    end_points = trajectories_np[-1, :, :2]

    if target_points is not None:
        ax.scatter(target_points[:, 0], target_points[:, 1], c="orange", s=5, alpha=0.5, zorder=4, label="Target")
    ax.scatter(start_points[:, 0], start_points[:, 1], c="red", s=5, alpha=0.5, zorder=4, label="Start")
    ax.scatter(end_points[:, 0], end_points[:, 1], c="green", s=5, alpha=0.5, zorder=4, label="End")
    mid_points = trajectories_np[trajectories_np.shape[0] // 2, :, :2]
    ax.scatter(mid_points[:, 0], mid_points[:, 1], c="blue", s=5, alpha=0.5, zorder=4, label="Mid")
    ax.grid(True, alpha=0.2)
    ax.set_title(title)

    lc = LineCollection(segments, colors="gray", alpha=0.5, linewidths=0.8, zorder=1)
    ax.add_collection(lc)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()