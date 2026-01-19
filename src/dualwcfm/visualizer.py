from typing import Optional, Tuple
from dualwcfm.cfm import ConditionalFlowMatching
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class VisualizationManager:
    """Handles visualization of training progress."""

    @staticmethod
    def plot_iteration_diagnostics(
        step: int,
        x_ref: torch.Tensor,
        y_ref: torch.Tensor,
        f_net: nn.Module,
        g_net: nn.Module,
        flow_model: nn.Module,
        device: torch.device,
    ):
        """Visualize coupling, learned weights, and vector field."""
        x_ref_np = x_ref.detach().cpu().numpy()
        y_ref_np = y_ref.detach().cpu().numpy()

        with torch.no_grad():
            a = f_net(x_ref).view(-1)
            b = g_net(y_ref).view(-1)
            log_weights = a + b

        # Create grid for vector field
        grid_n = 20
        x = np.linspace(-10, 10, grid_n)
        y = np.linspace(-10, 10, grid_n)
        X, Y = np.meshgrid(x, y)
        grid = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32).to(device)

        # Compute vector field
        t_zeros = torch.zeros(grid.shape[0], 1).to(device)
        with torch.no_grad():
            vectors = flow_model(torch.cat([grid, grid, t_zeros], dim=1)).cpu().numpy()

        U = vectors[:, 0].reshape(grid_n, grid_n)
        V = vectors[:, 1].reshape(grid_n, grid_n)

        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Source samples with weights
        sc = axes[0].scatter(
            x_ref_np[:, 0], x_ref_np[:, 1], c=log_weights.cpu().numpy(), cmap="viridis", s=5, alpha=0.6
        )
        axes[0].set_title(f"Step {step}: Source Samples\n(Color = Log Learned Weight)")
        plt.colorbar(sc, ax=axes[0])
        axes[0].set_xlim(-10, 10)
        axes[0].set_ylim(-10, 10)

        # Target samples
        axes[1].scatter(y_ref_np[:, 0], y_ref_np[:, 1], c=log_weights.cpu().numpy(), s=5, alpha=0.3)
        axes[1].set_title(f"Step {step}: Target Samples")
        axes[1].set_xlim(-10, 10)
        axes[1].set_ylim(-10, 10)

        # Vector field
        axes[2].quiver(X, Y, U, V, color="blue", alpha=0.8)
        axes[2].set_title(f"Step {step}: Learned Flow Field (v_t)")
        axes[2].set_xlim(-10, 10)
        axes[2].set_ylim(-10, 10)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cfm_trajectories(
        cfm: ConditionalFlowMatching,
        source_samples: torch.Tensor,
        target_dist,
        n_samples: int = 500,
        n_timesteps: int = 20,
        title: str = "W-CFM Trajectories",
    ):
        """Plot CFM trajectories."""
        time = torch.linspace(0, 1, n_timesteps)
        x_init_extended = torch.cat([source_samples, source_samples], dim=1)
        traj = cfm.get_trajectory(x0=x_init_extended, n_timesteps=n_timesteps)
        target_samples = target_dist.sample(n_samples).cpu().numpy()

        plot_2d_flow_trajectories(
            trajectories=traj, t=time, n_trajectories=n_samples, target_points=target_samples, title=title
        )
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
