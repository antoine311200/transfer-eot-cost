from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_2d_flow_trajectories(
    trajectories: torch.Tensor,
    t: torch.Tensor,
    n_trajectories: int = 50,
    target_points: Optional[torch.Tensor] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Flow Trajectories"
):
    """Plot sample flow trajectories"""
    fig, ax = plt.subplots(figsize=(6, 6))
    trajectories_np = trajectories.cpu().numpy()

    n_trajectories = min(n_trajectories, trajectories.shape[1])

    from matplotlib.collections import LineCollection

    segments = []
    for j in range(n_trajectories):
        points = trajectories_np[:, j, :]
        segs = np.stack([points[:-1], points[1:]], axis=1)
        segments.extend(segs)

    lc = LineCollection(segments, colors='blue', alpha=0.15, linewidths=0.5, zorder=1)
    ax.add_collection(lc)

    start_points = trajectories_np[0, :, :]
    end_points = trajectories_np[-1, :, :]

    if target_points is not None:
        ax.scatter(target_points[:, 0], target_points[:, 1], c='orange', s=5, alpha=0.5, zorder=4, label="Target")
    ax.scatter(start_points[:, 0], start_points[:, 1], c='red', s=5, alpha=0.5, zorder=4, label="Start")
    ax.scatter(end_points[:, 0], end_points[:, 1], c='green', s=5, alpha=0.5, zorder=4, label="End")
    ax.grid(True, alpha=0.2)
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    plt.show()