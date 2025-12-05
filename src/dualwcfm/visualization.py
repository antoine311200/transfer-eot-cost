from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_2d_flow_trajectories(
    trajectories: torch.Tensor,
    t: torch.Tensor,
    n_trajectories: int = 50,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Flow Trajectories"
):
    """Plot sample flow trajectories"""
    fig, ax = plt.subplots(figsize=(6, 6))
    trajectories_np = trajectories.cpu().numpy()

    # for i in range(min(n_trajectories, trajectories.shape[1])):
    #     traj = trajectories_np[:, i, :]
    #     plt.plot(traj[:, 0], traj[:, 1], "b-", alpha=0.3, linewidth=0.5)
    #     plt.scatter(traj[0, 0], traj[0, 1], c="red", s=10, alpha=0.7, label="Start" if i == 0 else "")
    #     plt.scatter(traj[-1, 0], traj[-1, 1], c="green", s=10, alpha=0.7, label="End" if i == 0 else "")
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

    # if xlim is not None:
    #     plt.xlim(xlim)
    # if ylim is not None:
    #     plt.ylim(ylim)

    # # plt.xlabel("X")
    # # plt.ylabel("Y")
    # # plt.title(title)
    # # plt.legend()
    # # plt.show()
    # plt.suptitle(title)
    # plt.show()