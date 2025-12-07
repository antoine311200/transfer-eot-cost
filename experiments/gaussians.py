import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchcfm.models.models import MLP

from dualwcfm.distributions import CouplingDistribution, RotatedShapeCoupling, Simple2DGaussianDistribution
from dualwcfm.trainers import IterativeTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define coupling distribution and new source/target distributions
coupling_dist = RotatedShapeCoupling(
    centers=[[0,0]],
    source_scale=0.5,
    target_scale=0.25,
    rotation_degrees=0.0,
    source_shift=(0.0, 0.0),
    target_shift=(5.0, 5.0),
    device=device
)
new_source_dist = Simple2DGaussianDistribution(
    mean=torch.tensor([0.0, 5.0], device=device),
    std=torch.tensor([0.5, 0.5], device=device)
)
new_target_dist = Simple2DGaussianDistribution(
    mean=torch.tensor([5.0, 0.0], device=device),
    std=torch.tensor([0.25, 0.25], device=device)
)

plt.figure(figsize=(8, 8))
coupling_samples = coupling_dist.coupling_sample(500)
source_samples = coupling_samples[0].cpu().numpy()
target_samples = coupling_samples[1].cpu().numpy()
new_source_samples = new_source_dist.sample(500).cpu().numpy()
new_target_samples = new_target_dist.sample(500).cpu().numpy()

plt.scatter(source_samples[:, 0], source_samples[:, 1], c='red', alpha=0.5, label='Original Source Dist Samples')
plt.scatter(target_samples[:, 0], target_samples[:, 1], c='green', alpha=0.5, label='Original Target Dist Samples')
plt.scatter(new_source_samples[:, 0], new_source_samples[:, 1], c='blue', alpha=0.5, label='New Source Dist Samples')
plt.scatter(new_target_samples[:, 0], new_target_samples[:, 1], c='orange', alpha=0.5, label='New Target Dist Samples')
plt.title('Samples from New Source and Target Distributions')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# Train iterative method

from dualwcfm.trainers import IterativeTrainer


f_net = MLP(dim=2, out_dim=1, w=64).to(device)
g_net = MLP(dim=2, out_dim=1, w=64).to(device)
flow_model = MLP(dim=2, out_dim=2, time_varying=True, w=64).to(device)

iterative_trainer = IterativeTrainer(
    coupling_dist=coupling_dist,
    source_dist=new_source_dist,
    target_dist=new_target_dist,
)
info, cfm_nets = iterative_trainer.train(
    n_iterations=10,
    f_net=f_net,
    g_net=g_net,
    flow_network=flow_model,
    batch_size=64,
    epochs=1000,
    learning_rate=1e-3,
    sigma=0.1,
    return_cfm=True,
    plot_progress=True,
    device=device,
)


fig, axes = plt.subplots(len(info), 2, figsize=(12, 4 * len(info)))
for i, (alpha, metrics) in enumerate(info.items()):
    axes[i, 0].plot(metrics['dual_info']["epoch_train_losses"], label='Dual Train Loss', color='blue', linewidth=1)
    axes[i, 0].plot([e for e, _ in metrics['dual_info']['eval_losses']], [l for _, l in metrics['dual_info']['eval_losses']], label='Dual Eval Loss', color='green', linewidth=1)
    axes[i, 0].set_title(f'Dual Loss at alpha={alpha:.2f}')
    axes[i, 0].set_xlabel('Epoch')
    axes[i, 0].set_ylabel('Loss')
    axes[i, 0].legend()

    axes[i, 1].plot(metrics['weighted_info']["epoch_train_losses"], label='Weighted CFM Train Loss', color='orange', linewidth=1)
    axes[i, 1].plot([e for e, _ in metrics['weighted_info']['eval_losses']], [l for _, l in metrics['weighted_info']['eval_losses']], label='Weighted CFM Val Loss', color='red', linewidth=1)
    axes[i, 1].set_title(f'Weighted CFM Loss at alpha={alpha:.2f}')
    axes[i, 1].set_xlabel('Epoch')
    axes[i, 1].set_ylabel('Loss')
    axes[i, 1].legend()
plt.tight_layout()
plt.show()

# Create a len(info) x 1 subplot of trajectories for each interpolation step from the new source to the new target distribution
from dualwcfm.models.cfm import ConditionalFlowMatching
import numpy as np

n_plots = len(cfm_nets)
n_cols = min(4, n_plots)
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)
if hasattr(axes, "flat"): axes_flat = list(axes.flat)
else: axes_flat = [axes]

time_steps = 10
n_trajectories = 500

for i, (alpha, cfm_net) in enumerate(cfm_nets.items()):
    ax = axes_flat[i]

    t = torch.linspace(0, 1, steps=time_steps).to(device)
    with torch.no_grad():
        samples = new_source_dist.sample(n_trajectories).to(device)

        cfm_new = ConditionalFlowMatching(
            source_dist=new_source_dist,
            target_dist=new_target_dist,
            network=cfm_net,
        )
        trajectories = cfm_new.get_trajectory(x0=samples, n_timesteps=time_steps).cpu().numpy()

    from matplotlib.collections import LineCollection

    segments = []
    for j in range(n_trajectories):
        points = trajectories[:, j, :]
        segs = np.stack([points[:-1], points[1:]], axis=1)
        segments.extend(segs)

    lc = LineCollection(segments, colors='blue', alpha=0.15, linewidths=0.5, zorder=1)
    ax.add_collection(lc)

    start_points = trajectories[0, :, :]
    end_points = trajectories[-1, :, :]

    ax.scatter(start_points[:, 0], start_points[:, 1], c='red', s=5, alpha=0.5, zorder=4, label="Start")
    ax.scatter(end_points[:, 0], end_points[:, 1], c='green', s=5, alpha=0.5, zorder=4, label="End")
    ax.set_title(f"α = {alpha:.2f}", fontsize=10)
    ax.grid(True, alpha=0.2)

    if i == 0:
        ax.legend(loc="upper right", fontsize=8)

    if i >= (n_rows - 1) * n_cols:
        ax.set_xlabel("x", fontsize=9)
    if i % n_cols == 0:
        ax.set_ylabel("y", fontsize=9)

for i in range(len(cfm_nets), len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()