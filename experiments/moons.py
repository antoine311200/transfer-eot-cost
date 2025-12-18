# %%
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from dualwcfm.distributions import CouplingDistribution, RotatedShapeCoupling, Simple2DGaussianDistribution, RotatedDistribution, MoonDistribution

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
from dualwcfm.distributions.distribution import Distribution

class ShiftedDistribution(Distribution):
    def __init__(self, base_distribution, shift):
        self.base_distribution = base_distribution
        self.shift = shift

    def sample(self, n):
        samples = self.base_distribution.sample(n)
        return samples + self.shift.to(samples.device)

# %%
coupling_dist = RotatedShapeCoupling(
    centers=[[0,0], [1.5,0], [-1.5,0], [0,1.5], [0,-1.5]],
    source_scale=0.2,
    target_scale=0.2,
    rotation_degrees=90.0,
    source_shift=(0.0, 0.0),
    target_shift=(0.0, 0.0),
    device=device
)
new_source_dist = ShiftedDistribution(MoonDistribution(), shift=torch.tensor([5.0, 5.0]))
new_target_dist = ShiftedDistribution(RotatedDistribution(new_source_dist, angle=torch.pi / 2), shift=torch.tensor([10.0, 0.0]))

plt.figure(figsize=(8, 8))
coupling_samples = coupling_dist.coupling_sample(2000)
source_samples = coupling_samples[0].cpu().numpy()
target_samples = coupling_samples[1].cpu().numpy()
new_source_samples = new_source_dist.sample(500).cpu().numpy()
new_target_samples = new_target_dist.sample(500).cpu().numpy()

fig, axes = plt.subplots(1, 4, figsize=(25, 8))
axes[0].scatter(source_samples[:, 0], source_samples[:, 1], c='red', alpha=0.5)
axes[0].set_title('Original Source Distribution Samples')
axes[1].scatter(target_samples[:, 0], target_samples[:, 1], c='green', alpha=0.5)
axes[1].set_title('Original Target Distribution Samples')
axes[2].scatter(new_source_samples[:, 0], new_source_samples[:, 1], c='blue', alpha=0.5)
axes[2].set_title('New Source Distribution Samples')
axes[3].scatter(new_source_samples[:, 0], new_source_samples[:, 1], c='blue', alpha=0.5)
axes[3].set_title('New Source Distribution Samples')
axes[3].scatter(new_target_samples[:, 0], new_target_samples[:, 1], c='orange', alpha=0.5)
axes[3].set_title('New Target Distribution Samples')
plt.tight_layout()
plt.show()

# %%
from dualwcfm.models.cfm import torch_wrapper
from dualwcfm.trainers.alternating_trainer import AlternatingConfig, AlternatingTrainer
from dualwcfm.distributions.coupling import WeightedCouplingDistribution
from dualwcfm.trainers.wcfm import CFMTrainerConfig
from torchcfm.models.models import MLP

alternating_config = AlternatingConfig(learning_rate=1e-4, epochs=1000, batch_size=2048)

f_net = MLP(dim=2, out_dim=1, w=16, time_varying=True).to(device)
g_net = MLP(dim=2, out_dim=1, w=16, time_varying=True).to(device)

weighted_coupling_dist = WeightedCouplingDistribution(coupling_dist, size=10_000_000)

log_weight_samples = [weighted_coupling_dist.log_weights[:30].cpu().numpy()]

n_iterations = 20
for step in range(n_iterations):
    print(f"Starting iteration {step}/{n_iterations}")
    t = (step + 1) / n_iterations
    delta = 1.0 / n_iterations

    dual_trainer = AlternatingTrainer(
        coupling_dist=weighted_coupling_dist,
        source_dist=new_source_dist,
        target_dist=new_target_dist,
        f_net=f_net,
        g_net=g_net,
        device=device,
        wrapper=torch_wrapper
    )
    dual_info = dual_trainer.train(
        t=t,
        config=alternating_config,
        print_every=1,
        eval_every=1
    )

    # Update weights of the coupling distribution
    with torch.no_grad():
        time = torch.ones(weighted_coupling_dist.dataset[0].shape[0], 1).to(device) * t
        weighted_coupling_dist.log_weights = weighted_coupling_dist.log_weights.to(device)
        weighted_coupling_dist.log_weights += delta * (
            dual_trainer.f_net(time, weighted_coupling_dist.dataset[0].to(device)).squeeze(1) +
            dual_trainer.g_net(time, weighted_coupling_dist.dataset[1].to(device)).squeeze(1)
        )
        weighted_coupling_dist.log_weights.detach_()
        print(weighted_coupling_dist.log_weights[:10])
        log_weight_samples.append(weighted_coupling_dist.log_weights[:30].cpu().numpy())


# %%
# Plot log weight samples over iterations
import numpy as np
import matplotlib.pyplot as plt
log_weight_samples = np.array(log_weight_samples)
plt.figure(figsize=(10, 6))
for i in range(log_weight_samples.shape[1]):
    plt.plot(log_weight_samples[:, i], label=f'Sample {i+1}', lw=0.6)
plt.xlabel('Iteration')
plt.ylabel('Log Weight')
plt.show()

# %%
from dualwcfm.models.cfm import ConditionalFlowMatching
from dualwcfm.utils import pad_t_like_x
from dualwcfm.visualization import plot_2d_flow_trajectories

flow_model = MLP(dim=2, out_dim=2, w=32, time_varying=True).to(device)

cfm_config = CFMTrainerConfig(learning_rate=1e-3, epochs=2000, batch_size=2048, sigma=1.0)
cfm_new = ConditionalFlowMatching(
    source_dist=new_source_dist,
    target_dist=new_target_dist,
    network=flow_model,
    wrapper=torch_wrapper,
    sigma=cfm_config.sigma
)

optimizer = optim.Adam(cfm_new.network.parameters(), lr=cfm_config.learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfm_config.epochs)

batch_size = cfm_config.batch_size

from tqdm import trange
pbar = trange(cfm_config.epochs)
for epoch in pbar:
    (x0_batch, x1_batch), log_weights = weighted_coupling_dist.coupling_sample(batch_size)

    t = torch.rand(batch_size, 1, device=device)
    t_pad = pad_t_like_x(t, x0_batch).to(device)
    x_t = cfm_new.compute_conditional_xt(x0_batch, x1_batch, t_pad)
    true_flow = cfm_new.compute_conditional_flow(x0_batch, x1_batch, t_pad)
    predicted_flow = cfm_new.network(t, x_t)

    loss = torch.mean((predicted_flow - true_flow) ** 2 * torch.exp(log_weights).unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    pbar.set_description(f"Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    if epoch % 500 == 0 and epoch > 0:
        n_samples, n_timesteps = 1500, 25
        plot_2d_flow_trajectories(
            trajectories=cfm_new.get_trajectory(n_samples=n_samples, n_timesteps=n_timesteps),
            target_points=new_target_dist.sample(n_samples).cpu().numpy(),
            t=torch.linspace(0, 1, n_timesteps),
            n_trajectories=n_samples,
            title=f"CFM Trajectories at Epoch {epoch+1}"
        )

# %%



