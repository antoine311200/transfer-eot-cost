# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from dualwcfm.trainers.dual_trainer import DualTrainer, DualTrainerConfig
from dualwcfm.visualization import plot_2d_flow_trajectories
from dualwcfm.models.cfm import ConditionalFlowMatching
from dualwcfm.trainers.dynamic_trainer import DynamicTrainer
from dualwcfm.models.cfm import torch_wrapper
from dualwcfm.trainers.wcfm import CFMTrainerConfig
from torchcfm.models.models import MLP
from dualwcfm.distributions.distribution import Distribution
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
    rotation_degrees=45.0,
    source_shift=(0.0, 0.0),
    target_shift=(0.0, 0.0),
    device=device
)
# new_source_dist = ShiftedDistribution(MoonDistribution(), shift=torch.tensor([1.0, 1.0]))
# new_target_dist = ShiftedDistribution(RotatedDistribution(new_source_dist, angle=torch.pi / 2), shift=torch.tensor([1.0, 1.0]))
new_coupling_dist = RotatedShapeCoupling(
    centers=[[0,0], [1.5,0], [-1.5,0], [0,1.5], [0,-1.5]],
    source_scale=0.2,
    target_scale=0.2,
    rotation_degrees=45.0,
    source_shift=(5.0, 5.0),
    target_shift=(5.0, 5.0),
)
new_source_dist, new_target_dist = new_coupling_dist.get_marginals()

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
import matplotlib.pyplot as plt
import torch
import numpy as np

def evaluate_2d_marginal_shift(x_ref, y_ref, f_net, g_net, new_source, new_target, batch_size=2048):
    # 1. Get data
    # x_ref, y_ref = current_coupling.coupling_sample(batch_size)
    x_ref, y_ref = x_ref.to(device), y_ref.to(device)

    # 2. Compute weights (Shape: [Batch])
    with torch.no_grad():
        # Ensure outputs are squeezed to [Batch]
        a = f_net(x_ref).view(-1)
        b = g_net(y_ref).view(-1)

    weights = 1.0 + a + b
    # Optional: Clip negative weights for cleaner histograms (since rho > 0)
    weights_clamped = torch.clamp(weights, min=0.0).cpu().numpy()

    mass_change = torch.mean(a + b).item()
    print(f"Average Mass Change (should be close to 0): {mass_change:.4f}")

    # 3. Ground Truth Data for comparison
    x_target_gt = new_source.sample(batch_size).cpu().numpy()
    y_target_gt = new_target.sample(batch_size).cpu().numpy()

    x_ref_np = x_ref.cpu().numpy()
    y_ref_np = y_ref.cpu().numpy()

    # 4. Plot Marginals for Dimension 0 and 1
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Check Source X (Dimensions 0 and 1) ---
    for dim in range(2):
        ax = axes[0, dim]
        # Hist of current distribution (Dim i)
        ax.hist(x_ref_np[:, dim], bins=50, alpha=0.5, density=True, label='Current')
        # Hist of Reweighted current (Dim i) -> Should match Target
        ax.hist(x_ref_np[:, dim], bins=50, weights=weights_clamped, alpha=0.5,
                density=True, histtype='step', linewidth=2, label='Predicted (Reweighted)')
        # Hist of Actual Target (Dim i)
        ax.hist(x_target_gt[:, dim], bins=50, alpha=0.3, density=True, color='green', label='Target GT')
        ax.set_title(f"Source: Dimension {dim}")
        if dim == 0: ax.legend()

    # --- Check Target Y (Dimensions 0 and 1) ---
    for dim in range(2):
        ax = axes[1, dim]
        ax.hist(y_ref_np[:, dim], bins=50, alpha=0.5, density=True, label='Current')
        ax.hist(y_ref_np[:, dim], bins=50, weights=weights_clamped, alpha=0.5,
                density=True, histtype='step', linewidth=2, label='Predicted')
        ax.hist(y_target_gt[:, dim], bins=50, alpha=0.3, density=True, color='green', label='Target GT')
        ax.set_title(f"Target: Dimension {dim}")

    plt.tight_layout()
    plt.show()

def visualize_weight_landscape(x_ref, y_ref, f_net, g_net):
    # x_ref, y_ref = current_coupling.coupling_sample(batch_size)
    x_ref = x_ref.to(device)
    y_ref = y_ref.to(device)

    with torch.no_grad():
        a = f_net(x_ref).view(-1)
        b = g_net(y_ref).view(-1)

    # The value (a + b) represents the relative change in density (d_rho / rho)
    change_intensity = (a + b).cpu().numpy()

    x_np = x_ref.cpu().numpy()

    plt.figure(figsize=(8, 6))
    # Scatter plot of the Source domain X
    sc = plt.scatter(x_np[:, 0], x_np[:, 1], c=change_intensity, cmap='coolwarm', s=10, alpha=0.6)
    plt.colorbar(sc, label='Log-Derivative (a+b)')
    plt.title("Where the model wants to move mass (Red = Add Mass, Blue = Remove)")
    plt.xlabel("Dim 0")
    plt.ylabel("Dim 1")
    plt.show()

def plot_iteration_diagnostics(step, x_ref, y_ref, weights, flow_model, device):
    """Visualizes the coupling, the learned weights, and the vector field."""
    x_ref_np = x_ref.detach().cpu().numpy()
    y_ref_np = y_ref.detach().cpu().numpy()
    w_np = weights.detach().cpu().numpy()
    
    # Grid for vector field
    grid_n = 20
    x = np.linspace(-4, 10, grid_n)
    y = np.linspace(-4, 10, grid_n)
    X, Y = np.meshgrid(x, y)
    grid = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32).to(device)
    
    # Flow Vector Field (at t=0 for this step)
    t_zeros = torch.zeros(grid.shape[0], 1).to(device)
    with torch.no_grad():
        vectors = flow_model(torch.cat([grid, t_zeros], dim=1)).cpu().numpy()
    U = vectors[:, 0].reshape(grid_n, grid_n)
    V = vectors[:, 1].reshape(grid_n, grid_n)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Current Coupling & Learned Weights (axes[0])
    sc = axes[0].scatter(x_ref_np[:, 0], x_ref_np[:, 1], c=np.log(w_np + 1e-6), cmap='viridis', s=5, alpha=0.6)
    axes[0].set_title(f"Step {step}: Source Samples\n(Color = Log Learned Weight)")
    plt.colorbar(sc, ax=axes[0])
    axes[0].set_xlim(-4, 10); axes[0].set_ylim(-4, 4)

    # Plot 2: Target Samples (axes[1])
    axes[1].scatter(y_ref_np[:, 0], y_ref_np[:, 1], c='orange', s=5, alpha=0.3, label='Target Coupling Y')
    axes[1].set_title(f"Step {step}: Target Samples")
    axes[1].set_xlim(-4, 10); axes[1].set_ylim(-4, 4)
    
    # Plot 3: Learned Vector Field (axes[2]) <-- FIXED (Was axes[3])
    axes[2].quiver(X, Y, U, V, color='blue', alpha=0.8)
    axes[2].set_title(f"Step {step}: Learned Flow Field (v_t)")
    axes[2].set_xlim(-4, 10); axes[2].set_ylim(-4, 10)
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Dataset initialization

# %%
from dualwcfm.distributions.coupling import FixedSizeCouplingDistribution, MarginalDistribution
from dualwcfm.distributions.distribution import EightGaussiansDistribution, LinearInterpolationDistribution
from tqdm import trange

dataset_size = 1_000_000

fixed_coupling_dist = FixedSizeCouplingDistribution(coupling_dist, size=dataset_size)
source_dist = MarginalDistribution(fixed_coupling_dist.sample_init)
target_dist = MarginalDistribution(fixed_coupling_dist.sample_target)
new_source_dist, new_target_dist = new_coupling_dist.get_marginals()

# %% [markdown]
# ### Learn the marginal bridges mu -> mu' and nu -> nu'

# %%
flow_mu = MLP(dim=2, out_dim=2, w=64, time_varying=True).to(device)
flow_nu = MLP(dim=2, out_dim=2, w=64, time_varying=True).to(device)

optimizer_mu = optim.Adam(flow_mu.parameters(), lr=1e-3)
optimizer_nu = optim.Adam(flow_nu.parameters(), lr=1e-3)

loss_fn = nn.MSELoss()

batch_size = 2048
epochs_marginal = 1000
sigma = 0.1


pbar = trange(epochs_marginal, desc="Training Marginal Bridges")
for epoch in pbar:
    # Sample from current marginals
    x_samples = source_dist.sample(batch_size).to(device)
    x_target_samples = new_source_dist.sample(batch_size).to(device)

    y_samples = target_dist.sample(batch_size).to(device)
    y_target_samples = new_target_dist.sample(batch_size).to(device)

    # Compute targets and predictions for marginal flows
    t = torch.rand(batch_size, 1).to(device)
    x_t = (1 - t) * x_samples + t * x_target_samples + torch.sqrt(2 * t * (1 - t)) * torch.randn_like(x_samples) * sigma
    v_mu_target = (x_target_samples - x_samples) / (1.0)
    v_mu_pred = flow_mu(torch.cat([x_t, t], dim=1))

    t = torch.rand(batch_size, 1).to(device)
    y_t = (1 - t) * y_samples + t * y_target_samples
    v_nu_target = (y_target_samples - y_samples) / (1.0)
    v_nu_pred = flow_nu(torch.cat([y_t, t], dim=1))

    optimizer_mu.zero_grad()
    optimizer_nu.zero_grad()

    loss_mu = loss_fn(v_mu_pred, v_mu_target)
    loss_mu.backward()
    optimizer_mu.step()

    loss_nu = loss_fn(v_nu_pred, v_nu_target)
    loss_nu.backward()
    optimizer_nu.step()

    pbar.set_postfix({"Loss mu": loss_mu.item(), "Loss nu": loss_nu.item()})

    if epoch > 0 and (epoch+1) % 500 == 0:
        # Visualize marginal shifts
        cfm_mu = ConditionalFlowMatching(source_dist, new_source_dist, flow_mu, sigma=sigma)
        cfm_nu = ConditionalFlowMatching(target_dist, new_target_dist, flow_nu, sigma=sigma)

        n_samples = 500
        n_timesteps = 20

        time = torch.linspace(0, 1, n_timesteps)
        traj_mu = cfm_mu.get_trajectory(n_samples=n_samples, n_timesteps=n_timesteps)
        traj_nu = cfm_nu.get_trajectory(n_samples=n_samples, n_timesteps=n_timesteps)
        target_mu_samples = new_source_dist.sample(n_samples).cpu().numpy()
        target_nu_samples = new_target_dist.sample(n_samples).cpu().numpy()

        plot_2d_flow_trajectories(trajectories=traj_mu, t=time, n_trajectories=n_samples, target_points=target_mu_samples, title=f"CFM Trajectories at Epoch {epoch+1} for mu -> mu'")
        plot_2d_flow_trajectories(trajectories=traj_nu, t=time, n_trajectories=n_samples, target_points=target_nu_samples, title=f"CFM Trajectories at Epoch {epoch+1} for nu -> nu'")

# %%
## Example of manually evolving a sample through learned marginal bridges
n_test_samples = 50

x_test = source_dist.sample(n_test_samples).to(device)
y_test = target_dist.sample(n_test_samples).to(device)

# For n iterations with step delta=1/n pushing the samples by delta * flow_mu and delta * flow_nu
n_iterations = 10
delta = 1.0 / n_iterations

trajectories_x = [x_test.cpu().numpy()]
trajectories_y = [y_test.cpu().numpy()]

flow_mu.eval()
flow_nu.eval()

for iteration in range(n_iterations):
    t_mu = torch.full((n_test_samples, 1), iteration * delta).to(device)
    t_nu = torch.full((n_test_samples, 1), iteration * delta).to(device)

    # Update x_test using flow_mu
    v_mu = flow_mu(torch.cat([x_test, t_mu], dim=1))
    x_test = x_test + delta * v_mu

    # Update y_test using flow_nu
    v_nu = flow_nu(torch.cat([y_test, t_nu], dim=1))
    y_test = y_test + delta * v_nu

    trajectories_x.append(x_test.detach().cpu().numpy())
    trajectories_y.append(y_test.detach().cpu().numpy())

trajectories_x = np.stack(trajectories_x, axis=0)  # Shape: [n_iterations+1, n_test_samples, 2]
trajectories_y = np.stack(trajectories_y, axis=0)  # Shape: [n_iterations+1, n_test_samples, 2]

# Plot the trajectories
time_points = np.linspace(0, 1, n_iterations + 1)
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
for i in range(n_test_samples):
    ax1.plot(trajectories_x[:, i, 0], trajectories_x[:, i, 1], marker='o', lw=1, markersize=4)
ax1.set_title("Trajectories through Marginal Bridge mu -> mu'")
ax1.set_xlim(-4, 8)
ax1.set_ylim(-4, 8)
ax1.set_xlabel("Dim 0")
ax1.set_ylabel("Dim 1")
ax2 = fig.add_subplot(122)
for i in range(n_test_samples):
    ax2.plot(trajectories_y[:, i, 0], trajectories_y[:, i, 1], marker='o', lw=1, markersize=4)
ax2.set_title("Trajectories through Marginal Bridge nu -> nu'")
ax2.set_xlim(-4, 8)
ax2.set_ylim(-4, 8)
ax2.set_xlabel("Dim 0")
ax2.set_ylabel("Dim 1")
plt.show()

# %% [markdown]
# ### Learn the iterations of the dual GAN

# %%
hidden_dim = 64
epochs_dual = 1000
epochs = 2000
n_iterations = 10
batch_size = 2048
delta = 1.0 / n_iterations
sigma = 0.1
current_time = 0.0

f_net = MLP(dim=2, out_dim=1, w=hidden_dim, time_varying=False).to(device)
g_net = MLP(dim=2, out_dim=1, w=hidden_dim, time_varying=False).to(device)
flow_model = MLP(dim=2, out_dim=2, w=64, time_varying=True).to(device)

# Create current coupling distribution by copying samples from initial coupling
current_coupling_dist = FixedSizeCouplingDistribution(coupling_dist, size=dataset_size)
current_coupling_dist.dataset = fixed_coupling_dist.dataset.clone()

def compute_div_term(potential_net, samples, vector_field_vals):
    samples.requires_grad_(True)
    potentials = potential_net(samples)
    grads = torch.autograd.grad(
        outputs=potentials.sum(), 
        inputs=samples, 
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True
    )[0]

    dot_prod = torch.sum(grads * vector_field_vals, dim=1)
    return dot_prod.mean()

for step in range(n_iterations):
    print(f"########### Iteration {step + 1} / {n_iterations} ({current_time + delta}) ###########")

    current_time += delta

    f_net.train()
    g_net.train()
    flow_mu.eval()
    flow_nu.eval()

    dual_f_optimizer = optim.Adam(f_net.parameters(), lr=1e-3)
    dual_g_optimizer = optim.Adam(g_net.parameters(), lr=1e-3)

    t_tensor = torch.full((batch_size, 1), current_time).to(device)

    print("Training Dual Network")
    pbar = trange(epochs_dual)
    for epoch in pbar:
        # Draw samples
        x0_init = source_dist.sample(batch_size).to(device)
        x1_init = target_dist.sample(batch_size).to(device)
        x0_target = new_source_dist.sample(batch_size).to(device)
        x1_target = new_target_dist.sample(batch_size).to(device)

        x0_ref, x1_ref = current_coupling_dist.coupling_sample(batch_size)
        x0_ref = x0_ref.to(device)
        x1_ref = x1_ref.to(device)

        # Compute dual loss
        a_init, a_target, a_ref = f_net(x0_init), f_net(x0_target), f_net(x0_ref)
        with torch.no_grad():
            b_ref = g_net(x1_ref)

        term_coupling_a = torch.mean(a_ref ** 2 + 2 * a_ref * b_ref)
        term_marginal_a = 2 * (torch.mean(a_target) - torch.mean(a_init))
        loss_f = term_coupling_a - term_marginal_a

        dual_f_optimizer.zero_grad()
        loss_f.backward()
        dual_f_optimizer.step()

        with torch.no_grad():
            a_ref = f_net(x0_ref)
        b_init, b_target, b_ref = g_net(x1_init), g_net(x1_target), g_net(x1_ref)

        term_coupling_b = torch.mean(b_ref ** 2 + 2 * a_ref * b_ref)
        term_marginal_b = 2 * (torch.mean(b_target) - torch.mean(b_init))
        loss_g = term_coupling_b - term_marginal_b

        dual_g_optimizer.zero_grad()
        loss_g.backward()
        dual_g_optimizer.step()


        pbar.set_description(f"Dual Losses: a: {loss_f.item():.4f} | b: {loss_g.item():.4f}")


    print("Evaluating Marginal Shift Prediction")
    x_ref, y_ref = current_coupling_dist.coupling_sample(10000)
    evaluate_2d_marginal_shift(x_ref, y_ref, f_net, g_net, new_source_dist, new_target_dist)
    visualize_weight_landscape(x_ref, y_ref, f_net, g_net)

    cfm_optimizer = optim.Adam(flow_model.parameters(), lr=1e-3)
    cfm_scheduler = optim.lr_scheduler.CosineAnnealingLR(cfm_optimizer, T_max=epochs)

    flow_model.train()

    pbar = trange(epochs)
    for epoch in pbar:
        x0, x1 = current_coupling_dist.coupling_sample(batch_size)
        x0 = x0.to(device)
        x1 = x1.to(device)

        t = torch.rand(batch_size, 1).to(device)
        x_t = (1 - t) * x0 + t * x1 + torch.sqrt(2 * t * (1 - t)) * torch.randn_like(x0) * sigma
        flow_true = (x1 - x0)
        flow_pred = flow_model(torch.cat([x_t, t], dim=1))

        with torch.no_grad():
            a, b = f_net(x0), g_net(x1)
            log_w = (a + b) * delta
            log_w = log_w - log_w.max()
            weight = torch.exp(log_w)
            weight = weight / (weight.mean() + 1e-8)
            weight = torch.clamp(weight, max=50.0)

        loss = torch.mean(weight * (flow_pred - flow_true) ** 2)

        cfm_optimizer.zero_grad()
        loss.backward()
        cfm_optimizer.step()
        cfm_scheduler.step()

        pbar.set_description(f"CFM Loss: {loss.item():.4f}")

    x_sample, y_sample = current_coupling_dist.coupling_sample(2000)
    with torch.no_grad():
        a_vis, b_vis = f_net(x_sample.to(device)), g_net(y_sample.to(device))
        w_vis = torch.exp(a_vis + b_vis)

    plot_iteration_diagnostics(step, x_sample, y_sample, w_vis, flow_model, device)

    # # Update current coupling distribution
    # # - Use the flow model to generate nu_t+delta from mu_t
    # # - Use generated samples nu_t+delta and reverse the CFM to get mu_t+delta
    chunks = 10_000
    new_initial_samples = []
    new_target_samples = []

    cfm = ConditionalFlowMatching(source_dist=source_dist, target_dist=target_dist, network=flow_model, sigma=0.0) # type: ignore

    f_net.eval()
    g_net.eval()
    flow_model.eval()

    for i in trange(0, dataset_size, chunks):
        samples = current_coupling_dist.dataset[:, i:i+chunks]
        samples_mu_t = samples[0].to(device)
        samples_nu_t = samples[1].to(device)

        # Use flow_mu and flow_nu to push samples
        t_mu = torch.full((samples_mu_t.shape[0], 1), (step + 1) * delta).to(device)
        t_nu = torch.full((samples_nu_t.shape[0], 1), (step + 1) * delta).to(device)

        # Update x_test using flow_mu
        v_mu = flow_mu(torch.cat([samples_mu_t, t_mu], dim=1))
        samples_mu_next = samples_mu_t + delta * v_mu

        # Update y_test using flow_nu
        v_nu = flow_nu(torch.cat([samples_nu_t, t_nu], dim=1))
        samples_nu_next = samples_nu_t + delta * v_nu

        current_coupling_dist.dataset[:, i:i+chunks] = torch.stack([samples_mu_next.detach().cpu(), samples_nu_next.detach().cpu()], dim=0)

    n_samples = 1000
    n_timesteps = 50

    plot_2d_flow_trajectories(
        trajectories=cfm.get_trajectory(x0=current_coupling_dist.sample_init(n_samples).to(device), n_timesteps=n_timesteps),
        t=torch.linspace(0, 1, n_timesteps),
        n_trajectories=n_samples,
        title=f"CFM Trajectories at Epoch {step+1}"
    )
    plot_2d_flow_trajectories(
        trajectories=cfm.get_trajectory(x0=new_source_dist.sample(n_samples).to(device), n_timesteps=n_timesteps),
        target_points=new_target_dist.sample(n_samples).cpu().numpy(),
        t=torch.linspace(0, 1, n_timesteps),
        n_trajectories=n_samples,
        title=f"CFM Trajectories at Epoch {step+1}"
    )

# %%



