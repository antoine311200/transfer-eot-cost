from dataclasses import dataclass, field
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dualwcfm.distributions.coupling import CouplingDistribution, FixedSizeCouplingDistribution, DiskCouplingDistribution
from dualwcfm.distributions.distribution import Distribution, FixedSizeDistribution, FixedSizeDistribution

from torchdyn.core import NeuralODE


def is_image_tensor(x: torch.Tensor) -> bool:
    """Check if tensor is image-shaped (4D: batch, channels, height, width)."""
    return x.dim() == 4


@dataclass
class TrainingConfig:
    device: torch.device
    dataset_size: int = 100_000

    marginal_epochs: int = 2500
    pretrain_epochs: int = 10000
    dual_epochs: int = 100
    cfm_epochs: int = 250

    iterations: int = 100
    loss: str = "implicit"
    advance_with_flow: bool = False

    sigma: float = 0.1
    l2_reg: float = 1e-3
    lip_reg: float = 1e-3

    batch_size: int = 4096

    # Image-specific settings
    image_shape: Optional[Tuple[int, int, int]] = None  # (C, H, W) if working with images

    # Disk-based coupling for memory efficiency with images
    use_disk_coupling: bool = False  # Set True for large image datasets
    coupling_save_dir: Optional[str] = None  # Directory for disk coupling (None = temp dir)
    coupling_chunk_size: int = 256  # Samples per chunk file


@dataclass
class ModelConfig:
    dual_learning_rate: float = 1e-4
    mu_learning_rate: float = 1e-3
    nu_learning_rate: float = 1e-3
    flow_learning_rate: float = 5e-3


@dataclass
class FeedForwardConfig(ModelConfig):
    dimension: int = 2
    hidden_size: int = 64
    # For image models
    image_shape: Optional[Tuple[int, int, int]] = None  # (C, H, W)

    @property
    def is_image_model(self) -> bool:
        return self.image_shape is not None

    @property
    def flat_dimension(self) -> int:
        """Return flattened dimension for compatibility."""
        if self.image_shape:
            return self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
        return self.dimension


@dataclass
class VisualizeConfig:
    show_each: int = 10
    n_trajectories: int = 500
    # Visualization intervals for training phases (0 = disabled)
    show_each_marginal: int = 0  # During marginal flow training
    show_each_pretrain: int = 0  # During pretraining


class Visualizer:
    def __init__(self, config: VisualizeConfig):
        self.config = config
        self.dim = 2

    def display(self, epoch: int, trainer: "Trainer"):
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
            self._display_evolution(x.cpu(), y.cpu(), y_target.cpu())
            plt.show()

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
        ax.grid(True, alpha=0.2)

        lc = LineCollection(segments, colors="black", alpha=0.75, linewidths=0.8, zorder=10)
        ax.add_collection(lc)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.set_aspect("equal")

    def display_marginal(self, epoch: int, trainer: "Trainer", model: nn.Module,
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

        self._display_evolution(start_points.cpu(), end_points.cpu(), target_points.cpu())
        plt.suptitle(f"{title} - Epoch {epoch}")
        plt.show()

    def display_pretrain(self, epoch: int, trainer: "Trainer", model: nn.Module, backward: bool = False):
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

        direction = "Backward" if backward else "Forward"
        self._display_evolution(x0.cpu(), end_points.cpu(), x1.cpu())
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

    def display(self, epoch: int, trainer: "Trainer"):
        """Override to handle image tensors properly."""
        if epoch % self.config.show_each == 0:
            n_samples = min(self.config.n_trajectories, 6)  # Show max 6 images
            x, _ = trainer.current_coupling.coupling_sample(n_samples)
            y_target = trainer.new_target.sample(n_samples)
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
            self._display_evolution(x.cpu(), y.cpu(), y_target.cpu())
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

    def display_marginal(self, epoch: int, trainer: "Trainer", model: nn.Module,
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

        self._display_evolution(start_points.cpu(), end_points.cpu(), target_points.cpu())
        plt.suptitle(f"{title} - Epoch {epoch}")
        plt.show()

    def display_pretrain(self, epoch: int, trainer: "Trainer", model: nn.Module, backward: bool = False):
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

        direction = "Backward" if backward else "Forward"
        self._display_evolution(x0.cpu(), end_points.cpu(), x1.cpu())
        plt.suptitle(f"Pretrain {direction} Flow - Epoch {epoch}")
        plt.show()


class node_wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        t = t.to(x.device)
        if t.dim() < 2:
            t = t.repeat(x.shape[0])[:, None]
        return self.model(torch.cat([x, t], dim=1))
        # return self.model(t, x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_mlp = nn.Linear(time_dim, dim)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        if t_emb is None:
            time_shift = 0
        else:
            time_shift = self.time_mlp(t_emb)
        h = self.mlp(x + time_shift)
        return x + h

class MLP2(nn.Module):
    def __init__(self, dim=3, out_dim=3, w=256, time_varying=False, time_dim=128):
        super().__init__()

        hidden_dim = w

        self.time_varying = time_varying
        if self.time_varying:
            self.t_embed = nn.Sequential(
                TimeEmbedding(time_dim),
                nn.Linear(time_dim, time_dim),
                nn.SiLU()
            )
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.layers = nn.ModuleList([
            ResBlock(hidden_dim, time_dim),
            ResBlock(hidden_dim, time_dim),
            ResBlock(hidden_dim, time_dim)
        ])
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, t, x):
        if self.time_varying:
            if t.dim() == 1: t = t.unsqueeze(-1)
            t_emb = self.t_embed(t)
        else:
            t_emb = None
        h = self.input_proj(x)
        for layer in self.layers: h = layer(h, t_emb)
        return self.out(h)

class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, n_layers=2, time_varying=False, use_norm=None):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        # self.net = torch.nn.Sequential(
        #     torch.nn.Linear(dim + (1 if time_varying else 0), w),
        #     (
        #         torch.nn.BatchNorm1d(w)
        #         if use_norm == "batch"
        #         else torch.nn.LayerNorm(w) if use_norm == "layer" else torch.nn.Identity()
        #     ),
        #     torch.nn.SELU(),
        #     torch.nn.Linear(w, w),
        #     (
        #         torch.nn.BatchNorm1d(w)
        #         if use_norm == "batch"
        #         else torch.nn.LayerNorm(w) if use_norm == "layer" else torch.nn.Identity()
        #     ),
        #     torch.nn.SELU(),
        #     torch.nn.Linear(w, out_dim),
        # )

        layers = []
        layers.append(torch.nn.Linear(dim + (1 if time_varying else 0), w))
        if use_norm == "batch":
            layers.append(torch.nn.BatchNorm1d(w))
        elif use_norm == "layer":
            layers.append(torch.nn.LayerNorm(w))
        layers.append(torch.nn.SELU())
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(w, w))
            if use_norm == "batch":
                layers.append(torch.nn.BatchNorm1d(w))
            elif use_norm == "layer":
                layers.append(torch.nn.LayerNorm(w))
            layers.append(torch.nn.SELU())
        layers.append(torch.nn.Linear(w, out_dim))
        self.net = torch.nn.Sequential(*layers)

        # # Initialize weights
        # for m in self.net.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ConditionalFlowMatching:
    def vector_field(
        self, x: torch.Tensor, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, extend: bool = True
    ) -> torch.Tensor:
        """Compute the vector field at position x and time t.

        Handles both vector (N, D) and image (N, C, H, W) tensors.
        """
        if extend:
            # For images, concatenate along channel dimension (dim=1)
            return torch.cat([x1 - x0, torch.zeros_like(x0)], dim=1)
        else:
            return x1 - x0

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma: float) -> torch.Tensor:
        """Interpolate between x0 and x1 at time t with noise sigma.

        Handles both vector (N, D) and image (N, C, H, W) tensors.
        """
        # Reshape t for proper broadcasting with images
        if is_image_tensor(x0):
            # t is (N, 1), reshape to (N, 1, 1, 1) for image broadcasting
            while t.dim() < x0.dim():
                t = t.unsqueeze(-1)
        # print("Interpolate shapes:", t.shape, x0.shape, x1.shape)

        return (1 - t) * x0 + t * x1 + sigma * torch.randn_like(x0) * torch.sqrt(t * (1 - t))

    def evolve(
        self, x_init: torch.Tensor, model: nn.Module, num_steps: int = 100, sigma: float = 0.0, extend: bool = True
    ) -> torch.Tensor:
        """Evolve the points x_init over time t using Euler integration.

        Handles both vector (N, D) and image (N, C, H, W) tensors.
        """
        device = x_init.device
        dt = 1.0 / num_steps

        with torch.no_grad():
            if extend:
                xt = torch.cat([x_init, x_init], dim=1)
            else:
                xt = x_init.clone()

            for i in range(num_steps):
                t = torch.ones(xt.shape[0], 1, device=device) * (i * dt)
                vt = model(t, xt)
                xt = xt + vt * dt

        model = model.to(device)
        xt = xt.to(device)

        return xt


class Strategy:
    """Base strategy for training transport models.

    This class can be subclassed to customize model architectures.
    For custom models (e.g., UNets for images), override _create_models().
    """

    def __init__(self, wrapper, config: FeedForwardConfig):
        self.wrapper = wrapper
        self.config = config

        dim = config.dimension
        hidden_size = config.hidden_size

        self._create_models(dim, hidden_size)
        self._create_optimizers(config)

    def _create_models(self, dim: int, hidden_size: int):
        """Create all models. Override this method for custom architectures."""
        wrapper = self.wrapper
        self.mu_bridge = wrapper(MLP(dim=dim, out_dim=dim, w=hidden_size, time_varying=True))
        self.nu_bridge = wrapper(MLP(dim=dim, out_dim=dim, w=hidden_size, time_varying=True))
        self.f_dual = wrapper(MLP(dim=dim, out_dim=1, w=hidden_size, time_varying=False))
        self.g_dual = wrapper(MLP(dim=dim, out_dim=1, w=hidden_size, time_varying=False))
        self.forward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, time_varying=True))
        self.previous_forward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, time_varying=True))

        # Save initial state
        self.initial_mu_bridge = wrapper(MLP(dim=dim, out_dim=dim, w=hidden_size, time_varying=True))
        self.initial_nu_bridge = wrapper(MLP(dim=dim, out_dim=dim, w=hidden_size, time_varying=True))
        self.initial_forward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, time_varying=True))

    def get_output_channels(self, x: torch.Tensor) -> int:
        """Get the number of channels to extract from model output for images."""
        if is_image_tensor(x):
            return x.shape[1]  # Number of channels
        return x.shape[1] // 2  # Half of vector dimension

    def _create_optimizers(self, config: FeedForwardConfig):
        dual_params = list(self.f_dual.parameters()) + list(self.g_dual.parameters())
        self.mu_optimizer = optim.Adam(self.mu_bridge.parameters(), lr=config.mu_learning_rate)
        self.nu_optimizer = optim.Adam(self.nu_bridge.parameters(), lr=config.nu_learning_rate)
        self.dual_optimizer = optim.Adam(dual_params, lr=config.dual_learning_rate)
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=config.flow_learning_rate)

    def get_duals(self, device):
        return self.f_dual.to(device), self.g_dual.to(device), self.dual_optimizer

    def get_bridges(self, device):
        return (self.mu_bridge.to(device), self.mu_optimizer), (self.nu_bridge.to(device), self.nu_optimizer)

    def get_flow(self, step: int, device):
        return self.forward_model.to(device), self.forward_optimizer

    def get_previous_flow(self, step: int, device):
        return self.previous_forward_model.to(device)

    def get_initial_bridges(self, device):
        return self.initial_mu_bridge.to(device), self.initial_nu_bridge.to(device)

    def get_initial_flow(self, step: int, device):
        return self.initial_forward_model.to(device)

    def advance(
        self, step: int, x_mu_next: torch.Tensor, y_nu_next: torch.Tensor, num_steps: int = 100, sigma: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the coupling using the current flow models."""
        x_condition = x_mu_next.clone()
        xt = x_mu_next.clone()
        dt = 1.0 / num_steps
        model = self.get_flow(step, x_mu_next.device)[0]
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.ones(xt.shape[0], 1).to(xt.device) * (i * dt)
                vt = model(t, xt, x_condition)[:, : xt.shape[1] // 2]
                xt = xt + vt * dt
        return x_mu_next, xt.clone()


class BidirectionalStrategy(Strategy):
    def __init__(self, wrapper, config: FeedForwardConfig):
        super().__init__(wrapper, config)

    def _create_models(self, dim: int, hidden_size: int):
        super()._create_models(dim, hidden_size)
        wrapper = self.wrapper
        self.backward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, time_varying=True))
        self.previous_backward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, time_varying=True))
        self.initial_backward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, time_varying=True))

    def _create_optimizers(self, config: FeedForwardConfig):
        super()._create_optimizers(config)
        self.backward_optimizer = optim.Adam(self.backward_model.parameters(), lr=config.flow_learning_rate)

    def get_flow(self, step: int, device):
        model = self.forward_model if step % 2 == 0 else self.backward_model
        optimizer = self.forward_optimizer if step % 2 == 0 else self.backward_optimizer
        return model.to(device), optimizer

    def get_previous_flow(self, step: int, device):
        previous_model = self.previous_forward_model if step % 2 == 0 else self.previous_backward_model
        return previous_model.to(device)

    def get_initial_flow(self, step: int, device):
        initial_model = self.initial_forward_model if step % 2 == 0 else self.initial_backward_model
        return initial_model.to(device)

    def advance(
        self, step: int, x_mu_next: torch.Tensor, y_nu_next: torch.Tensor, num_steps: int = 100, sigma: float = 0
    ):
        if step % 2 == 0:
            return super().advance(step, x_mu_next, y_nu_next, num_steps, sigma)
        else:
            y_next, x_next = super().advance(step, y_nu_next, x_mu_next, num_steps, sigma)
            return x_next, y_next


class CFMLoss:
    def execute(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        strategy: Strategy,
        config: TrainingConfig,
        flow_matching: ConditionalFlowMatching,
        step: int,
    ) -> tuple[torch.Tensor, nn.Module, optim.Adam]:
        """Compute the CFM loss."""
        raise NotImplementedError("method not implemented")

    @staticmethod
    def get_loss(loss_name: str):
        if loss_name == "weighted":
            return WeightedCFMLoss()
        elif loss_name == "explicit":
            return ExplicitCFMLoss()
        elif loss_name == "implicit":
            return ImplicitCFMLoss()
        else:
            raise NameError(f"loss name `{loss_name}` does not exist (choose between weighted, explicit, implicit)")


class WeightedCFMLoss(CFMLoss):
    def execute(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        strategy: Strategy,
        config: TrainingConfig,
        flow_matching: ConditionalFlowMatching,
        step: int,
    ) -> tuple[torch.Tensor, nn.Module, optim.Adam]:
        f_dual, g_dual, _ = strategy.get_duals(config.device)
        model, optimizer = strategy.get_flow(0, config.device)

        with torch.no_grad():
            a, b = f_dual(x0), g_dual(x1)
            r = a + b
            log_w = r / config.iterations
            weight = torch.exp(log_w)

            # For images, expand weight to broadcast properly
            if is_image_tensor(x0):
                while weight.dim() < x0.dim():
                    weight = weight.unsqueeze(-1)

            target_u_s = x1 - x0

        s = torch.rand(config.batch_size, 1).to(config.device)
        x_s = flow_matching.interpolate(x0, x1, s, config.sigma)
        u_s = model(s, x_s, x0)

        return (weight * (u_s - target_u_s) ** 2).mean(), model, optimizer


class ExplicitCFMLoss(CFMLoss):
    pass


class ImplicitCFMLoss(CFMLoss):
    def execute(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        strategy: Strategy,
        config: TrainingConfig,
        flow_matching: ConditionalFlowMatching,
        step: int,
    ) -> tuple[torch.Tensor, nn.Module, optim.Adam]:
        f_dual, g_dual, _ = strategy.get_duals(config.device)
        model, optimizer = strategy.get_flow(step, config.device)
        previous_model = strategy.get_previous_flow(step, config.device)

        s = torch.rand(config.batch_size, 1).to(config.device)
        xs = flow_matching.interpolate(x0, x1, s, config.sigma)

        with torch.no_grad():
            # For dual models: output shape is [batch, 1]
            a, b = f_dual(x0), g_dual(x1)
            r = a + b
            log_w = r / config.iterations
            weight = torch.exp(log_w)

            # For images, expand weight to broadcast properly: [batch, 1] -> [batch, 1, 1, 1]
            if is_image_tensor(x0):
                while weight.dim() < x0.dim():
                    weight = weight.unsqueeze(-1)

            if step % 2 != 0:
                x0, x1 = x1, x0

            b_hat_detached = model(s, xs, x0)
            b_ta = previous_model(s, xs, x0)

            u_t = flow_matching.vector_field(x0, torch.zeros_like(x0), x0, x1)
            target = b_ta + (weight - 1) * (u_t - b_hat_detached)
        u_s = model(s, xs, x0)
        return ((u_s - target) ** 2).mean(), model, optimizer


class Trainer:

    def __init__(
        self,
        initial_coupling: CouplingDistribution,
        new_source: Distribution,
        new_target: Distribution,
        flow_matching: ConditionalFlowMatching,
        strategy: Strategy,
        visualizer: Visualizer,
        config: TrainingConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.initial_coupling = initial_coupling
        self.new_source = new_source
        self.new_target = new_target

        self.visualizer = visualizer
        self.strategy = strategy
        self.flow_matching = flow_matching
        self.loss_cfm = CFMLoss.get_loss(config.loss)

        self.config = config
        self.device = device

        # Fixed sized mu_t, nu_t and coupling distributions
        self._init_distributions(initial_coupling)

    def _init_distributions(self, initial_coupling: CouplingDistribution):
        mu_distribution, nu_distribution = initial_coupling.get_marginals()
        self.mu_distribution = mu_distribution#.clone()
        self.nu_distribution = nu_distribution#.clone()
        # self.mu_distribution = FixedSizeDistribution(mu_distribution, size=self.config.dataset_size)
        # self.nu_distribution = FixedSizeDistribution(nu_distribution, size=self.config.dataset_size)

        # Use disk-based coupling for images to avoid OOM
        if self.config.use_disk_coupling or (self.config.image_shape is not None and self.config.dataset_size > 1000):
            print("Using disk-based coupling distribution for memory efficiency.")
            self.current_coupling = DiskCouplingDistribution(
                initial_coupling,
                size=self.config.dataset_size,
                save_dir=self.config.coupling_save_dir,
                chunk_size=self.config.coupling_chunk_size,
                device=self.device,
            )
        else:
            self.current_coupling = FixedSizeCouplingDistribution(initial_coupling, size=self.config.dataset_size)

        # Copy to Initial distribution for later use
        self.initial_mu_distribution = mu_distribution#.clone()
        self.initial_nu_distribution = nu_distribution#.clone()
        self.initial_coupling_distribution = self.current_coupling#.clone()

    def _load_pretrained_models(self, dir):
        # Load pretrained models if available
        # initial_forward_model.pth, initial_backward_model.pth, initial_mu_model.pth, initial_nu_model.pth
        forward_state_dict = torch.load(f"{dir}/initial_forward_model.pth", map_location=self.device)

        self.strategy.forward_model.load_state_dict(forward_state_dict)
        self.strategy.initial_forward_model.load_state_dict(forward_state_dict)

        if isinstance(self.strategy, BidirectionalStrategy):
            backward_state_dict = torch.load(f"{dir}/initial_backward_model.pth", map_location=self.device)
            self.strategy.backward_model.load_state_dict(backward_state_dict)
            self.strategy.initial_backward_model.load_state_dict(backward_state_dict)

        mu_state_dict = torch.load(f"{dir}/initial_mu_model.pth", map_location=self.device)
        nu_state_dict = torch.load(f"{dir}/initial_nu_model.pth", map_location=self.device)
        self.strategy.mu_bridge.load_state_dict(mu_state_dict)
        self.strategy.initial_mu_bridge.load_state_dict(mu_state_dict)
        self.strategy.nu_bridge.load_state_dict(nu_state_dict)
        self.strategy.initial_nu_bridge.load_state_dict(nu_state_dict)
        print("Pretrained models loaded.")

    def train(self, retrain: bool = False, save: str = None):
        self.strategy._create_models(self.strategy.config.dimension, self.strategy.config.hidden_size)
        self.strategy._create_optimizers(self.strategy.config)

        if not retrain:
            self.mu_distribution = self.initial_mu_distribution.clone()
            self.nu_distribution = self.initial_nu_distribution.clone()
            self.current_coupling = self.initial_coupling_distribution.clone()

        # Marginal flow matching phase
        self.train_marginal_flows(retrain, save)

        # Pretrain phase
        if isinstance(self.strategy, BidirectionalStrategy) or hasattr(self.strategy, "backward_model"):
            print("Pretraining bidirectional flows.")
            forward_model, forward_optimizer = self.strategy.get_flow(0, self.device)
            backward_model, backward_optimizer = self.strategy.get_flow(1, self.device)
            initial_forward_model = self.strategy.get_initial_flow(0, self.device)
            initial_backward_model = self.strategy.get_initial_flow(1, self.device)

            if retrain:
                self.pretrain(forward_model, forward_optimizer, backward=False)
                self.pretrain(backward_model, backward_optimizer, backward=True)
                initial_forward_model.load_state_dict(forward_model.state_dict())
                initial_backward_model.load_state_dict(backward_model.state_dict())

                # Save initial models
                if save:
                    # Create directory if it doesn't exist
                    import os
                    os.makedirs(save, exist_ok=True)
                    torch.save(
                        initial_forward_model.state_dict(),
                        f"{save}/initial_forward_model.pth",
                    )
                    torch.save(
                        initial_backward_model.state_dict(),
                        f"{save}/initial_backward_model.pth",
                    )
            else:
                print("Skipping pretraining.")
                print("Loading initial flow models.")

                if save:
                    self._load_pretrained_models(save)
                else:
                    forward_model.load_state_dict(initial_forward_model.state_dict())
                    backward_model.load_state_dict(initial_backward_model.state_dict())
        else:
            print("Pretraining unidirectional flow.")
            model, optimizer = self.strategy.get_flow(0, self.device)
            initial_model = self.strategy.get_initial_flow(0, self.device)
            if retrain:
                self.pretrain(model, optimizer)
                initial_model.load_state_dict(model.state_dict())
            else:
                print("Skipping pretraining.")
                print("Loading initial flow model.")
                model.load_state_dict(initial_model.state_dict())

        n_vis = 6 if self.config.image_shape else 2500
        start_points = self.mu_distribution.sample(n_vis)
        start_points = start_points.to(self.device)
        end_points = self.flow_matching.evolve(
            start_points, self.strategy.get_flow(0, self.device)[0], num_steps=50, sigma=self.config.sigma, extend=True
        )
        # Extract first half of output (evolved part, not frozen condition)
        if is_image_tensor(end_points):
            end_points = end_points[:, :start_points.shape[1], :, :]
        else:
            end_points = end_points[:, : self.strategy.config.dimension]
        target_points = self.nu_distribution.sample(n_vis)
        target_points = target_points.to(self.device)
        self.visualizer._display_evolution(start_points.cpu(), end_points.cpu(), target_points.cpu())
        plt.show()

        # Divide by 10 learning rates for cfm
        for param_group in self.strategy.get_flow(0, self.device)[1].param_groups:
            param_group["lr"] = param_group["lr"] / 10.0
        if isinstance(self.strategy, BidirectionalStrategy):
            for param_group in self.strategy.get_flow(1, self.device)[1].param_groups:
                param_group["lr"] = param_group["lr"] / 10.0

        # General training loop
        for iteration in range(self.config.iterations):  # , desc="Iteration loop"):
            print(f"Iteration {iteration+1}/{self.config.iterations}")
            self.train_duals_step(iteration)
            self.train_cfm_step(iteration)
            self.advance_step(iteration)

    def train_marginal_flows(self, retrain: bool = False, save: str = None):
        sigma = self.config.sigma
        batch_size = self.config.batch_size
        device = self.config.device
        loss_fn = nn.MSELoss()

        (mu_model, mu_optimizer), (nu_model, nu_optimizer) = self.strategy.get_bridges(device)
        initial_mu_model, initial_nu_model = self.strategy.get_initial_bridges(device)

        if retrain:

            pbar = trange(self.config.marginal_epochs, desc="Marginal Flow Matching")
            for epoch in pbar:
                # x0, y0 = self.initial_coupling.coupling_sample(batch_size)
                x0, y0 = self.initial_coupling.sample_init(batch_size), self.initial_coupling.sample_target(batch_size)
                x1, y1 = self.new_source.sample(batch_size), self.new_target.sample(batch_size)
                x0, x1, y0, y1 = x0.to(device), x1.to(device), y0.to(device), y1.to(device)

                t = torch.rand(batch_size, 1).to(device)
                xt, yt = self.flow_matching.interpolate(x0, x1, t, sigma), self.flow_matching.interpolate(
                    y0, y1, t, sigma
                )
                ut, vt = mu_model(t, xt), nu_model(t, yt)
                ut_true, vt_true = self.flow_matching.vector_field(
                    xt, t, x0, x1, extend=False
                ), self.flow_matching.vector_field(yt, t, y0, y1, extend=False)

                mu_optimizer.zero_grad()
                nu_optimizer.zero_grad()

                loss_mu = loss_fn(ut, ut_true)
                loss_mu.backward()
                mu_optimizer.step()

                loss_nu = loss_fn(vt, vt_true)
                loss_nu.backward()
                nu_optimizer.step()

                pbar.set_postfix({"Loss mu": loss_mu.item(), "Loss nu": loss_nu.item()})

                # Visualize marginal flows during training
                self.visualizer.display_marginal(
                    epoch, self, mu_model,
                    self.mu_distribution, self.new_source,
                    title="Mu Marginal Flow"
                )
                self.visualizer.display_marginal(
                    epoch, self, nu_model,
                    self.nu_distribution, self.new_target,
                    title="Nu Marginal Flow"
                )

            initial_mu_model.load_state_dict(mu_model.state_dict())
            initial_nu_model.load_state_dict(nu_model.state_dict())

            # Save initial models
            if save:
                import os
                os.makedirs(save, exist_ok=True)
                torch.save(
                    initial_mu_model.state_dict(),
                    f"{save}/initial_mu_model.pth",
                )
                torch.save(
                    initial_nu_model.state_dict(),
                    f"{save}/initial_nu_model.pth",
                )
                print("Initial marginal flow models saved.")
        else:
            print("Skipping marginal flow matching retraining.")
            print("Loading initial marginal flow models.")

            if save:
                self._load_pretrained_models(save)
            else:
                mu_model.load_state_dict(initial_mu_model.state_dict())
                nu_model.load_state_dict(initial_nu_model.state_dict())

        # Free cuda memory after pretraining
        torch.cuda.empty_cache()

        samples = 2500
        start_points = self.mu_distribution.sample(samples)
        start_points = start_points.to(self.device)

        end_points = self.flow_matching.evolve(
            start_points, self.strategy.mu_bridge, num_steps=50, sigma=self.config.sigma, extend=False
        )
        target_points = self.new_source.sample(samples)
        target_points = target_points.to(self.device)
        self.visualizer._display_evolution(start_points.cpu(), end_points.cpu(), target_points.cpu())
        plt.show()

        start_points = self.nu_distribution.sample(samples)
        start_points = start_points.to(self.device)

        end_points = self.flow_matching.evolve(
            start_points, self.strategy.nu_bridge, num_steps=50, sigma=self.config.sigma, extend=False
        )
        target_points = self.new_target.sample(samples)
        target_points = target_points.to(self.device)
        self.visualizer._display_evolution(start_points.cpu(), end_points.cpu(), target_points.cpu())
        plt.show()

    def pretrain(self, model, optimizer, backward: bool = False):
        sigma = self.config.sigma
        batch_size = self.config.batch_size
        loss_fn = nn.MSELoss()

        pbar = trange(self.config.pretrain_epochs, desc="Pretraining")

        for epoch in pbar:
            x0, x1 = self.initial_coupling.coupling_sample(batch_size)
            x0, x1 = x0.to(self.config.device), x1.to(self.config.device)
            if backward:
                x0, x1 = x1, x0

            s = torch.rand(batch_size, 1).to(self.config.device)
            x_s = self.flow_matching.interpolate(x0, x1, s, sigma)
            u_s = model(s, x_s, x0)
            u_s_true = self.flow_matching.vector_field(x_s, s, x0, x1)

            loss = loss_fn(u_s, u_s_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Pretrain Loss": loss.item()})

            # Visualize pretraining flow
            self.visualizer.display_pretrain(epoch, self, model, backward=backward)

    def train_duals_step(self, time_step: int):
        batch_size = self.config.batch_size
        t_tensor = torch.full((batch_size, 1), time_step / self.config.iterations, device=self.config.device)
        is_image = self.config.image_shape is not None

        (mu_model, _), (nu_model, _) = self.strategy.get_bridges(self.device)
        f_dual, g_dual, dual_optimizer = self.strategy.get_duals(self.device)
        f_dual.train()
        g_dual.train()
        mu_model.eval()
        nu_model.eval()

        pbar = trange(self.config.dual_epochs, desc="Dual Training")

        for epoch in pbar:
            x, y = self.current_coupling.coupling_sample(self.config.batch_size)
            # x_mu = self.mu_distribution.sample(self.config.batch_size)
            # y_nu = self.nu_distribution.sample(self.config.batch_size)

            # Create mu_t, nu_t sampels from interpolating between source and new_source, target and new_target for current batch
            x_mu = self.flow_matching.interpolate(
                self.initial_coupling.sample_init(batch_size).to(self.config.device),
                self.new_source.sample(batch_size).to(self.config.device),
                t_tensor,
                0.0#self.config.sigma
            )
            y_nu = self.flow_matching.interpolate(
                self.initial_coupling.sample_target(batch_size).to(self.config.device),
                self.new_target.sample(batch_size).to(self.config.device),
                t_tensor,
                0.0#self.config.sigma
            )

            x, y = x.to(self.config.device), y.to(self.config.device)
            x_mu, y_nu = x_mu.to(self.config.device), y_nu.to(self.config.device)

            with torch.no_grad():
                u_t = mu_model(t_tensor, x)
                v_t = nu_model(t_tensor, y)

            x, y = x.requires_grad_(True), y.requires_grad_(True)
            x_mu, y_nu = x_mu.requires_grad_(True), y_nu.requires_grad_(True)

            a, a_mu = f_dual(x), f_dual(x_mu)
            b, b_nu = g_dual(y), g_dual(y_nu)

            # Coupling term
            coupling_term = torch.mean((a - a.mean() + b - b.mean()) ** 2)

            # Divergence terms
            grad_a = torch.autograd.grad(a_mu.sum(), x_mu, create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad_b = torch.autograd.grad(b_nu.sum(), y_nu, create_graph=True, retain_graph=True, only_inputs=True)[0]

            # For images: sum over all dimensions except batch
            if is_image:
                # grad_a, u_t have shape [batch, C, H, W]
                a_term = torch.mean((grad_a * u_t).sum(dim=[1, 2, 3]))
                b_term = torch.mean((grad_b * v_t).sum(dim=[1, 2, 3]))
                # Compute gradient norm over flattened dimensions
                gp_f = ((grad_a.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
                gp_g = ((grad_b.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
            else:
                a_term = torch.mean(torch.sum(grad_a * u_t, dim=1))
                b_term = torch.mean(torch.sum(grad_b * v_t, dim=1))
                gp_f = ((grad_a.norm(2, dim=1) - 1) ** 2).mean()
                gp_g = ((grad_b.norm(2, dim=1) - 1) ** 2).mean()

            # Regularization (L2 + Lipshitz)
            l2_reg = self.config.l2_reg * (
                torch.mean(a**2) + torch.mean(b**2) + torch.mean(a_mu**2) + torch.mean(b_nu**2)
            )

            lip_reg = self.config.lip_reg * (gp_f + gp_g)

            loss = 0.5 * coupling_term - a_term - b_term + l2_reg + lip_reg

            dual_optimizer.zero_grad()
            loss.backward()
            dual_optimizer.step()

            pbar.set_postfix({"Dual Loss": loss.item()})

    def train_cfm_step(self, time_step: int):
        batch_size = self.config.batch_size

        pbar = trange(self.config.cfm_epochs, desc="CFM Training")

        model, _ = self.strategy.get_flow(time_step, self.config.device)
        previous_model = self.strategy.get_previous_flow(time_step, self.config.device)
        previous_model.load_state_dict(model.state_dict())
        previous_model.eval()

        for epoch in pbar:
            x0, x1 = self.current_coupling.coupling_sample(batch_size)
            x0, x1 = x0.to(self.config.device), x1.to(self.config.device)

            loss, model, optimizer = self.loss_cfm.execute(
                x0, x1, self.strategy, self.config, self.flow_matching, time_step
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"CFM Loss": loss.item()})

        self.visualizer.display(time_step, self)

    def advance_step(self, time_step: int):
        # Use coupling's chunk size for disk-based coupling, otherwise use defaults
        if isinstance(self.current_coupling, DiskCouplingDistribution):
            chunk_size = self.current_coupling.chunk_size
        else:
            chunk_size = 100_000 if not self.config.image_shape else 256

        (mu_model, _), (nu_model, _) = self.strategy.get_bridges(self.device)
        mu_model.eval()
        nu_model.eval()

        use_disk = isinstance(self.current_coupling, DiskCouplingDistribution)
        n_chunks = (self.current_coupling.size + chunk_size - 1) // chunk_size

        with torch.no_grad():
            for chunk_idx in trange(n_chunks, desc="Advancing coupling"):
                i = chunk_idx * chunk_size
                # Handle last chunk which may be smaller
                actual_chunk_size = min(chunk_size, self.current_coupling.size - i)

                if self.config.advance_with_flow:
                    x_mu, y_nu = self.mu_distribution.dataset[i : i + actual_chunk_size].to(
                        self.config.device
                    ), self.nu_distribution.dataset[i : i + actual_chunk_size].to(self.config.device)

                    t_tensor = torch.full((x_mu.shape[0], 1), time_step / self.config.iterations).to(self.config.device)

                    x_mu_next = x_mu.detach() + mu_model(t_tensor, x_mu) * (1 / self.config.iterations)
                    y_nu_next = y_nu.detach() + nu_model(t_tensor, y_nu) * (1 / self.config.iterations)
                else:
                    # Compute interpolation from source and new_source, target and new_target for current chunk
                    t_tensor = torch.full((actual_chunk_size, 1), time_step / self.config.iterations).to(self.config.device)
                    x_mu_next = self.flow_matching.interpolate(
                        self.initial_coupling.sample_init(actual_chunk_size).to(self.config.device),
                        self.new_source.sample(actual_chunk_size).to(self.config.device),
                        t_tensor,
                        0.0
                    )
                    y_nu_next = self.flow_matching.interpolate(
                        self.initial_coupling.sample_target(actual_chunk_size).to(self.config.device),
                        self.new_target.sample(actual_chunk_size).to(self.config.device),
                        t_tensor,
                        0.0
                    )

                num_steps = 10
                flow_model_mu = self.strategy.forward_model
                flow_model_nu = (
                    self.strategy.backward_model
                    if isinstance(self.strategy, BidirectionalStrategy)
                    else self.strategy.forward_model
                )
                device = self.config.device
                dt = 1.0 / num_steps

                if time_step % 2 == 0:
                    x_next = x_mu_next.detach().clone()
                    xt = x_next.clone()

                    for j in range(num_steps):
                        t = torch.ones(x_next.shape[0], 1).to(device) * (j * dt)
                        if is_image_tensor(xt):
                            x_concat = torch.cat([xt, x_next], dim=1)
                            vt = flow_model_mu(t, x_concat)
                            n_channels = xt.shape[1]
                            vt = vt[:, :n_channels, :, :]
                        else:
                            vt = flow_model_mu(torch.cat([xt, x_next, t], dim=1))[:, : self.strategy.config.dimension]
                        xt = xt + vt * dt
                    y_next = xt.clone()
                else:
                    y_next = y_nu_next.detach().clone()
                    yt = y_next.clone()

                    for j in range(num_steps):
                        t = torch.ones(y_next.shape[0], 1).to(device) * (j * dt)
                        if is_image_tensor(yt):
                            y_concat = torch.cat([yt, y_next], dim=1)
                            vt = flow_model_nu(t, y_concat)
                            n_channels = yt.shape[1]
                            vt = vt[:, :n_channels, :, :]
                        else:
                            vt = flow_model_nu(torch.cat([yt, y_next, t], dim=1))[:, : self.strategy.config.dimension]
                        yt = yt + vt * dt
                    x_next = yt.clone()

                # Update coupling dataset
                self.current_coupling.update_chunk(i, x_next.cpu(), y_next.cpu())

                if self.config.advance_with_flow:
                    self.mu_distribution.dataset[i : i + actual_chunk_size] = x_mu_next.cpu()
                    self.nu_distribution.dataset[i : i + actual_chunk_size] = y_nu_next.cpu()

        self.plot_distributions_after_advance(time_step)

    def plot_distributions_after_advance(self, time_step: int):
        """Plot distributions after advancing step. Handles both vectors and images."""

        if is_image_tensor(self.mu_distribution.sample(1)):
            self._plot_image_distributions_after_advance(time_step)
            return

        # Vector distributions visualization
        # mu_samples = self.mu_distribution.sample(2500).to(self.device)
        # nu_samples = self.nu_distribution.sample(2500).to(self.device)

        # Sample mu_t, nu_t from interpolating between source and new_source, target and new_target for current time step
        n_samples = 2500
        t_tensor = torch.full((n_samples // 10, 1), time_step / self.config.iterations, device=self.device)
        mu_samples = self.flow_matching.interpolate(
            self.initial_coupling.sample_init(n_samples // 10).to(self.device),
            self.new_source.sample(n_samples // 10).to(self.device),
            t_tensor,
            0.0#self.config.sigma
        )
        nu_samples = self.flow_matching.interpolate(
            self.initial_coupling.sample_target(n_samples // 10).to(self.device),
            self.new_target.sample(n_samples // 10).to(self.device),
            t_tensor,
            0.0#self.config.sigma
        )

        # Get coupling samples
        (mu_coupling_samples, nu_coupling_samples) = self.current_coupling.coupling_sample(250)
        mu_coupling_samples = mu_coupling_samples.to(self.device)
        nu_coupling_samples = nu_coupling_samples.to(self.device)

        # 3D plot
        if self.strategy.config.dimension == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(
                mu_samples[:, 0].cpu(),
                mu_samples[:, 1].cpu(),
                mu_samples[:, 2].cpu(),
                c="blue",
                alpha=0.1,
                depthshade=True,
                s=3,
            )
            ax.scatter(
                nu_samples[:, 0].cpu(),
                nu_samples[:, 1].cpu(),
                nu_samples[:, 2].cpu(),
                c="red",
                alpha=0.1,
                depthshade=True,
                s=3,
            )
            ax.scatter(
                mu_coupling_samples[:, 0].cpu(),
                mu_coupling_samples[:, 1].cpu(),
                mu_coupling_samples[:, 2].cpu(),
                c="purple",
                marker="x",
                alpha=0.75,
                depthshade=True,
            )
            ax.scatter(
                nu_coupling_samples[:, 0].cpu(),
                nu_coupling_samples[:, 1].cpu(),
                nu_coupling_samples[:, 2].cpu(),
                c="orange",
                marker="x",
                alpha=0.75,
                depthshade=True,
            )

            # Line between coupling samples
            dist = torch.norm(nu_coupling_samples - mu_coupling_samples, dim=1).cpu()

            for j in range(mu_coupling_samples.shape[0]):
                ax.plot(
                    [mu_coupling_samples[j, 0].cpu(), nu_coupling_samples[j, 0].cpu()],
                    [mu_coupling_samples[j, 1].cpu(), nu_coupling_samples[j, 1].cpu()],
                    [mu_coupling_samples[j, 2].cpu(), nu_coupling_samples[j, 2].cpu()],
                    # color=plt.cm.plasma(dist[j] / dist.max()),
                    color='gray',
                    alpha=0.8,
                )
            L = 1.2  # length of the axis lines

            ax.plot([-L, L], [0, 0], [0, 0], color="k", linewidth=1)  # X-axis
            ax.plot([0, 0], [-L, L], [0, 0], color="k", linewidth=1)  # Y-axis
            ax.plot([0, 0], [0, 0], [-L, L], color="k", linewidth=1)  # Z-axis

            # u = (nu_coupling_samples - mu_coupling_samples).cpu()

            # ax.quiver(
            #     mu_coupling_samples[:, 0].cpu(),
            #     mu_coupling_samples[:, 1].cpu(),
            #     mu_coupling_samples[:, 2].cpu(),
            #     u[:, 0],
            #     u[:, 1],
            #     u[:, 2],
            #     length=0.3,
            #     normalize=False,
            #     color="black",
            #     alpha=0.7,
            #     linewidth=0.4,
            # )

            # ax.view_init(elev=25, azim=45)

            # Set limits
            ax.set_xlim([-L, L])
            ax.set_ylim([-L, L])
            ax.set_zlim([-L, L])

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            sm = plt.cm.ScalarMappable(cmap="viridis")
            sm.set_array(dist.numpy())
            plt.colorbar(sm, ax=ax, shrink=0.6, label="Transport length")

            ax.set_title(f"Distributions after Advancing Step {time_step+1}")
            plt.show()
        else:
            plt.figure(figsize=(6, 6))
            plt.plot(mu_samples[:, 0].cpu(), mu_samples[:, 1].cpu(), '.', label='Mu Distribution', alpha=0.5)
            plt.plot(nu_samples[:, 0].cpu(), nu_samples[:, 1].cpu(), '.', label='Nu Distribution', alpha=0.5)
            plt.plot(mu_coupling_samples[:, 0].cpu(), mu_coupling_samples[:, 1].cpu(), 'x', label='Mu Coupling Samples', alpha=0.5)
            plt.plot(nu_coupling_samples[:, 0].cpu(), nu_coupling_samples[:, 1].cpu(), 'x', label='Nu Coupling Samples', alpha=0.5)
            # Line between coupling samples
            for j in range(mu_coupling_samples.shape[0]):
                plt.plot([mu_coupling_samples[j, 0].cpu(), nu_coupling_samples[j, 0].cpu()],
                        [mu_coupling_samples[j, 1].cpu(), nu_coupling_samples[j, 1].cpu()],
                        'k-', alpha=0.1)
            plt.title(f'Distributions after Advancing Step {time_step+1}')
            plt.show()

    def _plot_image_distributions_after_advance(self, time_step: int):
        """Plot image distributions after advancing step."""
        n_samples = 4
        (mu_coupling, nu_coupling) = self.current_coupling.coupling_sample(n_samples)
        mu_coupling = mu_coupling.cpu()
        nu_coupling = nu_coupling.cpu()

        fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

        for i in range(n_samples):
            # Source (mu) images
            if mu_coupling.dim() == 4:
                mu_img = mu_coupling[i].squeeze(0).numpy()
            else:
                mu_img = mu_coupling[i].reshape(self.config.image_shape[1], self.config.image_shape[2]).numpy()
            axes[0, i].imshow(mu_img, cmap="gray", vmin=-1, vmax=1)
            axes[0, i].set_title(f"Source {i+1}")
            axes[0, i].axis("off")

            # Target (nu) images
            if nu_coupling.dim() == 4:
                nu_img = nu_coupling[i].squeeze(0).numpy()
            else:
                nu_img = nu_coupling[i].reshape(self.config.image_shape[1], self.config.image_shape[2]).numpy()
            axes[1, i].imshow(nu_img, cmap="gray", vmin=-1, vmax=1)
            axes[1, i].set_title(f"Target {i+1}")
            axes[1, i].axis("off")

        plt.suptitle(f"Coupling after Step {time_step + 1}")
        plt.tight_layout()
        plt.show()
