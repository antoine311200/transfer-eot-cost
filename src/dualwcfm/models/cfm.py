from abc import ABC, abstractmethod
from typing import Optional

import torch
from torchdyn.core import NeuralODE

from dualwcfm.distributions import Distribution, CouplingDistribution

class torch_wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, t, x, *args, **kwargs):
        t = t.to(x.device)
        if t.dim() < 2:
            t = t.repeat(x.shape[0])[:, None]
        return self.model(torch.cat([x, t], dim=1))

class GenerativeModel(ABC):
    @abstractmethod
    def get_trajectory(self, n_samples: int = 0, x0: Optional[torch.Tensor] = None, n_timesteps: int = 100, device: Optional[torch.device] = None) -> torch.Tensor:
        pass

class ConditionalFlowMatching(GenerativeModel):
    """Main class for conditional flow matching

    Args:
        source_dist: Source distribution
        target_dist: Target distribution
        network: Neural network modeling the flow field
        sigma: Noise scale for conditional interpolation
        wrapper: Wrapper for the neural network to include time as input for Neural ODE support
    """

    def __init__(self, source_dist: Distribution, target_dist: Distribution, network, sigma: float = 0.0, wrapper=None):
        self.source_dist = source_dist
        self.target_dist = target_dist
        self.sigma = sigma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if wrapper is None:
            wrapper = torch_wrapper

        self.network = wrapper(network).to(self.device)
        self.node = NeuralODE(self.network, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

    def compute_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute conditional flow vector field

        Args:
            x0: Initial points, shape (batch_size, dim)
            x1: Target points, shape (batch_size, dim)
            t: Time steps, shape (batch_size,)

        Returns:
            Flow vectors, shape (batch_size, dim)
        """
        return x1 - x0

    def compute_conditional_xt(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute conditional interpolated points

        Args:
            x0: Initial points, shape (batch_size, dim)
            x1: Target points, shape (batch_size, dim)
            t: Time steps, shape (batch_size,)

        Returns:
            Interpolated points, shape (batch_size, dim)
        """
        return (1 - t) * x0 + t * x1 + torch.sqrt(2 * t * (1 - t)) * torch.randn_like(x0) * self.sigma

    def get_trajectory(self, n_samples: int = 0, x0: Optional[torch.Tensor] = None, n_timesteps: int = 100, device: Optional[torch.device] = None) -> torch.Tensor:
        """Sample entire flow trajectories

        Args:
            n_samples: Number of samples to generate if x0 is not provided
            x0: Initial points, shape (n_samples, dim). If provided, n_samples is ignored.
            n_timesteps: Number of time steps in the trajectory
            device: Device to perform computation on

        Returns:
            Trajectories, shape (n_timesteps, n_samples, dim)
        """
        if n_samples <= 0 and x0 is None:
            raise ValueError("Either n_samples must be positive or x0 must be provided.")

        self.network.eval()
        self.network.to(device if device is not None else self.device)

        with torch.no_grad():
            if x0 is None:
                x0 = self.source_dist.sample(n_samples).to(device if device is not None else self.device)
            trajectories = self.node.trajectory(x0, t_span=torch.linspace(0, 1, n_timesteps).to(x0.device))

        self.network.to(self.device)
        return trajectories


class CouplingDistributionFromTrajectory(CouplingDistribution):
    def __init__(self, base_dist: Distribution, coupling_model: GenerativeModel, **kwargs):
        self.base_dist = base_dist
        self.coupling_model = coupling_model
        self.n_timesteps = kwargs.get("n_timesteps", 100)

    def sample(self, n: int) -> torch.Tensor:
        return self.coupling_model.get_trajectory(n_samples=n, n_timesteps=self.n_timesteps)[-1, :, :] # type: ignore

    def sample_init(self, n: int) -> torch.Tensor:
        x0 = self.base_dist.sample(n)
        return x0

    def sample_target(self, n: int) -> torch.Tensor:
        return self.coupling_sample(n)[1]

    def coupling_sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = self.base_dist.sample(n)
        trajectories = self.coupling_model.get_trajectory(n_samples=n, n_timesteps=self.n_timesteps)
        x1 = trajectories[-1, :, :] # type: ignore
        return x0, x1

class FixedCouplingDistribution(CouplingDistribution):
    def __init__(self, base_dist: Distribution, coupling_model: GenerativeModel, **kwargs):
        self.base_dist = base_dist
        self.coupling_model = coupling_model
        self.n_timesteps = kwargs.get("n_timesteps", 100)
        self.total_samples = kwargs.get("total_samples", 10000)

        self.init_samples = self.base_dist.sample(self.total_samples)
        self.target_samples = self.coupling_model.get_trajectory(n_samples=self.total_samples, n_timesteps=self.n_timesteps)[-1, :, :]  # type: ignore
        print(self.init_samples.shape, self.target_samples.shape)

    def sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError("Use sample_target to sample from the target distribution.")

    def sample_init(self, n: int) -> torch.Tensor:
        indices = torch.randint(0, self.total_samples, (n,))
        return self.init_samples[indices]

    def sample_target(self, n: int) -> torch.Tensor:
        indices = torch.randint(0, self.total_samples, (n,))
        return self.target_samples[indices]

    def coupling_sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.total_samples, (n,))
        return self.init_samples[indices], self.target_samples[indices]
