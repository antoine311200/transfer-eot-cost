from abc import ABC, abstractmethod
import math

import torch

from dualwcfm.distributions.distribution import Distribution

class MarginalDistribution(Distribution):
    def __init__(self, sample_func):
        self.sample_func = sample_func

    def sample(self, n: int) -> torch.Tensor:
        return self.sample_func(n)

class CouplingDistribution(Distribution, ABC):
    @abstractmethod
    def coupling_sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:pass

    @abstractmethod
    def sample_init(self, n: int) -> torch.Tensor:pass

    @abstractmethod
    def sample_target(self, n: int) -> torch.Tensor:pass

    def get_marginals(self) -> tuple[Distribution, Distribution]:
        return MarginalDistribution(self.sample_init), MarginalDistribution(self.sample_target)

class RotatedShapeCoupling(CouplingDistribution):
    def __init__(self, centers, source_scale=0.1, target_scale=0.1, rotation_degrees=45.0, source_shift=(0,0), target_shift=(0,0), device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.centers = torch.tensor(centers, dtype=torch.float32).to(self.device)
        self.source_scale = source_scale
        self.target_scale = target_scale
        self.source_shift = torch.tensor(source_shift, dtype=torch.float32).to(self.device)
        self.target_shift = torch.tensor(target_shift, dtype=torch.float32).to(self.device)

        theta = torch.tensor(rotation_degrees * math.pi / 180.0)
        c, s = torch.cos(theta), torch.sin(theta)
        self.R = torch.tensor([[c, -s], [s, c]]).to(self.device)

        self.n_components = self.centers.shape[0]
        self.cat_dist = torch.distributions.Categorical(torch.ones(self.n_components, device=self.device))

    def sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError("Use sample_init or sample_target to sample from the marginals.")

    def sample_init(self, n: int) -> torch.Tensor:
        x, _ = self.coupling_sample(n)
        return x

    def sample_target(self, n: int) -> torch.Tensor:
        _, y = self.coupling_sample(n)
        return y

    def coupling_sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.cat_dist.sample((torch.Size([n])))
        chosen_centers = self.centers[indices]

        noise_x = torch.randn(n, 2, device=self.device)
        noise_y = torch.randn(n, 2, device=self.device)

        x_centered = chosen_centers + noise_x * self.source_scale
        x = x_centered + self.source_shift

        y_centered = (x_centered @ self.R.T) * self.target_scale / self.source_scale
        y = y_centered + self.target_shift

        return x, y
