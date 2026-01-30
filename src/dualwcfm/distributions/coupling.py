from abc import ABC, abstractmethod
import math

from typing import Callable, Union
import torch

from dualwcfm.distributions.distribution import Distribution


class MarginalDistribution(Distribution):
    def __init__(self, sample_func):
        self.sample_func = sample_func

    def sample(self, n: int) -> torch.Tensor:
        return self.sample_func(n)


class CouplingDistribution(Distribution, ABC):
    @abstractmethod
    def coupling_sample(self, n: int) -> Union[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor], torch.Tensor]:
        pass

    @abstractmethod
    def sample_init(self, n: int) -> torch.Tensor:
        pass

    @abstractmethod
    def sample_target(self, n: int) -> torch.Tensor:
        pass

    def get_marginals(self) -> tuple[Distribution, Distribution]:
        return MarginalDistribution(self.sample_init), MarginalDistribution(self.sample_target)

    def sample(self, n: int) -> torch.Tensor:
        return super().coupling_sample(n)  # type: ignore

    def clone(self):
        # Clone the coupling distribution and everything inside
        import copy

        return copy.deepcopy(self)


class FunctionalCouplingDistribution(CouplingDistribution):
    def __init__(self, coupling_func):
        self.coupling_func = coupling_func

    def sample(self, n: int) -> torch.Tensor:
        x0, x1 = self.coupling_func(n)
        return torch.stack([x0, x1], dim=0)

    def sample_init(self, n: int) -> torch.Tensor:
        x0, _ = self.coupling_func(n)
        return x0

    def sample_target(self, n: int) -> torch.Tensor:
        _, x1 = self.coupling_func(n)
        return x1

    def coupling_sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.coupling_func(n)


class RotatedShapeCoupling(CouplingDistribution):
    def __init__(
        self,
        centers,
        source_scale=0.1,
        target_scale=0.1,
        rotation_degrees=45.0,
        source_shift=(0, 0),
        target_shift=(0, 0),
        transformation: Callable[[torch.Tensor], torch.Tensor] = None,  # New parameter
        device=None,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.centers = torch.tensor(centers, dtype=torch.float32).to(self.device)
        self.source_scale = source_scale
        self.target_scale = target_scale
        self.source_shift = torch.tensor(source_shift, dtype=torch.float32).to(self.device)
        self.target_shift = torch.tensor(target_shift, dtype=torch.float32).to(self.device)

        # Set transformation to identity if none is provided
        self.transformation = transformation if transformation is not None else lambda x: x

        theta = torch.tensor(rotation_degrees * math.pi / 180.0)
        c, s = torch.cos(theta), torch.sin(theta)
        self.R = torch.tensor([[c, -s], [s, c]]).to(self.device)

        self.n_components = self.centers.shape[0]
        self.cat_dist = torch.distributions.Categorical(torch.ones(self.n_components, device=self.device))

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
        x_centered = chosen_centers + noise_x * self.source_scale

        # 1. Final source coordinates
        x = x_centered + self.source_shift

        # 2. Apply transformation T(x) before rotation
        x_transformed = self.transformation(x_centered)

        # 3. Rotate the transformed coordinates and scale
        y_centered = (x_transformed @ self.R.T) * (self.target_scale / self.source_scale)
        y = y_centered + self.target_shift

        return x, y

    def transform_source(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.source_shift
        x_transformed = self.transformation(x_centered)
        y_centered = (x_transformed @ self.R.T) * (self.target_scale / self.source_scale)
        y = y_centered + self.target_shift
        return y


class FixedSizeCouplingDistribution(CouplingDistribution):
    def __init__(self, coupling_distribution: CouplingDistribution, size: int = 1000):
        self.coupling_distribution = coupling_distribution
        self.size = size
        self.dataset = torch.stack(list(self.coupling_distribution.coupling_sample(self.size)), dim=0)

    @staticmethod
    def from_dataset(dataset: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
        obj = FixedSizeCouplingDistribution.__new__(FixedSizeCouplingDistribution)
        obj.size = dataset[0].shape[0]
        obj.dataset = dataset
        return obj

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def coupling_sample(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return (self.dataset[0][idx], self.dataset[1][idx])

    def sample_init(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return self.dataset[0][idx]

    def sample_target(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return self.dataset[1][idx]


class WeightedCouplingDistribution(CouplingDistribution):
    def __init__(self, base_fixed_coupling: CouplingDistribution, size: int = 1000):
        self.base_fixed_coupling = base_fixed_coupling
        self.size = size
        self.dataset = self.base_fixed_coupling.coupling_sample(self.size)
        self.log_weights = torch.zeros(self.size)

    def coupling_sample(self, n: int):  # type: ignore
        idx = torch.randint(0, self.size, (n,))
        samples = (self.dataset[0][idx], self.dataset[1][idx])
        log_w = self.log_weights[idx]
        return samples, log_w

    def sample_init(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return self.dataset[0][idx]

    def sample_target(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return self.dataset[1][idx]

    def sample(self, n: int):
        pass  # type: ignore
