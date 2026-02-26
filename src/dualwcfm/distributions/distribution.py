import math
from abc import ABC, abstractmethod
from typing import Optional, Callable

import torch
from torch.distributions import Categorical, MixtureSameFamily, Independent, Normal
from torchcfm.utils import sample_8gaussians, sample_moons, sample_single_moon


class Distribution(ABC):
    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        pass

    @property
    def dim(self) -> int:
        """Return the dimensionality of the distribution."""
        if hasattr(self, "_dim"):
            return self._dim
        sample = self.sample(1)
        self._dim = sample.shape[-1]
        return self._dim


class FixedSizeDistribution(Distribution):
    def __init__(self, distribution: Distribution, size: int = 1000):
        self.distribution = distribution
        self.size = size
        self.dataset = self.distribution.sample(self.size)

    def sample(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return self.dataset[idx]

    def clone(self):
        import copy

        return copy.deepcopy(self)


class WeightedDistribution(Distribution):
    def __init__(self, base_fixed_distribution: FixedSizeDistribution):
        self.base_fixed_distribution = base_fixed_distribution
        self.log_weights = torch.zeros(self.base_fixed_distribution.size)

    def sample(self, n: int):  # type: ignore
        idx = torch.randint(0, self.base_fixed_distribution.size, (n,))
        samples = self.base_fixed_distribution.dataset[idx]
        log_w = self.log_weights[idx]
        return samples, log_w


class MoonDistribution(Distribution):
    def sample(self, n: int) -> torch.Tensor:
        return sample_moons(n).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) * 0.25


class EightGaussiansDistribution(Distribution):
    def sample(self, n: int) -> torch.Tensor:
        return sample_8gaussians(n).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


class SingleMoonDistribution(Distribution):
    def sample(self, n: int) -> torch.Tensor:
        return sample_single_moon(n).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


class Simple2DGaussianDistribution(Distribution):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, device="cpu"):
        self.mean, self.std, self.device = mean, std, device

    def sample(self, n: int) -> torch.Tensor:
        return torch.randn(n, 2, device=self.device) * self.std + self.mean


class GMMShapeDistribution(Distribution):
    def __init__(self, centers, scale=0.1, rotation_degrees=0.0, shift=(0.0, 0.0), device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.centers = torch.tensor(centers, dtype=torch.float32).to(self.device)
        self.shift = torch.tensor(shift, dtype=torch.float32).to(self.device)

        theta = torch.tensor(rotation_degrees * math.pi / 180.0)
        c, s = torch.cos(theta), torch.sin(theta)
        self.R = torch.tensor([[c, -s], [s, c]]).to(self.device)
        self.centers = (self.centers @ self.R.T) + self.shift

        n_components = self.centers.shape[0]
        mix = Categorical(torch.ones(n_components, device=self.device))
        comp = Independent(Normal(self.centers, torch.ones_like(self.centers) * scale), 1)
        self.gmm = MixtureSameFamily(mix, comp)

    def sample(self, n: int) -> torch.Tensor:
        return self.gmm.sample(torch.Size([n]))


def get_cross_dist(rotation=0.0, shift=(0, 0), scale=0.1):
    centers = [[0, 0], [1.5, 0], [-1.5, 0], [0, 1.5], [0, -1.5]]
    return GMMShapeDistribution(centers, scale=scale, rotation_degrees=rotation, shift=shift)


def get_square_dist(rotation=0.0, shift=(5, 5), scale=0.1):
    d = 1.5
    centers = [[d, d], [d, -d], [-d, d], [-d, -d]]
    return GMMShapeDistribution(centers, scale=scale, rotation_degrees=rotation, shift=shift)


class DynamicLinearInterpolationDistribution(Distribution):
    def __init__(self, dist1: Distribution, dist2: Distribution, get_alpha_fn: Optional[Callable]):
        self.dist1 = dist1
        self.dist2 = dist2
        self.get_alpha_fn = get_alpha_fn if get_alpha_fn is not None else (lambda: torch.rand(1).item())

    def sample(self, n: int) -> torch.Tensor:
        self.current_alpha = self.get_alpha_fn()

        n2 = int(self.current_alpha * n)
        n1 = n - n2

        if n1 == 0:
            return self.dist2.sample(n2)
        if n2 == 0:
            return self.dist1.sample(n1)

        samples1 = self.dist1.sample(n1)
        samples2 = self.dist2.sample(n2)
        samples = torch.cat([samples1, samples2], dim=0)
        indices = torch.randperm(samples.shape[0])
        return samples[indices]


class LinearInterpolationDistribution(DynamicLinearInterpolationDistribution):
    def __init__(self, dist1: Distribution, dist2: Distribution, alpha: float):
        super().__init__(dist1, dist2, get_alpha_fn=lambda: alpha)
        self.alpha = alpha


class RotatedDistribution(Distribution):
    def __init__(
        self,
        base_dist: Distribution,
        angle: float,
        transformation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Rotates samples from base_dist by given angle (in radians).

        Args:
            base_dist: Base distribution to rotate
            angle: Rotation angle in radians
            transformation: Optional transformation function to apply to samples before rotation
        """
        self.base_dist = base_dist
        self.angle = angle
        self.transformation = transformation if transformation is not None else lambda x: x

        self._rotation_matrix = self._create_rotation_matrix(angle)

    def _create_rotation_matrix(self, angle: float) -> torch.Tensor:
        """Create 2D rotation matrix for given angle."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)

    def sample(self, n: int) -> torch.Tensor:
        """
        Sample n points from base distribution and rotate them.

        Args:
            n: Number of samples

        Returns:
            Rotated samples of shape (n, 2)
        """
        samples = self.base_dist.sample(n)

        if samples.shape[-1] != 2:
            raise ValueError(
                f"RotatedDistribution requires 2D samples, " f"but base distribution has dimension {samples.shape[-1]}"
            )

        samples = self.transformation(samples)
        self._rotation_matrix = self._rotation_matrix.to(samples.device)
        rotated_samples = torch.matmul(samples, self._rotation_matrix.T)
        return rotated_samples

    @property
    def dim(self) -> int:
        return self.base_dist.dim

    def rotate(self, new_angle: float) -> "RotatedDistribution":
        """Create a new RotatedDistribution with different angle."""
        return RotatedDistribution(self.base_dist, new_angle)


class SphereDistribution(Distribution):
    """Samples points uniformly inside a 3D sphere."""

    def __init__(self, mean: torch.Tensor, radius: float = 0.5):
        self.mean = mean
        self.radius = radius

    def sample(self, n: int) -> torch.Tensor:
        # Sample random directions
        phi = torch.rand(n) * 2 * math.pi
        costheta = torch.rand(n) * 2 - 1
        u = torch.rand(n)

        theta = torch.acos(costheta)
        r = self.radius * (u ** (1 / 3))  # Cube root for uniform volume sampling

        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)

        return torch.stack([x, y, z], dim=-1) + self.mean
