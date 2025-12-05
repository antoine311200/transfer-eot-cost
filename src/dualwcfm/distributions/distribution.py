import math
from abc import ABC, abstractmethod

import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily, Independent, Normal
from torchcfm.utils import sample_8gaussians, sample_moons, sample_single_moon

class Distribution(ABC):
    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:pass

class MoonDistribution(Distribution):
    def sample(self, n: int) -> torch.Tensor:
        return sample_moons(n).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

class EightGaussiansDistribution(Distribution):
    def sample(self, n: int) -> torch.Tensor:
        return sample_8gaussians(n).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

class SingleMoonDistribution(Distribution):
    def sample(self, n: int) -> torch.Tensor:
        return sample_single_moon(n).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

class Simple2DGaussianDistribution(Distribution):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
            self.mean, self.std, self.device = mean, std, torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def get_cross_dist(rotation=0.0, shift=(0,0), scale=0.1):
    centers = [[0,0], [1.5,0], [-1.5,0], [0,1.5], [0,-1.5]]
    return GMMShapeDistribution(centers, scale=scale, rotation_degrees=rotation, shift=shift)

def get_square_dist(rotation=0.0, shift=(5,5), scale=0.1):
    d = 1.5
    centers = [[d,d], [d,-d], [-d,d], [-d,-d]]
    return GMMShapeDistribution(centers, scale=scale, rotation_degrees=rotation, shift=shift)



class LinearInterpolationDistribution(Distribution):
    def __init__(self, dist1: Distribution, dist2: Distribution, alpha: float):
        self.dist1 = dist1
        self.dist2 = dist2
        self.alpha = alpha

    def sample(self, n: int) -> torch.Tensor:
        n2 = int(self.alpha * n)
        n1 = n - n2

        if n1 == 0: return self.dist2.sample(n2)
        if n2 == 0: return self.dist1.sample(n1)

        samples1 = self.dist1.sample(n1)
        samples2 = self.dist2.sample(n2)
        samples = torch.cat([samples1, samples2], dim=0)
        indices = torch.randperm(samples.shape[0])
        return samples[indices]