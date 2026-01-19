from .distribution import (
    Distribution,
    LinearInterpolationDistribution,
    Simple2DGaussianDistribution,
    MoonDistribution,
    RotatedDistribution,
    FixedSizeDistribution
)
from .coupling import CouplingDistribution, FixedSizeCouplingDistribution, MarginalDistribution, RotatedShapeCoupling

import torch
from typing import Tuple, Optional
from dualwcfm.distributions.coupling import FunctionalCouplingDistribution
from dualwcfm.distributions import Simple2DGaussianDistribution


class DistributionManager:
    """Simple wrapper for one coupling and two distributions."""

    def __init__(self, coupling_distribution: CouplingDistribution, new_source_distribution: Distribution, new_target_distribution: Distribution, dataset_size: int = 200000, device: Optional[torch.device] = None):
        self.dataset_size = dataset_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.coupling_dist = FixedSizeCouplingDistribution(coupling_distribution, size=dataset_size)
        self.current_coupling_dist = self.coupling_dist.clone()
        self.mu_dist, self.nu_dist = self.current_coupling_dist.get_marginals()
        self.mu_dist = FixedSizeDistribution(self.mu_dist, size=dataset_size)
        self.nu_dist = FixedSizeDistribution(self.nu_dist, size=dataset_size)
        self.new_source_dist = new_source_distribution
        self.new_target_dist = new_target_distribution

    def get_coupling_distribution(self):
        """Get the original coupling distribution."""
        return self.coupling_dist

    def get_current_coupling_distribution(self):
        """Get the current (evolving) coupling distribution."""
        return self.current_coupling_dist

    def get_marginal_distributions(self) -> Tuple:
        """Get the current marginal distributions (mu and nu)."""
        return self.mu_dist, self.nu_dist

    def get_new_source_distribution(self):
        """Get the new source distribution (mu')."""
        return self.new_source_dist

    def get_new_target_distribution(self):
        """Get the new target distribution (nu')."""
        return self.new_target_dist