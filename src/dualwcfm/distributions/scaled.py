from __future__ import annotations

import torch
from typing import Tuple, Union

from dualwcfm.distributions.distribution import Distribution
from dualwcfm.distributions.coupling import CouplingDistribution, MarginalDistribution


class ScaledDistribution(Distribution):
    """Wraps a Distribution, dividing samples by `scale` (normalization).

    By default all sampling methods return *normalized* samples (divided by
    `scale`).  Pass ``denormalized=True`` to recover the original scale.

    Args:
        base: The underlying distribution to wrap.
        scale: Positive scalar used for normalization.  Samples are divided by
               this value in normalized mode and returned as-is in denormalized
               mode.
    """

    def __init__(self, base: Distribution, scale: float):
        self.base = base
        self.scale = scale

    def sample(self, n: int, denormalized: bool = False) -> torch.Tensor:
        samples = self.base.sample(n)
        return samples if denormalized else samples / self.scale

    # ------------------------------------------------------------------
    # Helpers for transforming already-sampled tensors
    # ------------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Divide a tensor by scale (raw → normalized)."""
        return x / self.scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply a tensor by scale (normalized → raw)."""
        return x * self.scale

    def clone(self) -> ScaledDistribution:
        return self.base.clone()

class ScaledCouplingDistribution(CouplingDistribution):
    """Wraps a CouplingDistribution, dividing samples by `scale` (normalization).

    By default all sampling methods return *normalized* samples (divided by
    `scale`).  Pass ``denormalized=True`` to recover the original scale.

    ``get_marginals()`` returns :class:`ScaledDistribution` instances so that
    any downstream code that only calls ``.sample()`` on a marginal also
    receives normalized data by default.

    Args:
        base: The underlying coupling distribution to wrap.
        scale: Positive scalar used for normalization.
    """

    def __init__(self, base: CouplingDistribution, scale: float):
        self.base = base
        self.scale = scale

    # ------------------------------------------------------------------
    # Core sampling methods
    # ------------------------------------------------------------------

    def coupling_sample(
        self, n: int, denormalized: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = self.base.coupling_sample(n)
        if denormalized:
            return x0, x1
        return x0 / self.scale, x1 / self.scale

    def sample_init(self, n: int, denormalized: bool = False) -> torch.Tensor:
        x = self.base.sample_init(n)
        return x if denormalized else x / self.scale

    def sample_target(self, n: int, denormalized: bool = False) -> torch.Tensor:
        x = self.base.sample_target(n)
        return x if denormalized else x / self.scale

    def sample(self, n: int, denormalized: bool = False) -> torch.Tensor:
        x0, x1 = self.coupling_sample(n, denormalized=denormalized)
        return torch.stack([x0, x1], dim=0)

    # ------------------------------------------------------------------
    # Marginals — return ScaledDistribution so callers also get normalized data
    # ------------------------------------------------------------------

    def get_marginals(self):
        mu = ScaledDistribution(
            MarginalDistribution(self.base.sample_init), scale=self.scale
        )
        nu = ScaledDistribution(
            MarginalDistribution(self.base.sample_target), scale=self.scale
        )
        return mu, nu

    # ------------------------------------------------------------------
    # Helpers for transforming already-sampled / already-evolved tensors
    # ------------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Divide a tensor by scale (raw → normalized)."""
        return x / self.scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply a tensor by scale (normalized → raw)."""
        return x * self.scale
