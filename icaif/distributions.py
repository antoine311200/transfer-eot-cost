"""Finance-specific Distribution subclasses.

FinancialCouplingDistribution  — wraps precomputed (x, y) tensor pairs.
FinancialMarginalDistribution  — wraps a single precomputed tensor.

Both are thin wrappers; all financial data is precomputed and stored as
float32 tensors.  Sampling is random-index selection (with replacement).
"""

import torch
import numpy as np
from dualwcfm.distributions.coupling import CouplingDistribution
from dualwcfm.distributions.distribution import Distribution


class FinancialCouplingDistribution(CouplingDistribution):
    """Coupling distribution backed by precomputed (x, y) pairs.

    Args:
        x: numpy array or Tensor of shape (N, D) — source samples.
        y: numpy array or Tensor of shape (N, D) — target samples.
        device: torch device.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, device: torch.device):
        self.x = torch.as_tensor(x, dtype=torch.float32, device=device)
        self.y = torch.as_tensor(y, dtype=torch.float32, device=device)
        assert self.x.shape == self.y.shape, (
            f"x and y must have identical shapes; got {self.x.shape} vs {self.y.shape}"
        )
        self._device = device

    def _rand_idx(self, n: int) -> torch.Tensor:
        return torch.randint(0, len(self.x), (n,), device=self._device)

    def coupling_sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self._rand_idx(n)
        return self.x[idx], self.y[idx]

    def sample_init(self, n: int) -> torch.Tensor:
        return self.x[self._rand_idx(n)]

    def sample_target(self, n: int) -> torch.Tensor:
        return self.y[self._rand_idx(n)]

    def sample(self, n: int) -> torch.Tensor:
        return self.sample_init(n)

    @property
    def dim(self) -> int:
        return self.x.shape[-1]

    def __len__(self) -> int:
        return len(self.x)


class FinancialMarginalDistribution(Distribution):
    """Marginal distribution backed by a precomputed tensor.

    Args:
        data: numpy array or Tensor of shape (N, D).
        device: torch device.
    """

    def __init__(self, data: np.ndarray, device: torch.device):
        self.data = torch.as_tensor(data, dtype=torch.float32, device=device)
        self._device = device

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, len(self.data), (n,), device=self._device)
        return self.data[idx]

    @property
    def dim(self) -> int:
        return self.data.shape[-1]

    def __len__(self) -> int:
        return len(self.data)
