"""dualwcfm.experiments.exp_scalability
========================================
High-dimensional scalability experiments on top of the Gushchin framework.

Objective
---------
Test how DualWCFM scales to high-dimensional settings (d = 16, 32, 64, 128)
using a *simple, analytically-known* fixed-cost transport: the anticorrelation
map  T(x) = −x.

Why anticorrelation?
  - Ground truth is exact: T* = −I_d for any d
  - The entropic OT plan between N(0, I_d) and N(0, I_d) is known in closed form:
      π* = N(0, [[I, -I+εΛ]; [-I+εΛ, I]])  (approximately −I for small ε)
  - Easy to verify convergence: map_err should approach 0 as training proceeds
  - No dependence on dimension-specific structure (same map at every d)

Experiment design
-----------------
For each dimension d ∈ {16, 32, 64, 128}:

  Reference coupling
      x ~ N(0, I_d)
      y = −x  (exact anticorrelation; deterministic, no noise)

  New marginals (shifted to test generalisation)
      μ' = N([loc, …, loc], I_d)       with loc = 5.0
      ν' = N([−loc, …, −loc], I_d)

  Ground-truth transport (for the new marginals)
      T*(x) = −x   (same linear map, shift-invariant for linear transport)

Metrics reported
----------------
  SW₂(μ', ŷ)   — target marginal quality
  map_err        — ‖T*(x) − ŷ‖₂ (forward map error, RMSE)
  map_err_bwd    — ‖T*⁻¹(y) − x̂‖₂ (backward, same map since T* = T*⁻¹ here)
  sinkhorn_div   — debiased Sinkhorn divergence on target samples

Usage
-----
Direct::

    from dualwcfm.experiments.exp_scalability import make_scalability_experiments
    specs = make_scalability_experiments("results/scalability")
    for spec in specs:
        run_experiment(spec)

CLI (from project root)::

    python -m dualwcfm.experiments.run \\
        --exp scalability --dims 16 32 64 128 \\
        --output results/scalability \\
        --wandb-project exp_scalability

Baseline comparison::

    python -m dualwcfm.baselines.run \\
        --baseline genot --mode 2d \\
        --output results/scalability --wandb-project exp_scalability_baselines
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from dualwcfm.config import FeedForwardConfig, TrainingConfig, VisualizeConfig
from dualwcfm.distributions.coupling import FunctionalCouplingDistribution, CouplingDistribution
from dualwcfm.distributions.distribution import Distribution
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.experiments.base import ExperimentSpec, get_device
from dualwcfm.experiments.exp_gushchin import (
    GaussianDistribution,
    gaussian_eot_plan,
    _matrix_sqrt,
)

logger = logging.getLogger("dualwcfm")


# ---------------------------------------------------------------------------
# Anticorrelation coupling
# ---------------------------------------------------------------------------

class AnticorrelationDistribution(Distribution):
    """Isotropic Gaussian N(0, I_d).

    The anticorrelation coupling is  (x, y=-x), so the source distribution is
    simply a standard Gaussian and we compute the target by negation.
    """

    def __init__(self, dim: int, device: Optional[torch.device] = None):
        self._dev  = device or get_device()
        self._d    = dim

    def sample(self, n: int) -> torch.Tensor:
        return torch.randn(n, self._d, device=self._dev)


class ShiftedGaussian(Distribution):
    """Isotropic Gaussian N(loc * 1_d, I_d) for any shift vector loc."""

    def __init__(self, dim: int, loc: float, device: Optional[torch.device] = None):
        self._dev = device or get_device()
        self._d   = dim
        self.loc  = loc

    def sample(self, n: int) -> torch.Tensor:
        return torch.randn(n, self._d, device=self._dev) + self.loc


# ---------------------------------------------------------------------------
# Per-dimension experiment builder
# ---------------------------------------------------------------------------

def _anticorrelation_nd(
    dim: int,
    output_root: str,
    loc: float = 5.0,
    eps: float = 0.05,
    iterations: int = 100,
) -> ExperimentSpec:
    """Build the anticorrelation scalability experiment for R^dim.

    Parameters
    ----------
    dim:
        Ambient dimension (e.g. 16, 32, 64, 128).
    output_root:
        Results root directory.
    loc:
        Shift magnitude for the new source/target marginals.
    eps:
        Entropic regularisation level.  Small values (0.05) make the coupling
        nearly deterministic (T ≈ −I), which is ideal for this experiment.
    iterations:
        Number of dual adaptation iterations.
    """
    device = get_device()

    # ── Reference coupling: x ~ N(0, I_d), y = −x ──────────────────────
    def _anticorr_sample(n: int, _device=device):
        x = torch.randn(n, dim, device=_device)
        y = -x
        return x, y

    class AnticorrelatedCoupling(CouplingDistribution):
        def __init__(
            self,
            device=None,
        ):
            super().__init__()
            self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def sample_init(self, n: int) -> torch.Tensor:
            x, _ = self.coupling_sample(n)
            return x

        def sample_target(self, n: int) -> torch.Tensor:
            _, y = self.coupling_sample(n)
            return y

        def coupling_sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
            x = torch.randn(n, dim, device=self.device)
            y = -x
            return x, y

    coupling = FunctionalCouplingDistribution(_anticorr_sample)
    # coupling = AnticorrelatedCoupling(device=device)
    # print(coupling.coupling_sample(5)[0].shape)  # sanity check
    # from dualwcfm.distributions.coupling import FixedSizeCouplingDistribution, DiskCouplingDistribution
    # coupling = DiskCouplingDistribution(
    #             coupling,
    #             size=200000,
    #             save_dir='tmp/anticorr_coupling',
    #             chunk_size=100000,
    #             device='cuda',
    #         )
    # # coupling = FixedSizeCouplingDistribution(coupling, size=200000)
    # print(coupling.coupling_sample(5)[0].shape)  # sanity check
    # import sys
    # sys.exit(0)

    # ── New marginals: N([±loc]^d, I_d) ─────────────────────────────────
    new_src = ShiftedGaussian(dim, loc= loc, device=device)
    new_tgt = ShiftedGaussian(dim, loc=-loc, device=device)

    # ── Ground-truth transport map: T*(x) = −x (shift-invariant) ────────
    def transport_map(x: torch.Tensor) -> torch.Tensor:
        return -x

    # T* = T*⁻¹ for the anticorrelation map
    inverse_transport_map = transport_map

    # ── Hyperparameters (scale with dim) ─────────────────────────────────
    # Hidden size: 4× dim, capped at 1024
    hidden  = min(1024, max(256, 4 * dim))
    n_layers = 3 if dim <= 32 else 4

    train_cfg = TrainingConfig(
        device                    = device,
        dataset_size              = 500_000,
        marginal_epochs           = 3_000,
        pretrain_epochs           = 8_000,
        dual_epochs               = 100,
        cfm_epochs                = 1000,
        advance_steps             = 200,
        iterations                = iterations,
        scale_factor              = 50,       # auto
        batch_size                = 4_096,
        save_coupling_checkpoints = False,
        save_model_checkpoints    = True,
    )

    method_cfg = MixtureConfig(
        sigma           = 0.05,
        sigma2          = 0.05,
        loss            = "weighted",
        l2_reg          = 1e-2,
        lip_reg         = 0.5,
        mean_reg        = 0,
        use_noisy_field = False,
        particle_ratio  = 0.2,
        lr_cosine_decay = False,
        lr_min_factor   = 0.5,
        dual_warmup_epochs = 0,
    )

    model_cfg = FeedForwardConfig(
        dimension  = dim,
        hidden_size= hidden,
        num_layers = n_layers,
        flow_learning_rate = 1e-3,
        nu_learning_rate   = 1e-3,
        mu_learning_rate   = 1e-3,
        dual_learning_rate = 1e-3,
    )

    vis_cfg = VisualizeConfig(
        show_each          = 10,
        n_trajectories     = 0,      # no trajectory plots for high-dim
        trajectory_stride  = 1,
        show_each_marginal = 1000,
        show_each_pretrain = 1000,
    )

    return ExperimentSpec(
        name                  = f"scalability_anticorr_{dim}d",
        mode                  = "2d",   # use 2d metrics (SW, map_err) — no 3D scatter
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = transport_map,
        inverse_transport_map = inverse_transport_map,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


# ---------------------------------------------------------------------------
# Catalogue builder
# ---------------------------------------------------------------------------

def make_scalability_experiments(
    output_root: str = "results/scalability",
    dims: Tuple[int, ...] = (16, 32, 64, 128),
    eps:  float = 0.05,
    iterations: int = 100,
    loc: float = 5.0,
) -> List[ExperimentSpec]:
    """Return scalability experiment specs for all requested dimensions.

    Parameters
    ----------
    output_root:
        Root directory for all results.
    dims:
        Dimensions to run. Default: (16, 32, 64, 128).
    eps:
        Entropic regularisation for all experiments.
    iterations:
        Dual adaptation iterations per experiment.
    loc:
        Shift magnitude for new marginals.

    Returns
    -------
    List of ExperimentSpec, one per dimension.
    """
    return [
        _anticorrelation_nd(dim, output_root, loc=loc, eps=eps, iterations=iterations)
        for dim in dims
    ]


# ---------------------------------------------------------------------------
# Baseline spec builder (for running baselines on the same tasks)
# ---------------------------------------------------------------------------

def make_scalability_baseline_specs(
    output_root: str = "results/scalability",
    dims: Tuple[int, ...] = (16, 32, 64, 128),
    baseline_name: str = "genot",
) -> list:
    """Return BaselineSpec list for the given baseline on all scalability experiments.

    Usage::

        from dualwcfm.baselines import GENOTRunner, GENOTConfig
        from dualwcfm.experiments.exp_scalability import make_scalability_baseline_specs

        specs = make_scalability_baseline_specs()
        for spec in specs:
            runner = GENOTRunner(spec)
            runner.run()
    """
    from dualwcfm.experiments.base import ExperimentSpec
    import torch

    device = get_device()
    results = []
    for dim in dims:
        exp = _anticorrelation_nd(dim, output_root)
        results.append(exp)
    return results
