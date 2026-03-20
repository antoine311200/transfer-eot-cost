"""dualwcfm.experiments.exp_2d
===============================
2-D toy-distribution experiment catalogue.

Design principle
----------------
The reference coupling π encodes a known point-transport (e.g. anticorrelation,
60° rotation) between Gaussian blobs at positions S₀.  The new source μ' and
new target ν' live at *different* positions S₁ but follow the **same** transport
dynamic — testing the algorithm's ability to generalise the learned map to
unseen supports.

Experiments
-----------
simple  – anticorrelation map (x ↦ −x) with blobs far from origin
medium  – 60° rotation of a 2-blob GMM → generalised to a 4-blob square
complex – 45° rotation of a 5-blob cross → generalised to a 3× larger cross

Each experiment exists in a *standard* and *perturbed* variant.  In the
perturbed variant the new target ν' is shifted / extra-rotated, probing
robustness to distribution mismatch between reference and new marginals.
"""

from __future__ import annotations

import math
from typing import List

import torch

from dualwcfm.config import FeedForwardConfig, TrainingConfig, VisualizeConfig
from dualwcfm.distributions.coupling import FunctionalCouplingDistribution, RotatedShapeCoupling
from dualwcfm.distributions.distribution import GMMShapeDistribution, Simple2DGaussianDistribution
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.experiments.base import ExperimentSpec, get_device


# ---------------------------------------------------------------------------
# Default hyperparameter configs
# ---------------------------------------------------------------------------

def _default_train_cfg(device: torch.device) -> TrainingConfig:
    return TrainingConfig(
        device                   = device,
        dataset_size             = 1_000_000,
        marginal_epochs          = 5_000,
        pretrain_epochs          = 10_000,
        dual_epochs              = 100,
        cfm_epochs               = 250,
        advance_steps            = 100,
        iterations               = 50,
        num_rectified_flow       = 1,
        scale_factor             = 100.0,
        batch_size               = 2_048 * 4,
        coupling_chunk_size      = 1_000,
        save_coupling_checkpoints= False,
        save_model_checkpoints   = False,
    )


def _default_method_cfg() -> MixtureConfig:
    return MixtureConfig(
        sigma          = 0.01,
        sigma2         = 0.01,
        loss           = "weighted",
        l2_reg         = 0,
        lip_reg        = 0,
        mean_reg       = 0,
        use_noisy_field= False,
        particle_ratio = 0.1,
        lr_cosine_decay = False,
        lr_min_factor   = 0.5,
    )


# ---------------------------------------------------------------------------
# simple – anticorrelation map
# ---------------------------------------------------------------------------

def _simple(perturbed: bool, output_root: str) -> ExperimentSpec:
    """Anticorrelation map: x ↦ −x (+ optional y-shift on target).

    Reference coupling
        x ~ N(0, 0.5²·I₂),  y = −x + [0, perturbation]
    New marginals
        μ' = N([loc, loc], 0.5²·I₂),  ν' = N([−loc, −loc+shift], 0.5²·I₂)
    Ground-truth transport map
        T(x) = −x + [0, perturbation]
    """
    device      = get_device()
    perturbation = 2.0 if perturbed else 0.0
    loc          = 10.0

    def _anticorrelate(n: int):
        x = torch.randn(n, 2) * 0.5
        y = -x + torch.tensor([0.0, perturbation])
        return x, y

    coupling = FunctionalCouplingDistribution(_anticorrelate)
    new_src  = Simple2DGaussianDistribution(
        mean = torch.tensor([loc, loc]),
        std  = torch.tensor([0.5, 0.5]),
    )
    new_tgt  = Simple2DGaussianDistribution(
        mean = torch.tensor([-loc, -loc + perturbation]),
        std  = torch.tensor([0.5, 0.5]),
    )

    def transport_map(x: torch.Tensor) -> torch.Tensor:
        # T(x) = -x + [0, perturbation]
        A = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
        b = torch.tensor([0.0, perturbation])
        return x @ A.T + b

    # T is its own inverse: T(T(x)) = -(-x + [0,p]) + [0,p] = x
    inverse_transport_map = transport_map

    train_cfg  = _default_train_cfg(device)
    method_cfg = _default_method_cfg()
    model_cfg  = FeedForwardConfig(dimension=2, hidden_size=32)
    vis_cfg    = VisualizeConfig(show_each=1, n_trajectories=300, trajectory_stride=2)

    return ExperimentSpec(
        name                  = f"2d_simple{'_perturbed' if perturbed else ''}",
        mode                  = "2d",
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
# medium – 60° rotation, 2-blob → 4-blob GMM
# ---------------------------------------------------------------------------

def _medium(perturbed: bool, output_root: str) -> ExperimentSpec:
    """60° rotation of a 2-blob GMM generalised to a 4-blob square GMM.

    Reference coupling
        2 blobs at [±2, 0], transport = R(60°).
    New marginals
        4-blob square at [±5, ±5];  target rotated by 60° (+ 10° if perturbed).
    """
    device      = get_device()
    ref_centers = [[2.0, 0.0], [-2.0, 0.0]]

    coupling = RotatedShapeCoupling(
        ref_centers, source_scale=0.4, target_scale=0.4,
        rotation_degrees=60.0, device=device,
    )
    new_centers = [[5.0, 5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, -5.0]]
    new_src = GMMShapeDistribution(new_centers, scale=0.4, device=device)
    new_tgt = GMMShapeDistribution(
        new_centers, scale=0.4,
        rotation_degrees=60.0 + (10.0 if perturbed else 0.0),
        device=device,
    )

    _theta_med = math.radians(60.0)
    _c, _s     = math.cos(_theta_med), math.sin(_theta_med)
    _R60       = torch.tensor([[_c, -_s], [_s,  _c]], dtype=torch.float32)
    _R60_inv   = _R60.T  # rotation inverse = transpose

    def transport_map_med(x: torch.Tensor) -> torch.Tensor:
        return x @ _R60.T

    def inverse_transport_map_med(y: torch.Tensor) -> torch.Tensor:
        return y @ _R60_inv.T

    train_cfg              = _default_train_cfg(device)
    train_cfg.iterations   = 100
    method_cfg             = _default_method_cfg()
    model_cfg              = FeedForwardConfig(dimension=2, hidden_size=32)
    vis_cfg                = VisualizeConfig(show_each=1, n_trajectories=2_000, trajectory_stride=2)

    return ExperimentSpec(
        name                  = f"2d_medium{'_perturbed' if perturbed else ''}",
        mode                  = "2d",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = transport_map_med,
        inverse_transport_map = inverse_transport_map_med,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


# ---------------------------------------------------------------------------
# complex – 45° rotation, 5-blob cross → 3× larger cross
# ---------------------------------------------------------------------------

def _complex(perturbed: bool, output_root: str) -> ExperimentSpec:
    """45° rotation of a 5-blob cross generalised to a 3× larger cross.

    Reference coupling
        Cross at radius r₀=1.5, transport = R(45°).
    New marginals
        Cross at radius r₁=4.5 (3× scale);  target rotated by 45° (+ 10° if perturbed).
    """
    device = get_device()
    r0     = 1.5
    r1     = 4.5

    ref_centers = [[0, 0], [r0, 0], [-r0, 0], [0, r0], [0, -r0]]
    coupling = RotatedShapeCoupling(
        ref_centers, source_scale=0.15, target_scale=0.15,
        rotation_degrees=45.0, device=device,
    )
    new_centers = [[0, 0], [r1, 0], [-r1, 0], [0, r1], [0, -r1]]
    new_src = GMMShapeDistribution(new_centers, scale=0.4, device=device)
    new_tgt = GMMShapeDistribution(
        new_centers, scale=0.4,
        rotation_degrees=45.0 + (10.0 if perturbed else 0.0),
        device=device,
    )

    _theta_cpx = math.radians(45.0)
    _c2, _s2   = math.cos(_theta_cpx), math.sin(_theta_cpx)
    _R45       = torch.tensor([[_c2, -_s2], [_s2,  _c2]], dtype=torch.float32)
    _R45_inv   = _R45.T

    def transport_map_cpx(x: torch.Tensor) -> torch.Tensor:
        return x @ _R45.T

    def inverse_transport_map_cpx(y: torch.Tensor) -> torch.Tensor:
        return y @ _R45_inv.T

    train_cfg                = _default_train_cfg(device)
    train_cfg.iterations     = 150
    train_cfg.dual_epochs    = 200
    train_cfg.cfm_epochs     = 300
    method_cfg               = _default_method_cfg()
    model_cfg                = FeedForwardConfig(dimension=2, hidden_size=32)
    vis_cfg                  = VisualizeConfig(show_each=1, n_trajectories=400, trajectory_stride=2)

    return ExperimentSpec(
        name                  = f"2d_complex{'_perturbed' if perturbed else ''}",
        mode                  = "2d",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = transport_map_cpx,
        inverse_transport_map = inverse_transport_map_cpx,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


# ---------------------------------------------------------------------------
# Public catalogue
# ---------------------------------------------------------------------------

def make_2d_experiments(output_root: str = "results") -> List[ExperimentSpec]:
    """Return all 2-D experiment specs (simple / medium / complex × standard / perturbed)."""
    return [
        _simple(False,  output_root),
        _simple(True,   output_root),
        _medium(False,  output_root),
        _medium(True,   output_root),
        _complex(False, output_root),
        _complex(True,  output_root),
    ]
