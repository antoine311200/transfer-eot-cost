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
from dualwcfm.distributions.distribution import GMMShapeDistribution, Simple2DGaussianDistribution, Distribution
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.experiments.base import ExperimentSpec, get_device


# ---------------------------------------------------------------------------
# Default hyperparameter configs
# ---------------------------------------------------------------------------

def _default_train_cfg(device: torch.device) -> TrainingConfig:
    return TrainingConfig(
        device                   = device,
        dataset_size             = 1_000_000,
        marginal_epochs          = 2_000,
        pretrain_epochs          = 5_000,
        dual_epochs              = 100,
        cfm_epochs               = 500,
        advance_steps            = 100,
        iterations               = 50,
        num_rectified_flow       = 0,
        scale_factor             = 50,
        batch_size               = 2048 * 4,
        coupling_chunk_size      = 100_000,
        save_coupling_checkpoints= False,
        save_model_checkpoints   = False,
    )


def _default_method_cfg() -> MixtureConfig:
    return MixtureConfig(
        sigma          = 0.1, # t time
        sigma2         = 0.1, # s time
        loss           = "weighted",
        l2_reg         = 1e-2,
        lip_reg        = 1.0,
        mean_reg       = 0,
        use_noisy_field= False,
        particle_ratio = 0.1,
        lr_cosine_decay = False,
        lr_min_factor   = 0.5,
        dual_warmup_epochs = 0,
    )

def _default_model_cfg(dimension: int = 2, hidden_size: int = 32, num_layers: int = 2) -> FeedForwardConfig:
    return FeedForwardConfig(
        dimension=dimension,
        hidden_size=hidden_size,
        num_layers=num_layers,
        flow_learning_rate=1e-3,
        nu_learning_rate=1e-3,
        mu_learning_rate=1e-3,
        dual_learning_rate=1e-3,
        dual_l2_reg=0,#1e-3,
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
        A = torch.tensor([[-1.0, 0.0], [0.0, -1.0]], device=x.device, dtype=x.dtype)
        b = torch.tensor([0.0, perturbation], device=x.device, dtype=x.dtype)
        return x @ A.T + b

    # T is its own inverse: T(T(x)) = -(-x + [0,p]) + [0,p] = x
    inverse_transport_map = transport_map

    train_cfg  = _default_train_cfg(device)
    train_cfg.marginal_epochs    = 2_000   # linear transport converges fast; fewer phases needed
    # train_cfg.num_rectified_flow = 1       # one rectification pass is sufficient for a linear map
    method_cfg = _default_method_cfg()
    method_cfg.sigma = 0.03                # blobs are narrow (std=0.5) and far apart (loc=10); small σ avoids over-spreading
    model_cfg  = _default_model_cfg()
    vis_cfg    = VisualizeConfig(show_each=1, n_trajectories=2000, trajectory_stride=2, show_each_marginal=100, show_each_pretrain=1000)

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

    # Transport map for the *new* marginals: rotate by the same angle as the
    # new target (60° standard, 70° perturbed).  For the reference coupling the
    # rotation is always 60°; the perturbed experiment tests adaptation to an
    # extra 10° tilt on the target, so T must reflect the new marginals.
    _theta_med = math.radians(60.0 + (10.0 if perturbed else 0.0))
    _c, _s     = math.cos(_theta_med), math.sin(_theta_med)
    _R_med     = torch.tensor([[_c, -_s], [_s,  _c]], dtype=torch.float32)

    def transport_map_med(x: torch.Tensor) -> torch.Tensor:
        return x @ _R_med.to(x.device, x.dtype).T

    def inverse_transport_map_med(y: torch.Tensor) -> torch.Tensor:
        return y @ _R_med.to(y.device, y.dtype)

    train_cfg              = _default_train_cfg(device)
    train_cfg.iterations   = 100
    method_cfg             = _default_method_cfg()
    model_cfg              = _default_model_cfg(hidden_size=64)   # 4-blob target needs more capacity than simple
    vis_cfg    = VisualizeConfig(show_each=1, n_trajectories=2000, trajectory_stride=2, show_each_marginal=100, show_each_pretrain=1000)

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
    r1     = 10.0

    ref_centers = [[0, 0], [r0, 0], [-r0, 0], [0, r0], [0, -r0]]
    coupling = RotatedShapeCoupling(
        ref_centers, source_scale=0.15, target_scale=0.15,
        rotation_degrees=45.0, device=device,
    )
    new_centers = [[0, 0], [r1, 0], [-r1, 0], [0, r1], [0, -r1]]
    new_src = GMMShapeDistribution(new_centers, scale=0.15, device=device)
    new_tgt = GMMShapeDistribution(
        new_centers, scale=0.15,
        rotation_degrees=45.0 + (10.0 if perturbed else 0.0),
        device=device,
    )

    # As with medium: map angle matches the *new* target rotation angle.
    _theta_cpx = math.radians(45.0 + (10.0 if perturbed else 0.0))
    _c2, _s2   = math.cos(_theta_cpx), math.sin(_theta_cpx)
    _R_cpx     = torch.tensor([[_c2, -_s2], [_s2,  _c2]], dtype=torch.float32)

    def transport_map_cpx(x: torch.Tensor) -> torch.Tensor:
        return x @ _R_cpx.to(x.device, x.dtype).T

    def inverse_transport_map_cpx(y: torch.Tensor) -> torch.Tensor:
        return y @ _R_cpx.to(y.device, y.dtype)

    train_cfg                = _default_train_cfg(device)
    train_cfg.iterations     = 100              # complex needs more iterations than simple
    train_cfg.pretrain_epochs = 8_000           # larger flow pretraining for 5-blob cross with 6.7× scale jump
    train_cfg.dual_epochs    = 150              # duals need more steps for complex geometry
    train_cfg.scale_factor   = 30              # blobs at r1=10; scale_factor=50 over-normalises
    method_cfg               = _default_method_cfg()
    method_cfg.l2_reg        = 5e-3            # less dual potential regularisation for complex geometry
    method_cfg.lip_reg       = 1.0             # stronger Lipschitz stability for large scale jumps
    method_cfg.particle_ratio = 0.2            # more particle mixing for 5-blob cross
    method_cfg.lr_min_factor  = 0.3            # more LR headroom per iteration
    method_cfg.dual_warmup_epochs = 50         # warm-start duals for hard geometry
    model_cfg                = _default_model_cfg(hidden_size=128, num_layers=3)  # deeper network for complex transport
    vis_cfg                  = VisualizeConfig(show_each=1, n_trajectories=2000, trajectory_stride=2, show_each_marginal=100, show_each_pretrain=1000)

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
# moon – upper moon → lower moon transport
# ---------------------------------------------------------------------------

class _SingleMoonDist(Distribution):
    """Sample from one half of the two-moons dataset (class 0 = upper, 1 = lower).

    Uses sklearn make_moons for reliable label-paired sampling.  Samples are
    scaled and shifted to allow placing the moons at arbitrary positions.
    """

    def __init__(self, moon_class: int, scale: float = 1.0,
                 shift=(0.0, 0.0), noise: float = 0.07,
                 device=None):
        self._dev   = device or get_device()
        self._class = moon_class
        self._scale = scale
        self._shift = torch.tensor(shift, dtype=torch.float32)
        self._noise = noise

    def sample(self, n: int) -> torch.Tensor:
        from sklearn.datasets import make_moons
        # Over-sample to guarantee at least n points per class after filtering
        X, labels = make_moons(n_samples=max(6 * n, 2000), noise=self._noise)
        X = torch.tensor(X, dtype=torch.float32)
        pts = X[labels == self._class][:n]
        return (pts * self._scale + self._shift).to(self._dev)


def _moon(perturbed: bool, output_root: str) -> ExperimentSpec:
    """Upper-moon → lower-moon transport (sklearn two-moons dataset).

    Reference coupling
        x ~ upper moon (class 0), y ~ lower moon (class 1), randomly paired.
        Both at unit scale, centred near origin.
    New marginals
        μ' = upper moon scaled by 3× and shifted to [4, 4]
        ν' = lower moon scaled by 3× and shifted to [4, 4]
               (+ 20° global rotation if perturbed, testing distribution mismatch)
    Transport map
        No analytical ground truth — map-error metrics are omitted.
    """
    device = get_device()
    noise  = 0.07
    ref_scale  = 1.5          # reference coupling scale
    new_scale  = 4.5          # new-marginal scale (3× reference)
    new_shift  = (4.0, 4.0)   # move new marginals away from reference

    def _moon_coupling(n: int, _dev=device, _ns=noise, _rs=ref_scale):
        from sklearn.datasets import make_moons
        X, labels = make_moons(n_samples=max(6 * n, 2000), noise=_ns)
        X = torch.tensor(X, dtype=torch.float32) * _rs
        x = X[labels == 0][:n]
        y = X[labels == 1][:n]
        return x.to(_dev), y.to(_dev)

    coupling = FunctionalCouplingDistribution(_moon_coupling)

    if perturbed:
        # Rotate the new-target moon by 20° to create distribution mismatch
        _theta = math.radians(20.0)
        _c, _s = math.cos(_theta), math.sin(_theta)
        _R     = torch.tensor([[_c, -_s], [_s, _c]], dtype=torch.float32)

        class _RotatedMoonDist(Distribution):
            def __init__(self, base: _SingleMoonDist, R):
                self._base = base
                self._R    = R
            def sample(self, n: int) -> torch.Tensor:
                pts = self._base.sample(n)
                return (pts @ self._R.to(pts.device, pts.dtype).T)

        new_src = _SingleMoonDist(0, scale=new_scale, shift=new_shift, noise=noise, device=device)
        new_tgt = _RotatedMoonDist(
            _SingleMoonDist(1, scale=new_scale, shift=new_shift, noise=noise, device=device), _R
        )
    else:
        new_src = _SingleMoonDist(0, scale=new_scale, shift=new_shift, noise=noise, device=device)
        new_tgt = _SingleMoonDist(1, scale=new_scale, shift=new_shift, noise=noise, device=device)

    train_cfg               = _default_train_cfg(device)
    train_cfg.iterations    = 100
    train_cfg.pretrain_epochs = 8_000
    train_cfg.scale_factor  = 20
    method_cfg              = _default_method_cfg()
    method_cfg.sigma        = 0.05
    method_cfg.particle_ratio = 0.2
    method_cfg.lip_reg      = 1.0
    method_cfg.dual_warmup_epochs = 30
    model_cfg               = _default_model_cfg(hidden_size=128, num_layers=3)
    vis_cfg                 = VisualizeConfig(
        show_each=1, n_trajectories=2000, trajectory_stride=2,
        show_each_marginal=100, show_each_pretrain=1000,
    )

    return ExperimentSpec(
        name                  = f"2d_moon{'_perturbed' if perturbed else ''}",
        mode                  = "2d",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = None,   # no analytical map for moons
        inverse_transport_map = None,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


# ---------------------------------------------------------------------------
# Public catalogue
# ---------------------------------------------------------------------------

def make_2d_experiments(output_root: str = "results") -> List[ExperimentSpec]:
    """Return all 2-D experiment specs (simple / medium / complex / moon × standard / perturbed)."""
    return [
        _simple(False,  output_root),
        _simple(True,   output_root),
        _medium(False,  output_root),
        _medium(True,   output_root),
        _complex(False, output_root),
        _complex(True,  output_root),
        _moon(False,    output_root),
        _moon(True,     output_root),
    ]
