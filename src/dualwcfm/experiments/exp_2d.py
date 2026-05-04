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
simple  – anticorrelation map (x ↦ -x) with blobs far from origin
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
from dualwcfm.distributions.distribution import (
    GMMShapeDistribution, Simple2DGaussianDistribution, Distribution,
    CheckerboardDistribution, UniformRectDistribution, TransformedDistribution, UnionDistribution,
)
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
        advance_steps            = 20,
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
        sigma          = 0.01, # t time
        sigma2         = 0.01, # s time
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
    """Anticorrelation map: x ↦ -x (+ optional y-shift on target).

    Reference coupling
        x ~ N(0, 0.5²·I₂),  y = -x + [0, perturbation]
    New marginals
        μ' = N([loc, loc], 0.5²·I₂),  ν' = N([-loc, -loc+shift], 0.5²·I₂)
    Ground-truth transport map
        T(x) = -x + [0, perturbation]
    """
    device      = get_device()
    perturbation = 2.0 if perturbed else 0.0
    loc          = 10.0

    def _anticorrelate(n: int):
        x = torch.randn(n, 2) * 0.5
        y = -x# + torch.tensor([0.0, perturbation])
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
    method_cfg.sigma2 = 0.1               # wider bridge on s-time axis; confirmed optimal in W&B run te1d2h6c
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

def _far_away(perturbed: bool, output_root: str) -> ExperimentSpec:
    device      = get_device()
    perturbation = 2.0 if perturbed else 0.0
    loc          = 10000.0

    def _anticorrelate(n: int):
        x = torch.randn(n, 2) * 0.5
        y = -x# + torch.tensor([0.0, perturbation])
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
    train_cfg.marginal_epochs    = 10_000   # linear transport converges fast; fewer phases needed
    # Modify scale_factor to avoid numerical instability from large loc=10000 values; this also tests the algorithm's ability to handle large scale jumps.
    train_cfg.scale_factor   = 1000.0      # blobs at loc=10000; scale_factor=1000 normalises to ~10, avoiding overflow while preserving relative scale
    # train_cfg.num_rectified_flow = 1       # one rectification pass is sufficient for a linear map
    method_cfg = _default_method_cfg()
    method_cfg.sigma = 0.03                # blobs are narrow (std=0.5) and far apart (loc=10000); small σ avoids over-spreading
    method_cfg.sigma2 = 0.03
    model_cfg  = _default_model_cfg()
    vis_cfg    = VisualizeConfig(show_each=1, n_trajectories=2000, trajectory_stride=2, show_each_marginal=100, show_each_pretrain=1000)

    return ExperimentSpec(
        name                  = f"2d_far{'_perturbed' if perturbed else ''}",
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
    model_cfg.use_siren = True
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

def _moon_points(moon_class: int, t: torch.Tensor) -> torch.Tensor:
    """Analytical moon positions from angle parameter t ∈ [0, π].

    Class 0 (upper):  (cos t,       sin t)
    Class 1 (lower):  (1 - cos t,   0.5 - sin t)

    Both follow sklearn's make_moons parameterisation exactly, so the pairing
    by equal angle gives a valid, deterministic transport.
    """
    if moon_class == 0:
        return torch.stack([torch.cos(t), torch.sin(t)], dim=1)
    else:
        return torch.stack([1.0 - torch.cos(t), 0.5 - torch.sin(t)], dim=1)


class _SingleMoonDist(Distribution):
    """Sample from one half of the two-moons dataset (class 0 = upper, 1 = lower).

    Samples angles t ∼ Uniform(0, π) and returns the corresponding point on
    the moon arc ± Gaussian noise.  This ensures every call returns a fresh,
    unbiased random batch covering the full arc, unlike the legacy sklearn
    approach which took the first n points from a sorted grid (biased toward
    the first third of the arc) and was deterministic at noise=0.

    Scaled by *scale* and translated by *shift* to position the distribution.
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
        t   = torch.rand(n) * math.pi          # Uniform(0, π)
        pts = _moon_points(self._class, t)      # (n, 2)
        if self._noise > 0:
            pts = pts + self._noise * torch.randn_like(pts)
        return (pts * self._scale + self._shift).to(self._dev)


class _CircleDist(Distribution):
    """Uniform samples on a circle of given radius and center, with Gaussian noise."""

    def __init__(self, center=(0.0, 0.0), radius: float = 3.0,
                 noise: float = 0.07, device=None):
        self._dev    = device or get_device()
        self._center = torch.tensor(center, dtype=torch.float32)
        self._radius = radius
        self._noise  = noise

    def sample(self, n: int) -> torch.Tensor:
        theta = torch.rand(n) * 2 * math.pi
        pts   = torch.stack([
            self._center[0] + self._radius * torch.cos(theta),
            self._center[1] + self._radius * torch.sin(theta),
        ], dim=1)
        if self._noise > 0:
            pts = pts + self._noise * torch.randn_like(pts)
        return pts.to(self._dev)


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
    noise      = 0.07          # per-point Gaussian noise added to both x and y
    ref_scale  = 1.5           # reference coupling scale
    new_scale  = 4.5           # new-marginal scale (3× reference)
    new_shift  = (4.0, 4.0)    # move new marginals away from reference

    def _moon_coupling(n: int, _dev=device, _ns=noise, _rs=ref_scale):
        # Sample the SAME n random angles for both classes so (x[i], y[i]) are
        # truly paired: y[i] = T(x[i]) + noise, where T maps upper→lower moon
        # via the shared angle parameter.  Using max(6n,2000) sorted [:n] was
        # wrong: it covered only the first ~π/3 of the arc and was deterministic
        # at noise=0.
        t = torch.rand(n) * math.pi             # Uniform(0, π) — full arc coverage
        x = _moon_points(0, t) * _rs            # upper moon
        y = _moon_points(1, t) * _rs            # lower moon  (same angle → paired)
        if _ns > 0:
            x = x + _ns * torch.randn_like(x)
            y = y + _ns * torch.randn_like(y)
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
    method_cfg.sigma2       = 0.05
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
# circle – full circle → split semi-circles transport
# ---------------------------------------------------------------------------

def _circle(perturbed: bool, output_root: str) -> ExperimentSpec:
    """Circle → split semi-circles transport.

    Reference coupling
        μ:  full circle centered at origin, radius 3.
        ν:  upper semi-circle (y ≥ 0) shifted +5 in y (centred at y=+5);
            lower semi-circle (y < 0) shifted -5 in y (centred at y=-5).
        Transport: T(x, y) = (x, y + 5·sign(y))

    New marginals
        μ': circle centred at (8, -1), radius 2 — x-axis crosses at ¾ height,
            so ¾ of particles have y < 0 (go down) and ¼ have y > 0 (go up).
        ν': same T applied to μ', giving two arcs vertically separated by ±5.
            In the *perturbed* variant the new target is additionally rotated
            15° to create a distribution mismatch.
    """
    device      = get_device()
    ref_radius  = 3.0
    new_radius  = 2.0
    shift_y     = 5.0
    # x-axis at ¾ from bottom: center_y = radius - ¾·(2·radius) = -radius/2
    new_cx      = 0.0
    new_cy      = -new_radius / 2   # = -1.0
    noise       = 0.07

    # --- reference coupling ---
    def _circle_coupling(n: int):
        theta = torch.rand(n) * 2 * math.pi
        x = torch.stack([
            ref_radius * torch.cos(theta),
            ref_radius * torch.sin(theta),
        ], dim=1)
        if noise > 0:
            x = x + noise * torch.randn_like(x)
        y = x.clone()
        y[:, 1] = x[:, 1] + shift_y * torch.sign(x[:, 1])
        return x.to(device), y.to(device)

    coupling = FunctionalCouplingDistribution(_circle_coupling)

    # --- new source ---
    new_src = _CircleDist(center=(new_cx, new_cy), radius=new_radius, noise=noise, device=device)

    # --- new target: apply same ±shift transport to a fresh draw from new_src ---
    class _TransportedCircleDist(Distribution):
        def __init__(self, base: _CircleDist, shift: float):
            self._base  = base
            self._shift = shift

        def sample(self, n: int) -> torch.Tensor:
            pts = self._base.sample(n)
            out = pts.clone()
            out[:, 1] = pts[:, 1] + self._shift * torch.sign(pts[:, 1])
            return out

    base_tgt = _TransportedCircleDist(
        _CircleDist(center=(new_cx, new_cy), radius=new_radius, noise=noise, device=device),
        shift_y,
    )

    if perturbed:
        _theta = math.radians(15.0)
        _c, _s = math.cos(_theta), math.sin(_theta)
        _R     = torch.tensor([[_c, -_s], [_s, _c]], dtype=torch.float32)

        class _RotatedDist(Distribution):
            def __init__(self, base: Distribution, R):
                self._base = base
                self._R    = R

            def sample(self, n: int) -> torch.Tensor:
                pts = self._base.sample(n)
                return pts @ self._R.to(pts.device, pts.dtype).T

        new_tgt = _RotatedDist(base_tgt, _R)
    else:
        new_tgt = base_tgt

    # T(x,y) = (x, y + 5·sign(y));  T⁻¹(x,z) = (x, z - 5·sign(z))  [valid on target support |z|≥5]
    def transport_map(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, 1] = x[:, 1] + shift_y * torch.sign(x[:, 1])
        return out

    def inverse_transport_map(y: torch.Tensor) -> torch.Tensor:
        out = y.clone()
        out[:, 1] = y[:, 1] - shift_y * torch.sign(y[:, 1])
        return out

    train_cfg                  = _default_train_cfg(device)
    train_cfg.iterations       = 100
    train_cfg.pretrain_epochs  = 6_000
    train_cfg.scale_factor     = 50
    method_cfg                 = _default_method_cfg()
    method_cfg.sigma           = 0.07
    method_cfg.sigma2          = 0.07
    method_cfg.particle_ratio  = 0.15
    method_cfg.lip_reg         = 1.0
    method_cfg.use_ot_pretrain = True
    method_cfg.dual_warmup_epochs = 300
    model_cfg  = _default_model_cfg(hidden_size=128, num_layers=5)
    model_cfg.use_siren = True
    vis_cfg    = VisualizeConfig(
        show_each=1, n_trajectories=2000, trajectory_stride=2,
        show_each_marginal=100, show_each_pretrain=1000,
    )

    return ExperimentSpec(
        name                  = f"2d_circle{'_perturbed' if perturbed else ''}",
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
# radial_warp – cubic radial expansion  T(x) = x · (1 + 0.5‖x‖²)
# ---------------------------------------------------------------------------

def _radial_warp(perturbed: bool, output_root: str) -> ExperimentSpec:
    """Cubic radial expansion generalised from a Gaussian to a Checkerboard.

    Reference coupling
        x ~ N(0, 0.8²·I₂),  y = T(x) = x·(1 + 0.5‖x‖²)   [radially expanding]
    New marginals
        μ': 4×4 checkerboard over [-1.5, 1.5]²
        ν': T_# μ'  —  outer cells are stretched more than inner cells
    Perturbed
        ν' is additionally rotated 10° to create distribution mismatch.
    """
    device = get_device()

    def _T(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True)
        return x * (1.0 + 0.5 * r2)

    _ref_src = Simple2DGaussianDistribution(
        mean=torch.tensor([0.0, 0.0], device=device),
        std=torch.tensor([0.8, 0.8], device=device),
        device=device,
    )

    def _coupling_fn(n: int):
        x = _ref_src.sample(n)
        return x, _T(x)

    coupling = FunctionalCouplingDistribution(_coupling_fn)

    new_src = CheckerboardDistribution(k=4, scale=1.5, noise=0.02, device=device)
    new_tgt_base = TransformedDistribution(
        CheckerboardDistribution(k=4, scale=1.5, noise=0.02, device=device), _T
    )

    if perturbed:
        _theta = math.radians(10.0)
        _c, _s = math.cos(_theta), math.sin(_theta)
        _R = torch.tensor([[_c, -_s], [_s, _c]], dtype=torch.float32)
        new_tgt = TransformedDistribution(
            new_tgt_base, lambda x, R=_R: x @ R.to(x.device, x.dtype).T
        )
    else:
        new_tgt = new_tgt_base

    train_cfg              = _default_train_cfg(device)
    train_cfg.iterations   = 100
    train_cfg.pretrain_epochs = 8_000
    train_cfg.scale_factor = 30
    method_cfg             = _default_method_cfg()
    method_cfg.sigma       = 0.05
    method_cfg.sigma2      = 0.05
    method_cfg.particle_ratio = 0.15
    method_cfg.lip_reg     = 1.0
    method_cfg.dual_warmup_epochs = 50
    method_cfg.use_ot_pretrain = False
    model_cfg = _default_model_cfg(hidden_size=128, num_layers=3)
    model_cfg.use_siren = True  # cubic radial expansion has high-frequency content
    vis_cfg   = VisualizeConfig(show_each=1, n_trajectories=2000, trajectory_stride=2,
                                show_each_marginal=100, show_each_pretrain=1000)

    return ExperimentSpec(
        name                  = f"2d_radial_warp{'_perturbed' if perturbed else ''}",
        mode                  = "2d",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = _T,
        inverse_transport_map = None,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


# ---------------------------------------------------------------------------
# cubic_shear – polynomial shear  T(x) = (x₁,  x₂ + 0.3 x₁³ + 0.2 x₁)
# ---------------------------------------------------------------------------

class _TwoMoonsDist(Distribution):
    """Both arcs of the two-moons dataset, equally mixed."""

    def __init__(self, scale: float = 1.0, shift=(0.0, 0.0), noise: float = 0.07, device=None):
        self._dev   = device or get_device()
        self._scale = scale
        self._shift = torch.tensor(shift, dtype=torch.float32)
        self._noise = noise

    def sample(self, n: int) -> torch.Tensor:
        n0 = n // 2
        n1 = n - n0
        pts0 = _moon_points(0, torch.rand(n0) * math.pi)
        pts1 = _moon_points(1, torch.rand(n1) * math.pi)
        pts  = torch.cat([pts0, pts1], dim=0) * self._scale + self._shift
        if self._noise > 0:
            pts = pts + self._noise * torch.randn_like(pts)
        return pts[torch.randperm(n)].to(self._dev)


def _cubic_shear(perturbed: bool, output_root: str) -> ExperimentSpec:
    """Cubic shear generalised from a vertical strip to the two-moons.

    Reference coupling
        x ~ Uniform([-1.5, 1.5] × [-3, 3]),
        y = T(x) = (x₁,  x₂ + 0.3 x₁³ + 0.2 x₁)   [vertical lines → S-curves]
    New marginals
        μ': two-moons at scale 1.5
        ν': T_# μ'  —  the moons are bent into S-shaped arcs
    Perturbed
        ν' additionally rotated 10° to test distribution mismatch.
    """
    device = get_device()
    _strip = UniformRectDistribution(-1.5, 1.5, -3.0, 3.0, device=device)

    def _T(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, 1] = x[:, 1] + 0.3 * x[:, 0] ** 3 + 0.2 * x[:, 0]
        return out

    def _T_inv(y: torch.Tensor) -> torch.Tensor:
        out = y.clone()
        out[:, 1] = y[:, 1] - 0.3 * y[:, 0] ** 3 - 0.2 * y[:, 0]
        return out

    def _coupling_fn(n: int):
        x = _strip.sample(n)
        return x, _T(x)

    coupling = FunctionalCouplingDistribution(_coupling_fn)

    new_src = _TwoMoonsDist(scale=1.5, noise=0.07, device=device)
    new_tgt_base = TransformedDistribution(
        _TwoMoonsDist(scale=1.5, noise=0.07, device=device), _T
    )

    if perturbed:
        _theta = math.radians(10.0)
        _c, _s = math.cos(_theta), math.sin(_theta)
        _R = torch.tensor([[_c, -_s], [_s, _c]], dtype=torch.float32)
        new_tgt = TransformedDistribution(
            new_tgt_base, lambda x, R=_R: x @ R.to(x.device, x.dtype).T
        )
    else:
        new_tgt = new_tgt_base

    train_cfg              = _default_train_cfg(device)
    train_cfg.iterations   = 100
    train_cfg.pretrain_epochs = 8_000
    train_cfg.scale_factor = 20
    method_cfg             = _default_method_cfg()
    method_cfg.sigma       = 0.05
    method_cfg.sigma2      = 0.05
    method_cfg.particle_ratio = 0.2
    method_cfg.lip_reg     = 1.0
    method_cfg.dual_warmup_epochs = 30
    method_cfg.use_ot_pretrain = False
    model_cfg = _default_model_cfg(hidden_size=128, num_layers=3)
    model_cfg.use_siren = True  # polynomial shear bends low-freq inputs into high-freq S-curves
    vis_cfg   = VisualizeConfig(show_each=1, n_trajectories=2000, trajectory_stride=2,
                                show_each_marginal=100, show_each_pretrain=1000)

    return ExperimentSpec(
        name                  = f"2d_cubic_shear{'_perturbed' if perturbed else ''}",
        mode                  = "2d",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = _T,
        inverse_transport_map = _T_inv,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


# ---------------------------------------------------------------------------
# polar_twist – vortex map  T: r′ = r,  θ′ = θ + sin(r)
# ---------------------------------------------------------------------------

def _polar_twist(perturbed: bool, output_root: str) -> ExperimentSpec:
    """Polar vortex twist generalised from a thin rectangle to a cross.

    Reference coupling
        x ~ Uniform([1, 3] × [-0.4, 0.4])  (a thin strip offset from origin)
        y = T(x):  r unchanged,  θ → θ + sin(r)
    New marginals
        μ': cross  =  [-2, 2]×[-0.35, 0.35]  ∪  [-0.35, 0.35]×[-2, 2]
        ν': T_# μ'  —  the arms of the cross are individually swirled
    Perturbed
        ν' additionally rotated 10° to test distribution mismatch.
    """
    device = get_device()
    _rect = UniformRectDistribution(1.0, 3.0, -0.4, 0.4, device=device)

    def _T(x: torch.Tensor) -> torch.Tensor:
        r     = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
        theta = torch.atan2(x[:, 1:2], x[:, 0:1])
        theta_new = theta + torch.sin(r)
        return torch.cat([r * torch.cos(theta_new), r * torch.sin(theta_new)], dim=-1)

    def _T_inv(y: torch.Tensor) -> torch.Tensor:
        r     = torch.norm(y, dim=-1, keepdim=True).clamp(min=1e-6)
        theta = torch.atan2(y[:, 1:2], y[:, 0:1])
        theta_orig = theta - torch.sin(r)
        return torch.cat([r * torch.cos(theta_orig), r * torch.sin(theta_orig)], dim=-1)

    def _coupling_fn(n: int):
        x = _rect.sample(n)
        return x, _T(x)

    coupling = FunctionalCouplingDistribution(_coupling_fn)

    def _cross_dist(dev):
        return UnionDistribution([
            UniformRectDistribution(-2.0, 2.0, -0.35, 0.35, device=dev),
            UniformRectDistribution(-0.35, 0.35, -2.0, 2.0, device=dev),
        ])

    new_src      = _cross_dist(device)
    new_tgt_base = TransformedDistribution(_cross_dist(device), _T)

    if perturbed:
        _theta = math.radians(10.0)
        _c, _s = math.cos(_theta), math.sin(_theta)
        _R = torch.tensor([[_c, -_s], [_s, _c]], dtype=torch.float32)
        new_tgt = TransformedDistribution(
            new_tgt_base, lambda x, R=_R: x @ R.to(x.device, x.dtype).T
        )
    else:
        new_tgt = new_tgt_base

    train_cfg              = _default_train_cfg(device)
    train_cfg.iterations   = 100
    train_cfg.pretrain_epochs = 8_000
    train_cfg.scale_factor = 30
    method_cfg             = _default_method_cfg()
    method_cfg.sigma       = 0.05
    method_cfg.sigma2      = 0.05
    method_cfg.particle_ratio = 0.15
    method_cfg.lip_reg     = 1.0
    method_cfg.dual_warmup_epochs = 50
    method_cfg.use_ot_pretrain = False
    model_cfg = _default_model_cfg(hidden_size=128, num_layers=3)
    model_cfg.use_siren = True  # polar vortex θ' = θ + sin(r) has intrinsic sinusoidal structure
    vis_cfg   = VisualizeConfig(show_each=1, n_trajectories=2000, trajectory_stride=2,
                                show_each_marginal=100, show_each_pretrain=1000)

    return ExperimentSpec(
        name                  = f"2d_polar_twist{'_perturbed' if perturbed else ''}",
        mode                  = "2d",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = _T,
        inverse_transport_map = _T_inv,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


# ---------------------------------------------------------------------------
# Public catalogue
# ---------------------------------------------------------------------------

def make_2d_experiments(output_root: str = "results") -> List[ExperimentSpec]:
    """Return all 2-D experiment specs."""
    return [
        _simple(False,       output_root),
        _simple(True,        output_root),
        _medium(False,       output_root),
        _medium(True,        output_root),
        _complex(False,      output_root),
        _complex(True,       output_root),
        _moon(False,         output_root),
        _moon(True,          output_root),
        _circle(False,       output_root),
        _circle(True,        output_root),
        _radial_warp(False,  output_root),
        _radial_warp(True,   output_root),
        _cubic_shear(False,  output_root),
        _cubic_shear(True,   output_root),
        _polar_twist(False,  output_root),
        _polar_twist(True,   output_root),
        _far_away(False,       output_root),
        _far_away(True,        output_root),
    ]
