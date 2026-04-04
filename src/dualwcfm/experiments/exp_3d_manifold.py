"""dualwcfm.experiments.exp_3d_manifold
========================================
Manifold-structured 3-D toy experiments for EOT.

Four experiments with visually rich, non-GMM distributions:

A. Swiss Roll Transport
   Source  μ : Swiss Roll  (t ∈ [1.5π, 4.5π], height ∈ [0, 3])
   Target  ν : compressed (×0.6) + rotated (90°) Swiss Roll  — same embedding
   New     μ': Swiss Roll at a different radial/height range
   New     ν': T(μ') — closed-form via planar scale + rotation
   Transport : T(x,y,z) = 0.6 · R_z(90°) · (x,y) ∥ z

B. Interleaved Helices
   Source  μ : inner helix  (r=1, 2 turns, height 4)
   Target  ν : outer helix  (r=1.5, phase-shifted by π) at same z
   New     μ': inner helix at r=1.2, different z range
   New     ν': T(μ') — linear T = diag(-1.5, -1.5, 1)
   Transport : T(x,y,z) = (-1.5x, -1.5y, z)  (linear, closed-form)

C. Sphere with Latitude-Dependent Twist
   Source  μ : unit sphere (radius 1)
   Target  ν : twisted sphere — each point (x,y,z) rotated by α·z around z-axis
   New     μ': sphere of radius 1.4
   New     ν': T(μ') — same twist, sphere is preserved (only angles change)
   Transport : T(x,y,z) = (x·cos(α·z) − y·sin(α·z),  x·sin(α·z) + y·cos(α·z),  z)

D. Concentric Rings to Double Helix
   Source  μ : two concentric rings (r=1 and r=2) lying in the z=0 plane
   Target  ν : the two rings wound into concentric helices (z = H·θ/(2π))
   New     μ': rings at r=0.75 and r=1.5
   New     ν': T(μ') — same winding map, different radii
   Transport : T(x,y,0) = (x, y, H·atan2(y,x)/(2π))   (atan2 → z lift)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch

from dualwcfm.config import FeedForwardConfig, TrainingConfig, VisualizeConfig
from dualwcfm.distributions.coupling import CouplingDistribution
from dualwcfm.distributions.distribution import Distribution
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.experiments.base import ExperimentSpec, get_device
from dualwcfm.experiments.exp_3d import _default_train_cfg, _default_method_cfg


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _scale_rot_z(pts: torch.Tensor, scale: float, phi_rad: float) -> torch.Tensor:
    """Apply 2-D scale + rotation around the z-axis to (x, y), leave z unchanged."""
    cos_a, sin_a = math.cos(phi_rad), math.sin(phi_rad)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    tx = scale * (x * cos_a - y * sin_a)
    ty = scale * (x * sin_a + y * cos_a)
    return torch.stack([tx, ty, z], dim=1)


def _sphere_twist(pts: torch.Tensor, alpha: float) -> torch.Tensor:
    """Rotate each point (x, y, z) around the z-axis by alpha * z radians."""
    angle = alpha * pts[:, 2]
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return torch.stack([x * cos_a - y * sin_a, x * sin_a + y * cos_a, z], dim=1)


def _helix_lift(pts: torch.Tensor, height: float) -> torch.Tensor:
    """Lift each ring point (x, y, 0) to a helix: z = H * atan2(y, x) / (2π)."""
    theta = torch.atan2(pts[:, 1], pts[:, 0])   # ∈ (-π, π]
    tz = height * theta / (2.0 * math.pi)        # z ∈ (-H/2, H/2]
    return torch.stack([pts[:, 0], pts[:, 1], tz], dim=1)


# ---------------------------------------------------------------------------
# A: Swiss Roll distributions
# ---------------------------------------------------------------------------

class SwissRollCoupling(CouplingDistribution):
    """Paired Swiss Roll samples encoding a planar scale + rotation transport.

    Source x  ~ Swiss Roll(t_range, h_range) + noise
    Target y  ~ T(x_clean) + noise   where T = scale · R_z(phi)
    """

    def __init__(
        self,
        t_range: Tuple[float, float] = (1.5 * math.pi, 4.5 * math.pi),
        h_range: Tuple[float, float] = (0.0, 3.0),
        scale: float = 0.6,
        phi_deg: float = 90.0,
        noise: float = 0.0,
        device: torch.device = None,
    ):
        self._dev   = device or get_device()
        self.t_lo, self.t_hi = t_range
        self.h_lo, self.h_hi = h_range
        self.scale  = scale
        self.phi    = math.radians(phi_deg)
        self.noise  = noise

    def _sample_clean(self, n: int) -> torch.Tensor:
        t = torch.rand(n, device=self._dev) * (self.t_hi - self.t_lo) + self.t_lo
        h = torch.rand(n, device=self._dev) * (self.h_hi - self.h_lo) + self.h_lo
        # Normalize radius: r ∝ t so the roll expands outward
        r = t / self.t_hi
        x = r * torch.cos(t)
        y = r * torch.sin(t)
        return torch.stack([x, y, h], dim=1)

    def coupling_sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_clean = self._sample_clean(n)
        tgt_clean = _scale_rot_z(src_clean, self.scale, self.phi)
        src = src_clean + torch.randn_like(src_clean) * self.noise
        tgt = tgt_clean + torch.randn_like(tgt_clean) * self.noise
        return src, tgt

    def sample_init(self, n: int) -> torch.Tensor:
        return self.coupling_sample(n)[0]

    def sample_target(self, n: int) -> torch.Tensor:
        return self.coupling_sample(n)[1]

    def sample(self, n: int) -> torch.Tensor:
        x, y = self.coupling_sample(n)
        return torch.stack([x, y], dim=0)


class SwissRollDistribution(Distribution):
    """Samples from a Swiss Roll in R³."""

    def __init__(
        self,
        t_range: Tuple[float, float] = (1.5 * math.pi, 4.5 * math.pi),
        h_range: Tuple[float, float] = (0.0, 3.0),
        noise: float = 0.0,
        device: torch.device = None,
    ):
        self._dev   = device or get_device()
        self.t_lo, self.t_hi = t_range
        self.h_lo, self.h_hi = h_range
        self.noise  = noise

    def sample(self, n: int) -> torch.Tensor:
        t = torch.rand(n, device=self._dev) * (self.t_hi - self.t_lo) + self.t_lo
        h = torch.rand(n, device=self._dev) * (self.h_hi - self.h_lo) + self.h_lo
        r = t / self.t_hi
        x = r * torch.cos(t)
        y = r * torch.sin(t)
        pts = torch.stack([x, y, h], dim=1)
        return pts + torch.randn_like(pts) * self.noise


# ---------------------------------------------------------------------------
# B: Helix distributions
# ---------------------------------------------------------------------------

class HelixCoupling(CouplingDistribution):
    """Paired samples from two interleaved helices encoding a linear transport.

    Source: inner helix  (radius r_src, n_turns turns, total height H)
    Target: outer helix  (radius r_tgt, phase shifted by π)

    Transport map T(x, y, z) = (-r_tgt/r_src · x,  -r_tgt/r_src · y,  z)
    which maps the inner helix to the outer, phase-shifted helix.
    """

    def __init__(
        self,
        r_src: float = 1.0,
        r_tgt: float = 1.5,
        n_turns: float = 2.0,
        height: float = 4.0,
        noise: float = 0.0,
        device: torch.device = None,
    ):
        self._dev   = device or get_device()
        self.r_src  = r_src
        self.r_tgt  = r_tgt
        self.n_turns = n_turns
        self.height  = height
        self.noise   = noise

    def _sample_clean(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = torch.rand(n, device=self._dev) * 2.0 * math.pi * self.n_turns
        z     = theta / (2.0 * math.pi * self.n_turns) * self.height
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        src = torch.stack([self.r_src * cos_t, self.r_src * sin_t, z], dim=1)
        # outer helix: phase-shift by π → multiply cos/sin by -1
        tgt = torch.stack([-self.r_tgt * cos_t, -self.r_tgt * sin_t, z], dim=1)
        return src, tgt

    def coupling_sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self._sample_clean(n)
        src = src + torch.randn_like(src) * self.noise
        tgt = tgt + torch.randn_like(tgt) * self.noise
        return src, tgt

    def sample_init(self, n: int) -> torch.Tensor:
        return self.coupling_sample(n)[0]

    def sample_target(self, n: int) -> torch.Tensor:
        return self.coupling_sample(n)[1]

    def sample(self, n: int) -> torch.Tensor:
        x, y = self.coupling_sample(n)
        return torch.stack([x, y], dim=0)


class HelixDistribution(Distribution):
    """Samples from a single helix tube in R³."""

    def __init__(
        self,
        radius: float = 1.0,
        n_turns: float = 2.0,
        height: float = 4.0,
        z_offset: float = 0.0,
        noise: float = 0.0,
        device: torch.device = None,
    ):
        self._dev   = device or get_device()
        self.radius  = radius
        self.n_turns = n_turns
        self.height  = height
        self.z_offset = z_offset
        self.noise   = noise

    def sample(self, n: int) -> torch.Tensor:
        theta = torch.rand(n, device=self._dev) * 2.0 * math.pi * self.n_turns
        z     = theta / (2.0 * math.pi * self.n_turns) * self.height + self.z_offset
        pts   = torch.stack([self.radius * torch.cos(theta),
                              self.radius * torch.sin(theta), z], dim=1)
        return pts + torch.randn_like(pts) * self.noise


# ---------------------------------------------------------------------------
# C: Sphere distributions
# ---------------------------------------------------------------------------

class SphereCoupling(CouplingDistribution):
    """Paired samples from a sphere encoding a latitude-dependent twist transport.

    Source: uniform on sphere of radius r
    Target: T(source) where T(x,y,z) = R_z(alpha * z) · (x, y, z)

    alpha = π/2 means the north pole (z=1) rotates by 90°,
    the equator (z=0) stays fixed, and the south pole rotates by -90°.
    The sphere is mapped to itself — only angular positions change.
    """

    def __init__(
        self,
        radius: float = 1.0,
        alpha: float = math.pi / 2.0,
        noise: float = 0.0,
        device: torch.device = None,
    ):
        self._dev  = device or get_device()
        self.radius = radius
        self.alpha  = alpha
        self.noise  = noise

    def _sample_sphere(self, n: int) -> torch.Tensor:
        pts = torch.randn(n, 3, device=self._dev)
        pts = pts / pts.norm(dim=1, keepdim=True) * self.radius
        return pts

    def coupling_sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_clean = self._sample_sphere(n)
        tgt_clean = _sphere_twist(src_clean, self.alpha)
        src = src_clean + torch.randn_like(src_clean) * self.noise
        tgt = tgt_clean + torch.randn_like(tgt_clean) * self.noise
        return src, tgt

    def sample_init(self, n: int) -> torch.Tensor:
        return self.coupling_sample(n)[0]

    def sample_target(self, n: int) -> torch.Tensor:
        return self.coupling_sample(n)[1]

    def sample(self, n: int) -> torch.Tensor:
        x, y = self.coupling_sample(n)
        return torch.stack([x, y], dim=0)


class SphereDistribution(Distribution):
    """Uniform distribution on a sphere of the given radius."""

    def __init__(
        self,
        radius: float = 1.0,
        noise: float = 0.0,
        device: torch.device = None,
    ):
        self._dev   = device or get_device()
        self.radius = radius
        self.noise  = noise

    def sample(self, n: int) -> torch.Tensor:
        pts = torch.randn(n, 3, device=self._dev)
        pts = pts / pts.norm(dim=1, keepdim=True) * self.radius
        return pts + torch.randn_like(pts) * self.noise


# ---------------------------------------------------------------------------
# D: Concentric rings → double helix (GMM-based)
# ---------------------------------------------------------------------------

def _ring_gmm_centers(
    radii: List[float],
    n_blobs_per_ring: int = 20,
    z: float = 0.0,
) -> List[List[float]]:
    """Return GMM centers arranged on concentric rings in the z=*z* plane."""
    centers = []
    for r in radii:
        for k in range(n_blobs_per_ring):
            theta = 2.0 * math.pi * k / n_blobs_per_ring
            centers.append([r * math.cos(theta), r * math.sin(theta), z])
    return centers


def _ring_to_helix_centers(
    radii: List[float],
    n_blobs_per_ring: int = 20,
    height: float = 3.0,
    z_offset: float = 0.0,
) -> List[List[float]]:
    """Return helix GMM centers — one blob per ring point, lifted to z = H·θ/(2π)."""
    centers = []
    for r in radii:
        for k in range(n_blobs_per_ring):
            theta = 2.0 * math.pi * k / n_blobs_per_ring
            z = height * theta / (2.0 * math.pi) + z_offset
            centers.append([r * math.cos(theta), r * math.sin(theta), z])
    return centers


# ---------------------------------------------------------------------------
# Experiment factories
# ---------------------------------------------------------------------------

def _swiss_roll(output_root: str) -> ExperimentSpec:
    """Swiss Roll transport: scale × rotation in the xy-plane.

    Reference coupling
        Swiss Roll  t ∈ [1.5π, 4.5π],  height ∈ [0, 3]
        Transport  T = 3.0 · R_z(90°)  on xy, z unchanged.
    New marginals
        Wider roll  t ∈ [π, 5π],  height ∈ [0.5, 3.5]  (generalises radial range)
    """
    device = get_device()
    t_ref  = (1.5 * math.pi, 4.5 * math.pi)
    t_new  = (4.5 * math.pi,       10.0 * math.pi)
    h_ref  = (0.0, 3.0)
    h_new  = (3.0, 6.0)
    scale, phi_deg = 3.0, 90.0

    coupling = SwissRollCoupling(t_ref, h_ref, scale=scale, phi_deg=phi_deg,
                                 noise=0.0, device=device)
    new_src  = SwissRollDistribution(t_new, h_new, noise=0.0, device=device)

    phi_rad  = math.radians(phi_deg)

    def transport_map(x: torch.Tensor) -> torch.Tensor:
        return _scale_rot_z(x, scale, phi_rad)

    def inverse_transport_map(y: torch.Tensor) -> torch.Tensor:
        return _scale_rot_z(y, 1.0 / scale, -phi_rad)

    # new_tgt samples from T(new_src)
    class _TransportedDist(Distribution):
        def __init__(self, src_dist, T):
            self._src = src_dist
            self._T   = T
        def sample(self, n: int) -> torch.Tensor:
            return self._T(self._src.sample(n))

    new_tgt = _TransportedDist(new_src, transport_map)

    train_cfg  = _default_train_cfg(device)
    train_cfg.marginal_epochs    = 25_000          # continuous manifold needs more bridge training than GMM
    train_cfg.num_rectified_flow = 2               # extra rectification for curved roll geometry
    train_cfg.pretrain_epochs    = 8_000
    train_cfg.dual_epochs        = 150
    train_cfg.advance_steps      = 150             # more ODE steps for curved manifold integration
    method_cfg = _default_method_cfg()
    method_cfg.sigma             = 0.05            # σ=0.1 over-spreads the roll structure; 0.05 keeps paths on manifold
    method_cfg.lip_reg           = 5               # roll transport is smooth; less Lipschitz regularisation needed
    method_cfg.dual_warmup_epochs = 50
    model_cfg  = FeedForwardConfig(dimension=3, hidden_size=128, num_layers=3, flow_learning_rate=1e-4, nu_learning_rate=1e-4, mu_learning_rate=1e-4, dual_learning_rate=1e-4)
    vis_cfg    = VisualizeConfig(
        show_each=1, n_trajectories=600, trajectory_stride=2,
        show_each_marginal=1000, show_each_pretrain=1000,
        camera_elev=38, camera_azim=30,   # top-ish view to see roll structure
    )

    return ExperimentSpec(
        name                  = "3d_swiss_roll",
        mode                  = "3d",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        transport_map         = transport_map,
        inverse_transport_map = inverse_transport_map,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


def _helices(output_root: str) -> ExperimentSpec:
    """Interleaved helices: linear transport T = diag(-1.5, -1.5, 1).

    Reference coupling
        Inner helix  r=1,  2 turns,  height 4
        Outer helix  r=1.5, phase-shifted by π  (same z)
    New marginals
        Inner helix  r=1.2,  different z range  (height offset 0.5)
    """
    device   = get_device()
    r_src, r_tgt = 1.0, 1.5
    ratio    = r_tgt / r_src   # = 1.5

    coupling = HelixCoupling(r_src=r_src, r_tgt=r_tgt, n_turns=2, height=4,
                              noise=0.0, device=device)
    new_src  = HelixDistribution(radius=1.2, n_turns=2, height=4,
                                 z_offset=0.5, noise=0.0, device=device)

    def transport_map(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([-ratio * x[:, 0], -ratio * x[:, 1], x[:, 2]], dim=1)

    def inverse_transport_map(y: torch.Tensor) -> torch.Tensor:
        return torch.stack([-y[:, 0] / ratio, -y[:, 1] / ratio, y[:, 2]], dim=1)

    class _TransportedDist(Distribution):
        def __init__(self, src_dist, T):
            self._src = src_dist
            self._T   = T
        def sample(self, n: int) -> torch.Tensor:
            return self._T(self._src.sample(n))

    new_tgt = _TransportedDist(new_src, transport_map)

    train_cfg  = _default_train_cfg(device)
    train_cfg.num_rectified_flow = 2               # helical geometry benefits from trajectory straightening
    method_cfg = _default_method_cfg()
    method_cfg.sigma             = 0.05            # helix is a 1D structure; σ=0.1 blurs the tube unnecessarily
    model_cfg  = FeedForwardConfig(dimension=3, hidden_size=64, num_layers=3, flow_learning_rate=1e-4, nu_learning_rate=1e-4, mu_learning_rate=1e-4, dual_learning_rate=1e-4)
    vis_cfg    = VisualizeConfig(
        show_each=1, n_trajectories=600, trajectory_stride=2,
        show_each_marginal=1000, show_each_pretrain=1000,
        camera_elev=18, camera_azim=30,   # side view to appreciate helical structure (linear T, 64 sufficient)
    )

    return ExperimentSpec(
        name                  = "3d_helices",
        mode                  = "3d",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        transport_map         = transport_map,
        inverse_transport_map = inverse_transport_map,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


def _sphere_twist_exp(output_root: str) -> ExperimentSpec:
    """Sphere with latitude-dependent twist.

    Reference coupling
        Unit sphere (r=1)  →  T(sphere, α=π/2)
    New marginals
        Sphere r=1.4  →  T(sphere at r=1.4)  — T is scale-invariant in z,
        so the twist angle pattern is unchanged.
    """
    device = get_device()
    alpha  = math.pi / 2.0   # 90° at north pole

    coupling = SphereCoupling(radius=1.0, alpha=alpha, noise=0.0, device=device)
    new_src  = SphereDistribution(radius=1.4, noise=0.0, device=device)

    def transport_map(x: torch.Tensor) -> torch.Tensor:
        return _sphere_twist(x, alpha)

    def inverse_transport_map(y: torch.Tensor) -> torch.Tensor:
        return _sphere_twist(y, -alpha)

    class _TransportedDist(Distribution):
        def __init__(self, src_dist, T):
            self._src = src_dist
            self._T   = T
        def sample(self, n: int) -> torch.Tensor:
            return self._T(self._src.sample(n))

    new_tgt = _TransportedDist(new_src, transport_map)

    train_cfg  = _default_train_cfg(device)
    train_cfg.marginal_epochs    = 30_000          # latitude-dependent twist is the hardest bridge learning task
    train_cfg.num_rectified_flow = 3               # nonlinear twist requires the most trajectory straightening
    train_cfg.pretrain_epochs    = 10_000
    train_cfg.dual_epochs        = 200
    train_cfg.cfm_epochs         = 750
    train_cfg.iterations         = 100
    train_cfg.advance_steps      = 150
    method_cfg = _default_method_cfg()
    method_cfg.sigma             = 0.05            # sphere radius=1; σ=0.1 leaks outside the shell
    method_cfg.lip_reg           = 20              # nonlinear twist → sharper dual gradients → more Lipschitz control
    method_cfg.particle_ratio    = 0.35
    method_cfg.lr_min_factor     = 0.3
    method_cfg.dual_warmup_epochs = 100            # largest warmup for hardest dual problem
    model_cfg  = FeedForwardConfig(dimension=3, hidden_size=256, num_layers=4, flow_learning_rate=1e-4, nu_learning_rate=1e-4, mu_learning_rate=1e-4, dual_learning_rate=1e-4)
    vis_cfg    = VisualizeConfig(
        show_each=1, n_trajectories=600, trajectory_stride=2,
        show_each_marginal=1000, show_each_pretrain=1000,
        camera_elev=28, camera_azim=20,
    )

    return ExperimentSpec(
        name                  = "3d_sphere_twist",
        mode                  = "3d",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        transport_map         = transport_map,
        inverse_transport_map = inverse_transport_map,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


def _rings_to_helix(output_root: str) -> ExperimentSpec:
    """Concentric rings wound into a double helix.

    Reference coupling
        Two rings  r ∈ {1, 2}  in the z=0 plane  (20 blobs each)
        Transport  T(x,y,0) = (x, y, H·atan2(y,x)/(2π)),  H = 3
    New marginals
        Two smaller rings  r ∈ {0.75, 1.5}  in the z=0 plane
        Same T — atan2 is independent of radius, so z-lift is the same
    """
    device       = get_device()
    N_BLOBS      = 20
    H            = 3.0
    REF_RADII    = [1.0, 2.0]
    NEW_RADII    = [0.75, 1.5]
    BLOB_SCALE   = 0.07

    ref_src = _ring_gmm_centers(REF_RADII, N_BLOBS, z=0.0)
    ref_tgt = _ring_to_helix_centers(REF_RADII, N_BLOBS, height=H, z_offset=0.0)

    from dualwcfm.experiments.exp_3d import GMM3DCoupling, GMM3DDistribution

    coupling = GMM3DCoupling(ref_src, ref_tgt, blob_scale=BLOB_SCALE, device=device)

    new_src_centers = _ring_gmm_centers(NEW_RADII, N_BLOBS, z=0.0)
    new_tgt_centers = _ring_to_helix_centers(NEW_RADII, N_BLOBS, height=H, z_offset=0.0)
    new_src = GMM3DDistribution(new_src_centers, blob_scale=BLOB_SCALE, device=device)
    new_tgt = GMM3DDistribution(new_tgt_centers, blob_scale=BLOB_SCALE, device=device)

    _H = H

    def transport_map(x: torch.Tensor) -> torch.Tensor:
        theta = torch.atan2(x[:, 1], x[:, 0])
        tz    = _H * theta / (2.0 * math.pi)
        return torch.stack([x[:, 0], x[:, 1], tz], dim=1)

    def inverse_transport_map(y: torch.Tensor) -> torch.Tensor:
        return torch.stack([y[:, 0], y[:, 1], torch.zeros_like(y[:, 2])], dim=1)

    train_cfg  = _default_train_cfg(device)
    train_cfg.marginal_epochs    = 25_000          # 40-blob GMM with tiny blob_scale=0.07 needs long bridge training
    train_cfg.num_rectified_flow = 2
    train_cfg.pretrain_epochs    = 8_000
    train_cfg.dual_epochs        = 150
    train_cfg.cfm_epochs         = 750             # z-lift transport is nonlinear → more CFM steps
    train_cfg.iterations         = 100
    train_cfg.advance_steps      = 150
    method_cfg = _default_method_cfg()
    method_cfg.sigma             = 0.03            # BLOB_SCALE=0.07 → σ must stay ≤ 0.07; 0.03 is safe
    method_cfg.lip_reg           = 15
    method_cfg.particle_ratio    = 0.35
    method_cfg.lr_min_factor     = 0.3
    method_cfg.dual_warmup_epochs = 50
    model_cfg  = FeedForwardConfig(dimension=3, hidden_size=256, num_layers=4, flow_learning_rate=1e-4, nu_learning_rate=1e-4, mu_learning_rate=1e-4, dual_learning_rate=1e-4)
    vis_cfg    = VisualizeConfig(
        show_each=1, n_trajectories=600, trajectory_stride=2,
        show_each_marginal=1000, show_each_pretrain=1000,
        camera_elev=25, camera_azim=15,   # see both z-lift and ring structure
    )

    return ExperimentSpec(
        name                  = "3d_rings_helix",
        mode                  = "3d",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        transport_map         = transport_map,
        inverse_transport_map = inverse_transport_map,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


# ---------------------------------------------------------------------------
# Public catalogue
# ---------------------------------------------------------------------------

def make_3d_manifold_experiments(output_root: str = "results") -> List[ExperimentSpec]:
    """Return all 4 manifold-structured 3-D experiment specs."""
    return [
        _swiss_roll(output_root),
        _helices(output_root),
        _sphere_twist_exp(output_root),
        _rings_to_helix(output_root),
    ]
