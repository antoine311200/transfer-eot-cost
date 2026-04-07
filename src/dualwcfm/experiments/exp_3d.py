"""dualwcfm.experiments.exp_3d
===============================
3-D toy-distribution experiment catalogue.

Design principle
----------------
Same as the 2-D catalogue: the reference coupling π encodes a z-axis rotation
by θ° applied to Gaussian blobs at positions S₀.  The new source μ' and
new target ν' live at *different* 3-D positions S₁ and follow the **same**
rotation dynamic.

Critical design rule
--------------------
The reference coupling MUST have the **same number of modes** as the new
distributions.  An unconditional flow matching bridge can only learn a
deterministic velocity field v(t, y_t); with N_ref < N_new it would need
to "split" one source blob into multiple target blobs — impossible for a
deterministic map.  OT mini-batch pairing then provides clean 1-to-1 mode
assignments.

Experiments
-----------
simple  – 90° z-rotation of a 3-blob GMM → generalised to a 3-blob GMM
medium  – 120° z-rotation of a 4-blob GMM → 4-blob rectangle
complex – 60° z-rotation of a 6-blob hexagonal GMM → 6-blob hexagonal pattern

Each experiment exists in a *standard* and *perturbed* variant.  The perturbed
variant adds an extra 30° tilt around the x-axis on the new target, probing
robustness to out-of-plane displacement.
"""

from __future__ import annotations

import math
from typing import List

import torch

from dualwcfm.config import FeedForwardConfig, TrainingConfig, VisualizeConfig
from dualwcfm.distributions.coupling import CouplingDistribution
from dualwcfm.distributions.distribution import Distribution
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.experiments.base import ExperimentSpec, get_device


# ---------------------------------------------------------------------------
# 3-D geometry helpers
# ---------------------------------------------------------------------------

def _rot_z(centers: List[List[float]], degrees: float) -> List[List[float]]:
    """Rotate a list of 3-D centres by *degrees* around the z-axis."""
    rad = math.radians(degrees)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return [
        [cos_a * x - sin_a * y, sin_a * x + cos_a * y, z]
        for x, y, z in centers
    ]


def _rot_x(centers: List[List[float]], degrees: float) -> List[List[float]]:
    """Rotate a list of 3-D centres by *degrees* around the x-axis."""
    rad = math.radians(degrees)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return [
        [x, cos_a * y - sin_a * z, sin_a * y + cos_a * z]
        for x, y, z in centers
    ]


# ---------------------------------------------------------------------------
# 3-D rotation matrix helpers
# ---------------------------------------------------------------------------

def _Rz(degrees: float) -> torch.Tensor:
    """Rotation matrix around the z-axis."""
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _Rx(degrees: float) -> torch.Tensor:
    """Rotation matrix around the x-axis."""
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float32)


def _make_transport_maps(rot_z_deg: float, perturbed: bool):
    """Return (transport_map, inverse_transport_map) for a 3-D rotation experiment.

    Non-perturbed: T(x) = R_z(θ) · x
    Perturbed:     T(x) = R_x(30°) · R_z(θ) · x   (x-tilt applied after z-rotation)

    For row-vector convention (used throughout this codebase):
        T(x)      = x  @ M.T
        T⁻¹(y)   = y  @ M          (since M is orthogonal: M⁻¹ = M.T)
    """
    Rz = _Rz(rot_z_deg)
    if perturbed:
        Rx  = _Rx(30.0)
        M   = Rx @ Rz       # compose: apply Rz first, then Rx
    else:
        M   = Rz

    def transport_map(x: torch.Tensor) -> torch.Tensor:
        M_local = M.to(x.device)
        return x @ M_local.T

    def inverse_transport_map(y: torch.Tensor) -> torch.Tensor:
        M_local = M.to(y.device)
        return y @ M_local        # M.T is the inverse; (M.T).T = M

    return transport_map, inverse_transport_map


# ---------------------------------------------------------------------------
# 3-D distributions
# ---------------------------------------------------------------------------

class GMM3DCoupling(CouplingDistribution):
    """Paired Gaussian blobs in R³ encoding a known point-to-point transport.

    Each mode k maps src_centers[k] → tgt_centers[k] with isotropic Gaussian
    noise of standard deviation *blob_scale*.

    Parameters
    ----------
    src_centers, tgt_centers:
        Lists of 3-D centres of equal length.  Each (src, tgt) pair is one
        transport mode.
    blob_scale:
        Standard deviation of the Gaussian blobs (same for source and target).
    """

    def __init__(
        self,
        src_centers: List[List[float]],
        tgt_centers: List[List[float]],
        blob_scale: float = 0.2,
        device: torch.device = None,
    ):
        self._dev   = device or get_device()
        self.src    = torch.tensor(src_centers, dtype=torch.float32, device=self._dev)
        self.tgt    = torch.tensor(tgt_centers, dtype=torch.float32, device=self._dev)
        self.scale  = blob_scale
        assert self.src.shape == self.tgt.shape, "src and tgt must have the same number of blobs"
        self.K      = self.src.shape[0]

    def coupling_sample(self, n: int):
        idx   = torch.randint(self.K, (n,), device=self._dev)
        noise = torch.randn(n, 3, device=self._dev) * self.scale
        x     = self.src[idx] + noise
        y     = self.tgt[idx] + noise   # same noise → velocity y-x = tgt[k]-src[k] exactly
        return x, y

    def sample_init(self, n: int) -> torch.Tensor:
        return self.coupling_sample(n)[0]

    def sample_target(self, n: int) -> torch.Tensor:
        return self.coupling_sample(n)[1]

    def sample(self, n: int) -> torch.Tensor:
        x, y = self.coupling_sample(n)
        return torch.stack([x, y], dim=0)


class GMM3DDistribution(Distribution):
    """Isotropic Gaussian mixture in R³.

    Used for the new source μ' and new target ν' in 3-D experiments.

    Parameters
    ----------
    centers:
        List of 3-D blob centres.
    blob_scale:
        Isotropic standard deviation shared by all blobs.
    """

    def __init__(
        self,
        centers: List[List[float]],
        blob_scale: float = 0.25,
        device: torch.device = None,
    ):
        self._dev    = device or get_device()
        self.centers = torch.tensor(centers, dtype=torch.float32, device=self._dev)
        self.scale   = blob_scale
        self.K       = self.centers.shape[0]

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(self.K, (n,), device=self._dev)
        return self.centers[idx] + torch.randn(n, 3, device=self._dev) * self.scale


# ---------------------------------------------------------------------------
# Default hyperparameter configs
# ---------------------------------------------------------------------------

def _default_train_cfg(device: torch.device) -> TrainingConfig:
    return TrainingConfig(
        device                   = device,
        dataset_size             = 500_000,
        marginal_epochs          = 2_000,
        pretrain_epochs          = 5_000,
        dual_epochs              = 100,
        cfm_epochs               = 500,
        advance_steps            = 100,
        iterations               = 50,
        num_rectified_flow       = 0,
        scale_factor             = 50,
        batch_size               = 2_048 * 8,
        save_coupling_checkpoints= False,
        save_model_checkpoints   = False,
    )


def _default_method_cfg() -> MixtureConfig:
    return MixtureConfig(
        sigma           = 0.1,
        sigma2          = 0.1,
        loss            = "weighted",
        l2_reg          = 1e-2,
        lip_reg         = 1.0,
        mean_reg        = 0,
        use_noisy_field    = True,
        particle_ratio     = 0.25,
        lr_cosine_decay    = False,
        lr_min_factor      = 0.4,   # 0.1 was too aggressive: flow ran at 1e-4 every iteration
        dual_warmup_epochs = 0,
        use_ot_pretrain=True #(default): OT pairing gives each sample a consistent
        # (y0, y1) assignment so the conditional bridge model can learn per-sample
        # transport rather than the dataset mean velocity.
    )

def default_model_cfg(dimension: int = 3, hidden_size: int = 64, num_layers: int = 3) -> FeedForwardConfig:
    return FeedForwardConfig(
        dimension=dimension,
        hidden_size=hidden_size,
        num_layers=num_layers,
        flow_learning_rate=1e-3,
        nu_learning_rate=1e-3,
        mu_learning_rate=1e-3,
        dual_learning_rate=1e-3,
        dual_l2_reg=1e-3,
        # 3D reference couplings always have fewer modes than the new distributions
        # (2→3, 3→4, 4→6 blobs across simple/medium/complex).  An unconditional
        # bridge with independent (y0,y1) sampling can only learn the mean velocity
        # E[y1-y0] — it cannot split blobs.  With use_conditional_bridge=True the
        # model receives cat([yt, y0]) so OT-assigned starting points determine
        # which target blob each particle goes to.  The architecture fix in base.py
        # ensures bridge MLPs are built with bridge_dim=2*dim input from the start.
        use_conditional_bridge=True,
    )


# ---------------------------------------------------------------------------
# simple – 90° z-rotation, 3 blobs → 3 blobs
# ---------------------------------------------------------------------------

def _simple(perturbed: bool, output_root: str) -> ExperimentSpec:
    """90° z-rotation on 3 source blobs generalised to 3 new blobs.

    Reference coupling
        3 blobs with slight y-offset: [[1.5,±0.5,0], [-1.5,0,0]].
        The y-offset breaks the x-axis symmetry so that R_z(90°) maps each
        ref_src blob to a *distinct* ref_tgt position, giving OT a clean
        1-to-1 assignment.
        ref_tgt = R_z(90°) → [[-0.5,1.5,0], [0.5,1.5,0], [0,-1.5,0]]
    New marginals
        3 blobs at different 3-D positions;  target = R_z(90°) of source
        (+ R_x(30°) extra tilt if perturbed).
    """
    device   = get_device()
    # y-offset ±1.0 matches new_src's ±1.0 offset → same blob separation ratio (8σ).
    # ±0.5 gave only 4σ separation, causing OT cross-assignment and mode drift over epochs.
    ref_src  = [[1.5, 1.0, 0.0], [1.5, -1.0, 0.0], [-1.5, 0.0, 0.0]]
    ref_tgt  = _rot_z(ref_src, 90.0)   # → [[-1.0,1.5,0],[1.0,1.5,0],[0,-1.5,0]]
    coupling = GMM3DCoupling(ref_src, ref_tgt, blob_scale=0.25, device=device)

    new_src_c = [[3.0, 1.0, 1.0], [3.0, -1.0, 1.0], [-3.0, 0.0, 1.0]]
    new_tgt_c = _rot_z(new_src_c, 90.0)
    if perturbed:
        new_tgt_c = _rot_x(new_tgt_c, 30.0)
    new_src = GMM3DDistribution(new_src_c, blob_scale=0.25, device=device)
    new_tgt = GMM3DDistribution(new_tgt_c, blob_scale=0.25, device=device)

    transport_map, inverse_transport_map = _make_transport_maps(90.0, perturbed)

    train_cfg  = _default_train_cfg(device)
    train_cfg.marginal_epochs = 2_000          # 2-blob reference is simple; 20k is overkill
    train_cfg.iterations     = 100              # complex needs more iterations than simple
    train_cfg.pretrain_epochs = 8_000           # larger flow pretraining for 5-blob cross with 6.7× scale jump
    train_cfg.dual_epochs    = 150
    method_cfg = _default_method_cfg()
    method_cfg.dual_warmup_epochs = 50         # warm-start duals for hard geometry
    model_cfg  = default_model_cfg(dimension=3, hidden_size=128, num_layers=4)   # 128 is oversized for 2→3 blobs
    vis_cfg    = VisualizeConfig(show_each=1, n_trajectories=800, trajectory_stride=2, show_each_marginal=100, show_each_pretrain=1000)

    return ExperimentSpec(
        name                  = f"3d_simple{'_perturbed' if perturbed else ''}",
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
# medium – 120° z-rotation, 4 blobs → 4 blobs
# ---------------------------------------------------------------------------

def _medium(perturbed: bool, output_root: str) -> ExperimentSpec:
    """120° z-rotation on a 4-blob rectangular GMM generalised to a 4-blob GMM.

    Reference coupling
        4 blobs at [±2, ±1, 0] (small rectangle), transport = R_z(120°).
        Rectangle aspect ratio 2:1 ensures all 4 blobs are distinct under
        R_z(120°), giving OT a clean 1-to-1 assignment (no symmetric degeneracy).
    New marginals
        4-blob rectangle at [±4, ±2, 1];  target = R_z(120°) of source
        (+ R_x(30°) if perturbed).
    """
    device = get_device()
    ref_src = [
        [ 2.0,  1.0, 0.0],
        [ 2.0, -1.0, 0.0],
        [-2.0,  1.0, 0.0],
        [-2.0, -1.0, 0.0],
    ]
    ref_tgt  = _rot_z(ref_src, 120.0)
    coupling = GMM3DCoupling(ref_src, ref_tgt, blob_scale=0.3, device=device)

    new_src_c = [[4.0, 2.0, 1.0], [4.0, -2.0, 1.0], [-4.0, 2.0, 1.0], [-4.0, -2.0, 1.0]]
    new_tgt_c = _rot_z(new_src_c, 120.0)
    if perturbed:
        new_tgt_c = _rot_x(new_tgt_c, 30.0)
    new_src = GMM3DDistribution(new_src_c, blob_scale=0.3, device=device)
    new_tgt = GMM3DDistribution(new_tgt_c, blob_scale=0.3, device=device)

    transport_map, inverse_transport_map = _make_transport_maps(120.0, perturbed)

    train_cfg  = _default_train_cfg(device)
    train_cfg.marginal_epochs    = 2_000       # 3→4 blob case benefits from more bridge training
    # train_cfg.num_rectified_flow = 1            # extra rectification pass improves trajectory straightness
    method_cfg = _default_method_cfg()
    method_cfg.dual_warmup_epochs = 50          # warm-start duals for 4-blob target
    model_cfg  = default_model_cfg(dimension=3, hidden_size=256, num_layers=4)
    vis_cfg    = VisualizeConfig(show_each=1, n_trajectories=800, trajectory_stride=2, show_each_marginal=100, show_each_pretrain=1000)

    return ExperimentSpec(
        name                  = f"3d_medium{'_perturbed' if perturbed else ''}",
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
# complex – 60° z-rotation, 6-blob hexagon → 6-blob hexagon
# ---------------------------------------------------------------------------

def _complex(perturbed: bool, output_root: str) -> ExperimentSpec:
    """60° z-rotation on a 6-blob hexagonal GMM generalised to a larger hexagonal pattern.

    Reference coupling
        6-blob hexagon at r=1.5 in the z=0 plane, transport = R_z(60°).
        Under a 60° rotation a regular hexagon maps each vertex to the next,
        giving a clean 1-to-1 OT assignment.
    New marginals
        6-blob hexagon at r=4 in the z=2 plane;  target = R_z(60°) of source
        (+ R_x(30°) if perturbed).
    """
    device = get_device()
    r0     = 1.5
    ref_src = [
        [r0 * math.cos(k * math.pi / 3), r0 * math.sin(k * math.pi / 3), 0.0]
        for k in range(6)
    ]
    ref_tgt  = _rot_z(ref_src, 60.0)   # maps vertex k → vertex k+1 (mod 6)
    coupling = GMM3DCoupling(ref_src, ref_tgt, blob_scale=0.25, device=device)

    r1, z1 = 4.0, 2.0
    new_src_c = [
        [r1 * math.cos(k * math.pi / 3), r1 * math.sin(k * math.pi / 3), z1]
        for k in range(6)
    ]
    new_tgt_c = _rot_z(new_src_c, 60.0)
    if perturbed:
        new_tgt_c = _rot_x(new_tgt_c, 30.0)
    new_src = GMM3DDistribution(new_src_c, blob_scale=0.3, device=device)
    new_tgt = GMM3DDistribution(new_tgt_c, blob_scale=0.3, device=device)

    transport_map, inverse_transport_map = _make_transport_maps(60.0, perturbed)

    train_cfg                    = _default_train_cfg(device)
    train_cfg.iterations         = 100             # 200 risks coupling drift; 100 is more stable
    train_cfg.pretrain_epochs    = 8_000           # 6-blob hexagon needs more flow pretraining
    train_cfg.dual_epochs        = 200             # harder geometry → more dual convergence steps
    train_cfg.cfm_epochs         = 750             # more CFM steps for complex 3D geometry
    # train_cfg.num_rectified_flow = 1               # extra rectification for 4→6 blob jump
    method_cfg                   = _default_method_cfg()
    method_cfg.lr_min_factor     = 0.3             # slightly more aggressive decay for complex
    method_cfg.dual_warmup_epochs = 100            # larger warmup for hardest 3D geometry
    method_cfg.lip_reg           = 15              # stronger Lipschitz for 6-blob hexagonal potential
    method_cfg.particle_ratio    = 0.35            # more particle mixing for complex multi-mode case
    model_cfg  = default_model_cfg(dimension=3, hidden_size=256, num_layers=4)  # deeper for 6-blob hexagon
    vis_cfg                = VisualizeConfig(show_each=1, n_trajectories=800, trajectory_stride=2, show_each_marginal=100, show_each_pretrain=1000)

    return ExperimentSpec(
        name                  = f"3d_complex{'_perturbed' if perturbed else ''}",
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

def make_3d_experiments(output_root: str = "results") -> List[ExperimentSpec]:
    """Return all 3-D experiment specs (simple / medium / complex × standard / perturbed)."""
    return [
        _simple(False,  output_root),
        _simple(True,   output_root),
        _medium(False,  output_root),
        _medium(True,   output_root),
        _complex(False, output_root),
        _complex(True,  output_root),
    ]
