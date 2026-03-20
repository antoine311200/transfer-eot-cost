"""
DualWCFM Experiment Runner
==========================
Experiments for generative transfer of entropic optimal transport dynamics.

Goal: learn the transport dynamic encoded in a reference coupling π (e.g. "rotate by θ"),
then transfer it to new marginals (μ', ν') where ν' = T(μ') but μ' is in a DIFFERENT
support from the reference — demonstrating that the algorithm generalises the dynamic.

Usage (from repo root):
    cd src
    python run_experiments.py --mode 2d --level simple
    python run_experiments.py --mode 3d --level medium --perturbed
    python run_experiments.py --mode image --level simple
    python run_experiments.py --all          # run everything sequentially

Each experiment saves all plots and metrics under results/<name>/.
Per-type GIF timelapses are created after training.
"""

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torchvision.transforms.functional as TF

from dualwcfm.models.unet import UNetModel
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make sure src/ is on the path when running as __main__
sys.path.insert(0, os.path.dirname(__file__))

from dualwcfm.config import FeedForwardConfig, MixtureConfig, MethodConfig, TrainingConfig, VisualizeConfig
from dualwcfm.method import MixtureMethod
from dualwcfm.distributions.coupling import (
    FunctionalCouplingDistribution,
    RotatedShapeCoupling,
    CouplingDistribution,
    MarginalDistribution,
)
from dualwcfm.distributions.distribution import (
    Distribution,
    GMMShapeDistribution,
    Simple2DGaussianDistribution,
    Simple2DGaussianDistribution,
)
from dualwcfm.flow_matching import ConditionalFlowMatching
from dualwcfm.logging_visualizer import (
    LoggingVisualizer,
    LoggingVisualizer3D,
    LoggingVisualizerImage,
)
from dualwcfm.mlp import MLP, torch_wrapper
from dualwcfm.strategy import Strategy, BidirectionalStrategy
from dualwcfm.trainer import Trainer

logger = logging.getLogger("dualwcfm")


# =============================================================================
# Logging setup
# =============================================================================

def setup_logging(log_dir: str = None):
    """Configure the dualwcfm logger: StreamHandler + optional FileHandler."""
    root = logging.getLogger("dualwcfm")
    root.setLevel(logging.INFO)
    if not root.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        root.addHandler(sh)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        root.addHandler(fh)


# =============================================================================
# Helpers
# =============================================================================

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _wrapper():
    return torch_wrapper

class unet_wrapper(torch.nn.Module):
    """Wraps UNetModel for flow models.

    UNetModel.forward(t, x): first arg = timesteps, second arg = image.

    Caller convention (trainer/flow_matching):
        model(time_scalar, image, [condition_image])

    This wrapper passes: model(time_1d, cat([image, condition]))
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x=None, inp3=None, *args, **kwargs):
        # t = time (N,) or (N,1), x = image (N, C, H, W), inp3 = condition (N, C, H, W)
        # UNetModel.forward(t, x): first arg = timesteps, second arg = image
        t_flat = t.view(-1)  # (N,) required by timestep_embedding
        if inp3 is not None and x is not None:
            return self.model(t_flat, torch.cat([x, inp3], dim=1))
        if x is not None:
            return self.model(t_flat, x)
        else:
            # Single-arg call (e.g. during evolve with no condition):
            # t is the image, no explicit time — use zero timestep
            timesteps = torch.zeros(t.shape[0], device=t.device, dtype=torch.float32)
            return self.model(timesteps, t)


class dual_unet_wrapper(torch.nn.Module):
    """Wraps a stop_middle UNetModel for dual potential functions f, g, h.

    Dual potentials are time-independent: we always use a zero timestep.
    Input:  image  (N, C, H, W)
    Output: scalar (N, 1, 1, 1)  — global average of the encoder output,
            broadcast-compatible with image tensors in loss computations.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        t = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        # stop_middle=True: out has AdaptiveAvgPool2d + Flatten → shape (N, out_channels)
        return self.model(t, x)

# =============================================================================
# 3-D geometry helpers
# =============================================================================

def _rot_z(centers: List[List[float]], degrees: float) -> List[List[float]]:
    """Rotate a list of 3-D centres by `degrees` around the z-axis."""
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    out = []
    for x, y, z in centers:
        out.append([c * x - s * y, s * x + c * y, z])
    return out


# =============================================================================
# 3-D coupling / distribution helpers
# =============================================================================

class GMM3DCoupling(CouplingDistribution):
    """Multiple Gaussian blob pairs in R^3.

    Each pair (src_centers[k], tgt_centers[k]) encodes one mode of the transport.
    """

    def __init__(self, src_centers, tgt_centers, scale=0.2, device=None):
        self.dev = device or _device()
        self.src = torch.tensor(src_centers, dtype=torch.float32, device=self.dev)
        self.tgt = torch.tensor(tgt_centers, dtype=torch.float32, device=self.dev)
        assert self.src.shape == self.tgt.shape
        self.K = self.src.shape[0]
        self.scale = scale

    def coupling_sample(self, n):
        idx = torch.randint(self.K, (n,), device=self.dev)
        xs = self.src[idx] + torch.randn(n, 3, device=self.dev) * self.scale
        yt = self.tgt[idx] + torch.randn(n, 3, device=self.dev) * self.scale
        return xs, yt

    def sample_init(self, n):
        return self.coupling_sample(n)[0]

    def sample_target(self, n):
        return self.coupling_sample(n)[1]


class GMM3DDistribution(Distribution):
    """Simple 3-D Gaussian mixture (for new source / target marginals)."""

    def __init__(self, centers, scale=0.25, device=None):
        self.dev = device or _device()
        self.centers = torch.tensor(centers, dtype=torch.float32, device=self.dev)
        self.K = self.centers.shape[0]
        self.scale = scale

    def sample(self, n):
        idx = torch.randint(self.K, (n,), device=self.dev)
        return self.centers[idx] + torch.randn(n, 3, device=self.dev) * self.scale


# =============================================================================
# Image coupling helpers  (MNIST / FashionMNIST)
# =============================================================================

def _load_dataset_class(dataset_cls, cls: int, n: int, device, root: str):
    """Return up to n normalised images of class `cls` from dataset_cls, resized to 32×32."""
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize(32),          # UNet configured for 32×32 (powers-of-2 skip connections)
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dset = dataset_cls(root=root, train=True, download=True, transform=tfm)
    idx = [i for i, (_, label) in enumerate(dset) if label == cls][:n]
    return torch.stack([dset[i][0] for i in idx]).to(device)


def _make_flip_coupling(images: torch.Tensor, rotation_degrees: float = 0.0) -> FunctionalCouplingDistribution:
    """Coupling: (x, rotation(horizontal_flip(x))) for images in a fixed set."""
    def _sample(k):
        ix = torch.randint(len(images), (k,))
        x = images[ix]
        y = torch.flip(x, dims=[3])  # flip along width axis
        if rotation_degrees != 0.0:
            y = TF.rotate(y, rotation_degrees, fill=-1.0)
        return x, y
    return FunctionalCouplingDistribution(_sample)


def _make_flip_distributions(images: torch.Tensor, rotation_degrees: float = 0.0,
                              target_extra_rotation: float = 0.0):
    """Return (source_dist, target_dist) where target = extra_rotation(rotation(horizontal_flip(source))).

    target_extra_rotation: additional rotation applied on top of the coupling rotation,
    used in perturbed experiments to shift the final target marginal.
    """
    def _src(k):
        ix = torch.randint(len(images), (k,))
        return images[ix]
    def _tgt(k):
        ix = torch.randint(len(images), (k,))
        y = torch.flip(images[ix], dims=[3])
        if rotation_degrees != 0.0:
            y = TF.rotate(y, rotation_degrees, fill=-1.0)
        if target_extra_rotation != 0.0:
            y = TF.rotate(y, target_extra_rotation, fill=-1.0)
        return y
    return MarginalDistribution(_src), MarginalDistribution(_tgt)


# =============================================================================
# MLP Strategy factory
# =============================================================================

def _make_strategy(dim: int, hidden: int, bidirectional: bool = False):
    dev = _device()
    cfg = FeedForwardConfig(dimension=dim, hidden_size=hidden)
    cls = BidirectionalStrategy if bidirectional else Strategy
    strat = cls(_wrapper(), cfg)
    strat.to_device(dev)
    return strat, cfg


# =============================================================================
# Experiment dataclass
# =============================================================================

@dataclass
class ExperimentSpec:
    name: str
    mode: str                           # "2d" | "3d" | "image"
    initial_coupling: CouplingDistribution
    new_source: Distribution
    new_target: Distribution
    transport_map: Optional[Callable]   # ground-truth map (for diagnostic)
    train_cfg: TrainingConfig
    model_cfg: FeedForwardConfig
    vis_cfg: VisualizeConfig
    method_cfg: MethodConfig = field(default_factory=MixtureConfig)
    bidirectional: bool = True
    output_root: str = "results"

    @property
    def output_dir(self):
        return os.path.join(self.output_root, self.name)


# =============================================================================
# 2-D experiment catalogue
# =============================================================================
# Design principle:
#   - Reference coupling π encodes rotation by θ degrees on a shape S at scale r₀
#   - New source μ' = same shape S (or a different shape) at a DIFFERENT location / scale r₁ ≠ r₀
#   - New target ν' = μ' rotated by θ (respects the same dynamic, different support)
# =============================================================================

def _base_2d_train_cfg(device) -> TrainingConfig:
    return TrainingConfig(
        device=device,
        dataset_size=200_000,
        marginal_epochs=5000,
        pretrain_epochs=10000,
        dual_epochs=100,
        cfm_epochs=250,
        advance_steps=100,
        iterations=100,
        num_rectified_flow=1,
        scale_factor=100.0,
        batch_size=2048*4,
        coupling_chunk_size=1000,
    )


def _base_2d_method_cfg() -> MixtureConfig:
    return MixtureConfig(
        sigma=0.01,
        sigma2=0.01,
        loss="weighted",
        l2_reg=0,
        lip_reg=0,
        mean_reg=0,
        use_noisy_field=False,
        particle_ratio=0.25,
    )


def make_2d_experiments(output_root="results") -> List[ExperimentSpec]:
    dev = _device()
    specs = []

    # -------------------------------------------------------------------------
    # Simple (θ=90°):
    #   Reference π: single Gaussian blob at origin → rotated 90°
    #   New source μ': single Gaussian at [4, 2]       (different support)
    #   New target ν': single Gaussian at [-2, 4]      (= rotate90([4,2]))
    # -------------------------------------------------------------------------
    def _simple_2d(perturbed):
        # Reference coupling: single blob at origin, transport = rotate 90°
        # coup = RotatedShapeCoupling(
        #     [[0.0, 0.0]], source_scale=0.35, target_scale=0.35,
        #     rotation_degrees=90.0, device=dev,
        # )
        # # New marginals: single blob far from origin — same rotation dynamic
        # new_src = GMMShapeDistribution([[4.0, 2.0]], scale=0.35, device=dev)
        # new_tgt = GMMShapeDistribution([[-2.0, 4.0]], scale=0.35, device=dev)  # rotate90([4,2])
        # transport_map = lambda x: torch.stack([-x[:, 1], x[:, 0]], dim=1)

        def anticorrelate(n: int):
            source_samples = torch.randn(n, 2) * 0.5
            perturbation = torch.tensor([0, 0.0]) if not perturbed else torch.tensor([0, 2.0])
            return source_samples, -source_samples + perturbation

        loc = 10.0
        perturbation = 0.0 if not perturbed else 2.0
        coupling = FunctionalCouplingDistribution(anticorrelate)
        new_source = Simple2DGaussianDistribution(mean=torch.tensor([loc, loc]), std=torch.tensor([0.5, 0.5]))
        new_target = Simple2DGaussianDistribution(mean=torch.tensor([-loc, -loc + perturbation]), std=torch.tensor([0.5, 0.5]))
        def transport_map(x):
            A = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
            b = torch.tensor([0.0, +perturbation])
            return x @ A.T + b

        cfg = _base_2d_train_cfg(dev)
        method_cfg = _base_2d_method_cfg()
        mcfg = FeedForwardConfig(dimension=2, hidden_size=64)
        vcfg = VisualizeConfig(show_each=1, n_trajectories=300, trajectory_stride=2)
        return ExperimentSpec(
            name=f"2d_simple{'_perturbed' if perturbed else ''}",
            mode="2d", initial_coupling=coupling,
            new_source=new_source, new_target=new_target,
            transport_map=transport_map,
            train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            method_cfg=method_cfg,
            output_root=output_root,
        )
    specs += [_simple_2d(False), _simple_2d(True)]

    # -------------------------------------------------------------------------
    # Medium (θ=60°):
    #   Reference π: 2-mode GMM at [[2,0],[-2,0]] → rotated 60°
    #   New source μ': 4-mode square GMM at [±5, ±5]  (larger scale, different support)
    #   New target ν': same 4-mode square rotated 60°
    # -------------------------------------------------------------------------
    def _medium_2d(perturbed):
        ref_centers = [[2.0, 0.0], [-2.0, 0.0]]
        coup = RotatedShapeCoupling(
            ref_centers, source_scale=0.4, target_scale=0.4,
            rotation_degrees=60.0, device=dev,
        )
        # New marginals: square of 4 blobs at ±5 — different support, same rotation
        new_centers = [[5.0, 5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, -5.0]]
        new_src = GMMShapeDistribution(new_centers, scale=0.4, device=dev)
        new_tgt = GMMShapeDistribution(new_centers, scale=0.4, rotation_degrees=60.0, device=dev)

        cfg = _base_2d_train_cfg(dev)
        method_cfg = _base_2d_method_cfg()
        cfg.iterations = 100
        mcfg = FeedForwardConfig(dimension=2, hidden_size=128)
        vcfg = VisualizeConfig(show_each=1, n_trajectories=2000, trajectory_stride=2)
        return ExperimentSpec(
            name=f"2d_medium{'_perturbed' if perturbed else ''}",
            mode="2d", initial_coupling=coup,
            new_source=new_src, new_target=new_tgt,
            transport_map=None,
            train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            method_cfg=method_cfg,
            output_root=output_root,
        )
    specs += [_medium_2d(False), _medium_2d(True)]

    # -------------------------------------------------------------------------
    # Complex (θ=45°):
    #   Reference π: cross of 5 blobs at scale 1 → rotated 45°
    #   New source μ': cross of 5 blobs at scale 3  (larger scale = different support)
    #   New target ν': same cross at scale 3, rotated 45°
    # -------------------------------------------------------------------------
    def _complex_2d(perturbed):
        r0 = 1.5
        ref_centers = [[0, 0], [r0, 0], [-r0, 0], [0, r0], [0, -r0]]
        coup = RotatedShapeCoupling(
            ref_centers, source_scale=0.15, target_scale=0.15,
            rotation_degrees=45.0, device=dev,
        )
        # New marginals: cross at 3× the radius — different support, same rotation
        r1 = 4.5
        new_centers = [[0, 0], [r1, 0], [-r1, 0], [0, r1], [0, -r1]]
        new_src = GMMShapeDistribution(new_centers, scale=0.4, device=dev)
        new_tgt = GMMShapeDistribution(new_centers, scale=0.4, rotation_degrees=45.0, device=dev)

        cfg = _base_2d_train_cfg(dev)
        method_cfg = _base_2d_method_cfg()
        cfg.iterations = 150
        cfg.dual_epochs = 200
        cfg.cfm_epochs = 300
        mcfg = FeedForwardConfig(dimension=2, hidden_size=128)
        vcfg = VisualizeConfig(show_each=1, n_trajectories=400, trajectory_stride=2)
        return ExperimentSpec(
            name=f"2d_complex{'_perturbed' if perturbed else ''}",
            mode="2d", initial_coupling=coup,
            new_source=new_src, new_target=new_tgt,
            transport_map=None,
            train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            method_cfg=method_cfg,
            output_root=output_root,
        )
    specs += [_complex_2d(False), _complex_2d(True)]
    return specs


# =============================================================================
# 3-D experiment catalogue
# =============================================================================
# Design principle (same as 2-D):
#   - Reference coupling π encodes z-rotation by θ on blobs at positions S₀
#   - New source μ' = GMM at positions S₁ (DIFFERENT from S₀)
#   - New target ν' = rotate_z(θ)(S₁) — same dynamic, different support
# =============================================================================

def _base_3d_train_cfg(device) -> TrainingConfig:
    return TrainingConfig(
        device=device,
        dataset_size=200_000,
        marginal_epochs=5000,
        pretrain_epochs=10000,
        dual_epochs=200,
        cfm_epochs=300,
        advance_steps=100,
        iterations=200,
        scale_factor=100.0,
        batch_size=2048,
    )


def _base_3d_method_cfg() -> MixtureConfig:
    return MixtureConfig(
        sigma=0.05,
        sigma2=0.05,
        loss="weighted",
        l2_reg=0,
        lip_reg=0,
        mean_reg=0,
        use_noisy_field=False,
        particle_ratio=0.25,
    )


def make_3d_experiments(output_root="results") -> List[ExperimentSpec]:
    dev = _device()
    specs = []

    # -------------------------------------------------------------------------
    # Simple (θ=90° z-rotation):
    #   Reference π: 2 blobs at [±1.5, 0, 0] → [0, ±1.5, 0]  (90° z-rotation)
    #   New source μ': 3 blobs at [[3,1,1],[3,-1,1],[-3,0,1]]  (different region)
    #   New target ν': 3 blobs at [[-1,3,1],[1,3,1],[0,-3,1]]  (= rotate_z90(src))
    # -------------------------------------------------------------------------
    def _simple_3d(perturbed):
        ref_src = [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]]
        ref_tgt = _rot_z(ref_src, 90.0)
        coup = GMM3DCoupling(ref_src, ref_tgt, scale=0.25, device=dev)

        # New source at a different region with z offset
        new_src_c = [[3.0, 1.0, 1.0], [3.0, -1.0, 1.0], [-3.0, 0.0, 1.0]]
        new_tgt_c = _rot_z(new_src_c, 90.0)
        new_src = GMM3DDistribution(new_src_c, scale=0.25, device=dev)
        new_tgt = GMM3DDistribution(new_tgt_c, scale=0.25, device=dev)

        cfg = _base_3d_train_cfg(dev)
        method_cfg = _base_3d_method_cfg()
        mcfg = FeedForwardConfig(dimension=3, hidden_size=128)
        vcfg = VisualizeConfig(show_each=1, n_trajectories=200, trajectory_stride=2)
        return ExperimentSpec(
            name=f"3d_simple{'_perturbed' if perturbed else ''}",
            mode="3d", initial_coupling=coup,
            new_source=new_src, new_target=new_tgt,
            transport_map=None,
            train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            method_cfg=method_cfg,
            output_root=output_root,
        )
    specs += [_simple_3d(False), _simple_3d(True)]

    # -------------------------------------------------------------------------
    # Medium (θ=120° z-rotation):
    #   Reference π: 3 blobs at equilateral triangle, transport = rotate 120°
    #                (so each vertex maps to the next — the triangle is preserved)
    #   New source μ': 4 blobs at [±4, ±2, 1]           (different support)
    #   New target ν': 4 blobs rotated 120°
    # -------------------------------------------------------------------------
    def _medium_3d(perturbed):
        # Equilateral triangle in the xy-plane at r=2
        r = 2.0
        ref_src = [
            [r, 0.0, 0.0],
            [r * math.cos(2*math.pi/3), r * math.sin(2*math.pi/3), 0.0],
            [r * math.cos(4*math.pi/3), r * math.sin(4*math.pi/3), 0.0],
        ]
        ref_tgt = _rot_z(ref_src, 120.0)  # vertices cycle: v0→v1→v2→v0
        coup = GMM3DCoupling(ref_src, ref_tgt, scale=0.3, device=dev)

        # New marginals: 4-blob square far from origin
        new_src_c = [[4.0, 2.0, 1.0], [4.0, -2.0, 1.0],
                     [-4.0, 2.0, 1.0], [-4.0, -2.0, 1.0]]
        new_tgt_c = _rot_z(new_src_c, 120.0)
        new_src = GMM3DDistribution(new_src_c, scale=0.3, device=dev)
        new_tgt = GMM3DDistribution(new_tgt_c, scale=0.3, device=dev)

        cfg = _base_3d_train_cfg(dev)
        method_cfg = _base_3d_method_cfg()
        mcfg = FeedForwardConfig(dimension=3, hidden_size=128)
        vcfg = VisualizeConfig(show_each=1, n_trajectories=200, trajectory_stride=2)
        return ExperimentSpec(
            name=f"3d_medium{'_perturbed' if perturbed else ''}",
            mode="3d", initial_coupling=coup,
            new_source=new_src, new_target=new_tgt,
            transport_map=None,
            train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            method_cfg=method_cfg,
            output_root=output_root,
        )
    specs += [_medium_3d(False), _medium_3d(True)]

    # -------------------------------------------------------------------------
    # Complex (θ=60° z-rotation):
    #   Reference π: 4-blob square pattern in z=0, rotated 60°
    #   New source μ': 6-blob hexagonal pattern at z=2  (different support + shape)
    #   New target ν': same 6-blob hexagonal, rotated 60°
    # -------------------------------------------------------------------------
    def _complex_3d(perturbed):
        r0 = 2.0
        ref_src = [
            [ r0,  0.0, 0.0], [0.0,  r0, 0.0],
            [-r0,  0.0, 0.0], [0.0, -r0, 0.0],
        ]
        ref_tgt = _rot_z(ref_src, 60.0)
        coup = GMM3DCoupling(ref_src, ref_tgt, scale=0.25, device=dev)

        # New marginals: hexagonal pattern at z=2, radius 4
        r1 = 4.0
        z1 = 2.0
        new_src_c = [
            [r1 * math.cos(k * math.pi / 3), r1 * math.sin(k * math.pi / 3), z1]
            for k in range(6)
        ]
        new_tgt_c = _rot_z(new_src_c, 60.0)
        new_src = GMM3DDistribution(new_src_c, scale=0.3, device=dev)
        new_tgt = GMM3DDistribution(new_tgt_c, scale=0.3, device=dev)

        cfg = _base_3d_train_cfg(dev)
        method_cfg = _base_3d_method_cfg()
        cfg.iterations = 200
        mcfg = FeedForwardConfig(dimension=3, hidden_size=256)
        vcfg = VisualizeConfig(show_each=1, n_trajectories=200, trajectory_stride=2)
        return ExperimentSpec(
            name=f"3d_complex{'_perturbed' if perturbed else ''}",
            mode="3d", initial_coupling=coup,
            new_source=new_src, new_target=new_tgt,
            transport_map=None,
            train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            method_cfg=method_cfg,
            output_root=output_root,
        )
    specs += [_complex_3d(False), _complex_3d(True)]
    return specs


# =============================================================================
# Image experiment catalogue
# =============================================================================
# Design principle:
#   - Reference coupling π: (x, flip(x)) for images of class A
#     The transport dynamic is "horizontal flip"
#   - New source μ': images of class B  (DIFFERENT class / support)
#   - New target ν': flip(images of class B)
#     The algorithm must generalise the flip dynamic to a new class
# =============================================================================

def _base_image_train_cfg(device) -> TrainingConfig:
    return TrainingConfig(
        device=device,
        dataset_size=2_000,
        marginal_epochs=5000,
        pretrain_epochs=10000,
        dual_epochs=25,
        cfm_epochs=200,
        advance_steps=50,
        iterations=100,
        scale_factor=1.0,
        batch_size=64,
        use_disk_coupling=True,
        coupling_chunk_size=2048,
        save_coupling_checkpoints=True,
        save_model_checkpoints=True,
    )


def _base_image_method_cfg() -> MixtureConfig:
    return MixtureConfig(
        sigma=0.1,
        sigma2=0.1,
        loss="weighted",
        l2_reg=0,
        lip_reg=0,
        mean_reg=0,
        use_noisy_field=False,
        particle_ratio=0.1,
    )

class BidirectionalUnetStrategy(BidirectionalStrategy):
    def _create_models(self, dim: int, hidden_size: int):
        self._device = None  # Reset so next to_device() call actually moves new models
        wrapper = self.wrapper
        self.mu_bridge = wrapper(UNetModel(dim=(1, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1))
        self.nu_bridge = wrapper(UNetModel(dim=(1, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1))
        # Dual potentials use a dedicated wrapper: time-independent, outputs scalar (N,1,1,1)
        self.f_dual = dual_unet_wrapper(UNetModel(dim=(1, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1, stop_middle=True))
        self.g_dual = dual_unet_wrapper(UNetModel(dim=(1, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1, stop_middle=True))
        self.h_dual = dual_unet_wrapper(UNetModel(dim=(2, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1, stop_middle=True))

        self.forward_model = wrapper(UNetModel(dim=(2, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1))
        self.previous_forward_model = wrapper(UNetModel(dim=(2, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1))
        self.backward_model = wrapper(UNetModel(dim=(2, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1))
        self.previous_backward_model = wrapper(UNetModel(dim=(2, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1))

        self.initial_mu_bridge = wrapper(UNetModel(dim=(1, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1))
        self.initial_nu_bridge = wrapper(UNetModel(dim=(1, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1))
        self.initial_forward_model = wrapper(UNetModel(dim=(2, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1))
        self.initial_backward_model = wrapper(UNetModel(dim=(2, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1))

def make_image_experiments(output_root="results") -> List[ExperimentSpec]:
    try:
        from torchvision import datasets
    except ImportError:
        print("torchvision not found – skipping image experiments.")
        return []

    dev = _device()
    root = os.path.join(os.path.dirname(__file__), "..", "data")
    specs = []

    # -------------------------------------------------------------------------
    # Simple:
    #   Reference π: MNIST digit 3  →  horizontal flip + 10° rotation of digit 3
    #   New source μ': MNIST digit 7  (different class = different support)
    #   New target ν': horizontal flip + 10° rotation of digit 7
    #   Perturbed: new target ν' additionally rotated by 10° (20° total)
    #   Goal: the model must learn to apply "flip" to an unseen digit class
    # -------------------------------------------------------------------------
    def _mnist_simple(perturbed):
        ref_images   = _load_dataset_class(datasets.MNIST, cls=3, n=2000, device=dev, root=root)
        new_images   = _load_dataset_class(datasets.MNIST, cls=7, n=2000, device=dev, root=root)
        C, H, W = ref_images.shape[1:]

        coup = _make_flip_coupling(ref_images, rotation_degrees=0.0)
        new_src, new_tgt = _make_flip_distributions(
            new_images, rotation_degrees=0.0,
            target_extra_rotation=10.0 if perturbed else 0.0,
        )

        cfg = _base_image_train_cfg(dev)
        method_cfg = _base_image_method_cfg()
        cfg.image_shape = (C, H, W)
        mcfg = FeedForwardConfig(dimension=C, hidden_size=32, image_shape=(C, H, W))
        vcfg = VisualizeConfig(show_each=1, n_trajectories=6, trajectory_stride=0, show_each_marginal=1000, show_each_pretrain=1000)
        return ExperimentSpec(
            name=f"mnist_simple{'_perturbed' if perturbed else ''}",
            mode="image", initial_coupling=coup,
            new_source=new_src, new_target=new_tgt,
            transport_map=None,
            train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            method_cfg=method_cfg,
            output_root=output_root,
        )
    specs += [_mnist_simple(False), _mnist_simple(True)]

    # -------------------------------------------------------------------------
    # Medium:
    #   Reference π: FashionMNIST shirt (class 6) → horizontal flip + 10° rotation of shirt
    #   New source μ': FashionMNIST sneaker (class 7)  (different garment type)
    #   New target ν': horizontal flip + 10° rotation of sneaker
    #   Perturbed: new target ν' additionally rotated by 10° (20° total)
    # -------------------------------------------------------------------------
    def _fashion_medium(perturbed):
        ref_images = _load_dataset_class(datasets.FashionMNIST, cls=6, n=2000, device=dev, root=root)
        new_images = _load_dataset_class(datasets.FashionMNIST, cls=7, n=2000, device=dev, root=root)
        C, H, W = ref_images.shape[1:]

        coup = _make_flip_coupling(ref_images, rotation_degrees=10.0)
        new_src, new_tgt = _make_flip_distributions(
            new_images, rotation_degrees=10.0,
            target_extra_rotation=10.0 if perturbed else 0.0,
        )

        cfg = _base_image_train_cfg(dev)
        method_cfg = _base_image_method_cfg()
        cfg.image_shape = (C, H, W)
        mcfg = FeedForwardConfig(dimension=C, hidden_size=256, image_shape=(C, H, W))
        vcfg = VisualizeConfig(show_each=1, n_trajectories=6, trajectory_stride=0)
        return ExperimentSpec(
            name=f"fashion_medium{'_perturbed' if perturbed else ''}",
            mode="image", initial_coupling=coup,
            new_source=new_src, new_target=new_tgt,
            transport_map=None,
            train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            method_cfg=method_cfg,
            output_root=output_root,
        )
    specs += [_fashion_medium(False), _fashion_medium(True)]

    return specs


# =============================================================================
# Run a single experiment
# =============================================================================

def run_experiment(spec: ExperimentSpec, retrain: bool = True, save_models: bool = True):
    setup_logging(spec.output_dir)
    logger.info(f"\n{'='*70}")
    logger.info(f"  Experiment: {spec.name}")
    logger.info(f"{'='*70}")

    os.makedirs(spec.output_dir, exist_ok=True)
    model_dir = os.path.join(spec.output_dir, "models") if save_models else None

    dev = spec.train_cfg.device

    # Method
    method = MixtureMethod(spec.method_cfg)

    # Strategy
    # Check if image experiment with UNet config – if so, use BidirectionalUnetStrategy
    if spec.mode == "image" and spec.model_cfg.image_shape is not None:
        logger.info("  Using Bidirectional UNet strategy for image experiment.")
        strat = BidirectionalUnetStrategy(unet_wrapper, spec.model_cfg)
        strat.config = spec.model_cfg
        strat.to_device(dev)
    else:
        strat, _mcfg = _make_strategy(
            dim=spec.model_cfg.flat_dimension,
            hidden=spec.model_cfg.hidden_size,
            bidirectional=spec.bidirectional,
        )
        strat.config = spec.model_cfg

    # Flow matching
    fm = ConditionalFlowMatching()

    # Visualizer
    if spec.mode == "2d":
        vis = LoggingVisualizer(spec.vis_cfg, spec.transport_map, spec.output_dir)
    elif spec.mode == "3d":
        vis = LoggingVisualizer3D(spec.vis_cfg, spec.transport_map, spec.output_dir)
    else:
        assert spec.train_cfg.image_shape is not None
        vis = LoggingVisualizerImage(spec.vis_cfg, spec.train_cfg.image_shape, spec.output_dir)

    # Trainer
    trainer = Trainer(
        initial_coupling=spec.initial_coupling,
        new_source=spec.new_source,
        new_target=spec.new_target,
        flow_matching=fm,
        strategy=strat,
        visualizer=vis,
        config=spec.train_cfg,
        method=method,
        device=dev,
    )

    trainer.train(retrain=retrain, save=model_dir)

    # Post-training: one GIF per plot type + summary metrics plot
    vis.make_timelapse_all(fps=3)
    _save_metrics_plot(spec.output_dir)
    print(f"  Done → {spec.output_dir}")


def _save_metrics_plot(output_dir: str):
    import csv
    csv_path = os.path.join(output_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    iters = [int(r["iteration"]) for r in rows]
    keys = ["W1_mu", "W2_mu", "W1_nu", "W2_nu"]
    colors = ["royalblue", "steelblue", "tomato", "firebrick"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, color in zip(keys[:2], colors[:2]):
        vals = [float(r[key]) for r in rows]
        axes[0].plot(iters, vals, label=key, color=color)
    axes[0].set_title("Source marginal SW distances")
    axes[0].set_xlabel("Iteration")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for key, color in zip(keys[2:], colors[2:]):
        vals = [float(r[key]) for r in rows]
        axes[1].plot(iters, vals, label=key, color=color)
    axes[1].set_title("Target marginal SW distances")
    axes[1].set_xlabel("Iteration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics.png"), dpi=100)
    plt.close(fig)

    # Weight evolution plot
    fig, ax = plt.subplots(figsize=(7, 3))
    wm = [float(r["weight_mean"]) for r in rows]
    ws = [float(r["weight_std"]) for r in rows]
    ax.plot(iters, wm, label="mean(w)", color="navy")
    ax.fill_between(iters,
                    [m - s for m, s in zip(wm, ws)],
                    [m + s for m, s in zip(wm, ws)],
                    alpha=0.2, color="navy")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Importance weight evolution")
    ax.set_xlabel("Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "weights_evolution.png"), dpi=100)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def _all_experiments(output_root="results"):
    return (
        make_2d_experiments(output_root)
        + make_3d_experiments(output_root)
        + make_image_experiments(output_root)
    )


def main():
    parser = argparse.ArgumentParser(description="DualWCFM experiment runner")
    parser.add_argument("--mode", choices=["2d", "3d", "image"], default=None)
    parser.add_argument("--level", choices=["simple", "medium", "complex"], default=None)
    parser.add_argument("--perturbed", action="store_true")
    parser.add_argument("--all", action="store_true", dest="run_all")
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--load-pretrained", action="store_true", dest="load_pretrained",
        help="Skip marginal flow + pretrain phases and load previously saved models from results/<name>/models/",
    )
    args = parser.parse_args()

    if args.run_all:
        for spec in _all_experiments(args.output):
            run_experiment(spec, retrain=not args.load_pretrained)
        return

    if args.mode is None:
        parser.print_help()
        return

    catalogue = {
        "2d":    make_2d_experiments,
        "3d":    make_3d_experiments,
        "image": make_image_experiments,
    }[args.mode](args.output)

    name_parts = []
    if args.level:
        name_parts.append(args.level)
    if args.perturbed:
        name_parts.append("perturbed")

    matching = [s for s in catalogue if all(p in s.name for p in name_parts)]

    if not matching:
        print(f"No experiments matched mode={args.mode} level={args.level} "
              f"perturbed={args.perturbed}")
        return

    print(matching)
    for spec in matching:
        run_experiment(spec, retrain=not args.load_pretrained)


if __name__ == "__main__":
    main()
