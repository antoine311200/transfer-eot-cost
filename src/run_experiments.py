"""
DualWCFM Experiment Runner
==========================
Runs 2-D, 3-D and image experiments with both unperturbed and perturbed costs.

Usage (from repo root):
    cd src
    python run_experiments.py --mode 2d --level simple
    python run_experiments.py --mode 3d --level medium --perturbed
    python run_experiments.py --mode image --level simple
    python run_experiments.py --all          # run everything sequentially

Each experiment saves all plots and metrics under results/<name>/.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make sure src/ is on the path when running as __main__
sys.path.insert(0, os.path.dirname(__file__))

from dualwcfm.config import FeedForwardConfig, TrainingConfig, VisualizeConfig
from dualwcfm.distributions.coupling import (
    FixedSizeCouplingDistribution,
    FunctionalCouplingDistribution,
    RotatedShapeCoupling,
    CouplingDistribution,
)
from dualwcfm.distributions.distribution import (
    Distribution,
    GMMShapeDistribution,
    Simple2DGaussianDistribution,
    get_cross_dist,
    get_square_dist,
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


# =============================================================================
# Helpers
# =============================================================================

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _wrapper():
    return torch_wrapper


# =============================================================================
# 3-D coupling / distribution helpers
# =============================================================================

class Sphere3DCoupling(CouplingDistribution):
    """Pair of Gaussian blobs at two locations in R^3."""

    def __init__(self, center_src, center_tgt, scale=0.25, device=None):
        self.dev = device or _device()
        self.c_src = torch.tensor(center_src, dtype=torch.float32, device=self.dev)
        self.c_tgt = torch.tensor(center_tgt, dtype=torch.float32, device=self.dev)
        self.scale = scale

    def coupling_sample(self, n):
        noise = torch.randn(n, 3, device=self.dev)
        x = self.c_src + noise * self.scale
        y = self.c_tgt + noise * self.scale + torch.randn(n, 3, device=self.dev) * self.scale * 0.3
        return x, y

    def sample_init(self, n):
        return self.coupling_sample(n)[0]

    def sample_target(self, n):
        return self.coupling_sample(n)[1]


class GMM3DCoupling(CouplingDistribution):
    """Multiple sphere pairs in R^3."""

    def __init__(self, src_centers, tgt_centers, scale=0.2, device=None):
        self.dev = device or _device()
        self.src = torch.tensor(src_centers, dtype=torch.float32, device=self.dev)
        self.tgt = torch.tensor(tgt_centers, dtype=torch.float32, device=self.dev)
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


# Convenience 3-D marginal wrappers
class _SrcDist3D(Distribution):
    def __init__(self, coupling): self._c = coupling
    def sample(self, n): return self._c.sample_init(n)

class _TgtDist3D(Distribution):
    def __init__(self, coupling): self._c = coupling
    def sample(self, n): return self._c.sample_target(n)


# =============================================================================
# Image coupling helper  (MNIST / FashionMNIST)
# =============================================================================

def _load_mnist_coupling(dataset_cls, class_src: int, class_tgt: int,
                         n: int = 5000, device=None):
    """Return a FunctionalCouplingDistribution over two MNIST digit classes."""
    from torchvision import datasets, transforms

    dev = device or _device()
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # scale to [-1, 1]
    ])
    root = os.path.join(os.path.dirname(__file__), "..", "data")
    dset = dataset_cls(root=root, train=True, download=True, transform=tfm)

    def _get_class(cls, n_max):
        idx = [i for i, (_, label) in enumerate(dset) if label == cls][:n_max]
        return torch.stack([dset[i][0] for i in idx]).to(dev)

    xs = _get_class(class_src, n)
    ys = _get_class(class_tgt, n)

    def _sample(k):
        ix = torch.randint(len(xs), (k,))
        iy = torch.randint(len(ys), (k,))
        return xs[ix], ys[iy]

    return FunctionalCouplingDistribution(_sample), xs.shape[1:]  # (C, H, W)


# =============================================================================
# MLP Strategy factory that accepts 'dimension' correctly for both modes
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
    transport_map: Optional[Callable]   # ground-truth map (for cost diagnostic)
    train_cfg: TrainingConfig
    model_cfg: FeedForwardConfig
    vis_cfg: VisualizeConfig
    bidirectional: bool = False
    output_root: str = "results"

    @property
    def output_dir(self):
        return os.path.join(self.output_root, self.name)


# =============================================================================
# 2-D experiment catalogue
# =============================================================================

def _base_2d_train_cfg(device, perturbed=False) -> TrainingConfig:
    return TrainingConfig(
        device=device,
        dataset_size=20_000,
        marginal_epochs=2000,
        pretrain_epochs=5000,
        dual_epochs=150,
        cfm_epochs=200,
        advance_steps=100,
        iterations=50,
        loss="weighted",
        sigma=0.05,
        sigma2=0.05,
        l2_reg=1e-3,
        lip_reg=1e-3,
        mean_reg=0.1,
        batch_size=2048,
        coupling_particle_ratio=0.0,
        dual_mismatch_reg=0.5 if perturbed else 0.0,
        correction_each=0,
    )


def make_2d_experiments(output_root="results") -> List[ExperimentSpec]:
    dev = _device()
    specs = []

    # ---- Simple: Gaussian → rotated Gaussian --------------------------------
    def _simple_coupling(perturbed):
        centers = [[0.0, 0.0]]
        coup = RotatedShapeCoupling(
            centers, source_scale=0.5, target_scale=0.5,
            rotation_degrees=90.0, device=dev,
        )
        src_dist = Simple2DGaussianDistribution(
            torch.tensor([0.0, 0.0], device=dev), torch.tensor([0.5, 0.5], device=dev), device=str(dev))
        tgt_dist = Simple2DGaussianDistribution(
            torch.tensor([0.0, 0.0], device=dev), torch.tensor([0.5, 0.5], device=dev), device=str(dev))
        transport_map = lambda x: torch.stack([-x[:, 1], x[:, 0]], dim=1)
        cfg = _base_2d_train_cfg(dev, perturbed)
        mcfg = FeedForwardConfig(dimension=2, hidden_size=64)
        vcfg = VisualizeConfig(show_each=10, n_trajectories=300, trajectory_stride=2)
        return ExperimentSpec(
            name=f"2d_simple{'_perturbed' if perturbed else ''}",
            mode="2d", initial_coupling=coup, new_source=src_dist,
            new_target=tgt_dist, transport_map=transport_map,
            train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            output_root=output_root,
        )

    specs += [_simple_coupling(False), _simple_coupling(True)]

    # ---- Medium: 2-mode GMM → 2-mode GMM rotated 90° -----------------------
    def _medium_coupling(perturbed):
        centers = [[2.0, 0.0], [-2.0, 0.0]]
        coup = RotatedShapeCoupling(
            centers, source_scale=0.5, target_scale=0.5,
            rotation_degrees=90.0, device=dev,
        )
        src_dist = GMMShapeDistribution(centers, scale=0.5, device=dev)
        tgt_dist = GMMShapeDistribution(centers, scale=0.5, rotation_degrees=90.0, device=dev)

        cfg = _base_2d_train_cfg(dev, perturbed)
        cfg.iterations = 60
        cfg.coupling_particle_ratio = 0.3 if perturbed else 0.0
        mcfg = FeedForwardConfig(dimension=2, hidden_size=128)
        vcfg = VisualizeConfig(show_each=10, n_trajectories=400, trajectory_stride=2)
        return ExperimentSpec(
            name=f"2d_medium{'_perturbed' if perturbed else ''}",
            mode="2d", initial_coupling=coup,
            new_source=src_dist, new_target=tgt_dist,
            transport_map=None, train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            output_root=output_root,
        )

    specs += [_medium_coupling(False), _medium_coupling(True)]

    # ---- Complex: cross shape → square shape ---------------------------------
    def _complex_coupling(perturbed):
        coup = RotatedShapeCoupling(
            centers=[[0, 0], [1.5, 0], [-1.5, 0], [0, 1.5], [0, -1.5]],
            source_scale=0.15, target_scale=0.15,
            rotation_degrees=45.0,
            source_shift=(0, 0), target_shift=(3, 3),
            device=dev,
        )
        src_dist = get_cross_dist(scale=0.15)
        tgt_dist = get_square_dist(shift=(3, 3), scale=0.15)

        cfg = _base_2d_train_cfg(dev, perturbed)
        cfg.iterations = 80
        cfg.dual_epochs = 200
        cfg.cfm_epochs = 300
        cfg.coupling_particle_ratio = 0.4 if perturbed else 0.0
        cfg.dataset_size = 30_000
        mcfg = FeedForwardConfig(dimension=2, hidden_size=128)
        vcfg = VisualizeConfig(show_each=10, n_trajectories=400, trajectory_stride=2)
        return ExperimentSpec(
            name=f"2d_complex{'_perturbed' if perturbed else ''}",
            mode="2d", initial_coupling=coup,
            new_source=src_dist, new_target=tgt_dist,
            transport_map=None, train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            output_root=output_root,
        )

    specs += [_complex_coupling(False), _complex_coupling(True)]
    return specs


# =============================================================================
# 3-D experiment catalogue
# =============================================================================

def _base_3d_train_cfg(device, perturbed=False) -> TrainingConfig:
    return TrainingConfig(
        device=device,
        dataset_size=20_000,
        marginal_epochs=3000,
        pretrain_epochs=6000,
        dual_epochs=200,
        cfm_epochs=300,
        advance_steps=100,
        iterations=60,
        loss="weighted",
        sigma=0.05,
        sigma2=0.05,
        l2_reg=1e-3,
        lip_reg=1e-3,
        mean_reg=0.1,
        batch_size=2048,
        coupling_particle_ratio=0.0,
        dual_mismatch_reg=0.5 if perturbed else 0.0,
        correction_each=0,
    )


def make_3d_experiments(output_root="results") -> List[ExperimentSpec]:
    dev = _device()
    specs = []

    # ---- Simple: sphere → sphere --------------------------------------------
    def _simple3d(perturbed):
        coup = Sphere3DCoupling([-1, -1, -1], [1, 1, 1], scale=0.3, device=dev)
        src_d = _SrcDist3D(coup)
        tgt_d = _TgtDist3D(coup)
        cfg = _base_3d_train_cfg(dev, perturbed)
        mcfg = FeedForwardConfig(dimension=3, hidden_size=128)
        vcfg = VisualizeConfig(show_each=10, n_trajectories=200, trajectory_stride=2)
        return ExperimentSpec(
            name=f"3d_simple{'_perturbed' if perturbed else ''}",
            mode="3d", initial_coupling=coup, new_source=src_d, new_target=tgt_d,
            transport_map=None, train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            output_root=output_root,
        )
    specs += [_simple3d(False), _simple3d(True)]

    # ---- Medium: 2 spheres → 2 spheres (rearranged) -------------------------
    def _medium3d(perturbed):
        src_c = [[-1, 0, 0], [1, 0, 0]]
        tgt_c = [[0, -1, 0], [0, 1, 0]]
        coup = GMM3DCoupling(src_c, tgt_c, scale=0.25, device=dev)
        src_d = _SrcDist3D(coup)
        tgt_d = _TgtDist3D(coup)
        cfg = _base_3d_train_cfg(dev, perturbed)
        cfg.iterations = 70
        cfg.coupling_particle_ratio = 0.3 if perturbed else 0.0
        mcfg = FeedForwardConfig(dimension=3, hidden_size=128)
        vcfg = VisualizeConfig(show_each=10, n_trajectories=200, trajectory_stride=2)
        return ExperimentSpec(
            name=f"3d_medium{'_perturbed' if perturbed else ''}",
            mode="3d", initial_coupling=coup, new_source=src_d, new_target=tgt_d,
            transport_map=None, train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            output_root=output_root,
        )
    specs += [_medium3d(False), _medium3d(True)]

    # ---- Complex: cross of spheres → cube of spheres ------------------------
    def _complex3d(perturbed):
        r = 1.5
        src_c = [[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]]
        tgt_c = [[r, r, r], [r, r, -r], [r, -r, r], [r, -r, -r],
                 [-r, r, r], [-r, -r, -r]]
        # Ensure same number of centers
        tgt_c = tgt_c + tgt_c[:len(src_c) - len(tgt_c)]  # pad if needed
        coup = GMM3DCoupling(src_c, tgt_c[:len(src_c)], scale=0.2, device=dev)
        src_d = _SrcDist3D(coup)
        tgt_d = _TgtDist3D(coup)
        cfg = _base_3d_train_cfg(dev, perturbed)
        cfg.iterations = 100
        cfg.dual_epochs = 250
        cfg.cfm_epochs = 350
        cfg.coupling_particle_ratio = 0.45 if perturbed else 0.0
        cfg.dataset_size = 30_000
        mcfg = FeedForwardConfig(dimension=3, hidden_size=256)
        vcfg = VisualizeConfig(show_each=10, n_trajectories=200, trajectory_stride=2)
        return ExperimentSpec(
            name=f"3d_complex{'_perturbed' if perturbed else ''}",
            mode="3d", initial_coupling=coup, new_source=src_d, new_target=tgt_d,
            transport_map=None, train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            output_root=output_root,
        )
    specs += [_complex3d(False), _complex3d(True)]
    return specs


# =============================================================================
# Image experiment catalogue
# =============================================================================

def _base_image_train_cfg(device, perturbed=False) -> TrainingConfig:
    return TrainingConfig(
        device=device,
        dataset_size=2_000,
        marginal_epochs=500,
        pretrain_epochs=1000,
        dual_epochs=50,
        cfm_epochs=100,
        advance_steps=50,
        iterations=30,
        loss="weighted",
        sigma=0.05,
        sigma2=0.05,
        l2_reg=1e-3,
        lip_reg=1e-3,
        mean_reg=0.1,
        batch_size=128,
        coupling_particle_ratio=0.4 if perturbed else 0.0,
        dual_mismatch_reg=0.5 if perturbed else 0.0,
        use_disk_coupling=True,
        coupling_chunk_size=128,
        correction_each=0,
    )


def make_image_experiments(output_root="results") -> List[ExperimentSpec]:
    try:
        from torchvision import datasets
    except ImportError:
        print("torchvision not found – skipping image experiments.")
        return []

    dev = _device()
    specs = []

    # ---- MNIST 0 → 1 (simple) -----------------------------------------------
    def _mnist_simple(perturbed):
        coup, img_shape = _load_mnist_coupling(
            datasets.MNIST, class_src=0, class_tgt=1, n=2000, device=dev)
        C, H, W = img_shape

        from dualwcfm.distributions.coupling import MarginalDistribution
        src_d = MarginalDistribution(coup.sample_init)
        tgt_d = MarginalDistribution(coup.sample_target)

        cfg = _base_image_train_cfg(dev, perturbed)
        cfg.image_shape = (C, H, W)
        mcfg = FeedForwardConfig(dimension=C * H * W, hidden_size=256, image_shape=(C, H, W))
        vcfg = VisualizeConfig(show_each=5, n_trajectories=6, trajectory_stride=0)
        return ExperimentSpec(
            name=f"mnist_simple{'_perturbed' if perturbed else ''}",
            mode="image", initial_coupling=coup, new_source=src_d, new_target=tgt_d,
            transport_map=None, train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            output_root=output_root,
        )
    specs += [_mnist_simple(False), _mnist_simple(True)]

    # ---- FashionMNIST shirt (6) → sneaker (7) (medium) ----------------------
    def _fashion_medium(perturbed):
        coup, img_shape = _load_mnist_coupling(
            datasets.FashionMNIST, class_src=6, class_tgt=7, n=2000, device=dev)
        C, H, W = img_shape

        from dualwcfm.distributions.coupling import MarginalDistribution
        src_d = MarginalDistribution(coup.sample_init)
        tgt_d = MarginalDistribution(coup.sample_target)

        cfg = _base_image_train_cfg(dev, perturbed)
        cfg.image_shape = (C, H, W)
        cfg.iterations = 40
        mcfg = FeedForwardConfig(dimension=C * H * W, hidden_size=256, image_shape=(C, H, W))
        vcfg = VisualizeConfig(show_each=5, n_trajectories=6, trajectory_stride=0)
        return ExperimentSpec(
            name=f"fashion_medium{'_perturbed' if perturbed else ''}",
            mode="image", initial_coupling=coup, new_source=src_d, new_target=tgt_d,
            transport_map=None, train_cfg=cfg, model_cfg=mcfg, vis_cfg=vcfg,
            output_root=output_root,
        )
    specs += [_fashion_medium(False), _fashion_medium(True)]

    return specs


# =============================================================================
# Run a single experiment
# =============================================================================

def run_experiment(spec: ExperimentSpec, retrain: bool = True, save_models: bool = True):
    print(f"\n{'='*70}")
    print(f"  Experiment: {spec.name}")
    print(f"{'='*70}")

    os.makedirs(spec.output_dir, exist_ok=True)
    model_dir = os.path.join(spec.output_dir, "models") if save_models else None

    dev = spec.train_cfg.device

    # Strategy
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
        device=dev,
    )

    trainer.train(retrain=retrain, save=model_dir)

    # --- Post-training: assemble timelapse and save metrics plot ---
    vis.make_timelapse(fps=3)
    _save_metrics_plot(spec.output_dir)
    print(f"  Done → {spec.output_dir}")


def _save_metrics_plot(output_dir: str):
    import csv
    csv_path = os.path.join(output_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        return
    rows = []
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

    # Weight stats plot
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
    args = parser.parse_args()

    if args.run_all:
        for spec in _all_experiments(args.output):
            run_experiment(spec)
        return

    if args.mode is None:
        parser.print_help()
        return

    catalogue = {
        "2d": make_2d_experiments,
        "3d": make_3d_experiments,
        "image": make_image_experiments,
    }[args.mode](args.output)

    # Filter by level and perturbed flag
    name_parts = []
    if args.level:
        name_parts.append(args.level)
    if args.perturbed:
        name_parts.append("perturbed")

    matching = [s for s in catalogue
                if all(p in s.name for p in name_parts)]

    if not matching:
        print(f"No experiments matched mode={args.mode} level={args.level} "
              f"perturbed={args.perturbed}")
        return

    for spec in matching:
        run_experiment(spec)


if __name__ == "__main__":
    main()
