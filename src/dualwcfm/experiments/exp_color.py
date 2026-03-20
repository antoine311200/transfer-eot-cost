"""dualwcfm.experiments.exp_color
==================================
Color-domain image transfer experiments (E5 in the NeurIPS experiment plan).

Design principle
----------------
The reference coupling π encodes a deterministic color-channel transformation
applied to images of class A.  The new source μ' consists of images of class B
(a *different* class, hence different support) colored in a *different* color
scheme, and the new target ν' is the corresponding color-shifted class-B images.

This setup deliberately makes the pretrained forward flow inapplicable to the
new marginals: the source and target color distributions differ completely from
the reference.  DualWCFM must adapt the coupling via dual-potential reweighting.

Experiments
-----------
mnist_rgb_simple  – MNIST digit 3 (red→blue) transferred to digit 7 (green→purple)
mnist_rgb_complex – MNIST digit 3 (red→blue) transferred to digit 7 (yellow→cyan)
                    (trickier: orthogonal hue shift instead of complementary)
cifar_hue_simple  – CIFAR-10 automobile (+0.3 hue) transferred to airplane (+0.3 hue)
cifar_hue_medium  – CIFAR-10 automobile (+0.3 hue) transferred to ship  (+0.3 hue)

Each experiment has a *standard* and *perturbed* variant.  In the perturbed
variant the new target receives an additional +10° rotation so that the true
target marginal is slightly off from the reference-coupling target.

UNet architecture
-----------------
All experiments use 3-channel (C=3) inputs.  The ``BidirectionalUnetStrategy``
from ``exp_image`` is reused — it reads ``image_shape = (3, H, W)`` from the
model config and creates UNets with the correct number of input channels.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF

from dualwcfm.config import FeedForwardConfig, TrainingConfig, VisualizeConfig
from dualwcfm.distributions.coupling import FunctionalCouplingDistribution, MarginalDistribution
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.experiments.base import ExperimentSpec, get_device
from dualwcfm.experiments.exp_image import BidirectionalUnetStrategy, UNetWrapper


# ---------------------------------------------------------------------------
# Color-mask helpers
# ---------------------------------------------------------------------------

def _apply_color_mask(images: torch.Tensor, rgb_weights: Tuple[float, float, float]) -> torch.Tensor:
    """Apply an RGB color mask to a grayscale (1-channel) image tensor.

    Parameters
    ----------
    images:
        Tensor of shape (N, 1, H, W) with values in [-1, 1].
    rgb_weights:
        (r, g, b) multipliers in [0, 1].  For example, (1, 0, 0) makes the image
        all-red and (1, 0, 1) makes it purple (red + blue channel).

    Returns
    -------
    Tensor of shape (N, 3, H, W) with values in [-1, 1].
    """
    r, g, b = rgb_weights
    # Replicate grayscale to 3 channels then zero-out the unwanted ones.
    imgs_3c = images.expand(-1, 3, -1, -1).clone()
    imgs_3c[:, 0, :, :] *= r
    imgs_3c[:, 1, :, :] *= g
    imgs_3c[:, 2, :, :] *= b
    return imgs_3c


def _apply_hue_rotation(images: torch.Tensor, hue_factor: float) -> torch.Tensor:
    """Apply a HSV hue rotation to a batch of RGB images.

    Parameters
    ----------
    images:
        Tensor of shape (N, 3, H, W) with values in [-1, 1].
    hue_factor:
        Float in [-0.5, 0.5] for ``TF.adjust_hue``.

    Returns
    -------
    Tensor of same shape, still in [-1, 1].
    """
    if hue_factor == 0.0:
        return images
    # TF.adjust_hue expects [0, 1]
    imgs_01 = (images.clamp(-1, 1) + 1) / 2
    result = torch.stack([TF.adjust_hue(img, hue_factor) for img in imgs_01])
    return result * 2 - 1  # back to [-1, 1]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_images(dataset_cls, cls: int, n: int, device: torch.device,
                 root: str, image_size: int = 32) -> torch.Tensor:
    """Load up to *n* normalised images of class *cls* from *dataset_cls*.

    Images are resized to (*image_size* × *image_size*) and normalised to [−1, 1].
    """
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = dataset_cls(root=root, train=True, download=True, transform=transform)
    indices = [i for i, (_, label) in enumerate(dataset) if label == cls][:n]
    return torch.stack([dataset[i][0] for i in indices]).to(device)


def _load_rgb_images(dataset_cls, cls: int, n: int, device: torch.device,
                     root: str, image_size: int = 32) -> torch.Tensor:
    """Load up to *n* normalised RGB images of class *cls* from *dataset_cls*.

    Images are resized to (*image_size* × *image_size*) and normalised to [−1, 1].
    Grayscale images are converted to 3-channel RGB.
    """
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = dataset_cls(root=root, train=True, download=True, transform=transform)
    indices = [i for i, (_, label) in enumerate(dataset) if label == cls][:n]
    imgs = torch.stack([dataset[i][0] for i in indices]).to(device)
    # Expand 1-channel (MNIST) to 3-channel.
    if imgs.shape[1] == 1:
        imgs = imgs.expand(-1, 3, -1, -1).clone()
    return imgs


# ---------------------------------------------------------------------------
# Coupling / distribution factories
# ---------------------------------------------------------------------------

def _make_color_coupling(
    images: torch.Tensor,
    src_rgb: Tuple[float, float, float],
    tgt_rgb: Tuple[float, float, float],
    rotation_degrees: float = 0.0,
) -> FunctionalCouplingDistribution:
    """Build a coupling (color_src(x), color_tgt(x)) from a fixed image set.

    Optionally applies an additional rotation to the target image.
    """
    def _sample(k: int):
        idx = torch.randint(len(images), (k,))
        x   = _apply_color_mask(images[idx], src_rgb)
        y   = _apply_color_mask(images[idx], tgt_rgb)
        if rotation_degrees != 0.0:
            y = TF.rotate(y, rotation_degrees, fill=-1.0)
        return x, y

    return FunctionalCouplingDistribution(_sample)


def _make_color_distributions(
    images: torch.Tensor,
    src_rgb: Tuple[float, float, float],
    tgt_rgb: Tuple[float, float, float],
    rotation_degrees: float = 0.0,
    extra_target_rotation: float = 0.0,
) -> Tuple[MarginalDistribution, MarginalDistribution]:
    """Build (source_dist, target_dist) with the given color masks."""
    def _src(k: int):
        idx = torch.randint(len(images), (k,))
        return _apply_color_mask(images[idx], src_rgb)

    def _tgt(k: int):
        idx = torch.randint(len(images), (k,))
        y = _apply_color_mask(images[idx], tgt_rgb)
        if rotation_degrees != 0.0:
            y = TF.rotate(y, rotation_degrees, fill=-1.0)
        if extra_target_rotation != 0.0:
            y = TF.rotate(y, extra_target_rotation, fill=-1.0)
        return y

    return MarginalDistribution(_src), MarginalDistribution(_tgt)


def _make_hue_coupling(
    images: torch.Tensor,
    hue_factor: float,
) -> FunctionalCouplingDistribution:
    """Build a coupling (x, hue_rotate(x)) from a fixed image set."""
    def _sample(k: int):
        idx = torch.randint(len(images), (k,))
        x   = images[idx]
        y   = _apply_hue_rotation(x, hue_factor)
        return x, y

    return FunctionalCouplingDistribution(_sample)


def _make_hue_distributions(
    images: torch.Tensor,
    hue_factor: float,
    extra_target_rotation: float = 0.0,
) -> Tuple[MarginalDistribution, MarginalDistribution]:
    """Build (source_dist, hue-rotated target_dist)."""
    def _src(k: int):
        idx = torch.randint(len(images), (k,))
        return images[idx]

    def _tgt(k: int):
        idx = torch.randint(len(images), (k,))
        y   = _apply_hue_rotation(images[idx], hue_factor)
        if extra_target_rotation != 0.0:
            y = TF.rotate(y, extra_target_rotation, fill=-1.0)
        return y

    return MarginalDistribution(_src), MarginalDistribution(_tgt)


# ---------------------------------------------------------------------------
# Default hyperparameter configs
# ---------------------------------------------------------------------------

def _default_train_cfg_mnist(device: torch.device) -> TrainingConfig:
    return TrainingConfig(
        device                   = device,
        dataset_size             = 20_000,
        marginal_epochs          = 20_000,
        pretrain_epochs          = 25_000,
        dual_epochs              = 100,
        cfm_epochs               = 500,
        advance_steps            = 30,
        iterations               = 40,
        num_rectified_flow       = 1,
        scale_factor             = 1.0,
        batch_size               = 128,
        use_disk_coupling        = True,
        coupling_chunk_size      = 2_048,
        save_coupling_checkpoints= False,
        save_model_checkpoints   = False,
    )


def _default_train_cfg_cifar(device: torch.device) -> TrainingConfig:
    return TrainingConfig(
        device                   = device,
        dataset_size             = 50_000,
        marginal_epochs          = 30_000,
        pretrain_epochs          = 50_000,
        dual_epochs              = 200,
        cfm_epochs               = 500,
        advance_steps            = 30,
        iterations               = 50,
        num_rectified_flow       = 1,
        scale_factor             = 1.0,
        batch_size               = 64,
        use_disk_coupling        = True,
        coupling_chunk_size      = 2_048,
        save_coupling_checkpoints= False,
        save_model_checkpoints   = False,
    )


def _default_method_cfg() -> MixtureConfig:
    return MixtureConfig(
        sigma           = 0.05,
        sigma2          = 0.05,
        loss            = "weighted",
        l2_reg          = 0,
        lip_reg         = 0,
        mean_reg        = 0,
        use_noisy_field = False,
        particle_ratio  = 0.25,
        lr_cosine_decay = False,
        lr_min_factor   = 0.5,
    )


# ---------------------------------------------------------------------------
# MNIST-RGB color channel experiments
# ---------------------------------------------------------------------------

# Color mask definitions (r, g, b) ∈ [0,1]
_RED    = (1.0, 0.0, 0.0)
_BLUE   = (0.0, 0.0, 1.0)
_GREEN  = (0.0, 1.0, 0.0)
_PURPLE = (1.0, 0.0, 1.0)
_YELLOW = (1.0, 1.0, 0.0)
_CYAN   = (0.0, 1.0, 1.0)


def _mnist_rgb_simple(perturbed: bool, output_root: str) -> ExperimentSpec:
    """MNIST: digit 3 (red→blue) reference; digit 7 (green→purple) new marginals.

    Reference coupling
        (red_digit3, blue_digit3) — deterministic color shift.
    New marginals
        Source = green_digit7,  Target = purple_digit7
        (+ 10° rotation if perturbed).
    Why pretrained model fails
        Reference was trained on red source; new source is green. The pretrained
        forward flow maps red→blue but outputs are blue when the input is green,
        which does not match the true purple target.
    """
    device    = get_device()
    data_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    from torchvision import datasets
    ref_images = _load_rgb_images(datasets.MNIST, cls=3, n=2_000, device=device, root=data_root)
    new_images = _load_rgb_images(datasets.MNIST, cls=7, n=2_000, device=device, root=data_root)
    C, H, W    = ref_images.shape[1:]   # C=3

    coupling         = _make_color_coupling(ref_images, src_rgb=_RED, tgt_rgb=_BLUE)
    new_src, new_tgt = _make_color_distributions(
        new_images,
        src_rgb               = _GREEN,
        tgt_rgb               = _PURPLE,
        extra_target_rotation = 10.0 if perturbed else 0.0,
    )

    train_cfg              = _default_train_cfg_mnist(device)
    train_cfg.image_shape  = (C, H, W)
    method_cfg             = _default_method_cfg()
    model_cfg              = FeedForwardConfig(
        dimension          = C,
        hidden_size        = 32,
        image_shape        = (C, H, W),
        dual_learning_rate = 1e-4,
        mu_learning_rate   = 2e-4,
        nu_learning_rate   = 2e-4,
        flow_learning_rate = 2e-4,
    )
    vis_cfg = VisualizeConfig(
        show_each           = 1,
        n_trajectories      = 6,
        trajectory_stride   = 0,
        show_each_marginal  = 1_000,
        show_each_pretrain  = 1_000,
    )

    return ExperimentSpec(
        name             = f"color_mnist_rgb_simple{'_perturbed' if perturbed else ''}",
        mode             = "image",
        initial_coupling = coupling,
        new_source       = new_src,
        new_target       = new_tgt,
        transport_map    = None,
        train_cfg        = train_cfg,
        model_cfg        = model_cfg,
        vis_cfg          = vis_cfg,
        method_cfg       = method_cfg,
        output_root      = output_root,
    )


def _mnist_rgb_complex(perturbed: bool, output_root: str) -> ExperimentSpec:
    """MNIST: digit 3 (red→blue) reference; digit 7 (yellow→cyan) new marginals.

    The yellow→cyan shift is orthogonal to the red→blue reference, making
    generalisation harder: the model must identify the abstract "complementary
    color channel" rule and apply it to an entirely different hue pair.
    """
    device    = get_device()
    data_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    from torchvision import datasets
    ref_images = _load_rgb_images(datasets.MNIST, cls=3, n=2_000, device=device, root=data_root)
    new_images = _load_rgb_images(datasets.MNIST, cls=7, n=2_000, device=device, root=data_root)
    C, H, W    = ref_images.shape[1:]

    coupling         = _make_color_coupling(ref_images, src_rgb=_RED, tgt_rgb=_BLUE)
    new_src, new_tgt = _make_color_distributions(
        new_images,
        src_rgb               = _YELLOW,
        tgt_rgb               = _CYAN,
        extra_target_rotation = 10.0 if perturbed else 0.0,
    )

    train_cfg              = _default_train_cfg_mnist(device)
    train_cfg.image_shape  = (C, H, W)
    method_cfg             = _default_method_cfg()
    model_cfg              = FeedForwardConfig(
        dimension          = C,
        hidden_size        = 32,
        image_shape        = (C, H, W),
        dual_learning_rate = 1e-4,
        mu_learning_rate   = 2e-4,
        nu_learning_rate   = 2e-4,
        flow_learning_rate = 2e-4,
    )
    vis_cfg = VisualizeConfig(show_each=1, n_trajectories=6, trajectory_stride=0)

    return ExperimentSpec(
        name             = f"color_mnist_rgb_complex{'_perturbed' if perturbed else ''}",
        mode             = "image",
        initial_coupling = coupling,
        new_source       = new_src,
        new_target       = new_tgt,
        transport_map    = None,
        train_cfg        = train_cfg,
        model_cfg        = model_cfg,
        vis_cfg          = vis_cfg,
        method_cfg       = method_cfg,
        output_root      = output_root,
    )


# ---------------------------------------------------------------------------
# CIFAR-10 hue-rotation experiments
# ---------------------------------------------------------------------------

def _cifar_hue(new_class: int, new_class_name: str, perturbed: bool,
               output_root: str, hue_factor: float = 0.3) -> ExperimentSpec:
    """CIFAR-10: automobile (+hue) reference; *new_class* (+hue) new marginals.

    Reference coupling
        (automobile, hue_rotated_automobile) — deterministic hue shift.
    New marginals
        Source = new_class images (original color),
        Target = new_class images with same +hue rotation
        (+ 10° spatial rotation if perturbed).
    Transport law
        "Shift hue by +hue_factor" — same law as reference, new support.
    Why interesting
        CIFAR-10 classes have very different intrinsic color distributions
        (automobiles are red-heavy, airplanes are blue-heavy, ships are blue-grey).
        The reference coupling is trained on one color profile; the new marginals
        have a completely different profile.
    """
    device    = get_device()
    data_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    from torchvision import datasets, transforms
    # CIFAR-10 is already RGB; use standard normalisation.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def _load_cifar_class(cls: int, n: int) -> torch.Tensor:
        ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        indices = [i for i, (_, label) in enumerate(ds) if label == cls][:n]
        return torch.stack([ds[i][0] for i in indices]).to(device)

    # CIFAR-10 class indices: automobile=1, airplane=0, ship=8, truck=9
    ref_images = _load_cifar_class(cls=1, n=5_000)   # automobile
    new_images = _load_cifar_class(cls=new_class, n=5_000)
    C, H, W    = ref_images.shape[1:]   # (3, 32, 32)

    coupling         = _make_hue_coupling(ref_images, hue_factor=hue_factor)
    new_src, new_tgt = _make_hue_distributions(
        new_images,
        hue_factor            = hue_factor,
        extra_target_rotation = 10.0 if perturbed else 0.0,
    )

    train_cfg              = _default_train_cfg_cifar(device)
    train_cfg.image_shape  = (C, H, W)
    method_cfg             = _default_method_cfg()
    model_cfg              = FeedForwardConfig(
        dimension          = C,
        hidden_size        = 64,
        image_shape        = (C, H, W),
        dual_learning_rate = 1e-4,
        mu_learning_rate   = 2e-4,
        nu_learning_rate   = 2e-4,
        flow_learning_rate = 2e-4,
    )
    vis_cfg = VisualizeConfig(show_each=1, n_trajectories=6, trajectory_stride=0)

    suffix = f"_perturbed" if perturbed else ""
    return ExperimentSpec(
        name             = f"color_cifar_hue_automobile2{new_class_name}{suffix}",
        mode             = "image",
        initial_coupling = coupling,
        new_source       = new_src,
        new_target       = new_tgt,
        transport_map    = None,
        train_cfg        = train_cfg,
        model_cfg        = model_cfg,
        vis_cfg          = vis_cfg,
        method_cfg       = method_cfg,
        output_root      = output_root,
    )


# ---------------------------------------------------------------------------
# Public catalogue
# ---------------------------------------------------------------------------

def make_color_experiments(output_root: str = "results") -> List[ExperimentSpec]:
    """Return all color-domain transfer experiment specs.

    Requires torchvision.  Returns [] with a warning if not installed.
    """
    try:
        import torchvision  # noqa: F401
    except ImportError:
        import warnings
        warnings.warn("torchvision not found – skipping color experiments.", stacklevel=2)
        return []

    return [
        # MNIST-RGB: simple color-channel shift
        _mnist_rgb_simple(False, output_root),
        _mnist_rgb_simple(True,  output_root),

        # MNIST-RGB: complex (orthogonal hue pair)
        _mnist_rgb_complex(False, output_root),
        _mnist_rgb_complex(True,  output_root),

        # CIFAR-10 hue rotation: automobile → airplane
        _cifar_hue(new_class=0, new_class_name="airplane", perturbed=False, output_root=output_root),
        _cifar_hue(new_class=0, new_class_name="airplane", perturbed=True,  output_root=output_root),

        # CIFAR-10 hue rotation: automobile → ship
        _cifar_hue(new_class=8, new_class_name="ship", perturbed=False, output_root=output_root),
        _cifar_hue(new_class=8, new_class_name="ship", perturbed=True,  output_root=output_root),
    ]
