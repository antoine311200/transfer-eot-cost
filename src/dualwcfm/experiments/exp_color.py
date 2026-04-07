"""dualwcfm.experiments.exp_color
==================================
Color-domain image transfer experiments (E5 in the NeurIPS experiment plan).

Design principle
----------------
All experiments share the **same underlying transport map**

    T(r, g, b) = (g, b, r)          # cyclic left-shift of RGB channels

applied per-pixel to the (normalised) 3-channel image.  Because T is a fixed
linear transformation, the quadratic transport cost is the same for every pair
on the same orbit:

    c(x, T(x)) = ‖x − T(x)‖² = 2v²   per pixel (v = grayscale intensity)

This means reference and new marginals present the *same difficulty* to the
optimal transport problem; only the image content (digit class / CIFAR class)
and the specific colour pair differ, creating a genuine domain shift while
keeping the transport cost controlled.

The T-cyclic orbits used:
    RED   (1,0,0) → BLUE   (0,0,1) → GREEN  (0,1,0) → RED    [reference orbit]
    PURPLE(1,0,1) → CYAN   (0,1,1) → YELLOW (1,1,0) → PURPLE [new-marginal orbit]

Experiments
-----------
mnist_rgb_simple      – digit 3 (RED→BLUE) reference; digit 7 (PURPLE→CYAN) new marginals
                        T(PURPLE) = CYAN, same cost 2v²
mnist_rgb_complex     – digit 3 (RED→BLUE) reference; digit 7 (YELLOW→PURPLE) new marginals
                        T(YELLOW) = PURPLE, same cost 2v²; harder because the source
                        two-channel structure (R+G) is more distant from reference (R only)
mnist_rgb_flip_simple – digit 3 (RED→flip(BLUE)) reference; digit 7 (PURPLE→flip(CYAN)) new
                        T = hflip ∘ T_cyclic; combines color shift with spatial flip
mnist_rgb_flip_complex– digit 3 (RED→flip(BLUE)) reference; digit 7 (YELLOW→flip(PURPLE)) new
                        Hardest MNIST variant: two-channel source + spatial flip mismatch
cifar_hue_simple      – CIFAR-10 automobile (hue+0.3) reference; airplane (hue+0.3) new
cifar_hue_medium      – CIFAR-10 automobile (hue+0.3) reference; ship    (hue+0.3) new

Standard variants: Supp(T(μ')) = Supp(ν') exactly.
Perturbed variants: the new target receives an additional +10° spatial rotation,
breaking the exact support match and probing robustness to slight distribution
mismatch — analogous to the perturbed variants in E2/E3.

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
    flip_target: bool = False,
) -> FunctionalCouplingDistribution:
    """Build a coupling (color_src(x), color_tgt(x)) from a fixed image set.

    Optionally applies a horizontal flip and/or rotation to the target image.
    """
    def _sample(k: int):
        idx = torch.randint(len(images), (k,))
        x   = _apply_color_mask(images[idx], src_rgb)
        y   = _apply_color_mask(images[idx], tgt_rgb)
        if flip_target:
            y = TF.hflip(y)
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
    flip_target: bool = False,
) -> Tuple[MarginalDistribution, MarginalDistribution]:
    """Build (source_dist, target_dist) with the given color masks."""
    def _src(k: int):
        idx = torch.randint(len(images), (k,))
        return _apply_color_mask(images[idx], src_rgb)

    def _tgt(k: int):
        idx = torch.randint(len(images), (k,))
        y = _apply_color_mask(images[idx], tgt_rgb)
        if flip_target:
            y = TF.hflip(y)
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
        dataset_size             = 2_000,
        marginal_epochs          = 10_000,
        pretrain_epochs          = 5_000,
        dual_epochs              = 100,
        cfm_epochs               = 1000,
        advance_steps            = 200,
        iterations               = 31,
        num_rectified_flow       = 0,
        scale_factor             = 1.0,
        batch_size               = 128,
        use_disk_coupling        = False,
        coupling_chunk_size      = 256,
        save_coupling_checkpoints= False,
        save_model_checkpoints   = False,
    )


def _default_train_cfg_cifar(device: torch.device) -> TrainingConfig:
    return TrainingConfig(
        device                   = device,
        dataset_size             = 50_000,
        marginal_epochs          = 5_000,
        pretrain_epochs          = 1_000,
        dual_epochs              = 100,
        cfm_epochs               = 500,
        advance_steps            = 200,
        iterations               = 31,
        num_rectified_flow       = 0,
        scale_factor             = 1.0,
        batch_size               = 64,
        use_disk_coupling        = True,
        coupling_chunk_size      = 256,
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
        use_noisy_field = False,
        particle_ratio  = 0.25,
        lr_cosine_decay = False,
        lr_min_factor   = 0.5,
        dual_warmup_epochs = 1000,
        use_ot_pretrain=True
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


# ---------------------------------------------------------------------------
# Shared transport map  T(r,g,b) = (g,b,r)  — cyclic left-shift of channels
# ---------------------------------------------------------------------------

def _cyclic_color_shift(x: torch.Tensor) -> torch.Tensor:
    """T(r,g,b) = (g,b,r): cyclic left-shift of RGB channels.

    Works on batches of shape (N, 3, H, W).
    Orbits: RED→BLUE→GREEN→RED  and  PURPLE→CYAN→YELLOW→PURPLE.
    Cost per pixel: ‖x − T(x)‖² = 2v² (v = pixel intensity).
    """
    return torch.stack([x[:, 1], x[:, 2], x[:, 0]], dim=1)


def _cyclic_color_shift_inv(y: torch.Tensor) -> torch.Tensor:
    """T⁻¹(r,g,b) = (b,r,g): inverse cyclic shift (right-shift).

    T⁻¹(BLUE)=(RED), T⁻¹(CYAN)=(PURPLE), T⁻¹(PURPLE)=(YELLOW).
    """
    return torch.stack([y[:, 2], y[:, 0], y[:, 1]], dim=1)


def _flip_cyclic(x: torch.Tensor) -> torch.Tensor:
    """T_flip(x) = hflip(T_cyclic(x)): cyclic channel shift then horizontal flip.

    Combines the colour transformation with a spatial reflection.
    Orbits (colour component only):
        RED→BLUE→GREEN→RED  and  PURPLE→CYAN→YELLOW→PURPLE
    The flip is applied after the channel permutation, so the target is both
    colour-shifted AND mirrored left–right.
    """
    return TF.hflip(_cyclic_color_shift(x))


def _flip_cyclic_inv(y: torch.Tensor) -> torch.Tensor:
    """T_flip⁻¹(y) = T_cyclic⁻¹(hflip(y)): inverse of hflip ∘ T_cyclic.

    Since hflip is its own inverse: T_flip⁻¹ = T_cyclic⁻¹ ∘ hflip.
    """
    return _cyclic_color_shift_inv(TF.hflip(y))


def _mnist_rgb_simple(perturbed: bool, output_root: str) -> ExperimentSpec:
    """MNIST: digit 3 (RED→BLUE) reference; digit 7 (PURPLE→CYAN) new marginals.

    Transport map  T(r,g,b) = (g,b,r):
        T(RED)    = T(1,0,0) = (0,0,1) = BLUE   [reference]
        T(PURPLE) = T(1,0,1) = (0,1,1) = CYAN   [new marginals]

    Both pairs share the same quadratic transport cost: 2v² per pixel,
    so the transport *difficulty* is identical between reference and new marginals.
    Supp(T(purple_digit7)) = Supp(cyan_digit7) exactly in the standard variant;
    the perturbed variant adds a +10° spatial rotation on the cyan target to
    break the exact support match.

    Why the pretrained model fails
        The model was trained on RED sources; PURPLE inputs have an extra B channel
        that the model has never seen.  It outputs BLUE-ish images, not CYAN.
        Dual adaptation must re-weight the coupling to discover that purple digit 7
        should be paired with cyan digit 7 of the same image content.
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
        src_rgb               = _PURPLE,
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
        use_conditional_bridge=True,
    )
    vis_cfg = VisualizeConfig(
        show_each           = 1,
        n_trajectories      = 6,
        trajectory_stride   = 0,
        show_each_marginal  = 1_000,
        show_each_pretrain  = 1_000,
    )

    return ExperimentSpec(
        name                  = f"color_mnist_rgb_simple{'_perturbed' if perturbed else ''}",
        mode                  = "image",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        transport_map         = _cyclic_color_shift,
        inverse_transport_map = _cyclic_color_shift_inv,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


def _mnist_rgb_complex(perturbed: bool, output_root: str) -> ExperimentSpec:
    """MNIST: digit 3 (RED→BLUE) reference; digit 7 (YELLOW→PURPLE) new marginals.

    Transport map  T(r,g,b) = (g,b,r):
        T(RED)    = T(1,0,0) = (0,0,1) = BLUE   [reference]
        T(YELLOW) = T(1,1,0) = (1,0,1) = PURPLE [new marginals]

    Same quadratic cost 2v² per pixel as the reference and the simple experiment.

    Harder than 'simple' (PURPLE→CYAN) because the YELLOW source has two active
    channels (R+G) while the reference source has only one (R).  The G channel is
    entirely absent from the reference support, so the pretrained flow has never
    seen it and the mismatch is more severe: the model outputs BLUE (ignoring G),
    whereas the true target is PURPLE (R+B active).
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
        use_conditional_bridge=True,
    )
    vis_cfg = VisualizeConfig(show_each=1, n_trajectories=6, trajectory_stride=0)

    return ExperimentSpec(
        name                  = f"color_mnist_rgb_complex{'_perturbed' if perturbed else ''}",
        mode                  = "image",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        transport_map         = _cyclic_color_shift,
        inverse_transport_map = _cyclic_color_shift_inv,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


# ---------------------------------------------------------------------------
# MNIST-RGB + horizontal flip experiments
# ---------------------------------------------------------------------------

def _mnist_rgb_flip_simple(perturbed: bool, output_root: str) -> ExperimentSpec:
    """MNIST: digit 3 (RED→flip(BLUE)) reference; digit 7 (PURPLE→flip(CYAN)) new marginals.

    Transport map  T_flip = hflip ∘ T_cyclic:
        T_flip(RED)    = hflip(BLUE)   [reference]
        T_flip(PURPLE) = hflip(CYAN)   [new marginals]

    Same colour cost 2v² per pixel as the non-flip variants; the horizontal flip
    adds a *spatial* component to the transport, making the pretrained model
    doubly mismatched: wrong colour (PURPLE≠RED) and wrong orientation (flipped).
    Dual adaptation must simultaneously identify the colour-shift rule and the
    spatial reflection from the reference coupling.

    Supp(T_flip(purple_digit7)) = Supp(hflip(cyan_digit7)) exactly in standard variant.
    Perturbed: an additional +10° rotation on the target, on top of the flip.
    """
    device    = get_device()
    data_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    from torchvision import datasets
    ref_images = _load_rgb_images(datasets.MNIST, cls=3, n=2_000, device=device, root=data_root)
    new_images = _load_rgb_images(datasets.MNIST, cls=7, n=2_000, device=device, root=data_root)
    C, H, W    = ref_images.shape[1:]

    coupling         = _make_color_coupling(ref_images, src_rgb=_RED, tgt_rgb=_BLUE,
                                            flip_target=True)
    new_src, new_tgt = _make_color_distributions(
        new_images,
        src_rgb               = _PURPLE,
        tgt_rgb               = _CYAN,
        flip_target           = True,
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
        use_conditional_bridge=True,
    )
    vis_cfg = VisualizeConfig(
        show_each           = 1,
        n_trajectories      = 6,
        trajectory_stride   = 0,
        show_each_marginal  = 1_000,
        show_each_pretrain  = 1_000,
    )

    return ExperimentSpec(
        name                  = f"color_mnist_rgb_flip_simple{'_perturbed' if perturbed else ''}",
        mode                  = "image",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        transport_map         = _flip_cyclic,
        inverse_transport_map = _flip_cyclic_inv,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


def _mnist_rgb_flip_complex(perturbed: bool, output_root: str) -> ExperimentSpec:
    """MNIST: digit 3 (RED→flip(BLUE)) reference; digit 7 (YELLOW→flip(PURPLE)) new marginals.

    Transport map  T_flip = hflip ∘ T_cyclic:
        T_flip(RED)    = hflip(BLUE)   [reference]
        T_flip(YELLOW) = hflip(PURPLE) [new marginals]

    Hardest MNIST variant: combines the two-channel source mismatch of
    'mnist_rgb_complex' (YELLOW has R+G while reference source is R only) with
    the spatial flip of 'mnist_rgb_flip_simple'.  The pretrained model has never
    seen a two-channel source nor flipped targets, so the initial transport error
    is maximal.

    Supp(T_flip(yellow_digit7)) = Supp(hflip(purple_digit7)) exactly in standard variant.
    Perturbed: an additional +10° rotation on the target, on top of the flip.
    """
    device    = get_device()
    data_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    from torchvision import datasets
    ref_images = _load_rgb_images(datasets.MNIST, cls=3, n=2_000, device=device, root=data_root)
    new_images = _load_rgb_images(datasets.MNIST, cls=7, n=2_000, device=device, root=data_root)
    C, H, W    = ref_images.shape[1:]

    coupling         = _make_color_coupling(ref_images, src_rgb=_RED, tgt_rgb=_BLUE,
                                            flip_target=True)
    new_src, new_tgt = _make_color_distributions(
        new_images,
        src_rgb               = _YELLOW,
        tgt_rgb               = _PURPLE,
        flip_target           = True,
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
        use_conditional_bridge=True,
    )
    vis_cfg = VisualizeConfig(show_each=1, n_trajectories=6, trajectory_stride=0)

    return ExperimentSpec(
        name                  = f"color_mnist_rgb_flip_complex{'_perturbed' if perturbed else ''}",
        mode                  = "image",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        transport_map         = _flip_cyclic,
        inverse_transport_map = _flip_cyclic_inv,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        method_cfg            = method_cfg,
        output_root           = output_root,
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
        use_conditional_bridge=True,
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

        # MNIST-RGB: complex (two-channel source)
        _mnist_rgb_complex(False, output_root),
        _mnist_rgb_complex(True,  output_root),

        # MNIST-RGB + flip: simple (color shift + horizontal flip)
        _mnist_rgb_flip_simple(False, output_root),
        _mnist_rgb_flip_simple(True,  output_root),

        # MNIST-RGB + flip: complex (two-channel source + horizontal flip)
        _mnist_rgb_flip_complex(False, output_root),
        _mnist_rgb_flip_complex(True,  output_root),

        # CIFAR-10 hue rotation: automobile → airplane
        _cifar_hue(new_class=0, new_class_name="airplane", perturbed=False, output_root=output_root),
        _cifar_hue(new_class=0, new_class_name="airplane", perturbed=True,  output_root=output_root),

        # CIFAR-10 hue rotation: automobile → ship
        _cifar_hue(new_class=8, new_class_name="ship", perturbed=False, output_root=output_root),
        _cifar_hue(new_class=8, new_class_name="ship", perturbed=True,  output_root=output_root),
    ]
