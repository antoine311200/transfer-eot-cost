"""dualwcfm.experiments.exp_image
==================================
Image experiment catalogue (MNIST / FashionMNIST).

Design principle
----------------
The reference coupling π encodes "horizontal flip + optional rotation" applied to
images of class A.  The new source μ' consists of images of class B (a *different*
class, hence different support) and the new target ν' is the same flip/rotation
applied to class B — testing whether the model generalises the transport dynamic
to an unseen image distribution.

Experiments
-----------
mnist_simple    – MNIST digit 3 → digit 7,  transport = horizontal flip
fashion_medium  – FashionMNIST shirt → sneaker, transport = flip + 10° rotation

Each experiment exists in a *standard* and *perturbed* variant.  In the perturbed
variant the new target ν' is additionally rotated by 10°, introducing a systematic
mismatch between the reference and new target marginals.

UNet architecture
-----------------
Image experiments use a UNet instead of an MLP.  Two thin wrapper classes adapt
the UNet calling conventions to match those expected by the strategy/trainer:

  UNetWrapper      – flow models and bridges: (t, x[, cond]) → velocity
  DualUNetWrapper  – dual potentials f, g, h: (x,) → scalar  (N, 1, 1, 1)

BidirectionalUnetStrategy overrides ``_create_models`` to build UNets for all
roles (bridges, duals, forward/backward flow).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF

from dualwcfm.config import FeedForwardConfig, TrainingConfig, VisualizeConfig
from dualwcfm.distributions.coupling import FunctionalCouplingDistribution, MarginalDistribution
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.models.unet import UNetModel
from dualwcfm.strategy import BidirectionalStrategy
from dualwcfm.experiments.base import ExperimentSpec, get_device


# ---------------------------------------------------------------------------
# UNet wrappers
# ---------------------------------------------------------------------------

class UNetWrapper(torch.nn.Module):
    """Adapts UNetModel to the (t, x[, condition]) calling convention.

    The trainer / flow-matching code calls flow models as::

        velocity = model(t, x, condition)   # CFM  – channels doubled
        velocity = model(t, x)              # bridges / no condition

    UNetModel expects ``(timesteps: Tensor[N], x: Tensor[N,C,H,W])``.
    When a condition is provided, it is concatenated with *x* along the
    channel axis before passing to the UNet.
    """

    def __init__(self, model: UNetModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor = None,
        condition: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Flatten time to 1-D as required by the UNet timestep embedding.
        t_flat = t.view(-1)

        if x is not None:
            inp = torch.cat([x, condition], dim=1) if condition is not None else x
            return self.model(t_flat, inp)
        else:
            # Fallback: called with a single positional arg (image in t's slot).
            # Use zero timestep – happens only in some diagnostic code paths.
            timesteps = torch.zeros(t.shape[0], device=t.device, dtype=torch.float32)
            return self.model(timesteps, t)


class DualUNetWrapper(torch.nn.Module):
    """Adapts a ``stop_middle=True`` UNetModel for dual potential functions.

    Dual potentials are time-independent scalars.  This wrapper:
    - Always passes a zero timestep to the UNet.
    - Returns shape ``(N, 1, 1, 1)`` for broadcast-safe arithmetic with image tensors.
    """

    def __init__(self, model: UNetModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        t = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        out = self.model(t, x)  # (N, C) after Flatten when stop_middle=True
        # Collapse to (N, 1, 1, 1) regardless of input channels
        return out.view(x.shape[0], -1).mean(dim=1).view(x.shape[0], 1, 1, 1)


# ---------------------------------------------------------------------------
# BidirectionalUnetStrategy
# ---------------------------------------------------------------------------

#: UNet channel multiplier schedule used for all image experiments.
_CHANNEL_MULT = (1, 2, 3, 4)
#: Base number of UNet channels.
_NUM_CHANNELS = 32
#: Number of residual blocks per resolution level.
_NUM_RES_BLOCKS = 1


class BidirectionalUnetStrategy(BidirectionalStrategy):
    """BidirectionalStrategy that creates UNet models for all roles.

    Overrides ``_create_models`` to build UNet-based bridges, dual potentials,
    and forward/backward flow models for a grayscale 32×32 image setup.

    The image shape is read from ``self.config.image_shape`` which must be a
    ``(C, H, W)`` tuple set by the caller before ``_create_models`` is invoked.
    """

    def _create_models(self, dim: int, hidden_size: int) -> None:  # noqa: D102
        self._device = None  # Reset so the next to_device() call moves the new models.
        C, H, W = self.config.image_shape
        num_channels   = hidden_size if hidden_size > 0 else _NUM_CHANNELS
        num_res_blocks = getattr(self.config, "num_res_blocks", _NUM_RES_BLOCKS)

        def _make_unet(in_channels: int, stop_middle: bool = False) -> UNetModel:
            return UNetModel(
                dim            = (in_channels, H, W),
                channel_mult   = _CHANNEL_MULT,
                num_channels   = num_channels,
                num_res_blocks = num_res_blocks,
                stop_middle    = stop_middle,
            )

        wrap  = self.wrapper   # UNetWrapper
        dwrap = DualUNetWrapper

        # Bridge models: (t, x) → velocity, single-channel input
        self.mu_bridge = wrap(_make_unet(C))
        self.nu_bridge = wrap(_make_unet(C))

        # Dual potentials: (x,) → scalar, stop_middle=True for global pool
        self.f_dual = dwrap(_make_unet(C,     stop_middle=True))
        self.g_dual = dwrap(_make_unet(C,     stop_middle=True))
        self.h_dual = dwrap(_make_unet(2 * C, stop_middle=True))

        # Flow models: (t, x, cond) → velocity, 2C input (x concat cond)
        self.forward_model          = wrap(_make_unet(2 * C))
        self.previous_forward_model = wrap(_make_unet(2 * C))
        self.backward_model         = wrap(_make_unet(2 * C))
        self.previous_backward_model= wrap(_make_unet(2 * C))

        # Frozen initial copies for reference
        self.initial_mu_bridge      = wrap(_make_unet(C))
        self.initial_nu_bridge      = wrap(_make_unet(C))
        self.initial_forward_model  = wrap(_make_unet(2 * C))
        self.initial_backward_model = wrap(_make_unet(2 * C))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_images(
    dataset_cls,
    cls: int,
    n: int,
    device: torch.device,
    root: str,
    image_size: int = 32,
) -> torch.Tensor:
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


def _make_flip_coupling(
    images: torch.Tensor,
    rotation_degrees: float = 0.0,
) -> FunctionalCouplingDistribution:
    """Build a coupling (x, flip(x)) from a fixed image set.

    Optionally applies a rotation on top of the horizontal flip.
    """
    def _sample(k: int):
        idx = torch.randint(len(images), (k,))
        x   = images[idx]
        y   = torch.flip(x, dims=[3])       # horizontal flip along width axis
        if rotation_degrees != 0.0:
            y = TF.rotate(y, rotation_degrees, fill=-1.0)
        return x, y

    return FunctionalCouplingDistribution(_sample)


def _make_flip_distributions(
    images: torch.Tensor,
    rotation_degrees: float = 0.0,
    extra_target_rotation: float = 0.0,
) -> Tuple[MarginalDistribution, MarginalDistribution]:
    """Build (source_dist, target_dist) from a fixed image set.

    Target = horizontal flip + *rotation_degrees* + *extra_target_rotation*.
    The extra rotation is used in perturbed experiments to shift the target
    marginal away from the reference coupling's target marginal.
    """
    def _src(k: int):
        idx = torch.randint(len(images), (k,))
        return images[idx]

    def _tgt(k: int):
        idx = torch.randint(len(images), (k,))
        y   = torch.flip(images[idx], dims=[3])
        if rotation_degrees != 0.0:
            y = TF.rotate(y, rotation_degrees, fill=-1.0)
        if extra_target_rotation != 0.0:
            y = TF.rotate(y, extra_target_rotation, fill=-1.0)
        return y

    return MarginalDistribution(_src), MarginalDistribution(_tgt)


# ---------------------------------------------------------------------------
# Default hyperparameter configs
# ---------------------------------------------------------------------------

def _default_train_cfg(device: torch.device) -> TrainingConfig:
    return TrainingConfig(
        device                   = device,
        dataset_size             = 2_000,
        marginal_epochs          = 30_000,
        pretrain_epochs          = 30_000,
        dual_epochs              = 100,
        cfm_epochs               = 1000,
        advance_steps            = 200,
        iterations               = 10,
        num_rectified_flow       = 2,
        scale_factor             = 1.0,
        batch_size               = 128,
        use_disk_coupling        = True,
        coupling_chunk_size      = 128,
        save_coupling_checkpoints= False,
        save_model_checkpoints   = False,
    )


def _default_method_cfg() -> MixtureConfig:
    return MixtureConfig(
        sigma           = 0.01,
        sigma2          = 0.01,
        loss            = "weighted",
        l2_reg          = 0,
        lip_reg         = 0,
        mean_reg        = 0,
        use_noisy_field = False,
        particle_ratio  = 0.1,
        lr_cosine_decay = False,
        lr_min_factor   = 1.0,
    )


# ---------------------------------------------------------------------------
# Experiment constructors
# ---------------------------------------------------------------------------

def _mnist_simple(perturbed: bool, output_root: str) -> ExperimentSpec:
    """MNIST digit 3 → digit 7, transport = horizontal flip.

    Reference coupling
        (x, flip(x)) for class-3 MNIST images.
    New marginals
        Class-7 MNIST images;  target = flip of class-7
        (+ 10° rotation if perturbed).
    Goal
        Generalise the "flip" dynamic to an unseen digit class.
    """
    device = get_device()
    data_root   = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError("torchvision is required for image experiments.") from exc

    ref_images = _load_images(datasets.MNIST, cls=3, n=2_000, device=device, root=data_root)
    new_images = _load_images(datasets.MNIST, cls=7, n=2_000, device=device, root=data_root)
    C, H, W    = ref_images.shape[1:]

    coupling      = _make_flip_coupling(ref_images, rotation_degrees=0.0)
    new_src, new_tgt = _make_flip_distributions(
        new_images,
        rotation_degrees       = 0.0,
        extra_target_rotation  = 10.0 if perturbed else 0.0,
    )

    train_cfg              = _default_train_cfg(device)
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
    vis_cfg                = VisualizeConfig(
        show_each           = 1,
        n_trajectories      = 6,
        trajectory_stride   = 0,
        show_each_marginal  = 1_000,
        show_each_pretrain  = 1_000,
    )

    # T = horizontal flip (self-inverse: flip(flip(x)) = x)
    def _flip_map(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-1])  # last dim = width for both 3D and 4D tensors

    return ExperimentSpec(
        name                  = f"mnist_simple{'_perturbed' if perturbed else ''}",
        mode                  = "image",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = _flip_map,
        inverse_transport_map = _flip_map,  # self-inverse
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


def _fashion_medium(perturbed: bool, output_root: str) -> ExperimentSpec:
    """FashionMNIST shirt → sneaker, transport = horizontal flip + 10° rotation.

    Reference coupling
        (x, rotate(flip(x), 10°)) for class-6 (shirt) FashionMNIST images.
    New marginals
        Class-7 (sneaker) FashionMNIST;  target = flip + 10° rotation of sneaker
        (+ extra 10° rotation if perturbed).
    Goal
        Generalise the "flip + rotation" dynamic across garment types.
    """
    device = get_device()
    data_root   = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError("torchvision is required for image experiments.") from exc

    ref_images = _load_images(datasets.FashionMNIST, cls=6, n=2_000, device=device, root=data_root)
    new_images = _load_images(datasets.FashionMNIST, cls=7, n=2_000, device=device, root=data_root)
    C, H, W    = ref_images.shape[1:]

    coupling         = _make_flip_coupling(ref_images, rotation_degrees=10.0)
    new_src, new_tgt = _make_flip_distributions(
        new_images,
        rotation_degrees       = 10.0,
        extra_target_rotation  = 10.0 if perturbed else 0.0,
    )

    train_cfg              = _default_train_cfg(device)
    train_cfg.image_shape  = (C, H, W)
    method_cfg             = _default_method_cfg()
    model_cfg              = FeedForwardConfig(
        dimension          = C,
        hidden_size        = 256,
        image_shape        = (C, H, W),
        dual_learning_rate = 1e-4,
        mu_learning_rate   = 5e-4,
        nu_learning_rate   = 5e-4,
        flow_learning_rate = 2e-4,
    )
    vis_cfg                = VisualizeConfig(show_each=1, n_trajectories=6, trajectory_stride=0)

    # T = flip then rotate(10°).  T⁻¹ = rotate(-10°) then flip.
    def _flip_rot10_map(x: torch.Tensor) -> torch.Tensor:
        y = torch.flip(x, dims=[-1])
        return TF.rotate(y, 10.0, fill=-1.0)

    def _flip_rot10_inv(y: torch.Tensor) -> torch.Tensor:
        x = TF.rotate(y, -10.0, fill=-1.0)
        return torch.flip(x, dims=[-1])

    return ExperimentSpec(
        name                  = f"fashion_medium{'_perturbed' if perturbed else ''}",
        mode                  = "image",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = _flip_rot10_map,
        inverse_transport_map = _flip_rot10_inv,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


# ---------------------------------------------------------------------------
# Public catalogue
# ---------------------------------------------------------------------------

def make_image_experiments_v2(output_root: str = "results") -> List[ExperimentSpec]:
    """Image experiments with improved hyperparameters for better DL convergence.

    Key changes vs v1
    -----------------
    - UNet: 64 base channels (was 32), 2 res blocks (was 1)  → more capacity
    - dual_epochs: 300 (was 100)  → dual potentials converge per iteration
    - cfm_epochs:  1000 (was 500) → flow model converges per iteration
    - advance_steps: 20 (was 30)  → slower coupling advance, UNets can track it
    - iterations: 60 (was 30)     → more outer loop steps
    - batch_size: 256 (was 128)   → more stable gradient estimates
    - particle_ratio: 0.5 (was 0.25) → more new-marginal samples in CFM loss
    - sigma: 0.1 (was 0.05)       → slightly more regularisation for image scale
    - lr_cosine_decay: True       → learning rate annealing for stability
    - Learning rates halved       → slower, more careful updates
    """
    try:
        import torchvision  # noqa: F401
    except ImportError:
        import warnings
        warnings.warn("torchvision not found – skipping image experiments.", stacklevel=2)
        return []

    # Build v1 specs then override configs in place
    specs = make_image_experiments(output_root)
    for spec in specs:
        t = spec.train_cfg
        t.iterations      = 60
        t.dual_epochs     = 300
        t.cfm_epochs      = 1_000
        t.advance_steps   = 20
        t.batch_size      = 256
        t.marginal_epochs = 15_000
        t.pretrain_epochs = 20_000

        m = spec.method_cfg
        m.particle_ratio  = 0.5
        m.sigma           = 0.1
        m.sigma2          = 0.1
        m.lr_cosine_decay = True

        c = spec.model_cfg
        c.hidden_size        = 64
        c.num_res_blocks     = 2
        c.dual_learning_rate = 5e-5
        c.mu_learning_rate   = 1e-4
        c.nu_learning_rate   = 1e-4
        c.flow_learning_rate = 1e-4

    return specs


def make_image_experiments(output_root: str = "results") -> List[ExperimentSpec]:
    """Return all image experiment specs (requires torchvision).

    Returns an empty list with a warning if torchvision is not installed.
    """
    try:
        import torchvision  # noqa: F401
    except ImportError:
        import warnings
        warnings.warn("torchvision not found – skipping image experiments.", stacklevel=2)
        return []

    return [
        _mnist_simple(False,    output_root),
        _mnist_simple(True,     output_root),
        _fashion_medium(False,  output_root),
        _fashion_medium(True,   output_root),
    ]
