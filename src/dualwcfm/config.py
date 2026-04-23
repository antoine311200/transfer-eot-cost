from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class TrainingConfig:
    device: torch.device
    dataset_size: int = 100_000

    iterations: int = 100
    marginal_epochs: int = 2500
    pretrain_epochs: int = 10000
    dual_epochs: int = 100
    cfm_epochs: int = 250
    advance_steps: int = 100

    num_rectified_flow: int = 0

    scale_factor: float = -1  # Overall scale factor for normalization (tune based on data scale)

    batch_size: int = 4096

    # Image-specific settings
    image_shape: Optional[Tuple[int, int, int]] = None  # (C, H, W) if working with images

    # Disk-based coupling for memory efficiency with images
    # None = auto (disk when image_shape set + dataset_size > 1000), True = force disk, False = force RAM
    use_disk_coupling: Optional[bool] = None
    coupling_save_dir: Optional[str] = None  # Directory for disk coupling (None = temp dir)
    coupling_chunk_size: int = 256  # Samples per chunk file

    # Checkpointing: save state periodically for later visualization / analysis.
    # Files land in <model_dir>/checkpoints/coupling/  and  <model_dir>/checkpoints/models/
    save_coupling_checkpoints: bool = False  # Save coupling dataset at checkpoint iterations
    save_model_checkpoints: bool = False     # Save CFM forward/backward weights at checkpoint iterations
    # checkpoint_every=0 → disabled, N → save at iter 0, every N-th iter, and the last iter
    checkpoint_every: int = 1

    # Final model persistence: save all learned models at the end of training.
    save_final_models: bool = True  # Save forward/backward/bridges/duals after all iterations

    # Weights & Biases integration (optional).
    wandb_project: Optional[str] = None   # W&B project name; None = no W&B logging
    wandb_run_name: Optional[str] = None  # W&B run display name; None = auto


@dataclass
class ModelConfig:
    dual_learning_rate: float = 1e-4
    mu_learning_rate: float = 1e-3
    nu_learning_rate: float = 1e-3
    flow_learning_rate: float = 5e-3
    dual_l2_reg: float = 0.0  # L2 regularization strength for dual networks (default: no regularization)


@dataclass
class FeedForwardConfig(ModelConfig):
    dimension: int = 2
    hidden_size: int = 64
    num_layers: int = 2  # Number of hidden layers in the MLP (bridges, duals, flow models)
    # For image models
    image_shape: Optional[Tuple[int, int, int]] = None  # (C, H, W)
    num_res_blocks: int = 1  # UNet residual blocks per resolution level
    # Bridge conditioning: if True, mu/nu bridge models accept x0 as a second input
    # (model(t, xt, x0)) so each particle's velocity is conditioned on its starting
    # point.  This eliminates crossing-path ambiguity and stabilises bridge training.
    use_conditional_bridge: bool = False

    # SIREN (anti-spectral-bias) settings.
    # When True, BidirectionalSirenStrategy replaces SELU MLPs with sinusoidal ones.
    # Recommended for high-frequency transports: shear, radial warp, polar vortex.
    use_siren: bool = False
    siren_omega_0: float = 30.0      # first-layer angular frequency
    siren_hidden_omega: float = 30.0 # hidden-layer angular frequency

    @property
    def is_image_model(self) -> bool:
        return self.image_shape is not None

    @property
    def flat_dimension(self) -> int:
        """Return flattened dimension for compatibility."""
        if self.image_shape:
            return self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
        return self.dimension


@dataclass
class VisualizeConfig:
    show_each: int = 10
    n_trajectories: int = 500
    # Visualization intervals for training phases (0 = disabled)
    show_each_marginal: int = 0  # During marginal flow training
    show_each_pretrain: int = 0  # During pretraining
    trajectory_stride: int = 2  # Plot one trajectory point every N steps (0 = only start/end)
    # 3-D camera angle (elevation and azimuth in degrees)
    camera_elev: float = 22.0
    camera_azim: float = -55.0

