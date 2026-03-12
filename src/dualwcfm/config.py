from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class TrainingConfig:
    device: torch.device
    dataset_size: int = 100_000

    marginal_epochs: int = 2500
    pretrain_epochs: int = 10000
    dual_epochs: int = 100
    cfm_epochs: int = 250
    advance_steps: int = 100

    iterations: int = 100
    loss: str = "implicit"
    advance_with_flow: bool = False
    num_rectified_flow: int = 0
    use_noisy_field: bool = False
    correction_epoch_multiplier: int = 5
    dual_mismatch_reg: float = 0.0

    sigma: float = 0.1
    sigma2: float = 0.1
    l2_reg: float = 1e-3
    lip_reg: float = 1e-3
    mean_reg: float = 0.1
    scale_factor: float = -1  # Overall scale factor for normalization (tune based on data scale)

    batch_size: int = 4096

    # Image-specific settings
    image_shape: Optional[Tuple[int, int, int]] = None  # (C, H, W) if working with images

    # Disk-based coupling for memory efficiency with images
    use_disk_coupling: bool = False  # Set True for large image datasets
    coupling_save_dir: Optional[str] = None  # Directory for disk coupling (None = temp dir)
    coupling_chunk_size: int = 256  # Samples per chunk file

    # Mixed precision training (float16 forward/backward, float32 optimizer)
    use_amp: bool = False  # Set True for ~2x speedup on modern GPUs with tensor cores

    # Error-accumulation correction
    # correction_each > 0: run a correction step every `correction_each` iterations.
    # The correction retrains the nu bridge (and mu bridge if bidirectional) from
    # nu_t_with_errors (actual forward-model output at time t) to nu_1, then uses
    # these corrected bridges to retrain the dual networks f and g.
    correction_each: int = 0    # 0 = disabled
    correction_epochs: int = 100

    # Fraction of coupling pairs built by Euler-stepping the previous (x,y) via the
    # marginal bridges rather than running the forward flow model.  Range [0, 1].
    # 0 = pure flow (default), 1 = pure particle, 0.5 = half-and-half.
    coupling_particle_ratio: float = 0.0

    # Checkpointing: save state at every iteration for later visualization / analysis.
    # Files land in <model_dir>/checkpoints/coupling/  and  <model_dir>/checkpoints/models/
    save_coupling_checkpoints: bool = False  # Save coupling dataset after each advance step
    save_model_checkpoints: bool = False     # Save CFM forward/backward weights after each iteration


@dataclass
class ModelConfig:
    dual_learning_rate: float = 1e-4
    mu_learning_rate: float = 1e-3
    nu_learning_rate: float = 1e-3
    flow_learning_rate: float = 5e-3


@dataclass
class FeedForwardConfig(ModelConfig):
    dimension: int = 2
    hidden_size: int = 64
    # For image models
    image_shape: Optional[Tuple[int, int, int]] = None  # (C, H, W)

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
