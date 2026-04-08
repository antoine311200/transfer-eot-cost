import logging
import os
import shutil

import torch

from dualwcfm.config import TrainingConfig
from dualwcfm.flow_matching import ConditionalFlowMatching
from dualwcfm.strategy import Strategy, BidirectionalStrategy
from dualwcfm.methods.base import Method
from dualwcfm.visualization import Visualizer
from dualwcfm.distributions.coupling import (
    CouplingDistribution,
    FixedSizeCouplingDistribution,
    DiskCouplingDistribution,
)
from dualwcfm.distributions.distribution import Distribution
from dualwcfm.distributions.scaled import ScaledCouplingDistribution, ScaledDistribution

logger = logging.getLogger("dualwcfm")


class Trainer:

    def __init__(
        self,
        initial_coupling: CouplingDistribution,
        new_source: Distribution,
        new_target: Distribution,
        flow_matching: ConditionalFlowMatching,
        strategy: Strategy,
        visualizer: Visualizer,
        config: TrainingConfig,
        method: Method,
        device: torch.device = torch.device("cpu"),
    ):

        self.config = config
        self.device = device
        self.method = method

        # Log the configurations for reproducibility
        logger.info("Training configuration:")
        for key, value in vars(config).items():
            logger.info(f"  {key}: {value}")
        for key, value in vars(method.config).items():
            logger.info(f"  method.{key}: {value}")

        # Compute normalization scale: max absolute value across a probe batch
        # of all four marginals so every sample lies in [-1, 1] after scaling.
        if self.config.scale_factor > 0:
            self.scale = self.config.scale_factor
            logger.info(f"Using provided normalization scale: {self.scale:.4f}")
        else:
            with torch.no_grad():
                probe_n = 1000
                x0, x1 = initial_coupling.coupling_sample(probe_n)
                s = new_source.sample(probe_n)
                t = new_target.sample(probe_n)
                self.scale = max(
                    x0.abs().max().item(),
                    x1.abs().max().item(),
                    s.abs().max().item(),
                    t.abs().max().item(),
                )
            logger.info(f"Normalization scale set to {self.scale:.4f} based on probe samples.")

        self.initial_coupling = ScaledCouplingDistribution(initial_coupling, self.scale)
        self.initial_source, self.initial_target = self.initial_coupling.get_marginals()
        self.new_source = ScaledDistribution(new_source, self.scale)
        self.new_target = ScaledDistribution(new_target, self.scale)

        # Scaled noise levels — never mutate method.config
        self._sigma = method.config.sigma / self.scale
        self._sigma2 = method.config.sigma2 / self.scale

        self.visualizer = visualizer
        self.strategy = strategy
        self.flow_matching = flow_matching

        # Fixed-size coupling and marginal distributions
        self._init_distributions(self.initial_coupling)

    def _init_distributions(self, initial_coupling: CouplingDistribution):
        mu_distribution, nu_distribution = initial_coupling.get_marginals()

        self.mu_distribution = mu_distribution
        self.nu_distribution = nu_distribution

        # Use disk-based coupling for images to avoid OOM.
        # use_disk_coupling=None → auto (disk when image_shape + large dataset)
        # use_disk_coupling=True → always disk; use_disk_coupling=False → always RAM
        _auto_disk = self.config.image_shape is not None and self.config.dataset_size > 1000
        if self.config.use_disk_coupling is True or (self.config.use_disk_coupling is None and _auto_disk):
            logger.info("Using disk-based coupling distribution for memory efficiency.")
            self.current_coupling = DiskCouplingDistribution(
                initial_coupling,
                size=self.config.dataset_size,
                save_dir=self.config.coupling_save_dir,
                chunk_size=self.config.coupling_chunk_size,
                device=self.device,
            )
        else:
            self.current_coupling = FixedSizeCouplingDistribution(initial_coupling, size=self.config.dataset_size)

        # Copy to Initial distribution for later use
        if hasattr(self.mu_distribution, "clone") and hasattr(self.nu_distribution, "clone"):
            self.initial_mu_distribution = self.mu_distribution.clone()
            self.initial_nu_distribution = self.nu_distribution.clone()
        self.initial_coupling_distribution = self.current_coupling

    def _save_iteration_checkpoint(self, iteration: int, save: str):
        """Persist coupling dataset and/or CFM weights for this iteration."""
        checkpoint_root = os.path.join(save, "checkpoints")

        if self.config.save_coupling_checkpoints:
            coupling_dir = os.path.join(checkpoint_root, "coupling")
            os.makedirs(coupling_dir, exist_ok=True)
            coupling = self.current_coupling

            if isinstance(coupling, DiskCouplingDistribution):
                iter_dir = os.path.join(coupling_dir, f"iter_{iteration:04d}")
                os.makedirs(iter_dir, exist_ok=True)

                if coupling._x_mmap is not None:
                    coupling._x_mmap.flush()
                    coupling._y_mmap.flush()

                for fname in ("x.dat", "y.dat", "meta.pt"):
                    src = os.path.join(coupling.save_dir, fname)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(iter_dir, fname))

                logger.info(f"  [ckpt] coupling saved -> {iter_dir}")
            elif isinstance(coupling, FixedSizeCouplingDistribution):
                path = os.path.join(coupling_dir, f"iter_{iteration:04d}.pt")
                torch.save({"x": coupling.dataset[0].cpu(), "y": coupling.dataset[1].cpu()}, path)

                logger.info(f"  [ckpt] coupling saved -> {path}")

        if self.config.save_model_checkpoints:
            mdl_dir = os.path.join(checkpoint_root, "models")
            os.makedirs(mdl_dir, exist_ok=True)
            fwd, _ = self.strategy.get_flow(0, self.device)
            path_fwd = os.path.join(mdl_dir, f"iter_{iteration:04d}_forward.pth")
            torch.save(fwd.state_dict(), path_fwd)

            logger.info(f"  [ckpt] forward model saved -> {path_fwd}")

            if isinstance(self.strategy, BidirectionalStrategy):
                bwd, _ = self.strategy.get_flow(1, self.device)
                path_bwd = os.path.join(mdl_dir, f"iter_{iteration:04d}_backward.pth")
                torch.save(bwd.state_dict(), path_bwd)

                logger.info(f"  [ckpt] backward model saved -> {path_bwd}")

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(self, retrain: bool = False, save: str = None, save_checkpoints: str = None):
        self.strategy._create_models(self.strategy.config.dimension, self.strategy.config.hidden_size)
        self.strategy._create_optimizers(self.strategy.config)
        self.strategy.to_device(self.device)

        if not retrain:
            self.mu_distribution = self.initial_mu_distribution.clone()
            self.nu_distribution = self.initial_nu_distribution.clone()
            self.current_coupling = self.initial_coupling_distribution.clone()

        # Marginal flow matching + flow pretraining
        self.method.pretrain(self, retrain, save)

        # Main iteration loop
        for iteration in range(self.config.iterations):
            logger.info(f"################# Iteration {iteration+1}/{self.config.iterations} #################")

            ctx = self.method.step_preprocess(self, iteration)
            self.method.step_duals(self, iteration, ctx)
            self.method.step_flow_matching(self, iteration, ctx)
            self.method.step_postprocess(self, iteration, ctx)

            if save_checkpoints and (self.config.save_coupling_checkpoints or self.config.save_model_checkpoints):
                n = self.config.checkpoint_every
                if n > 0:
                    is_first = iteration == 0
                    is_last  = iteration == self.config.iterations - 1
                    is_nth   = (iteration + 1) % n == 0
                    if is_first or is_last or is_nth:
                        self._save_iteration_checkpoint(iteration, save_checkpoints)

            self.visualizer.display(iteration, self)
            distances = self.visualizer.analyse(iteration, self)
            logger.info(
                f"  SW distances — "
                f"W1_mu={distances['W1_mu']:.4f}  W2_mu={distances['W2_mu']:.4f}  "
                f"W1_nu={distances['W1_nu']:.4f}  W2_nu={distances['W2_nu']:.4f}"
            )

        # Save final trained models after all iterations.
        if save and self.config.save_final_models:
            self.strategy._save_final(save)

