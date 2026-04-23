from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import torch
from tqdm import trange

from dualwcfm.distributions.coupling import DiskCouplingDistribution
from dualwcfm.distributions.distribution import DiskDistribution
from dualwcfm.methods.standard import StandardMethod, StandardConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dualwcfm.trainer import Trainer


@dataclass
class MixtureConfig(StandardConfig):
    """StandardConfig + particle bridge mixing in the advance step."""
    particle_ratio: float = 0.0


class MixtureMethod(StandardMethod):
    """StandardMethod + particle/flow split in the advance step.

    A fraction `config.particle_ratio` of each chunk is advanced by Euler-stepping
    the stored (x, y) pair through the marginal bridges, breaking the self-referential
    y = flow(x) loop.  The remainder uses the standard flow model transport.

    The particle slots are always the first n_particle indices of each chunk so that
    each particle accumulates one Euler step per iteration and travels the full path
    from t=0 to t=1 over the course of training.  Randomising slot assignment would
    cause each particle to be stepped only particle_ratio fraction of the time,
    reducing the total displacement to particle_ratio instead of 1.

    Fix vs naive implementation:
    - True t=0 origins stored per particle slot and used for conditional bridge calls
      instead of x_prev (which drifts away from x_0 as t increases).
      Origins are DiskDistribution-backed when the coupling is disk-based so RAM is
      not blown up for large datasets.
    """

    def __init__(self, config: MixtureConfig):
        super().__init__(config)
        # Either two DiskDistributions (disk coupling) or two plain CPU tensors (in-memory).
        self._x_origins: DiskDistribution | torch.Tensor | None = None
        self._y_origins: DiskDistribution | torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Origins management
    # ------------------------------------------------------------------

    def _init_origins(self, trainer: Trainer) -> None:
        """Build the t=0 origin buffers from the coupling's current (t=0) state."""
        coupling = trainer.current_coupling
        if isinstance(coupling, DiskCouplingDistribution):
            coupling._ensure_memmap()
            sample_shape = tuple(coupling._x_mmap.shape[1:])
            np_dtype = coupling._x_mmap.dtype
            for attr, suffix in [("_x_origins", "ox"), ("_y_origins", "oy")]:
                save_dir = os.path.join(coupling.save_dir, f"origins_{suffix}")
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                dist = DiskDistribution(coupling.size, sample_shape, np_dtype, save_dir)
                for chunk_idx in range(coupling.n_chunks):
                    start = chunk_idx * coupling.chunk_size
                    end = min(start + coupling.chunk_size, coupling.size)
                    x_chunk, y_chunk = coupling._load_chunk(chunk_idx)
                    dist.write(torch.arange(start, end), x_chunk if suffix == "ox" else y_chunk)
                setattr(self, attr, dist)
        else:
            self._x_origins = coupling.dataset[0].clone().cpu()
            self._y_origins = coupling.dataset[1].clone().cpu()

    def _read_origins(self, global_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self._x_origins, DiskDistribution):
            return self._x_origins.read(global_idx), self._y_origins.read(global_idx)
        return self._x_origins[global_idx].clone(), self._y_origins[global_idx].clone()

    # ------------------------------------------------------------------
    # Advance step (postprocess)
    # ------------------------------------------------------------------

    def step_postprocess(self, trainer: Trainer, iteration: int, ctx: dict) -> None:
        backward = ctx["backward"]
        device = trainer.device
        particle_ratio = self.config.particle_ratio
        conditional = getattr(trainer.strategy.config, "use_conditional_bridge", False)

        # Initialise / reset origins at the start of each training run (conditional only).
        if iteration == 0 and conditional:
            self._init_origins(trainer)

        chunk_size = (
            trainer.current_coupling.chunk_size
            if isinstance(trainer.current_coupling, DiskCouplingDistribution)
            else (100_000 if not trainer.config.image_shape else 256)
        )

        (mu_model, _), (nu_model, _) = trainer.strategy.get_bridges(device)
        mu_model.eval()
        nu_model.eval()

        flow_fwd = trainer.strategy.get_flow(0, device)[0]
        flow_bwd = trainer.strategy.get_flow(1, device)[0]

        n_chunks = (trainer.current_coupling.size + chunk_size - 1) // chunk_size

        with torch.no_grad():
            for chunk_idx in trange(n_chunks, desc="Advancing coupling"):
                i = chunk_idx * chunk_size
                actual = min(chunk_size, trainer.current_coupling.size - i)
                n_particle = int(actual * particle_ratio) if particle_ratio > 0 else 0
                n_flow = actual - n_particle

                # --- Particle pairs: Euler-step the fixed first n_particle slots ---
                # Fixed slots are required: each particle must be stepped at every
                # iteration to accumulate the full t=0→1 displacement.
                if n_particle > 0:
                    t_par = torch.full(
                        (n_particle, 1),
                        iteration / trainer.config.iterations,
                        device=device,
                    )

                    if isinstance(trainer.current_coupling, DiskCouplingDistribution):
                        _chunk = trainer.current_coupling._load_chunk(chunk_idx)
                        x_prev = _chunk[0][:n_particle].to(device)
                        y_prev = _chunk[1][:n_particle].to(device)
                    else:
                        x_prev = trainer.current_coupling.dataset[0][i : i + n_particle].to(device)
                        y_prev = trainer.current_coupling.dataset[1][i : i + n_particle].to(device)

                    # Use stored t=0 origins for conditioning instead of x_prev.
                    if conditional:
                        x0_par, y0_par = self._read_origins(torch.arange(i, i + n_particle))
                        x0_par, y0_par = x0_par.to(device), y0_par.to(device)
                    else:
                        x0_par, y0_par = x_prev, y_prev  # ignored by unconditional bridge

                    x_par = x_prev + self._bridge_call(mu_model, t_par, x_prev, x0_par, conditional) * (1 / trainer.config.iterations)
                    y_par = y_prev + self._bridge_call(nu_model, t_par, y_prev, y0_par, conditional) * (1 / trainer.config.iterations)

                # --- Flow pairs: standard transport ---
                if n_flow > 0:
                    t_next = torch.full(
                        (n_flow, 1),
                        (iteration + 1) / trainer.config.iterations,
                        device=device,
                    )
                    x_mu_flow = self._interpolate(
                        trainer.initial_source, trainer.new_source, trainer,
                        n_flow, t_next, trainer._sigma, device
                    )
                    y_nu_flow = self._interpolate(
                        trainer.initial_target, trainer.new_target, trainer,
                        n_flow, t_next, trainer._sigma, device
                    )
                    x_flow, y_flow = self._advance_chunk(
                        trainer, x_mu_flow, y_nu_flow, flow_fwd, flow_bwd, backward,
                    )

                # Assemble: particles first, flow second (matches the fixed-slot layout).
                if n_particle > 0 and n_flow > 0:
                    x_next = torch.cat([x_par, x_flow], dim=0)
                    y_next = torch.cat([y_par, y_flow], dim=0)
                elif n_particle > 0:
                    x_next, y_next = x_par, y_par
                else:
                    x_next, y_next = x_flow, y_flow

                trainer.current_coupling.update_chunk(i, x_next.detach().cpu(), y_next.detach().cpu())

        trainer.visualizer.at_postprocess_end(iteration, trainer, ctx)
