from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import trange

from dualwcfm.distributions.coupling import DiskCouplingDistribution
from dualwcfm.method.standard import StandardMethod, StandardConfig
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
    """

    def __init__(self, config: MixtureConfig):
        super().__init__(config)

    def step_postprocess(self, trainer: Trainer, iteration: int, ctx: dict) -> None:
        backward = ctx["backward"]
        device = trainer.device
        particle_ratio = self.config.particle_ratio

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

                # --- Particle pairs: Euler-step stored (x_prev, y_prev) via bridges ---
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
                    x_par = x_prev + mu_model(t_par, x_prev) * (1 / trainer.config.iterations)
                    y_par = y_prev + nu_model(t_par, y_prev) * (1 / trainer.config.iterations)

                # --- Flow pairs: standard transport ---
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
                    trainer,
                    x_mu_flow,
                    y_nu_flow,
                    flow_fwd,
                    flow_bwd,
                    backward,
                )

                if n_particle > 0:
                    x_next = torch.cat([x_par, x_flow], dim=0)
                    y_next = torch.cat([y_par, y_flow], dim=0)
                else:
                    x_next, y_next = x_flow, y_flow

                trainer.current_coupling.update_chunk(i, x_next.detach().cpu(), y_next.detach().cpu())

        trainer.visualizer.at_postprocess_end(iteration, trainer, ctx)