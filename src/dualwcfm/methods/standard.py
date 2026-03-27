from __future__ import annotations

import logging
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import trange

from dualwcfm.losses import FlowLoss
from dualwcfm.strategy import BidirectionalStrategy
from dualwcfm.utils import is_image_tensor
from dualwcfm.distributions.coupling import DiskCouplingDistribution
from dualwcfm.methods.base import Method, MethodConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dualwcfm.trainer import Trainer

logger = logging.getLogger("dualwcfm")

@dataclass
class StandardConfig(MethodConfig):
    """Config for StandardMethod: interpolation-based marginals + weighted/implicit CFM."""
    sigma: float = 0.1
    sigma2: float = 0.1
    loss: str = "weighted"
    l2_reg: float = 1e-3
    lip_reg: float = 1e-3
    mean_reg: float = 0.1
    use_noisy_field: bool = False

    # Flow LR schedule for the CFM training loop.
    #
    # lr_min_factor scales the base flow_learning_rate and is used by both modes:
    #
    #   lr_cosine_decay=True  (cosine within each iteration)
    #       LR starts at flow_lr, decays to flow_lr * lr_min_factor over cfm_epochs,
    #       then resets to flow_lr at the next iteration.
    #       Useful when cfm_epochs is large enough (~1000+) to benefit from refinement.
    #
    #   lr_cosine_decay=False  (constant reduced LR)
    #       LR is fixed at flow_lr * lr_min_factor for every iteration.
    #       Simpler and more predictable; recommended when cfm_epochs is small (~500).
    #       Set lr_min_factor=1.0 to keep the original flat flow_learning_rate.
    lr_cosine_decay: bool = False
    lr_min_factor: float = 1.0  # 1.0 = no change; 0.5 = half the base flow_lr

class StandardMethod(Method):
    """Standard DualWCFM method: interpolation-based marginals, weighted/implicit CFM.

    Does not include correction steps or particle mixing.
    """

    def __init__(self, config: StandardConfig):
        super().__init__(config)
        self._loss_fn = FlowLoss.get_loss(config.loss)

    @property
    def use_noisy_field(self) -> bool:
        return self.config.use_noisy_field

    # ------------------------------------------------------------------
    # Pretrain
    # ------------------------------------------------------------------

    def pretrain(self, trainer: Trainer, retrain: bool = True, save: str = None) -> None:
        self._pretrain_bridges(trainer, retrain, save)

        if not retrain:
            logger.info("Skipping flow pretraining, loading pretrained models.")
            if not save:
                # Copy initial states to active models (already done for marginals above)
                trainer.strategy.forward_model.load_state_dict(trainer.strategy.initial_forward_model.state_dict())
                if isinstance(trainer.strategy, BidirectionalStrategy):
                    trainer.strategy.backward_model.load_state_dict(
                        trainer.strategy.initial_backward_model.state_dict()
                    )
            trainer.visualizer.at_flow_pretrain_end(trainer, {})
            return

        strategy = trainer.strategy
        if isinstance(strategy, BidirectionalStrategy):
            logger.info("Pretraining bidirectional flows.")
            fwd_model, fwd_opt = strategy.get_flow(0, trainer.device)
            bwd_model, bwd_opt = strategy.get_flow(1, trainer.device)
            self._pretrain_model(trainer, fwd_model, fwd_opt, backward=False)
            self._pretrain_model(trainer, bwd_model, bwd_opt, backward=True)
            strategy.initial_forward_model.load_state_dict(fwd_model.state_dict())
            strategy.initial_backward_model.load_state_dict(bwd_model.state_dict())
        else:
            logger.info("Pretraining unidirectional flow.")
            model, optimizer = strategy.get_flow(0, trainer.device)
            self._pretrain_model(trainer, model, optimizer, backward=False)
            strategy.initial_forward_model.load_state_dict(model.state_dict())

        trainer.strategy._save_pretrained(save)
        trainer.visualizer.at_flow_pretrain_end(trainer, {})

    # ------------------------------------------------------------------
    # Pretrain marginal flow matching
    # ------------------------------------------------------------------

    def _pretrain_bridges(self, trainer: Trainer, retrain: bool = True, save: str = None) -> None:
        """Train (or load) mu/nu bridge models for marginal flow matching."""
        sigma = trainer._sigma
        batch_size = trainer.config.batch_size
        device = trainer.device
        loss_fn = nn.MSELoss()

        (mu_model, mu_optimizer), (nu_model, nu_optimizer) = trainer.strategy.get_bridges(device)
        initial_mu_model, initial_nu_model = trainer.strategy.get_initial_bridges(device)

        if retrain:
            previous_mu_model = copy.deepcopy(mu_model)
            previous_nu_model = copy.deepcopy(nu_model)
            mu_model.train()
            nu_model.train()

            for i in range(trainer.config.num_rectified_flow + 1):
                logger.info(f"Marginal flow matching phase {i + 1}/{trainer.config.num_rectified_flow + 1}")

                n_epochs = trainer.config.marginal_epochs if i == 0 else trainer.config.marginal_epochs // 10
                mu_scheduler = lr_scheduler.CosineAnnealingLR(mu_optimizer, T_max=n_epochs, eta_min=1e-6)
                nu_scheduler = lr_scheduler.CosineAnnealingLR(nu_optimizer, T_max=n_epochs, eta_min=1e-6)

                pbar = trange(n_epochs, desc="Marginal Flow Matching")
                for epoch in pbar:
                    # Sample (x0, y0) from initial coupling and (x1, y1) from current marginals.
                    # In the first pass, use initial coupling for both.
                    # In rectified passes, use previous models to get better (x1, y1) samples representing μ_1 and ν_1 solving the ODE.
                    if i == 0:
                        x0 = trainer.initial_coupling.sample_init(batch_size)
                        y0 = trainer.initial_coupling.sample_target(batch_size)
                        x1 = trainer.new_source.sample(batch_size)
                        y1 = trainer.new_target.sample(batch_size)
                    else:
                        x0 = trainer.initial_coupling.sample_init(batch_size).to(device)
                        y0 = trainer.initial_coupling.sample_target(batch_size).to(device)
                        previous_mu_model.eval()
                        previous_nu_model.eval()
                        x1 = trainer.flow_matching.evolve(x0, previous_mu_model, num_steps=50, extend=False)
                        y1 = trainer.flow_matching.evolve(y0, previous_nu_model, num_steps=50, extend=False)
                        previous_mu_model.train()
                        previous_nu_model.train()

                    x0, x1, y0, y1 = (
                        x0.to(device),
                        x1.to(device),
                        y0.to(device),
                        y1.to(device),
                    )

                    # Sample random time and interpolate to get (x_t, y_t) representing μ_t and ν_t
                    t = trainer.flow_matching.random_time(batch_size, device)
                    xt = trainer.flow_matching.interpolate(x0, x1, t, sigma)
                    yt = trainer.flow_matching.interpolate(y0, y1, t, sigma)

                    mu_optimizer.zero_grad()
                    nu_optimizer.zero_grad()

                    ut = mu_model(t, xt)
                    ut_true = trainer.flow_matching.vector_field(
                        xt, t, x0, x1, extend=False, noisy=self.use_noisy_field
                    )
                    loss_mu = loss_fn(ut, ut_true)
                    loss_mu.backward()
                    mu_optimizer.step()

                    vt = nu_model(t, yt)
                    vt_true = trainer.flow_matching.vector_field(
                        yt, t, y0, y1, extend=False, noisy=self.use_noisy_field
                    )
                    loss_nu = loss_fn(vt, vt_true)
                    loss_nu.backward()
                    nu_optimizer.step()

                    mu_scheduler.step()
                    nu_scheduler.step()

                    pbar.set_postfix({
                        "Loss mu": loss_mu.item(),
                        "Loss nu": loss_nu.item(),
                        "lr": mu_scheduler.get_last_lr()[0],
                    })

                    trainer.visualizer.during_pretrain(
                        epoch,
                        trainer,
                        {
                            "phase": "bridges",
                            "mu_model": mu_model,
                            "nu_model": nu_model,
                            "phase_idx": i,
                            "force": i != 0 and epoch == trainer.config.marginal_epochs // 10 - 1,
                        },
                    )

                previous_mu_model.load_state_dict(mu_model.state_dict())
                previous_nu_model.load_state_dict(nu_model.state_dict())

            initial_mu_model.load_state_dict(mu_model.state_dict())
            initial_nu_model.load_state_dict(nu_model.state_dict())
        else:
            logger.info("Skipping marginal flow matching retraining.")
            if save:
                trainer.strategy._load_pretrained_models(save, device)
            else:
                mu_model.load_state_dict(initial_mu_model.state_dict())
                nu_model.load_state_dict(initial_nu_model.state_dict())

        torch.cuda.empty_cache()
        trainer.visualizer.at_pretrain_end(trainer, {})

    # ------------------------------------------------------------------
    # Pretrain forward (and backward) flow models
    # ------------------------------------------------------------------

    def _pretrain_model(self, trainer: Trainer, model, optimizer, backward: bool = False) -> None:
        batch_size = trainer.config.batch_size
        device = trainer.device
        loss_fn = nn.MSELoss()
        model.train()

        pbar = trange(trainer.config.pretrain_epochs, desc="Pretraining")
        for epoch in pbar:
            x0, x1 = trainer.initial_coupling.coupling_sample(batch_size)
            x0, x1 = x0.to(device), x1.to(device)
            if backward:
                x0, x1 = x1, x0

            s = trainer.flow_matching.random_time(batch_size, device)
            x_s = trainer.flow_matching.interpolate(x0, x1, s, trainer._sigma2)

            optimizer.zero_grad()
            u_s = model(s, x_s, x0)
            u_s_true = trainer.flow_matching.vector_field(x_s, s, x0, x1, noisy=self.config.use_noisy_field)
            loss = loss_fn(u_s, u_s_true)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Pretrain Loss": loss.item()})
            trainer.visualizer.during_pretrain(
                epoch,
                trainer,
                {
                    "phase": "flow",
                    "model": model,
                    "backward": backward,
                },
            )

    # ------------------------------------------------------------------
    # Step preprocess
    # ------------------------------------------------------------------

    def step_preprocess(self, trainer: Trainer, iteration: int) -> dict:
        backward = isinstance(trainer.strategy, BidirectionalStrategy) and iteration % 2 != 0
        is_image = trainer.config.image_shape is not None
        t = iteration / trainer.config.iterations
        t_tensor = torch.full((trainer.config.batch_size, 1), t, device=trainer.device)

        # Apply LR policy at the start of every iteration:
        #   cosine mode  – reset to base_lr so the within-iteration decay starts fresh.
        #   constant mode – set to base_lr * lr_min_factor and keep it flat.
        # No-op when lr_cosine_decay=False and lr_min_factor=1.0 (default).
        if self.config.lr_cosine_decay or self.config.lr_min_factor != 1.0:
            self._reset_flow_lr(trainer)

        return {
            "t": t,
            "t_tensor": t_tensor,
            "backward": backward,
            "is_image": is_image,
        }

    def _reset_flow_lr(self, trainer: Trainer) -> None:
        """Set the flow optimizer(s) to the iteration starting LR.

        For cosine mode: resets to the full base LR so the within-iteration
        schedule always decays from the same ceiling.
        For constant mode: sets to base_lr * lr_min_factor directly, which then
        stays flat throughout the cfm_epochs loop.
        """
        base_lr    = trainer.strategy.config.flow_learning_rate
        start_lr   = base_lr if self.config.lr_cosine_decay else base_lr * self.config.lr_min_factor
        for opt in self._flow_optimizers(trainer):
            for pg in opt.param_groups:
                pg["lr"] = start_lr

    def _flow_optimizers(self, trainer: Trainer):
        """Yield the flow optimizer(s) that participate in the within-iteration schedule."""
        strat = trainer.strategy
        yield strat.forward_optimizer
        if isinstance(strat, BidirectionalStrategy) and hasattr(strat, "backward_optimizer"):
            yield strat.backward_optimizer

    # ------------------------------------------------------------------
    # Dual training step
    # ------------------------------------------------------------------

    def step_duals(self, trainer: Trainer, iteration: int, ctx: dict) -> None:
        batch_size = trainer.config.batch_size
        device = trainer.device
        t_tensor = ctx["t_tensor"]
        backward = ctx["backward"]
        is_image = ctx["is_image"]

        (mu_model, _), (nu_model, _) = trainer.strategy.get_bridges(device)
        f_dual, g_dual, dual_optimizer = trainer.strategy.get_duals(device)
        f_dual.train()
        g_dual.train()
        mu_model.eval()
        nu_model.eval()

        pbar = trange(trainer.config.dual_epochs, desc="Dual Training")
        for _ in pbar:
            x, y = trainer.current_coupling.coupling_sample(batch_size)
            x, y = x.to(device), y.to(device)

            # Sample marginals at time t, (x_mu, y_nu) ~ (μ_t, ν_t)
            x_mu, y_nu = self._sample_marginals(trainer, t_tensor, backward, batch_size, device)

            with torch.no_grad():
                u_t = mu_model(t_tensor, x_mu)
                v_t = nu_model(t_tensor, y_nu)

            x, y = x.requires_grad_(True), y.requires_grad_(True)
            x_mu, y_nu = x_mu.requires_grad_(True), y_nu.requires_grad_(True)

            a, a_mu = f_dual(x), f_dual(x_mu)
            b, b_nu = g_dual(y), g_dual(y_nu)

            # Gauge invariance
            a_mean = a.mean()
            a = a - a_mean
            a_mu = a_mu - a_mean

            # Coupling term (with optional change_of_measure)
            # com = trainer.strategy.change_of_measure
            # com_opt = trainer.strategy.get_com_optimizer()
            # if com is not None:
            #     xy = torch.cat([x.view(x.shape[0], -1), y.view(y.shape[0], -1)], dim=1)
            #     d = com(xy)
            #     coupling_term = torch.mean((a + b + d) ** 2)
            # else:
            coupling_term = torch.mean((a + b) ** 2)

            # Gradient / divergence terms
            grad_a = torch.autograd.grad(a_mu.sum(), x_mu, create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad_b = torch.autograd.grad(b_nu.sum(), y_nu, create_graph=True, retain_graph=True, only_inputs=True)[0]

            if is_image:
                a_term = torch.mean((grad_a * u_t).sum(dim=[1, 2, 3]))
                b_term = torch.mean((grad_b * v_t).sum(dim=[1, 2, 3]))
                gp_f = ((grad_a.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
                gp_g = ((grad_b.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
            else:
                a_term = torch.mean(torch.sum(grad_a * u_t, dim=1))
                b_term = torch.mean(torch.sum(grad_b * v_t, dim=1))
                gp_f = ((grad_a.norm(2, dim=1) - 1) ** 2).mean()
                gp_g = ((grad_b.norm(2, dim=1) - 1) ** 2).mean()

            # Regularization terms
            penalty = torch.tensor(0.0, device=device)
            if self.config.l2_reg > 0 or self.config.lip_reg > 0 or self.config.mean_reg > 0:
                mean_a = torch.mean(a)
                mean_b = torch.mean(b)
                mean_a_sq = torch.mean(a**2)
                mean_b_sq = torch.mean(b**2)
                mean_a_mu_sq = torch.mean(a_mu**2)
                mean_b_nu_sq = torch.mean(b_nu**2)
                var_a = mean_a_sq - mean_a**2
                var_b = mean_b_sq - mean_b**2

                l2_reg = self.config.l2_reg * (mean_a_sq + mean_b_sq + mean_a_mu_sq + mean_b_nu_sq)
                lip_reg = self.config.lip_reg * (gp_f + gp_g)
                drift_penalty = self.config.mean_reg * (mean_a**2 + mean_b**2)
                var_penalty = self.config.mean_reg * (var_a + var_b)

                penalty = l2_reg + lip_reg + drift_penalty + var_penalty

            loss = coupling_term - 2 * a_term - 2 * b_term
            loss += penalty

            dual_optimizer.zero_grad()
            loss.backward()
            dual_optimizer.step()

            pbar.set_postfix({"Dual Loss": loss.item()})

        # Diagnostic after training
        x0_test, x1_test = trainer.current_coupling.coupling_sample(batch_size)
        x0_test, x1_test = x0_test.to(device), x1_test.to(device)
        with torch.no_grad():
            log_weights = trainer.strategy.log_weight(x0_test, x1_test, trainer.config.iterations)

        trainer.visualizer.at_duals_end(
            iteration,
            trainer,
            {
                "x0": x0_test,
                "x1": x1_test,
                "log_w": log_weights,
                "is_image": is_image,
            },
        )

    def _sample_marginals(self, trainer: Trainer, t_tensor, backward, batch_size, device):
        """Return (x_mu, y_nu) samples representing μ_t and ν_t."""
        x_mu = trainer.flow_matching.interpolate(
            trainer.initial_coupling.sample_init(batch_size).to(device),
            trainer.new_source.sample(batch_size).to(device),
            t_tensor,
            sigma=trainer._sigma,
        )
        y_nu = trainer.flow_matching.interpolate(
            trainer.initial_coupling.sample_target(batch_size).to(device),
            trainer.new_target.sample(batch_size).to(device),
            t_tensor,
            sigma=trainer._sigma,
        )
        return x_mu, y_nu

    def _interpolate(self, initial_dist, final_dist, trainer, batch_size, t_tensor, sigma, device):
        """Helper to interpolate between two distributions."""
        return trainer.flow_matching.interpolate(
            initial_dist.sample(batch_size).to(device),
            final_dist.sample(batch_size).to(device),
            t_tensor,
            sigma=sigma,
        )

    # ------------------------------------------------------------------
    # CFM training step
    # ------------------------------------------------------------------

    def step_flow_matching(self, trainer: Trainer, iteration: int, ctx: dict) -> None:
        batch_size = trainer.config.batch_size
        device = trainer.device
        is_image = ctx["is_image"]
        backward = ctx["backward"]

        model, _ = trainer.strategy.get_flow(iteration, device)
        previous_model = trainer.strategy.get_previous_flow(iteration, device)
        # Copy current → previous BEFORE the epoch loop (critical for ImplicitFlowLoss)
        previous_model.load_state_dict(model.state_dict())
        previous_model.eval()
        model.train()

        mu_t, nu_t = trainer.current_coupling.coupling_sample(batch_size)
        mu_t, nu_t = mu_t.to(device), nu_t.to(device)

        t_full = torch.full((batch_size, 1), ctx["t"], device=device)
        nu_t_target = self._interpolate(
            trainer.initial_coupling.get_marginals()[1],
            trainer.new_target,
            trainer,
            batch_size,
            t_full,
            trainer._sigma,
            device,
        )
        mu_t_quiver = mu_t[: min(400, batch_size)] if not is_image else None
        flow_ctx = {
            "x0_samp": mu_t,
            "x1_samp": nu_t,
            "x1_target": nu_t_target,
            "model": model,
            "x0_quiver": mu_t_quiver,
            "is_image": is_image,
        }
        start_ctx = trainer.visualizer.at_flow_matching_start(iteration, trainer, flow_ctx)

        import math
        cfm_epochs = trainer.config.cfm_epochs
        base_lr    = trainer.strategy.config.flow_learning_rate
        min_factor = self.config.lr_min_factor
        use_cosine = self.config.lr_cosine_decay and cfm_epochs > 1

        pbar = trange(cfm_epochs, desc="CFM Training")
        for epoch in pbar:
            if use_cosine:
                # Cosine decay from base_lr to base_lr * min_factor within this iteration.
                cos_factor = min_factor + (1.0 - min_factor) * (
                    0.5 * (1.0 + math.cos(math.pi * epoch / (cfm_epochs - 1)))
                )
                for opt in self._flow_optimizers(trainer):
                    for pg in opt.param_groups:
                        pg["lr"] = base_lr * cos_factor
            # Constant-reduced mode: LR was already set in _reset_flow_lr, nothing to do.

            x0, x1 = trainer.current_coupling.coupling_sample(batch_size)
            x0, x1 = x0.clone().to(device), x1.clone().to(device)

            loss, model, optimizer = self._loss_fn.execute(
                x0,
                x1,
                trainer.strategy,
                trainer.flow_matching,
                iteration,
                trainer.config.iterations,
                trainer._sigma2,
                backward=backward,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"CFM Loss": loss.item()})

        trainer.visualizer.at_flow_matching_end(iteration, trainer, flow_ctx, start_ctx)

        # Transport cost diagnostic — always evaluate on new marginal samples so
        # that T(x) is well-defined and the metric tracks adaptation progress.
        n_diag = 10000 if not is_image else min(256, batch_size)
        if backward:
            xt_tc = trainer.new_target.sample(n_diag).to(device)
        else:
            xt_tc = trainer.new_source.sample(n_diag).to(device)

        model.eval()
        _evolved = trainer.flow_matching.evolve(xt_tc, model, num_steps=100, sigma=trainer._sigma, extend=True)
        yt = (
            _evolved[:, : xt_tc.shape[1], :, :]
            if is_image_tensor(xt_tc)
            else _evolved[:, : trainer.strategy.config.dimension]
        )
        transport_cost = trainer.visualizer.transport_cost(xt_tc.cpu(), yt.cpu())
        if transport_cost is not None:
            cost_flat = transport_cost.view(transport_cost.shape[0], -1)
            logger.info(f"Cost error {torch.norm(cost_flat, dim=1).mean().item():.6f}")

    # ------------------------------------------------------------------
    # Advance step (postprocess)
    # ------------------------------------------------------------------

    def step_postprocess(self, trainer: Trainer, iteration: int, ctx: dict) -> None:
        backward = ctx["backward"]
        device = trainer.device

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

                t_next = torch.full(
                    (actual, 1),
                    (iteration + 1) / trainer.config.iterations,
                    device=device,
                )
                x_mu_next = self._interpolate(
                    trainer.initial_mu_distribution,
                    trainer.new_source,
                    trainer,
                    actual,
                    t_next,
                    trainer._sigma,
                    device,
                )
                y_nu_next = self._interpolate(
                    trainer.initial_nu_distribution,
                    trainer.new_target,
                    trainer,
                    actual,
                    t_next,
                    trainer._sigma,
                    device,
                )

                x_next, y_next = self._advance_chunk(
                    trainer,
                    x_mu_next,
                    y_nu_next,
                    flow_fwd,
                    flow_bwd,
                    backward,
                )
                trainer.current_coupling.update_chunk(i, x_next.detach().cpu(), y_next.detach().cpu())

        trainer.visualizer.at_postprocess_end(iteration, trainer, ctx)

    def _advance_chunk(
        self,
        trainer: Trainer,
        x_mu_next: torch.Tensor,
        y_nu_next: torch.Tensor,
        flow_fwd,
        flow_bwd,
        backward: bool,
    ):
        """Evolve one chunk through the flow model. Returns (x_next, y_next)."""
        num_steps = trainer.config.advance_steps
        # sigma2 = trainer._sigma2
        dim = trainer.strategy.config.dimension

        if not backward:
            x_flow = x_mu_next.clone()
            _evolved = trainer.flow_matching.evolve(x_flow, flow_fwd, num_steps=num_steps, extend=True)
            y_flow = _evolved[:, : x_flow.shape[1], :, :] if is_image_tensor(x_flow) else _evolved[:, :dim]
            # y_flow = y_flow + sigma2 * torch.randn_like(y_flow)
            return x_flow, y_flow
        else:
            y_flow = y_nu_next.clone()
            _evolved = trainer.flow_matching.evolve(y_flow, flow_bwd, num_steps=num_steps, extend=True)
            x_flow = _evolved[:, : y_flow.shape[1], :, :] if is_image_tensor(y_flow) else _evolved[:, :dim]
            # x_flow = x_flow + sigma2 * torch.randn_like(x_flow)
            return x_flow, y_flow
