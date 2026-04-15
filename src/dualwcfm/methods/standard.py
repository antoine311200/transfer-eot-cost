from __future__ import annotations

import logging
import copy
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import trange

from dualwcfm.losses import FlowLoss
from dualwcfm.strategy import BidirectionalStrategy
from dualwcfm.utils import is_image_tensor
from dualwcfm.distributions.coupling import DiskCouplingDistribution
from dualwcfm.methods.base import Method, MethodConfig

if TYPE_CHECKING:
    from dualwcfm.trainer import Trainer

logger = logging.getLogger("dualwcfm")



class OTPlanSampler:
    """Mini-batch OT plan sampler following Tong et al. (2023).

    Computes the Wasserstein-2 optimal transport plan between two mini-batches
    using the POT library and resamples pairs (x0[i], x1[j]) according to the plan.

    This avoids the degenerate fixed point that arises when (x0, x1) are sampled
    independently for conditional bridge training: with OT coupling,
    E[x1 - x0 | x0] = T_OT(x0) - x0 ≠ 0, so the model learns a non-trivial
    velocity at the ODE starting condition xt = x0.

    Reference: A. Tong et al., "Improving and Generalizing Flow Matching", NeurIPS 2023.
               https://github.com/atong01/conditional-flow-matching
    """

    def __init__(self, method: str = "exact", reg: float = 0.05, normalize_cost: bool = False):
        """
        Parameters
        ----------
        method : "exact" | "sinkhorn"
            "exact"    – Wasserstein-2 via POT ``ot.emd`` (LP, deterministic).
            "sinkhorn" – Entropic regularised OT via ``ot.sinkhorn`` (faster, approximate).
        reg : float
            Entropic regularisation strength (only used when method="sinkhorn").
        normalize_cost : bool
            Divide the cost matrix by its maximum before solving (improves numerical
            stability for large-scale data).
        """
        try:
            import ot as pot
        except ImportError:
            raise ImportError(
                "POT (Python Optimal Transport) is required for OT pretraining. "
                "Install it with:  pip install POT"
            )
        self._pot = pot
        self.method = method
        self.reg = reg
        self.normalize_cost = normalize_cost

        if method == "exact":
            self._ot_fn = lambda a, b, M: pot.emd(a, b, M)
        elif method == "sinkhorn":
            self._ot_fn = lambda a, b, M: pot.sinkhorn(a, b, M, reg=reg)
        else:
            raise ValueError(f"Unknown OT method '{method}'. Choose 'exact' or 'sinkhorn'.")

    def get_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        """Return the OT transport plan π of shape (N, N).

        Cost is squared Euclidean (W2 ground metric).  Multi-dimensional inputs
        (e.g. images) are automatically flattened for the cost computation.
        """
        n = x0.shape[0]
        a = self._pot.unif(n)
        b = self._pot.unif(n)

        x0_flat = x0.detach().view(n, -1).float()
        x1_flat = x1.detach().view(n, -1).float()

        M = torch.cdist(x0_flat, x1_flat) ** 2
        if self.normalize_cost:
            M = M / (M.max() + 1e-8)
        M_np = M.cpu().numpy()

        try:
            pi = self._ot_fn(a, b, M_np)
        except Exception as e:
            logger.warning("OT solver failed (%s); falling back to uniform plan.", e)
            pi = np.outer(a, b)

        if not np.isfinite(pi).all():
            logger.warning("OT plan contained non-finite values; falling back to uniform plan.")
            pi = np.outer(a, b)

        return pi

    def sample_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Resample N pairs (x0[i], x1[j]) drawn from the OT plan π.

        Sampling is with replacement (standard mini-batch OT CFM).  Each returned
        batch has the same size N as the inputs, but pairs are now approximately
        OT-coupled rather than independent.
        """
        n = x0.shape[0]
        pi = self.get_plan(x0, x1)

        p = pi.flatten()
        p = p / p.sum()

        choices = np.random.choice(n * n, p=p, size=n, replace=True)
        i, j = np.divmod(choices, n)

        return x0[i], x1[j]

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
    gauge_ema_momentum: float = 0.99  # EMA momentum for gauge fix (used when batch_size < 512)
    dual_warmup_epochs: int = 0       # Extra dual epochs at iteration 0 to warm-start f and g (0 = disabled)

    # Mini-batch OT pretraining (Tong et al., 2023).
    # When True, (x0, x1) and (y0, y1) pairs in bridge pretraining are resampled
    # from the W2 optimal transport plan instead of being drawn independently.
    # This ensures E[x1-x0 | x0] ≠ 0 and gives the bridge a well-defined
    # non-trivial target velocity, which is especially important with conditional
    # bridges but also speeds up convergence for unconditional ones.
    use_ot_pretrain: bool = True
    ot_method: str = "exact"          # "exact" (W2 via EMD) or "sinkhorn" (entropic OT)
    ot_reg: float = 0.05              # Entropic regularisation (only used when ot_method="sinkhorn")
    ot_normalize_cost: bool = False   # Normalise cost matrix by its max before solving

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

    @property
    def _use_conditional_bridge(self) -> bool:
        """Read the conditional-bridge flag from the strategy config at call time."""
        # Accessed lazily so no trainer reference is needed at construction time.
        return False  # overridden per-call via trainer.strategy.config

    def _bridge_call(self, model, t, xt, x0, conditional: bool):
        """Call a bridge model with or without x0 conditioning.

        When conditional, uses the extend convention: cat([xt, x0]) is passed as
        a single 2-arg input, mirroring how the flow model receives cat([x, cond]).
        The model wrapper then appends t, giving MLP input cat([xt, x0, t]).
        """
        if conditional:
            return model(t, torch.cat([xt, x0], dim=1))
        return model(t, xt)

    @staticmethod
    def _ot_pair(a: torch.Tensor, b: torch.Tensor) -> tuple:
        """Approximately OT-pair two batches via random-projection sorting.

        Projects both onto a shared random direction, ranks them, and pairs
        samples with matching ranks.  This creates an approximate bijection
        that is O(N log N) and fully GPU-compatible.

        With OT-coupled pairs the conditional bridge model learns
        E[b - a | a, yt, t] = T_approx(a) - a ≠ 0 at t=0,
        avoiding the degenerate zero-velocity fixed point that arises when
        a and b are sampled independently (where the model reduces to
        computing (yt - a)/t, which is 0 at the ODE starting point yt=a).
        """
        n = a.shape[0]
        d = a.view(n, -1).shape[1]
        direction = torch.randn(d, device=a.device)
        direction = direction / direction.norm()

        proj_a = a.view(n, -1) @ direction   # [N]
        proj_b = b.view(n, -1) @ direction   # [N]

        a_rank = proj_a.argsort()            # indices that sort a
        b_rank = proj_b.argsort()            # indices that sort b
        a_inv_rank = a_rank.argsort()        # rank of each a[i]

        b_paired = b[b_rank[a_inv_rank]]     # b sample with same projection rank as a[i]
        return a, b_paired

    @staticmethod
    def _evolve_bridge_conditional(x0: torch.Tensor, model, num_steps: int, device) -> torch.Tensor:
        """Euler-integrate a *conditional* bridge ODE using the extend convention.

        Mirrors how the forward/backward flow models are evolved: the extended
        state is cat([x_state, x0_frozen], dim=1), the model is called 2-arg as
        model(t, xt_full), and only the first half (the state) is updated.
        """
        n_ch = x0.shape[1]                            # C for images, D for vectors
        xt = torch.cat([x0, x0], dim=1)               # [state=x0, condition=x0]
        dt = 1.0 / num_steps
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((x0.shape[0], 1), i * dt, device=device)
                vt = model(t, xt)                     # 2-arg call, outputs n_ch channels
                state = xt[:, :n_ch] + vt[:, :n_ch] * dt
                xt = torch.cat([state, xt[:, n_ch:]], dim=1)  # condition stays frozen
        return xt[:, :n_ch]

    def _pretrain_bridges(self, trainer: Trainer, retrain: bool = True, save: str = None) -> None:
        """Train (or load) mu/nu bridge models for marginal flow matching."""
        sigma = trainer._sigma
        batch_size = trainer.config.batch_size if not self.config.use_ot_pretrain else min(trainer.config.batch_size, 256+128)
        device = trainer.device
        loss_fn = nn.MSELoss()
        conditional = getattr(trainer.strategy.config, "use_conditional_bridge", False)

        # Build OT sampler once if requested (POT import validated at construction).
        use_ot = self.config.use_ot_pretrain
        ot_sampler: Optional[OTPlanSampler] = None
        if use_ot:
            ot_sampler = OTPlanSampler(
                method=self.config.ot_method,
                reg=self.config.ot_reg,
                normalize_cost=self.config.ot_normalize_cost,
            )
            logger.info(
                "OT pretraining enabled (method=%s). Pairs will be resampled from "
                "the mini-batch W2 OT plan each epoch.",
                self.config.ot_method,
            )

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
                # Keep a constant eta_min floor across rectified-flow phases so subsequent
                # phases still have enough LR to correct residual bridge errors.
                eta_min = 1e-6
                mu_scheduler = lr_scheduler.CosineAnnealingLR(mu_optimizer, T_max=n_epochs, eta_min=eta_min)
                nu_scheduler = lr_scheduler.CosineAnnealingLR(nu_optimizer, T_max=n_epochs, eta_min=eta_min)

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
                        # if conditional:
                        #     # For conditional bridges, integrate with x0 fixed as context.
                        #     x1 = self._evolve_bridge_conditional(x0, previous_mu_model, trainer.config.advance_steps, device)
                        #     y1 = self._evolve_bridge_conditional(y0, previous_nu_model, trainer.config.advance_steps, device)
                        # else:
                        with torch.no_grad():
                            x1 = trainer.flow_matching.evolve(x0, previous_mu_model, num_steps=trainer.config.advance_steps, extend=conditional)
                            y1 = trainer.flow_matching.evolve(y0, previous_nu_model, num_steps=trainer.config.advance_steps, extend=conditional)
                            # evolve() with extend=True returns [N, 2*dim]; extract state half.
                            if conditional:
                                x1 = x1[:, :x0.shape[1]]
                                y1 = y1[:, :y0.shape[1]]

                    x0, x1, y0, y1 = (
                        x0.to(device),
                        x1.to(device),
                        y0.to(device),
                        y1.to(device),
                    )

                    # Pair (x0,x1) and (y0,y1) so the bridge learns a non-trivial
                    # non-degenerate conditional velocity.  Two modes:
                    #
                    #   use_ot_pretrain=True  – W2 OT plan via POT (Tong et al. 2023).
                    #                          Applies every epoch and to all RF phases.
                    #                          Works for both conditional and unconditional bridges.
                    #
                    #   use_conditional_bridge=True, use_ot_pretrain=False
                    #                        – Cheap random-projection sorting fallback.
                    #                          Only applied in phase i=0 (sufficient for RF=0).
                    #
                    # Without any pairing + conditional=True, independent sampling causes
                    # the model to learn v ≈ (yt-y0)/t, which equals 0 at yt=y0 (ODE start).
                    if use_ot and ot_sampler is not None:
                        with torch.no_grad():
                            x0, x1 = ot_sampler.sample_plan(x0, x1)
                            y0, y1 = ot_sampler.sample_plan(y0, y1)
                    elif conditional and i == 0:
                        with torch.no_grad():
                            x0, x1 = self._ot_pair(x0, x1)
                            y0, y1 = self._ot_pair(y0, y1)

                    # Sample random time and interpolate to get (x_t, y_t) representing μ_t and ν_t
                    t = trainer.flow_matching.random_time(batch_size, device)
                    xt = trainer.flow_matching.interpolate(x0, x1, t, sigma / (1 if i == 0 else 10))
                    yt = trainer.flow_matching.interpolate(y0, y1, t, sigma / (1 if i == 0 else 10))

                    mu_optimizer.zero_grad()
                    nu_optimizer.zero_grad()

                    ut = self._bridge_call(mu_model, t, xt, x0, conditional)
                    # Bridge pretrain uses plain flow matching (noisy=False) regardless of
                    # self.use_noisy_field.  The noisy correction term (1-2t)/(2t(1-t))*sigma*noise
                    # has std ≈ sigma_normalized / (2*sqrt(eps)) near t=0, which can equal or exceed
                    # the drift x1-x0 when normalized data spans small coordinates (e.g. 3D with
                    # scale_factor=50).  The result is SNR ≈ 1:1, making the bridge fail to converge.
                    # The score correction is only meaningful for the main SDE-based CFM training.
                    ut_true = trainer.flow_matching.vector_field(
                        xt, t, x0, x1, extend=False, noisy=False
                    )
                    loss_mu = loss_fn(ut, ut_true)
                    loss_mu.backward()
                    nn.utils.clip_grad_norm_(mu_model.parameters(), max_norm=1.0)
                    mu_optimizer.step()

                    vt = self._bridge_call(nu_model, t, yt, y0, conditional)
                    vt_true = trainer.flow_matching.vector_field(
                        yt, t, y0, y1, extend=False, noisy=False
                    )
                    loss_nu = loss_fn(vt, vt_true)
                    loss_nu.backward()
                    nn.utils.clip_grad_norm_(nu_model.parameters(), max_norm=1.0)
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

        n_epochs = trainer.config.pretrain_epochs
        # Decay to lr_min_factor of the base LR — matches the floor used in tuning iterations.
        base_lr = trainer.strategy.config.flow_learning_rate
        eta_min = base_lr * self.config.lr_min_factor
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=eta_min)

        pbar = trange(n_epochs, desc="Pretraining")
        for epoch in pbar:
            x0, x1 = trainer.initial_coupling.coupling_sample(batch_size)
            x0, x1 = x0.to(device), x1.to(device)
            if backward:
                x0, x1 = x1, x0

            s = trainer.flow_matching.random_time(batch_size, device)
            x_s = trainer.flow_matching.interpolate(x0, x1, s, trainer._sigma2)

            optimizer.zero_grad()
            u_s = model(s, x_s, x0)
            # Flow pretrain uses noisy=False for the same reason as bridge pretrain:
            # the (1-2t)/(2t(1-t))*sigma*noise correction term has std ~sigma/(2*sqrt(eps))
            # near t=0, which can equal or exceed the drift x1-x0 on normalized data.
            # The score correction is only meaningful for the main SDE-based CFM training.
            u_s_true = trainer.flow_matching.vector_field(x_s, s, x0, x1, noisy=False)
            loss = loss_fn(u_s, u_s_true)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"Pretrain Loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
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

        use_ema_gauge = batch_size < 512
        ema_a_mean = torch.tensor(0.0, device=device)
        ema_b_mean = torch.tensor(0.0, device=device)
        gauge_mom = self.config.gauge_ema_momentum

        warmup = self.config.dual_warmup_epochs if iteration == 0 else 0
        n_dual_epochs = trainer.config.dual_epochs + warmup
        pbar = trange(n_dual_epochs, desc="Dual Training" + (" (warmup)" if warmup else ""))
        for _ in pbar:
            x, y = trainer.current_coupling.coupling_sample(batch_size)
            x, y = x.to(device), y.to(device)

            # Sample marginals at time t, (x_mu, y_nu) ~ (μ_t, ν_t)
            # x0/y0 are the un-interpolated starting points used by conditional bridges.
            x_mu, y_nu, x0_bridge, y0_bridge = self._sample_marginals(trainer, t_tensor, backward, batch_size, device)
            conditional = x0_bridge is not None

            with torch.no_grad():
                u_t = self._bridge_call(mu_model, t_tensor, x_mu, x0_bridge, conditional)
                v_t = self._bridge_call(nu_model, t_tensor, y_nu, y0_bridge, conditional)

            x, y = x.requires_grad_(True), y.requires_grad_(True)
            x_mu, y_nu = x_mu.requires_grad_(True), y_nu.requires_grad_(True)

            a, a_mu = f_dual(x), f_dual(x_mu)
            b, b_nu = g_dual(y), g_dual(y_nu)

            # Gauge invariance: subtract mean of f to fix the additive constant.
            # With small batches the batch mean is too noisy; use an EMA instead.
            if use_ema_gauge:
                ema_a_mean = gauge_mom * ema_a_mean + (1 - gauge_mom) * a.detach().mean()
                ema_b_mean = gauge_mom * ema_b_mean + (1 - gauge_mom) * b.detach().mean()
                a_mean = ema_a_mean
                b_mean = ema_b_mean
            else:
                a_mean = a.detach().mean()
                b_mean = b.detach().mean()

            a, a_mu = a - a_mean, a_mu - a_mean
            b, b_nu = b - b_mean, b_nu - b_mean

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
                # gp_f = ((grad_a.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
                # gp_g = ((grad_b.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
            else:
                a_term = torch.mean(torch.sum(grad_a * u_t, dim=1))
                b_term = torch.mean(torch.sum(grad_b * v_t, dim=1))
                # gp_f = ((grad_a.norm(2, dim=1) - 1) ** 2).mean()
                # gp_g = ((grad_b.norm(2, dim=1) - 1) ** 2).mean()

            # Regularization terms
            # penalty = torch.tensor(0.0, device=device)
            # if self.config.l2_reg > 0 or self.config.lip_reg > 0 or self.config.mean_reg > 0:
            #     mean_a = torch.mean(a)
            #     mean_b = torch.mean(b)
            #     mean_a_sq = torch.mean(a**2)
            #     mean_b_sq = torch.mean(b**2)
            #     mean_a_mu_sq = torch.mean(a_mu**2)
            #     mean_b_nu_sq = torch.mean(b_nu**2)
            #     var_a = mean_a_sq - mean_a**2
            #     var_b = mean_b_sq - mean_b**2

            #     l2_reg = self.config.l2_reg * (mean_a_sq + mean_b_sq + mean_a_mu_sq + mean_b_nu_sq)
            #     lip_reg = self.config.lip_reg * (gp_f + gp_g)
            #     drift_penalty = self.config.mean_reg * (mean_a**2 + mean_b**2)
            #     var_penalty = self.config.mean_reg * (var_a + var_b)

            #     penalty = l2_reg + lip_reg + drift_penalty + var_penalty
            penalty = torch.tensor(0.0, device=device)
            if self.config.l2_reg > 0 or self.config.lip_reg > 0:
                l2_val = (a**2).mean() + (b**2).mean() + (a_mu**2).mean() + (b_nu**2).mean()

                # grad_energy = (grad_a.pow(2).sum(dim=1).mean() + grad_b.pow(2).sum(dim=1).mean())
                # Create interpolated samples to "fill the gaps" between modes
                # alpha = torch.rand(batch_size, 1, 1, 1, device=device) # Adjust dims for images
                # x_interp = (alpha * x + (1 - alpha) * x_mu).requires_grad_(True)
                # a_interp = f_dual(x_interp)

                # grad_interp = torch.autograd.grad(a_interp.sum(), x_interp, create_graph=True)[0]
                # grad_energy = (grad_interp.norm(2, dim=1)**2).mean()
                # This forces the "bridge" to be smooth even between the modes.

                alpha = torch.rand(batch_size, 1, device=device)
                if is_image:
                    alpha = alpha.view(-1, 1, 1, 1)

                # We detach the base points to ensure we only backprop through f_dual
                x_hat = (alpha * x.detach() + (1 - alpha) * x_mu.detach()).requires_grad_(True)
                y_hat = (alpha * y.detach() + (1 - alpha) * y_nu.detach()).requires_grad_(True)

                # 2. Forward pass at interpolated points
                a_hat = f_dual(x_hat)
                b_hat = g_dual(y_hat)

                # 3. The "Stable" Gradient Calculation
                # Note: Use a_hat.sum() for the graph, and create_graph=True for the penalty
                grad_a_hat = torch.autograd.grad(a_hat.sum(), x_hat, create_graph=True, only_inputs=True)[0]
                grad_b_hat = torch.autograd.grad(b_hat.sum(), y_hat, create_graph=True, only_inputs=True)[0]

                # 4. Final Penalty (Squared Norm centered at 0)
                gp_f = grad_a_hat.view(batch_size, -1).pow(2).sum(dim=1).mean()
                gp_g = grad_b_hat.view(batch_size, -1).pow(2).sum(dim=1).mean()

                grad_energy = gp_f + gp_g

                penalty = (self.config.l2_reg * l2_val) + (self.config.lip_reg * grad_energy)

            loss = coupling_term - 2 * a_term - 2 * b_term
            loss += penalty

            dual_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(f_dual.parameters()) + list(g_dual.parameters()), max_norm=1.0)
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
        """Return (x_mu, y_nu, x0, y0) samples representing μ_t and ν_t.

        x0 / y0 are the un-interpolated starting points; they are needed when
        use_conditional_bridge=True so the bridge can be called as model(t, xt, x0).
        They are returned as None when conditional bridges are not in use.
        """
        conditional = getattr(trainer.strategy.config, "use_conditional_bridge", False)
        x0 = trainer.initial_coupling.sample_init(batch_size).to(device)
        x1 = trainer.new_source.sample(batch_size).to(device)
        y0 = trainer.initial_coupling.sample_target(batch_size).to(device)
        y1 = trainer.new_target.sample(batch_size).to(device)
        x_mu = trainer.flow_matching.interpolate(x0, x1, t_tensor, sigma=trainer._sigma)
        y_nu = trainer.flow_matching.interpolate(y0, y1, t_tensor, sigma=trainer._sigma)
        if conditional:
            return x_mu, y_nu, x0, y0
        return x_mu, y_nu, None, None

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

        # Map-error diagnostic — same metric as map_err_fwd / map_err_bwd in W&B.
        # Evaluates sqrt(E[||T(x) - y_hat||²]) or sqrt(E[||T^{-1}(y) - x_hat||²])
        # in original (denormalized) coordinates on new-marginal samples.
        n_diag = 10000 if not is_image else min(256, batch_size)
        direction = "backward" if backward else "forward"
        transport_fn = (
            getattr(trainer.visualizer, "_inverse_transport_map", None)
            if backward
            else getattr(trainer.visualizer, "_transport_map", None)
        )
        if transport_fn is not None:
            from dualwcfm.metrics import map_error as _map_err
            s = trainer.scale
            if backward:
                xt_diag = trainer.new_target.sample(n_diag).to(device)
            else:
                xt_diag = trainer.new_source.sample(n_diag).to(device)
            model.eval()
            _evolved = trainer.flow_matching.evolve(xt_diag, model, num_steps=100, sigma=trainer._sigma, extend=True)
            dim = trainer.strategy.config.dimension
            yt_diag = (
                _evolved[:, : xt_diag.shape[1], :, :]
                if is_image_tensor(xt_diag)
                else _evolved[:, :dim]
            )
            err = _map_err((xt_diag * s).cpu(), (yt_diag * s).cpu(), transport_fn)
            logger.info(f"Map error ({direction}): {err:.6f}")

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
        dim = trainer.strategy.config.dimension

        if not backward:
            x_flow = x_mu_next.clone()
            _evolved = trainer.flow_matching.evolve(x_flow, flow_fwd, num_steps=num_steps, extend=True)
            y_flow = _evolved[:, : x_flow.shape[1], :, :] if is_image_tensor(x_flow) else _evolved[:, :dim]
            return x_flow, y_flow
        else:
            y_flow = y_nu_next.clone()
            _evolved = trainer.flow_matching.evolve(y_flow, flow_bwd, num_steps=num_steps, extend=True)
            x_flow = _evolved[:, : y_flow.shape[1], :, :] if is_image_tensor(y_flow) else _evolved[:, :dim]
            return x_flow, y_flow
