import copy
import os

import torch
import torch.nn as nn

from tqdm import trange
import matplotlib.pyplot as plt

from dualwcfm.config import TrainingConfig
from dualwcfm.utils import is_image_tensor
from dualwcfm.flow_matching import ConditionalFlowMatching
from dualwcfm.strategy import Strategy, BidirectionalStrategy
from dualwcfm.losses import CFMLoss
from dualwcfm.visualization import Visualizer
from dualwcfm.distributions.coupling import (
    CouplingDistribution,
    FixedSizeCouplingDistribution,
    DiskCouplingDistribution,
)
from dualwcfm.distributions.distribution import Distribution, FixedSizeDistribution
from dualwcfm.distributions.scaled import ScaledCouplingDistribution, ScaledDistribution


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
        device: torch.device = torch.device("cpu"),
    ):

        self.config = config
        self.device = device

        # Compute normalization scale: max absolute value across a probe batch
        # of all four marginals so every sample lies in [-1, 1] after scaling.
        if self.config.scale_factor > 0:
            self.scale = self.config.scale_factor
            print(f"Using provided normalization scale: {self.scale:.4f}")
        else:
            with torch.no_grad():
                _probe_n = 1000
                _x0, _x1 = initial_coupling.coupling_sample(_probe_n)
                _s = new_source.sample(_probe_n)
                _t = new_target.sample(_probe_n)
                self.scale = max(
                    _x0.abs().max().item(),
                    _x1.abs().max().item(),
                    _s.abs().max().item(),
                    _t.abs().max().item(),
                )

            print(f"Normalization scale set to {self.scale:.4f} based on probe samples.")
        self.initial_coupling = ScaledCouplingDistribution(initial_coupling, self.scale)
        self.new_source = ScaledDistribution(new_source, self.scale)
        self.new_target = ScaledDistribution(new_target, self.scale)
        # self.initial_coupling = initial_coupling
        # self.new_source = new_source
        # self.new_target = new_target

        # Scale noises
        self.config.sigma = config.sigma / self.scale
        self.config.sigma2 = config.sigma2 / self.scale

        self.visualizer = visualizer
        self.strategy = strategy
        self.flow_matching = flow_matching
        self.loss_cfm = CFMLoss.get_loss(config.loss)

        # AMP scaler for mixed precision
        self._amp_enabled = config.use_amp and device.type == "cuda"
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)

        # Fixed sized mu_t, nu_t and coupling distributions
        self._init_distributions(self.initial_coupling)
        self.last_correction_time = 0.0

    def _autocast(self):
        """Return autocast context manager (no-op on CPU or when AMP is disabled)."""
        return torch.cuda.amp.autocast(enabled=self._amp_enabled)

    def _init_distributions(self, initial_coupling: CouplingDistribution):
        mu_distribution, nu_distribution = initial_coupling.get_marginals()

        # When advancing with flow, track marginal particles in a fixed-size dataset
        # (Euler-stepped each iteration, like the reference notebook)
        if self.config.advance_with_flow:
            self.mu_distribution = FixedSizeDistribution(mu_distribution, size=self.config.dataset_size)
            self.nu_distribution = FixedSizeDistribution(nu_distribution, size=self.config.dataset_size)
        else:
            self.mu_distribution = mu_distribution
            self.nu_distribution = nu_distribution

        # Use disk-based coupling for images to avoid OOM
        if self.config.use_disk_coupling or (self.config.image_shape is not None and self.config.dataset_size > 1000):
            print("Using disk-based coupling distribution for memory efficiency.")
            self.current_coupling = DiskCouplingDistribution(
                initial_coupling,
                size=self.config.dataset_size,
                save_dir=self.config.coupling_save_dir,
                chunk_size=self.config.coupling_chunk_size,
                device=self.device,
            )
        else:
            self.current_coupling = FixedSizeCouplingDistribution(initial_coupling, size=self.config.dataset_size)

        # Copy to Initial distribution for later use (clone so training mutations don't affect the template)
        self.initial_mu_distribution = self.mu_distribution.clone()
        self.initial_nu_distribution = self.nu_distribution.clone()
        self.initial_coupling_distribution = self.current_coupling

    def _load_pretrained_models(self, dir):
        # Load pretrained models if available
        forward_state_dict = torch.load(f"{dir}/initial_forward_model.pth", map_location=self.device)

        self.strategy.forward_model.load_state_dict(forward_state_dict)
        self.strategy.initial_forward_model.load_state_dict(forward_state_dict)

        if isinstance(self.strategy, BidirectionalStrategy):
            backward_state_dict = torch.load(f"{dir}/initial_backward_model.pth", map_location=self.device)
            self.strategy.backward_model.load_state_dict(backward_state_dict)
            self.strategy.initial_backward_model.load_state_dict(backward_state_dict)

        mu_state_dict = torch.load(f"{dir}/initial_mu_model.pth", map_location=self.device)
        nu_state_dict = torch.load(f"{dir}/initial_nu_model.pth", map_location=self.device)
        self.strategy.mu_bridge.load_state_dict(mu_state_dict)
        self.strategy.initial_mu_bridge.load_state_dict(mu_state_dict)
        self.strategy.nu_bridge.load_state_dict(nu_state_dict)
        self.strategy.initial_nu_bridge.load_state_dict(nu_state_dict)
        print("Pretrained models loaded.")

    def train(self, retrain: bool = False, save: str = None):
        self.strategy._create_models(self.strategy.config.dimension, self.strategy.config.hidden_size)
        self.strategy._create_optimizers(self.strategy.config)
        # Move all models to device once
        self.strategy.to_device(self.device)

        if not retrain:
            self.mu_distribution = self.initial_mu_distribution.clone()
            self.nu_distribution = self.initial_nu_distribution.clone()
            self.current_coupling = self.initial_coupling_distribution.clone()

        # Marginal flow matching phase
        self.train_marginal_flows(retrain, save)

        # Pretrain phase
        if isinstance(self.strategy, BidirectionalStrategy) or hasattr(self.strategy, "backward_model"):
            print("Pretraining bidirectional flows.")
            forward_model, forward_optimizer = self.strategy.get_flow(0, self.device)
            backward_model, backward_optimizer = self.strategy.get_flow(1, self.device)
            initial_forward_model = self.strategy.get_initial_flow(0, self.device)
            initial_backward_model = self.strategy.get_initial_flow(1, self.device)

            if retrain:
                self.pretrain(forward_model, forward_optimizer, backward=False)
                self.pretrain(backward_model, backward_optimizer, backward=True)
                initial_forward_model.load_state_dict(forward_model.state_dict())
                initial_backward_model.load_state_dict(backward_model.state_dict())

                # Save initial models
                if save:
                    os.makedirs(save, exist_ok=True)
                    torch.save(
                        initial_forward_model.state_dict(),
                        f"{save}/initial_forward_model.pth",
                    )
                    torch.save(
                        initial_backward_model.state_dict(),
                        f"{save}/initial_backward_model.pth",
                    )
            else:
                print("Skipping pretraining.")
                print("Loading initial flow models.")

                if save:
                    self._load_pretrained_models(save)
                else:
                    forward_model.load_state_dict(initial_forward_model.state_dict())
                    backward_model.load_state_dict(initial_backward_model.state_dict())
        else:
            print("Pretraining unidirectional flow.")
            model, optimizer = self.strategy.get_flow(0, self.device)
            initial_model = self.strategy.get_initial_flow(0, self.device)
            if retrain:
                self.pretrain(model, optimizer)
                initial_model.load_state_dict(model.state_dict())

                if save:
                    os.makedirs(save, exist_ok=True)
                    torch.save(
                        initial_model.state_dict(),
                        f"{save}/initial_forward_model.pth",
                    )
            else:
                print("Skipping pretraining.")
                print("Loading initial flow model.")
                if save:
                    self._load_pretrained_models(save)
                else:
                    model.load_state_dict(initial_model.state_dict())

        n_vis = 6 if self.config.image_shape else 2500
        start_points = self.mu_distribution.sample(n_vis)
        start_points = start_points.to(self.device)
        end_points = self.flow_matching.evolve(
            start_points, self.strategy.get_flow(0, self.device)[0], num_steps=50, sigma=self.config.sigma, extend=True
        )
        # Extract first half of output (evolved part, not frozen condition)
        if is_image_tensor(end_points):
            end_points = end_points[:, : start_points.shape[1], :, :]
        else:
            end_points = end_points[:, : self.strategy.config.dimension]
        target_points = self.nu_distribution.sample(n_vis)
        target_points = target_points.to(self.device)
        self.visualizer._display_evolution(
            (start_points * self.scale).cpu(),
            (end_points * self.scale).cpu(),
            (target_points * self.scale).cpu(),
        )
        plt.show()

        # Divide by 10 learning rates for cfm
        # for param_group in self.strategy.get_flow(0, self.device)[1].param_groups:
        #     param_group["lr"] = param_group["lr"] / 5.0
        # if isinstance(self.strategy, BidirectionalStrategy):
        #     for param_group in self.strategy.get_flow(1, self.device)[1].param_groups:
        #         param_group["lr"] = param_group["lr"] / 5.0

        print("Is bidirectional?", isinstance(self.strategy, BidirectionalStrategy))
        # General training loop
        for iteration in range(self.config.iterations):
            print(f"Iteration {iteration+1}/{self.config.iterations}")
            # self.plot_distributions_after_advance(iteration)
            # Error-accumulation correction: retrain bridges from nu_t_errors → nu_1,
            # then use corrected bridges to refine dual networks f and g.
            if self.config.correction_each > 0 and (iteration + 1) % self.config.correction_each == 0 and iteration > 0:
                print(f"\n################# Correction step at iteration {iteration + 1} #################")
                self.correction_step(iteration)
            # Add a correction after the previous one if bidirectional to correct both forward and backward models
            if isinstance(self.strategy, BidirectionalStrategy) and self.config.correction_each > 0 and iteration % self.config.correction_each == 0 and iteration > 0:
                print(f"\n################# Second correction step for bidirectional strategy at iteration {iteration + 1} #################")
                self.correction_step(iteration)

            self.train_duals_step(iteration)
            self.train_cfm_step(iteration)
            self.advance_step(iteration, backward=isinstance(self.strategy, BidirectionalStrategy))

            # Analyse marginal drift after each iteration
            distances = self.visualizer.analyse(iteration, self)
            print(
                f"  SW distances — "
                f"W1_mu={distances['W1_mu']:.4f}  W2_mu={distances['W2_mu']:.4f}  "
                f"W1_nu={distances['W1_nu']:.4f}  W2_nu={distances['W2_nu']:.4f}"
            )


    # ------------------------------------------------------------------
    # Error-accumulation correction
    # ------------------------------------------------------------------

    def correction_step(self, time_step: int):
        """Correct for error accumulated in the current coupling.

        Due to first-order exponential-weight approximations the forward model
        maps mu_t → nu_t_with_errors rather than the ideal nu_t.  This method:

        1. Generates a buffer of *nu_t_with_errors* samples by evolving mu_t
           through the current forward model.
        2. Creates *separate* correction bridges (deep-copies of the originals)
           and retrains them from nu_t_with_errors → nu_1 (and mu_t_with_errors
           → mu_1 if bidirectional).  The original bridges used by advance_step
           are **never modified**.
        3. Stores the correction bridges and the error buffers as instance
           variables so that every subsequent call to :meth:`train_duals_step`
           uses local correction time until the next correction point replaces
           them.
        4. Runs :meth:`_correction_dual_step` for an immediate boundary update
           of f and g at correction time s = eps.

        The correction is triggered from the main loop every
        ``config.correction_each`` iterations.
        """
        batch_size = self.config.batch_size
        device = self.config.device
        loss_fn = nn.MSELoss()

        t = time_step / self.config.iterations
        t_tensor = torch.full((batch_size, 1), t, device=device)

        # nu_model, nu_optimizer = self.strategy.get_bridges(device)[1]
        # if bidirectional and even step take nu bridge for backward correction, else take forward nu bridge
        if isinstance(self.strategy, BidirectionalStrategy):
            if time_step % 2 == 0:
                bridge_model, bridge_optimizer = self.strategy.get_bridges(device)[1]
            else:
                bridge_model, bridge_optimizer = self.strategy.get_bridges(device)[0]
        else:
            bridge_model, bridge_optimizer = self.strategy.get_bridges(device)[1]
        model, _ = self.strategy.get_flow(time_step, device)
        model.eval()

        bridge_model.train()

        # Print bridge_model weights before correction for debugging
        # print("bridge_model weights before correction:")
        # for name, param in bridge_model.named_parameters():
        #     print(f"  {name}: {param.mean().item()}")
        #     break

        previous_bridge_model = copy.deepcopy(bridge_model)

        for i in range(self.config.num_rectified_flow + 1):
            print(f"Correction flow matching phase {i + 1}/{self.config.num_rectified_flow + 1}")

            pbar = trange(self.config.correction_epochs if i == 0 else self.config.correction_epochs // 10, desc="Correct Marginal Flow Matching")
            for epoch in pbar:
                if isinstance(self.strategy, BidirectionalStrategy):
                    if time_step % 2 == 0:
                        # Even step: correct forward mu bridge from mu_t_with_errors → mu_1
                        x0 = self.initial_coupling.sample_init(batch_size).to(device)
                        x1 = self.new_source.sample(batch_size).to(device)
                    else:
                        # Odd step: correct backward nu bridge from nu_t_with_errors → nu_1
                        x0 = self.initial_coupling.sample_target(batch_size).to(device)
                        x1 = self.new_target.sample(batch_size).to(device)
                else:
                    x0 = self.initial_coupling.sample_init(batch_size)
                    x1 = self.new_source.sample(batch_size)
                x0, x1 = x0.to(device), x1.to(device)

                # Sample from mu_t, then use the forward model to get nu_t_with_errors samples for correction training.
                xt = self.flow_matching.interpolate(x0, x1, t_tensor, self.config.sigma)
                yt = self.flow_matching.evolve(xt, model, num_steps=50, extend=True)[:, : self.strategy.config.dimension]

                if i == 0:
                    if isinstance(self.strategy, BidirectionalStrategy) and time_step % 2 == 1:
                        y1 = self.new_source.sample(batch_size).to(device)
                    else:
                        y1 = self.new_target.sample(batch_size).to(device)
                else:
                    # Rectified flow: use the previous bridge model to generate paired endpoints.
                    _num_steps = 50
                    previous_bridge_model.eval()
                    y1 = self.flow_matching.evolve(
                        yt, previous_bridge_model, num_steps=_num_steps, extend=False,
                        start_index=int(t * _num_steps), start_time=t,
                    )
                    previous_bridge_model.train()

                # Interpolate between nu_t_with_errors and nu_1 to get training samples for the correction bridge.
                # s is local time in [0, 1]; s_dilate is global time in [t, 1].
                s = self.flow_matching.random_time(batch_size, self.config.device)
                zs = self.flow_matching.interpolate(yt, y1, s, self.config.sigma)
                s_dilate = t + s * (1 - t)

                bridge_optimizer.zero_grad()

                with self._autocast():
                    vt = bridge_model(s_dilate, zs)
                    # vector_field returns velocity in local-time units (dx/ds).
                    # Euler integration uses global-time dt, so scale by 1/(1-t) to get dx/d(tau).
                    vt_true = self.flow_matching.vector_field(zs, s, yt, y1, extend=False, noisy=self.config.use_noisy_field) / (1 - t)
                    loss = loss_fn(vt, vt_true)

                if self._amp_enabled:
                    self._scaler.scale(loss).backward()
                    self._scaler.step(bridge_optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    bridge_optimizer.step()

                pbar.set_postfix({"Correction Loss": loss.item()})

            previous_bridge_model.load_state_dict(bridge_model.state_dict())

        self.last_correction_time = t

        # print("bridge_model weights after correction:")
        # for name, param in bridge_model.named_parameters():
        #     print(f"  {name}: {param.mean().item()}")
        #     break

        # Visualize: show how the corrected bridge transports the erroneous marginal to the true target.
        # For even step / unidirectional: mu_t → forward_model → nu_t_err → nu_bridge → nu_1
        # For odd step (bidirectional):   nu_t → backward_model → mu_t_err → mu_bridge → mu_1
        _vis_n = batch_size
        _num_steps = 100
        if isinstance(self.strategy, BidirectionalStrategy) and time_step % 2 == 1:
            # Odd step: backward direction
            p0 = self.initial_coupling.sample_target(_vis_n).to(device)
            p1_samples = self.new_target.sample(_vis_n).to(device)
            pt = self.flow_matching.interpolate(p0, p1_samples, t_tensor, self.config.sigma)
            start = self.flow_matching.evolve(pt, model, num_steps=50, extend=True)[:, :self.strategy.config.dimension]
            ref_target = self.new_source.sample(_vis_n).to(device)
        else:
            # Even step / unidirectional: forward direction
            p0 = self.initial_coupling.sample_init(_vis_n).to(device)
            p1_samples = self.new_source.sample(_vis_n).to(device)
            pt = self.flow_matching.interpolate(p0, p1_samples, t_tensor, self.config.sigma)
            start = self.flow_matching.evolve(pt, model, num_steps=50, extend=True)[:, :self.strategy.config.dimension]
            ref_target = self.new_target.sample(_vis_n).to(device)

        end = self.flow_matching.evolve(
            start,
            bridge_model,
            num_steps=_num_steps,
            extend=False,
            start_index=int(t * _num_steps),
            start_time=t,
            verbose=True,
        )
        self.visualizer._display_evolution(
            (start * self.scale).cpu(),
            (end * self.scale).cpu(),
            (ref_target * self.scale).cpu(),
        )
        plt.show()

    def train_marginal_flows(self, retrain: bool = False, save: str = None):
        sigma = self.config.sigma
        batch_size = self.config.batch_size
        device = self.config.device
        loss_fn = nn.MSELoss()

        (mu_model, mu_optimizer), (nu_model, nu_optimizer) = self.strategy.get_bridges(device)
        initial_mu_model, initial_nu_model = self.strategy.get_initial_bridges(device)

        previous_mu_model = copy.deepcopy(mu_model)
        previous_nu_model = copy.deepcopy(nu_model)

        if retrain:
            mu_model.train()
            nu_model.train()

            for i in range(self.config.num_rectified_flow + 1):
                print(f"Marginal flow matching phase {i + 1}/{self.config.num_rectified_flow + 1}")

                pbar = trange(self.config.marginal_epochs if i == 0 else self.config.marginal_epochs // 10, desc="Marginal Flow Matching")
                for epoch in pbar:
                    if i == 0:
                        x0, y0 = self.initial_coupling.sample_init(batch_size), self.initial_coupling.sample_target(batch_size)
                        x1, y1 = self.new_source.sample(batch_size), self.new_target.sample(batch_size)
                    else:
                        x0 = self.initial_coupling.sample_init(batch_size).to(device)
                        y0 = self.initial_coupling.sample_target(batch_size).to(device)
                        previous_mu_model.eval()
                        previous_nu_model.eval()
                        x1 = self.flow_matching.evolve(x0, previous_mu_model, num_steps=50, extend=False)
                        y1 = self.flow_matching.evolve(y0, previous_nu_model, num_steps=50, extend=False)
                        previous_mu_model.train()
                        previous_nu_model.train()

                    x0, x1, y0, y1 = x0.to(device), x1.to(device), y0.to(device), y1.to(device)

                    t = self.flow_matching.random_time(batch_size, self.config.device)
                    xt = self.flow_matching.interpolate(x0, x1, t, sigma)
                    yt = self.flow_matching.interpolate(y0, y1, t, sigma)

                    mu_optimizer.zero_grad()
                    nu_optimizer.zero_grad()

                    with self._autocast():
                        ut = mu_model(t, xt)
                        ut_true = self.flow_matching.vector_field(xt, t, x0, x1, extend=False, noisy=self.config.use_noisy_field)
                        loss_mu = loss_fn(ut, ut_true)

                    if self._amp_enabled:
                        self._scaler.scale(loss_mu).backward()
                        self._scaler.step(mu_optimizer)
                    else:
                        loss_mu.backward()
                        mu_optimizer.step()

                    with self._autocast():
                        vt = nu_model(t, yt)
                        vt_true = self.flow_matching.vector_field(yt, t, y0, y1, extend=False, noisy=self.config.use_noisy_field)
                        loss_nu = loss_fn(vt, vt_true)

                    if self._amp_enabled:
                        self._scaler.scale(loss_nu).backward()
                        self._scaler.step(nu_optimizer)
                        self._scaler.update()
                    else:
                        loss_nu.backward()
                        nu_optimizer.step()

                    pbar.set_postfix({"Loss mu": loss_mu.item(), "Loss nu": loss_nu.item()})

                    # Visualize marginal flows during training
                    self.visualizer.display_marginal(
                        epoch, self, mu_model, self.mu_distribution, self.new_source, title="Mu Marginal Flow", display=i!=0 and epoch == self.config.marginal_epochs//10 - 1
                    )
                    self.visualizer.display_marginal(
                        epoch, self, nu_model, self.nu_distribution, self.new_target, title="Nu Marginal Flow", display=i!=0 and epoch == self.config.marginal_epochs//10 - 1
                    )

                previous_mu_model.load_state_dict(mu_model.state_dict())
                previous_nu_model.load_state_dict(nu_model.state_dict())

            initial_mu_model.load_state_dict(mu_model.state_dict())
            initial_nu_model.load_state_dict(nu_model.state_dict())

            # Save initial models
            if save:
                os.makedirs(save, exist_ok=True)
                torch.save(
                    initial_mu_model.state_dict(),
                    f"{save}/initial_mu_model.pth",
                )
                torch.save(
                    initial_nu_model.state_dict(),
                    f"{save}/initial_nu_model.pth",
                )
                print("Initial marginal flow models saved.")
        else:
            print("Skipping marginal flow matching retraining.")
            print("Loading initial marginal flow models.")

            if save:
                self._load_pretrained_models(save)
            else:
                mu_model.load_state_dict(initial_mu_model.state_dict())
                nu_model.load_state_dict(initial_nu_model.state_dict())

        # Free cuda memory after pretraining
        torch.cuda.empty_cache()

        samples = 2500
        start_points = self.mu_distribution.sample(samples)
        start_points = start_points.to(self.device)

        mu_model.eval()
        end_points = self.flow_matching.evolve(
            start_points, self.strategy.mu_bridge, num_steps=50, sigma=self.config.sigma, extend=False
        )
        target_points = self.new_source.sample(samples)
        target_points = target_points.to(self.device)
        self.visualizer._display_evolution(
            (start_points * self.scale).cpu(),
            (end_points * self.scale).cpu(),
            (target_points * self.scale).cpu(),
        )
        plt.show()

        start_points = self.nu_distribution.sample(samples)
        start_points = start_points.to(self.device)

        nu_model.eval()
        end_points = self.flow_matching.evolve(
            start_points, self.strategy.nu_bridge, num_steps=50, sigma=self.config.sigma, extend=False
        )
        target_points = self.new_target.sample(samples)
        target_points = target_points.to(self.device)
        self.visualizer._display_evolution(
            (start_points * self.scale).cpu(),
            (end_points * self.scale).cpu(),
            (target_points * self.scale).cpu(),
        )
        plt.show()

    def pretrain(self, model, optimizer, backward: bool = False):
        sigma2 = self.config.sigma2
        batch_size = self.config.batch_size
        loss_fn = nn.MSELoss()

        model.train()

        pbar = trange(self.config.pretrain_epochs, desc="Pretraining")

        for epoch in pbar:
            x0, x1 = self.initial_coupling.coupling_sample(batch_size)
            x0, x1 = x0.to(self.config.device), x1.to(self.config.device)
            if backward:
                x0, x1 = x1, x0

            s = self.flow_matching.random_time(batch_size, self.config.device)
            x_s = self.flow_matching.interpolate(x0, x1, s, sigma2)

            optimizer.zero_grad()

            with self._autocast():
                u_s = model(s, x_s, x0)
                u_s_true = self.flow_matching.vector_field(x_s, s, x0, x1, noisy=self.config.use_noisy_field)
                loss = loss_fn(u_s, u_s_true)

            if self._amp_enabled:
                self._scaler.scale(loss).backward()
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                loss.backward()
                optimizer.step()

            pbar.set_postfix({"Pretrain Loss": loss.item()})

            # Visualize pretraining flow
            self.visualizer.display_pretrain(epoch, self, model, backward=backward)

    def train_duals_step(self, time_step: int):
        batch_size = self.config.batch_size
        device = self.config.device
        t_tensor = torch.full((batch_size, 1), time_step / self.config.iterations, device=device)
        is_image = self.config.image_shape is not None

        (mu_model, _), (nu_model, _) = self.strategy.get_bridges(self.device)
        f_dual, g_dual, dual_optimizer = self.strategy.get_duals(self.device)
        forward_model, _ = self.strategy.get_flow(0, self.device)
        f_dual.train()
        g_dual.train()
        mu_model.eval()
        nu_model.eval()
        forward_model.eval()

        # If not after a correction step, train for the full number of epochs.  If after a correction step, train 5 times more epochs to allow duals to fit the corrected coupling.
        # if self.last_correction_time > 0 and time_step > self.last_correction_time * self.config.iterations:

        # If the previous step is a correction step, train for more epochs to allow duals to fit the corrected coupling.
        if self.config.correction_each > 0 and (time_step + 1) % self.config.correction_each == 0 and self.last_correction_time > 0:
            print("Increasing dual training epochs for correction fitting.")
            num_epochs = 5 * self.config.dual_epochs
        else:
            num_epochs = self.config.dual_epochs


        print("Last correction time:", self.last_correction_time)
        print("Current time:", time_step / self.config.iterations)

        x0_test, x1_test = self.current_coupling.coupling_sample(batch_size)
        x0_test, x1_test = x0_test.to(self.config.device), x1_test.to(self.config.device)
        f_dual, g_dual, _ = self.strategy.get_duals(self.device)
        with torch.no_grad():
            a, b = f_dual(x0_test), g_dual(x1_test)
            sm = torch.exp((a+b) / self.config.iterations)
            print(f"   f_dual(x0) mean: {a.mean().item():.4e}, std: {a.std().item():.4e}")
            print(f"   g_dual(x1) mean: {b.mean().item():.4e}, std: {b.std().item():.4e}")
            print(f"   sum mean: {sm.mean().item():.4e}, std:{sm.std().item():.4e}")

        pbar = trange(num_epochs, desc="Dual Training")
        for epoch in pbar:
            x, y = self.current_coupling.coupling_sample(self.config.batch_size)
            x, y = x.to(device), y.to(device)

            if self.config.advance_with_flow:
                x_mu = self.mu_distribution.sample(batch_size).to(device)
                y_nu = self.nu_distribution.sample(batch_size).to(device)
            else:
                # Check if the strategy is bidirectional
                if (isinstance(self.strategy, BidirectionalStrategy) and time_step % 2 == 0) or (not isinstance(self.strategy, BidirectionalStrategy)):
                    x0 = self.initial_coupling.sample_init(batch_size).to(device)
                    x1 = self.new_source.sample(batch_size).to(device)
                    x_mu = self.flow_matching.interpolate(x0, x1, t_tensor, sigma=self.config.sigma)

                    # y_nu: forward model applied to mu_t — same distribution as the y-side of current_coupling
                    y_nu = self.flow_matching.evolve(x_mu, forward_model, num_steps=50, extend=True)[:, :self.strategy.config.dimension]

                    if epoch == 0:
                        print(f"W1 (nu_err vs p1(pi)): {self.visualizer._sliced_wasserstein(y.cpu(), y_nu.cpu(), n_proj=1000, p=1):.4e} | W2 (nu_err vs p1(pi)): {self.visualizer._sliced_wasserstein(y.cpu(), y_nu.cpu(), n_proj=1000, p=2):.4e}")

                else:# elif isinstance(self.strategy, BidirectionalStrategy) and time_step % 2 == 1:
                    y0 = self.initial_coupling.sample_target(batch_size).to(device)
                    y1 = self.new_target.sample(batch_size).to(device)
                    y_nu = self.flow_matching.interpolate(y0, y1, t_tensor, sigma=self.config.sigma)

                    # x_mu: backward model applied to nu_t — same distribution as the x-side of current_coupling (odd step)
                    backward_model, _ = self.strategy.get_flow(1, self.device)
                    x_mu = self.flow_matching.evolve(y_nu, backward_model, num_steps=50, extend=True)[:, :self.strategy.config.dimension]

            with torch.no_grad():
                u_t = mu_model(t_tensor, x_mu)
                v_t = nu_model(t_tensor, y_nu)

            x, y = x.requires_grad_(True), y.requires_grad_(True)
            x_mu, y_nu = x_mu.requires_grad_(True), y_nu.requires_grad_(True)

            a, a_mu = f_dual(x), f_dual(x_mu)
            b, b_nu = g_dual(y), g_dual(y_nu)

            # Coupling term
            # coupling_term = torch.mean((a - a.mean() + b - b.mean()) ** 2)
            coupling_term = torch.mean((a + b) ** 2)

            # Divergence terms
            grad_a = torch.autograd.grad(a_mu.sum(), x_mu, create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad_b = torch.autograd.grad(b_nu.sum(), y_nu, create_graph=True, retain_graph=True, only_inputs=True)[0]

            # For images: sum over all dimensions except batch
            if is_image:
                # grad_a, u_t have shape [batch, C, H, W]
                a_term = torch.mean((grad_a * u_t).sum(dim=[1, 2, 3]))
                b_term = torch.mean((grad_b * v_t).sum(dim=[1, 2, 3]))
                # Compute gradient norm over flattened dimensions
                gp_f = ((grad_a.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
                gp_g = ((grad_b.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
            else:
                a_term = torch.mean(torch.sum(grad_a * u_t, dim=1))
                b_term = torch.mean(torch.sum(grad_b * v_t, dim=1))
                gp_f = ((grad_a.norm(2, dim=1) - 1) ** 2).mean()
                gp_g = ((grad_b.norm(2, dim=1) - 1) ** 2).mean()

            # Regularization (L2 + Lipshitz)
            l2_reg = self.config.l2_reg * (
                torch.mean(a**2) + torch.mean(b**2) + torch.mean(a_mu**2) + torch.mean(b_nu**2)
            )

            lip_reg = self.config.lip_reg * (gp_f + gp_g)

            # # Penalize a-E(a) and b-E(q) to avoid trivial solutions where a and b go to infinity but their sum remains constant
            # mean_a = torch.mean(a)
            # mean_a_mu = torch.mean(a_mu)
            # loss_mean_a = torch.mean((a - mean_a) ** 2)  # Force a to have low variance around its mean
            # loss_mean_b = torch.mean((b + mean_a) ** 2)  # Force b to be close to -a to prevent trivial solutions
            # mean_loss = self.config.mean_reg * (torch.mean(a + b))**2

            mean_a = torch.mean(a)
            mean_b = torch.mean(b)
            drift_penalty = self.config.mean_reg * (mean_a**2 + mean_b**2)
            mean_sum_constraint = self.config.mean_reg * (mean_a + mean_b)**2
            l2_reg_centered = self.config.mean_reg * (torch.mean((a - mean_a)**2) + torch.mean((b - mean_b)**2))

            loss = 0.5 * coupling_term - a_term - b_term
            loss += drift_penalty + l2_reg_centered + lip_reg + l2_reg
            # + l2_reg + lip_reg + drift_penalty + mean_sum_constraint #+ l2_reg_centered
            # loss += 0.1 * mean_a_loss + 0.1 * mean_b_loss + 0.1 * mean_a_mu_loss + 0.1 * mean_b_nu_loss

            dual_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(f_dual.parameters(), max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(g_dual.parameters(), max_norm=10.0)
            dual_optimizer.step()

            pbar.set_postfix({"Dual Loss": loss.item()})

        x0_test, x1_test = self.current_coupling.coupling_sample(batch_size)
        x0_test, x1_test = x0_test.to(self.config.device), x1_test.to(self.config.device)
        f_dual, g_dual, _ = self.strategy.get_duals(self.device)
        with torch.no_grad():
            a, b = f_dual(x0_test), g_dual(x1_test)
            sm = torch.exp((a+b) / self.config.iterations)
            print(f"   f_dual(x0) mean: {a.mean().item():.4f}, std: {a.std().item():.4f}")
            print(f"   g_dual(x1) mean: {b.mean().item():.4f}, std: {b.std().item():.4f}")
            print(f"   sum mean: {sm.mean().item():.4f}, std:{sm.std().item():.4f}")#, ex: {sm[:5]}")
            x0_plot = x0_test.cpu().numpy()
            x1_plot = x1_test.cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        cmap = 'viridis'
        sc0 = axes[0].scatter(x0_plot[:, 0], x0_plot[:, 1], c=sm.detach().cpu().numpy(), cmap=cmap, edgecolors='k', alpha=0.7)
        axes[0].set_title(r"Source $x_0$ colored by $\exp((a+b)/\epsilon)$")
        axes[0].set_xlabel("Dim 1")
        axes[0].set_ylabel("Dim 2")
        sc1 = axes[1].scatter(x1_plot[:, 0], x1_plot[:, 1], c=sm.detach().cpu().numpy(), cmap=cmap, edgecolors='k', alpha=0.7)
        axes[1].set_title(r"Target $x_1$ colored by $\exp((a+b)/\epsilon)$")
        axes[1].set_xlabel("Dim 1")
        axes[1].set_ylabel("Dim 2")
        # cbar = fig.colorbar(sc1, ax=axes.ravel().tolist())
        # Set the colorbar to the right of the second plot
        cbar = fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label(r'$\exp(\frac{a+b}{iter})$ magnitude')

        plt.tight_layout()
        plt.show()

    def train_cfm_step(self, time_step: int):
        batch_size = self.config.batch_size

        if self.config.correction_each > 0 and (time_step + 1) % self.config.correction_each == 0 and self.last_correction_time > 0:
            print("Increasing dual training epochs for correction fitting.")
            num_epochs = 5 * self.config.cfm_epochs
        else:
            num_epochs = self.config.cfm_epochs


        model, _ = self.strategy.get_flow(time_step, self.config.device)
        previous_model = self.strategy.get_previous_flow(time_step, self.config.device)
        previous_model.load_state_dict(model.state_dict())
        previous_model.eval()
        model.train()

        x0_test, x1_test = self.current_coupling.coupling_sample(batch_size)
        x0_test, x1_test = x0_test.to(self.config.device), x1_test.to(self.config.device)
        with torch.no_grad():
            u = model(0.5 * torch.ones((batch_size, 1), device=self.config.device), (x0_test + x1_test) / 2, x0_test)
            print(f"   Flow model output mean: {u.mean().item():.4f}, std: {u.std().item():.4f}")

        pbar = trange(num_epochs, desc="CFM Training")
        for epoch in pbar:
            x0, x1 = self.current_coupling.coupling_sample(batch_size)
            x0, x1 = x0.clone().to(self.config.device), x1.clone().to(self.config.device)

            with self._autocast():
                loss, model, optimizer = self.loss_cfm.execute(
                    x0, x1, self.strategy, self.config, self.flow_matching, time_step, backward=isinstance(self.strategy, BidirectionalStrategy)
                )

            optimizer.zero_grad()
            if self._amp_enabled:
                self._scaler.scale(loss).backward()
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                loss.backward()
                optimizer.step()

            pbar.set_postfix({"CFM Loss": loss.item()})

        with torch.no_grad():
            u = model(0.5 * torch.ones((batch_size, 1), device=self.config.device), (x0_test + x1_test) / 2, x0_test)
            print(f"   Flow model output mean: {u.mean().item():.4f}, std: {u.std().item():.4f}")


        self.visualizer.display(time_step, self)

        x0, x0_back = self.current_coupling.coupling_sample(10000)
        x0 = x0.clone()
        x0_back = x0_back.clone()
        # If bidirectional and step is odd, swap x0 and x0_back to get correct source samples for transport cost calculation
        if isinstance(self.strategy, BidirectionalStrategy) and time_step % 2 == 1:
            x0, x0_back = x0_back, x0
        model.eval()
        x1 = self.flow_matching.evolve(x0.to(self.config.device), model, num_steps=100, sigma=self.config.sigma, extend=True)[:, :self.strategy.config.dimension]
        transport_cost = self.visualizer.transport_cost(x0.cpu(), x1.cpu())
        print(torch.norm(transport_cost, dim=1).mean().item())

    def advance_step(self, time_step: int, backward: bool = False):
        # Use coupling's chunk size for disk-based coupling, otherwise use defaults
        if isinstance(self.current_coupling, DiskCouplingDistribution):
            chunk_size = self.current_coupling.chunk_size
        else:
            chunk_size = 100_000 if not self.config.image_shape else 256

        (mu_model, _), (nu_model, _) = self.strategy.get_bridges(self.device)
        mu_model.eval()
        nu_model.eval()

        n_chunks = (self.current_coupling.size + chunk_size - 1) // chunk_size
        device = self.config.device

        with torch.no_grad():
            for chunk_idx in trange(n_chunks, desc="Advancing coupling"):
                i = chunk_idx * chunk_size
                # Handle last chunk which may be smaller
                actual_chunk_size = min(chunk_size, self.current_coupling.size - i)

                if self.config.advance_with_flow:
                    x_mu = self.mu_distribution.dataset[i : i + actual_chunk_size].to(device)
                    y_nu = self.nu_distribution.dataset[i : i + actual_chunk_size].to(device)

                    t_tensor = torch.full((x_mu.shape[0], 1), time_step / self.config.iterations, device=device)

                    x_mu_next = x_mu + mu_model(t_tensor, x_mu) * (1 / self.config.iterations)
                    y_nu_next = y_nu + nu_model(t_tensor, y_nu) * (1 / self.config.iterations)
                else:
                    # Compute interpolation from source and new_source, target and new_target for current chunk
                    t_tensor = torch.full((actual_chunk_size, 1), (time_step + 1) / self.config.iterations, device=device)
                    x_mu_next = self.flow_matching.interpolate(
                        self.initial_coupling.sample_init(actual_chunk_size).to(device),
                        self.new_source.sample(actual_chunk_size).to(device),
                        t_tensor,
                        # 0.0,
                        sigma=self.config.sigma
                    )
                    y_nu_next = self.flow_matching.interpolate(
                        self.initial_coupling.sample_target(actual_chunk_size).to(device),
                        self.new_target.sample(actual_chunk_size).to(device),
                        t_tensor,
                        # 0.0,
                        sigma=self.config.sigma
                    )

                num_steps = self.config.advance_steps
                flow_model_mu, _ = self.strategy.get_flow(0, device=self.device)
                flow_model_nu, _ = self.strategy.get_flow(1, device=self.device)
                dt = 1.0 / num_steps

                if not backward or time_step % 2 == 0:
                    x_next = x_mu_next.clone()
                    xt = x_next.clone()

                    # for j in range(num_steps):
                    #     t = torch.full((x_next.shape[0], 1), j * dt, device=device)
                    #     if is_image_tensor(xt):
                    #         x_concat = torch.cat([xt, x_next], dim=1)
                    #         vt = flow_model_mu(t, x_concat)
                    #         vt = vt[:, :xt.shape[1], :, :]
                    #     else:
                    #         vt = flow_model_mu(t, xt, x_next)[:, : self.strategy.config.dimension]
                    #     xt = xt + vt * dt
                    # y_next = xt
                    y_next = self.flow_matching.evolve(x_next, flow_model_mu, num_steps=num_steps, extend=True)[:, :self.strategy.config.dimension]
                else:
                    y_next = y_nu_next.clone()
                    yt = y_next.clone()

                    for j in range(num_steps):
                        t = torch.full((y_next.shape[0], 1), j * dt, device=device)
                        if is_image_tensor(yt):
                            y_concat = torch.cat([yt, y_next], dim=1)
                            vt = flow_model_nu(t, y_concat)
                            vt = vt[:, :yt.shape[1], :, :]
                        else:
                            # vt = flow_model_nu(torch.cat([yt, y_next, t], dim=1))[:, : self.strategy.config.dimension]
                            vt = flow_model_nu(t, yt, y_next)[:, : self.strategy.config.dimension]
                        yt = yt + vt * dt
                    x_next = yt

                # Update coupling dataset
                self.current_coupling.update_chunk(i, x_next.detach().cpu(), y_next.detach().cpu())

                if self.config.advance_with_flow:
                    self.mu_distribution.dataset[i : i + actual_chunk_size] = x_mu_next.cpu()
                    self.nu_distribution.dataset[i : i + actual_chunk_size] = y_nu_next.cpu()

        # Plot every N steps to avoid excessive plotting overhead
        if (time_step + 1) % self.visualizer.config.show_each == 0 or time_step == self.config.iterations - 1:
            self.plot_distributions_after_advance(time_step)

    def plot_distributions_after_advance(self, time_step: int):
        """Plot distributions after advancing step. Handles both vectors and images."""

        if is_image_tensor(self.mu_distribution.sample(1)):
            self._plot_image_distributions_after_advance(time_step)
            return

        # Vector distributions visualization
        n_samples = 2500
        t_tensor = torch.full((n_samples // 10, 1), (time_step + 1) / self.config.iterations, device=self.device)
        mu_samples = self.flow_matching.interpolate(
            self.initial_coupling.sample_init(n_samples // 10).to(self.device),
            self.new_source.sample(n_samples // 10).to(self.device),
            t_tensor,
            0.0,
        )
        nu_samples = self.flow_matching.interpolate(
            self.initial_coupling.sample_target(n_samples // 10).to(self.device),
            self.new_target.sample(n_samples // 10).to(self.device),
            t_tensor,
            0.0,
        )

        # Get coupling samples and move to CPU once
        (mu_coupling_samples, nu_coupling_samples) = self.current_coupling.coupling_sample(250)
        mu_cpu = mu_coupling_samples.clone().cpu()
        nu_cpu = nu_coupling_samples.clone().cpu()

        nu_flow = self.flow_matching.evolve(mu_cpu.to(self.device), self.strategy.get_flow(0, self.device)[0], num_steps=100, sigma=self.config.sigma, extend=True)[:, :self.strategy.config.dimension]
        nu_flow_cpu = nu_flow.cpu()

        # 3D plot
        if self.strategy.config.dimension == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")

            mu_s_cpu = mu_samples.cpu()
            nu_s_cpu = nu_samples.cpu()

            ax.scatter(mu_s_cpu[:, 0], mu_s_cpu[:, 1], mu_s_cpu[:, 2], c="blue", alpha=0.1, depthshade=True, s=3)
            ax.scatter(nu_s_cpu[:, 0], nu_s_cpu[:, 1], nu_s_cpu[:, 2], c="red", alpha=0.1, depthshade=True, s=3)
            ax.scatter(mu_cpu[:, 0], mu_cpu[:, 1], mu_cpu[:, 2], c="purple", marker="x", alpha=0.75, depthshade=True)
            ax.scatter(nu_cpu[:, 0], nu_cpu[:, 1], nu_cpu[:, 2], c="orange", marker="x", alpha=0.75, depthshade=True)
            ax.scatter(nu_flow_cpu[:, 0], nu_flow_cpu[:, 1], nu_flow_cpu[:, 2], c="green", marker="o", alpha=0.75, depthshade=True, s=50, label="Flow Evolved")

            # Line between coupling samples
            dist = torch.norm(nu_cpu - mu_cpu, dim=1)

            for j in range(mu_cpu.shape[0]):
                ax.plot(
                    [mu_cpu[j, 0], nu_cpu[j, 0]],
                    [mu_cpu[j, 1], nu_cpu[j, 1]],
                    [mu_cpu[j, 2], nu_cpu[j, 2]],
                    color="gray",
                    alpha=0.8,
                )
            L = 1.2  # length of the axis lines

            ax.plot([-L, L], [0, 0], [0, 0], color="k", linewidth=1)  # X-axis
            ax.plot([0, 0], [-L, L], [0, 0], color="k", linewidth=1)  # Y-axis
            ax.plot([0, 0], [0, 0], [-L, L], color="k", linewidth=1)  # Z-axis

            # Set limits
            ax.set_xlim([-L, L])
            ax.set_ylim([-L, L])
            ax.set_zlim([-L, L])

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            sm = plt.cm.ScalarMappable(cmap="viridis")
            sm.set_array(dist.numpy())
            plt.colorbar(sm, ax=ax, shrink=0.6, label="Transport length")

            ax.set_title(f"Distributions after Advancing Step {time_step+1}")
            plt.show()
        else:
            mu_s_cpu = mu_samples.cpu()
            nu_s_cpu = nu_samples.cpu()

            plt.figure(figsize=(6, 6))
            plt.plot(mu_s_cpu[:, 0], mu_s_cpu[:, 1], ".", label="Mu Distribution", alpha=0.5)
            plt.plot(nu_s_cpu[:, 0], nu_s_cpu[:, 1], ".", label="Nu Distribution", alpha=0.5)
            plt.plot(mu_cpu[:, 0], mu_cpu[:, 1], "x", label="Mu Coupling Samples", alpha=0.5)
            plt.plot(nu_cpu[:, 0], nu_cpu[:, 1], "x", label="Nu Coupling Samples", alpha=0.5)
            # Line between coupling samples
            for j in range(mu_cpu.shape[0]):
                plt.plot(
                    [mu_cpu[j, 0], nu_cpu[j, 0]],
                    [mu_cpu[j, 1], nu_cpu[j, 1]],
                    "k-",
                    alpha=0.1,
                )
            plt.legend()
            plt.title(f"Distributions after Advancing Step {time_step+1}")
            plt.show()

    def _plot_image_distributions_after_advance(self, time_step: int):
        """Plot image distributions after advancing step."""
        n_samples = 4
        (mu_coupling, nu_coupling) = self.current_coupling.coupling_sample(n_samples)
        mu_coupling = mu_coupling.cpu()
        nu_coupling = nu_coupling.cpu()

        fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

        for i in range(n_samples):
            # Source (mu) images
            if mu_coupling.dim() == 4:
                mu_img = mu_coupling[i].squeeze(0).numpy()
            else:
                mu_img = mu_coupling[i].reshape(self.config.image_shape[1], self.config.image_shape[2]).numpy()
            axes[0, i].imshow(mu_img, cmap="gray", vmin=-1, vmax=1)
            axes[0, i].set_title(f"Source {i+1}")
            axes[0, i].axis("off")

            # Target (nu) images
            if nu_coupling.dim() == 4:
                nu_img = nu_coupling[i].squeeze(0).numpy()
            else:
                nu_img = nu_coupling[i].reshape(self.config.image_shape[1], self.config.image_shape[2]).numpy()
            axes[1, i].imshow(nu_img, cmap="gray", vmin=-1, vmax=1)
            axes[1, i].set_title(f"Target {i+1}")
            axes[1, i].axis("off")

        plt.suptitle(f"Coupling after Step {time_step + 1}")
        plt.tight_layout()
        plt.show()
