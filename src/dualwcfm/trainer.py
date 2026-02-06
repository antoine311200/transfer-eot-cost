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
from dualwcfm.distributions.distribution import Distribution


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
        self.initial_coupling = initial_coupling
        self.new_source = new_source
        self.new_target = new_target

        self.visualizer = visualizer
        self.strategy = strategy
        self.flow_matching = flow_matching
        self.loss_cfm = CFMLoss.get_loss(config.loss)

        self.config = config
        self.device = device

        # Fixed sized mu_t, nu_t and coupling distributions
        self._init_distributions(initial_coupling)

    def _init_distributions(self, initial_coupling: CouplingDistribution):
        mu_distribution, nu_distribution = initial_coupling.get_marginals()
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

        # Copy to Initial distribution for later use
        self.initial_mu_distribution = mu_distribution
        self.initial_nu_distribution = nu_distribution
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
            else:
                print("Skipping pretraining.")
                print("Loading initial flow model.")
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
        self.visualizer._display_evolution(start_points.cpu(), end_points.cpu(), target_points.cpu())
        plt.show()

        # Divide by 10 learning rates for cfm
        for param_group in self.strategy.get_flow(0, self.device)[1].param_groups:
            param_group["lr"] = param_group["lr"] / 10.0
        if isinstance(self.strategy, BidirectionalStrategy):
            for param_group in self.strategy.get_flow(1, self.device)[1].param_groups:
                param_group["lr"] = param_group["lr"] / 10.0

        # General training loop
        for iteration in range(self.config.iterations):
            print(f"Iteration {iteration+1}/{self.config.iterations}")
            self.train_duals_step(iteration)
            self.train_cfm_step(iteration)
            self.advance_step(iteration)

    def train_marginal_flows(self, retrain: bool = False, save: str = None):
        sigma = self.config.sigma
        batch_size = self.config.batch_size
        device = self.config.device
        loss_fn = nn.MSELoss()

        (mu_model, mu_optimizer), (nu_model, nu_optimizer) = self.strategy.get_bridges(device)
        initial_mu_model, initial_nu_model = self.strategy.get_initial_bridges(device)

        if retrain:

            pbar = trange(self.config.marginal_epochs, desc="Marginal Flow Matching")
            for epoch in pbar:
                x0, y0 = self.initial_coupling.sample_init(batch_size), self.initial_coupling.sample_target(batch_size)
                x1, y1 = self.new_source.sample(batch_size), self.new_target.sample(batch_size)
                x0, x1, y0, y1 = x0.to(device), x1.to(device), y0.to(device), y1.to(device)

                t = torch.rand(batch_size, 1).to(device)
                xt, yt = self.flow_matching.interpolate(x0, x1, t, sigma), self.flow_matching.interpolate(
                    y0, y1, t, sigma
                )
                ut, vt = mu_model(t, xt), nu_model(t, yt)
                ut_true, vt_true = self.flow_matching.vector_field(
                    xt, t, x0, x1, extend=False
                ), self.flow_matching.vector_field(yt, t, y0, y1, extend=False)

                mu_optimizer.zero_grad()
                nu_optimizer.zero_grad()

                loss_mu = loss_fn(ut, ut_true)
                loss_mu.backward()
                mu_optimizer.step()

                loss_nu = loss_fn(vt, vt_true)
                loss_nu.backward()
                nu_optimizer.step()

                pbar.set_postfix({"Loss mu": loss_mu.item(), "Loss nu": loss_nu.item()})

                # Visualize marginal flows during training
                self.visualizer.display_marginal(
                    epoch, self, mu_model, self.mu_distribution, self.new_source, title="Mu Marginal Flow"
                )
                self.visualizer.display_marginal(
                    epoch, self, nu_model, self.nu_distribution, self.new_target, title="Nu Marginal Flow"
                )

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

        end_points = self.flow_matching.evolve(
            start_points, self.strategy.mu_bridge, num_steps=50, sigma=self.config.sigma, extend=False
        )
        target_points = self.new_source.sample(samples)
        target_points = target_points.to(self.device)
        self.visualizer._display_evolution(start_points.cpu(), end_points.cpu(), target_points.cpu())
        plt.show()

        start_points = self.nu_distribution.sample(samples)
        start_points = start_points.to(self.device)

        end_points = self.flow_matching.evolve(
            start_points, self.strategy.nu_bridge, num_steps=50, sigma=self.config.sigma, extend=False
        )
        target_points = self.new_target.sample(samples)
        target_points = target_points.to(self.device)
        self.visualizer._display_evolution(start_points.cpu(), end_points.cpu(), target_points.cpu())
        plt.show()

    def pretrain(self, model, optimizer, backward: bool = False):
        sigma = self.config.sigma
        batch_size = self.config.batch_size
        loss_fn = nn.MSELoss()

        pbar = trange(self.config.pretrain_epochs, desc="Pretraining")

        for epoch in pbar:
            x0, x1 = self.initial_coupling.coupling_sample(batch_size)
            x0, x1 = x0.to(self.config.device), x1.to(self.config.device)
            if backward:
                x0, x1 = x1, x0

            s = torch.rand(batch_size, 1).to(self.config.device)
            x_s = self.flow_matching.interpolate(x0, x1, s, sigma)
            u_s = model(s, x_s, x0)
            u_s_true = self.flow_matching.vector_field(x_s, s, x0, x1)

            loss = loss_fn(u_s, u_s_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Pretrain Loss": loss.item()})

            # Visualize pretraining flow
            self.visualizer.display_pretrain(epoch, self, model, backward=backward)

    def train_duals_step(self, time_step: int):
        batch_size = self.config.batch_size
        t_tensor = torch.full((batch_size, 1), time_step / self.config.iterations, device=self.config.device)
        is_image = self.config.image_shape is not None

        (mu_model, _), (nu_model, _) = self.strategy.get_bridges(self.device)
        f_dual, g_dual, dual_optimizer = self.strategy.get_duals(self.device)
        f_dual.train()
        g_dual.train()
        mu_model.eval()
        nu_model.eval()

        pbar = trange(self.config.dual_epochs, desc="Dual Training")

        for epoch in pbar:
            x, y = self.current_coupling.coupling_sample(self.config.batch_size)

            # Create mu_t, nu_t samples from interpolating between source and new_source, target and new_target for current batch
            x_mu = self.flow_matching.interpolate(
                self.initial_coupling.sample_init(batch_size).to(self.config.device),
                self.new_source.sample(batch_size).to(self.config.device),
                t_tensor,
                0.0,
            )
            y_nu = self.flow_matching.interpolate(
                self.initial_coupling.sample_target(batch_size).to(self.config.device),
                self.new_target.sample(batch_size).to(self.config.device),
                t_tensor,
                0.0,
            )

            x, y = x.to(self.config.device), y.to(self.config.device)
            x_mu, y_nu = x_mu.to(self.config.device), y_nu.to(self.config.device)

            with torch.no_grad():
                u_t = mu_model(t_tensor, x)
                v_t = nu_model(t_tensor, y)

            x, y = x.requires_grad_(True), y.requires_grad_(True)
            x_mu, y_nu = x_mu.requires_grad_(True), y_nu.requires_grad_(True)

            a, a_mu = f_dual(x), f_dual(x_mu)
            b, b_nu = g_dual(y), g_dual(y_nu)

            # Coupling term
            coupling_term = torch.mean((a - a.mean() + b - b.mean()) ** 2)

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

            loss = 0.5 * coupling_term - a_term - b_term + l2_reg + lip_reg

            dual_optimizer.zero_grad()
            loss.backward()
            dual_optimizer.step()

            pbar.set_postfix({"Dual Loss": loss.item()})

    def train_cfm_step(self, time_step: int):
        batch_size = self.config.batch_size

        pbar = trange(self.config.cfm_epochs, desc="CFM Training")

        model, _ = self.strategy.get_flow(time_step, self.config.device)
        previous_model = self.strategy.get_previous_flow(time_step, self.config.device)
        previous_model.load_state_dict(model.state_dict())
        previous_model.eval()

        for epoch in pbar:
            x0, x1 = self.current_coupling.coupling_sample(batch_size)
            x0, x1 = x0.to(self.config.device), x1.to(self.config.device)

            loss, model, optimizer = self.loss_cfm.execute(
                x0, x1, self.strategy, self.config, self.flow_matching, time_step
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"CFM Loss": loss.item()})

        self.visualizer.display(time_step, self)

    def advance_step(self, time_step: int):
        # Use coupling's chunk size for disk-based coupling, otherwise use defaults
        if isinstance(self.current_coupling, DiskCouplingDistribution):
            chunk_size = self.current_coupling.chunk_size
        else:
            chunk_size = 100_000 if not self.config.image_shape else 256

        (mu_model, _), (nu_model, _) = self.strategy.get_bridges(self.device)
        mu_model.eval()
        nu_model.eval()

        use_disk = isinstance(self.current_coupling, DiskCouplingDistribution)
        n_chunks = (self.current_coupling.size + chunk_size - 1) // chunk_size

        with torch.no_grad():
            for chunk_idx in trange(n_chunks, desc="Advancing coupling"):
                i = chunk_idx * chunk_size
                # Handle last chunk which may be smaller
                actual_chunk_size = min(chunk_size, self.current_coupling.size - i)

                if self.config.advance_with_flow:
                    x_mu, y_nu = self.mu_distribution.dataset[i : i + actual_chunk_size].to(
                        self.config.device
                    ), self.nu_distribution.dataset[i : i + actual_chunk_size].to(self.config.device)

                    t_tensor = torch.full((x_mu.shape[0], 1), time_step / self.config.iterations).to(
                        self.config.device
                    )

                    x_mu_next = x_mu.detach() + mu_model(t_tensor, x_mu) * (1 / self.config.iterations)
                    y_nu_next = y_nu.detach() + nu_model(t_tensor, y_nu) * (1 / self.config.iterations)
                else:
                    # Compute interpolation from source and new_source, target and new_target for current chunk
                    t_tensor = torch.full((actual_chunk_size, 1), time_step / self.config.iterations).to(
                        self.config.device
                    )
                    x_mu_next = self.flow_matching.interpolate(
                        self.initial_coupling.sample_init(actual_chunk_size).to(self.config.device),
                        self.new_source.sample(actual_chunk_size).to(self.config.device),
                        t_tensor,
                        0.0,
                    )
                    y_nu_next = self.flow_matching.interpolate(
                        self.initial_coupling.sample_target(actual_chunk_size).to(self.config.device),
                        self.new_target.sample(actual_chunk_size).to(self.config.device),
                        t_tensor,
                        0.0,
                    )

                num_steps = 10
                flow_model_mu = self.strategy.forward_model
                flow_model_nu = (
                    self.strategy.backward_model
                    if isinstance(self.strategy, BidirectionalStrategy)
                    else self.strategy.forward_model
                )
                device = self.config.device
                dt = 1.0 / num_steps

                if time_step % 2 == 0:
                    x_next = x_mu_next.detach().clone()
                    xt = x_next.clone()

                    for j in range(num_steps):
                        t = torch.ones(x_next.shape[0], 1).to(device) * (j * dt)
                        if is_image_tensor(xt):
                            x_concat = torch.cat([xt, x_next], dim=1)
                            vt = flow_model_mu(t, x_concat)
                            n_channels = xt.shape[1]
                            vt = vt[:, :n_channels, :, :]
                        else:
                            vt = flow_model_mu(torch.cat([xt, x_next, t], dim=1))[:, : self.strategy.config.dimension]
                        xt = xt + vt * dt
                    y_next = xt.clone()
                else:
                    y_next = y_nu_next.detach().clone()
                    yt = y_next.clone()

                    for j in range(num_steps):
                        t = torch.ones(y_next.shape[0], 1).to(device) * (j * dt)
                        if is_image_tensor(yt):
                            y_concat = torch.cat([yt, y_next], dim=1)
                            vt = flow_model_nu(t, y_concat)
                            n_channels = yt.shape[1]
                            vt = vt[:, :n_channels, :, :]
                        else:
                            vt = flow_model_nu(torch.cat([yt, y_next, t], dim=1))[:, : self.strategy.config.dimension]
                        yt = yt + vt * dt
                    x_next = yt.clone()

                # Update coupling dataset
                self.current_coupling.update_chunk(i, x_next.cpu(), y_next.cpu())

                if self.config.advance_with_flow:
                    self.mu_distribution.dataset[i : i + actual_chunk_size] = x_mu_next.cpu()
                    self.nu_distribution.dataset[i : i + actual_chunk_size] = y_nu_next.cpu()

        self.plot_distributions_after_advance(time_step)

    def plot_distributions_after_advance(self, time_step: int):
        """Plot distributions after advancing step. Handles both vectors and images."""

        if is_image_tensor(self.mu_distribution.sample(1)):
            self._plot_image_distributions_after_advance(time_step)
            return

        # Vector distributions visualization
        n_samples = 2500
        t_tensor = torch.full((n_samples // 10, 1), time_step / self.config.iterations, device=self.device)
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

        # Get coupling samples
        (mu_coupling_samples, nu_coupling_samples) = self.current_coupling.coupling_sample(250)
        mu_coupling_samples = mu_coupling_samples.to(self.device)
        nu_coupling_samples = nu_coupling_samples.to(self.device)

        # 3D plot
        if self.strategy.config.dimension == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(
                mu_samples[:, 0].cpu(),
                mu_samples[:, 1].cpu(),
                mu_samples[:, 2].cpu(),
                c="blue",
                alpha=0.1,
                depthshade=True,
                s=3,
            )
            ax.scatter(
                nu_samples[:, 0].cpu(),
                nu_samples[:, 1].cpu(),
                nu_samples[:, 2].cpu(),
                c="red",
                alpha=0.1,
                depthshade=True,
                s=3,
            )
            ax.scatter(
                mu_coupling_samples[:, 0].cpu(),
                mu_coupling_samples[:, 1].cpu(),
                mu_coupling_samples[:, 2].cpu(),
                c="purple",
                marker="x",
                alpha=0.75,
                depthshade=True,
            )
            ax.scatter(
                nu_coupling_samples[:, 0].cpu(),
                nu_coupling_samples[:, 1].cpu(),
                nu_coupling_samples[:, 2].cpu(),
                c="orange",
                marker="x",
                alpha=0.75,
                depthshade=True,
            )

            # Line between coupling samples
            dist = torch.norm(nu_coupling_samples - mu_coupling_samples, dim=1).cpu()

            for j in range(mu_coupling_samples.shape[0]):
                ax.plot(
                    [mu_coupling_samples[j, 0].cpu(), nu_coupling_samples[j, 0].cpu()],
                    [mu_coupling_samples[j, 1].cpu(), nu_coupling_samples[j, 1].cpu()],
                    [mu_coupling_samples[j, 2].cpu(), nu_coupling_samples[j, 2].cpu()],
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
            plt.figure(figsize=(6, 6))
            plt.plot(mu_samples[:, 0].cpu(), mu_samples[:, 1].cpu(), ".", label="Mu Distribution", alpha=0.5)
            plt.plot(nu_samples[:, 0].cpu(), nu_samples[:, 1].cpu(), ".", label="Nu Distribution", alpha=0.5)
            plt.plot(
                mu_coupling_samples[:, 0].cpu(),
                mu_coupling_samples[:, 1].cpu(),
                "x",
                label="Mu Coupling Samples",
                alpha=0.5,
            )
            plt.plot(
                nu_coupling_samples[:, 0].cpu(),
                nu_coupling_samples[:, 1].cpu(),
                "x",
                label="Nu Coupling Samples",
                alpha=0.5,
            )
            # Line between coupling samples
            for j in range(mu_coupling_samples.shape[0]):
                plt.plot(
                    [mu_coupling_samples[j, 0].cpu(), nu_coupling_samples[j, 0].cpu()],
                    [mu_coupling_samples[j, 1].cpu(), nu_coupling_samples[j, 1].cpu()],
                    "k-",
                    alpha=0.1,
                )
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
