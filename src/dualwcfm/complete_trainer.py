from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dualwcfm.distributions.coupling import CouplingDistribution, FixedSizeCouplingDistribution
from dualwcfm.distributions.distribution import Distribution, FixedSizeDistribution, FixedSizeDistribution

from torchdyn.core import NeuralODE


@dataclass
class TrainingConfig:
    device: torch.device
    dataset_size: int = 100_000

    marginal_epochs: int = 2500
    pretrain_epochs: int = 10000
    dual_epochs: int = 100
    cfm_epochs: int = 250

    iterations: int = 100
    loss: str = "implicit"

    sigma: float = 0.1
    l2_reg: float = 1e-3
    lip_reg: float = 1e-3

    batch_size: int = 4096


@dataclass
class ModelConfig:
    dual_learning_rate = 1e-4
    mu_learning_rate = 1e-3
    nu_learning_rate = 1e-3
    flow_learning_rate = 5e-3


@dataclass
class FeedForwardConfig(ModelConfig):
    dimension: int
    hidden_size: int


@dataclass
class VisualizeConfig:
    show_each: int = 10
    n_trajectories: int = 500


class Visualizer:
    def __init__(self, config: VisualizeConfig):
        self.config = config
        self.dim = 2

    def display(self, epoch: int, trainer: "Trainer"):
        if epoch % self.config.show_each == 0:
            x, _ = trainer.current_coupling.coupling_sample(self.config.n_trajectories)
            y_target = trainer.new_target.sample(self.config.n_trajectories)
            x, y_target = x.to(trainer.device), y_target.to(trainer.device)
            y = trainer.flow_matching.evolve(
                x,
                trainer.strategy.get_flow(epoch, trainer.device)[0],
                num_steps=25,
                sigma=trainer.config.sigma,
                extend=True,
            )
            y = y[:, : self.dim]
            self._display_evolution(x.cpu(), y.cpu(), y_target.cpu())
            plt.show()

    def _display_evolution(
        self, start_points: torch.Tensor, end_points: torch.Tensor, target_points: Optional[torch.Tensor] = None
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        segments = []
        for j in range(self.config.n_trajectories):
            points = torch.stack([start_points[j], end_points[j]], dim=0).numpy()
            segs = np.stack([points[:-1], points[1:]], axis=1)
            segments.extend(segs)

        if target_points is not None:
            ax.scatter(target_points[:, 0], target_points[:, 1], c="orange", s=5, alpha=0.5, zorder=4, label="Target")
        ax.scatter(start_points[:, 0], start_points[:, 1], c="red", s=5, alpha=0.5, zorder=4, label="Start")
        ax.scatter(end_points[:, 0], end_points[:, 1], c="green", s=5, alpha=0.5, zorder=4, label="End")
        mid_points = (start_points + end_points) / 2
        ax.scatter(mid_points[:, 0], mid_points[:, 1], c="blue", s=5, alpha=0.5, zorder=4, label="Mid")
        ax.grid(True, alpha=0.2)

        lc = LineCollection(segments, colors="gray", alpha=0.5, linewidths=0.8, zorder=1)
        ax.add_collection(lc)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.set_aspect("equal")


class Visualizer3D(Visualizer):
    def __init__(self, config: VisualizeConfig):
        super().__init__(config)
        self.dim = 3

    def _display_evolution(
        self, start_points: torch.Tensor, end_points: torch.Tensor, target_points: Optional[torch.Tensor] = None
    ):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        for j in range(self.config.n_trajectories):
            points = torch.stack([start_points[j], end_points[j]], dim=0).numpy()
            ax.plot(points[:, 0], points[:, 1], points[:, 2], c="gray", alpha=0.5, linewidth=0.8)

        if target_points is not None:
            ax.scatter(
                target_points[:, 0],
                target_points[:, 1],
                target_points[:, 2],
                c="orange",
                s=5,
                alpha=0.5,
                zorder=4,
                label="Target",
            )
        ax.scatter(
            start_points[:, 0],
            start_points[:, 1],
            start_points[:, 2],
            c="red",
            s=5,
            alpha=0.5,
            zorder=4,
            label="Start",
        )
        ax.scatter(
            end_points[:, 0], end_points[:, 1], end_points[:, 2], c="green", s=5, alpha=0.5, zorder=4, label="End"
        )
        mid_points = (start_points + end_points) / 2
        ax.scatter(
            mid_points[:, 0], mid_points[:, 1], mid_points[:, 2], c="blue", s=5, alpha=0.5, zorder=4, label="Mid"
        )
        ax.grid(True, alpha=0.2)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore
        ax.legend()
        ax.set_box_aspect([1, 1, 1])  # type: ignore


class node_wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        t = t.to(x.device)
        if t.dim() < 2:
            t = t.repeat(x.shape[0])[:, None]
        return self.model(torch.cat([x, t], dim=1))


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, use_norm=None):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            (
                torch.nn.BatchNorm1d(w)
                if use_norm == "batch"
                else torch.nn.LayerNorm(w) if use_norm == "layer" else torch.nn.Identity()
            ),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            (
                torch.nn.BatchNorm1d(w)
                if use_norm == "batch"
                else torch.nn.LayerNorm(w) if use_norm == "layer" else torch.nn.Identity()
            ),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalFlowMatching:
    def vector_field(
        self, x: torch.Tensor, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, extend: bool = True
    ) -> torch.Tensor:
        """Compute the vector field at position x and time t."""
        if extend:
            return torch.cat([x1 - x0, torch.zeros_like(x0)], dim=1)
        else:
            return x1 - x0

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma: float) -> torch.Tensor:
        """Interpolate between x0 and x1 at time t with noise sigma."""
        return (1 - t) * x0 + t * x1 + sigma * torch.randn_like(x0) * torch.sqrt(t * (1 - t))

    def evolve(
        self, x_init: torch.Tensor, model: nn.Module, num_steps: int = 100, sigma: float = 0.0, extend: bool = True
    ) -> torch.Tensor:
        """Evolve the points x_init over time t."""
        node = NeuralODE(node_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        t_span = torch.linspace(0, 1, num_steps).to(x_init.device)
        with torch.no_grad():
            if extend:
                x_init = torch.cat([x_init, x_init], dim=1)
            trajectory = node.trajectory(x_init, t_span)
        return trajectory[-1]


class Strategy:
    def __init__(self, wrapper, config: FeedForwardConfig):
        self.wrapper = wrapper

        self.mu_bridge = wrapper(
            MLP(dim=config.dimension, out_dim=config.dimension, w=config.hidden_size, time_varying=True)
        )
        self.nu_bridge = wrapper(
            MLP(dim=config.dimension, out_dim=config.dimension, w=config.hidden_size, time_varying=True)
        )
        self.f_dual = wrapper(MLP(dim=config.dimension, out_dim=1, w=config.hidden_size, time_varying=False))
        self.g_dual = wrapper(MLP(dim=config.dimension, out_dim=1, w=config.hidden_size, time_varying=False))

        self.mu_optimizer = optim.Adam(self.mu_bridge.parameters(), lr=config.mu_learning_rate)
        self.nu_optimizer = optim.Adam(self.nu_bridge.parameters(), lr=config.nu_learning_rate)
        self.dual_optimizer = optim.Adam(
            list(self.f_dual.parameters()) + list(self.g_dual.parameters()), lr=config.dual_learning_rate
        )

        self.forward_model = wrapper(
            MLP(dim=2 * config.dimension, out_dim=2 * config.dimension, w=config.hidden_size, time_varying=True)
        )
        self.previous_forward_model = wrapper(
            MLP(dim=2 * config.dimension, out_dim=2 * config.dimension, w=config.hidden_size, time_varying=True)
        )

        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=config.flow_learning_rate)

        self.config = config

        # Save initial state
        self.initial_mu_bridge = wrapper(
            MLP(dim=config.dimension, out_dim=config.dimension, w=config.hidden_size, time_varying=True)
        )
        self.initial_nu_bridge = wrapper(
            MLP(dim=config.dimension, out_dim=config.dimension, w=config.hidden_size, time_varying=True)
        )
        self.initial_forward_model = wrapper(
            MLP(dim=2 * config.dimension, out_dim=2 * config.dimension, w=config.hidden_size, time_varying=True)
        )

    def get_duals(self, device):
        return self.f_dual.to(device), self.g_dual.to(device), self.dual_optimizer

    def get_bridges(self, device):
        return (self.mu_bridge.to(device), self.mu_optimizer), (self.nu_bridge.to(device), self.nu_optimizer)

    def get_flow(self, step: int, device):
        return self.forward_model.to(device), self.forward_optimizer

    def get_previous_flow(self, step: int, device):
        return self.previous_forward_model.to(device)

    def get_initial_bridges(self, device):
        return self.initial_mu_bridge.to(device), self.initial_nu_bridge.to(device)

    def get_initial_flow(self, step: int, device):
        return self.initial_forward_model.to(device)

    def advance(
        self, step: int, x_mu_next: torch.Tensor, y_nu_next: torch.Tensor, num_steps: int = 100, sigma: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the coupling using the current flow models."""
        x_condition = x_mu_next.clone()
        xt = x_mu_next.clone()
        dt = 1.0 / num_steps
        model = self.get_flow(step, x_mu_next.device)[0]
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.ones(xt.shape[0], 1).to(xt.device) * (i * dt)
                vt = model(t, xt, x_condition)[:, : xt.shape[1] // 2]
                xt = xt + vt * dt
        return x_mu_next, xt.clone()


class BidirectionalStrategy(Strategy):
    def __init__(self, wrapper, config: FeedForwardConfig):
        super().__init__(wrapper, config)

        self.backward_model = wrapper(
            MLP(dim=2 * config.dimension, out_dim=2 * config.dimension, w=config.hidden_size, time_varying=True)
        )
        self.previous_backward_model = wrapper(
            MLP(dim=2 * config.dimension, out_dim=2 * config.dimension, w=config.hidden_size, time_varying=True)
        )
        self.backward_optimizer = optim.Adam(self.backward_model.parameters(), lr=config.flow_learning_rate)

        self.initial_backward_model = wrapper(
            MLP(dim=2 * config.dimension, out_dim=2 * config.dimension, w=config.hidden_size, time_varying=True)
        )

    def get_flow(self, step: int, device):
        model = self.forward_model if step % 2 == 0 else self.backward_model
        optimizer = self.forward_optimizer if step % 2 == 0 else self.backward_optimizer
        return model.to(device), optimizer

    def get_previous_flow(self, step: int, device):
        previous_model = self.previous_forward_model if step % 2 == 0 else self.previous_backward_model
        return previous_model.to(device)

    def get_initial_flow(self, step: int, device):
        initial_model = self.initial_forward_model if step % 2 == 0 else self.initial_backward_model
        return initial_model.to(device)

    def advance(
        self, step: int, x_mu_next: torch.Tensor, y_nu_next: torch.Tensor, num_steps: int = 100, sigma: float = 0
    ):
        if step % 2 == 0:
            return super().advance(step, x_mu_next, y_nu_next, num_steps, sigma)
        else:
            y_next, x_next = super().advance(step, y_nu_next, x_mu_next, num_steps, sigma)
            return x_next, y_next


class CFMLoss:
    def execute(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        strategy: Strategy,
        config: TrainingConfig,
        flow_matching: ConditionalFlowMatching,
        step: int,
    ) -> tuple[torch.Tensor, nn.Module, optim.Adam]:
        """Compute the CFM loss."""
        raise NotImplementedError("method not implemented")

    @staticmethod
    def get_loss(loss_name: str):
        if loss_name == "weighted":
            return WeightedCFMLoss()
        elif loss_name == "explicit":
            return ExplicitCFMLoss()
        elif loss_name == "implicit":
            return ImplicitCFMLoss()
        else:
            raise NameError(f"loss name `{loss_name}` does not exist (choose between weighted, explicit, implicit)")


class WeightedCFMLoss(CFMLoss):
    def execute(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        strategy: Strategy,
        config: TrainingConfig,
        flow_matching: ConditionalFlowMatching,
        step: int,
    ) -> tuple[torch.Tensor, nn.Module, optim.Adam]:
        f_dual, g_dual, _ = strategy.get_duals(config.device)
        model, optimizer = strategy.get_flow(0, config.device)

        with torch.no_grad():
            a, b = f_dual(x0), g_dual(x1)
            r = a + b
            log_w = r / config.iterations
            weight = torch.exp(log_w)

            target_u_s = x1 - x0

        s = torch.rand(config.batch_size, 1).to(config.device)
        x_s = flow_matching.interpolate(x0, x1, s, config.sigma)
        u_s = model(s, x_s, x0)

        return (weight * (u_s - target_u_s) ** 2).mean(), model, optimizer


class ExplicitCFMLoss(CFMLoss):
    pass


class ImplicitCFMLoss(CFMLoss):
    def execute(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        strategy: Strategy,
        config: TrainingConfig,
        flow_matching: ConditionalFlowMatching,
        step: int,
    ) -> tuple[torch.Tensor, nn.Module, optim.Adam]:
        f_dual, g_dual, _ = strategy.get_duals(config.device)
        model, optimizer = strategy.get_flow(step, config.device)
        previous_model = strategy.get_previous_flow(step, config.device)

        s = torch.rand(config.batch_size, 1).to(config.device)
        xs = flow_matching.interpolate(x0, x1, s, config.sigma)

        with torch.no_grad():
            a, b = f_dual(x0), g_dual(x1)
            r = a + b
            log_w = r / config.iterations
            weight = torch.exp(log_w)

            if step % 2 != 0:
                x0, x1 = x1, x0

            b_hat_detached = model(s, xs, x0)
            b_ta = previous_model(s, xs, x0)

            u_t = flow_matching.vector_field(x0, torch.zeros_like(x0), x0, x1)
            target = b_ta + (weight - 1) * (u_t - b_hat_detached)
        u_s = model(s, xs, x0)
        return ((u_s - target) ** 2).mean(), model, optimizer


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
        self.mu_distribution = FixedSizeDistribution(mu_distribution, size=self.config.dataset_size)
        self.nu_distribution = FixedSizeDistribution(nu_distribution, size=self.config.dataset_size)
        self.current_coupling = FixedSizeCouplingDistribution(initial_coupling.clone(), size=self.config.dataset_size)

        # Copy to Initial distribution for later use
        self.initial_mu_distribution = self.mu_distribution.clone()
        self.initial_nu_distribution = self.nu_distribution.clone()
        self.initial_coupling_distribution = self.current_coupling.clone()

    def train(self, retrain: bool = False):
        if not retrain:
            self.mu_distribution = self.initial_mu_distribution.clone()
            self.nu_distribution = self.initial_nu_distribution.clone()
            self.current_coupling = self.initial_coupling_distribution.clone()

        # Marginal flow matching phase
        self.train_marginal_flows(retrain)

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
            else:
                print("Skipping pretraining.")
                print("Loading initial flow models.")
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

        # General training loop
        for iteration in range(self.config.iterations):  # , desc="Iteration loop"):
            print(f"Iteration {iteration+1}/{self.config.iterations}")
            self.train_duals_step(iteration)
            self.train_cfm_step(iteration)
            self.advance_step(iteration)

    def train_marginal_flows(self, retrain: bool = False):
        sigma = self.config.sigma
        batch_size = self.config.batch_size
        device = self.config.device
        loss_fn = nn.MSELoss()

        (mu_model, mu_optimizer), (nu_model, nu_optimizer) = self.strategy.get_bridges(device)
        initial_mu_model, initial_nu_model = self.strategy.get_initial_bridges(device)

        if retrain:

            pbar = trange(self.config.marginal_epochs, desc="Marginal Flow Matching")
            for epoch in pbar:
                x0, y0 = self.initial_coupling.coupling_sample(batch_size)
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

                initial_mu_model.load_state_dict(mu_model.state_dict())
                initial_nu_model.load_state_dict(nu_model.state_dict())
        else:
            print("Skipping marginal flow matching retraining.")
            print("Loading initial marginal flow models.")
            mu_model.load_state_dict(initial_mu_model.state_dict())
            nu_model.load_state_dict(initial_nu_model.state_dict())

        start_points = self.mu_distribution.sample(2500)
        start_points = start_points.to(self.device)

        end_points = self.flow_matching.evolve(
            start_points, self.strategy.mu_bridge, num_steps=50, sigma=self.config.sigma, extend=False
        )
        self.visualizer._display_evolution(start_points.cpu(), end_points.cpu())
        plt.show()

        start_points = self.nu_distribution.sample(2500)
        start_points = start_points.to(self.device)

        end_points = self.flow_matching.evolve(
            start_points, self.strategy.nu_bridge, num_steps=50, sigma=self.config.sigma, extend=False
        )
        self.visualizer._display_evolution(start_points.cpu(), end_points.cpu())
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

    def train_duals_step(self, time_step: int):
        batch_size = self.config.batch_size
        t_tensor = torch.full((batch_size, 1), time_step / self.config.iterations, device=self.config.device)

        (mu_model, _), (nu_model, _) = self.strategy.get_bridges(self.device)
        f_dual, g_dual, dual_optimizer = self.strategy.get_duals(self.device)
        f_dual.train()
        g_dual.train()
        mu_model.eval()
        nu_model.eval()

        pbar = trange(self.config.dual_epochs, desc="Dual Training")

        for epoch in pbar:
            x, y = self.current_coupling.coupling_sample(self.config.batch_size)
            x_mu = self.mu_distribution.sample(self.config.batch_size)
            y_nu = self.nu_distribution.sample(self.config.batch_size)

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

            a_term = torch.mean(torch.sum(grad_a * u_t, dim=1))
            b_term = torch.mean(torch.sum(grad_b * v_t, dim=1))

            # Regularization (L2 + Lipshitz)
            l2_reg = self.config.l2_reg * (
                torch.mean(a**2) + torch.mean(b**2) + torch.mean(a_mu**2) + torch.mean(b_nu**2)
            )

            gp_f = ((grad_a.norm(2, dim=1) - 1) ** 2).mean()
            gp_g = ((grad_b.norm(2, dim=1) - 1) ** 2).mean()
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
        chunks = 100_000

        (mu_model, _), (nu_model, _) = self.strategy.get_bridges(self.device)
        mu_model.eval()
        nu_model.eval()

        with torch.no_grad():
            for i in range(0, self.current_coupling.size, chunks):
                x_mu, y_nu = self.mu_distribution.dataset[i : i + chunks].to(
                    self.config.device
                ), self.nu_distribution.dataset[i : i + chunks].to(self.config.device)

                t_tensor = torch.full((x_mu.shape[0], 1), time_step / self.config.iterations).to(self.config.device)

                x_mu_next = x_mu.detach() + mu_model(t_tensor, x_mu) * (1 / self.config.iterations)
                y_nu_next = y_nu.detach() + nu_model(t_tensor, y_nu) * (1 / self.config.iterations)

                # x_next, y_next = self.strategy.advance(time_step, x_mu_next.clone(), y_nu_next.clone())

                num_steps = 50
                flow_model_mu = self.strategy.forward_model
                flow_model_nu = (
                    self.strategy.backward_model
                    if isinstance(self.strategy, BidirectionalStrategy)
                    else self.strategy.forward_model
                )
                device = self.config.device
                dt = 1.0 / num_steps
                if time_step % 2 == 0:
                    # x_next = x.detach() + delta * flow_mu(torch.cat([x, t_tensor], dim=1))
                    x_next = x_mu_next.detach().clone()

                    xt = x_next.clone()
                    for j in range(num_steps):
                        t = torch.ones(x_next.shape[0], 1).to(device) * (j * dt)
                        vt = flow_model_mu(torch.cat([xt, x_next, t], dim=1))[:, :2]
                        xt = xt + vt * dt
                    y_next = xt.clone()
                else:
                    # y_next = y.detach() + delta * flow_nu(torch.cat([y, t_tensor], dim=1))
                    y_next = y_nu_next.detach().clone()

                    yt = y_next.clone()
                    for j in range(num_steps):
                        t = torch.ones(y_next.shape[0], 1).to(device) * (j * dt)
                        vt = flow_model_nu(torch.cat([yt, y_next, t], dim=1))[:, :2]
                        yt = yt + vt * dt
                    x_next = yt.clone()

                self.current_coupling.dataset[:, i : i + chunks] = torch.stack([x_next, y_next], dim=0).cpu()
                self.mu_distribution.dataset[i : i + chunks] = x_mu_next.cpu()
                self.nu_distribution.dataset[i : i + chunks] = y_nu_next.cpu()

        # # Plot mu and nu distributions after advancing
        # mu_samples = self.mu_distribution.sample(2500).to(self.device)
        # nu_samples = self.nu_distribution.sample(2500).to(self.device)
        # # Get coupling samples
        # (mu_coupling_samples, nu_coupling_samples) = self.current_coupling.coupling_sample(250)
        # mu_coupling_samples = mu_coupling_samples.to(self.device)
        # nu_coupling_samples = nu_coupling_samples.to(self.device)
        # plt.figure(figsize=(6, 6))
        # plt.plot(mu_samples[:, 0].cpu(), mu_samples[:, 1].cpu(), '.', label='Mu Distribution', alpha=0.5)
        # plt.plot(nu_samples[:, 0].cpu(), nu_samples[:, 1].cpu(), '.', label='Nu Distribution', alpha=0.5)
        # plt.plot(mu_coupling_samples[:, 0].cpu(), mu_coupling_samples[:, 1].cpu(), 'x', label='Mu Coupling Samples', alpha=0.5)
        # plt.plot(nu_coupling_samples[:, 0].cpu(), nu_coupling_samples[:, 1].cpu(), 'x', label='Nu Coupling Samples', alpha=0.5)
        # # Line between coupling samples
        # for j in range(mu_coupling_samples.shape[0]):
        #     plt.plot([mu_coupling_samples[j, 0].cpu(), nu_coupling_samples[j, 0].cpu()],
        #              [mu_coupling_samples[j, 1].cpu(), nu_coupling_samples[j, 1].cpu()],
        #              'k-', alpha=0.1)
        # plt.title(f'Distributions after Advancing Step {time_step+1}')
        # plt.show()
