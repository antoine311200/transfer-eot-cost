from typing import Optional

from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.optim as optim
from torch import nn

from dualwcfm.models.cfm import ConditionalFlowMatching
from dualwcfm.trainers.dual_trainer import DualTrainer, DualTrainerConfig
from dualwcfm.trainers.trainer import Trainer
from dualwcfm.distributions.coupling import CouplingDistribution, MarginalDistribution, WeightedCouplingDistribution
from dualwcfm.distributions.distribution import Distribution, DynamicLinearInterpolationDistribution


@dataclass
class AlternatingConfig(DualTrainerConfig):
    # learning_rate_f: float = 1e-3
    # learning_rate_g: float = 1e-3
    learning_rate: float = 1e-3
    epochs: int = 1000
    batch_size: int = 64
    eval_batch_size: int = 32
    mode: str = "integrator"  # "integrator" or "simple"


class AlternatingTrainer(DualTrainer):
    def __init__(
        self,
        coupling_dist: WeightedCouplingDistribution,
        source_dist: Distribution,
        target_dist: Distribution,
        f_net: nn.Module,
        g_net: nn.Module,
        device: torch.device,
        dynamic: bool = False,
        wrapper=None,
    ):
        super().__init__(coupling_dist, source_dist, target_dist, f_net, g_net, device, dynamic)
        self.wrapper = wrapper

        if self.wrapper is None:
            self.wrapper = lambda net: net

        self.f_net = self.wrapper(self.f_net)
        self.g_net = self.wrapper(self.g_net)

    def step(self, t: float, training: bool, **kwargs) -> float:  # type: ignore
        x0_new = self.source_dist.sample(self.config.batch_size).to(self.device)
        x1_new = self.target_dist.sample(self.config.batch_size).to(self.device)
        x0_init = self.coupling_dist.sample_init(self.config.batch_size).to(self.device)
        x1_init = self.coupling_dist.sample_target(self.config.batch_size).to(self.device)

        (x0_ref, x1_ref), log_weights = self.coupling_dist.coupling_sample(self.config.batch_size)
        x0_ref = x0_ref.to(self.device)
        x1_ref = x1_ref.to(self.device)
        log_weights = log_weights.to(self.device)

        loss = self.loss(t, x0_new, x1_new, x0_init, x1_init, x0_ref, x1_ref, log_weights, training=training)

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return loss.item()

    def loss(self, t, x0_new, x1_new, x0_init, x1_init, x0_ref, x1_ref, log_weights, training: bool = True):
        time = torch.ones(x0_new.shape[0], 1, device=self.device) * t if self.config.mode == "simple" else torch.rand(x0_new.shape[0], 1, device=self.device)

        if training:
            if self.config.mode == "integrator":
                time.requires_grad_(True)

            with torch.enable_grad():
                return self._compute_loss(time, x0_new, x1_new, x0_init, x1_init, x0_ref, x1_ref, log_weights, training)
        else:
            with torch.no_grad():
                if self.config.mode == "integrator":
                    time.requires_grad_(True)
                    with torch.enable_grad():
                        return self._compute_loss(time, x0_new, x1_new, x0_init, x1_init, x0_ref, x1_ref, log_weights, training)
                else:
                    return self._compute_loss(time, x0_new, x1_new, x0_init, x1_init, x0_ref, x1_ref, log_weights, training)


    def _compute_loss(self, time, x0_new, x1_new, x0_init, x1_init, x0_ref, x1_ref, log_weights, training):
        a_new = self.f_net(time, x0_new)
        b_new = self.g_net(time, x1_new)
        a_init = self.f_net(time, x0_init)
        b_init = self.g_net(time, x1_init)
        a_ref = self.f_net(time, x0_ref)
        b_ref = self.g_net(time, x1_ref)

        if self.config.mode == "simple":
            term_coupling = torch.mean((a_ref + b_ref) ** 2 * torch.exp(log_weights).unsqueeze(1))
            term_marginal = 2 * (torch.mean(a_new) + torch.mean(b_new) - torch.mean(a_ref) - torch.mean(b_ref))

            return term_coupling - term_marginal
        elif self.config.mode == "integrator":
            create_graph = training

            da_t_new = torch.autograd.grad(a_new, time, grad_outputs=torch.ones_like(a_new), create_graph=create_graph, retain_graph=create_graph)[0]
            db_t_new = torch.autograd.grad(b_new, time, grad_outputs=torch.ones_like(b_new), create_graph=create_graph, retain_graph=create_graph)[0]
            da_t_init = torch.autograd.grad(a_init, time, grad_outputs=torch.ones_like(a_init), create_graph=create_graph, retain_graph=create_graph)[0]
            db_t_init = torch.autograd.grad(b_init, time, grad_outputs=torch.ones_like(b_init), create_graph=create_graph, retain_graph=create_graph)[0]
            da_t_ref = torch.autograd.grad(a_ref, time, grad_outputs=torch.ones_like(a_ref), create_graph=create_graph, retain_graph=create_graph)[0]
            db_t_ref = torch.autograd.grad(b_ref, time, grad_outputs=torch.ones_like(b_ref), create_graph=create_graph, retain_graph=create_graph)[0]

            term_coupling = torch.mean((da_t_ref**2 + db_t_ref**2 + 4 * da_t_ref * db_t_ref) * torch.exp(a_ref + b_ref))
            term_marginal = 2 * (torch.mean(da_t_new + db_t_new) - torch.mean(da_t_init + db_t_init))

            return term_coupling - term_marginal
        else:
            raise ValueError(f"Unknown mode {self.config.mode}")

    def _init_train_tools(self):
        self.optimizer = optim.Adam(list(self.f_net.parameters()) + list(self.g_net.parameters()), lr=self.config.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=self.config.learning_rate * 1e-3)

    def train(  # type: ignore
        self,
        t: float,
        config: AlternatingConfig,
        plot_func=None,
        plot_every: int = 0,
        print_every: int = 100,
        eval_every: int = 10,
    ):
        self.config = config
        print(f"Starting alternating training for {config.epochs} epochs")
        print(f"Batch size: {config.batch_size}, Print every: {print_every}, Eval every: {eval_every}")
        print("-" * 80)

        self.train_losses = []
        self.eval_losses = []
        self._init_train_tools()

        pbar = tqdm(range(self.config.epochs))
        for epoch in pbar:
            loss = self.step(t, training=True)

            self.train_losses.append(loss)

            if (epoch + 1) % print_every == 0:
                pbar.set_description(
                    f"Epoch {epoch+1}/{self.config.epochs}, Loss: {loss:.4f}"
                )

            if (epoch + 1) % eval_every == 0:
                eval_loss = self.evaluate(t=t)
                self.eval_losses.append(eval_loss)

            if plot_func is not None and plot_every > 0 and (epoch + 1) % plot_every == 0:
                plot_func(f_net=self.f_net, g_net=self.g_net, epoch=epoch + 1)

        print("Training completed.")
        return self.train_losses, self.eval_losses