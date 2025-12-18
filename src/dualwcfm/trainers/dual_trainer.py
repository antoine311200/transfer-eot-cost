from typing import Any, Optional

from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.optim as optim
from torch import nn

from dualwcfm.models.cfm import ConditionalFlowMatching
from dualwcfm.trainers.trainer import Trainer, TrainerConfig
from dualwcfm.distributions.coupling import CouplingDistribution, MarginalDistribution
from dualwcfm.distributions.distribution import Distribution, DynamicLinearInterpolationDistribution

@dataclass
class DualTrainerConfig(TrainerConfig):
    learning_rate: float = 1e-3
    epochs: int = 1000
    batch_size: int = 64
    eval_batch_size: int = 32
    lambda_reg: float = 0.0

class DualTrainer(Trainer):
    def __init__(
        self,
        coupling_dist: CouplingDistribution,
        source_dist: Distribution,
        target_dist: Distribution,
        f_net: nn.Module,
        g_net: nn.Module,
        device: torch.device,
        dynamic: bool = False,
    ):
        self.dynamic = dynamic

        self.coupling_dist = coupling_dist

        if not dynamic:
            self.source_dist = source_dist
            self.target_dist = target_dist
        else:
            self.source_dist = DynamicLinearInterpolationDistribution(
                MarginalDistribution(coupling_dist.sample_init),
                source_dist,
                get_alpha_fn=lambda: torch.rand(1).item()
            )
            self.target_dist = DynamicLinearInterpolationDistribution(
                MarginalDistribution(coupling_dist.sample_target),
                target_dist,
                get_alpha_fn=lambda: torch.rand(1).item()
            )

        self.device = device

        self.f_net = f_net.to(device)
        self.g_net = g_net.to(device)

    def step(self, training: bool, **kwargs) -> float:
        batch_size = self.config.batch_size if training else self.config.eval_batch_size
        x0_new = self.source_dist.sample(batch_size).to(self.device)
        x1_new = self.target_dist.sample(batch_size).to(self.device)

        x0_ref, x1_ref = self.coupling_dist.coupling_sample(batch_size)
        x0_ref = x0_ref.to(self.device)
        x1_ref = x1_ref.to(self.device)

        if not self.dynamic:
            f_new = self.f_net(x0_new)
            g_new = self.g_net(x1_new)

            f_ref = self.f_net(x0_ref)
            g_ref = self.g_net(x1_ref)
        else:
            # Typing linter workaround (but unnecessary at runtime)
            if isinstance(self.source_dist, DynamicLinearInterpolationDistribution) and isinstance(self.target_dist, DynamicLinearInterpolationDistribution):
                f_alpha = torch.tensor(self.source_dist.current_alpha, device=self.device)
                g_alpha = torch.tensor(self.target_dist.current_alpha, device=self.device)
                ones = torch.ones(batch_size, 1, device=self.device)

                # Repeat alphas to match batch size
                f_alpha = f_alpha.repeat(batch_size, 1)
                g_alpha = g_alpha.repeat(batch_size, 1)
            else:
                raise ValueError("For dynamic training, source and target distributions must be DynamicLinearInterpolationDistribution.")

            # f_new = self.f_net(torch.cat([f_alpha, x0_new], dim=1))
            # g_new = self.g_net(torch.cat([g_alpha, x1_new], dim=1))

            # f_ref = self.f_net(torch.cat([ones, x0_ref], dim=1))
            # g_ref = self.g_net(torch.cat([ones, x1_ref], dim=1))

            f_new = self.f_net(f_alpha, x0_new)
            g_new = self.g_net(g_alpha, x1_new)
            f_ref = self.f_net(ones, x0_ref)
            g_ref = self.g_net(ones, x1_ref)

        loss_marginals = torch.mean(f_new) + torch.mean(g_new)
        loss_coupling = torch.mean(torch.exp(f_ref + g_ref))
        loss_reg = torch.mean(f_new**2) + torch.mean(g_new**2)
        loss = loss_coupling - loss_marginals + self.config.lambda_reg * loss_reg

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return loss.item()

    def evaluate(self, **kwargs) -> Any:
        self.f_net.eval()
        self.g_net.eval()

        with torch.no_grad():
            loss = self.step(training=False, **kwargs)

        self.f_net.train()
        self.g_net.train()
        return loss

    def _init_train_tools(self):
        self.optimizer = optim.Adam(list(self.f_net.parameters()) + list(self.g_net.parameters()), lr=self.config.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=self.config.learning_rate * 1e-3)

    def train( # type: ignore
        self,
        config: DualTrainerConfig,
        print_every: int = 100,
        eval_every: int = 10,
        **kwargs
    ):
        print(f"Starting training for {config.epochs} epochs")
        print(f"Batch size: {config.batch_size}, Print every: {print_every}, Eval every: {eval_every}")
        print("-" * 80)

        self.config = config

        self.train_losses = []
        self.epoch_train_losses = []
        self.eval_losses = []

        self._init_train_tools()

        pbar = tqdm(range(config.epochs), desc="Training")

        for epoch in pbar:

            loss = self.step(batch_size=config.batch_size, training=True)
            self.train_losses.append(loss)
            self.epoch_train_losses.append(loss)

            if (epoch + 1) % eval_every == 0 or epoch == 0:
                eval_loss = self.evaluate(batch_size=config.batch_size)
                self.eval_losses.append((epoch, eval_loss))

            if (epoch + 1) % print_every == 0 or epoch == 0:
                # current_lr = self.scheduler.get_last_lr()[0]
                if isinstance(loss, tuple):
                    postfix = {
                        'train_loss_1': f"{loss[0]:.4f}",
                        'train_loss_2': f"{loss[1]:.4f}",
                        # 'lr': f"{current_lr:.2e}"
                    }
                else:
                    postfix = {
                        'train_loss': f"{loss:.4f}",
                        # 'lr': f"{current_lr:.2e}"
                    }

                if (epoch + 1) % eval_every == 0 or epoch == 0:
                    eval_loss = self.eval_losses[-1][1]
                    if isinstance(loss, tuple):
                        postfix['eval_loss_1'] = f"{eval_loss[0]:.4f}"
                        postfix['eval_loss_2'] = f"{eval_loss[1]:.4f}"
                    else:
                        postfix['eval_loss'] = f"{eval_loss:.4f}"

                pbar.set_postfix(postfix)

        print("-" * 80)
        print("Training completed!")
        # print(f"Final train loss: {self.epoch_train_losses[-1]:.6f}")
        # if self.eval_losses:
        #     print(f"Final eval loss: {self.eval_losses[-1][1]:.6f}")
        print(f"Total training steps: {len(self.train_losses)}")

        return {
            'train_losses': self.train_losses,
            'epoch_train_losses': self.epoch_train_losses,
            'eval_losses': self.eval_losses
        }