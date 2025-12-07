from typing import Optional

from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn

from dualwcfm.models.cfm import ConditionalFlowMatching
from dualwcfm.trainers.trainer import Trainer
from dualwcfm.distributions.coupling import CouplingDistribution
from dualwcfm.distributions.distribution import Distribution

class DualTrainer(Trainer):
    def __init__(
        self,
        coupling_dist: CouplingDistribution,
        source_dist: Distribution,
        target_dist: Distribution,
        f_net: nn.Module,
        g_net: nn.Module,
        device: torch.device
    ):
        self.coupling_dist = coupling_dist
        self.source_dist = source_dist
        self.target_dist = target_dist

        self.device = device

        self.f_net = f_net.to(device)
        self.g_net = g_net.to(device)

    def step(self, batch_size: int, training: bool) -> float:
        x0_new = self.source_dist.sample(batch_size).to(self.device)
        x1_new = self.target_dist.sample(batch_size).to(self.device)

        x0_ref, x1_ref = self.coupling_dist.coupling_sample(batch_size)
        x0_ref = x0_ref.to(self.device)
        x1_ref = x1_ref.to(self.device)

        f_new = self.f_net(x0_new)
        g_new = self.g_net(x1_new)

        f_ref = self.f_net(x0_ref)
        g_ref = self.g_net(x1_ref)

        loss_marginals = torch.mean(f_new) + torch.mean(g_new)
        loss_coupling = torch.mean(torch.exp(f_ref + g_ref))
        loss_reg = torch.mean(f_new**2) + torch.mean(g_new**2)
        loss = loss_coupling - loss_marginals + self.lambda_reg * loss_reg

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return loss.item()

    def evaluate(self, batch_size: int = 1000) -> float:
        self.f_net.eval()
        self.g_net.eval()

        with torch.no_grad():
            loss = self.step(batch_size=batch_size, training=False)

        self.f_net.train()
        self.g_net.train()
        return loss

    def train(self, epochs: int, batch_size: int, learning_rate: float, print_every: int = 100, eval_every: int = 10, lambda_reg: float = 0.0):
        print(f"Starting training for {epochs} epochs")
        print(f"Batch size: {batch_size}, Print every: {print_every}, Eval every: {eval_every}")
        print("-" * 80)

        self.lambda_reg = lambda_reg

        self.train_losses = []
        self.epoch_train_losses = []
        self.eval_losses = []

        self.optimizer = optim.Adam(list(self.f_net.parameters()) + list(self.g_net.parameters()), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=learning_rate * 1e-3)

        pbar = tqdm(range(epochs), desc="Training")

        for epoch in pbar:

            loss = self.step(batch_size=batch_size, training=True)

            self.train_losses.append(loss)
            self.epoch_train_losses.append(loss)

            if (epoch + 1) % eval_every == 0 or epoch == 0:
                eval_loss = self.evaluate(batch_size=batch_size)
                self.eval_losses.append((epoch, eval_loss))

            if (epoch + 1) % print_every == 0 or epoch == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                postfix = {
                    'train_loss': f"{loss:.4f}",
                    'lr': f"{current_lr:.2e}"
                }

                if (epoch + 1) % eval_every == 0 or epoch == 0:
                    eval_loss = self.eval_losses[-1][1]
                    postfix['eval_loss'] = f"{eval_loss:.4f}"

                pbar.set_postfix(postfix)

        print("-" * 80)
        print("Training completed!")
        print(f"Final train loss: {self.epoch_train_losses[-1]:.6f}")
        if self.eval_losses:
            print(f"Final eval loss: {self.eval_losses[-1][1]:.6f}")
        print(f"Total training steps: {len(self.train_losses)}")

        return {
            'train_losses': self.train_losses,
            'epoch_train_losses': self.epoch_train_losses,
            'eval_losses': self.eval_losses
        }