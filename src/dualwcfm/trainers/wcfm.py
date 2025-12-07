from typing import Optional

from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn

from dualwcfm.models.cfm import ConditionalFlowMatching
from dualwcfm.trainers.trainer import Trainer
from dualwcfm.utils import pad_t_like_x

class WeightedConditionalFlowMatchingTrainer(Trainer):
    """Conditional flow matching with learned vector field

    Args:
        flow_matching: Conditional flow matching model
        f_net: Optional neural network for source log-density estimation
        g_net: Optional neural network for target log-density estimation
        mode: Training mode, either "classic" or "weighted"
    """

    def __init__(
        self,
        flow_matching: ConditionalFlowMatching,
        f_net: Optional[nn.Module] = None,
        g_net: Optional[nn.Module] = None,
        mode: str = "classic"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.f_net = f_net.to(self.device) if f_net is not None else None
        self.g_net = g_net.to(self.device) if g_net is not None else None
        self.mode = mode

        self.flow_matching = flow_matching
        self.flow_matching.network.to(self.device)

    def step(self, batch_size: int, training: bool) -> float:
        x0_batch = self.flow_matching.source_dist.sample(batch_size).to(self.device)
        x1_batch = self.flow_matching.target_dist.sample(batch_size).to(self.device)

        t = torch.rand(batch_size, 1, device=self.device)
        t = pad_t_like_x(t, x0_batch).to(self.device)
        x_t = self.flow_matching.compute_conditional_xt(x0_batch, x1_batch, t)
        true_flow = self.flow_matching.compute_conditional_flow(x0_batch, x1_batch, t)
        # predicted_flow = self.flow_matching.network(torch.cat([x_t, t], dim=1))
        predicted_flow = self.flow_matching.network(t, x_t)


        if self.mode == "weighted" and self.f_net and self.g_net:
            self.f_net.eval()
            self.g_net.eval()

            with torch.no_grad():
                f_x0 = self.f_net(x0_batch)
                g_x1 = self.g_net(x1_batch)

                if f_x0.shape[1] > 1: log_w = f_x0.sum(dim=1) + g_x1.sum(dim=1)
                else: log_w = f_x0.squeeze(1) + g_x1.squeeze(1)

                weights = torch.exp(log_w)
                weights = weights / (weights.mean() + 1e-8)
                weights = weights.unsqueeze(1).unsqueeze(2)

            loss = torch.mean(weights * (predicted_flow - true_flow) ** 2)
        else:
            loss = torch.mean((predicted_flow - true_flow) ** 2)

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return loss.item()

    def evaluate(self, batch_size: int = 1000) -> float:
        self.flow_matching.network.eval()

        with torch.no_grad():
            loss = self.step(batch_size=batch_size, training=False)

        self.flow_matching.network.train()
        return loss

    def train(self, epochs: int, batch_size: int, learning_rate: float, print_every: int = 100, eval_every: int = 10):
        self.train_losses = []
        self.epoch_train_losses = []
        self.eval_losses = []

        self.optimizer = optim.Adam(self.flow_matching.network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)

        self.flow_matching.network.train()

        print(f"Starting training for {epochs} epochs")
        print(f"Batch size: {batch_size}, Print every: {print_every}, Eval every: {eval_every}")
        print("-" * 80)

        pbar = tqdm(range(epochs), desc="Training")

        for epoch in pbar:
            epoch_losses = []

            loss = self.step(batch_size=batch_size, training=True)

            epoch_losses.append(loss)
            self.train_losses.append(loss)

            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.epoch_train_losses.append(avg_epoch_loss)

            if (epoch + 1) % eval_every == 0 or epoch == 0:
                eval_loss = self.evaluate(batch_size=batch_size)
                self.eval_losses.append((epoch, eval_loss))

            if (epoch + 1) % print_every == 0 or epoch == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                postfix = {
                    'train_loss': f"{avg_epoch_loss:.4f}",
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