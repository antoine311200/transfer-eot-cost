from typing import Optional

from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn

from dualwcfm.models.cfm import ConditionalFlowMatching
from dualwcfm.trainers.dual_trainer import DualTrainer
from dualwcfm.trainers.trainer import Trainer
from dualwcfm.distributions.coupling import CouplingDistribution, MarginalDistribution
from dualwcfm.distributions.distribution import Distribution, DynamicLinearInterpolationDistribution

class AlternatingTrainer(DualTrainer):

    def step(self, batch_size: int, training: bool) -> tuple[float, float]: # type: ignore
        x0_new = self.source_dist.sample(batch_size).to(self.device)
        x1_new = self.target_dist.sample(batch_size).to(self.device)

        x0_ref, x1_ref = self.coupling_dist.coupling_sample(batch_size)
        x0_ref = x0_ref.to(self.device)
        x1_ref = x1_ref.to(self.device)

        loss_a = self._loss_a(x0_new, x0_ref, x1_ref)
        if training:
            self.optimizer_f.zero_grad()
            loss_a.backward()
            self.optimizer_f.step()
            self.scheduler_f.step()

        loss_b = self._loss_b(x1_new, x0_ref, x1_ref)
        if training:
            self.optimizer_g.zero_grad()
            loss_b.backward()
            self.optimizer_g.step()
            self.scheduler_g.step()

        return loss_a.item(), loss_b.item()

    def _loss_a(self, x0_new, x0_ref, x1_ref):
        a_new = self.f_net(x0_new)
        a_ref = self.f_net(x0_ref)
        b_ref = self.g_net(x1_ref).detach() # Not optimized here

        term_coupling = torch.mean(a_ref ** 2 + 2 * a_ref * b_ref)
        term_marginal = 2 * (torch.mean(a_new) - torch.mean(a_ref))

        return term_coupling - term_marginal

    def _loss_b(self, x1_new, x0_ref, x1_ref):
        b_new = self.g_net(x1_new)
        a_ref = self.f_net(x0_ref).detach() # Not optimized here
        b_ref = self.g_net(x1_ref)

        term_coupling = torch.mean(b_ref ** 2 + 2 * a_ref * b_ref)
        term_marginal = 2 * (torch.mean(b_new) - torch.mean(b_ref))

        return term_coupling - term_marginal

    def _init_train_tools(self, learning_rate: float, epochs: int):
        self.optimizer_f = optim.Adam(self.f_net.parameters(), lr=learning_rate)
        self.optimizer_g = optim.Adam(self.g_net.parameters(), lr=learning_rate)
        self.scheduler_f = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_f, T_max=epochs, eta_min=learning_rate*1e-3)
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_g, T_max=epochs, eta_min=learning_rate*1e-3)