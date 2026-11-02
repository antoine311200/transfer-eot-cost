import torch
import torch.nn as nn
import torch.optim as optim

from dualwcfm.strategy import Strategy
from dualwcfm.flow_matching import ConditionalFlowMatching
from dualwcfm.losses.base import FlowLoss, _expand_weight, _training_log_weight

class WeightedFlowLoss(FlowLoss):
    def execute(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        strategy: Strategy,
        flow_matching: ConditionalFlowMatching,
        step: int,
        n_iter: int,
        sigma: float,
        backward: bool = False,
    ) -> tuple[torch.Tensor, nn.Module, optim.Adam]:
        device = x0.device
        model, optimizer = strategy.get_flow(step, device)

        with torch.no_grad():
            # Compute log-weight before the backward swap
            log_w = _training_log_weight(strategy, x0, x1, n_iter)
            weight = _expand_weight(torch.exp(log_w), x0)

            if backward and step % 2 != 0:
                x0, x1 = x1, x0

        # x0 is the condition; the flow state starts at z (= x0 unless a noise
        # source is enabled, see ConditionalFlowMatching.source).
        z = flow_matching.source(x0)
        batch_size = x0.shape[0]
        s = flow_matching.random_time(batch_size, device)
        x_s = flow_matching.interpolate(z, x1, s, sigma)
        u_s = model(s, x_s, x0)
        target_u_s = flow_matching.vector_field(x_s, s, z, x1, extend=True, noisy=True)

        return (weight * (u_s - target_u_s) ** 2).mean(), model, optimizer
