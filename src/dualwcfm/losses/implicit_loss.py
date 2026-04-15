import torch
import torch.nn as nn
import torch.optim as optim

from dualwcfm.strategy import Strategy
from dualwcfm.flow_matching import ConditionalFlowMatching
from dualwcfm.losses.base import FlowLoss, _expand_weight

class ImplicitFlowLoss(FlowLoss):
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
        previous_model = strategy.get_previous_flow(step, device)

        with torch.no_grad():
            # Compute log-weight before the backward swap
            log_w = strategy.log_weight(x0, x1, n_iter)
            weight = _expand_weight(torch.exp(log_w), x0)

        if backward and step % 2 != 0:
            x0, x1 = x1, x0

        batch_size = x0.shape[0]
        s = flow_matching.random_time(batch_size, device)
        xs = flow_matching.interpolate(x0, x1, s, sigma)

        with torch.no_grad():
            b_ta = previous_model(s, xs, x0)
            u_t = flow_matching.vector_field(xs, s, x0, x1, noisy=True)

        # Single forward pass -- use detach() for the target instead of running model twice.
        # Fixed-point analysis with factor (w - 1):
        #   u_s* = b_ta/w + (1 - 1/w)*u_t  →  diverges as w → 0, wrong direction for w < 1.
        # Correct factor is w:
        #   u_s* = (b_ta + w·u_t) / (1 + w)  →  stable for all w ≥ 0.
        u_s = model(s, xs, x0)
        with torch.no_grad():
            target = b_ta + weight * (u_t - u_s.detach())

        return ((u_s - target) ** 2).mean(), model, optimizer
