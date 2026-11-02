import torch
import torch.nn as nn
import torch.optim as optim

from dualwcfm.strategy import Strategy
from dualwcfm.flow_matching import ConditionalFlowMatching
from dualwcfm.losses.base import FlowLoss, _expand_weight, _training_log_weight


class ExplicitFlowLoss(FlowLoss):
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
            # Weight factor for the ETM update.  The first-order Taylor
            # exp(h) ≈ 1 + h gives factor (1 + log_w), not bare log_w.
            # Using bare log_w drops the constant 1, so particles with
            # log_w < 0 (w < 1) get a negative factor that pushes the model
            # *away* from u_t and particles with log_w = 0 receive zero
            # correction (no learning at all).
            log_w = _training_log_weight(strategy, x0, x1, n_iter)
            h_r = _expand_weight(1.0 + log_w, x0)

        if backward and step % 2 != 0:
            x0, x1 = x1, x0

        # x0 is the condition; the flow state starts at z (noise source if enabled).
        z = flow_matching.source(x0)
        batch_size = x0.shape[0]
        s = flow_matching.random_time(batch_size, device)
        xs = flow_matching.interpolate(z, x1, s, sigma)

        with torch.no_grad():
            # b_ta: velocity field of the current model before update [cite: 361]
            b_ta = previous_model(s, xs, x0)
            # u_t: velocity of the interpolant (I_dot) [cite: 361, 1580]
            u_t = flow_matching.vector_field(xs, s, z, x1, noisy=True)

            # ETM Target calculation:
            # target = b_ta + (h * r) * (u_t - b_ta)
            target = b_ta + h_r * (u_t - b_ta)

        # Regression: Train the new model to match this first-order update [cite: 419, 1585]
        u_s = model(s, xs, x0)

        return ((u_s - target) ** 2).mean(), model, optimizer