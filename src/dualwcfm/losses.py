import torch
import torch.nn as nn
import torch.optim as optim

from dualwcfm.config import TrainingConfig
from dualwcfm.utils import is_image_tensor
from dualwcfm.strategy import Strategy
from dualwcfm.flow_matching import ConditionalFlowMatching


def _expand_weight(weight: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expand weight [batch, 1] to match target dims (e.g. [batch, 1, 1, 1] for images)."""
    if is_image_tensor(target):
        return weight.view(weight.shape[0], *([1] * (target.dim() - 1)))
    return weight


class CFMLoss:
    def execute(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        strategy: Strategy,
        config: TrainingConfig,
        flow_matching: ConditionalFlowMatching,
        step: int,
        backward: bool = False,
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
        backward: bool = False,
    ) -> tuple[torch.Tensor, nn.Module, optim.Adam]:
        f_dual, g_dual, _ = strategy.get_duals(config.device)
        h_dual, _ = strategy.get_dual_mismatch(config.device)
        model, optimizer = strategy.get_flow(step, config.device)

        with torch.no_grad():
            a, b = f_dual(x0), g_dual(x1)
            d = h_dual(torch.cat([x0, x1], dim=1)) if config.dual_mismatch_reg > 0 else 0.0
            log_w = (a + b + d) / config.iterations
            weight = _expand_weight(torch.exp(log_w), x0)

            # Network a and b take non-permuted inputs

            if backward and step % 2 != 0:
                x0, x1 = x1, x0

            # target_u_s = torch.cat([x1 - x0, torch.zeros_like(x0)], dim=-1)

        # s = torch.rand(config.batch_size, 1, device=config.device)
        s = flow_matching.random_time(config.batch_size, config.device)
        x_s = flow_matching.interpolate(x0, x1, s, config.sigma2)
        u_s = model(s, x_s, x0)
        target_u_s = flow_matching.vector_field(x_s, s, x0, x1, extend=True, noisy=True)

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
        backward: bool = False,
    ) -> tuple[torch.Tensor, nn.Module, optim.Adam]:
        f_dual, g_dual, _ = strategy.get_duals(config.device)
        h_dual, _ = strategy.get_dual_mismatch(config.device)
        model, optimizer = strategy.get_flow(step, config.device)
        previous_model = strategy.get_previous_flow(step, config.device)

        with torch.no_grad():
            a, b = f_dual(x0), g_dual(x1)
            d = h_dual(torch.cat([x0, x1], dim=1)) if config.dual_mismatch_reg > 0 else 0.0
            log_w = (a + b + d) / config.iterations
            weight = _expand_weight(torch.exp(log_w), x0)

        if backward and step % 2 != 0:
            x0, x1 = x1, x0

        # s = torch.rand(config.batch_size, 1, device=config.device)
        s = flow_matching.random_time(config.batch_size, config.device)
        xs = flow_matching.interpolate(x0, x1, s, config.sigma)

        with torch.no_grad():
            b_ta = previous_model(s, xs, x0)
            u_t = flow_matching.vector_field(xs, s, x0, x1, noisy=True)

        # Single forward pass -- use detach() for the target instead of running model twice
        u_s = model(s, xs, x0)
        with torch.no_grad():
            target = b_ta + (weight - 1) * (u_t - u_s.detach())

        return ((u_s - target) ** 2).mean(), model, optimizer
