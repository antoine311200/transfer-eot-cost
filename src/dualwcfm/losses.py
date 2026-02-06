import torch
import torch.nn as nn
import torch.optim as optim

from dualwcfm.config import TrainingConfig
from dualwcfm.utils import is_image_tensor
from dualwcfm.strategy import Strategy
from dualwcfm.flow_matching import ConditionalFlowMatching


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

            # For images, expand weight to broadcast properly
            if is_image_tensor(x0):
                while weight.dim() < x0.dim():
                    weight = weight.unsqueeze(-1)

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
            # For dual models: output shape is [batch, 1]
            a, b = f_dual(x0), g_dual(x1)
            r = a + b
            log_w = r / config.iterations
            weight = torch.exp(log_w)

            # For images, expand weight to broadcast properly: [batch, 1] -> [batch, 1, 1, 1]
            if is_image_tensor(x0):
                while weight.dim() < x0.dim():
                    weight = weight.unsqueeze(-1)

            if step % 2 != 0:
                x0, x1 = x1, x0

            b_hat_detached = model(s, xs, x0)
            b_ta = previous_model(s, xs, x0)

            u_t = flow_matching.vector_field(x0, torch.zeros_like(x0), x0, x1)
            target = b_ta + (weight - 1) * (u_t - b_hat_detached)
        u_s = model(s, xs, x0)
        return ((u_s - target) ** 2).mean(), model, optimizer
