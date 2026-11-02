import torch
import torch.nn as nn
import torch.optim as optim

from dualwcfm.utils import is_image_tensor
from dualwcfm.strategy import Strategy
from dualwcfm.flow_matching import ConditionalFlowMatching


def _expand_weight(weight: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expand weight [batch, 1] to match target dims (e.g. [batch, 1, 1, 1] for images)."""
    if is_image_tensor(target):
        return weight.view(weight.shape[0], *([1] * (target.dim() - 1)))
    return weight


def _training_log_weight(strategy: Strategy, x0: torch.Tensor, x1: torch.Tensor, n_iter: int) -> torch.Tensor:
    """Log importance weight used to re-train the coupling flow.

    Returns zeros when ``strategy.disable_tilt`` is set: the flow is then
    re-trained on its own samples without tilting, which isolates the
    contribution of the tilt in ablations.
    """
    log_w = strategy.log_weight(x0, x1, n_iter)
    if getattr(strategy, "disable_tilt", False):
        return torch.zeros_like(log_w)
    return log_w


class FlowLoss:
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
        """Compute the CFM loss."""
        raise NotImplementedError("method not implemented")

    @staticmethod
    def get_loss(loss_name: str):
        from dualwcfm.losses.weight_loss import WeightedFlowLoss
        from dualwcfm.losses.explicit_loss import ExplicitFlowLoss
        from dualwcfm.losses.implicit_loss import ImplicitFlowLoss

        if loss_name == "weighted":
            return WeightedFlowLoss()
        elif loss_name == "explicit":
            return ExplicitFlowLoss()
        elif loss_name == "implicit":
            return ImplicitFlowLoss()
        else:
            raise NameError(f"loss name `{loss_name}` does not exist (choose between weighted, explicit, implicit)")
