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
