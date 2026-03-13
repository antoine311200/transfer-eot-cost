import torch
import torch.nn as nn
import torch.optim as optim

from dualwcfm.strategy import Strategy
from dualwcfm.flow_matching import ConditionalFlowMatching
from dualwcfm.losses.base import FlowLoss, _expand_weight


class ExplicitCFMLoss(FlowLoss):
    pass