import copy
from typing import Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from dualwcfm.trainers.iterative_trainer import IterativeTrainer
from dualwcfm.distributions import CouplingDistribution, Distribution, LinearInterpolationDistribution, MarginalDistribution
from dualwcfm.trainers import DualTrainer, WeightedConditionalFlowMatchingTrainer
from dualwcfm.models.cfm import ConditionalFlowMatching
from dualwcfm.utils import safe_clone

class DynamicTrainer(IterativeTrainer):
    """Class to handle dynamic training of dual and weighted CFM models between source and target distributions using the coupling distribution
    in a dynamic manner performing random marginal interpolation without needing explicit iterations.

    Attributes:
        coupling_dist (CouplingDistribution): The coupling distribution used for sampling.
        source_dist (Distribution): The initial source distribution.
        target_dist (Distribution): The initial target distribution.
    """

    def __init__(self, source_dist: Distribution, target_dist: Distribution, coupling_dist: CouplingDistribution, eval_coupling_dist: Optional[CouplingDistribution] = None):
        super().__init__(source_dist, target_dist, coupling_dist, eval_coupling_dist)

    def train( # type: ignore
        self,
        f_net: nn.Module,
        g_net: nn.Module,
        flow_network: nn.Module,
        device: torch.device,
        batch_size: int,
        epochs_dual: int,
        epochs: int,
        dual_learning_rate: float,
        learning_rate: float,
        sigma: float,
        lambda_reg: float = 0.1,
        torch_wrapper = None,
        return_cfm: bool = False,
        plot_every: int = 0,
        print_every: int = 100,
        eval_every: int = 10,
        plot_func = None
    ) -> Union[tuple[dict, dict], dict]:
        return super().train(
            n_iterations=1,
            f_net=f_net,
            g_net=g_net,
            flow_network=flow_network,
            device=device,
            batch_size=batch_size,
            epochs_dual=epochs_dual,
            epochs=epochs,
            dual_learning_rate=dual_learning_rate,
            learning_rate=learning_rate,
            sigma=sigma,
            lambda_reg=lambda_reg,
            torch_wrapper=torch_wrapper,
            return_cfm=return_cfm,
            plot_every=plot_every,
            print_every=print_every,
            eval_every=eval_every,
            plot_func=plot_func
        )