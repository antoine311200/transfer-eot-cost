import copy
from typing import Optional, Union
from functools import partial
from dualwcfm.trainers.alternating_trainer import AlternatingTrainer
from typing_extensions import Literal

from dualwcfm.trainers.wcfm import CFMTrainerConfig
import torch
import torch.nn as nn

from dualwcfm.distributions import CouplingDistribution, Distribution, LinearInterpolationDistribution, MarginalDistribution
from dualwcfm.trainers import DualTrainer, WeightedConditionalFlowMatchingTrainer
from dualwcfm.trainers.trainer import Trainer, TrainerConfig
from dualwcfm.trainers.dual_trainer import DualTrainerConfig
from dualwcfm.models.cfm import ConditionalFlowMatching
from dualwcfm.utils import safe_clone

class IterativeTrainer:
    """Class to handle iterative training of dual and weighted CFM models between source and target distributions using the coupling distribution.

    Attributes:
        coupling_dist (CouplingDistribution): The coupling distribution used for sampling.
        source_dist (Distribution): The initial source distribution.
        target_dist (Distribution): The initial target distribution.
    """

    def __init__(
        self,
        source_dist: Distribution,
        target_dist: Distribution,
        coupling_dist: CouplingDistribution,
        eval_coupling_dist: Optional[CouplingDistribution] = None,
        mode: str = "static"
    ):
        self.coupling_dist = coupling_dist
        self.source_dist = MarginalDistribution(self.coupling_dist.sample_init)
        self.target_dist = MarginalDistribution(self.coupling_dist.sample_target)
        self.eval_coupling_dist = eval_coupling_dist if eval_coupling_dist else None
        self.new_source_dist = source_dist
        self.new_target_dist = target_dist
        self.dynamic = True if mode == "dynamic" else False
        self.mode = mode

    def train(
        self,
        n_iterations: int,
        f_net: nn.Module,
        g_net: nn.Module,
        flow_network: nn.Module,
        device: torch.device,

        dual_config: DualTrainerConfig,
        cfm_config: CFMTrainerConfig,
        weight_mode: Literal['classic', 'exp_weighted', 'exact_weighted'] = "exact_weighted",

        torch_wrapper = None,
        return_cfm: bool = False,
        reinitialize_duals: bool = False,
        plot_every: int = 100,
        print_every: int = 100,
        eval_every: int = 10,
        plot_func = None,
        **kwargs
    ) -> Union[tuple[dict, dict], dict]:
        """Trains dual and weighted CFM models iteratively between the source and target distributions.

        Args:
            n_iterations (int): Number of interpolation steps between source and target distributions.
            f_net (nn.Module): Neural network for the dual potential f.
            g_net (nn.Module): Neural network for the dual potential g.
            flow_network (nn.Module): Neural network for the flow model.
            device (torch.device): Device to run the training on.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs for each training phase.
            learning_rate (float): Learning rate for the optimizers.
            sigma (float): Noise parameter for the CFM model.
            torch_wrapper: Wrapper for PyTorch models.

        Returns:
            dict: A dictionary containing training information for each interpolation step.
        """
        if torch_wrapper is None:
            torch_wrapper = lambda net: net

        f_net_clone = torch_wrapper(safe_clone(f_net))
        g_net_clone = torch_wrapper(safe_clone(g_net))

        self.flow_network = flow_network

        self.weight_mode = weight_mode

        info_dict = {}
        cfms = {}

        for step in range(n_iterations):
            print(f"Starting iteration {step}/{n_iterations}")

            if self.mode == "static":
                print(f" with alpha={step / (n_iterations):.2f}")
                alpha = step / (n_iterations - 1)
                mixed_source_dist = LinearInterpolationDistribution(self.source_dist, self.new_source_dist, alpha)
                mixed_target_dist = LinearInterpolationDistribution(self.target_dist, self.new_target_dist, alpha)
            else:
                mixed_source_dist = self.new_source_dist
                mixed_target_dist = self.new_target_dist

            if self.weight_mode == "classic" or self.weight_mode == "exp_weighted":
                dual_trainer = DualTrainer(
                    coupling_dist=self.coupling_dist,
                    source_dist=mixed_source_dist,
                    target_dist=mixed_target_dist,
                    f_net=f_net_clone if not reinitialize_duals else safe_clone(f_net),
                    g_net=g_net_clone if not reinitialize_duals else safe_clone(g_net),
                    dynamic=self.dynamic,
                    device=device
                )
            else:  # exact_weighted
                print("Using Alternating Trainer for exact weighted mode")
                dual_trainer = AlternatingTrainer(
                    coupling_dist=self.coupling_dist,
                    source_dist=mixed_source_dist,
                    target_dist=mixed_target_dist,
                    f_net=f_net_clone if not reinitialize_duals else safe_clone(f_net),
                    g_net=g_net_clone if not reinitialize_duals else safe_clone(g_net),
                    dynamic=self.dynamic,
                    device=device
                )
            dual_info = dual_trainer.train(
                config=dual_config,
                print_every=print_every,
                eval_every=eval_every
            )

            print(
                f"Completed training duals"
                f" for alpha={step / (n_iterations - 1):.2f}" if not self.dynamic else ""
            )

            cfm_new = ConditionalFlowMatching(
                source_dist=mixed_source_dist,
                target_dist=mixed_target_dist,
                network=self.flow_network,
                wrapper=torch_wrapper,
                sigma=cfm_config.sigma
            )
            weighted_trainer = WeightedConditionalFlowMatchingTrainer(
                flow_matching=cfm_new,
                mode=weight_mode,
                f_net=dual_trainer.f_net,
                g_net=dual_trainer.g_net,
                dynamic=self.dynamic
            )
            print("Delta", (step + 1) / (n_iterations - 1))
            weighted_info = weighted_trainer.train(
                config=cfm_config,
                delta=(step + 1) / n_iterations,
                device=device,
                plot_func=partial(plot_func, coupling_dist=self.coupling_dist, eval_coupling_dist=self.eval_coupling_dist, new_source_dist=self.new_source_dist, new_target_dist=self.new_target_dist, device=device) if plot_func is not None else None,
                plot_every=plot_every,
                print_every=print_every,
                eval_every=eval_every
            )

            if return_cfm and not self.dynamic:
                cfm_net = safe_clone(self.flow_network)
                cfms[step] = cfm_net

            if plot_func is None:
                from dualwcfm.visualization import plot_2d_flow_trajectories
                plot_2d_flow_trajectories(
                    trajectories=cfm_new.get_trajectory(n_samples=500, n_timesteps=25),
                    t=torch.linspace(0, 1, 25),
                    n_trajectories=500,
                    title=f"CFM Trajectories"
                )
            else:
                plot_func(cfm_new, step, self.coupling_dist, self.eval_coupling_dist, self.new_source_dist, self.new_target_dist, device)

            print(f"Completed training weighted CFM")

            f_net_clone.load_state_dict(dual_trainer.f_net.state_dict())
            g_net_clone.load_state_dict(dual_trainer.g_net.state_dict())

            info_dict[step] = {
                "dual_info": dual_info,
                "weighted_info": weighted_info
            }

        if return_cfm:
            return info_dict, cfms
        else:
            return info_dict