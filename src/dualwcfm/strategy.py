from typing import Optional
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim

from dualwcfm.config import FeedForwardConfig
from dualwcfm.utils import is_image_tensor
from dualwcfm.mlp import MLP

logger = logging.getLogger("dualwcfm")


class PairwiseMLP(nn.Module):
    """MLP accepting concatenated (x, y) → scalar. Use as change_of_measure."""

    def __init__(self, dim: int, hidden_size: int):
        super().__init__()
        self.net = MLP(dim=2 * dim, out_dim=1, w=hidden_size, time_varying=False)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


class Strategy:
    """Base strategy for training transport models.

    This class can be subclassed to customize model architectures.
    For custom models (e.g., UNets for images), override _create_models().
    """

    def __init__(
        self,
        wrapper,
        config: FeedForwardConfig,
        change_of_measure: Optional[nn.Module] = None,
    ):
        self.wrapper = wrapper
        self.config = config
        self._device = None
        self.change_of_measure = change_of_measure

        dim = config.dimension
        hidden_size = config.hidden_size
        num_layers = config.num_layers

        self._create_models(dim, hidden_size, num_layers)
        self._create_optimizers(config)

        if change_of_measure is not None:
            self._com_optimizer = optim.Adam(change_of_measure.parameters(), lr=config.dual_learning_rate)
        else:
            self._com_optimizer = None

    def _create_models(self, dim: int, hidden_size: int, num_layers: int = 2):
        """Create all models. Override this method for custom architectures."""
        self._device = None  # Reset so next to_device() call actually moves models
        wrapper = self.wrapper
        self.mu_bridge = wrapper(MLP(dim=dim, out_dim=dim, w=hidden_size, n_layers=num_layers, time_varying=True))
        self.nu_bridge = wrapper(MLP(dim=dim, out_dim=dim, w=hidden_size, n_layers=num_layers, time_varying=True))
        self.f_dual = wrapper(MLP(dim=dim, out_dim=1, w=hidden_size, n_layers=num_layers, time_varying=False))
        self.g_dual = wrapper(MLP(dim=dim, out_dim=1, w=hidden_size, n_layers=num_layers, time_varying=False))
        self.h_dual = MLP(dim=2*dim, out_dim=1, w=hidden_size, n_layers=num_layers, time_varying=False)
        self.forward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, n_layers=num_layers, time_varying=True))
        self.previous_forward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, n_layers=num_layers, time_varying=True))

        # Save initial state
        self.initial_mu_bridge = wrapper(MLP(dim=dim, out_dim=dim, w=hidden_size, n_layers=num_layers, time_varying=True))
        self.initial_nu_bridge = wrapper(MLP(dim=dim, out_dim=dim, w=hidden_size, n_layers=num_layers, time_varying=True))
        self.initial_forward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, n_layers=num_layers, time_varying=True))

    def to_device(self, device: torch.device):
        """Move all models to device once. Call this before training."""
        if self._device == device:
            return
        self._device = device
        for attr in self._model_attrs():
            getattr(self, attr).to(device)
        if self.change_of_measure is not None:
            self.change_of_measure.to(device)

    def _model_attrs(self):
        """Return names of all model attributes."""
        return [
            "mu_bridge", "nu_bridge", "f_dual", "g_dual", "h_dual",
            "forward_model", "previous_forward_model",
            "initial_mu_bridge", "initial_nu_bridge", "initial_forward_model",
        ]

    def get_output_channels(self, x: torch.Tensor) -> int:
        """Get the number of channels to extract from model output for images."""
        if is_image_tensor(x):
            return x.shape[1]  # Number of channels
        return x.shape[1] // 2  # Half of vector dimension

    def _create_optimizers(self, config: FeedForwardConfig):
        dual_params = list(self.f_dual.parameters()) + list(self.g_dual.parameters())
        self.mu_optimizer = optim.Adam(self.mu_bridge.parameters(), lr=config.mu_learning_rate, weight_decay=config.dual_l2_reg)
        self.nu_optimizer = optim.Adam(self.nu_bridge.parameters(), lr=config.nu_learning_rate, weight_decay=config.dual_l2_reg)
        self.dual_optimizer = optim.Adam(dual_params, lr=config.dual_learning_rate, weight_decay=config.dual_l2_reg)
        self.dual_mismatch_optimizer = optim.Adam(self.h_dual.parameters(), lr=config.dual_learning_rate)
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=config.flow_learning_rate, weight_decay=config.dual_l2_reg)

    def log_weight(self, x: torch.Tensor, y: torch.Tensor, n_iter: int) -> torch.Tensor:
        """Compute log-weight (f(x) + g(y) [+ com(x,y)]) / n_iter."""
        f, g, _ = self.get_duals(x.device)
        a, b = f(x), g(y)
        if self.change_of_measure is not None:
            xy = torch.cat([x.view(x.shape[0], -1), y.view(y.shape[0], -1)], dim=1)
            d = self.change_of_measure(xy)
        else:
            d = 0.0
        return (a + b + d) / n_iter

    def weight(self, x: torch.Tensor, y: torch.Tensor, n_iter: int) -> torch.Tensor:
        """Compute importance weight exp(log_weight(x, y, n_iter))."""
        return torch.exp(self.log_weight(x, y, n_iter))

    def get_com_optimizer(self) -> Optional[optim.Optimizer]:
        return self._com_optimizer

    def get_duals(self, device):
        self.to_device(device)
        return self.f_dual, self.g_dual, self.dual_optimizer

    def get_dual_mismatch(self, device):
        self.to_device(device)
        return self.h_dual, self.dual_mismatch_optimizer

    def get_bridges(self, device):
        self.to_device(device)
        return (self.mu_bridge, self.mu_optimizer), (self.nu_bridge, self.nu_optimizer)

    def get_flow(self, step: int, device):
        self.to_device(device)
        return self.forward_model, self.forward_optimizer

    def get_previous_flow(self, step: int, device):
        self.to_device(device)
        return self.previous_forward_model

    def get_initial_bridges(self, device):
        self.to_device(device)
        return self.initial_mu_bridge, self.initial_nu_bridge

    def get_initial_flow(self, step: int, device):
        self.to_device(device)
        return self.initial_forward_model

    def advance(
        self, step: int, x_mu_next: torch.Tensor, y_nu_next: torch.Tensor, num_steps: int = 100, sigma: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the coupling using the current flow models."""
        x_condition = x_mu_next.clone()
        xt = x_mu_next.clone()
        dt = 1.0 / num_steps
        device = x_mu_next.device
        model = self.get_flow(step, device)[0]
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((xt.shape[0], 1), i * dt, device=device)
                vt = model(t, xt, x_condition)[:, : xt.shape[1] // 2]
                xt = xt + vt * dt
        return x_mu_next, xt.clone()

    def _load_pretrained_models(self, dir, device):
        """Load pretrained models from disk."""
        forward_state_dict = torch.load(f"{dir}/initial_forward_model.pth", map_location=device)
        self.forward_model.load_state_dict(forward_state_dict)
        self.initial_forward_model.load_state_dict(forward_state_dict)

        if isinstance(self, BidirectionalStrategy):
            backward_state_dict = torch.load(f"{dir}/initial_backward_model.pth", map_location=device)
            self.backward_model.load_state_dict(backward_state_dict)
            self.initial_backward_model.load_state_dict(backward_state_dict)

        mu_state_dict = torch.load(f"{dir}/initial_mu_model.pth", map_location=device)
        nu_state_dict = torch.load(f"{dir}/initial_nu_model.pth", map_location=device)
        self.mu_bridge.load_state_dict(mu_state_dict)
        self.initial_mu_bridge.load_state_dict(mu_state_dict)
        self.nu_bridge.load_state_dict(nu_state_dict)
        self.initial_nu_bridge.load_state_dict(nu_state_dict)
        logger.info("Pretrained models loaded.")

    def _save_pretrained(self, save_dir: str):
        """Save all pretrained models (flow + marginal bridges) to disk."""
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.initial_forward_model.state_dict(), f"{save_dir}/initial_forward_model.pth")
        if isinstance(self, BidirectionalStrategy):
            torch.save(self.initial_backward_model.state_dict(), f"{save_dir}/initial_backward_model.pth")
        torch.save(self.initial_mu_bridge.state_dict(), f"{save_dir}/initial_mu_model.pth")
        torch.save(self.initial_nu_bridge.state_dict(), f"{save_dir}/initial_nu_model.pth")
        logger.info("Pretrained models saved (flow + marginal bridges).")

    def _save_final(self, save_dir: str) -> None:
        """Save the fully-trained models (after all iterations) to *save_dir*.

        Files written
        -------------
        forward_model.pth       – final forward CFM model
        backward_model.pth      – final backward CFM model (BidirectionalStrategy only)
        mu_bridge.pth           – source marginal bridge
        nu_bridge.pth           – target marginal bridge
        f_dual.pth              – dual potential f
        g_dual.pth              – dual potential g
        h_dual.pth              – dual mismatch potential h
        """
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.forward_model.state_dict(), f"{save_dir}/forward_model.pth")
        if isinstance(self, BidirectionalStrategy):
            torch.save(self.backward_model.state_dict(), f"{save_dir}/backward_model.pth")
        torch.save(self.mu_bridge.state_dict(),  f"{save_dir}/mu_bridge.pth")
        torch.save(self.nu_bridge.state_dict(),  f"{save_dir}/nu_bridge.pth")
        torch.save(self.f_dual.state_dict(),     f"{save_dir}/f_dual.pth")
        torch.save(self.g_dual.state_dict(),     f"{save_dir}/g_dual.pth")
        torch.save(self.h_dual.state_dict(),     f"{save_dir}/h_dual.pth")
        logger.info("Final trained models saved -> %s", save_dir)


class BidirectionalStrategy(Strategy):
    def __init__(self, wrapper, config: FeedForwardConfig, change_of_measure: Optional[nn.Module] = None):
        super().__init__(wrapper, config, change_of_measure)

    def _create_models(self, dim: int, hidden_size: int, num_layers: int = 2):
        super()._create_models(dim, hidden_size, num_layers)
        wrapper = self.wrapper
        self.backward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, n_layers=num_layers, time_varying=True))
        self.previous_backward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, n_layers=num_layers, time_varying=True))
        self.initial_backward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, n_layers=num_layers, time_varying=True))

    def _model_attrs(self):
        return super()._model_attrs() + [
            "backward_model", "previous_backward_model", "initial_backward_model",
        ]

    def _create_optimizers(self, config: FeedForwardConfig):
        super()._create_optimizers(config)
        self.backward_optimizer = optim.Adam(self.backward_model.parameters(), lr=config.flow_learning_rate, weight_decay=config.dual_l2_reg)

    def get_flow(self, step: int, device):
        self.to_device(device)
        model = self.forward_model if step % 2 == 0 else self.backward_model
        optimizer = self.forward_optimizer if step % 2 == 0 else self.backward_optimizer
        return model, optimizer

    def get_previous_flow(self, step: int, device):
        self.to_device(device)
        previous_model = self.previous_forward_model if step % 2 == 0 else self.previous_backward_model
        return previous_model

    def get_initial_flow(self, step: int, device):
        self.to_device(device)
        initial_model = self.initial_forward_model if step % 2 == 0 else self.initial_backward_model
        return initial_model

    def advance(
        self, step: int, x_mu_next: torch.Tensor, y_nu_next: torch.Tensor, num_steps: int = 100, sigma: float = 0
    ):
        if step % 2 == 0:
            return super().advance(step, x_mu_next, y_nu_next, num_steps, sigma)
        else:
            y_next, x_next = super().advance(step, y_nu_next, x_mu_next, num_steps, sigma)
            return x_next, y_next
