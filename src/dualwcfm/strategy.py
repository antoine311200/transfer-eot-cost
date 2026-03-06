import torch
import torch.nn as nn
import torch.optim as optim

from dualwcfm.config import FeedForwardConfig
from dualwcfm.utils import is_image_tensor
from dualwcfm.mlp import MLP


class Strategy:
    """Base strategy for training transport models.

    This class can be subclassed to customize model architectures.
    For custom models (e.g., UNets for images), override _create_models().
    """

    def __init__(self, wrapper, config: FeedForwardConfig):
        self.wrapper = wrapper
        self.config = config
        self._device = None

        dim = config.dimension
        hidden_size = config.hidden_size

        self._create_models(dim, hidden_size)
        self._create_optimizers(config)

    def _create_models(self, dim: int, hidden_size: int):
        """Create all models. Override this method for custom architectures."""
        wrapper = self.wrapper
        self.mu_bridge = wrapper(MLP(dim=dim, out_dim=dim, w=hidden_size, time_varying=True))
        self.nu_bridge = wrapper(MLP(dim=dim, out_dim=dim, w=hidden_size, time_varying=True))
        self.f_dual = wrapper(MLP(dim=dim, out_dim=1, w=hidden_size, time_varying=False))
        self.g_dual = wrapper(MLP(dim=dim, out_dim=1, w=hidden_size, time_varying=False))
        self.h_dual = wrapper(MLP(dim=2*dim, out_dim=1, w=hidden_size, time_varying=False))
        self.forward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, time_varying=True))
        self.previous_forward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, time_varying=True))

        # Save initial state
        self.initial_mu_bridge = wrapper(MLP(dim=dim, out_dim=dim, w=hidden_size, time_varying=True))
        self.initial_nu_bridge = wrapper(MLP(dim=dim, out_dim=dim, w=hidden_size, time_varying=True))
        self.initial_forward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, time_varying=True))

    def to_device(self, device: torch.device):
        """Move all models to device once. Call this before training."""
        if self._device == device:
            return
        self._device = device
        for attr in self._model_attrs():
            getattr(self, attr).to(device)

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
        self.mu_optimizer = optim.Adam(self.mu_bridge.parameters(), lr=config.mu_learning_rate)
        self.nu_optimizer = optim.Adam(self.nu_bridge.parameters(), lr=config.nu_learning_rate)
        self.dual_optimizer = optim.Adam(dual_params, lr=config.dual_learning_rate)
        self.dual_mismatch_optimizer = optim.Adam(self.h_dual.parameters(), lr=config.dual_learning_rate)
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=config.flow_learning_rate)

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


class BidirectionalStrategy(Strategy):
    def __init__(self, wrapper, config: FeedForwardConfig):
        super().__init__(wrapper, config)

    def _create_models(self, dim: int, hidden_size: int):
        super()._create_models(dim, hidden_size)
        wrapper = self.wrapper
        self.backward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, time_varying=True))
        self.previous_backward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, time_varying=True))
        self.initial_backward_model = wrapper(MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, time_varying=True))

    def _model_attrs(self):
        return super()._model_attrs() + [
            "backward_model", "previous_backward_model", "initial_backward_model",
        ]

    def _create_optimizers(self, config: FeedForwardConfig):
        super()._create_optimizers(config)
        self.backward_optimizer = optim.Adam(self.backward_model.parameters(), lr=config.flow_learning_rate)

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
