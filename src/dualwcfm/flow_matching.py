import torch
import torch.nn as nn

from dualwcfm.utils import is_image_tensor


class ConditionalFlowMatching:
    def vector_field(
        self, x: torch.Tensor, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, extend: bool = True
    ) -> torch.Tensor:
        """Compute the vector field at position x and time t.

        Handles both vector (N, D) and image (N, C, H, W) tensors.
        """
        if extend:
            # For images, concatenate along channel dimension (dim=1)
            return torch.cat([x1 - x0, torch.zeros_like(x0)], dim=1)
        else:
            return x1 - x0

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma: float) -> torch.Tensor:
        """Interpolate between x0 and x1 at time t with noise sigma.

        Handles both vector (N, D) and image (N, C, H, W) tensors.
        """
        # Reshape t for proper broadcasting with images
        if is_image_tensor(x0):
            # t is (N, 1), reshape to (N, 1, 1, 1) for image broadcasting
            while t.dim() < x0.dim():
                t = t.unsqueeze(-1)

        return (1 - t) * x0 + t * x1 + sigma * torch.randn_like(x0) * torch.sqrt(t * (1 - t))

    def evolve(
        self, x_init: torch.Tensor, model: nn.Module, num_steps: int = 100, sigma: float = 0.0, extend: bool = True
    ) -> torch.Tensor:
        """Evolve the points x_init over time t using Euler integration.

        Handles both vector (N, D) and image (N, C, H, W) tensors.
        """
        device = x_init.device
        dt = 1.0 / num_steps

        with torch.no_grad():
            if extend:
                xt = torch.cat([x_init, x_init], dim=1)
            else:
                xt = x_init.clone()

            for i in range(num_steps):
                t = torch.ones(xt.shape[0], 1, device=device) * (i * dt)
                vt = model(t, xt)
                xt = xt + vt * dt

        model = model.to(device)
        xt = xt.to(device)

        return xt
