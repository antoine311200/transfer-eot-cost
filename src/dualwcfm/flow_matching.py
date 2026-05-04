import torch
import torch.nn as nn

from dualwcfm.utils import is_image_tensor


class ConditionalFlowMatching:

    def random_time(self, batch_size, device, eps=1e-3):
        s = torch.rand(batch_size, 1, device=device)
        s = s * (1 - 2 * eps) + eps
        return s

    def vector_field(
        self, x: torch.Tensor, t, x0: torch.Tensor, x1: torch.Tensor, extend: bool = True, noisy: bool = False#, noise: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute the vector field at position x and time t.

        Handles both vector (N, D) and image (N, C, H, W) tensors.
        Note: x and t are unused in this linear flow -- kept for API compatibility.
        """
        if noisy:
            # Reshape t for proper broadcasting with images: (N, 1) -> (N, 1, 1, 1)
            if is_image_tensor(x0):
                t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
            field = x1 - x0 + (1 - 2*t) / (2 * t * (1 - t)) * (x - (t * x1 + (1 - t) * x0))
        else:
            field = x1 - x0
        if extend:
            return torch.cat([field, torch.zeros_like(x0)], dim=1)
        else:
            return field

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma: float) -> torch.Tensor:
        """Interpolate between x0 and x1 at time t with noise sigma.

        Handles both vector (N, D) and image (N, C, H, W) tensors.
        """
        # Reshape t for proper broadcasting with images: (N, 1) -> (N, 1, 1, 1)
        if is_image_tensor(x0):
            t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))

        if sigma == 0.0:
            return (1 - t) * x0 + t * x1

        noise = sigma * torch.randn_like(x0)

        return (1 - t) * x0 + t * x1 + noise * torch.sqrt(t * (1 - t))

    def interpolate_from_correction(self, x1: torch.Tensor, y_corr: torch.Tensor, t_corr: torch.Tensor, t: torch.Tensor, model: nn.Module, sigma: float, config) -> torch.Tensor:
        x_corr = self.evolve(y_corr, model, num_steps=25, sigma=sigma, extend=True)[..., :x1.shape[1]]
        t_shrink = (t - t_corr) / (1 - t_corr) if config.correction_each > 0 else t
        return self.interpolate(x_corr, x1, t_shrink, sigma)

    def evolve(
        self, x_init: torch.Tensor, model: nn.Module, num_steps: int = 100, sigma: float = 0.0, extend: bool = True, start_index: int = 0, start_time: float = 0.0, verbose: bool = False,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """Evolve the points x_init over time t using Euler integration.

        Handles both vector (N, D) and image (N, C, H, W) tensors.

        When *condition* is provided the model is called as model(t, xt, condition)
        with the condition tensor held fixed throughout integration (e.g. the t=0
        starting point for a conditional bridge model).
        """
        device = x_init.device
        dt = 1.0 / num_steps

        with torch.no_grad():
            if extend:
                xt = torch.cat([x_init, x_init], dim=1)
            else:
                xt = x_init.clone()

            for i in range(num_steps - start_index):
                t = torch.full((xt.shape[0], 1), i * dt + start_time, device=device)
                if condition is not None:
                    vt = model(t, xt, condition)
                else:
                    vt = model(t, xt)

                if extend:
                    xt = torch.cat([xt[:, :x_init.shape[1]] + vt[:, :x_init.shape[1]] * dt, xt[:, x_init.shape[1]:]], dim=1)
                else:
                    xt = xt + vt * dt
                # Set xt to be xt+vt*dt on the first half of the dimensions, and keep the second half x_init for stability:
            if verbose:
                print(i * dt + start_time)
        return xt

    def evolve_trajectory(
        self, x_init: torch.Tensor, model: nn.Module, num_steps: int = 100, sigma: float = 0.0, extend: bool = True, start_index: int = 0, start_time: float = 0.0,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """Like evolve, but returns all intermediate states as (num_steps+1, N, D).

        When extend=True the full doubled tensor is stored; slice [:, :, :D] to get
        only the evolved half.

        When *condition* is provided the model is called as model(t, xt, condition)
        with the condition tensor held fixed throughout (e.g. x0 for conditional bridges).
        """
        device = x_init.device
        dt = 1.0 / num_steps

        with torch.no_grad():
            if extend:
                xt = torch.cat([x_init, x_init], dim=1)
            else:
                xt = x_init.clone()

            states = [xt.clone()]
            for i in range(num_steps - start_index):
                t = torch.full((xt.shape[0], 1), i * dt + start_time, device=device)
                if condition is not None:
                    vt = model(t, xt, condition)
                else:
                    vt = model(t, xt)

                if extend:
                    xt = torch.cat([xt[:, :x_init.shape[1]] + vt[:, :x_init.shape[1]] * dt, xt[:, x_init.shape[1]:]], dim=1)
                else:
                    xt = xt + vt * dt
                states.append(xt.clone())

        return torch.stack(states, dim=0)  # (num_steps+1, N, D)

    def evolve_to_time(
        self, x_init: torch.Tensor, model: nn.Module, t_end: float,
        n_steps: int = 50, extend: bool = False,
    ) -> torch.Tensor:
        """Euler-integrate a bridge ODE from t=0 to t=t_end (not t=1).

        Unlike evolve(), dt = t_end / n_steps so integration stops at t_end.
        When extend=True the frozen-condition convention is used (same as evolve):
        state is doubled, only first half updated, second half stays frozen.
        Returns the full doubled tensor when extend=True.
        """
        if t_end <= 0.0:
            if extend:
                return torch.cat([x_init, x_init], dim=1)
            return x_init.clone()
        n_steps = max(1, n_steps)
        dt = t_end / n_steps
        device = x_init.device
        with torch.no_grad():
            if extend:
                xt = torch.cat([x_init, x_init], dim=1)
            else:
                xt = x_init.clone()
            for i in range(n_steps):
                t = torch.full((x_init.shape[0], 1), i * dt, device=device)
                vt = model(t, xt)
                if extend:
                    xt = torch.cat([
                        xt[:, :x_init.shape[1]] + vt[:, :x_init.shape[1]] * dt,
                        xt[:, x_init.shape[1]:],
                    ], dim=1)
                else:
                    xt = xt + vt * dt
        return xt
