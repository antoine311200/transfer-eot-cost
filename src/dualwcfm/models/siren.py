"""SIREN (Sinusoidal Representation Network) MLP.

Sitzmann et al. 2020 – "Implicit Neural Representations with Periodic Activation Functions".
Replaces SELU activations with sin(ω₀·Wx+b) to combat spectral bias and faithfully
represent high-frequency functions – crucial for shear, radial-warp, and vortex transports.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SirenLayer(nn.Module):
    """Single SIREN layer: Linear followed by sin(ω₀ · x)."""

    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Sitzmann et al.: first-layer weights uniform in [-1/fan_in, 1/fan_in]
                bound = 1.0 / self.in_features
            else:
                # Hidden layers: uniform in [-√(6/fan_in)/ω₀, √(6/fan_in)/ω₀]
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenMLP(nn.Module):
    """SIREN network with the same interface as :class:`~dualwcfm.mlp.MLP`.

    Drop-in replacement that uses sinusoidal activations throughout the hidden
    layers.  The final projection is a plain linear layer (no activation) with
    small-gain init to avoid wild trajectories early in training.

    Parameters
    ----------
    dim:
        Input feature dimension.  When *time_varying* is ``True`` the actual
        first-layer width is ``dim + 1`` (time is concatenated by the caller,
        e.g. :class:`~dualwcfm.mlp.torch_wrapper`).
    out_dim:
        Output dimension (defaults to *dim*).
    w:
        Width of every hidden SIREN layer.
    n_layers:
        Number of **hidden** SIREN layers (≥ 1).
    time_varying:
        If ``True``, expects a ``dim + 1`` wide input (time already appended).
    omega_0:
        Angular frequency for the **first** layer (controls the initial
        frequency spectrum seen by the network).
    hidden_omega:
        Angular frequency for all **subsequent** hidden layers.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        w: int = 64,
        n_layers: int = 2,
        time_varying: bool = False,
        omega_0: float = 30.0,
        hidden_omega: float = 30.0,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = dim

        in_dim = dim + (1 if time_varying else 0)

        layers: list[nn.Module] = []
        layers.append(SirenLayer(in_dim, w, omega_0=omega_0, is_first=True))
        for _ in range(n_layers - 1):
            layers.append(SirenLayer(w, w, omega_0=hidden_omega, is_first=False))
        self.net = nn.Sequential(*layers)

        # Final linear projection – no sinusoidal activation, small-gain init
        # to prevent wild flow trajectories at the start of training.
        self.final = nn.Linear(w, out_dim)
        nn.init.xavier_uniform_(self.final.weight, gain=0.01)
        nn.init.zeros_(self.final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.net(x))
