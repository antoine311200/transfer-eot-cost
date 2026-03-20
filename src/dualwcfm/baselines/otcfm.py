"""dualwcfm.baselines.otcfm
===========================
OT-CFM (Optimal Transport Conditional Flow Matching) trained from scratch on
(mu', nu') — the primary "from-scratch" comparison baseline.

The model learns a velocity field v_theta(t, x) using the conditional flow
matching loss with minibatch OT pairing implemented via a pure-PyTorch
log-domain Sinkhorn (no POT / external OT library required).

References
----------
Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
Tong et al., "Improving and Generalizing Flow-Matching", NeurIPS 2023 (OT-CFM).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from dualwcfm.baselines.base import BaselineRunner, BaselineSpec

logger = logging.getLogger("dualwcfm")


# ---------------------------------------------------------------------------
# Pure-PyTorch minibatch Sinkhorn (no POT dependency)
# ---------------------------------------------------------------------------

def _sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    eps: float = 0.05,
    n_iter: int = 50,
) -> torch.Tensor:
    """Log-domain Sinkhorn; returns transport plan P of shape (N, M)."""
    log_a = a.log()
    log_b = b.log()
    log_K = -C / eps          # (N, M)
    u = torch.zeros_like(a)   # (N,)
    for _ in range(n_iter):
        v = log_b - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)
        u = log_a - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
    return (log_K + u.unsqueeze(1) + v.unsqueeze(0)).exp()


def _ot_pair(
    x0: torch.Tensor,
    x1: torch.Tensor,
    eps: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return OT-paired (x0', x1') sampled proportionally to the Sinkhorn plan."""
    N, M = len(x0), len(x1)
    a = torch.full((N,), 1.0 / N, device=x0.device)
    b = torch.full((M,), 1.0 / M, device=x1.device)
    with torch.no_grad():
        C = torch.cdist(x0.flatten(1), x1.flatten(1)).pow(2)
        P = _sinkhorn(a, b, C, eps=eps)
    idx = torch.multinomial(P.reshape(-1), num_samples=N, replacement=True)
    return x0[idx // M], x1[idx % M]


# ---------------------------------------------------------------------------
# MLP velocity field  v_theta(t, x) -> R^d
# ---------------------------------------------------------------------------

class _VelocityMLP(nn.Module):
    """Time-conditioned MLP velocity field for low-dimensional data."""

    def __init__(self, dim: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(dim + 1, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_in = t.view(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([t_in, x.flatten(1)], dim=1))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OTCFMConfig:
    """Hyperparameters for the OT-CFM baseline."""
    n_steps:    int   = 10_000  # gradient steps
    batch_size: int   = 512
    lr:         float = 1e-3
    hidden:     int   = 256
    n_layers:   int   = 4
    ot_eps:     float = 0.05    # Sinkhorn epsilon; None = random pairing (plain CFM)
    sigma:      float = 0.01    # noise added to interpolation midpoints
    euler_steps: int  = 100     # integration steps at inference


# ---------------------------------------------------------------------------
# OTCFMRunner
# ---------------------------------------------------------------------------

class OTCFMRunner(BaselineRunner):
    """OT-CFM trained from scratch on (mu', nu') — no reference coupling used.

    This is the primary quantitative baseline for 2-D / 3-D / Gushchin tables.
    Set ``config.ot_eps = None`` to disable OT pairing and use plain random-pair
    CFM instead (useful as a secondary ablation).

    Parameters
    ----------
    spec:
        Experiment specification (source/target distributions, device, …).
    config:
        Training hyperparameters.  ``None`` uses sensible defaults.
    wandb_run:
        Optional W&B run for per-step loss logging.
    """

    def __init__(
        self,
        spec:     BaselineSpec,
        config:   Optional[OTCFMConfig] = None,
        wandb_run=None,
    ) -> None:
        super().__init__(spec, wandb_run)
        self.config  = config or OTCFMConfig()
        self._model: Optional[_VelocityMLP] = None

    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg    = self.config
        device = self.spec.device
        dim    = self.spec.dim

        model = _VelocityMLP(dim, hidden=cfg.hidden, n_layers=cfg.n_layers).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_steps, eta_min=cfg.lr * 0.1)

        log_every = max(1, cfg.n_steps // 20)
        model.train()

        for step in range(cfg.n_steps):
            x0 = _to_tensor(self.spec.source.sample(cfg.batch_size), device)
            x1 = _to_tensor(self.spec.target.sample(cfg.batch_size), device)

            # Minibatch OT pairing (or plain random pairing)
            if cfg.ot_eps is not None:
                x0, x1 = _ot_pair(x0, x1, eps=cfg.ot_eps)

            # Sample time, interpolate, add optional noise
            t  = torch.rand(len(x0), device=device)
            xt = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1
            if cfg.sigma > 0:
                xt = xt + cfg.sigma * torch.randn_like(xt)

            # CFM objective: regress conditional vector field (x1 - x0)
            loss = (model(t, xt) - (x1 - x0)).pow(2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            if (step + 1) % log_every == 0:
                logger.info(
                    "  [OT-CFM %s] step %5d/%d  loss=%.5f",
                    self.spec.name, step + 1, cfg.n_steps, loss.item(),
                )
                self._wandb.log({"otcfm/loss": loss.item()}, step=step)

        # Save weights
        torch.save(model.state_dict(), os.path.join(self.spec.output_dir, "otcfm_model.pth"))
        self._model = model
        logger.info("  [OT-CFM] Training complete -> %s", self.spec.output_dir)

    # ------------------------------------------------------------------

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._model is not None, "Call train() first."
        model  = self._model.eval()
        device = self.spec.device

        x0 = _to_tensor(self.spec.source.sample(n), device)
        x  = x0.clone()
        dt = 1.0 / self.config.euler_steps

        with torch.no_grad():
            for i in range(self.config.euler_steps):
                t = torch.full((n,), i * dt, device=device)
                x = x + model(t, x) * dt

        return x0, x


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_tensor(x, device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    return x.float().to(device)
