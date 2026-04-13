"""dualwcfm.baselines.lightsb
==============================
LightSB baseline — "Light and Optimal Schrödinger Bridge Matching".

Algorithm (Korotin et al., ICML 2023)
--------------------------------------
LightSB parametrises the Schrödinger Bridge solution via dual potentials
f(x), g(y) and trains them with the entropic OT dual objective.  Generation
proceeds via the stochastic interpolant: given an (x, y) pair drawn from the
learned SB plan, the bridge X_t = (1-t)x + ty + σ√(t(1-t))ξ is used to
produce the output sample.

Step 1 — Dual-potential training
  Maximise the Sinkhorn dual:
    L(f, g) = E_{x~μ}[f(x)] + E_{y~ν}[g(y)]
              − ε · E_{x~μ, y~ν}[exp((f(x) + g(y) − ‖x−y‖²/ε))]

  Optimised with alternating gradient ascent on f and g.

Step 2 — SB-plan sampling
  Given a batch (x_i, y_j), the plan weight is
    w(x_i, y_j) = exp((f(x_i) + g(y_j) − ‖x_i−y_j‖²) / ε)
  Rows of the weight matrix are normalised to form conditional distributions
  p(y | x_i).  We resample y_i ~ p(· | x_i) to obtain coupled pairs (x_i, ŷ_i).

Step 3 — Stochastic interpolant flow
  Train a velocity field v_θ(t, x) on the resampled pairs using the standard
  CFM loss:
    L_CFM = E_{(x,y)~plan, t~U[0,1]}[‖v_θ(t, X_t) − (y − x)‖²]
  where X_t = (1-t)x + ty.

At inference: Euler integration of v_θ from x₀ ~ μ.

References
----------
Korotin et al., "Light and Optimal Schrödinger Bridge Matching", ICML 2023.
De Bortoli et al., "Diffusion Schrödinger Bridge", NeurIPS 2021.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from dualwcfm.baselines.base import BaselineRunner, BaselineSpec

logger = logging.getLogger("dualwcfm")


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class _PotentialMLP(nn.Module):
    """Scalar potential network  h_θ(x) → ℝ."""

    def __init__(self, dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).squeeze(-1)   # (N,)


class _VelocityMLP(nn.Module):
    """Time-conditioned velocity field  v_θ(t, x) → ℝ^d."""

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
class LightSBConfig:
    """Hyperparameters for the LightSB baseline."""
    eps:               float = 0.1     # entropic regularisation ε
    # Potential training
    n_dual_steps:      int   = 10_000  # gradient steps for (f, g) training
    dual_batch:        int   = 512     # batch size for potential training
    dual_lr:           float = 1e-3
    potential_hidden:  int   = 256
    potential_layers:  int   = 3
    # CFM on resampled plan
    n_cfm_steps:       int   = 10_000
    cfm_batch:         int   = 512
    cfm_lr:            float = 1e-3
    cfm_hidden:        int   = 256
    cfm_layers:        int   = 4
    # Plan sampling buffer
    plan_buffer_size:  int   = 4_000   # number of (x, y) plan pairs to maintain
    plan_refresh_every: int  = 2_000   # re-sample the plan buffer every N CFM steps
    sigma_bridge:      float = 0.0     # noise on the interpolant; 0 = ODE
    # Inference
    euler_steps:       int   = 100


# ---------------------------------------------------------------------------
# LightSBRunner
# ---------------------------------------------------------------------------

class LightSBRunner(BaselineRunner):
    """LightSB trained from scratch on (mu', nu') — Korotin et al., ICML 2023.

    Two-stage pipeline:
      1. Train Kantorovich potentials f, g via the entropic OT dual.
      2. Resample an SB-plan buffer, then train a CFM velocity field on it.
    """

    def __init__(
        self,
        spec:     BaselineSpec,
        config:   Optional[LightSBConfig] = None,
        wandb_run=None,
    ) -> None:
        super().__init__(spec, wandb_run)
        self.config  = config or LightSBConfig()
        self._flow:  Optional[_VelocityMLP]  = None

    # ------------------------------------------------------------------
    # Stage 1 — dual-potential training
    # ------------------------------------------------------------------

    def _train_potentials(self) -> Tuple[_PotentialMLP, _PotentialMLP]:
        """Train f(x) and g(y) via the Sinkhorn dual loss.

        Returns the trained (f, g) pair.
        """
        cfg    = self.config
        device = self.spec.device
        dim    = self.spec.dim
        eps    = cfg.eps

        f = _PotentialMLP(dim, cfg.potential_hidden, cfg.potential_layers).to(device)
        g = _PotentialMLP(dim, cfg.potential_hidden, cfg.potential_layers).to(device)

        opt_f = torch.optim.Adam(f.parameters(), lr=cfg.dual_lr)
        opt_g = torch.optim.Adam(g.parameters(), lr=cfg.dual_lr)
        log_every = max(1, cfg.n_dual_steps // 20)

        for step in range(cfg.n_dual_steps):
            x = _to_tensor(self.spec.source.sample(cfg.dual_batch), device)
            y = _to_tensor(self.spec.target.sample(cfg.dual_batch), device)

            fx = f(x)   # (N,)
            gy = g(y)   # (M,)

            # Cost matrix C_ij = ‖x_i − y_j‖²
            C = torch.cdist(x.flatten(1), y.flatten(1)).pow(2)  # (N, M)

            # Sinkhorn dual: maximise E[f] + E[g] - ε·mean(exp((f+g-C)/ε))
            log_W = (fx.unsqueeze(1) + gy.unsqueeze(0) - C) / eps  # (N, M)

            # Numerically stable mean(exp(log_W)) via logsumexp:
            #   mean(exp(log_W)) = exp(logsumexp(log_W.flat) - log(N*M))
            # NOTE: do NOT just subtract max without restoring exp(max) —
            # that silently drops the penalty to ~0 and causes divergence.
            log_n = math.log(log_W.numel())
            exp_term = eps * (torch.logsumexp(log_W.reshape(-1), dim=0) - log_n).exp()

            # Negate because we want to *maximise* the dual
            loss = -(fx.mean() + gy.mean() - exp_term)

            opt_f.zero_grad()
            opt_g.zero_grad()
            loss.backward()
            # Clip to guard against any remaining large gradients
            torch.nn.utils.clip_grad_norm_(
                list(f.parameters()) + list(g.parameters()), 5.0
            )
            opt_f.step()
            opt_g.step()

            if (step + 1) % log_every == 0:
                logger.info(
                    "  [LightSB dual %s] step %5d/%d  loss=%.5f",
                    self.spec.name, step + 1, cfg.n_dual_steps, -loss.item(),
                )
                self._wandb.log({"lightsb/dual_loss": -loss.item()})

        return f, g

    # ------------------------------------------------------------------
    # Stage 2 — sample SB plan
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_plan(
        self,
        f: _PotentialMLP,
        g: _PotentialMLP,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw (x, y) pairs from the SB plan using importance resampling.

        For each x_i:
          p(y_j | x_i) ∝ exp((f(x_i) + g(y_j) − ‖x_i − y_j‖²) / ε)
        We normalise per row and draw one y per x.

        Returns (x_plan, y_plan) each of shape (plan_buffer_size, d).
        """
        cfg    = self.config
        device = self.spec.device
        N      = cfg.plan_buffer_size
        eps    = cfg.eps

        x = _to_tensor(self.spec.source.sample(N), device)
        y = _to_tensor(self.spec.target.sample(N), device)

        fx = f(x)    # (N,)
        gy = g(y)    # (N,)
        C  = torch.cdist(x.flatten(1), y.flatten(1)).pow(2)   # (N, N)

        log_w = (fx.unsqueeze(1) + gy.unsqueeze(0) - C) / eps  # (N, N)
        # Row-wise softmax: numerically stable; handles -inf rows gracefully
        p = torch.softmax(log_w, dim=1)   # (N, N)
        # Guard: if any row collapsed to NaN (all-inf costs), replace with uniform
        bad = p.isnan().any(dim=1)
        if bad.any():
            p[bad] = 1.0 / N

        # Sample one y per x according to p(· | x_i)
        y_idx = torch.multinomial(p, num_samples=1).squeeze(1)  # (N,)
        return x, y[y_idx]

    # ------------------------------------------------------------------
    # Stage 3 — CFM velocity field on resampled pairs
    # ------------------------------------------------------------------

    def _train_cfm(
        self,
        f: _PotentialMLP,
        g: _PotentialMLP,
    ) -> _VelocityMLP:
        """Train CFM on the SB plan, periodically refreshing the plan buffer."""
        cfg    = self.config
        device = self.spec.device
        dim    = self.spec.dim

        flow = _VelocityMLP(dim, cfg.cfm_hidden, cfg.cfm_layers).to(device)
        opt  = torch.optim.Adam(flow.parameters(), lr=cfg.cfm_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.n_cfm_steps, eta_min=cfg.cfm_lr * 0.1
        )
        log_every = max(1, cfg.n_cfm_steps // 20)

        # Initial plan buffer
        x_plan, y_plan = self._sample_plan(f, g)
        N = len(x_plan)

        for step in range(cfg.n_cfm_steps):
            # Periodically refresh the plan buffer to avoid overfitting
            if step > 0 and cfg.plan_refresh_every > 0 and step % cfg.plan_refresh_every == 0:
                x_plan, y_plan = self._sample_plan(f, g)
                N = len(x_plan)

            idx = torch.randint(N, (cfg.cfm_batch,), device=device)
            x0b = x_plan[idx]
            x1b = y_plan[idx]

            t  = torch.rand(len(x0b), device=device)
            xt = (1 - t.view(-1, 1)) * x0b + t.view(-1, 1) * x1b
            if cfg.sigma_bridge > 0:
                xt = xt + cfg.sigma_bridge * torch.randn_like(xt)

            # CFM regression target: conditional vector field x1 - x0
            loss = (flow(t, xt) - (x1b - x0b)).pow(2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            if (step + 1) % log_every == 0:
                logger.info(
                    "  [LightSB CFM %s] step %5d/%d  loss=%.5f",
                    self.spec.name, step + 1, cfg.n_cfm_steps, loss.item(),
                )
                self._wandb.log({"lightsb/cfm_loss": loss.item()})

        return flow

    # ------------------------------------------------------------------
    # BaselineRunner interface
    # ------------------------------------------------------------------

    def train(self) -> None:
        logger.info("  [LightSB] Stage 1: training dual potentials …")
        f, g = self._train_potentials()

        # Log plan quality after potential training
        with torch.no_grad():
            x_probe, y_probe = self._sample_plan(f, g)
        logger.info(
            "  [LightSB] Plan sampled: %d pairs,  E[‖x−y‖²]=%.4f",
            len(x_probe),
            (x_probe - y_probe).pow(2).sum(-1).mean().item(),
        )

        logger.info("  [LightSB] Stage 3: training CFM on plan (with periodic refresh) …")
        flow = self._train_cfm(f, g)

        # Save artefacts
        torch.save(f.state_dict(),    os.path.join(self.spec.output_dir, "lightsb_f.pth"))
        torch.save(g.state_dict(),    os.path.join(self.spec.output_dir, "lightsb_g.pth"))
        torch.save(flow.state_dict(), os.path.join(self.spec.output_dir, "lightsb_flow.pth"))
        self._flow = flow
        logger.info("  [LightSB] Training complete -> %s", self.spec.output_dir)

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._flow is not None, "Call train() first."
        flow   = self._flow.eval()
        device = self.spec.device
        cfg    = self.config
        dt     = 1.0 / cfg.euler_steps

        x0 = _to_tensor(self.spec.source.sample(n), device)
        x  = x0.clone()

        with torch.no_grad():
            for i in range(cfg.euler_steps):
                t = torch.full((n,), i * dt, device=device)
                x = x + flow(t, x) * dt

        return x0, x


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_tensor(x, device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    return x.float().to(device)
