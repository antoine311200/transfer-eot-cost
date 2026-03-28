"""dualwcfm.baselines.dsb
========================
Diffusion Schrödinger Bridge baseline via IPF (Iterative Proportional Fitting).

Algorithm (De Bortoli et al., NeurIPS 2021 — "Diffusion Schrödinger Bridge")
-----------------------------------------------------------------------------
The Schrödinger Bridge between μ and ν is the path measure that:
  - Matches marginals π_0 = μ and π_1 = ν
  - Is closest in KL divergence to a reference Wiener measure W_σ

IPF solves this by alternating half-bridge problems:

  Iteration k (forward half):
    - Simulate trajectories {X_t}_{t∈[0,1]} starting from X_0 ~ μ
      using the current *backward* drift b^k
    - Train *forward* drift f^{k+1} by regressing on the conditional
      vector field pointing toward X_1:
        L_f = E[‖ f(t, X_t) − (X_1 − X_t)/(1−t) ‖²]

  Iteration k (backward half):
    - Simulate trajectories {X_t}_{t∈[0,1]} ending at X_1 ~ ν
      using the current *forward* drift f^{k+1}
    - Train *backward* drift b^{k+1} by regressing on the conditional
      vector field pointing toward X_0:
        L_b = E[‖ b(t, X_t) − (X_0 − X_t)/t ‖²]

At inference: Euler-Maruyama from X_0 ~ μ using the final forward drift f.

Complexity note
---------------
For the paper comparison, we use σ = 0 (deterministic bridges) which turns
the SDE into a regularised ODE — equivalent to CFM with IPF-coupled pairs.
Setting sigma > 0 recovers the full stochastic SB formulation.

References
----------
De Bortoli et al., "Diffusion Schrödinger Bridge", NeurIPS 2021.
Shi et al., "Diffusion Schrödinger Bridge Matching", NeurIPS 2023.
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
# MLP drift network  u_θ(t, x) → ℝ^d
# ---------------------------------------------------------------------------

class _DriftMLP(nn.Module):
    """Time-conditioned MLP used for both forward and backward drifts."""

    def __init__(self, dim: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(dim + 1, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_in = t.view(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([t_in, x], dim=1))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DSBConfig:
    """Hyperparameters for the DSB-IPF baseline."""
    n_ipf_iters:     int   = 10      # number of full IPF iterations (fwd + bwd each)
    n_steps_per_half: int  = 2_000   # gradient steps per half-bridge training
    batch_size:      int   = 512
    lr:              float = 1e-3
    hidden:          int   = 256
    n_layers:        int   = 4
    sigma:           float = 0.0     # SDE noise; 0 = deterministic ODE bridge
    n_sim_steps:     int   = 50      # Euler steps when simulating trajectories
    n_sim_samples:   int   = 2_000   # samples kept in the replay buffer per half
    euler_steps:     int   = 100     # steps at inference


# ---------------------------------------------------------------------------
# DSBRunner
# ---------------------------------------------------------------------------

class DSBRunner(BaselineRunner):
    """Diffusion Schrödinger Bridge via IPF (De Bortoli et al., NeurIPS 2021).

    Learns forward and backward drift networks by alternating half-bridge
    regressions.  No external SDE library required — all simulation is done
    with plain Euler-Maruyama in PyTorch.

    Parameters
    ----------
    spec:
        Baseline specification (source/target distributions, device, …).
    config:
        IPF hyperparameters.
    wandb_run:
        Optional W&B run for loss logging.
    """

    def __init__(
        self,
        spec:     BaselineSpec,
        config:   Optional[DSBConfig] = None,
        wandb_run=None,
    ) -> None:
        super().__init__(spec, wandb_run)
        self.config  = config or DSBConfig()
        self._fwd_model: Optional[_DriftMLP] = None

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _simulate_forward(
        self,
        x0: torch.Tensor,
        fwd: _DriftMLP,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate SDE forward from x0 using fwd drift.

        Returns (X_t, t, X_0_broadcast) all of shape (N * n_steps, d).
        Used to build the backward regression target.
        """
        cfg    = self.config
        device = self.spec.device
        N, d   = x0.shape
        dt     = 1.0 / cfg.n_sim_steps

        xs, ts, x0s = [], [], []
        x = x0.clone()
        with torch.no_grad():
            for i in range(cfg.n_sim_steps):
                t_val = i * dt
                t = torch.full((N,), t_val, device=device)
                # Euler-Maruyama step
                drift = fwd(t, x)
                x = x + drift * dt
                if cfg.sigma > 0:
                    x = x + cfg.sigma * (dt ** 0.5) * torch.randn_like(x)
                xs.append(x.clone())
                ts.append(torch.full((N, 1), t_val + dt, device=device))
                x0s.append(x0.clone())

        return torch.cat(xs, 0), torch.cat(ts, 0), torch.cat(x0s, 0)

    def _simulate_backward(
        self,
        x1: torch.Tensor,
        bwd: _DriftMLP,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate SDE backward from x1 using bwd drift (time-reversed).

        Returns (X_t, t, X_1_broadcast).
        Used to build the forward regression target.
        """
        cfg    = self.config
        device = self.spec.device
        N, d   = x1.shape
        dt     = 1.0 / cfg.n_sim_steps

        xs, ts, x1s = [], [], []
        x = x1.clone()
        with torch.no_grad():
            for i in range(cfg.n_sim_steps - 1, -1, -1):
                t_val = (i + 1) * dt                   # current time (going backward)
                t = torch.full((N,), t_val, device=device)
                drift = bwd(t, x)
                x = x - drift * dt                      # reverse direction
                if cfg.sigma > 0:
                    x = x + cfg.sigma * (dt ** 0.5) * torch.randn_like(x)
                xs.append(x.clone())
                ts.append(torch.full((N, 1), i * dt, device=device))
                x1s.append(x1.clone())

        return torch.cat(xs, 0), torch.cat(ts, 0), torch.cat(x1s, 0)

    # ------------------------------------------------------------------
    # Half-bridge training helpers
    # ------------------------------------------------------------------

    def _train_half(
        self,
        model:     _DriftMLP,
        X_t:       torch.Tensor,   # (M, d)  — midpoint samples
        t_vals:    torch.Tensor,   # (M, 1)  — corresponding times
        X_target:  torch.Tensor,   # (M, d)  — terminal / initial endpoint
        toward_future: bool,       # True → regress on (X_T - X_t)/(1-t), False → (X_0-X_t)/t
        label:     str,
    ) -> None:
        """Run n_steps_per_half gradient steps regressing the drift on X_t→X_target."""
        cfg    = self.config
        M      = X_t.shape[0]
        opt    = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.n_steps_per_half, eta_min=cfg.lr * 0.1
        )
        log_every = max(1, cfg.n_steps_per_half // 10)

        for step in range(cfg.n_steps_per_half):
            idx = torch.randint(M, (cfg.batch_size,), device=X_t.device)
            xt = X_t[idx]
            tt = t_vals[idx].squeeze(1)          # (B,)
            xt_end = X_target[idx]

            if toward_future:
                # target = (X_1 - X_t) / (1 - t)
                denom = (1.0 - tt).clamp(min=1e-3).unsqueeze(1)
                target = (xt_end - xt) / denom
            else:
                # target = (X_0 - X_t) / t
                denom = tt.clamp(min=1e-3).unsqueeze(1)
                target = (xt_end - xt) / denom

            pred = model(tt, xt)
            loss = (pred - target).pow(2).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if (step + 1) % log_every == 0:
                logger.debug("  [DSB %s] %s step %4d loss=%.5f",
                             self.spec.name, label, step + 1, loss.item())
                self._wandb.log({f"dsb/{label}_loss": loss.item()})

    # ------------------------------------------------------------------
    # BaselineRunner interface
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg    = self.config
        device = self.spec.device
        dim    = self.spec.dim

        fwd = _DriftMLP(dim, cfg.hidden, cfg.n_layers).to(device)
        bwd = _DriftMLP(dim, cfg.hidden, cfg.n_layers).to(device)

        # ── Initialise forward drift with a trivial linear field ─────────
        # f^0(t, x) ≈ (mean(nu) - x) / (1 - t)  — points toward target mean
        nu_mean = _to_tensor(
            self.spec.target.sample(512), device
        ).mean(0, keepdim=True)      # (1, d)

        def _init_fwd_pass(n_init_steps: int = 500) -> None:
            opt_init = torch.optim.Adam(fwd.parameters(), lr=cfg.lr)
            for _ in range(n_init_steps):
                x = _to_tensor(self.spec.source.sample(cfg.batch_size), device)
                t = torch.rand(len(x), device=device)
                denom = (1.0 - t).clamp(min=1e-3).unsqueeze(1)
                target = (nu_mean - x) / denom
                loss = (fwd(t, x) - target).pow(2).mean()
                opt_init.zero_grad(); loss.backward(); opt_init.step()

        _init_fwd_pass()
        logger.info("  [DSB] Initialised forward drift  (trivial bridge)")

        # ── IPF main loop ─────────────────────────────────────────────────
        for k in range(cfg.n_ipf_iters):
            logger.info("  [DSB] IPF iter %d/%d", k + 1, cfg.n_ipf_iters)

            # ── Backward half: simulate fwd → train bwd ────────────────
            x0_buf = _to_tensor(self.spec.source.sample(cfg.n_sim_samples), device)
            Xt_f, t_f, X0_f = self._simulate_forward(x0_buf, fwd)
            self._train_half(bwd, Xt_f, t_f, X0_f,
                             toward_future=False, label=f"bwd_k{k}")

            # ── Forward half: simulate bwd → train fwd ─────────────────
            x1_buf = _to_tensor(self.spec.target.sample(cfg.n_sim_samples), device)
            Xt_b, t_b, X1_b = self._simulate_backward(x1_buf, bwd)
            self._train_half(fwd, Xt_b, t_b, X1_b,
                             toward_future=True,  label=f"fwd_k{k}")

        # Save both drifts
        torch.save(fwd.state_dict(), os.path.join(self.spec.output_dir, "dsb_fwd.pth"))
        torch.save(bwd.state_dict(), os.path.join(self.spec.output_dir, "dsb_bwd.pth"))
        self._fwd_model = fwd
        logger.info("  [DSB] Training complete -> %s", self.spec.output_dir)

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._fwd_model is not None, "Call train() first."
        fwd    = self._fwd_model.eval()
        device = self.spec.device
        cfg    = self.config
        dt     = 1.0 / cfg.euler_steps

        x0 = _to_tensor(self.spec.source.sample(n), device)
        x  = x0.clone()

        with torch.no_grad():
            for i in range(cfg.euler_steps):
                t = torch.full((n,), i * dt, device=device)
                x = x + fwd(t, x) * dt
                if cfg.sigma > 0:
                    x = x + cfg.sigma * (dt ** 0.5) * torch.randn_like(x)

        return x0, x


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_tensor(x, device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    return x.float().to(device)
