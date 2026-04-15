"""dualwcfm.baselines.genot
==========================
GENOT — Entropic (Gromov) Wasserstein Flow Matching.

Algorithm (Klein, Palla, Theis et al., 2023 — arXiv:2310.09614)
----------------------------------------------------------------
GENOT learns a **stochastic** map from source μ to target ν by combining
mini-batch entropic OT with conditional flow matching.  Unlike deterministic
OT-CFM / LightSB (which learn a point-to-point map), GENOT produces a
stochastic conditional map  (x, z) → y  where z ~ N(0, I_d) is auxiliary
noise in **target space** and x is the source condition.

This naturally handles non-deterministic OT plans (e.g. entropic regularisation
or Gromov-Wasserstein cross-domain transport).  For our synthetic benchmarks
we use the standard W₂ (quadratic-cost) variant.

Training (W₂ same-space variant)
---------------------------------
At each gradient step:
  1. Sample x ~ μ and y ~ ν (mini-batch, size ``batch``)
  2. Compute the entropic OT plan T ∈ ℝ^{n×n} via log-domain Sinkhorn:
        T[i,j] ∝ exp( (f_i + g_j − ‖x_i − y_j‖²) / ε )
  3. For each x_i, sample y_i ~ T[i, :] / Σ_j T[i,j]  (conditional draw)
  4. Source of the flow: z_i = x_i  (p_z = μ for same-space W₂)
  5. Linear interpolant:  z_t = (1−t) · x_i + t · y_i,  t ~ U[0,1]
  6. CFM regression:  L = ‖v_θ(t, z_t, x_i) − (y_i − x_i)‖²

Inference
---------
Given x ~ μ:
  1. z_0 = x  (start from source, not from noise)
  2. Euler-integrate  dz/dt = v_θ(t, z_t, x)  from t=0 to t=1
  3. Return z_1  as the transported sample

Notes
-----
- p_z = μ (source distribution) is the correct choice for same-space W₂
  transport.  For cross-domain / Gromov-Wasserstein transport (source and
  target in different metric spaces), Klein et al. use p_z = N(0, I) in the
  target space — that variant is NOT used here.
- Conditioning v_θ on x_i (the starting point) is the key difference from
  plain OT-CFM: OT-CFM marginalises over x_0 at each step, whereas GENOT
  keeps x_0 as an explicit condition throughout the flow.  This lets the
  model learn the full conditional distribution p(y | x) under the OT plan,
  naturally handling stochastic / multi-valued maps.

References
----------
Klein, D., Palla, G., Alavizadeh, S., Hrovatin, K., Heiber, R.,
Moinfar, A.A., ... & Theis, F.J. (2023).
"GENOT: Entropic (Gromov) Wasserstein Flow Matching with Applications
to Single-Cell Genomics." arXiv:2310.09614.
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
# Networks
# ---------------------------------------------------------------------------

class _ConditionalVelocityMLP(nn.Module):
    """Conditional velocity field  v_θ(t, z_t, x) → ℝ^d.

    Input concatenation: [t | z_t | x]  of dimension  1 + 2*d.
    The source condition x is concatenated with the current state z_t so the
    model learns to adapt the flow based on which source point it starts from.
    """

    def __init__(self, dim: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        in_dim = 2 * dim + 1   # [z_t, x, t]
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        t:  torch.Tensor,   # (N,) or (N,1)
        zt: torch.Tensor,   # (N, d)
        x:  torch.Tensor,   # (N, d)
    ) -> torch.Tensor:
        t_in = t.view(-1, 1)
        inp  = torch.cat([t_in, zt.flatten(1), x.flatten(1)], dim=1)  # (N, 2d+1)
        return self.net(inp)  # (N, d)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GENOTConfig:
    """Hyperparameters for the GENOT baseline."""
    eps:          float = 0.1     # entropic regularisation ε for mini-batch Sinkhorn
    n_sinkhorn:   int   = 50      # Sinkhorn iterations per step
    # Training
    n_steps:      int   = 20_000  # gradient steps
    batch:        int   = 512     # mini-batch size (both source and target)
    lr:           float = 1e-3
    hidden:       int   = 256
    n_layers:     int   = 4
    # Inference
    euler_steps:  int   = 100


# ---------------------------------------------------------------------------
# Sinkhorn (log-domain, no-grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sinkhorn_plan(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    n_iter: int = 50,
) -> torch.Tensor:
    """Return the entropic OT plan T ∈ ℝ^{n×m} via log-domain Sinkhorn.

    Parameters
    ----------
    x : (n, d) source mini-batch
    y : (m, d) target mini-batch
    eps : entropic regularisation
    n_iter : Sinkhorn iterations

    Returns
    -------
    T : (n, m) transport plan (rows sum to 1/n, cols sum to 1/m)
    """
    n, m = x.shape[0], y.shape[0]
    device = x.device

    # Cost matrix  C_ij = ‖x_i − y_j‖²
    C = torch.cdist(x.flatten(1), y.flatten(1)).pow(2)   # (n, m)

    # Log-domain Sinkhorn: dual variables f (n,), g (m,)
    log_a = torch.full((n,), -torch.log(torch.tensor(float(n))), device=device)
    log_b = torch.full((m,), -torch.log(torch.tensor(float(m))), device=device)

    f = torch.zeros(n, device=device)
    g = torch.zeros(m, device=device)

    for _ in range(n_iter):
        # f_i = ε·log_a_i − ε·logsumexp_j((g_j − C_ij/ε))
        f = eps * log_a - eps * torch.logsumexp(
            g.unsqueeze(0) - C / eps, dim=1
        )
        g = eps * log_b - eps * torch.logsumexp(
            f.unsqueeze(1) - C / eps, dim=0
        )

    # T_ij = exp((f_i + g_j − C_ij) / ε)
    log_T = (f.unsqueeze(1) + g.unsqueeze(0) - C) / eps
    T = log_T.exp()
    # Guard against NaN/Inf
    T = torch.nan_to_num(T, nan=1.0 / (n * m), posinf=1.0 / (n * m))
    return T


def _row_sample(T: torch.Tensor) -> torch.Tensor:
    """For each row i, sample one column index j ~ T[i, :].

    Returns (n,) index tensor.
    """
    # Normalise rows (they may not sum to exactly 1/n due to numerical drift)
    p = T / T.sum(dim=1, keepdim=True).clamp(min=1e-12)
    # Guard NaN rows → uniform
    bad = p.isnan().any(dim=1)
    if bad.any():
        p[bad] = 1.0 / T.shape[1]
    return torch.multinomial(p, num_samples=1).squeeze(1)   # (n,)


# ---------------------------------------------------------------------------
# GENOTRunner
# ---------------------------------------------------------------------------

class GENOTRunner(BaselineRunner):
    """GENOT: Generalized Neural OT (Bunne et al., NeurIPS 2023).

    Learns a stochastic conditional map (x, z) → y by training a conditional
    velocity field on mini-batch entropic OT plans.

    Parameters
    ----------
    spec:
        Baseline specification (source/target distributions, device, …).
    config:
        GENOT hyperparameters.
    wandb_run:
        Optional W&B run for loss logging.
    """

    def __init__(
        self,
        spec:     BaselineSpec,
        config:   Optional[GENOTConfig] = None,
        wandb_run=None,
    ) -> None:
        super().__init__(spec, wandb_run)
        self.config = config or GENOTConfig()
        self._model: Optional[_ConditionalVelocityMLP] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg    = self.config
        device = self.spec.device
        dim    = self.spec.dim

        model = _ConditionalVelocityMLP(dim, cfg.hidden, cfg.n_layers).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.n_steps, eta_min=cfg.lr * 0.1
        )
        log_every = max(1, cfg.n_steps // 20)

        logger.info("  [GENOT] Training conditional velocity field  (dim=%d, steps=%d)",
                    dim, cfg.n_steps)

        use_oracle = self.spec.coupling_fn is not None
        if use_oracle:
            logger.info("  [GENOT] coupling_fn provided — replacing Sinkhorn pairing "
                        "with oracle (x, T(x)) pairs.")

        for step in range(cfg.n_steps):
            if use_oracle:
                # Oracle coupling: directly sample (x, T(x)) pairs — no Sinkhorn needed
                x, y_matched = self.spec.coupling_fn(cfg.batch)
                x, y_matched = x.to(device), y_matched.to(device)
            else:
                # 1. Sample mini-batches
                x = _to_tensor(self.spec.source.sample(cfg.batch), device)
                y = _to_tensor(self.spec.target.sample(cfg.batch), device)

                # 2. Sinkhorn plan  (no grad; used only for coupling)
                T = _sinkhorn_plan(x, y, eps=cfg.eps, n_iter=cfg.n_sinkhorn)

                # 3. Sample y_i ~ plan(· | x_i) for each i
                y_idx     = _row_sample(T)          # (n,)
                y_matched = y[y_idx]                # (n, d)

            # 4. Source of the flow: z_0 = x  (p_z = μ, same-space W₂ variant)
            # 5. Interpolate  z_t = (1-t)·x + t·y_matched
            t  = torch.rand(x.shape[0], device=device)
            zt = (1 - t.view(-1, 1)) * x + t.view(-1, 1) * y_matched
            # 6. CFM loss:  ‖v_θ(t, z_t, x) − (y − x)‖²
            target = y_matched - x               # conditional vector field
            pred   = model(t, zt, x)
            loss   = (pred - target).pow(2).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if (step + 1) % log_every == 0:
                logger.info(
                    "  [GENOT %s] step %5d/%d  loss=%.5f",
                    self.spec.name, step + 1, cfg.n_steps, loss.item(),
                )
                self._wandb.log({"genot/cfm_loss": loss.item()})

        torch.save(model.state_dict(),
                   os.path.join(self.spec.output_dir, "genot_model.pth"))
        self._model = model
        logger.info("  [GENOT] Training complete -> %s", self.spec.output_dir)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._model is not None, "Call train() first."
        model  = self._model.eval()
        device = self.spec.device
        cfg    = self.config
        dt     = 1.0 / cfg.euler_steps

        x0 = _to_tensor(self.spec.source.sample(n), device)

        # Start from z = x0 (p_z = μ, same-space W₂ variant) and integrate
        # conditioned on x0 throughout.
        z = x0.clone()

        with torch.no_grad():
            for i in range(cfg.euler_steps):
                t  = torch.full((n,), i * dt, device=device)
                vt = model(t, z, x0)
                z  = z + vt * dt
        return x0, z   # (x0, T(x0) ≈ OT map output)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_tensor(x, device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    return x.float().to(device)
