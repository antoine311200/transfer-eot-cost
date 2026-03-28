"""dualwcfm.baselines.metaot
============================
MetaOT / Neural-W2 OT baseline.

This module implements a "single-task" version of the MetaOT framework
(Amos et al., NeurIPS 2022) that learns the W₂-OT map T: μ → ν via an
Input Convex Neural Network (ICNN) potential.

Background
----------
MetaOT meta-learns a hypernetwork that maps (μ, ν) summary statistics to
Kantorovich potentials, enabling fast adaptation at test time.  For the
purpose of a *from-scratch* per-experiment comparison we implement the inner
solver that MetaOT would be warm-started with:

  Let g_φ: ℝ^d → ℝ be an ICNN (convex by construction).
  The W₂-OT map (Brenier's theorem) satisfies  T(x) = ∇g*(x),
  where g*(y) = sup_x [⟨x, y⟩ − g(x)] is the Legendre-Fenchel conjugate.

Rather than computing g* analytically, we follow the dual formulation of
Makkuva et al. (2020) and Amos et al. (2022):

  max_{g, h}  E_{y~ν}[g(y)] + E_{x~μ}[h(x)] − E_{x~μ}[max_{y} (⟨x,y⟩ − g(y))]

where h is the conjugate approximation.  We parametrise both g (ICNN) and h
(ICNN or free MLP) and optimise via stochastic gradient ascent.

At inference: T(x) = ∇_x g*(x) ≈ argmax_{y} [⟨x, y⟩ − g(y)], solved by
a single Euler step with the gradient of g (i.e. the soft-map approximation):

  T̃(x) = x + η · ∇_x [max_y ⟨x,y⟩ − g(y)]

We approximate this by training an explicit map network T_θ alongside g,
guided by the convex conjugate correspondence.

Practical implementation (self-contained, no CVXPY / external OT library)
--------------------------------------------------------------------------
1. Train ICNN g_φ(x) and a free potential h_ψ(y) (ICNN or MLP) via:
     L = E_{y~ν}[g(y)] − E_{x~μ}[T̃_θ(x)ᵀ x − g(T̃_θ(x))]
   combined with a transport-matching term:
     + λ · E_{x~μ}[sinkhorn_grad_penalty(T̃_θ(x), ν)]

2. Train explicit map T_θ: ℝ^d → ℝ^d (free MLP) to approximate ∇g*:
     L_T = −E_{y~ν}[g(y)] + E_{x~μ}[‖T_θ(x)‖²/2 − g(T_θ(x))]

3. At inference: x ↦ T_θ(x).

This recovers the Monge map for ε→0 and is the test-time inner loop of
MetaOT when starting from a random initialisation (as opposed to the
meta-learned warm start MetaOT uses in the paper).

References
----------
Amos et al., "Meta Optimal Transport", NeurIPS 2022.
Makkuva et al., "Optimal Transport Mapping via Input Convex Neural Networks",
  ICML 2020.
Amos et al., "Input Convex Neural Networks", ICML 2017.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dualwcfm.baselines.base import BaselineRunner, BaselineSpec

logger = logging.getLogger("dualwcfm")


# ---------------------------------------------------------------------------
# Input Convex Neural Network (ICNN)
# ---------------------------------------------------------------------------

class ICNN(nn.Module):
    """Partially Input-Convex Neural Network (Amos et al., ICML 2017).

    Guarantees g(x) is convex in x by:
      - Using non-negative weights on all "pass-through" (z) paths.
      - Using SiLU activations (convex, non-decreasing).

    Architecture:
        z_0 = x
        z_{k+1} = σ(W_z^k · z_k + W_x^k · x + b^k),  W_z^k ≥ 0
        output  = w^T · z_K + b_out

    The non-negativity constraint on W_z is enforced via softplus
    reparametrisation during the forward pass.
    """

    def __init__(self, dim: int, hidden: int = 256, n_layers: int = 4,
                 strongly_convex_reg: float = 0.5):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.strongly_convex_reg = strongly_convex_reg

        sizes = [dim] + [hidden] * n_layers

        # Input → hidden projections (unconstrained Linear)
        self.W_x = nn.ModuleList([
            nn.Linear(dim, sizes[i + 1], bias=(i == 0))
            for i in range(n_layers)
        ])
        # Pass-through hidden → hidden weights (non-negative; stored as raw params)
        # Layer 0 has no z input so we only need layers 1..n_layers-1
        self.W_z_raw = nn.ParameterList([
            nn.Parameter(torch.randn(sizes[i + 1], sizes[i]) * 0.1)
            for i in range(1, n_layers)
        ])
        self.final = nn.Linear(hidden, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return g(x), shape (N,)."""
        x_flat = x.flatten(1)
        z = F.softplus(self.W_x[0](x_flat))           # layer 0: no z pass-through
        for i in range(1, self.n_layers):
            Wz_pos = F.softplus(self.W_z_raw[i - 1])   # non-negative weights
            h = self.W_x[i](x_flat) + z @ Wz_pos.T
            z = F.softplus(h)
        out = self.final(z).squeeze(-1)
        # Strongly convex regularisation: add (reg/2) * ‖x‖²
        return out + self.strongly_convex_reg * 0.5 * x_flat.pow(2).sum(-1)


# ---------------------------------------------------------------------------
# Map network  T_θ: ℝ^d → ℝ^d
# ---------------------------------------------------------------------------

class _MapMLP(nn.Module):
    """Free MLP that approximates the OT map T: x ↦ T(x)."""

    def __init__(self, dim: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MetaOTConfig:
    """Hyperparameters for the MetaOT / Neural-W2 OT baseline."""
    n_steps:      int   = 10_000   # total joint training steps
    batch_size:   int   = 512
    lr_g:         float = 5e-4     # ICNN potential learning rate
    lr_T:         float = 1e-3     # map network learning rate
    hidden:       int   = 256
    n_layers:     int   = 4
    strongly_convex_reg: float = 0.5
    # Gradient penalty weight for enforcing T(x) ≈ ∇g*(x) via conjugate
    lam_conj:     float = 1.0
    # Inner maximisation steps for estimating g*(x) via y* = argmax_y [<x,y>-g(y)]
    n_inner:      int   = 3
    inner_lr:     float = 0.1
    euler_steps:  int   = 1        # MetaOT is a 1-step map (no ODE needed)


# ---------------------------------------------------------------------------
# MetaOTRunner
# ---------------------------------------------------------------------------

class MetaOTRunner(BaselineRunner):
    """Neural W₂-OT map via ICNN potential (inner loop of MetaOT).

    Learns the Brenier potential g_φ (ICNN) and an explicit transport map
    T_θ (free MLP) jointly.  The ICNN guarantees that g is convex and its
    gradient ∇g* provides the OT map.  T_θ is trained as an amortised
    approximation of ∇g* to enable fast inference without inner-loop
    optimisation.

    At inference: x ↦ T_θ(x)  (single forward pass, no ODE).

    Parameters
    ----------
    spec:
        Baseline specification.
    config:
        Training hyperparameters.
    wandb_run:
        Optional W&B run.
    """

    def __init__(
        self,
        spec:     BaselineSpec,
        config:   Optional[MetaOTConfig] = None,
        wandb_run=None,
    ) -> None:
        super().__init__(spec, wandb_run)
        self.config = config or MetaOTConfig()
        self._T: Optional[_MapMLP] = None

    # ------------------------------------------------------------------
    # Conjugate approximation: given x, estimate T*(x) = argmax_y[<x,y>-g(y)]
    # We approximate this via gradient ascent on y starting from x.
    # ------------------------------------------------------------------

    def _conjugate_approx(
        self,
        g: ICNN,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate y* = argmax_y [⟨x, y⟩ − g(y)] via gradient ascent.

        Initialises y = x and takes n_inner steps of step size inner_lr.
        """
        cfg = self.config
        y   = x.detach().clone().requires_grad_(True)
        for _ in range(cfg.n_inner):
            obj = (x.detach() * y).sum(-1) - g(y)   # (N,)
            grad_y = torch.autograd.grad(obj.sum(), y)[0]
            y = (y + cfg.inner_lr * grad_y).detach().requires_grad_(True)
        return y.detach()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg    = self.config
        device = self.spec.device
        dim    = self.spec.dim

        g = ICNN(dim, cfg.hidden, cfg.n_layers,
                 strongly_convex_reg=cfg.strongly_convex_reg).to(device)
        T = _MapMLP(dim, cfg.hidden, cfg.n_layers).to(device)

        opt_g = torch.optim.Adam(g.parameters(), lr=cfg.lr_g)
        opt_T = torch.optim.Adam(T.parameters(), lr=cfg.lr_T)
        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_g, T_max=cfg.n_steps, eta_min=cfg.lr_g * 0.1
        )
        sched_T = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_T, T_max=cfg.n_steps, eta_min=cfg.lr_T * 0.1
        )

        log_every = max(1, cfg.n_steps // 20)

        for step in range(cfg.n_steps):
            x = _to_tensor(self.spec.source.sample(cfg.batch_size), device)
            y = _to_tensor(self.spec.target.sample(cfg.batch_size), device)

            # ── Step 1: Update ICNN g via the W₂ dual ──────────────────
            # W₂ dual (Kantorovich with cost c(x,y) = ‖x−y‖²/2):
            #   max_g  E_ν[g(y)] + E_μ[g*(x)]
            # We approximate g*(x) via the inner maximisation above.
            T_x       = T(x).detach()            # map approximation (no gradient through T here)
            # g*(x) ≈ ⟨x, T(x)⟩ − g(T(x))      (first-order Legendre)
            g_conj_x  = (x * T_x).sum(-1) - g(T_x)
            loss_g    = -(g(y).mean() + g_conj_x.mean())

            opt_g.zero_grad()
            loss_g.backward()
            nn.utils.clip_grad_norm_(g.parameters(), 1.0)
            opt_g.step()
            sched_g.step()

            # ── Step 2: Update map T to approximate ∇g*(x) ────────────
            # T_θ(x) should satisfy: T_θ(x) = argmax_y [⟨x,y⟩ − g(y)]
            # Loss: minimise ‖T_θ(x)‖²/2 − g(T_θ(x)) − ⟨x, T_θ(x)⟩
            # Equivalently: minimise −[⟨x, T_θ(x)⟩ − g(T_θ(x))] = −g*(x)
            # + push T#μ toward ν via Sinkhorn-free SW term
            T_x2      = T(x)
            # Conjugate term: maximise ⟨x, T(x)⟩ − g(T(x))
            conj_loss = -(x.detach() * T_x2).sum(-1).mean() + g(T_x2).mean()

            # Push-forward term: penalise W₂(T#μ, ν) via sample Sinkhorn
            push_loss = _sample_sinkhorn_loss(T_x2, y, eps=0.05)

            loss_T = cfg.lam_conj * conj_loss + push_loss

            opt_T.zero_grad()
            loss_T.backward()
            nn.utils.clip_grad_norm_(T.parameters(), 1.0)
            opt_T.step()
            sched_T.step()

            if (step + 1) % log_every == 0:
                logger.info(
                    "  [MetaOT %s] step %5d/%d  L_g=%.4f  L_T=%.4f",
                    self.spec.name, step + 1, cfg.n_steps,
                    -loss_g.item(), loss_T.item(),
                )
                self._wandb.log({
                    "metaot/loss_g": -loss_g.item(),
                    "metaot/loss_T": loss_T.item(),
                })

        torch.save(g.state_dict(), os.path.join(self.spec.output_dir, "metaot_g.pth"))
        torch.save(T.state_dict(), os.path.join(self.spec.output_dir, "metaot_T.pth"))
        self._T = T
        logger.info("  [MetaOT] Training complete -> %s", self.spec.output_dir)

    # ------------------------------------------------------------------

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._T is not None, "Call train() first."
        T      = self._T.eval()
        device = self.spec.device

        x0 = _to_tensor(self.spec.source.sample(n), device)
        with torch.no_grad():
            y = T(x0)
        return x0, y


# ---------------------------------------------------------------------------
# Minibatch Sinkhorn loss for the push-forward term
# ---------------------------------------------------------------------------

def _sample_sinkhorn_loss(
    x_gen: torch.Tensor,
    y_ref: torch.Tensor,
    eps:   float = 0.05,
    n_iter: int  = 20,
) -> torch.Tensor:
    """Sinkhorn divergence S_ε(x_gen, y_ref) — differentiable w.r.t. x_gen."""
    def _ot_eps(A, B):
        N, M = len(A), len(B)
        C = torch.cdist(A, B).pow(2)
        log_K = -C / eps
        u = torch.zeros(N, device=A.device)
        for _ in range(n_iter):
            v = -torch.logsumexp(log_K + u.unsqueeze(1), dim=0)
            u = -torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
        log_P = log_K + u.unsqueeze(1) + v.unsqueeze(0)
        return (log_P.exp() * C).sum()

    ot_xy = _ot_eps(x_gen, y_ref.detach())
    with torch.no_grad():
        ot_xx = _ot_eps(x_gen.detach(), x_gen.detach())
        ot_yy = _ot_eps(y_ref, y_ref)
    return (ot_xy - 0.5 * ot_xx - 0.5 * ot_yy).clamp(min=0.0)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_tensor(x, device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    return x.float().to(device)
