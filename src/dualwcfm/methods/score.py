from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tqdm import trange

from dualwcfm.distributions.coupling import DiskCouplingDistribution
from dualwcfm.methods.standard import StandardMethod, StandardConfig
from dualwcfm.mlp import MLP
from dualwcfm.utils import is_image_tensor

if TYPE_CHECKING:
    from dualwcfm.trainer import Trainer

logger = logging.getLogger("dualwcfm")


# ---------------------------------------------------------------------------
# Module-level helpers (not class methods — avoids any static-method binding
# issues with aggressive linters/formatters)
# ---------------------------------------------------------------------------

def _compute_ksd(z: torch.Tensor, score: torch.Tensor, n_sub: int = 200) -> torch.Tensor:
    """V-statistic estimate of the kernelised Stein discrepancy.

    Uses an RBF kernel k(x,y) = exp(−‖x−y‖²/2h²) with the median heuristic
    for bandwidth h.  KSD²(q,p) = E_{x,y~q}[k_p(x,y)] where

        k_p(x,y) = s(x)ᵀs(y)·k + s(x)ᵀ∇_y k + ∇_x kᵀs(y) + tr(∇_x∇_y k)

    Returns the (non-negative) scalar KSD = √max(0, mean(k_p)).
    """
    N = min(n_sub, z.shape[0])
    idx = torch.randperm(z.shape[0], device=z.device)[:N]
    z, score = z[idx], score[idx]
    d = z.shape[1]

    dists_sq = torch.cdist(z, z).pow(2)                              # (N, N)
    h2 = (dists_sq.sqrt().median().pow(2) / (2 * math.log(N + 1))).clamp(min=1e-6)
    K = torch.exp(-dists_sq / (2 * h2))                              # (N, N)

    diff = z.unsqueeze(1) - z.unsqueeze(0)                           # (N, N, d)
    grad_y_K = diff / h2 * K.unsqueeze(-1)                           # (N, N, d)

    term1 = K * (score @ score.T)                                     # s(x)ᵀs(y)·k
    term2 = (score.unsqueeze(1) * grad_y_K).sum(dim=-1)              # s(x)ᵀ∇_y k
    term3 = (-grad_y_K * score.unsqueeze(0)).sum(dim=-1)             # ∇_x kᵀs(y)
    term4 = K * (d / h2 - dists_sq / h2.pow(2))                      # tr(∇_x∇_y k)

    return (term1 + term2 + term3 + term4).mean().clamp(min=0.0).sqrt()


def _hutchinson_div(
    model: nn.Module, t: torch.Tensor, x: torch.Tensor, n_probes: int = 1
) -> torch.Tensor:
    """Estimate tr(∂ model(t,x) / ∂x) via Hutchinson's trace estimator.

    Returns shape (B,).
    """
    total = torch.zeros(x.shape[0], device=x.device)
    for _ in range(n_probes):
        v = torch.randn_like(x)
        x_req = x.detach().requires_grad_(True)
        with torch.enable_grad():
            out = model(t.detach(), x_req)
            vjp = torch.autograd.grad((out * v).sum(), x_req, create_graph=False)[0]
        total = total + (vjp.detach() * v).sum(dim=1)
    return total / n_probes


@dataclass
class ScoreConfig(StandardConfig):
    """StandardConfig + score matching of the initial coupling pi_0."""
    score_epochs: int = 1000
    score_sigma: float = 0.1
    score_lr: float = 1e-3


class ScoreMethod(StandardMethod):
    """StandardMethod extended with a time-evolving score model for π_t.

    Pretrain phase
    --------------
    After the standard marginal + flow pretrain, learns s_θ ≈ ∇ log π_0 via
    denoising score matching (Vincent 2011):

        L = E_{Z~π_0} E_{ε~N(0,I)} [ ‖ s_θ(Z + σε) + ε/σ ‖² ]

    Post-process (per iteration)
    ----------------------------
    Advances the coupling by one Euler step using the marginal bridge velocities:

        (x, y)  ←  (x + Δ·u_t(x),  y + Δ·v_t(y))

    CFM + score combined step (per iteration)
    ------------------------------------------
    A single training loop jointly trains the flow model and the score model
    using the full change-of-measure weight:

        w̃_t(Z) = a_t(X) + b_t(Y)
                + Δ [ u_{t-Δ}(X)·S^x_{t-Δ}(Z) + v_{t-Δ}(Y)·S^y_{t-Δ}(Z)
                      + ∇·V_{t-Δ}(Z) ]

    CFM loss:
        E_Z [ w̃_t(Z) · ‖ u_θ(s, Z_s, X) − target_s(Z) ‖² ]

    Score loss (≡ E_{π_{t+Δ}}[ ‖ s_θ − Target ‖² ]):
        E_Z [ w̃_t(Z) · ‖ s_θ(Z) − (s_{prev}(Z) + [∇_x a_t, ∇_y b_t]) ‖² ]

    Both losses share the same weight computed once per mini-batch.
    """

    def __init__(self, config: ScoreConfig):
        super().__init__(config)
        self.score_model: nn.Module | None = None
        self.previous_score_model: nn.Module | None = None
        self.score_optimizer = None
        # Fixed held-out batch (CPU) sampled from π_0, kept throughout training
        # for consistent drift and KSD tracking across iterations.
        self._score_probe: tuple[torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Pretrain
    # ------------------------------------------------------------------

    def pretrain(self, trainer: Trainer, retrain: bool = True, save: str = None) -> None:
        super().pretrain(trainer, retrain, save)
        self._pretrain_score(trainer)
        self.previous_score_model = copy.deepcopy(self.score_model)
        self.previous_score_model.eval()

    def _pretrain_score(self, trainer: Trainer) -> None:
        """Learn the score of π_0 via denoising score matching."""
        dim = trainer.strategy.config.dimension
        hidden_size = trainer.strategy.config.hidden_size
        device = trainer.device
        batch_size = trainer.config.batch_size
        sigma = self.config.score_sigma

        self.score_model = MLP(
            dim=2 * dim, out_dim=2 * dim, w=hidden_size, time_varying=False
        ).to(device)
        self.score_optimizer = torch.optim.Adam(
            self.score_model.parameters(), lr=self.config.score_lr
        )
        self.score_model.train()

        pbar = trange(self.config.score_epochs, desc="Score Matching (pi_0)")
        for _ in pbar:
            x0, y0 = trainer.initial_coupling.coupling_sample(batch_size)
            x0, y0 = x0.to(device), y0.to(device)
            xy = torch.cat([x0, y0], dim=1)

            eps = torch.randn_like(xy)
            score = self.score_model(xy + sigma * eps)
            # loss = ((score + eps / sigma) ** 2).mean()
            loss = ((score * sigma + eps) ** 2).sum(dim=1).mean()

            self.score_optimizer.zero_grad()
            loss.backward()
            self.score_optimizer.step()
            pbar.set_postfix({"Score Loss": loss.item()})

        # Initialise fixed probe batch (once) and log pretrain diagnostics
        with torch.no_grad():
            xp, yp = trainer.initial_coupling.coupling_sample(256)
            self._score_probe = (xp.cpu(), yp.cpu())
        self._log_score_diagnostics(trainer, tag="pretrain")

    # ------------------------------------------------------------------
    # Post-process: Euler step with marginal bridges
    # ------------------------------------------------------------------

    def step_postprocess(self, trainer: Trainer, iteration: int, ctx: dict) -> None:
        """Advance every coupling pair one Euler step: x ← x + Δ·u_t(x), y ← y + Δ·v_t(y)."""
        device = trainer.device
        delta = 1.0 / trainer.config.iterations
        t = ctx["t"]

        (mu_model, _), (nu_model, _) = trainer.strategy.get_bridges(device)
        mu_model.eval()
        nu_model.eval()

        chunk_size = (
            trainer.current_coupling.chunk_size
            if isinstance(trainer.current_coupling, DiskCouplingDistribution)
            else (100_000 if not trainer.config.image_shape else 256)
        )
        n_chunks = (trainer.current_coupling.size + chunk_size - 1) // chunk_size

        with torch.no_grad():
            for chunk_idx in trange(n_chunks, desc="Advancing coupling (bridge Euler)"):
                i = chunk_idx * chunk_size
                actual = min(chunk_size, trainer.current_coupling.size - i)
                t_chunk = torch.full((actual, 1), t, device=device)

                if isinstance(trainer.current_coupling, DiskCouplingDistribution):
                    x, y = trainer.current_coupling._load_chunk(chunk_idx)
                    x, y = x.to(device), y.to(device)
                else:
                    x = trainer.current_coupling.dataset[0][i : i + actual].to(device)
                    y = trainer.current_coupling.dataset[1][i : i + actual].to(device)

                x_new = x + delta * mu_model(t_chunk, x)
                y_new = y + delta * nu_model(t_chunk, y)
                trainer.current_coupling.update_chunk(i, x_new.cpu(), y_new.cpu())

        trainer.visualizer.at_postprocess_end(iteration, trainer, ctx)

    # ------------------------------------------------------------------
    # CFM + Score: combined training loop
    # ------------------------------------------------------------------

    def step_flow_matching(self, trainer: Trainer, iteration: int, ctx: dict) -> None:
        """Jointly train the flow model and the score model with the full COM weight."""
        device = trainer.device
        batch_size = trainer.config.batch_size
        is_image = ctx["is_image"]
        backward = ctx["backward"]
        delta = 1.0 / trainer.config.iterations
        t = ctx["t"]
        t_prev = max(0.0, t - delta)
        t_prev_tensor = torch.full((batch_size, 1), t_prev, device=device)

        # --- Models ---
        model, optimizer = trainer.strategy.get_flow(iteration, device)
        previous_model = trainer.strategy.get_previous_flow(iteration, device)
        previous_model.load_state_dict(model.state_dict())
        previous_model.eval()
        model.train()

        (mu_model, _), (nu_model, _) = trainer.strategy.get_bridges(device)
        f_dual, g_dual, _ = trainer.strategy.get_duals(device)
        for m in (mu_model, nu_model, f_dual, g_dual):
            m.eval()
        self.score_model.train()

        # --- Visualizer setup (mirrors StandardMethod) ---
        mu_t, nu_t = trainer.current_coupling.coupling_sample(batch_size)
        mu_t, nu_t = mu_t.to(device), nu_t.to(device)
        t_full = torch.full((batch_size, 1), t, device=device)
        nu_t_target = self._interpolate(
            trainer.initial_coupling.get_marginals()[1],
            trainer.new_target,
            trainer,
            batch_size,
            t_full,
            trainer._sigma,
            device,
        )
        flow_ctx = {
            "x0_samp": mu_t,
            "x1_samp": nu_t,
            "x1_target": nu_t_target,
            "model": model,
            "x0_quiver": mu_t[: min(400, batch_size)] if not is_image else None,
            "is_image": is_image,
        }
        start_ctx = trainer.visualizer.at_flow_matching_start(iteration, trainer, flow_ctx)

        # --- Combined training loop ---
        pbar = trange(trainer.config.cfm_epochs, desc="CFM + Score Training")
        for _ in pbar:
            x0, x1 = trainer.current_coupling.coupling_sample(batch_size)
            x0, x1 = x0.clone().to(device), x1.clone().to(device)

            # ---- Full change-of-measure weight (computed on original X=x0, Y=x1) ----
            with torch.no_grad():
                log_w = trainer.strategy.log_weight(x0, x1, trainer.config.iterations).squeeze()

            if iteration > 0:
                # Δ [ u_{t-Δ}·S^x_{t-Δ} + v_{t-Δ}·S^y_{t-Δ} ]
                with torch.no_grad():
                    u_prev = mu_model(t_prev_tensor, x0)
                    v_prev = nu_model(t_prev_tensor, x1)
                    xy = torch.cat([x0, x1], dim=1)
                    score_prev = self.previous_score_model(xy)
                    s_x = score_prev[:, : x0.shape[1]]
                    s_y = score_prev[:, x0.shape[1] :]
                    dot = (u_prev * s_x).sum(dim=1) + (v_prev * s_y).sum(dim=1)  # (B,)

                # ∇·V_{t-Δ} = ∇_x·u_{t-Δ}(x) + ∇_y·v_{t-Δ}(y)  via Hutchinson
                div_u = _hutchinson_div(mu_model, t_prev_tensor, x0)  # (B,)
                div_v = _hutchinson_div(nu_model, t_prev_tensor, x1)  # (B,)

                with torch.no_grad():
                    log_w = log_w + delta * (dot + div_u.detach() + div_v.detach())

            with torch.no_grad():
            #     log_w = log_w - log_w.max()
                w = torch.exp(log_w)
            #     w = w / w.sum() * batch_size  # (B,), E[w] = 1

            # ---- CFM loss ----
            # Backward direction swap applied only to the CFM interpolation
            x0_cfm = x1 if (backward and iteration % 2 != 0) else x0
            x1_cfm = x0 if (backward and iteration % 2 != 0) else x1

            s = trainer.flow_matching.random_time(batch_size, device)
            x_s = trainer.flow_matching.interpolate(x0_cfm, x1_cfm, s, trainer._sigma2)
            u_s = model(s, x_s, x0_cfm)
            target_u_s = trainer.flow_matching.vector_field(
                x_s, s, x0_cfm, x1_cfm, extend=True, noisy=True
            )
            # Broadcast weight to match output dims (vector: (B,1); image: (B,1,1,1))
            w_cfm = w.view(batch_size, *([1] * (u_s.dim() - 1)))
            loss_cfm = (w_cfm * (u_s - target_u_s) ** 2).mean()

            # ---- Score loss ----
            # Target: s_prev(Z) + [∇_x a_t(X), ∇_y b_t(Y)]
            x_req = x0.detach().requires_grad_(True)
            y_req = x1.detach().requires_grad_(True)
            with torch.enable_grad():
                grad_a = torch.autograd.grad(f_dual(x_req).sum(), x_req, create_graph=False)[0]
                grad_b = torch.autograd.grad(g_dual(y_req).sum(), y_req, create_graph=False)[0]

            xy = torch.cat([x0, x1], dim=1)
            with torch.no_grad():
                target_score = self.previous_score_model(xy) + torch.cat(
                    [grad_a.detach(), grad_b.detach()], dim=1
                )

            score_pred = self.score_model(xy.detach())
            loss_score = (w.unsqueeze(1) * (score_pred - target_score) ** 2).mean()

            # ---- Joint backward ----
            loss = loss_cfm + loss_score
            optimizer.zero_grad()
            self.score_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.score_optimizer.step()

            pbar.set_postfix({"CFM": loss_cfm.item(), "Score": loss_score.item()})

        trainer.visualizer.at_flow_matching_end(iteration, trainer, flow_ctx, start_ctx)

        # Transport cost diagnostic (mirrors StandardMethod)
        n_diag = 10000 if not is_image else min(256, batch_size)
        x0_tc = (trainer.initial_coupling.sample_target if backward else trainer.initial_coupling.sample_init)(n_diag).to(device)
        x1_tc = (trainer.new_target if backward else trainer.new_source).sample(n_diag).to(device)
        t_full = torch.full((n_diag, 1), t, device=device)
        xt_tc = trainer.flow_matching.interpolate(x0_tc, x1_tc, t_full, trainer._sigma)
        model.eval()
        _evolved = trainer.flow_matching.evolve(xt_tc, model, num_steps=100, sigma=trainer._sigma, extend=True)
        yt = (
            _evolved[:, : xt_tc.shape[1], :, :]
            if is_image_tensor(xt_tc)
            else _evolved[:, : trainer.strategy.config.dimension]
        )
        transport_cost = trainer.visualizer.transport_cost(xt_tc.cpu(), yt.cpu())
        if transport_cost is not None:
            cost_flat = transport_cost.view(transport_cost.shape[0], -1)
            logger.info(f"Cost error {torch.norm(cost_flat, dim=1).mean().item():.6f}")

        # Diagnostics (uses previous_score_model = s_{t-1} for drift, before overwrite)
        self._log_score_diagnostics(trainer, tag=f"iter {iteration}")

        # Store current score as s_{t} for next iteration's s_{t-Δ}
        self.previous_score_model = copy.deepcopy(self.score_model)
        self.previous_score_model.eval()

    # ------------------------------------------------------------------
    # Score diagnostics  (helpers are module-level: _compute_ksd, _hutchinson_div)
    # ------------------------------------------------------------------

    def _log_score_diagnostics(self, trainer: Trainer, tag: str) -> None:
        """Log score norm, KSD, and drift on the fixed probe batch.

        Metrics
        -------
        norm  : mean ‖s_θ(Z)‖₂ over a fresh batch from the current coupling.
                For a Gaussian π_t with variance σ², expected ≈ √(2d)/σ.
        KSD   : kernelised Stein discrepancy (RBF kernel, V-statistic) on the
                fixed probe.  KSD→0 iff s_θ = ∇ log π_t a.e.
        drift : mean ‖s_θ^new(Z) − s_θ^prev(Z)‖₂ on the fixed probe.
                Large drift = the score is still changing; near-zero = converged.
        """
        device = trainer.device
        n = min(256, trainer.config.batch_size)

        self.score_model.eval()
        with torch.no_grad():
            # --- norm: fresh samples from current coupling ---
            xf, yf = trainer.current_coupling.coupling_sample(n)
            xf, yf = xf.to(device), yf.to(device)
            xy_fresh = torch.cat([xf, yf], dim=1)
            s_fresh = self.score_model(xy_fresh)
            norm = s_fresh.norm(dim=1).mean().item()

            # --- KSD + drift: fixed probe from π_0 ---
            xp = self._score_probe[0].to(device)
            yp = self._score_probe[1].to(device)
            xy_probe = torch.cat([xp, yp], dim=1)
            s_probe = self.score_model(xy_probe)

            ksd = _compute_ksd(xy_probe, s_probe).item()

            if self.previous_score_model is not None:
                s_prev_probe = self.previous_score_model(xy_probe)
                drift = (s_probe - s_prev_probe).norm(dim=1).mean().item()
            else:
                drift = float("nan")

        logger.info(
            f"[Score {tag}]  ||s||={norm:.4f}  KSD={ksd:.6f}  drift={drift:.4f}"
        )
        self.score_model.train()

