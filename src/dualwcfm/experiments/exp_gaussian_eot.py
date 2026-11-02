"""dualwcfm.experiments.exp_gaussian_eot
=======================================
Entropic Gaussian transfer with a closed-form ground truth.

Why this experiment
-------------------
The 2D benchmarks of ``exp_2d`` use *deterministic* reference couplings
(``y = T(x)``, no entropy).  A deterministic coupling cannot be tilted into a
different one, and transporting the reference pairs along OT marginal bridges
already solves those benchmarks when ``T`` is a translation, rotation or
scaling.  They therefore cannot tell whether the tilting step does anything.

Here the reference coupling is a genuine entropic OT plan: a non-degenerate
joint Gaussian, which is the EOT plan of the bilinear cost
``c(x, y) = -eps * q * <x, y>`` between its marginals.  Every tilt
``exp(phi(x) + psi(y)) * pi_0`` keeps the cross block ``q`` of the precision
matrix, so for new Gaussian marginals the transferred plan is known in closed
form:

* isotropic marginal variances ``vx, vy`` and precision cross coefficient ``q``
  give ``D = (1 + sqrt(1 + 4 vx vy q^2)) / (2 vx vy)``, cross covariance
  ``c = -q / D`` and correlation ``rho = c / sqrt(vx vy)``.

Translating the marginals leaves ``rho`` unchanged; **scaling** them changes it.
Pairs transported along the OT bridges (particle mixing) keep the reference
correlation ``rho_0``, so for scaled marginals particles alone are biased,
while the tilting flow targets ``rho*``.

Diagnostics recorded at every outer iteration, on the current coupling
dataset:

``q_ratio``   empirical precision cross block / reference ``q``.  Invariant
              under tilts: it measures the *latent cost* carried by the
              current coupling (1 = exact, drift = cost error).
``rho_hat``   empirical correlation of the current coupling.
``rho_own``   correlation of the EOT plan of the latent cost between the
              current coupling's own marginals (``rho_hat == rho_own`` iff the
              iterate is an EOT plan of the latent cost, Lemma "tilt class").
``rho_path``  correlation of the exact plan pi_t on the target marginal path.

Usage
-----
    python -m dualwcfm.experiments.exp_gaussian_eot --variants all --scale 2.0

Variants (see ``VARIANTS``):

``original``      Dirac source, absolute gradient penalty (lip_reg=1), alpha=0.
``fixed``         noise source, relative gradient penalty, bridge marginals, alpha=0.
``fixed_notilt``  as ``fixed`` but the coupling flow is re-trained without weights.
``particles``     as ``fixed`` with alpha=1 (particle transport only).
``fixed_mix``     as ``fixed`` with alpha=0.2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, replace
from typing import Dict, List

import torch

from dualwcfm.config import FeedForwardConfig, TrainingConfig, VisualizeConfig
from dualwcfm.distributions.coupling import FunctionalCouplingDistribution
from dualwcfm.distributions.distribution import Simple2DGaussianDistribution
from dualwcfm.flow_matching import ConditionalFlowMatching
from dualwcfm.methods.mixture import MixtureConfig, MixtureMethod
from dualwcfm.mlp import torch_wrapper
from dualwcfm.strategy import BidirectionalStrategy
from dualwcfm.trainer import Trainer


# ---------------------------------------------------------------------------
# Closed-form Gaussian EOT transfer
# ---------------------------------------------------------------------------

@dataclass
class GaussianEOTProblem:
    """Isotropic Gaussian reference plan and new isotropic Gaussian marginals."""
    dim: int = 2
    sx0: float = 1.0          # reference source std
    sy0: float = 1.0          # reference target std
    rho0: float = -0.8        # reference correlation (anticorrelation, as y ~ -x)
    shift: float = 4.0        # new marginals: mu_1 = N(+shift, ...), nu_1 = N(-shift, ...)
    scale: float = 2.0        # new marginal stds = scale * reference stds

    @property
    def c0(self) -> float:
        return self.rho0 * self.sx0 * self.sy0

    @property
    def q(self) -> float:
        """Cross coefficient of the precision matrix (invariant under tilts)."""
        return -self.c0 / (self.sx0 ** 2 * self.sy0 ** 2 - self.c0 ** 2)

    def rho_for(self, vx: float, vy: float) -> float:
        """Correlation of the EOT plan of the latent cost between N(., vx I) and N(., vy I)."""
        q = self.q
        D = (1.0 + math.sqrt(1.0 + 4.0 * vx * vy * q * q)) / (2.0 * vx * vy)
        return (-q / D) / math.sqrt(vx * vy)

    @property
    def rho_star(self) -> float:
        return self.rho_for((self.scale * self.sx0) ** 2, (self.scale * self.sy0) ** 2)

    # -- distributions ------------------------------------------------------

    def reference_coupling(self) -> FunctionalCouplingDistribution:
        d, sx, sy, r = self.dim, self.sx0, self.sy0, self.rho0

        def sample(n: int):
            x = torch.randn(n, d) * sx
            y = r * (sy / sx) * x + sy * math.sqrt(1.0 - r * r) * torch.randn(n, d)
            return x, y

        return FunctionalCouplingDistribution(sample)

    def new_marginals(self):
        m = torch.full((self.dim,), self.shift)
        src = Simple2DGaussianDistribution(mean=m, std=torch.full((self.dim,), self.scale * self.sx0))
        tgt = Simple2DGaussianDistribution(mean=-m, std=torch.full((self.dim,), self.scale * self.sy0))
        return src, tgt


def coupling_statistics(x: torch.Tensor, y: torch.Tensor, problem: GaussianEOTProblem) -> Dict[str, float]:
    """Moments, correlation and precision cross block of an empirical coupling (original coordinates)."""
    d = x.shape[1]
    z = torch.cat([x, y], dim=1).double()
    z = z - z.mean(0, keepdim=True)
    cov = z.T @ z / (z.shape[0] - 1)
    vx = cov[:d, :d].diagonal().mean().item()
    vy = cov[d:, d:].diagonal().mean().item()
    cxy = cov[:d, d:].diagonal().mean().item()
    prec = torch.linalg.inv(cov)
    q_hat = prec[:d, d:].diagonal().mean().item()
    return {
        "mean_x": x.mean().item(),
        "mean_y": y.mean().item(),
        "var_x": vx,
        "var_y": vy,
        "rho_hat": cxy / math.sqrt(vx * vy),
        "rho_own": problem.rho_for(vx, vy),
        "q_ratio": q_hat / problem.q,
    }


# ---------------------------------------------------------------------------
# Lightweight visualiser: records statistics only (no plots)
# ---------------------------------------------------------------------------

class GaussianStatsRecorder:
    """Implements the visualiser hooks used by Trainer/Methods; records Gaussian statistics."""

    def __init__(self, problem: GaussianEOTProblem, n_eval: int = 50_000):
        self.problem = problem
        self.n_eval = n_eval
        self.rows: List[Dict[str, float]] = []

    def _record(self, iteration: int, trainer) -> Dict[str, float]:
        x, y = trainer.current_coupling.coupling_sample(min(self.n_eval, trainer.current_coupling.size))
        s = trainer.scale
        stats = coupling_statistics(x.cpu() * s, y.cpu() * s, self.problem)
        t = (iteration + 1) / trainer.config.iterations if iteration >= 0 else 0.0
        p = self.problem
        lam = 1.0 + (p.scale - 1.0) * t          # std multiplier along the OT path
        stats.update({"iteration": iteration + 1, "t": t,
                      "rho_path": p.rho_for((lam * p.sx0) ** 2, (lam * p.sy0) ** 2)})
        self.rows.append(stats)
        return stats

    # hooks used by Trainer / Methods
    def analyse(self, iteration, trainer):
        st = self._record(iteration, trainer)
        return {"W1_mu": st["var_x"], "W2_mu": st["mean_x"], "W1_nu": st["var_y"], "W2_nu": st["mean_y"]}

    def at_flow_pretrain_end(self, trainer, ctx):
        self._record(-1, trainer)

    def at_flow_matching_start(self, iteration, trainer, ctx):
        return {}

    def transport_cost(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        # display, during_pretrain, at_pretrain_end, at_duals_end,
        # at_flow_matching_end, at_postprocess_end: no-ops
        if name.startswith("at_") or name in ("display", "during_pretrain"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def _base_method_cfg() -> MixtureConfig:
    return MixtureConfig(
        sigma=0.01, sigma2=0.01, loss="weighted",
        l2_reg=0.0, lip_reg=0.0, mean_reg=0.0,
        use_noisy_field=False, particle_ratio=0.0,
        lr_cosine_decay=False, lr_min_factor=0.5,
    )


VARIANTS = {
    # Original pipeline: Dirac source, absolute gradient penalty, independent-interpolation marginals.
    "original":     dict(source_std=0.0, lip_reg=1.0, l2_reg=1e-2, use_bridge_marginals=False, particle_ratio=0.0),
    # Fixes: noise source, relative penalty, marginals consistent with the bridges.
    "fixed":        dict(source_std=1.0, lip_reg=1e-2, lip_reg_relative=True, use_bridge_marginals=True, particle_ratio=0.0),
    "fixed_notilt": dict(source_std=1.0, lip_reg=1e-2, lip_reg_relative=True, use_bridge_marginals=True, particle_ratio=0.0, disable_tilt=True),
    "particles":    dict(source_std=1.0, lip_reg=1e-2, lip_reg_relative=True, use_bridge_marginals=True, particle_ratio=1.0),
    "fixed_mix":    dict(source_std=1.0, lip_reg=1e-2, lip_reg_relative=True, use_bridge_marginals=True, particle_ratio=0.2),
}


def make_configs(variant: str, device: torch.device, quick: bool = True):
    method_cfg = replace(_base_method_cfg(), **VARIANTS[variant])
    train_cfg = TrainingConfig(
        device=device,
        dataset_size=20_000 if quick else 200_000,
        marginal_epochs=1_500 if quick else 4_000,
        pretrain_epochs=3_000 if quick else 8_000,
        dual_epochs=150,
        cfm_epochs=300 if quick else 600,
        advance_steps=20,
        iterations=20 if quick else 50,
        scale_factor=10.0,
        batch_size=1024 if quick else 4096,
        save_final_models=False,
    )
    model_cfg = FeedForwardConfig(
        dimension=2, hidden_size=64, num_layers=2,
        flow_learning_rate=1e-3, nu_learning_rate=1e-3, mu_learning_rate=1e-3,
        dual_learning_rate=1e-3, dual_l2_reg=0.0,
    )
    return method_cfg, train_cfg, model_cfg


def run_variant(variant: str, problem: GaussianEOTProblem, out_dir: str, quick: bool = True, seed: int = 0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method_cfg, train_cfg, model_cfg = make_configs(variant, device, quick)
    src, tgt = problem.new_marginals()
    recorder = GaussianStatsRecorder(problem)
    strategy = BidirectionalStrategy(torch_wrapper, model_cfg)
    trainer = Trainer(
        initial_coupling=problem.reference_coupling(),
        new_source=src,
        new_target=tgt,
        flow_matching=ConditionalFlowMatching(),
        strategy=strategy,
        visualizer=recorder,
        config=train_cfg,
        method=MixtureMethod(method_cfg),
        device=device,
    )
    trainer.train(retrain=True, save=None)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{variant}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recorder.rows[0].keys()))
        w.writeheader()
        w.writerows(recorder.rows)
    return recorder.rows


def plot_summary(out_dir: str, problem: GaussianEOTProblem, names: List[str]) -> str:
    """Plot rho_hat and q_ratio along the iterations for the given variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    path_drawn = False
    for name in names:
        fname = os.path.join(out_dir, f"{name}.csv")
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        t = [r["t"] for r in rows]
        if not path_drawn:
            axes[0].plot(t, [r["rho_path"] for r in rows], "k--", lw=1.5, label=r"exact $\pi_t$")
            path_drawn = True
        axes[0].plot(t, [r["rho_hat"] for r in rows], marker="o", ms=3, label=name)
        axes[1].plot(t, [r["q_ratio"] for r in rows], marker="o", ms=3, label=name)
    axes[0].axhline(problem.rho0, color="grey", lw=0.8, ls=":")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("correlation of current coupling")
    axes[0].set_title(rf"$\rho_0={problem.rho0:+.2f}$, target $\rho^*={problem.rho_star:+.3f}$")
    axes[1].axhline(1.0, color="k", lw=1.0, ls="--")
    axes[1].set_xlabel("t"); axes[1].set_ylabel(r"$\hat q / q$ (latent cost carried)")
    axes[1].set_yscale("symlog", linthresh=1.0)
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "gaussian_eot_summary.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default="all", help="comma-separated list or 'all'")
    parser.add_argument("--scale", type=float, default=2.0, help="std multiplier of the new marginals")
    parser.add_argument("--shift", type=float, default=4.0)
    parser.add_argument("--rho0", type=float, default=-0.8)
    parser.add_argument("--full", action="store_true", help="larger budget (GPU)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/gaussian_eot")
    args = parser.parse_args()

    problem = GaussianEOTProblem(rho0=args.rho0, shift=args.shift, scale=args.scale)
    names = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    print(f"reference rho0 = {problem.rho0:+.4f}   target rho* = {problem.rho_star:+.4f}   q = {problem.q:.4f}")
    summary = {}
    for name in names:
        rows = run_variant(name, problem, args.output, quick=not args.full, seed=args.seed)
        last = rows[-1]
        summary[name] = {k: round(v, 4) for k, v in last.items()}
        print(f"[{name}] final: rho_hat={last['rho_hat']:+.4f} (rho*={problem.rho_star:+.4f}, rho_own={last['rho_own']:+.4f}) "
              f"q_ratio={last['q_ratio']:.3f} var_x={last['var_x']:.3f} var_y={last['var_y']:.3f} "
              f"mean_x={last['mean_x']:+.3f} mean_y={last['mean_y']:+.3f}")
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump({"rho0": problem.rho0, "rho_star": problem.rho_star, "q": problem.q, "variants": summary}, f, indent=2)
    print("figure:", plot_summary(args.output, problem, names))


if __name__ == "__main__":
    main()
