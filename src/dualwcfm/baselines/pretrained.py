"""dualwcfm.baselines.pretrained
================================
"Pretrained-only" baseline: apply the DualWCFM pretrained forward model
to new marginals (mu', nu') *without* running the dual-potential adaptation loop.

This isolates the contribution of the adaptation phase.  If pretrained-only
already matches DualWCFM quality, adaptation adds no value.  If it fails,
adaptation is essential — providing a clear ablation story for the paper.

How it works
------------
After the DualWCFM pretrain phase the strategy holds:

  ``initial_forward_model.pth``  — CFM flow trained on the reference coupling pi_0
  ``initial_mu_model.pth``       — bridge from mu_0 to mu' (source marginal flow)
  ``initial_backward_model.pth`` — same for backward direction

At inference, given x ~ mu':
  1. Integrate the mu_bridge forward from x to get a "landing" point x_mu.
  2. Draw y_cond ~ nu_0 (reference target marginal, independent of x).
  3. Apply the forward model conditioned on y_cond: integrate v(t, x, y_cond) to get y.
  4. Measure SW(y, nu').

The gap between this and full DualWCFM quantifies what the dual-potential
adaptation phase contributes.

Usage
-----
Requires a prior DualWCFM run with ``save_models=True`` (the default) on the
same experiment spec.  The weights are read from ``{output_dir}/models/``.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch

from dualwcfm.baselines.base import BaselineRunner, BaselineSpec

logger = logging.getLogger("dualwcfm")


class PretrainedOnlyRunner(BaselineRunner):
    """Apply a pretrained DualWCFM forward model to new marginals without adaptation.

    Parameters
    ----------
    spec:
        Baseline specification (source/target distributions, device, …).
    pretrain_dir:
        Path to the ``models/`` directory produced by a DualWCFM run,
        e.g. ``results/2d_simple/models/``.  Must contain at minimum
        ``initial_forward_model.pth`` and ``initial_mu_model.pth``.
    ref_target:
        Distribution for the reference target nu_0 (used as the conditioning
        signal for the forward model).  When ``None``, the forward model is
        run unconditionally — i.e. the condition channel is set to zeros.
    hidden_size:
        Hidden size of the MLP used to reconstruct the model architecture.
        Must match the value used in the original DualWCFM run.
    euler_steps:
        Number of Euler integration steps for ODE solving.
    wandb_run:
        Optional W&B run.
    """

    def __init__(
        self,
        spec:         BaselineSpec,
        pretrain_dir: str,
        ref_target=None,
        hidden_size:  int = 64,
        euler_steps:  int = 100,
        wandb_run=None,
    ) -> None:
        super().__init__(spec, wandb_run)
        self.pretrain_dir = pretrain_dir
        self.ref_target   = ref_target
        self.hidden_size  = hidden_size
        self.euler_steps  = euler_steps
        self._strategy    = None

    # ------------------------------------------------------------------

    def train(self) -> None:
        """Load pretrained weights (no gradient steps required)."""
        fwd_path = os.path.join(self.pretrain_dir, "initial_forward_model.pth")
        mu_path  = os.path.join(self.pretrain_dir, "initial_mu_model.pth")

        for p in (fwd_path,):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Pretrained weight not found: {p}\n"
                    f"Run DualWCFM with save_models=True on the same experiment first."
                )

        # Reconstruct the MLP strategy (architecture only — weights loaded below)
        from dualwcfm.config import FeedForwardConfig
        from dualwcfm.strategy import BidirectionalStrategy
        from dualwcfm.mlp import torch_wrapper

        cfg      = FeedForwardConfig(dimension=self.spec.dim, hidden_size=self.hidden_size)
        strategy = BidirectionalStrategy(torch_wrapper, cfg)
        strategy.to_device(self.spec.device)

        strategy.initial_forward_model.load_state_dict(
            torch.load(fwd_path, map_location=self.spec.device)
        )
        if os.path.exists(mu_path):
            strategy.initial_mu_model.load_state_dict(
                torch.load(mu_path, map_location=self.spec.device)
            )
            logger.info("  [Pretrained-only] Loaded mu_bridge from %s", mu_path)
        else:
            logger.warning("  [Pretrained-only] mu_bridge not found; will use identity.")

        self._strategy = strategy
        logger.info("  [Pretrained-only] Loaded forward model from %s", fwd_path)

    # ------------------------------------------------------------------

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._strategy is not None, "Call train() first."
        device   = self.spec.device
        strategy = self._strategy
        dt       = 1.0 / self.euler_steps

        x0 = _to_tensor(self.spec.source.sample(n), device)

        with torch.no_grad():
            # Step 1 — integrate mu_bridge to get the adapted source sample
            x_mu = x0.clone()
            for i in range(self.euler_steps):
                t = torch.full((n,), i * dt, device=device)
                x_mu = x_mu + strategy.initial_mu_model(t, x_mu) * dt

            # Step 2 — conditioning signal from reference target nu_0
            if self.ref_target is not None:
                y_cond = _to_tensor(self.ref_target.sample(n), device)
            else:
                # No reference target available — condition on zeros
                y_cond = torch.zeros_like(x0)

            # Step 3 — integrate forward model conditioned on y_cond
            x = x_mu.clone()
            for i in range(self.euler_steps):
                t = torch.full((n,), i * dt, device=device)
                x = x + strategy.initial_forward_model(t, x, y_cond) * dt

        return x0, x


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_tensor(x, device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    return x.float().to(device)
