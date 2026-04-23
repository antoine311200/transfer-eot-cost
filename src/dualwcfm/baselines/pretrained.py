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

At inference, given x ~ mu':
  1. Extend x → cat([x, x]) to form the doubled input expected by the model
     (the second half is frozen as the "start" condition throughout integration).
  2. Run Euler integration with the forward model.
  3. Take the first ``dim`` components of the output as the transported sample.

This matches exactly how the visualizer evaluates the pretrained model before
adaptation starts (``visualization.py:Visualizer.display_pretrained``).

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
        e.g. ``results/2d_simple/models/``.  Must contain
        ``initial_forward_model.pth``.
    hidden_size:
        Hidden size of the MLP used to reconstruct the model architecture.
        Must match the value used in the original DualWCFM run.
    num_layers:
        Number of hidden layers.  Inferred from the checkpoint when called
        via ``compute_pretrained_results``; otherwise must match training.
    euler_steps:
        Number of Euler integration steps for ODE solving.
    wandb_run:
        Optional W&B run.
    """

    def __init__(
        self,
        spec:         BaselineSpec,
        pretrain_dir: str,
        ref_target=None,          # kept for API compat, unused
        hidden_size:  int = 64,
        num_layers:   int = 2,
        euler_steps:  int = 100,
        wandb_run=None,
    ) -> None:
        super().__init__(spec, wandb_run)
        self.pretrain_dir = pretrain_dir
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.euler_steps  = euler_steps
        self._fwd_model   = None

    # ------------------------------------------------------------------

    def train(self) -> None:
        """Load pretrained forward model weights (no gradient steps)."""
        fwd_path = os.path.join(self.pretrain_dir, "initial_forward_model.pth")
        if not os.path.exists(fwd_path):
            raise FileNotFoundError(
                f"Pretrained weight not found: {fwd_path}\n"
                f"Run DualWCFM with save_models=True on the same experiment first."
            )

        from dualwcfm.config import FeedForwardConfig
        from dualwcfm.strategy import BidirectionalStrategy
        from dualwcfm.mlp import torch_wrapper

        cfg      = FeedForwardConfig(dimension=self.spec.dim, hidden_size=self.hidden_size,
                                     num_layers=self.num_layers)
        strategy = BidirectionalStrategy(torch_wrapper, cfg)
        strategy.to_device(self.spec.device)
        strategy.initial_forward_model.load_state_dict(
            torch.load(fwd_path, map_location=self.spec.device)
        )
        self._fwd_model = strategy.initial_forward_model
        logger.info("  [Pretrained-only] Loaded forward model from %s", fwd_path)

    # ------------------------------------------------------------------

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transport n samples from new_source using the pretrained forward model.

        Matches the canonical inference in ``Visualizer.display_pretrained``:
          evolved = flow_matching.evolve(x0, fwd_model, extend=True)
          y_hat   = evolved[:, :dim]
        """
        assert self._fwd_model is not None, "Call train() first."
        device = self.spec.device
        dim    = self.spec.dim

        from dualwcfm.flow_matching import ConditionalFlowMatching
        fm = ConditionalFlowMatching()

        x0     = _to_tensor(self.spec.source.sample(n), device)
        self._fwd_model.eval()
        evolved = fm.evolve(x0, self._fwd_model,
                            num_steps=self.euler_steps, sigma=0.0, extend=True)
        y_hat = evolved[:, :dim]
        return x0, y_hat


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_tensor(x, device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    return x.float().to(device)
