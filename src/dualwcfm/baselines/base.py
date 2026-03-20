"""dualwcfm.baselines.base
==========================
Shared infrastructure for all baselines: spec, mode-aware metric evaluation,
CSV/W&B logging, and the abstract BaselineRunner.

Metric sets by mode
-------------------
2d / 3d / gushchin (low-dimensional):
    W1_mu, W2_mu, W1_nu, W2_nu   — Sliced-Wasserstein (source & target)
    mmd_nu                        — Maximum Mean Discrepancy (RBF, median BW)
    energy_dist_nu                — Energy distance (kernel-free)
    sinkhorn_div                  — Debiased Sinkhorn divergence
    bw2                           — Bures-Wasserstein₂ (Gushchin only, when ref given)
    mean_err_nu                   — L2 mean error (Gushchin only)
    cov_frob_nu                   — Frobenius covariance error (Gushchin only)

image / color:
    W1_nu, W2_nu                  — Sliced-Wasserstein (pixel space)
    mmd_nu                        — MMD (pixel space)
    fid                           — Fréchet Inception Distance
    kid                           — Kernel Inception Distance
"""
from __future__ import annotations

import csv
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from dualwcfm import metrics as M

logger = logging.getLogger("dualwcfm")


# ---------------------------------------------------------------------------
# BaselineSpec
# ---------------------------------------------------------------------------

@dataclass
class BaselineSpec:
    """Complete specification for one baseline evaluation run.

    Attributes
    ----------
    name:
        Unique identifier used as the W&B run name and CSV row label.
    mode:
        ``"2d"``, ``"3d"``, ``"gushchin"``, ``"image"``, or ``"color"``.
    source:
        Distribution for mu' (new source marginal).
    target:
        Distribution for nu' (new target marginal).
    dim:
        Flattened sample dimension.
    output_dir:
        Per-run artefact directory (model weights, per-step CSV).
    device:
        Torch device.
    # Gushchin-specific (optional)
    ref_mean:
        Ground-truth mean of the target Gaussian (for BW₂).
    ref_cov:
        Ground-truth covariance of the target Gaussian (for BW₂).
    """
    name:       str
    mode:       str
    source:     Any
    target:     Any
    dim:        int
    output_dir: str
    device:     torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # Optional: Gushchin ground-truth for BW₂
    ref_mean:      Optional[torch.Tensor] = None
    ref_cov:       Optional[torch.Tensor] = None
    # Optional: ground-truth forward transport map T s.t. y ≈ T(x)
    transport_map: Optional[Callable]     = None


# ---------------------------------------------------------------------------
# Mode-aware metric evaluation
# ---------------------------------------------------------------------------

_LOW_DIM_MODES  = {"2d", "3d", "gushchin"}
_IMAGE_MODES    = {"image", "color"}
_GUSHCHIN_MODES = {"gushchin"}

# Columns written to baselines_summary.csv
_SUMMARY_FIELDS = [
    "experiment", "baseline",
    # Universal
    "W1_mu", "W2_mu", "W1_nu", "W2_nu",
    "mmd_nu", "energy_dist_nu", "sinkhorn_div",
    # Ground-truth transport map error (when T is known)
    "map_err_fwd",
    # Gushchin
    "bw2", "mean_err_nu", "cov_frob_nu",
    # Image
    "fid", "kid",
]


def evaluate_metrics(
    spec:    BaselineSpec,
    x_gen:   torch.Tensor,
    y_gen:   torch.Tensor,
    n_ref:   int = 2_000,
    n_proj:  int = 500,
) -> Dict[str, float]:
    """Compute all applicable metrics for *spec.mode*.

    Parameters
    ----------
    spec:
        Baseline specification (source/target distributions, mode, device).
    x_gen:
        Source samples used as input (same pairing as y_gen).
    y_gen:
        Transported target samples from the model.
    n_ref:
        Number of reference samples drawn from the true marginals.
    n_proj:
        Number of random projections for sliced-Wasserstein.
    """
    device = spec.device
    result: Dict[str, float] = {}

    # ── Reference samples from the true marginals ────────────────────────
    x_ref = _sample(spec.source, n_ref, device)
    y_ref = _sample(spec.target, n_ref, device)

    n = min(len(x_gen), n_ref)
    x_g = x_gen[:n]
    y_g = y_gen[:n]

    # ── Universal: SW on both marginals ─────────────────────────────────
    result["W1_mu"] = M.sliced_wasserstein(x_g, x_ref, n_proj=n_proj, p=1)
    result["W2_mu"] = M.sliced_wasserstein(x_g, x_ref, n_proj=n_proj, p=2)
    result["W1_nu"] = M.sliced_wasserstein(y_g, y_ref, n_proj=n_proj, p=1)
    result["W2_nu"] = M.sliced_wasserstein(y_g, y_ref, n_proj=n_proj, p=2)

    # ── Ground-truth transport map error (when T is known) ──────────────
    if spec.transport_map is not None:
        result["map_err_fwd"] = M.map_error(x_g, y_g, spec.transport_map)

    # ── Low-dimensional metrics ──────────────────────────────────────────
    if spec.mode in _LOW_DIM_MODES:
        result["mmd_nu"]         = M.mmd(y_g, y_ref)
        result["energy_dist_nu"] = M.energy_distance(y_g, y_ref)
        result["sinkhorn_div"]   = M.sinkhorn_divergence(y_g, y_ref)

    # ── Gushchin: Gaussian-specific ─────────────────────────────────────
    if spec.mode in _GUSHCHIN_MODES:
        result["bw2"] = M.bures_wasserstein_2(
            y_g, y_ref,
            mu_ref    = spec.ref_mean,
            Sigma_ref = spec.ref_cov,
        )
        result["mean_err_nu"] = (
            M.mean_err(y_g, spec.ref_mean) if spec.ref_mean is not None
            else M.mean_err(y_g, y_ref.mean(0))
        )
        result["cov_frob_nu"] = (
            M.cov_frob_err(y_g, spec.ref_cov) if spec.ref_cov is not None
            else M.cov_frob_err(y_g, M._empirical_cov(y_ref.float()))
        )

    # ── Image-specific ───────────────────────────────────────────────────
    if spec.mode in _IMAGE_MODES:
        result["mmd_nu"] = M.mmd(y_g, y_ref)
        # FID and KID need enough samples; skip silently if too few
        if n >= 100:
            result["fid"] = M.fid(y_g, y_ref, device=device)
            result["kid"] = M.kid(y_g, y_ref, device=device)

    return result


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def append_summary_csv(output_root: str, row: Dict[str, Any]) -> None:
    """Append one result row to ``baselines_summary.csv`` in *output_root*."""
    os.makedirs(output_root, exist_ok=True)
    path = os.path.join(output_root, "baselines_summary.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in _SUMMARY_FIELDS})


def _write_run_csv(output_dir: str, metrics: Dict[str, Any]) -> None:
    path = os.path.join(output_dir, "result.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


# ---------------------------------------------------------------------------
# Abstract BaselineRunner
# ---------------------------------------------------------------------------

class BaselineRunner(ABC):
    """Abstract base class for all from-scratch baselines.

    Subclasses implement :meth:`train` and :meth:`sample`.
    :meth:`evaluate` and :meth:`run` are fully provided here.
    """

    def __init__(self, spec: BaselineSpec, wandb_run=None) -> None:
        self.spec   = spec
        self._wandb = wandb_run or _NoOpRun()
        os.makedirs(spec.output_dir, exist_ok=True)

    # ------------------------------------------------------------------

    @abstractmethod
    def train(self) -> None:
        """Train (or load) the baseline model."""

    @abstractmethod
    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (x, T(x)) pairs from the trained model."""

    # ------------------------------------------------------------------

    def evaluate(
        self,
        n_eval:      int = 4_000,
        output_root: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute all metrics, write to CSV and W&B, return dict."""
        x, y = self.sample(n_eval)
        metrics = evaluate_metrics(self.spec, x, y)
        metrics["baseline"]   = self.spec.name
        metrics["experiment"] = os.path.basename(os.path.dirname(self.spec.output_dir))

        _log_metrics(self.spec.name, metrics)
        _write_run_csv(self.spec.output_dir, metrics)

        root = output_root or os.path.dirname(os.path.dirname(self.spec.output_dir))
        append_summary_csv(root, metrics)

        self._wandb.log(metrics)
        self._wandb.summary(metrics)
        return metrics

    def run(self, output_root: Optional[str] = None) -> Dict[str, float]:
        """Full pipeline: train → evaluate."""
        logger.info("\n  Baseline : %s  (mode=%s)", self.spec.name, self.spec.mode)
        self.train()
        return self.evaluate(output_root=output_root)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample(dist, n: int, device: torch.device) -> torch.Tensor:
    x = dist.sample(n)
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    return x.float().to(device)


def _log_metrics(name: str, m: Dict[str, float]) -> None:
    parts = []
    for key in ("W2_nu", "mmd_nu", "energy_dist_nu", "sinkhorn_div", "bw2", "fid"):
        if key in m and m[key] != "":
            parts.append(f"{key}={float(m[key]):.4f}")
    logger.info("  [%s] %s", name, "  ".join(parts))


class _NoOpRun:
    def log(self, *a, **k):     pass
    def summary(self, *a, **k): pass
    def finish(self):           pass
