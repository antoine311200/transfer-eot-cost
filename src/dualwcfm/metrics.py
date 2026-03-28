"""dualwcfm.metrics
===================
Canonical, mode-aware metric implementations shared by the main experiment
logging (``logging_visualizer.py``) and the baselines package.

All functions accept raw ``torch.Tensor`` point clouds and return plain Python
floats so they can be safely written to CSV / sent to W&B.

Metric catalogue
----------------
Universal (2-D, 3-D, Gushchin, image):
  sliced_wasserstein(X, Y, n_proj, p)   — SW_p between two point clouds
  mmd(X, Y)                             — MMD with RBF kernel (median bandwidth)
  energy_distance(X, Y)                 — Energy distance (kernel-free)
  sinkhorn_divergence(X, Y, eps)        — Debiased Sinkhorn divergence

Transport-map quality (when ground-truth T is known):
  map_error(x_gen, y_gen, transport_fn) — RMSE of E[||T(x) - y||²]

Gaussian-specific (Gushchin benchmark):
  bures_wasserstein_2(X, Y, mu, Sigma)  — BW2 to a reference Gaussian
  mean_err(X, mu_ref)                   — L2 error of empirical mean
  cov_frob_err(X, Sigma_ref)            — Frobenius error of empirical covariance

Image-specific:
  fid(generated, reference)             — Fréchet Inception Distance
  kid(generated, reference)             — Kernel Inception Distance
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Tuple

import torch

logger = logging.getLogger("dualwcfm")

# ---------------------------------------------------------------------------
# Universal metrics
# ---------------------------------------------------------------------------

def sliced_wasserstein(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_proj: int = 500,
    p: int = 2,
) -> float:
    """Sliced Wasserstein distance SW_p between point clouds X and Y.

    Tensors are flattened to 2-D before projection so image tensors work too.
    Handles unequal sample sizes via linear interpolation of sorted projections.
    """
    X = X.flatten(1).float()
    Y = Y.flatten(1).float()
    d = X.shape[1]
    dirs = torch.randn(d, n_proj, device=X.device, dtype=X.dtype)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp(min=1e-8)
    Xp = (X @ dirs).sort(dim=0).values  # (N, n_proj)
    Yp = (Y @ dirs).sort(dim=0).values  # (M, n_proj)
    n, m = Xp.shape[0], Yp.shape[0]
    if n != m:
        k = min(n, m)
        Xp = Xp[torch.linspace(0, n - 1, k, device=X.device).long()]
        Yp = Yp[torch.linspace(0, m - 1, k, device=Y.device).long()]
    diff = (Xp - Yp).abs()
    return diff.mean().item() if p == 1 else diff.pow(2).mean().sqrt().item()


def mmd(
    X: torch.Tensor,
    Y: torch.Tensor,
    max_samples: int = 1_000,
) -> float:
    """Maximum Mean Discrepancy with RBF kernel (median bandwidth heuristic).

    MMD²(X,Y) = E[k(X,X')] - 2·E[k(X,Y)] + E[k(Y,Y')]

    Parameters
    ----------
    X, Y:
        Point clouds to compare (e.g. generated target vs. true target).
    max_samples:
        Cap on samples used (O(N²) cost).
    """
    X = X.flatten(1).float()
    Y = Y.flatten(1).float()
    # Sub-sample for efficiency
    if X.shape[0] > max_samples:
        X = X[torch.randperm(X.shape[0], device=X.device)[:max_samples]]
    if Y.shape[0] > max_samples:
        Y = Y[torch.randperm(Y.shape[0], device=Y.device)[:max_samples]]

    # Median bandwidth heuristic on the combined set
    Z = torch.cat([X, Y], dim=0)
    with torch.no_grad():
        dists = torch.cdist(Z, Z)
        sigma2 = dists[dists > 0].median().item() ** 2
        if sigma2 < 1e-10:
            return 0.0

    def _rbf(A, B):
        d2 = torch.cdist(A, B).pow(2)
        return torch.exp(-d2 / (2.0 * sigma2))

    with torch.no_grad():
        kxx = _rbf(X, X).mean()
        kyy = _rbf(Y, Y).mean()
        kxy = _rbf(X, Y).mean()
    return float((kxx + kyy - 2.0 * kxy).clamp(min=0.0).sqrt().item())


def energy_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    max_samples: int = 1_000,
) -> float:
    """Energy distance between two point clouds — kernel-free, statistically consistent.

    E(X,Y) = 2·E[‖X−Y‖] − E[‖X−X'‖] − E[‖Y−Y'‖]

    Unlike MMD, requires no kernel bandwidth choice and is always non-negative.

    Parameters
    ----------
    X, Y:
        Point clouds to compare.
    max_samples:
        Cap on samples used (O(N²) cost).
    """
    X = X.flatten(1).float()
    Y = Y.flatten(1).float()
    if X.shape[0] > max_samples:
        X = X[torch.randperm(X.shape[0], device=X.device)[:max_samples]]
    if Y.shape[0] > max_samples:
        Y = Y[torch.randperm(Y.shape[0], device=Y.device)[:max_samples]]

    with torch.no_grad():
        e_xy = torch.cdist(X, Y).mean()
        e_xx = torch.cdist(X, X).mean()
        e_yy = torch.cdist(Y, Y).mean()
    return float((2.0 * e_xy - e_xx - e_yy).clamp(min=0.0).item())


def sinkhorn_divergence(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 0.1,
    n_iter: int = 50,
    max_samples: int = 500,
) -> float:
    """Debiased Sinkhorn divergence — a smooth proxy for the W₂ OT cost.

    S_ε(X,Y) = OT_ε(X,Y) − ½·OT_ε(X,X) − ½·OT_ε(Y,Y)

    Debiasing removes the self-entropy term so S_ε(X,X) = 0.
    Unlike the raw L2 transport cost this metric does NOT penalise
    stochastic plans — it purely measures distributional discrepancy.

    Parameters
    ----------
    X, Y:
        Point clouds.
    eps:
        Entropic regularisation.  Larger values = smoother but more biased.
    n_iter:
        Sinkhorn iterations.
    max_samples:
        Sub-sample cap (O(N²·n_iter) cost).
    """
    X = X.flatten(1).float()
    Y = Y.flatten(1).float()
    if X.shape[0] > max_samples:
        X = X[torch.randperm(X.shape[0], device=X.device)[:max_samples]]
    if Y.shape[0] > max_samples:
        Y = Y[torch.randperm(Y.shape[0], device=Y.device)[:max_samples]]

    def _ot_eps(A, B):
        N, M = len(A), len(B)
        a = torch.full((N,), 1.0 / N, device=A.device)
        b = torch.full((M,), 1.0 / M, device=B.device)
        log_K = -torch.cdist(A, B).pow(2) / eps
        u = torch.zeros(N, device=A.device)
        for _ in range(n_iter):
            v = b.log() - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)
            u = a.log() - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
        log_P = log_K + u.unsqueeze(1) + v.unsqueeze(0)
        C = torch.cdist(A, B).pow(2)
        return (log_P.exp() * C).sum()

    with torch.no_grad():
        ot_xy = _ot_eps(X, Y)
        ot_xx = _ot_eps(X, X)
        ot_yy = _ot_eps(Y, Y)
    return float((ot_xy - 0.5 * ot_xx - 0.5 * ot_yy).clamp(min=0.0).item())


# ---------------------------------------------------------------------------
# Transport-map quality (requires knowing the ground-truth map T)
# ---------------------------------------------------------------------------

def map_error(
    x_gen: torch.Tensor,
    y_gen: torch.Tensor,
    transport_fn: Callable,
) -> float:
    """Ground-truth transport map RMSE: sqrt(E[||T(x) − y||²]).

    Measures how accurately the model transports each source sample to the
    location prescribed by the true map T.  Unlike marginal metrics (SW, MMD),
    this is a per-sample quality measure that rewards deterministic, low-cost
    transport.

    Parameters
    ----------
    x_gen:
        Source samples used as input (same pairing as y_gen).
    y_gen:
        Transported target samples from the model.
    transport_fn:
        Callable T: ℝ^d → ℝ^d (or image tensors).  Called on CPU float32
        tensors; must return a tensor of the same shape.

    Returns
    -------
    RMSE (float).
    """
    x_cpu = x_gen.float().cpu()
    y_cpu = y_gen.float().cpu()
    with torch.no_grad():
        T_x = transport_fn(x_cpu)
        err = (T_x.flatten(1) - y_cpu.flatten(1)).pow(2).sum(dim=-1).mean()
    return float(err.sqrt().item())


# ---------------------------------------------------------------------------
# Gaussian-specific metrics (Gushchin benchmark)
# ---------------------------------------------------------------------------

def _matrix_sqrt(A: torch.Tensor) -> torch.Tensor:
    """Symmetric PSD matrix square root via eigendecomposition."""
    L, V = torch.linalg.eigh(A)
    L = L.clamp(min=0.0)
    return V @ torch.diag(L.sqrt()) @ V.T


def bures_wasserstein_2(
    X: torch.Tensor,
    Y: torch.Tensor,
    mu_ref: Optional[torch.Tensor] = None,
    Sigma_ref: Optional[torch.Tensor] = None,
) -> float:
    """BW₂ distance between the empirical distribution of Y and a reference Gaussian.

    When ``mu_ref`` / ``Sigma_ref`` are *None*, they are estimated from Y itself
    and compared against X, computing BW₂ between the Gaussian fits to X and Y.

    BW₂²(N(m1,C1), N(m2,C2)) = ‖m1−m2‖² + tr(C1) + tr(C2) − 2·tr((C1^{1/2}·C2·C1^{1/2})^{1/2})

    Parameters
    ----------
    X:
        Generated samples (from the model/baseline).
    Y:
        Reference samples OR a draw from the ground-truth plan.
        When mu_ref/Sigma_ref are provided, Y is compared against that Gaussian.
    mu_ref, Sigma_ref:
        Known reference mean / covariance.  Supply for Gushchin benchmark where
        the ground-truth EOT plan is analytically known.
    """
    X = X.flatten(1).float()
    Y = Y.flatten(1).float()

    m1 = X.mean(0)
    C1 = _empirical_cov(X)

    if mu_ref is not None and Sigma_ref is not None:
        m2 = mu_ref.to(X.device).float()
        C2 = Sigma_ref.to(X.device).float()
    else:
        m2 = Y.mean(0)
        C2 = _empirical_cov(Y)

    mean_term = (m1 - m2).pow(2).sum()
    C1_sqrt    = _matrix_sqrt(C1)
    M          = _matrix_sqrt(C1_sqrt @ C2 @ C1_sqrt)
    cov_term   = C1.trace() + C2.trace() - 2.0 * M.trace()
    return float((mean_term + cov_term).clamp(min=0.0).sqrt().item())


def mean_err(X: torch.Tensor, mu_ref: torch.Tensor) -> float:
    """L2 distance between empirical mean of X and reference mean mu_ref."""
    return float((X.float().mean(0) - mu_ref.to(X.device).float()).norm().item())


def cov_frob_err(X: torch.Tensor, Sigma_ref: torch.Tensor) -> float:
    """Frobenius norm between empirical covariance of X and reference Sigma_ref."""
    C = _empirical_cov(X.float())
    S = Sigma_ref.to(X.device).float()
    return float((C - S).norm("fro").item())


def _empirical_cov(X: torch.Tensor) -> torch.Tensor:
    X_c = X - X.mean(0, keepdim=True)
    return (X_c.T @ X_c) / max(len(X) - 1, 1)


# ---------------------------------------------------------------------------
# Image-specific metrics
# ---------------------------------------------------------------------------

def fid(
    generated: torch.Tensor,
    reference: torch.Tensor,
    device: Optional[torch.device] = None,
) -> float:
    """Fréchet Inception Distance between generated and reference image tensors.

    Requires ``torchmetrics``.  Falls back to a pixel-statistics FID
    (FID on raw pixel vectors) if torchmetrics / InceptionV3 is unavailable.

    Parameters
    ----------
    generated, reference:
        Image tensors of shape (N, C, H, W), values in [-1, 1] or [0, 1].
    """
    device = device or generated.device
    generated  = _to_01(generated).to(device)
    reference  = _to_01(reference).to(device)

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        metric = FrechetInceptionDistance(normalize=True).to(device)
        metric.update(reference,  real=True)
        metric.update(generated,  real=False)
        return float(metric.compute().item())
    except Exception:
        pass

    # Pixel-statistics fallback (Gaussian fit on flattened pixels)
    return float(bures_wasserstein_2(generated.flatten(1), reference.flatten(1)))


def kid(
    generated: torch.Tensor,
    reference: torch.Tensor,
    device: Optional[torch.device] = None,
    subset_size: int = 1_000,
) -> float:
    """Kernel Inception Distance — more statistically efficient than FID.

    Requires ``torchmetrics >= 0.11``.  Falls back gracefully to MMD on
    raw pixels when torchmetrics / InceptionV3 is unavailable.

    Parameters
    ----------
    generated, reference:
        Image tensors (N, C, H, W), values in [-1, 1] or [0, 1].
    subset_size:
        KID sub-sampling size for the unbiased estimator.
    """
    device     = device or generated.device
    generated  = _to_01(generated).to(device)
    reference  = _to_01(reference).to(device)

    try:
        from torchmetrics.image.kid import KernelInceptionDistance
        metric = KernelInceptionDistance(
            normalize=True, subset_size=min(subset_size, len(generated), len(reference))
        ).to(device)
        metric.update(reference, real=True)
        metric.update(generated, real=False)
        kid_mean, _ = metric.compute()
        return float(kid_mean.item())
    except Exception:
        pass

    return mmd(generated.flatten(1), reference.flatten(1))


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Shift from [-1,1] to [0,1] if needed; ensure float32 and 3-channel."""
    x = x.float()
    if x.min() < -0.01:                 # assume [-1, 1] normalisation
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    if x.shape[1] == 1:                 # grayscale → RGB
        x = x.expand(-1, 3, -1, -1)
    return x


# ---------------------------------------------------------------------------
# Joint coupling metric (Gushchin BW2)
# ---------------------------------------------------------------------------

def joint_bw2(
    x: torch.Tensor,
    y: torch.Tensor,
    x_ref: torch.Tensor,
    y_ref: torch.Tensor,
) -> float:
    """Bures–Wasserstein² distance between two joint coupling distributions.

    Fits Gaussians to (x‖y) and (x_ref‖y_ref) then evaluates BW₂ analytically.
    This is the standard Gushchin benchmark metric for coupling quality.

    Parameters
    ----------
    x, y     : learned coupling samples (N, d) each
    x_ref, y_ref : reference coupling samples (M, d) each

    Returns
    -------
    BW₂ distance (float)
    """
    xy     = torch.cat([x.flatten(1), y.flatten(1)], dim=1).float()
    xy_ref = torch.cat([x_ref.flatten(1), y_ref.flatten(1)], dim=1).float()
    return bures_wasserstein_2(xy, xy_ref)


# ---------------------------------------------------------------------------
# Convenience: compute all universal metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_gen: torch.Tensor,
    y_ref: torch.Tensor,
    x_gen: Optional[torch.Tensor] = None,
    x_ref: Optional[torch.Tensor] = None,
    n_proj: int = 500,
    prefix: str = "",
) -> dict:
    """Compute the full universal metric suite comparing y_gen vs y_ref.

    Optionally computes source-marginal metrics when x_gen / x_ref are given.

    Parameters
    ----------
    y_gen:
        Transported target samples from the model.
    y_ref:
        True target samples drawn from nu'.
    x_gen:
        Source samples used as input to the model (same pairing as y_gen).
    x_ref:
        True source samples drawn from mu'.
    n_proj:
        Number of random projections for SW.
    prefix:
        String prepended to all metric keys (e.g. ``"nu_"``).

    Returns
    -------
    dict mapping metric name -> float.
    """
    p = prefix
    metrics: dict = {}

    # --- Target marginal quality ------------------------------------------
    metrics[f"{p}W1_nu"]          = sliced_wasserstein(y_gen, y_ref, n_proj=n_proj, p=1)
    metrics[f"{p}W2_nu"]          = sliced_wasserstein(y_gen, y_ref, n_proj=n_proj, p=2)
    metrics[f"{p}mmd_nu"]         = mmd(y_gen, y_ref)
    metrics[f"{p}energy_dist_nu"] = energy_distance(y_gen, y_ref)
    metrics[f"{p}sinkhorn_div"]   = sinkhorn_divergence(y_gen, y_ref)

    # --- Source marginal quality (optional) --------------------------------
    if x_gen is not None and x_ref is not None:
        metrics[f"{p}W1_mu"] = sliced_wasserstein(x_gen, x_ref, n_proj=n_proj, p=1)
        metrics[f"{p}W2_mu"] = sliced_wasserstein(x_gen, x_ref, n_proj=n_proj, p=2)

    return metrics
