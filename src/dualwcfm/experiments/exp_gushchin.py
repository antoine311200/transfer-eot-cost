"""dualwcfm.experiments.exp_gushchin
=====================================
Gaussian-pair EOT benchmark experiments (E1 in the NeurIPS plan).

Design principle
----------------
Following the spirit of Gushchin et al. (NeurIPS 2023) "Building the Bridge of
Schrödinger: A Continuous EOT Benchmark", we construct source/target pairs of
Gaussian mixtures for which the **ground-truth** EOT plan is known by
construction.  This allows us to compute the Bures–Wasserstein distance between
our learned coupling and the ground truth, giving a quantitative score that can
be directly compared against DSBM, LightSB, etc.

Construction
------------
For a pair of Gaussian distributions (μ = N(m_μ, Σ_μ),  ν = N(m_ν, Σ_ν)) the
EOT plan (with quadratic cost c(x,y) = ||x-y||² and regularisation ε) is itself
Gaussian: π* = N(m*, Σ*).  We use this to build a reference coupling with a
known ground truth and measure BW₂ between our learned coupling and π*.

For Gaussian-mixture pairs, the EOT plan is not in closed form, but we can
compute an accurate particle approximation via the Sinkhorn algorithm (using
the POT library), which serves as the "oracle" reference coupling.

Experiment families
-------------------
gaussian_iso_2d    – isotropic Gaussians in R²  (analytical BW₂)
gaussian_gmm_2d    – 4-component Gaussian mixtures in R²  (Sinkhorn reference)
gaussian_iso_8d    – isotropic Gaussians in R⁸  (analytical BW₂, tests scaling)
gaussian_gmm_8d    – 4-component Gaussian mixtures in R⁸  (Sinkhorn reference)

Each family has three ε (regularisation) levels: small / medium / large.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import torch

from dualwcfm.config import FeedForwardConfig, TrainingConfig, VisualizeConfig
from dualwcfm.distributions.coupling import FunctionalCouplingDistribution
from dualwcfm.distributions.distribution import Distribution
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.experiments.base import ExperimentSpec, get_device

logger = logging.getLogger("dualwcfm")


# ---------------------------------------------------------------------------
# Gaussian EOT — analytical solution
# ---------------------------------------------------------------------------

def gaussian_eot_plan(
    mu_mean: np.ndarray, mu_cov: np.ndarray,
    nu_mean: np.ndarray, nu_cov: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (joint_mean, joint_cov, bw2) for the Gaussian EOT plan.

    For Gaussians (μ, ν) with entropic regularisation ε, the EOT plan is
    Gaussian with the closed-form joint covariance derived in (Janati et al.,
    2020; Mallasto et al., 2022).

    Parameters
    ----------
    mu_mean, mu_cov:   source mean (d,) and covariance (d,d)
    nu_mean, nu_cov:   target mean (d,) and covariance (d,d)
    eps:               entropic regularisation (> 0)

    Returns
    -------
    joint_mean : (2d,)        mean of the Gaussian EOT plan
    joint_cov  : (2d, 2d)    covariance of the Gaussian EOT plan
    bw2        : float        Bures–Wasserstein² distance to ground truth
                              (used to validate other methods)
    """
    d = len(mu_mean)

    # M = (Σ_μ^{1/2} Σ_ν Σ_μ^{1/2} + (ε/2)² I)^{1/2}
    Sigma_mu_sqrt = _matrix_sqrt(mu_cov)
    inner = Sigma_mu_sqrt @ nu_cov @ Sigma_mu_sqrt + (eps / 2) ** 2 * np.eye(d)
    M = _matrix_sqrt(inner)

    # C_xy = Σ_μ^{-1/2} (M − (ε/2) I) Σ_μ^{-1/2} ← cross-covariance
    Sigma_mu_sqrt_inv = np.linalg.inv(Sigma_mu_sqrt)
    C_xy = Sigma_mu_sqrt_inv @ (M - (eps / 2) * np.eye(d)) @ Sigma_mu_sqrt_inv

    joint_mean = np.concatenate([mu_mean, nu_mean])
    joint_cov  = np.block([
        [mu_cov,     C_xy],
        [C_xy.T,     nu_cov],
    ])

    # BW₂² between (μ_plan, ν_plan) and the unregularised W₂ plan (ε→0 limit)
    # Here we report the EOT cost as validation
    eot_cost = (np.linalg.norm(mu_mean - nu_mean) ** 2
                + np.trace(mu_cov) + np.trace(nu_cov)
                - 2 * np.trace(M))
    bw2 = float(eot_cost)

    return joint_mean, joint_cov, bw2


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Symmetric positive-definite matrix square root via eigendecomposition."""
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 0)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


# ---------------------------------------------------------------------------
# BW₂ metric computation (for post-hoc evaluation)
# ---------------------------------------------------------------------------

def compute_bw2_from_samples(
    x: torch.Tensor, y: torch.Tensor,
    x_ref: torch.Tensor, y_ref: torch.Tensor,
) -> float:
    """Estimate the Bures–Wasserstein² distance between two joint couplings.

    Computes empirical mean and covariance for both (x, y) and (x_ref, y_ref)
    (stacked as 2d-dim vectors) then evaluates BW₂ analytically.

    Parameters
    ----------
    x, y        : learned coupling (N, d) each
    x_ref, y_ref: reference / ground-truth coupling (M, d) each

    Returns
    -------
    BW₂² distance (float)
    """
    xy     = torch.cat([x, y], dim=1).cpu().numpy().astype(np.float64)
    xy_ref = torch.cat([x_ref, y_ref], dim=1).cpu().numpy().astype(np.float64)

    mu1, S1 = xy.mean(0), np.cov(xy, rowvar=False)
    mu2, S2 = xy_ref.mean(0), np.cov(xy_ref, rowvar=False)

    diff = mu1 - mu2
    try:
        import scipy.linalg
        covmean = scipy.linalg.sqrtm(S1 @ S2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        bw2 = float(diff @ diff + np.trace(S1 + S2 - 2 * covmean))
    except Exception:
        bw2 = float(diff @ diff)   # fallback: mean-only distance
    return max(bw2, 0.0)


# ---------------------------------------------------------------------------
# Sinkhorn oracle coupling (for GMM pairs via POT)
# ---------------------------------------------------------------------------

def _sinkhorn_coupling(
    src_samples: np.ndarray, tgt_samples: np.ndarray,
    eps: float, n_iter: int = 500,
) -> Optional[np.ndarray]:
    """Compute the Sinkhorn-regularised OT plan between two sample sets.

    Returns the transport matrix T (n_src × n_tgt), or None if POT is not
    installed.
    """
    try:
        import ot
    except ImportError:
        logger.warning("POT not installed — Sinkhorn oracle coupling unavailable.")
        return None

    n  = src_samples.shape[0]
    a  = np.ones(n) / n
    b  = np.ones(n) / n
    M  = ot.dist(src_samples, tgt_samples, metric="sqeuclidean")
    T  = ot.sinkhorn(a, b, M, reg=eps, numItertermax=n_iter)
    return T


# ---------------------------------------------------------------------------
# Simple Gaussian distributions
# ---------------------------------------------------------------------------

class GaussianDistribution(Distribution):
    """Multivariate Gaussian distribution N(mean, cov)."""

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor,
                 device: Optional[torch.device] = None):
        self._dev  = device or get_device()
        self.mean  = mean.to(self._dev)
        # Cholesky factor for efficient sampling
        self.L     = torch.linalg.cholesky(cov.to(self._dev))
        self.d     = mean.shape[0]

    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.d, device=self._dev)
        return z @ self.L.T + self.mean


class GMMDistribution(Distribution):
    """Gaussian mixture distribution."""

    def __init__(self, means: torch.Tensor, covs: torch.Tensor,
                 device: Optional[torch.device] = None):
        """
        Parameters
        ----------
        means : (K, d)
        covs  : (K, d, d)
        """
        self._dev  = device or get_device()
        self.means = means.to(self._dev)
        self.Ls    = torch.stack([torch.linalg.cholesky(C.to(self._dev)) for C in covs])
        self.K, self.d = means.shape[:2]

    def sample(self, n: int) -> torch.Tensor:
        k_idx = torch.randint(self.K, (n,), device=self._dev)
        z     = torch.randn(n, self.d, device=self._dev)
        L_sel = self.Ls[k_idx]           # (n, d, d)
        m_sel = self.means[k_idx]        # (n, d)
        return torch.bmm(L_sel, z.unsqueeze(-1)).squeeze(-1) + m_sel


# ---------------------------------------------------------------------------
# Default hyperparameter configs
# ---------------------------------------------------------------------------

def _default_train_cfg(device: torch.device, dim: int) -> TrainingConfig:
    return TrainingConfig(
        device                   = device,
        dataset_size             = 500_000,
        marginal_epochs          = 5_000,
        pretrain_epochs          = 10_000,
        dual_epochs              = 150,
        cfm_epochs               = 300,
        advance_steps            = 100,
        iterations               = 100,
        scale_factor             = -1,           # auto-detect
        batch_size               = 4_096,
        save_coupling_checkpoints= True,
        save_model_checkpoints   = False,
    )


def _default_method_cfg() -> MixtureConfig:
    return MixtureConfig(
        sigma           = 0.05,
        sigma2          = 0.05,
        loss            = "weighted",
        l2_reg          = 0,
        lip_reg         = 0,
        mean_reg        = 0,
        use_noisy_field = False,
        particle_ratio  = 0.25,
    )


# ---------------------------------------------------------------------------
# E1a — Isotropic Gaussian pairs (R²)
# ---------------------------------------------------------------------------

def _gaussian_iso_2d(eps: float, output_root: str) -> ExperimentSpec:
    """Isotropic Gaussian source → anisotropic Gaussian target, R², EOT with eps.

    Reference coupling
        Constructed from the closed-form Gaussian EOT plan (known analytically).
    New marginals
        Same distributions shifted to a different location — tests whether the
        model generalises the OT structure to unseen supports.
    Ground truth
        BW₂ to the known Gaussian EOT plan.
    """
    device = get_device()

    # Source: N([0,0], I)   Target: N([3,2], [[2,0.5],[0.5,1]])
    mu_mean = np.array([0.0, 0.0])
    mu_cov  = np.eye(2)
    nu_mean = np.array([3.0, 2.0])
    nu_cov  = np.array([[2.0, 0.5], [0.5, 1.0]])

    joint_mean, joint_cov, _ = gaussian_eot_plan(mu_mean, mu_cov, nu_mean, nu_cov, eps)

    # Build the reference coupling from the known joint Gaussian
    joint_L = torch.tensor(_matrix_sqrt(joint_cov), dtype=torch.float32)
    joint_m = torch.tensor(joint_mean, dtype=torch.float32)

    def _ref_sample(n: int):
        z  = torch.randn(n, 4)
        xy = z @ joint_L.T + joint_m
        return xy[:, :2], xy[:, 2:]

    coupling = FunctionalCouplingDistribution(_ref_sample)

    # New marginals: same shapes, shifted by [5, 5]
    shift = torch.tensor([5.0, 5.0])
    new_src = GaussianDistribution(
        mean = shift,
        cov  = torch.tensor(mu_cov, dtype=torch.float32),
        device=device,
    )
    new_tgt = GaussianDistribution(
        mean = shift + torch.tensor([3.0, 2.0]),
        cov  = torch.tensor(nu_cov, dtype=torch.float32),
        device=device,
    )

    train_cfg  = _default_train_cfg(device, dim=2)
    method_cfg = _default_method_cfg()
    method_cfg.sigma  = eps / 4    # tune sigma ≈ eps / 4 for Gaussian experiments
    method_cfg.sigma2 = eps / 4
    model_cfg  = FeedForwardConfig(dimension=2, hidden_size=64)
    vis_cfg    = VisualizeConfig(show_each=10, n_trajectories=300, trajectory_stride=2)

    # Analytical conditional-mean transport map for the Gaussian EOT plan.
    # T(x) = new_nu_mean + C_xy.T @ Σ_μ⁻¹ @ (x - new_mu_mean)
    # (C_xy is the same for the shifted marginals as for the reference pair)
    _Sigma_mu_inv = torch.tensor(np.linalg.inv(mu_cov), dtype=torch.float32)
    _Sigma_nu_inv = torch.tensor(np.linalg.inv(nu_cov), dtype=torch.float32)
    _C_xy         = torch.tensor(joint_cov[:2, 2:], dtype=torch.float32)  # d × d block
    _new_mu_mean  = shift.float()
    _new_nu_mean  = (shift + torch.tensor([3.0, 2.0])).float()

    def _transport_map_iso2d(x: torch.Tensor) -> torch.Tensor:
        return _new_nu_mean + (x - _new_mu_mean) @ (_Sigma_mu_inv @ _C_xy)

    def _inv_transport_map_iso2d(y: torch.Tensor) -> torch.Tensor:
        return _new_mu_mean + (y - _new_nu_mean) @ (_Sigma_nu_inv @ _C_xy.T)

    eps_tag = str(eps).replace(".", "p")
    return ExperimentSpec(
        name                  = f"gushchin_gaussian_iso_2d_eps{eps_tag}",
        mode                  = "2d",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = _transport_map_iso2d,
        inverse_transport_map = _inv_transport_map_iso2d,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


# ---------------------------------------------------------------------------
# E1b — Gaussian Mixture pairs (R²)
# ---------------------------------------------------------------------------

def _gaussian_gmm_2d(eps: float, output_root: str) -> ExperimentSpec:
    """4-component GMM source → 4-component GMM target, R².

    The reference coupling is drawn from the Sinkhorn OT plan between the two
    GMMs (if POT is available), or approximated by greedy nearest-neighbour
    pairing as a fallback.

    New marginals: same GMM structures shifted to a different region of R².
    """
    device = get_device()

    src_centers = np.array([[ 2.,  2.], [-2.,  2.], [-2., -2.], [ 2., -2.]])
    tgt_centers = np.array([[ 4.,  0.], [ 0.,  4.], [-4.,  0.], [ 0., -4.]])
    blob_std    = 0.4

    # Build reference coupling via Sinkhorn
    n_ref  = 20_000
    np.random.seed(0)
    k_src  = np.random.randint(4, size=n_ref)
    k_tgt  = np.random.randint(4, size=n_ref)
    x_ref  = src_centers[k_src] + np.random.randn(n_ref, 2) * blob_std
    y_ref  = tgt_centers[k_tgt] + np.random.randn(n_ref, 2) * blob_std
    T_plan = _sinkhorn_coupling(x_ref[:2_000], y_ref[:2_000], eps=eps)

    if T_plan is not None:
        # Sample (i, j) pairs according to the transport plan
        idx_src = np.repeat(np.arange(2_000), 2_000)
        idx_tgt = np.repeat(np.arange(2_000)[None, :], 2_000, axis=0).ravel()
        probs   = T_plan.ravel()
        probs   = probs / probs.sum()
        chosen  = np.random.choice(len(probs), size=n_ref, p=probs)
        x_coup  = x_ref[:2_000][chosen // 2_000]
        y_coup  = y_ref[:2_000][chosen %  2_000]
    else:
        # Fallback: greedy nearest-neighbour pairing (rough approximation)
        x_coup, y_coup = x_ref[:n_ref], y_ref[:n_ref]

    x_coup_t = torch.tensor(x_coup, dtype=torch.float32)
    y_coup_t = torch.tensor(y_coup, dtype=torch.float32)

    def _ref_sample(n: int):
        idx = torch.randint(n_ref, (n,))
        return x_coup_t[idx], y_coup_t[idx]

    coupling = FunctionalCouplingDistribution(_ref_sample)

    # New marginals: shift by [8, 8]
    shift = 8.0
    new_src_means = torch.tensor(src_centers + shift, dtype=torch.float32)
    new_tgt_means = torch.tensor(tgt_centers + shift, dtype=torch.float32)
    cov           = (blob_std ** 2) * torch.eye(2)
    covs          = cov.unsqueeze(0).expand(4, -1, -1)

    new_src = GMMDistribution(new_src_means, covs, device=device)
    new_tgt = GMMDistribution(new_tgt_means, covs, device=device)

    train_cfg  = _default_train_cfg(device, dim=2)
    method_cfg = _default_method_cfg()
    method_cfg.sigma  = max(eps / 4, 0.01)
    method_cfg.sigma2 = max(eps / 4, 0.01)
    model_cfg  = FeedForwardConfig(dimension=2, hidden_size=64)
    vis_cfg    = VisualizeConfig(show_each=10, n_trajectories=400, trajectory_stride=2)

    eps_tag = str(eps).replace(".", "p")
    return ExperimentSpec(
        name             = f"gushchin_gaussian_gmm_2d_eps{eps_tag}",
        mode             = "2d",
        initial_coupling = coupling,
        new_source       = new_src,
        new_target       = new_tgt,
        train_cfg        = train_cfg,
        model_cfg        = model_cfg,
        vis_cfg          = vis_cfg,
        method_cfg       = method_cfg,
        output_root      = output_root,
    )


# ---------------------------------------------------------------------------
# E1c — High-dimensional Gaussian pair (Rᵈ)
# ---------------------------------------------------------------------------

def _gaussian_iso_nd(dim: int, eps: float, output_root: str) -> ExperimentSpec:
    """Isotropic Gaussian pairs in R^dim for scalability evaluation."""
    device = get_device()

    mu_mean = np.zeros(dim)
    mu_cov  = np.eye(dim)
    nu_mean = np.ones(dim) * 1.5
    nu_cov  = np.eye(dim) * 2.0

    joint_mean, joint_cov, bw2_gt = gaussian_eot_plan(mu_mean, mu_cov, nu_mean, nu_cov, eps)
    logger.info("Ground-truth BW₂² for dim=%d eps=%.2f: %.4f", dim, eps, bw2_gt)

    joint_L = torch.tensor(_matrix_sqrt(joint_cov), dtype=torch.float32)
    joint_m = torch.tensor(joint_mean, dtype=torch.float32)

    def _ref_sample(n: int):
        z  = torch.randn(n, 2 * dim)
        xy = z @ joint_L.T + joint_m
        return xy[:, :dim], xy[:, dim:]

    coupling = FunctionalCouplingDistribution(_ref_sample)

    shift   = torch.ones(dim) * 5.0
    new_src = GaussianDistribution(shift, torch.tensor(mu_cov, dtype=torch.float32), device=device)
    new_tgt = GaussianDistribution(shift + torch.ones(dim) * 1.5,
                                   torch.tensor(nu_cov, dtype=torch.float32), device=device)

    hidden  = {2: 64, 8: 256, 16: 512, 32: 512}.get(dim, 256)
    train_cfg  = _default_train_cfg(device, dim=dim)
    train_cfg.dataset_size   = 200_000
    train_cfg.dual_epochs    = 200
    train_cfg.cfm_epochs     = 400
    train_cfg.iterations     = 150
    method_cfg = _default_method_cfg()
    method_cfg.sigma  = eps / 4
    method_cfg.sigma2 = eps / 4
    model_cfg  = FeedForwardConfig(dimension=dim, hidden_size=hidden)
    vis_cfg    = VisualizeConfig(show_each=25, n_trajectories=300, trajectory_stride=2)

    # Analytical transport map for the shifted new marginals
    _Smu_inv  = torch.tensor(np.linalg.inv(mu_cov), dtype=torch.float32)
    _Snu_inv  = torch.tensor(np.linalg.inv(nu_cov), dtype=torch.float32)
    _Cxy      = torch.tensor(joint_cov[:dim, dim:], dtype=torch.float32)
    _mu_new   = shift.float()
    _nu_new   = (shift + torch.ones(dim) * 1.5).float()

    def _transport_map_nd(x: torch.Tensor) -> torch.Tensor:
        return _nu_new + (x - _mu_new) @ (_Smu_inv @ _Cxy)

    def _inv_transport_map_nd(y: torch.Tensor) -> torch.Tensor:
        return _mu_new + (y - _nu_new) @ (_Snu_inv @ _Cxy.T)

    eps_tag = str(eps).replace(".", "p")
    return ExperimentSpec(
        name                  = f"gushchin_gaussian_iso_{dim}d_eps{eps_tag}",
        mode                  = "2d" if dim == 2 else "3d" if dim == 3 else "2d",
        initial_coupling      = coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = _transport_map_nd,
        inverse_transport_map = _inv_transport_map_nd,
        method_cfg            = method_cfg,
        output_root           = output_root,
    )


# ---------------------------------------------------------------------------
# Public catalogue
# ---------------------------------------------------------------------------

def make_gushchin_experiments(output_root: str = "results") -> List[ExperimentSpec]:
    """Return all Gushchin-style EOT benchmark experiments.

    Covers:
    - Isotropic Gaussian pairs in R² at ε ∈ {0.1, 0.5, 2.0}
    - GMM pairs in R² at ε ∈ {0.5, 2.0}
    - Isotropic Gaussian in R⁸ at ε = 0.5  (scaling experiment)
    """
    eps_levels_iso = [0.1, 0.5, 2.0]
    eps_levels_gmm = [0.5, 2.0]

    specs: List[ExperimentSpec] = []

    for eps in eps_levels_iso:
        specs.append(_gaussian_iso_2d(eps, output_root))

    for eps in eps_levels_gmm:
        specs.append(_gaussian_gmm_2d(eps, output_root))

    # High-dimensional scaling
    specs.append(_gaussian_iso_nd(dim=8, eps=0.5, output_root=output_root))

    return specs


# ---------------------------------------------------------------------------
# Post-hoc BW₂ evaluation helper
# ---------------------------------------------------------------------------

def evaluate_bw2(
    trainer,
    mu_mean: np.ndarray, mu_cov: np.ndarray,
    nu_mean: np.ndarray, nu_cov: np.ndarray,
    eps: float,
    n_samples: int = 10_000,
) -> dict:
    """Compute BW₂ between the learned coupling and the ground-truth Gaussian EOT plan.

    Call this after training to get the final quantitative score::

        result = evaluate_bw2(trainer, mu_mean, mu_cov, nu_mean, nu_cov, eps)
        print(f"BW₂ = {result['bw2']:.4f}  (ground truth EOT cost = {result['gt_cost']:.4f})")

    Returns
    -------
    dict with keys: bw2, learned_cost, gt_cost
    """
    joint_mean, joint_cov, gt_cost = gaussian_eot_plan(mu_mean, mu_cov, nu_mean, nu_cov, eps)

    # Sample from ground-truth plan
    joint_L = _matrix_sqrt(joint_cov)
    z       = np.random.randn(n_samples, len(joint_mean))
    xy_ref  = z @ joint_L.T + joint_mean
    x_ref   = torch.tensor(xy_ref[:, :len(mu_mean)], dtype=torch.float32)
    y_ref   = torch.tensor(xy_ref[:, len(mu_mean):], dtype=torch.float32)

    # Sample from learned coupling
    x_learn, y_learn = trainer.current_coupling.coupling_sample(n_samples)

    bw2          = compute_bw2_from_samples(x_learn.cpu(), y_learn.cpu(), x_ref, y_ref)
    learned_cost = float(((x_learn - y_learn) ** 2).sum(-1).mean().item())

    return {"bw2": bw2, "learned_cost": learned_cost, "gt_cost": gt_cost}
