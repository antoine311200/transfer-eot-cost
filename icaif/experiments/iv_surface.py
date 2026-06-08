"""ExperimentSpec factories for E1: IV Surface Regime Transfer.

Two variants
------------
Synthetic (two-factor)
    Reference coupling π': daily (surface_today, surface_tomorrow) pairs
    simulated from a two-factor stochastic-volatility model (Heston variance
    process + independent stochastic skew factor) in the calm regime.

    The surface is driven by a genuine 2D state (v_t, s_t):
      - v_t  — instantaneous variance (drives ATM vol level)
      - s_t  — stochastic skew perturbation (shifts effective rho)

    This is critical: in a plain single-factor Heston model the entire 10D
    surface collapses to a deterministic function of one scalar, so the
    coupling reduces to a 1D Markov chain and a simple rescaling trivially
    solves the transfer.  The two-factor model prevents this degeneracy.

    Calm →  Stress transfer: theta_v and v0 increase (higher ATM vol) AND
    theta_s and s0 become more negative (steeper put skew, rho_eff ≈ −0.80
    vs −0.70 calm).  The structural dynamics (kappa_v, xi_v, rho, kappa_s,
    xi_s) are IDENTICAL — they encode the latent cost TACO must transfer.

SPX real data
    Reference coupling π': (surface_t, surface_{t+1}) pairs built from the
    CBOE VIX term structure (^VIX, ^VIX3M) and SKEW index (^SKEW) via the
    Bergomi-Gatheral expansion.  Calm regime: all of 2019.  Stress regime:
    Feb–Jun 2020 COVID crash.  Requires yfinance (pip install yfinance).
    Falls back gracefully to the synthetic two-factor experiment if download
    fails or yfinance is not installed.
"""

import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import numpy as np

from dualwcfm.config import TrainingConfig, FeedForwardConfig, VisualizeConfig
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.experiments.base import ExperimentSpec, get_device

from ..data.heston import (
    generate_iv_surface_dataset, CALM_PARAMS, STRESS_PARAMS,    # single-factor (kept for reference)
    generate_iv_surface_dataset_2f, CALM_2F, STRESS_2F, D_IV,  # two-factor (new default)
)
from ..distributions import FinancialCouplingDistribution, FinancialMarginalDistribution

logger = logging.getLogger(__name__)

_N_CALM   = 3000
_N_STRESS = 800


def _train_cfg(device: torch.device) -> TrainingConfig:
    return TrainingConfig(
        device=device,
        marginal_epochs=2000,
        pretrain_epochs=6000,
        dual_epochs=60,
        cfm_epochs=120,
        iterations=120,
        dataset_size=min(_N_CALM, 8000),
        batch_size=1024,
        scale_factor=-1.0,
    )


def _model_cfg() -> FeedForwardConfig:
    return FeedForwardConfig(
        dimension=D_IV,
        hidden_size=128,
        num_layers=3,
        flow_learning_rate=5e-4,
        dual_learning_rate=5e-5,
        mu_learning_rate=5e-4,
        nu_learning_rate=5e-4,
    )


def _method_cfg() -> MixtureConfig:
    return MixtureConfig(
        sigma=0.05,
        sigma2=0.05,
        loss="weighted",
        l2_reg=0.0,
        lip_reg=0.0,
        mean_reg=0.0,
        particle_ratio=0.20,
        use_ot_pretrain=False,
    )


# ── Synthetic two-factor experiment ──────────────────────────────────────────

def make_iv_surface_experiment(
    output_root: str = "results/icaif",
    device: torch.device = None,
) -> ExperimentSpec:
    """IV-surface regime transfer using the two-factor synthetic model."""
    device = device or get_device()
    logger.info("Building IV surface experiment (two-factor, D=%d)…", D_IV)

    x_calm,   y_calm   = generate_iv_surface_dataset_2f(CALM_2F,   _N_CALM,   seed=0)
    x_stress, y_stress = generate_iv_surface_dataset_2f(STRESS_2F, _N_STRESS, seed=1)

    logger.info(
        "  Calm  : %d pairs, ATM vol ≈ %.1f%%, long-run skew factor θ_s=%.2f",
        len(x_calm), float(CALM_2F.sigma_atm * 100), CALM_2F.theta_s,
    )
    logger.info(
        "  Stress: %d pairs, ATM vol ≈ %.1f%%, long-run skew factor θ_s=%.2f",
        len(x_stress), float(STRESS_2F.sigma_atm * 100), STRESS_2F.theta_s,
    )

    ref_coupling = FinancialCouplingDistribution(x_calm, y_calm, device)
    new_source   = FinancialMarginalDistribution(x_stress, device)
    new_target   = FinancialMarginalDistribution(y_stress, device)

    return ExperimentSpec(
        name="iv_surface",
        mode="2d",
        initial_coupling=ref_coupling,
        new_source=new_source,
        new_target=new_target,
        train_cfg=_train_cfg(device),
        model_cfg=_model_cfg(),
        vis_cfg=VisualizeConfig(show_each=20, n_trajectories=300),
        method_cfg=_method_cfg(),
        output_root=output_root,
        visualizer_kwargs={
            "truth_x": x_stress,
            "truth_y": y_stress,
            "ref_x":   x_calm,
            "ref_y":   y_calm,
        },
    )


# ── SPX real-data experiment ──────────────────────────────────────────────────

def make_iv_surface_spx_experiment(
    output_root: str = "results/icaif",
    device: torch.device = None,
    cache_dir: str = "data/spx_cache",
) -> ExperimentSpec:
    """IV-surface regime transfer on real SPX data (CBOE VIX/SKEW).

    Reference: all of 2019 (calm, VIX 12–20).
    Stress   : Feb–Jun 2020 COVID crash (VIX 20–85).

    Falls back to the synthetic two-factor experiment if yfinance is not
    installed or the download fails (e.g. offline environment).
    """
    device = device or get_device()

    try:
        from ..data.spx import (
            generate_spx_coupling, CALM_DATES, STRESS_DATES,
        )
        calm_cache   = os.path.join(cache_dir, "spx_calm_2017.parquet")
        stress_cache = os.path.join(cache_dir, "spx_stress.parquet")

        logger.info("Building IV surface experiment (SPX real data)…")
        x_calm,   y_calm   = generate_spx_coupling(*CALM_DATES,   cache_path=calm_cache)
        x_stress, y_stress = generate_spx_coupling(*STRESS_DATES, cache_path=stress_cache)

        logger.info(
            "  Calm  : %d pairs (%s → %s), median ATM ≈ %.1f%%",
            len(x_calm), *CALM_DATES, float(np.median(x_calm[:, 2]) * 100),
        )
        logger.info(
            "  Stress: %d pairs (%s → %s), median ATM ≈ %.1f%%",
            len(x_stress), *STRESS_DATES, float(np.median(x_stress[:, 2]) * 100),
        )

        name = "iv_surface_spx"

    except Exception as exc:
        logger.warning(
            "SPX data unavailable (%s); falling back to synthetic two-factor.", exc
        )
        return make_iv_surface_experiment(output_root=output_root, device=device)

    train_cfg = _train_cfg(device)
    # Adjust dataset size to actual data lengths
    train_cfg.dataset_size = min(len(x_calm), 8000)

    ref_coupling = FinancialCouplingDistribution(x_calm, y_calm, device)
    new_source   = FinancialMarginalDistribution(x_stress, device)
    new_target   = FinancialMarginalDistribution(y_stress, device)

    return ExperimentSpec(
        name=name,
        mode="2d",
        initial_coupling=ref_coupling,
        new_source=new_source,
        new_target=new_target,
        train_cfg=train_cfg,
        model_cfg=_model_cfg(),
        vis_cfg=VisualizeConfig(show_each=20, n_trajectories=300),
        method_cfg=_method_cfg(),
        output_root=output_root,
        visualizer_kwargs={
            "truth_x": x_stress,
            "truth_y": y_stress,
            "ref_x":   x_calm,
            "ref_y":   y_calm,
        },
    )
