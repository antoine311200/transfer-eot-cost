"""ExperimentSpec factory for E2: Yield Curve Dynamics Transfer.

Setup
-----
Reference coupling π': daily (curve_today, curve_tomorrow) pairs during the
ZIRP epoch (2015–2019 equivalent, low rates, upward-sloping curve).

New task: same structural factor loadings λ (latent cost), but the *factor
levels* (β₀, β₁, β₂) shift to a higher-rate hiking regime (2022–2023).

TACO transfers the yield-curve transition dynamics without re-fitting the
Nelson–Siegel model on the new epoch.
"""

import logging
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import numpy as np

from dualwcfm.config import TrainingConfig, FeedForwardConfig, VisualizeConfig
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.experiments.base import ExperimentSpec, get_device

from ..data.yield_curve import (
    generate_yield_curve_dataset, ZIRP_PARAMS, HIKING_PARAMS, D_YC,
)
from ..distributions import FinancialCouplingDistribution, FinancialMarginalDistribution

logger = logging.getLogger(__name__)

_N_ZIRP   = 2500
_N_HIKING = 600


def _train_cfg(device: torch.device) -> TrainingConfig:
    return TrainingConfig(
        device=device,
        marginal_epochs=2000,
        pretrain_epochs=6000,
        dual_epochs=60,
        cfm_epochs=120,
        iterations=120,
        dataset_size=min(_N_ZIRP, 8000),
        batch_size=1024,
        scale_factor=-1.0,
    )


def _model_cfg() -> FeedForwardConfig:
    return FeedForwardConfig(
        dimension=D_YC,
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


def make_yield_curve_experiment(
    output_root: str = "results/icaif",
    device: torch.device = None,
) -> ExperimentSpec:
    """Build the ExperimentSpec for the yield-curve dynamics-transfer experiment."""
    device = device or get_device()
    logger.info("Building yield curve experiment (D=%d)…", D_YC)

    x_zirp,   y_zirp   = generate_yield_curve_dataset(ZIRP_PARAMS,   _N_ZIRP,   seed=10)
    x_hiking, y_hiking = generate_yield_curve_dataset(HIKING_PARAMS, _N_HIKING, seed=11)

    logger.info("  ZIRP:   %d pairs, mean 10Y ≈ %.2f%%",
                len(x_zirp),   float(x_zirp[:, 7].mean()))
    logger.info("  Hiking: %d pairs, mean 10Y ≈ %.2f%%",
                len(x_hiking), float(x_hiking[:, 7].mean()))

    ref_coupling = FinancialCouplingDistribution(x_zirp,   y_zirp,   device)
    new_source   = FinancialMarginalDistribution(x_hiking, device)
    new_target   = FinancialMarginalDistribution(y_hiking, device)

    return ExperimentSpec(
        name="yield_curve",
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
            "truth_x": x_hiking,
            "truth_y": y_hiking,
            "ref_x":   x_zirp,
            "ref_y":   y_zirp,
        },
    )
