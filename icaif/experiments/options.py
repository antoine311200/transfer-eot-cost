"""ExperimentSpec factories for the cross-maturity options EMOT experiment.

Reference coupling
------------------
π': same-day (T1 call prices, T2 call prices) from the CALM regime.
    Each pair (x, y) is the joint option strip on one simulation day.

New marginals
-------------
µ': T1 call prices from the STRESS regime.
ν': T2 call prices from the STRESS regime.

Task: Transfer the calm-regime coupling to match today's stressed marginals.

Two variants
------------
make_options_eot_experiment   — pure EOT  (no martingale constraint)
make_options_emot_experiment  — EMOT with the m_t(X)·(Y−X) dual constraint

The martingale constraint E_π[Y | X = x] = x encodes the heuristic that
T2 call prices are a conditional martingale of T1 prices (term-structure
driftless condition).  When active, a third network m_t : R^5 → R^5 is
trained jointly with the f/g dual potentials.
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from dualwcfm.config import TrainingConfig, FeedForwardConfig, VisualizeConfig
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.experiments.base import ExperimentSpec, get_device

from ..data.options import (
    generate_options_coupling,
    D_OPTIONS, N_CALM_COUPLING, N_STRESS,
)
from ..distributions import FinancialCouplingDistribution, FinancialMarginalDistribution

logger = logging.getLogger(__name__)

_D = D_OPTIONS   # 5


def _train_cfg(device: torch.device) -> TrainingConfig:
    return TrainingConfig(
        device=device,
        marginal_epochs=1500,
        pretrain_epochs=3000,
        dual_epochs=40,
        cfm_epochs=80,
        iterations=60,
        dataset_size=N_CALM_COUPLING,
        batch_size=512,
        scale_factor=-1.0,
    )


def _model_cfg(use_martingale: bool = False) -> FeedForwardConfig:
    return FeedForwardConfig(
        dimension=_D,
        hidden_size=128,
        num_layers=3,
        flow_learning_rate=5e-4,
        dual_learning_rate=5e-5,
        mu_learning_rate=5e-4,
        nu_learning_rate=5e-4,
        use_martingale=use_martingale,
    )


def _method_cfg() -> MixtureConfig:
    return MixtureConfig(
        sigma=0.02,
        sigma2=0.02,
        loss="weighted",
        l2_reg=0.0,
        lip_reg=0.0,
        mean_reg=0.0,
        particle_ratio=0.20,
        use_ot_pretrain=False,
    )


def _build_spec(
    name: str,
    use_martingale: bool,
    output_root: str,
    device: torch.device,
) -> ExperimentSpec:
    logger.info("Building options coupling experiment (%s, martingale=%s)…",
                name, use_martingale)

    x_calm,   y_calm   = generate_options_coupling(N_CALM_COUPLING, seed=0,  stress=False)
    x_stress, y_stress = generate_options_coupling(N_STRESS,        seed=1,  stress=True)

    logger.info(
        "  Calm  : %d pairs  | ATM T1 µ=%.4f  ATM T2 µ=%.4f",
        len(x_calm), float(x_calm[:, 2].mean()), float(y_calm[:, 2].mean()),
    )
    logger.info(
        "  Stress: %d samples | ATM T1 µ=%.4f  ATM T2 µ=%.4f",
        len(x_stress), float(x_stress[:, 2].mean()), float(y_stress[:, 2].mean()),
    )

    ref_coupling = FinancialCouplingDistribution(x_calm, y_calm, device)
    new_source   = FinancialMarginalDistribution(x_stress, device)
    new_target   = FinancialMarginalDistribution(y_stress, device)

    return ExperimentSpec(
        name=name,
        mode="2d",
        initial_coupling=ref_coupling,
        new_source=new_source,
        new_target=new_target,
        train_cfg=_train_cfg(device),
        model_cfg=_model_cfg(use_martingale=use_martingale),
        vis_cfg=VisualizeConfig(show_each=10, n_trajectories=200),
        method_cfg=_method_cfg(),
        output_root=output_root,
        visualizer_kwargs={
            "truth_x": x_stress,
            "truth_y": y_stress,
            "ref_x":   x_calm,
            "ref_y":   y_calm,
        },
    )


def make_options_eot_experiment(
    output_root: str = "results/icaif",
    device: torch.device = None,
) -> ExperimentSpec:
    """EOT regime transfer — no martingale constraint."""
    return _build_spec("options_eot", False, output_root, device or get_device())


def make_options_emot_experiment(
    output_root: str = "results/icaif",
    device: torch.device = None,
) -> ExperimentSpec:
    """EMOT regime transfer — with martingale dual m_t(X)·(Y−X)."""
    return _build_spec("options_emot", True, output_root, device or get_device())
