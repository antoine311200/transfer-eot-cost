"""ExperimentSpec factory for E3: Cross-Regime Risk Factor Scenario Generation.

Setup
-----
Five risk factors: equity, vol, rates, FX, credit.
Reference coupling π': daily (factors_today, factors_tomorrow) pairs in the
normal VIX regime (VIX 12–20).

New task: same correlation structure / dependence skeleton (latent cost) but
the stressed VIX regime (VIX > 25) where all marginal vols are 2–3× higher.

TACO transfers the joint dynamics from normal to stressed without re-estimating
the full multivariate distribution on the (scarce) stressed data.
"""

import logging
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from dualwcfm.config import TrainingConfig, FeedForwardConfig, VisualizeConfig
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.experiments.base import ExperimentSpec, get_device

from ..data.risk_factors import load_risk_factors, D_RF
from ..distributions import FinancialCouplingDistribution, FinancialMarginalDistribution

logger = logging.getLogger(__name__)


def _train_cfg(device: torch.device) -> TrainingConfig:
    return TrainingConfig(
        device=device,
        marginal_epochs=1500,
        pretrain_epochs=4000,
        dual_epochs=50,
        cfm_epochs=100,
        iterations=100,
        dataset_size=4000,
        batch_size=512,
        scale_factor=-1.0,
    )


def _model_cfg() -> FeedForwardConfig:
    return FeedForwardConfig(
        dimension=D_RF,
        hidden_size=64,
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
        particle_ratio=0.25,
        use_ot_pretrain=False,
    )


def make_risk_factors_experiment(
    output_root: str = "results/icaif",
    device: torch.device = None,
    try_real_data: bool = True,
) -> ExperimentSpec:
    """Build the ExperimentSpec for the risk-factor scenario generation experiment."""
    device = device or get_device()
    logger.info("Building risk factor experiment (D=%d)…", D_RF)

    (x_normal, y_normal), (x_stress, y_stress) = load_risk_factors(
        n_normal=2000, n_stress=200, try_real=try_real_data
    )

    logger.info("  Normal:  %d coupling pairs", len(x_normal))
    logger.info("  Stress:  %d coupling pairs", len(x_stress))

    ref_coupling = FinancialCouplingDistribution(x_normal, y_normal, device)
    new_source   = FinancialMarginalDistribution(x_stress, device)
    new_target   = FinancialMarginalDistribution(y_stress, device)

    return ExperimentSpec(
        name="risk_factors",
        mode="2d",
        initial_coupling=ref_coupling,
        new_source=new_source,
        new_target=new_target,
        train_cfg=_train_cfg(device),
        model_cfg=_model_cfg(),
        vis_cfg=VisualizeConfig(show_each=20, n_trajectories=200),
        method_cfg=_method_cfg(),
        output_root=output_root,
        visualizer_kwargs={
            "truth_x": x_stress,
            "truth_y": y_stress,
            "ref_x":   x_normal,
            "ref_y":   y_normal,
        },
    )
