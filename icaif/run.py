"""Main entry point for TACO finance experiments (ICAIF 2026).

Usage
-----
    # Run all three experiments
    python -m icaif.run --all

    # Single experiment
    python -m icaif.run --exp iv_surface
    python -m icaif.run --exp yield_curve
    python -m icaif.run --exp risk_factors

    # With W&B logging
    python -m icaif.run --all --wandb-project TACO-Finance --wandb-entity myname

    # Skip re-training (load saved models)
    python -m icaif.run --exp iv_surface --no-retrain

    # Override iterations
    python -m icaif.run --exp risk_factors --iterations 50
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys

# Ensure the src directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")

from dualwcfm.experiments.base import (
    ExperimentSpec, _make_mlp_strategy, get_device, setup_logging,
    save_metrics_plot, METHOD_REGISTRY,
)
from dualwcfm.flow_matching import ConditionalFlowMatching
from dualwcfm.trainer import Trainer

logger = logging.getLogger("dualwcfm")


# ── Custom run function ───────────────────────────────────────────────────────

def run_finance_experiment(
    spec: ExperimentSpec,
    retrain: bool = True,
    save_models: bool = True,
    wandb_project: str = None,
    wandb_entity: str = None,
    wandb_run_name: str = None,
    wandb_mode: str = None,
) -> None:
    """Like base.run_experiment but wires the finance-specific visualizer.

    The custom visualizer class is selected based on spec.name and the
    ground-truth data is retrieved from spec.visualizer_kwargs.
    """
    setup_logging(spec.output_dir)
    os.makedirs(spec.output_dir, exist_ok=True)
    model_dir = os.path.join(spec.output_dir, "models")
    device = spec.train_cfg.device

    logger.info("=" * 70)
    logger.info("  Finance Experiment : %s", spec.name)
    logger.info("  Method             : %s", type(spec.method_cfg).__name__)
    logger.info("=" * 70)

    # ── W&B init ──────────────────────────────────────────────────────────
    from dualwcfm import wandb_logger as _wl

    def _cfg_dict(obj):
        return {f.name: str(getattr(obj, f.name)) for f in dataclasses.fields(obj)}

    _wb_cfg = {
        "experiment": spec.name,
        **_cfg_dict(spec.train_cfg),
        **_cfg_dict(spec.method_cfg),
    }
    wandb_run = _wl.init(
        project=wandb_project,
        name=wandb_run_name or spec.name,
        entity=wandb_entity,
        config=_wb_cfg,
        mode=wandb_mode,
    )

    # ── Visualizer ─────────────────────────────────────────────────────────
    from icaif.visualizer import (
        IVSurfaceVisualizer, YieldCurveVisualizer, RiskFactorVisualizer,
    )
    viz_kw = spec.visualizer_kwargs or {}
    viz_cls = {
        "iv_surface":     IVSurfaceVisualizer,
        "iv_surface_spx": IVSurfaceVisualizer,
        "yield_curve":    YieldCurveVisualizer,
        "risk_factors":   RiskFactorVisualizer,
    }.get(spec.name, IVSurfaceVisualizer)

    visualizer = viz_cls(
        vis_cfg=spec.vis_cfg,
        output_dir=spec.output_dir,
        wandb_run=wandb_run,
        **viz_kw,
    )

    # ── Method + Strategy + Flow matching ─────────────────────────────────
    method_cls = METHOD_REGISTRY.get(type(spec.method_cfg))
    method     = method_cls(spec.method_cfg)
    strategy   = _make_mlp_strategy(spec.model_cfg, bidirectional=True, device=device)
    fm         = ConditionalFlowMatching()

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        initial_coupling=spec.initial_coupling,
        new_source=spec.new_source,
        new_target=spec.new_target,
        flow_matching=fm,
        strategy=strategy,
        visualizer=visualizer,
        config=spec.train_cfg,
        method=method,
        device=device,
    )

    trainer.train(
        retrain=retrain,
        save=model_dir,
        save_checkpoints=model_dir if save_models else None,
    )

    # ── Post-training ──────────────────────────────────────────────────────
    logger.info("Generating final finance plots…")
    visualizer.make_final_plots(trainer)

    visualizer.make_timelapse_all(fps=4)
    save_metrics_plot(spec.output_dir)
    wandb_run.finish()
    logger.info("Done -> %s", spec.output_dir)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TACO Finance Experiments (ICAIF 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--exp",
                   choices=["iv_surface", "iv_surface_spx", "yield_curve", "risk_factors"],
                   help="Which experiment to run.")
    p.add_argument("--all",       action="store_true", help="Run all three experiments.")
    p.add_argument("--output",    default="results/icaif", help="Output root directory.")
    p.add_argument("--no-retrain",action="store_true", dest="no_retrain",
                   help="Skip training; load saved models.")
    p.add_argument("--no-save",   action="store_false", dest="save_models",
                   help="Do not save model checkpoints.")
    p.add_argument("--iterations",type=int, default=None)
    p.add_argument("--no-real-data", action="store_false", dest="real_data",
                   help="Force synthetic data even if yfinance is available.")

    wb = p.add_argument_group("Weights & Biases")
    wb.add_argument("--wandb-project", default=None, dest="wandb_project")
    wb.add_argument("--wandb-entity",  default=None, dest="wandb_entity")
    wb.add_argument("--wandb-offline", action="store_true", dest="wandb_offline")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if not args.exp and not args.all:
        print("Specify --exp <name> or --all.  Run with -h for help.")
        sys.exit(1)

    from icaif.experiments.iv_surface   import (
        make_iv_surface_experiment, make_iv_surface_spx_experiment,
    )
    from icaif.experiments.yield_curve  import make_yield_curve_experiment
    from icaif.experiments.risk_factors import make_risk_factors_experiment

    factories = {
        "iv_surface":     lambda: make_iv_surface_experiment(args.output),
        "iv_surface_spx": lambda: make_iv_surface_spx_experiment(args.output),
        "yield_curve":    lambda: make_yield_curve_experiment(args.output),
        "risk_factors":   lambda: make_risk_factors_experiment(
            args.output, try_real_data=args.real_data),
    }
    # --all runs the three core experiments (synthetic iv, yield curve, risk factors)
    _all_keys = ["iv_surface", "yield_curve", "risk_factors"]
    keys = _all_keys if args.all else [args.exp]

    for key in keys:
        spec = factories[key]()
        if args.iterations is not None:
            spec.train_cfg.iterations = args.iterations

        run_finance_experiment(
            spec,
            retrain=not args.no_retrain,
            save_models=args.save_models,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_mode="offline" if args.wandb_offline else None,
        )


if __name__ == "__main__":
    main()
