"""dualwcfm.baselines.run
=========================
Command-line interface for running all baselines and generating comparison tables.

This module is intentionally decoupled from the main experiment runner.  It
reads ExperimentSpec objects from the existing catalogue factories and converts
them into BaselineSpec objects, then trains and evaluates baselines independently.

Usage examples
--------------
Run OT-CFM on 2D simple::

    python -m dualwcfm.baselines.run --baseline otcfm --mode 2d --level simple

Run the pretrained-only baseline (requires a prior DualWCFM run)::

    python -m dualwcfm.baselines.run --baseline pretrained --mode 2d --level simple

Run all baselines on all 2D experiments::

    python -m dualwcfm.baselines.run --baseline all --mode 2d

With W&B logging::

    python -m dualwcfm.baselines.run --baseline otcfm --mode 2d \\
        --wandb-project "Data-Based EOT" --wandb-entity antoine311200

Generate comparison table from existing results::

    python -m dualwcfm.baselines.run --table --output results --metric W2_nu
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logger = logging.getLogger("dualwcfm")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "python -m dualwcfm.baselines.run",
        description = "DualWCFM baseline runner",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog      = __doc__,
    )

    sel = parser.add_argument_group("experiment selection")
    sel.add_argument(
        "--mode", choices=["2d", "3d", "gushchin"],
        help="Experiment mode (required unless --table).",
    )
    sel.add_argument("--level", choices=["simple", "medium", "complex"], default=None)
    sel.add_argument("--perturbed", action="store_true")

    bl = parser.add_argument_group("baseline")
    bl.add_argument(
        "--baseline",
        choices=["otcfm", "cfm", "pretrained", "dsb", "lightsb", "metaot", "all"],
        default="otcfm",
        help=(
            "otcfm=OT-paired CFM from scratch, "
            "cfm=random-paired CFM from scratch, "
            "pretrained=DualWCFM without adaptation, "
            "dsb=Diffusion Schrödinger Bridge (IPF), "
            "lightsb=LightSB (dual potentials + interpolant), "
            "metaot=Neural W2 OT via ICNN (MetaOT inner loop), "
            "all=run every applicable baseline."
        ),
    )
    bl.add_argument(
        "--pretrain-dir", default=None, dest="pretrain_dir",
        metavar="PATH",
        help=(
            "Path to models/ directory for the pretrained-only baseline. "
            "Defaults to {output}/{exp_name}/models/."
        ),
    )

    tr = parser.add_argument_group("training (shared)")
    tr.add_argument("--epochs",     type=int,   default=None,  help="Gradient steps (default per-method).")
    tr.add_argument("--batch-size", type=int,   default=None,  dest="batch_size")
    tr.add_argument("--hidden",     type=int,   default=256,   help="MLP hidden size.")
    tr.add_argument("--n-layers",   type=int,   default=4,     dest="n_layers")
    tr.add_argument(
        "--ot-eps", type=float, default=0.05, dest="ot_eps",
        help="Sinkhorn epsilon for OT pairing / LightSB entropic reg (0 = plain CFM).",
    )
    tr.add_argument(
        "--dsb-iters", type=int, default=None, dest="dsb_iters",
        help="Number of IPF iterations for DSB (default: 10).",
    )
    tr.add_argument(
        "--dsb-sigma", type=float, default=0.0, dest="dsb_sigma",
        help="SDE noise level for DSB (0 = deterministic ODE bridge).",
    )
    tr.add_argument(
        "--lightsb-eps", type=float, default=None, dest="lightsb_eps",
        help="Entropic regularisation ε for LightSB (default: 0.1).",
    )

    wb = parser.add_argument_group("Weights & Biases")
    wb.add_argument("--wandb-project", default=None, dest="wandb_project")
    wb.add_argument("--wandb-entity",  default=None, dest="wandb_entity")

    out = parser.add_argument_group("output")
    out.add_argument("--output", default="results", help="Root results directory.")

    tbl = parser.add_argument_group("table generation")
    tbl.add_argument(
        "--table", action="store_true",
        help="Aggregate existing results into a comparison table (no training).",
    )
    tbl.add_argument("--metric", default="W2_nu", help="Metric for the table (default: W2_nu).")

    return parser


def _baseline_spec_from_exp(exp_spec, baseline_name: str, output_root: str):
    """Convert an ExperimentSpec into a BaselineSpec, including Gushchin GT if present."""
    from dualwcfm.baselines.base import BaselineSpec
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gushchin experiments optionally carry ground-truth Gaussian parameters
    ref_mean = getattr(exp_spec, "ref_mean", None)
    ref_cov  = getattr(exp_spec, "ref_cov",  None)

    return BaselineSpec(
        name          = baseline_name,
        mode          = exp_spec.mode,
        source        = exp_spec.new_source,
        target        = exp_spec.new_target,
        dim           = exp_spec.model_cfg.flat_dimension,
        output_dir    = os.path.join(output_root, exp_spec.name, "baselines", baseline_name),
        device        = device,
        ref_mean      = ref_mean,
        ref_cov       = ref_cov,
        transport_map = getattr(exp_spec, "transport_map", None),
    )


def _make_wandb_run(args, exp_name: str, baseline_name: str):
    from dualwcfm import wandb_logger as _wl
    return _wl.init(
        project = getattr(args, "wandb_project", None),
        entity  = getattr(args, "wandb_entity", None),
        name    = f"{exp_name}__{baseline_name}",
        group   = f"baselines__{baseline_name}",
        config  = {
            "baseline":   baseline_name,
            "experiment": exp_name,
            "mode":       args.mode,
        },
    )


def _filter(catalogue, level, perturbed):
    parts = ([level] if level else []) + (["perturbed"] if perturbed else [])
    if not parts:
        return catalogue
    return [s for s in catalogue if all(p in s.name for p in parts)]


# ---------------------------------------------------------------------------
# Baseline runners
# ---------------------------------------------------------------------------

def _run_otcfm(args, exp_specs, use_ot: bool = True) -> None:
    from dualwcfm.baselines.otcfm import OTCFMRunner, OTCFMConfig

    label = "otcfm" if use_ot else "cfm"
    for exp_spec in exp_specs:
        bl_spec   = _baseline_spec_from_exp(exp_spec, label, args.output)
        wandb_run = _make_wandb_run(args, exp_spec.name, label)

        cfg = OTCFMConfig(
            n_steps    = args.epochs    or 10_000,
            batch_size = args.batch_size or 512,
            hidden     = args.hidden,
            n_layers   = args.n_layers,
            ot_eps     = (args.ot_eps if use_ot and args.ot_eps > 0 else None),
        )
        runner = OTCFMRunner(bl_spec, config=cfg, wandb_run=wandb_run)
        metrics = runner.run(output_root=args.output)
        wandb_run.finish()
        logger.info("  %s | %s -> W2_nu=%.4f", exp_spec.name, label, metrics["W2_nu"])


def _run_dsb(args, exp_specs) -> None:
    from dualwcfm.baselines.dsb import DSBRunner, DSBConfig

    for exp_spec in exp_specs:
        bl_spec   = _baseline_spec_from_exp(exp_spec, "dsb", args.output)
        wandb_run = _make_wandb_run(args, exp_spec.name, "dsb")

        cfg = DSBConfig(
            n_ipf_iters      = args.dsb_iters or 10,
            n_steps_per_half = args.epochs    or 2_000,
            batch_size       = args.batch_size or 512,
            hidden           = args.hidden,
            n_layers         = args.n_layers,
            sigma            = args.dsb_sigma,
        )
        runner  = DSBRunner(bl_spec, config=cfg, wandb_run=wandb_run)
        metrics = runner.run(output_root=args.output)
        wandb_run.finish()
        logger.info("  %s | dsb -> W2_nu=%.4f", exp_spec.name, metrics["W2_nu"])


def _run_lightsb(args, exp_specs) -> None:
    from dualwcfm.baselines.lightsb import LightSBRunner, LightSBConfig

    eps = args.lightsb_eps if args.lightsb_eps is not None else (
        args.ot_eps if args.ot_eps > 0 else 0.1
    )
    for exp_spec in exp_specs:
        bl_spec   = _baseline_spec_from_exp(exp_spec, "lightsb", args.output)
        wandb_run = _make_wandb_run(args, exp_spec.name, "lightsb")

        n_steps = args.epochs or 5_000
        cfg = LightSBConfig(
            eps              = eps,
            n_dual_steps     = n_steps,
            n_cfm_steps      = n_steps,
            dual_batch       = args.batch_size or 256,
            cfm_batch        = args.batch_size or 512,
            potential_hidden = args.hidden,
            cfm_hidden       = args.hidden,
            potential_layers = max(2, args.n_layers - 1),
            cfm_layers       = args.n_layers,
        )
        runner  = LightSBRunner(bl_spec, config=cfg, wandb_run=wandb_run)
        metrics = runner.run(output_root=args.output)
        wandb_run.finish()
        logger.info("  %s | lightsb -> W2_nu=%.4f", exp_spec.name, metrics["W2_nu"])


def _run_metaot(args, exp_specs) -> None:
    from dualwcfm.baselines.metaot import MetaOTRunner, MetaOTConfig

    for exp_spec in exp_specs:
        bl_spec   = _baseline_spec_from_exp(exp_spec, "metaot", args.output)
        wandb_run = _make_wandb_run(args, exp_spec.name, "metaot")

        cfg = MetaOTConfig(
            n_steps    = args.epochs    or 10_000,
            batch_size = args.batch_size or 512,
            hidden     = args.hidden,
            n_layers   = args.n_layers,
        )
        runner  = MetaOTRunner(bl_spec, config=cfg, wandb_run=wandb_run)
        metrics = runner.run(output_root=args.output)
        wandb_run.finish()
        logger.info("  %s | metaot -> W2_nu=%.4f", exp_spec.name, metrics["W2_nu"])


def _run_pretrained(args, exp_specs) -> None:
    from dualwcfm.baselines.pretrained import PretrainedOnlyRunner

    for exp_spec in exp_specs:
        pretrain_dir = args.pretrain_dir or os.path.join(args.output, exp_spec.name, "models")
        bl_spec      = _baseline_spec_from_exp(exp_spec, "pretrained_only", args.output)
        wandb_run    = _make_wandb_run(args, exp_spec.name, "pretrained_only")

        # Pass the reference target marginal as conditioning signal
        ref_target = getattr(exp_spec.initial_coupling, "sample_target", None)
        if ref_target is not None:
            from dualwcfm.distributions.coupling import MarginalDistribution
            ref_target_dist = MarginalDistribution(ref_target)
        else:
            ref_target_dist = None

        runner = PretrainedOnlyRunner(
            bl_spec,
            pretrain_dir  = pretrain_dir,
            ref_target    = ref_target_dist,
            hidden_size   = exp_spec.model_cfg.hidden_size,
            wandb_run     = wandb_run,
        )
        metrics = runner.run(output_root=args.output)
        wandb_run.finish()
        logger.info(
            "  %s | pretrained_only -> W2_nu=%.4f", exp_spec.name, metrics["W2_nu"]
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    from dualwcfm.experiments.base import setup_logging
    setup_logging()

    # Table-only mode — no training
    if args.table:
        from dualwcfm.baselines.table import build_comparison_table, build_all_tables
        if args.metric == "all":
            build_all_tables(args.output)
        else:
            latex = build_comparison_table(args.output, metric=args.metric)
            print(latex)
        return

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    # Load experiment catalogue
    from dualwcfm.experiments.exp_2d       import make_2d_experiments
    from dualwcfm.experiments.exp_3d       import make_3d_experiments
    from dualwcfm.experiments.exp_gushchin import make_gushchin_experiments

    catalogues = {
        "2d":       make_2d_experiments,
        "3d":       make_3d_experiments,
        "gushchin": make_gushchin_experiments,
    }
    all_specs = catalogues[args.mode](args.output)
    matching  = _filter(all_specs, args.level, args.perturbed)

    if not matching:
        print(f"No experiments found: mode={args.mode!r} level={args.level!r}")
        sys.exit(1)

    logger.info(
        "Running baseline=%s on %d experiment(s)", args.baseline, len(matching)
    )

    if args.baseline in ("otcfm", "all"):
        _run_otcfm(args, matching, use_ot=True)
    if args.baseline in ("cfm", "all"):
        _run_otcfm(args, matching, use_ot=False)
    if args.baseline in ("pretrained", "all"):
        _run_pretrained(args, matching)
    if args.baseline in ("dsb", "all"):
        _run_dsb(args, matching)
    if args.baseline in ("lightsb", "all"):
        _run_lightsb(args, matching)
    if args.baseline in ("metaot", "all"):
        _run_metaot(args, matching)

    # Auto-generate one comparison table per available metric
    from dualwcfm.baselines.table import build_all_tables
    build_all_tables(args.output)
    logger.info("Done.  All comparison tables written to %s/", args.output)


if __name__ == "__main__":
    main()
