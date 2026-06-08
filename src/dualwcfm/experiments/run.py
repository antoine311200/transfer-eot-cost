"""dualwcfm.experiments.run
===========================
Command-line interface for running TACO experiments and ablation studies.

Usage examples
--------------
Run a single experiment::

    python -m dualwcfm.experiments.run --mode 2d --level simple
    python -m dualwcfm.experiments.run --mode 3d --level medium --perturbed
    python -m dualwcfm.experiments.run --mode image --level simple

Run all experiments sequentially::

    python -m dualwcfm.experiments.run --all

Run an ablation study (compare methods on the 2D simple experiment)::

    python -m dualwcfm.experiments.run --ablation --mode 2d --level simple

Seed sweep::

    python -m dualwcfm.experiments.run \\
        --ablation --mode 2d --level simple \\
        --methods mixture --seeds 0 1 2

Load pre-trained models instead of retraining::

    python -m dualwcfm.experiments.run --mode 2d --level simple --load-pretrained

Control training length::

    python -m dualwcfm.experiments.run --mode 2d --level simple --iterations 50

Output goes to ``results/<name>/`` by default;  use ``--output`` to override.
"""

import argparse
import sys
import os

# Ensure the src directory is on the Python path when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "python -m dualwcfm.experiments.run",
        description = "TACO experiment runner",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog      = __doc__,
    )

    # ── Experiment selection ─────────────────────────────────────────────────
    sel = parser.add_argument_group("experiment selection")
    sel.add_argument(
        "--mode", choices=["2d", "3d", "manifold", "image", "image-v2", "color",
                           "gushchin", "scalability", "singlecell", "singlecell-siren",
                           "singlecell-extra", "singlecell-otcfm"],
        help="Experiment mode.  Required unless --all is given.",
    )
    sel.add_argument(
        "--level",
        help=(
            "Filter experiments by name substring.  "
            "Standard levels: simple, medium, complex.  "
            "Gushchin examples: 8d, gmm, eps0p1, eps0p5, eps2p0."
        ),
    )
    sel.add_argument(
        "--dims", nargs="+", type=int, default=[16, 32, 64, 128],
        metavar="D",
        help="Dimensions for scalability experiments (default: 16 32 64 128).",
    )
    sel.add_argument(
        "--perturbed", action="store_true",
        help="Use the perturbed variant of the experiment.",
    )
    sel.add_argument(
        "--no-perturbed", action="store_true", dest="no_perturbed",
        help="Exclude perturbed variants (run only non-perturbed experiments).",
    )
    sel.add_argument(
        "--no-ot-cfm", action="store_true", dest="no_ot_cfm",
        help="Do not use OT-CFM for training the velocity fields between source and new source (resp. target and new target)"
    )
    sel.add_argument(
        "--use-siren", action="store_true", dest="use_siren",
        help="Replace MLP backbone with SIREN network (3D and singlecell modes).",
    )
    sel.add_argument(
        "--all", action="store_true", dest="run_all",
        help="Run every experiment (2D + 3D + image) sequentially.",
    )
    sel.add_argument(
        "--data-path", default=None, dest="data_path", metavar="PATH",
        help=(
            "Path to the Sci-Plex3 h5ad file "
            "(SrivatsanTrapnell2020_sciplex3.h5ad). "
            "Required for --mode singlecell unless pertpy is installed "
            "with the dataset cached locally."
        ),
    )

    # ── Method ──────────────────────────────────────────────────────────────
    meth = parser.add_argument_group("method")
    meth.add_argument(
        "--method", choices=["standard", "mixture", "score"], default="mixture",
        dest="method",
        help="Training method to use (default: mixture).",
    )
    meth.add_argument(
        "--sigma", type=float, default=None,
        help="Override sigma hyper-parameter of the method config.",
    )
    meth.add_argument(
        "--particle-ratio", type=float, default=None, dest="particle_ratio",
        help="Override particle_ratio for MixtureMethod (default: 0.25).",
    )
    meth.add_argument(
        "--loss", choices=["weighted", "explicit", "implicit"], default="weighted",
        dest="loss",
        help="Loss function to use (default: weighted).",
    )

    # ── Training ─────────────────────────────────────────────────────────────
    train = parser.add_argument_group("training")
    train.add_argument(
        "--iterations", type=int, default=None,
        help="Override the number of outer training iterations.",
    )
    train.add_argument(
        "--batch-size", type=int, default=None, dest="batch_size",
        help="Override the batch size.",
    )
    train.add_argument(
        "--load-pretrained", action="store_true", dest="load_pretrained",
        help=(
            "Skip marginal flow + pretrain phases and load previously saved "
            "models from <output>/<name>/models/."
        ),
    )
    train.add_argument(
        "--pretrain-dir", default=None, dest="pretrain_dir", metavar="PATH",
        help=(
            "Path to a directory containing pre-trained .pth weight files "
            "(initial_forward_model.pth, initial_mu_model.pth, etc.). "
            "The pretrain phase is skipped and these weights are used instead. "
            "Takes precedence over --load-pretrained."
        ),
    )
    train.add_argument(
        "--no-save-models", action="store_false", dest="save_models",
        help="Do not write model checkpoints to disk.",
    )
    train.add_argument(
        "--checkpoint-every", type=int, default=0, dest="checkpoint_every", metavar="N",
        help=(
            "Save model checkpoints every N iterations, plus the first and last. "
            "0 = disabled (default)."
        ),
    )
    train.add_argument(
        "--checkpoint-coupling", action="store_true", dest="checkpoint_coupling",
        help="Also checkpoint the coupling dataset at each checkpoint iteration (can be large).",
    )
    train.add_argument(
        "--seed", type=int, default=None,
        help="Global random seed for reproducibility in single-experiment runs.",
    )

    # ── Ablation ─────────────────────────────────────────────────────────────
    abl = parser.add_argument_group("ablation study")
    abl.add_argument(
        "--ablation", action="store_true",
        help="Run an ablation study comparing methods instead of a single run.",
    )
    abl.add_argument(
        "--ablation-type",
        choices=["method", "particle_ratio", "sigma", "iterations", "seeds", "all"],
        default="method", dest="ablation_type",
        help=(
            "Which ablation to run (default: method). "
            "method=Standard vs Mixture vs Score, "
            "particle_ratio=sweep alpha in {0,0.1,0.25,0.5,1.0}, "
            "sigma=sweep sigma in {0.001,0.01,0.05,0.1,0.5}, "
            "iterations=sweep N in {10,25,50,100,200}, "
            "seeds=multi-seed stability, "
            "all=run every pre-defined ablation suite."
        ),
    )
    abl.add_argument(
        "--methods", nargs="+",
        choices=["standard", "mixture", "score"],
        default=["standard", "mixture", "score"],
        help="Methods to compare in an ablation study (default: all three).",
    )
    abl.add_argument(
        "--seeds", nargs="+", type=int, default=[0],
        help="Random seeds for the ablation study (default: [0]).",
    )
    abl.add_argument(
        "--ablation-name", default=None, dest="ablation_name",
        help="Name for the ablation suite output directory.  Auto-generated if omitted.",
    )

    # ── Weights & Biases ─────────────────────────────────────────────────────
    wb = parser.add_argument_group("Weights & Biases (optional)")
    wb.add_argument(
        "--wandb-project", default=None, dest="wandb_project", metavar="PROJECT",
        help="W&B project name (e.g. 'Data-Based EOT').  Omit to disable W&B logging.",
    )
    wb.add_argument(
        "--wandb-entity", default=None, dest="wandb_entity", metavar="ENTITY",
        help="W&B entity (username or team, e.g. 'antoine311200').  Defaults to logged-in user.",
    )
    wb.add_argument(
        "--wandb-run", default=None, dest="wandb_run_name", metavar="RUN",
        help="W&B run display name.  Defaults to the experiment name.",
    )
    wb.add_argument(
        "--wandb-offline", action="store_true", dest="wandb_offline",
        help=(
            "Log W&B data offline (no network calls).  "
            "Sync later with: wandb sync wandb/offline-run-*/"
        ),
    )

    # ── Output ───────────────────────────────────────────────────────────────
    out = parser.add_argument_group("output")
    out.add_argument(
        "--output", default="results",
        help="Root directory for experiment output (default: 'results').",
    )
    out.add_argument(
        "--iter-subdir", action="store_true", dest="iter_subdir",
        help="Save each run in a subdirectory iter_N inside the experiment folder.",
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers: build method config from CLI args
# ---------------------------------------------------------------------------

def _build_method_cfg(method_name: str, args: argparse.Namespace):
    """Construct a MethodConfig from CLI arguments."""
    from dualwcfm.methods.standard import StandardConfig
    from dualwcfm.methods.mixture import MixtureConfig
    from dualwcfm.methods.score import ScoreConfig

    _shared = dict(
        sigma           = 0.05,
        sigma2          = 0.05,
        loss            = "weighted",
        l2_reg          = 0,
        lip_reg         = 0,
        mean_reg        = 0,
        use_noisy_field = False,
        lr_cosine_decay = False,
        lr_min_factor   = 0.5,
    )
    defaults = {
        "standard": StandardConfig(**_shared),
        "mixture":  MixtureConfig( **_shared, particle_ratio=0.25),
        "score":    ScoreConfig(   **_shared),
    }
    cfg = defaults[method_name]

    if args.sigma is not None:
        cfg.sigma  = args.sigma
        cfg.sigma2 = args.sigma
    if args.particle_ratio is not None and hasattr(cfg, "particle_ratio"):
        cfg.particle_ratio = args.particle_ratio
    if args.loss is not None:
        cfg.loss = args.loss

    print(f"Using method config: {cfg}")

    return cfg


# ---------------------------------------------------------------------------
# Helpers: filter catalogue
# ---------------------------------------------------------------------------

def _filter_catalogue(catalogue, level, perturbed, no_perturbed=False):
    """Return specs matching *level* / *perturbed* / *no_perturbed* filters."""
    parts = []
    if level:
        parts.append(level)
    if perturbed:
        parts.append("perturbed")

    matching = catalogue if not parts else [s for s in catalogue if all(p in s.name for p in parts)]

    if no_perturbed:
        matching = [s for s in matching if "perturbed" not in s.name]

    return matching


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    from dualwcfm.experiments.exp_2d            import make_2d_experiments
    from dualwcfm.experiments.exp_3d            import make_3d_experiments
    from dualwcfm.experiments.exp_3d_manifold   import make_3d_manifold_experiments
    from dualwcfm.experiments.exp_image         import make_image_experiments, make_image_experiments_v2
    from dualwcfm.experiments.exp_color         import make_color_experiments
    from dualwcfm.experiments.exp_gushchin      import make_gushchin_experiments
    from dualwcfm.experiments.exp_scalability   import make_scalability_experiments
    from dualwcfm.experiments.exp_singlecell    import (
        make_singlecell_experiments, make_singlecell_siren_experiments,
        make_singlecell_extra_experiments, make_singlecell_otcfm_baselines,
    )
    from dualwcfm.experiments.base              import run_experiment

    dims = tuple(getattr(args, "dims", [16, 32, 64, 128]))

    # ── Catalogue factory ────────────────────────────────────────────────────
    catalogue_factories = {
        "2d":          make_2d_experiments,
        "3d":          lambda out: make_3d_experiments(out, use_siren=getattr(args, "use_siren", False)),
        "manifold":    make_3d_manifold_experiments,
        "image":       make_image_experiments,
        "image-v2":    make_image_experiments_v2,
        "color":       make_color_experiments,
        "gushchin":    make_gushchin_experiments,
        "scalability": lambda out: make_scalability_experiments(out, dims=dims),
        "singlecell":  lambda out: make_singlecell_experiments(
            data_path=getattr(args, "data_path", None), output_root=out
        ),
        "singlecell-siren": lambda out: make_singlecell_siren_experiments(
            data_path=getattr(args, "data_path", None), output_root=out
        ),
        "singlecell-extra": lambda out: make_singlecell_extra_experiments(
            data_path=getattr(args, "data_path", None), output_root=out
        ),
        "singlecell-otcfm": lambda out: make_singlecell_otcfm_baselines(
            data_path=getattr(args, "data_path", None), output_root=out
        ),
    }

    # ── Run everything ───────────────────────────────────────────────────────
    if args.run_all:
        all_specs = (
            make_2d_experiments(args.output)
            + make_3d_experiments(args.output)
            + make_image_experiments(args.output)
        )
        if args.ablation:
            # One AblationSuite per spec, comparing all requested methods.
            for spec in all_specs:
                _apply_train_overrides(spec, args)
                _run_ablation(args, [spec])
        else:
            for spec in all_specs:
                _apply_train_overrides(spec, args)
                spec.method_cfg = _build_method_cfg(args.method, args)
                run_experiment(
                    spec,
                    retrain          = not args.load_pretrained,
                    save_models      = args.save_models,
                    pretrain_dir     = args.pretrain_dir,
                    wandb_project    = getattr(args, "wandb_project", None),
                    wandb_entity     = getattr(args, "wandb_entity", None),
                    wandb_run_name   = getattr(args, "wandb_run_name", None),
                    wandb_mode       = "offline" if getattr(args, "wandb_offline", False) else None,
                )
        return

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    catalogue = catalogue_factories[args.mode](args.output)
    matching  = _filter_catalogue(catalogue, args.level, args.perturbed, getattr(args, "no_perturbed", False))

    if not matching:
        print(
            f"No experiments match: mode={args.mode!r}, level={args.level!r}, "
            f"perturbed={args.perturbed}"
        )
        sys.exit(1)

    # ── Pre-defined ablation suites ──────────────────────────────────────────
    if args.ablation and args.ablation_type != "method":
        _run_predefined_ablation(args)
        return

    # ── Method-comparison ablation ───────────────────────────────────────────
    if args.ablation:
        _run_ablation(args, matching)
        return

    # ── Single experiment ────────────────────────────────────────────────────
    _seed = getattr(args, "seed", None)
    if _seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)

    for spec in matching:
        _apply_train_overrides(spec, args)
        # spec.method_cfg = _build_method_cfg(args.method, args)
        run_experiment(
            spec,
            retrain          = not args.load_pretrained,
            save_models      = args.save_models,
            pretrain_dir     = args.pretrain_dir,
            wandb_project    = getattr(args, "wandb_project", None),
            wandb_entity     = getattr(args, "wandb_entity", None),
            wandb_run_name   = getattr(args, "wandb_run_name", None),
            wandb_mode       = "offline" if getattr(args, "wandb_offline", False) else None,
        )


def _apply_train_overrides(spec, args: argparse.Namespace) -> None:
    """Apply CLI training overrides (iterations, batch_size, checkpointing) to *spec* in place."""
    if args.iterations is not None:
        spec.train_cfg.iterations = args.iterations
        if getattr(args, "iter_subdir", False):
            spec.output_dir_override = os.path.join(spec.output_dir, f"iter_{args.iterations}")
    if args.batch_size is not None:
        spec.train_cfg.batch_size = args.batch_size
    if args.checkpoint_every > 0:
        spec.train_cfg.save_model_checkpoints   = True
        spec.train_cfg.checkpoint_every         = args.checkpoint_every
        if args.checkpoint_coupling:
            spec.train_cfg.save_coupling_checkpoints = True
    if args.sigma is not None:
        spec.method_cfg.sigma  = args.sigma
    if args.sigma is not None and hasattr(spec.method_cfg, "sigma2"):
         spec.method_cfg.sigma2 = args.sigma
    if args.particle_ratio is not None and hasattr(spec.method_cfg, "particle_ratio"):
        spec.method_cfg.particle_ratio = args.particle_ratio
    if args.loss is not None:
        spec.method_cfg.loss = args.loss
    if args.no_ot_cfm:
        spec.method_cfg.use_ot_pretrain = False



def _run_ablation(args: argparse.Namespace, base_specs) -> None:
    """Build and run an AblationSuite from CLI arguments."""
    from dualwcfm.experiments.ablation import AblationSuite, AblationVariant, AblationRunner
    from dualwcfm.methods.standard import StandardConfig
    from dualwcfm.methods.mixture import MixtureConfig
    from dualwcfm.methods.score import ScoreConfig

    # Use the first matching spec as the base template
    base_spec = base_specs[0]
    _apply_train_overrides(base_spec, args)

    # Build variants: methods × seeds
    variants = []
    for method_name in args.methods:
        for seed in args.seeds:
            cfg   = _build_method_cfg(method_name, args)
            label = method_name.capitalize()
            name  = f"{method_name}_seed{seed}" if len(args.seeds) > 1 else method_name
            variants.append(
                AblationVariant(
                    name       = name,
                    method_cfg = cfg,
                    seed       = seed,
                    label      = label if len(args.seeds) == 1 else f"{label} (seed {seed})",
                )
            )

    suite_name = args.ablation_name or f"{base_spec.name}_ablation"
    suite = AblationSuite(
        name      = suite_name,
        base_spec = base_spec,
        variants  = variants,
    )

    runner = AblationRunner(
        suite          = suite,
        output_root    = args.output,
        retrain        = not args.load_pretrained,
        save_models    = args.save_models,
        wandb_project  = getattr(args, "wandb_project", None),
        wandb_entity   = getattr(args, "wandb_entity", None),
    )
    runner.run()

    # Additional plots
    runner.plot_final_metrics_bar()
    runner.plot_final_distributions()
    if base_spec.mode == "2d":
        runner.plot_checkpoint_evolution()
    if len(args.seeds) > 1:
        runner.plot_seed_aggregate(metric="W2_nu")


def _run_predefined_ablation(args: argparse.Namespace) -> None:
    """Run one of the pre-defined ablation suites (E7a-f)."""
    from dualwcfm.experiments.ablation import AblationRunner
    from dualwcfm.experiments.exp_ablations import (
        make_particle_ratio_ablation,
        make_sigma_ablation,
        make_iterations_ablation,
        make_seed_ablation,
        make_method_ablation,
        make_all_ablation_suites,
    )

    atype = args.ablation_type
    out   = args.output

    if atype == "particle_ratio":
        suites = [make_particle_ratio_ablation(out)]
    elif atype == "sigma":
        suites = [make_sigma_ablation(out)]
    elif atype == "iterations":
        suites = [make_iterations_ablation(out)]
    elif atype == "seeds":
        suites = [make_seed_ablation(out)]
    elif atype == "all":
        suites = make_all_ablation_suites(out)
    else:
        suites = []

    for suite in suites:
        runner = AblationRunner(
            suite          = suite,
            output_root    = out,
            retrain        = not args.load_pretrained,
            save_models    = args.save_models,
            wandb_project  = getattr(args, "wandb_project", None),
            wandb_entity   = getattr(args, "wandb_entity", None),
        )
        runner.run()
        runner.plot_convergence()
        runner.plot_weight_evolution()
        runner.plot_final_metrics_bar()
        runner.export_latex_table()
        if suite.base_spec.mode == "2d":
            runner.plot_final_distributions()
            runner.plot_checkpoint_evolution()
        if atype in ("seeds", "all"):
            runner.plot_seed_aggregate(metric="W2_nu")


if __name__ == "__main__":
    main()
