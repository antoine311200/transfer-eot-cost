"""dualwcfm.experiments.grid_search
=====================================
Hyperparameter grid search for 2-D and 3-D experiments.

The key design principle is **amortised pretraining**: the expensive marginal
and pretrain phases are run only once per base experiment; all grid-search
runs share those weights and only iterate the adaptation loop.

Usage examples
--------------
# Grid search on 2d_simple with 2 values for sigma and particle_ratio:
  python -m dualwcfm.experiments.grid_search --mode 2d --level simple \\
      --sigma 0.01,0.05 --particle-ratio 0.1,0.25 \\
      --output results_grid

# Provide pretrained weights yourself (skip pretraining entirely):
  python -m dualwcfm.experiments.grid_search --mode 2d --level simple \\
      --pretrain-dir results/2d_simple/models \\
      --sigma 0.01,0.05,0.1 --particle-ratio 0.1,0.25,0.5 \\
      --output results_grid

# Log to Weights & Biases (all runs grouped under one sweep):
  python -m dualwcfm.experiments.grid_search --mode 2d --level simple \\
      --sigma 0.01,0.05 --particle-ratio 0.1,0.25 \\
      --wandb-project dualwcfm-sweep --wandb-entity <entity> \\
      --output results_grid

# Use wandb sweep agent (wandb controls sampling):
  python -m dualwcfm.experiments.grid_search --mode 2d --level simple \\
      --sigma 0.01,0.05 --particle-ratio 0.1,0.25 \\
      --wandb-project dualwcfm-sweep --wandb-sweep \\
      --sweep-method random --output results_grid
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import itertools
import logging
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from dualwcfm.config import FeedForwardConfig, TrainingConfig, VisualizeConfig
from dualwcfm.experiments.base import (
    ExperimentSpec,
    get_device,
    run_experiment,
    save_metrics_plot,
    setup_logging,
)
from dualwcfm.experiments.exp_2d import make_2d_experiments
from dualwcfm.experiments.exp_3d import make_3d_experiments
from dualwcfm.methods.mixture import MixtureConfig

logger = logging.getLogger("dualwcfm")


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def _floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",")]

def _ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",")]

def _bools(s: str) -> List[bool]:
    out = []
    for x in s.split(","):
        x = x.strip().lower()
        if x in ("true", "1", "yes"):
            out.append(True)
        elif x in ("false", "0", "no"):
            out.append(False)
        else:
            raise argparse.ArgumentTypeError(f"Cannot parse bool: {x!r}")
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "python -m dualwcfm.experiments.grid_search",
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    # ── Experiment selection ────────────────────────────────────────────
    sel = p.add_argument_group("experiment selection")
    sel.add_argument("--mode",   choices=["2d", "3d"], default="2d",
                     help="Experiment catalogue to search.")
    sel.add_argument("--level", nargs="+", default=None, metavar="SUBSTR",
                     help="One or more substring filters on experiment name. "
                          "A spec matches if ANY substring is found in its name. "
                          "E.g. --level simple medium  runs both levels with the same grid.")
    sel.add_argument("--perturbed", action="store_true",
                     help="Run only the perturbed variant(s) of the matched experiments.")
    sel.add_argument("--both", action="store_true",
                     help="Run both perturbed and non-perturbed variants. "
                          "Each gets its own pretrain phase (run once per variant, "
                          "reused across all combos of that variant).")
    sel.add_argument("--output", default="results_grid",
                     help="Root output directory for grid-search results.")
    sel.add_argument("--pretrain-dir", default=None, metavar="DIR",
                     help="Path to a models/ directory to reuse for ALL selected specs "
                          "(skip pretraining).  When running multiple specs this must contain "
                          "weights valid for every spec — prefer --reuse-from for the "
                          "multi-experiment case.")
    sel.add_argument("--reuse-from",  default="results", metavar="DIR",
                     help="Root results dir to scan for pre-existing {name}/models/ weights "
                          "per spec (default: results).  Each spec looks up its own subdir "
                          "independently.  Set to '' to always retrain.")

    # ── W&B ───────────────────────────────────────────────────────────
    wb = p.add_argument_group("Weights & Biases")
    wb.add_argument("--wandb-project", default=None,
                    help="W&B project name.  Omit to disable W&B logging.")
    wb.add_argument("--wandb-entity",  default=None,
                    help="W&B entity (username or team).")
    wb.add_argument("--wandb-sweep",   action="store_true",
                    help="Create a W&B sweep and run via wandb.agent().")
    wb.add_argument("--sweep-method",  choices=["grid", "random", "bayes"],
                    default="grid",
                    help="Sweep sampling strategy (default: grid).")
    wb.add_argument("--sweep-count",   type=int, default=None,
                    help="Max number of sweep runs (default: full grid).")

    # ── Grid parameters ────────────────────────────────────────────────
    g = p.add_argument_group(
        "grid parameters",
        "Comma-separated lists of values.  All combinations are tried (grid search).",
    )
    g.add_argument("--sigma",           type=_floats, default=[0.01, 0.05],
                   metavar="F[,F…]",    help="Bridge noise σ (method_cfg.sigma). (default: 0.01,0.05)")
    g.add_argument("--sigma2",          type=_floats, default=None,
                   metavar="F[,F…]",
                   help="CFM interpolation noise σ₂ (method_cfg.sigma2). "
                        "If omitted, mirrors --sigma values.")
    g.add_argument("--particle-ratio",  type=_floats, default=[0.1, 0.25],
                   metavar="F[,F…]",    help="New-marginal particle fraction. (default: 0.1,0.25)")
    g.add_argument("--scale-factor",    type=_floats, default=[1.0],
                   metavar="F[,F…]",    help="Data scale factor (train_cfg.scale_factor). (default: 1.0)")
    g.add_argument("--use-noisy-field", type=_bools,  default=[False],
                   metavar="B[,B…]",    help="Noisy vector field (method_cfg.use_noisy_field). (default: false)")
    g.add_argument("--dual-epochs",     type=_ints,   default=[100],
                   metavar="N[,N…]",    help="Dual potential training epochs/iter. (default: 100)")
    g.add_argument("--cfm-epochs",      type=_ints,   default=[250],
                   metavar="N[,N…]",    help="CFM training epochs/iter. (default: 250)")
    g.add_argument("--advance-steps",   type=_ints,   default=[100],
                   metavar="N[,N…]",    help="ODE steps for coupling advance. (default: 100)")
    g.add_argument("--iterations",      type=_ints,   default=[50],
                   metavar="N[,N…]",    help="Outer adaptation iterations. (default: 50)")
    g.add_argument("--lr-cosine-decay", type=_bools,  default=[False],
                   metavar="B[,B…]",    help="Enable cosine LR decay. (default: false)")
    g.add_argument("--batch-size",      type=_ints,   default=[8192],
                   metavar="N[,N…]",    help="Training batch size. (default: 8192)")
    g.add_argument("--seeds",           type=_ints,   default=[0],
                   metavar="N[,N…]",
                   help="Random seeds.  Multiple seeds run each combo N times and "
                        "aggregate metrics (mean ± std).  (default: 0)")

    return p


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

_GRID_KEYS = [
    "sigma", "sigma2", "particle_ratio", "scale_factor", "use_noisy_field",
    "dual_epochs", "cfm_epochs", "advance_steps", "iterations",
    "lr_cosine_decay", "batch_size",
]

def _build_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Return a list of hyperparameter dicts — one per grid point."""
    # sigma2 mirrors sigma when not specified explicitly
    sigma2_vals = args.sigma2 if args.sigma2 is not None else args.sigma
    grid = {
        "sigma":           args.sigma,
        "sigma2":          sigma2_vals,
        "particle_ratio":  args.particle_ratio,
        "scale_factor":    args.scale_factor,
        "use_noisy_field": args.use_noisy_field,
        "dual_epochs":     args.dual_epochs,
        "cfm_epochs":      args.cfm_epochs,
        "advance_steps":   args.advance_steps,
        "iterations":      args.iterations,
        "lr_cosine_decay": args.lr_cosine_decay,
        "batch_size":      args.batch_size,
    }
    keys, values = zip(*grid.items())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _run_name(base_name: str, combo: Dict[str, Any]) -> str:
    """Short but unique run name encoding the hyperparameter combo."""
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.0e}".replace("+0", "").replace("-0", "-").replace("e", "e")
        if isinstance(v, bool):
            return "T" if v else "F"
        return str(v)

    tag = "_".join(
        f"{k[:2]}{_fmt(v)}"
        for k, v in sorted(combo.items())
        if len(args_from_combo(combo)) > 1   # skip constant dims
        or True                               # always include
    )
    # Shorten keys: sigma→s, particle_ratio→pr, dual_epochs→de …
    tag = (tag
           .replace("si", "s")
           .replace("pa", "pr")
           .replace("du", "de")
           .replace("cf", "ce")
           .replace("ad", "as")
           .replace("it", "it")
           .replace("lr", "lrd")
           .replace("ba", "bs"))
    return f"{base_name}__{tag}"


def args_from_combo(combo):
    return combo


def _apply_combo(spec: ExperimentSpec, combo: Dict[str, Any]) -> ExperimentSpec:
    """Deep-copy *spec* and apply *combo* hyperparameters."""
    new_spec = copy.copy(spec)
    new_spec.train_cfg  = copy.copy(spec.train_cfg)
    new_spec.method_cfg = copy.copy(spec.method_cfg)

    new_spec.train_cfg.dual_epochs    = combo["dual_epochs"]
    new_spec.train_cfg.cfm_epochs     = combo["cfm_epochs"]
    new_spec.train_cfg.advance_steps  = combo["advance_steps"]
    new_spec.train_cfg.iterations     = combo["iterations"]
    new_spec.train_cfg.batch_size     = combo["batch_size"]
    new_spec.train_cfg.scale_factor   = combo["scale_factor"]
    new_spec.method_cfg.sigma          = combo["sigma"]
    new_spec.method_cfg.sigma2         = combo["sigma2"]
    new_spec.method_cfg.particle_ratio = combo["particle_ratio"]
    new_spec.method_cfg.lr_cosine_decay= combo["lr_cosine_decay"]
    new_spec.method_cfg.use_noisy_field= combo["use_noisy_field"]

    return new_spec


# ---------------------------------------------------------------------------
# Pretraining helper
# ---------------------------------------------------------------------------

def _ensure_pretrain(
    spec:          ExperimentSpec,
    pretrain_dir:  Optional[str],
    wandb_project: Optional[str],
    wandb_entity:  Optional[str],
    reuse_from:    str = "results",
) -> str:
    """Run pretraining for *spec* if *pretrain_dir* is not given; return path.

    Resolution order
    ----------------
    1. ``pretrain_dir`` (explicit path from --pretrain-dir)
    2. ``{reuse_from}/{spec.name}/models/`` — auto-detected existing weights
    3. Fresh pretrain run (saves into ``spec.output_dir/models/``)
    """
    if pretrain_dir is not None:
        if not os.path.isdir(pretrain_dir):
            raise FileNotFoundError(f"--pretrain-dir not found: {pretrain_dir}")
        logger.info("  Re-using pretrained models from %s", pretrain_dir)
        return pretrain_dir

    # Auto-detect from a previous full-experiment run
    if reuse_from:
        candidate = os.path.join(reuse_from, spec.name, "models")
        if os.path.isdir(candidate) and any(f.endswith(".pth") for f in os.listdir(candidate)):
            logger.info("  Auto-detected pretrained models at %s", candidate)
            return candidate

    logger.info("  Pretraining base experiment: %s", spec.name)
    run_experiment(
        spec,
        retrain       = True,
        save_models   = True,
        wandb_project = wandb_project,
        wandb_entity  = wandb_entity,
        wandb_run_name= f"{spec.name}__pretrain",
        wandb_group   = f"grid__{spec.name}",
    )
    return os.path.join(spec.output_dir, "models")


# ---------------------------------------------------------------------------
# Local grid search (no wandb agent)
# ---------------------------------------------------------------------------

def _seed_run_name(base_run_name: str, seed: int, multi_seed: bool) -> str:
    """Append seed suffix only when running multiple seeds."""
    return f"{base_run_name}__s{seed}" if multi_seed else base_run_name


def _aggregate_seed_metrics(
    output_root: str,
    spec_name:   str,
    combo_base_names: List[str],
    seeds:       List[int],
) -> None:
    """Read final-iteration metrics for all (combo × seed) runs and write a
    seed_summary.csv with mean ± std per combo into output_root/{spec_name}__summary/.
    """
    multi_seed = len(seeds) > 1
    summary_dir = os.path.join(output_root, f"{spec_name}__summary")
    os.makedirs(summary_dir, exist_ok=True)

    rows = []
    for base_name in combo_base_names:
        seed_rows: List[Dict[str, float]] = []
        for seed in seeds:
            run_name = _seed_run_name(base_name, seed, multi_seed)
            csv_path = os.path.join(output_root, run_name, "metrics.csv")
            if not os.path.exists(csv_path):
                logger.warning("  Missing metrics.csv for %s — skipping in aggregation", run_name)
                continue
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                all_rows = list(reader)
            if all_rows:
                # Use last row = final iteration metrics
                last = {k: float(v) for k, v in all_rows[-1].items()
                        if v not in ("", None) and k != "iteration"}
                seed_rows.append(last)

        if not seed_rows:
            continue

        # Aggregate across seeds
        keys = seed_rows[0].keys()
        agg: Dict[str, Any] = {"combo": base_name, "n_seeds": len(seed_rows)}
        for k in keys:
            vals = np.array([r[k] for r in seed_rows if k in r], dtype=float)
            agg[f"{k}_mean"] = float(vals.mean())
            agg[f"{k}_std"]  = float(vals.std())
        rows.append(agg)

    if not rows:
        return

    out_path = os.path.join(summary_dir, "seed_summary.csv")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("  Seed aggregation summary -> %s", out_path)

    # Also print a compact table to stdout
    metric_cols = [c for c in fieldnames if c.endswith("_mean")]
    header = f"{'combo':<55}" + "".join(f"  {c[:-5]:>18}" for c in metric_cols)
    logger.info("\n%s\n%s", header, "-" * len(header))
    for row in rows:
        line = f"{row['combo']:<55}"
        for c in metric_cols:
            std_key = c[:-4] + "std"
            line += f"  {row[c]:>8.4f}±{row.get(std_key, 0.0):>6.4f}"
        logger.info(line)


def run_grid_search(
    specs:          List[ExperimentSpec],
    combos:         List[Dict[str, Any]],
    output_root:    str,
    pretrain_dir:   Optional[str],
    wandb_project:  Optional[str],
    wandb_entity:   Optional[str],
    reuse_from:     str = "results",
    seeds:          List[int] = (0,),
) -> None:
    failed = []
    multi_seed = len(seeds) > 1

    for spec in specs:
        n_total = len(combos) * len(seeds)
        logger.info("\n%s\nGrid search: %s  (%d combos × %d seeds = %d runs)\n%s",
                    "=" * 68, spec.name, len(combos), len(seeds), n_total, "=" * 68)

        # 1. Pretrain once (or reuse)
        pt_dir = _ensure_pretrain(spec, pretrain_dir, wandb_project, wandb_entity, reuse_from)

        # 2. Iterate over combos × seeds
        combo_base_names = []
        run_idx = 0
        for combo in combos:
            base_name = _run_name(spec.name, combo)
            combo_base_names.append(base_name)

            for seed in seeds:
                run_idx += 1
                run_name = _seed_run_name(base_name, seed, multi_seed)
                logger.info("  [%d/%d] %s  (seed=%d)", run_idx, n_total, run_name, seed)

                gs_spec = _apply_combo(spec, combo)
                gs_spec = dataclasses.replace(gs_spec, name=run_name, output_root=output_root)

                torch.manual_seed(seed)
                np.random.seed(seed)

                try:
                    run_experiment(
                        gs_spec,
                        retrain             = False,
                        save_models         = True,
                        pretrain_dir        = pt_dir,
                        wandb_project       = wandb_project,
                        wandb_entity        = wandb_entity,
                        wandb_run_name      = run_name,
                        wandb_group         = f"grid__{spec.name}",
                        wandb_extra_config  = {"base_experiment": spec.name,
                                               "seed": seed,
                                               **combo},
                        wandb_tags          = [spec.name],
                    )
                except Exception as exc:
                    logger.error("  FAILED %s: %s", run_name, exc)
                    failed.append(run_name)

        # 3. Aggregate across seeds (if multiple seeds were used)
        if multi_seed:
            logger.info("\nAggregating metrics across %d seeds...", len(seeds))
            _aggregate_seed_metrics(output_root, spec.name, combo_base_names, seeds)

    if failed:
        logger.warning("Failed runs:\n  %s", "\n  ".join(failed))
    else:
        logger.info("All grid-search runs completed successfully.")


# ---------------------------------------------------------------------------
# W&B sweep
# ---------------------------------------------------------------------------

def _build_sweep_config(
    combo_keys_values: Dict[str, List[Any]],
    method: str,
) -> Dict:
    """Build a wandb sweep configuration dict."""
    params = {}
    for key, vals in combo_keys_values.items():
        params[key] = {"values": vals}
    return {
        "method":     method,
        "parameters": params,
        "metric":     {"name": "sinkhorn_div", "goal": "minimize"},
    }


def run_wandb_sweep(
    specs:        List[ExperimentSpec],
    args:         argparse.Namespace,
    combos_dict:  Dict[str, List[Any]],
    pretrain_dir: Optional[str],
) -> None:
    """Create and run a W&B sweep for the given experiment specs."""
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb is required for --wandb-sweep. Install with: pip install wandb")

    for spec in specs:
        logger.info("Creating W&B sweep for %s", spec.name)

        # Pretrain once
        pt_dir = _ensure_pretrain(spec, pretrain_dir, args.wandb_project, args.wandb_entity)

        sweep_config = _build_sweep_config(combos_dict, args.sweep_method)
        sweep_id = wandb.sweep(
            sweep_config,
            project = args.wandb_project,
            entity  = args.wandb_entity,
        )
        logger.info("  Sweep ID: %s", sweep_id)

        # Capture current spec and args for the closure
        _spec     = spec
        _pt_dir   = pt_dir
        _out_root = args.output

        def sweep_train_fn():
            run = wandb.init(
                project = args.wandb_project,
                entity  = args.wandb_entity,
                group   = f"grid__{_spec.name}",
            )
            combo = {
                "sigma":            run.config.get("sigma",            args.sigma[0]),
                "particle_ratio":   run.config.get("particle_ratio",   args.particle_ratio[0]),
                "dual_epochs":      run.config.get("dual_epochs",      args.dual_epochs[0]),
                "cfm_epochs":       run.config.get("cfm_epochs",       args.cfm_epochs[0]),
                "advance_steps":    run.config.get("advance_steps",    args.advance_steps[0]),
                "iterations":       run.config.get("iterations",       args.iterations[0]),
                "lr_cosine_decay":  run.config.get("lr_cosine_decay",  args.lr_cosine_decay[0]),
                "batch_size":       run.config.get("batch_size",       args.batch_size[0]),
            }
            run_name = _run_name(_spec.name, combo)
            if run.name != run_name:
                run.name = run_name
                run.save()
            gs_spec  = _apply_combo(_spec, combo)
            gs_spec  = dataclasses.replace(gs_spec, name=run_name, output_root=_out_root)

            # Run experiment — don't re-init wandb (agent already did it)
            from dualwcfm.experiments.base import (
                _make_mlp_strategy, METHOD_REGISTRY, setup_logging,
            )
            from dualwcfm.flow_matching import ConditionalFlowMatching
            from dualwcfm.logging_visualizer import LoggingVisualizer, LoggingVisualizer3D
            from dualwcfm.trainer import Trainer
            from dualwcfm import wandb_logger as _wl

            setup_logging(gs_spec.output_dir)
            os.makedirs(gs_spec.output_dir, exist_ok=True)
            model_dir = os.path.join(gs_spec.output_dir, "models")
            device    = gs_spec.train_cfg.device

            # Copy pretrained weights
            os.makedirs(model_dir, exist_ok=True)
            for fname in os.listdir(_pt_dir):
                if fname.endswith(".pth"):
                    shutil.copy2(os.path.join(_pt_dir, fname), model_dir)

            method_cls = METHOD_REGISTRY.get(type(gs_spec.method_cfg))
            method     = method_cls(gs_spec.method_cfg)
            strategy   = _make_mlp_strategy(
                gs_spec.model_cfg.flat_dimension, gs_spec.model_cfg.hidden_size,
                gs_spec.bidirectional, device,
            )
            strategy.config = gs_spec.model_cfg

            # Wrap the active wandb run
            wb_run = _wl._WandbRun(run)

            if gs_spec.mode == "2d":
                visualizer = LoggingVisualizer(gs_spec.vis_cfg, gs_spec.transport_map,
                                               gs_spec.output_dir, wandb_run=wb_run,
                                               inverse_transport_map=gs_spec.inverse_transport_map)
            else:
                visualizer = LoggingVisualizer3D(gs_spec.vis_cfg, gs_spec.transport_map,
                                                 gs_spec.output_dir, wandb_run=wb_run,
                                                 inverse_transport_map=gs_spec.inverse_transport_map)

            trainer = Trainer(
                initial_coupling = gs_spec.initial_coupling,
                new_source       = gs_spec.new_source,
                new_target       = gs_spec.new_target,
                flow_matching    = ConditionalFlowMatching(),
                strategy         = strategy,
                visualizer       = visualizer,
                config           = gs_spec.train_cfg,
                method           = method,
                device           = device,
            )
            trainer.train(retrain=False, save=model_dir, save_checkpoints=None)
            visualizer.make_timelapse_all(fps=3)
            save_metrics_plot(gs_spec.output_dir)
            wandb.finish()

        count = args.sweep_count or len(_build_grid(args))
        wandb.agent(sweep_id, function=sweep_train_fn, count=count)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    setup_logging()

    # ── Select experiments ───────────────────────────────────────────────
    factory = make_2d_experiments if args.mode == "2d" else make_3d_experiments
    all_specs = factory(args.output)

    # Filter by level substrings (any match)
    if args.level:
        all_specs = [s for s in all_specs if any(lvl in s.name for lvl in args.level)]

    # Filter by perturbed/non-perturbed:
    # --both       → keep all (perturbed + non-perturbed); each gets its own pretrain
    # --perturbed  → keep only specs whose name contains "perturbed"
    # (default)    → keep only specs whose name does NOT contain "perturbed"
    if not args.both:
        if args.perturbed:
            all_specs = [s for s in all_specs if "perturbed" in s.name]
        else:
            all_specs = [s for s in all_specs if "perturbed" not in s.name]

    if not all_specs:
        raise SystemExit(
            f"No experiments match --level={args.level!r}, --mode={args.mode}, "
            f"perturbed={args.perturbed}"
        )

    logger.info("Selected experiments: %s", [s.name for s in all_specs])

    combos = _build_grid(args)
    logger.info("Grid: %d combinations", len(combos))

    # ── Run ─────────────────────────────────────────────────────────────
    if args.wandb_sweep:
        if not args.wandb_project:
            raise SystemExit("--wandb-project is required when using --wandb-sweep")
        combos_dict = {
            "sigma":           args.sigma,
            "particle_ratio":  args.particle_ratio,
            "dual_epochs":     args.dual_epochs,
            "cfm_epochs":      args.cfm_epochs,
            "advance_steps":   args.advance_steps,
            "iterations":      args.iterations,
            "lr_cosine_decay": args.lr_cosine_decay,
            "batch_size":      args.batch_size,
        }
        run_wandb_sweep(all_specs, args, combos_dict, args.pretrain_dir)
    else:
        run_grid_search(
            all_specs,
            combos,
            output_root   = args.output,
            pretrain_dir  = args.pretrain_dir,
            wandb_project = args.wandb_project,
            wandb_entity  = args.wandb_entity,
            reuse_from    = args.reuse_from,
            seeds         = args.seeds,
        )


if __name__ == "__main__":
    main()
