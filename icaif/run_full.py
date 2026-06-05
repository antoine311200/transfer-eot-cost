"""Full-quality runner for IV surface experiments — TACO Finance ICAIF.

Epoch counts scaled between the ultra-fast diagnostic and the over-generous
defaults.  Iterations capped at 25: the diagnostic shows convergence by
iteration 12 and weight drift accelerates after that.

Usage:
    python -m icaif.run_full --exp synthetic
    python -m icaif.run_full --exp spx
    python -m icaif.run_full --exp all
"""
from __future__ import annotations
import argparse, dataclasses, logging, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib; matplotlib.use("Agg")

WANDB_PROJECT = "TACO-Finance-ICAIF"
OUTPUT_ROOT   = "results/icaif_full"

FULL_OVERRIDES = dict(
    marginal_epochs = 1500,
    pretrain_epochs = 2500,
    dual_epochs     = 40,
    cfm_epochs      = 60,
    iterations      = 25,
    batch_size      = 1024,
)


def patch_cfg(cfg, extra: dict | None = None):
    for k, v in {**FULL_OVERRIDES, **(extra or {})}.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def run_one(spec, run_name: str):
    from icaif.run import run_finance_experiment
    run_finance_experiment(
        spec,
        retrain=True,
        save_models=True,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=run_name,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["synthetic", "spx", "all"], default="all")
    args = p.parse_args()

    from dualwcfm.experiments.base import get_device
    device = get_device()

    if args.exp in ("synthetic", "all"):
        from icaif.experiments.iv_surface import make_iv_surface_experiment
        spec = make_iv_surface_experiment(OUTPUT_ROOT, device=device)
        spec.train_cfg = patch_cfg(spec.train_cfg, {"dataset_size": 3000})
        run_one(spec, "iv_surface_2f_full")

    if args.exp in ("spx", "all"):
        from icaif.experiments.iv_surface import make_iv_surface_spx_experiment
        spec = make_iv_surface_spx_experiment(
            OUTPUT_ROOT, device=device, cache_dir="data/spx_cache"
        )
        # Calm window now 2017-2019 (~752 pairs) — use full dataset
        spec.train_cfg = patch_cfg(spec.train_cfg, {
            "dataset_size": min(len(spec.initial_coupling), 752),
            "batch_size":   256,
        })
        run_one(spec, "iv_surface_spx_full_v2")


if __name__ == "__main__":
    main()
