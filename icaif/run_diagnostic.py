"""Diagnostic runner — reduced epochs, quick convergence check.

Usage:
    python -m icaif.run_diagnostic --exp synthetic
    python -m icaif.run_diagnostic --exp spx
    python -m icaif.run_diagnostic --exp all
"""
from __future__ import annotations
import argparse
import dataclasses
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib
matplotlib.use("Agg")

WANDB_PROJECT = "TACO-Finance-ICAIF"
OUTPUT_ROOT   = "results/icaif_diag"

# ── Diagnostic hyperparameters ────────────────────────────────────────────────
DIAG_OVERRIDES = dict(
    marginal_epochs = 500,
    pretrain_epochs = 1000,
    dual_epochs     = 15,
    cfm_epochs      = 30,
    iterations      = 12,
    dataset_size    = 1024,
    batch_size      = 512,
)


def patch_cfg(cfg):
    for k, v in DIAG_OVERRIDES.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def run_one(spec, run_name: str):
    from icaif.run import run_finance_experiment
    spec.train_cfg = patch_cfg(spec.train_cfg)
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
        run_one(spec, "iv_surface_2f_diag")

    if args.exp in ("spx", "all"):
        from icaif.experiments.iv_surface import make_iv_surface_spx_experiment
        spec = make_iv_surface_spx_experiment(
            OUTPUT_ROOT, device=device, cache_dir="data/spx_cache"
        )
        # SPX has fewer pairs — keep dataset_size at actual length
        spec.train_cfg = patch_cfg(spec.train_cfg)
        spec.train_cfg.dataset_size = min(len(spec.initial_coupling), 250)
        run_one(spec, f"iv_surface_spx_diag_{spec.name}")


if __name__ == "__main__":
    main()
