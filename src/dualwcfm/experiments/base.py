"""dualwcfm.experiments.base
============================
Core infrastructure shared by all experiment types.

Provides:
  - get_device()          – select CUDA or CPU
  - setup_logging()       – configure the ``dualwcfm`` logger
  - ExperimentSpec        – single fully-specified experiment
  - run_experiment()      – instantiate all objects and call trainer.train()
  - save_metrics_plot()   – post-training metric summaries from metrics.csv
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Type

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dualwcfm.config import FeedForwardConfig, TrainingConfig, VisualizeConfig
from dualwcfm.distributions.coupling import CouplingDistribution
from dualwcfm.distributions.distribution import Distribution
from dualwcfm.flow_matching import ConditionalFlowMatching
from dualwcfm.logging_visualizer import (
    LoggingVisualizer,
    LoggingVisualizer3D,
    LoggingVisualizerImage,
)
from dualwcfm.methods.base import Method, MethodConfig
from dualwcfm.methods.mixture import MixtureConfig, MixtureMethod
from dualwcfm.methods.score import ScoreConfig, ScoreMethod
from dualwcfm.methods.standard import StandardConfig, StandardMethod
from dualwcfm.mlp import torch_wrapper
from dualwcfm.strategy import BidirectionalStrategy, Strategy
from dualwcfm.trainer import Trainer

logger = logging.getLogger("dualwcfm")


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Optional[str] = None) -> None:
    """Configure the ``dualwcfm`` logger with a StreamHandler and optional FileHandler.

    Safe to call multiple times – handlers are added only if not already present.
    """
    root = logging.getLogger("dualwcfm")
    root.setLevel(logging.INFO)

    if not root.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        root.addHandler(stream_handler)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "training.log")
        # Avoid duplicate file handlers across repeated calls
        existing_paths = {
            getattr(h, "baseFilename", None) for h in root.handlers
        }
        if os.path.abspath(log_path) not in existing_paths:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

#: Maps config type → (Method class, config class) for automatic dispatch.
METHOD_REGISTRY: Dict[Type[MethodConfig], Type[Method]] = {
    StandardConfig: StandardMethod,
    MixtureConfig:  MixtureMethod,
    ScoreConfig:    ScoreMethod,
}


# ---------------------------------------------------------------------------
# ExperimentSpec
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSpec:
    """Complete specification for a single training run.

    All hyperparameters, distributions, and output locations live here so that
    experiments are fully reproducible from a single object.

    Attributes
    ----------
    name:
        Unique human-readable identifier used as the output sub-directory name.
    mode:
        Dimensionality tag – ``"2d"``, ``"3d"``, or ``"image"``.  Determines
        which :class:`~dualwcfm.logging_visualizer.LoggingVisualizer` subclass
        is instantiated.
    initial_coupling:
        Reference coupling π that encodes the transport dynamic to learn.
    new_source, new_target:
        New marginals (μ', ν') on which the learned dynamic is applied.
    transport_map:
        Optional ground-truth T: ℝ^d → ℝ^d for diagnostic overlay plots.
    train_cfg, model_cfg, vis_cfg:
        Training / model / visualisation hyperparameters.
    method_cfg:
        Config for the training method (determines which Method is used).
    bidirectional:
        Whether to use :class:`~dualwcfm.strategy.BidirectionalStrategy`.
    output_root:
        Root directory; results land in ``{output_root}/{name}/``.
    """
    name:             str
    mode:             str
    initial_coupling: CouplingDistribution
    new_source:       Distribution
    new_target:       Distribution
    train_cfg:        TrainingConfig
    model_cfg:        FeedForwardConfig
    vis_cfg:          VisualizeConfig
    # Ground-truth transport maps for diagnostic map-error metrics.
    # transport_map:         T   s.t. y ≈ T(x)   (forward direction)
    # inverse_transport_map: T⁻¹ s.t. x ≈ T⁻¹(y) (backward direction)
    transport_map:         Optional[Callable] = None
    inverse_transport_map: Optional[Callable] = None
    method_cfg:       MethodConfig = field(default_factory=MixtureConfig)
    bidirectional:    bool = True
    output_root:      str = "results"

    @property
    def output_dir(self) -> str:
        return os.path.join(self.output_root, self.name)


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def _make_mlp_strategy(
    dim: int,
    hidden: int,
    bidirectional: bool = True,
    device: Optional[torch.device] = None,
) -> Strategy:
    """Build a MLP-based Strategy, move it to *device*, and return it."""
    device = device or get_device()
    cfg = FeedForwardConfig(dimension=dim, hidden_size=hidden)
    cls = BidirectionalStrategy if bidirectional else Strategy
    strategy = cls(torch_wrapper, cfg)
    strategy.to_device(device)
    return strategy


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------

def run_experiment(
    spec: ExperimentSpec,
    retrain: bool = True,
    save_models: bool = True,
    pretrain_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_extra_config: Optional[Dict] = None,
    wandb_tags: Optional[list] = None,
) -> None:
    """Build all objects from *spec* and run the full training loop.

    Parameters
    ----------
    spec:
        Fully-specified experiment.
    retrain:
        If *True* (default) run the full pretrain phase.  Set *False* to skip
        marginal training and load weights saved under ``{output_dir}/models/``.
        Automatically set to *False* when *pretrain_dir* is provided.
    save_models:
        Whether to write model checkpoints to disk after training.
    pretrain_dir:
        Path to a directory containing pre-trained ``.pth`` weight files
        (``initial_forward_model.pth``, ``initial_mu_model.pth``, etc.).
        When supplied the pretrain phase is skipped entirely — the weights are
        copied into the experiment's own ``models/`` directory and loaded from
        there, so iteration checkpoints still land in the right place.
    """
    setup_logging(spec.output_dir)
    logger.info("\n%s", "=" * 70)
    logger.info("  Experiment : %s", spec.name)
    logger.info("  Mode       : %s", spec.mode)
    logger.info("  Method     : %s", type(spec.method_cfg).__name__)
    logger.info("%s", "=" * 70)

    os.makedirs(spec.output_dir, exist_ok=True)
    model_dir = os.path.join(spec.output_dir, "models")
    device    = spec.train_cfg.device

    # Resolve W&B fields: CLI args > spec.train_cfg fields
    _wb_project = wandb_project or spec.train_cfg.wandb_project
    _wb_entity  = wandb_entity  or getattr(spec.train_cfg, "wandb_entity", None)
    _wb_name    = wandb_run_name or spec.train_cfg.wandb_run_name or spec.name
    _wb_group   = wandb_group

    # Initialise W&B (no-op if project is None or wandb not installed)
    from dualwcfm import wandb_logger as _wl
    import dataclasses
    def _safe_asdict(obj):
        """dataclasses.asdict but stringify non-serialisable values."""
        d = {}
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            try:
                import json; json.dumps(v)
                d[f.name] = v
            except (TypeError, ValueError):
                d[f.name] = str(v)
        return d
    _wb_config = {
        "experiment":    spec.name,
        "mode":          spec.mode,
        "method":        type(spec.method_cfg).__name__,
        **_safe_asdict(spec.train_cfg),
        **_safe_asdict(spec.method_cfg),
        **(wandb_extra_config or {}),
    }
    # Always tag the run with the experiment name so runs are filterable in
    # the W&B UI even when the run display-name encodes hyperparameters.
    _wb_tags = list(dict.fromkeys([spec.name] + (wandb_tags or [])))
    wandb_run = _wl.init(
        project = _wb_project,
        name    = _wb_name,
        entity  = _wb_entity,
        group   = _wb_group,
        config  = _wb_config,
        tags    = _wb_tags,
    )

    # --- Pre-trained weights ----------------------------------------------
    if pretrain_dir is not None:
        import shutil
        if not os.path.isdir(pretrain_dir):
            raise FileNotFoundError(f"pretrain_dir does not exist: {pretrain_dir}")
        os.makedirs(model_dir, exist_ok=True)
        copied = []
        for fname in os.listdir(pretrain_dir):
            if fname.endswith(".pth"):
                shutil.copy2(os.path.join(pretrain_dir, fname), model_dir)
                copied.append(fname)
        if not copied:
            raise FileNotFoundError(f"No .pth files found in pretrain_dir: {pretrain_dir}")
        logger.info("  Loaded pretrained weights from %s (%s)", pretrain_dir, ", ".join(sorted(copied)))
        retrain = False

    # --- Method -----------------------------------------------------------
    method_class = METHOD_REGISTRY.get(type(spec.method_cfg), MixtureMethod)
    method       = method_class(spec.method_cfg)

    # --- Strategy ---------------------------------------------------------
    if spec.mode == "image" and spec.model_cfg.is_image_model:
        from dualwcfm.experiments.exp_image import BidirectionalUnetStrategy, UNetWrapper
        logger.info("  Using BidirectionalUnetStrategy (UNet) for image experiment.")
        strategy        = BidirectionalUnetStrategy(UNetWrapper, spec.model_cfg)
        strategy.config = spec.model_cfg
        strategy.to_device(device)
    else:
        strategy = _make_mlp_strategy(
            dim          = spec.model_cfg.flat_dimension,
            hidden       = spec.model_cfg.hidden_size,
            bidirectional= spec.bidirectional,
            device       = device,
        )
        strategy.config = spec.model_cfg

    # --- Flow matching ----------------------------------------------------
    flow_matching = ConditionalFlowMatching()

    # --- Visualizer -------------------------------------------------------
    if spec.mode == "2d":
        visualizer = LoggingVisualizer(spec.vis_cfg, spec.transport_map, spec.output_dir,
                                       wandb_run=wandb_run,
                                       inverse_transport_map=spec.inverse_transport_map)
    elif spec.mode == "3d":
        visualizer = LoggingVisualizer3D(spec.vis_cfg, spec.transport_map, spec.output_dir,
                                         wandb_run=wandb_run,
                                         inverse_transport_map=spec.inverse_transport_map)
    else:
        if spec.train_cfg.image_shape is None:
            raise ValueError("image_shape must be set in train_cfg for image experiments.")
        visualizer = LoggingVisualizerImage(
            spec.vis_cfg, spec.train_cfg.image_shape, spec.output_dir,
            wandb_run=wandb_run,
            transport_map=spec.transport_map,
            inverse_transport_map=spec.inverse_transport_map,
        )

    # --- Trainer ----------------------------------------------------------
    trainer = Trainer(
        initial_coupling = spec.initial_coupling,
        new_source       = spec.new_source,
        new_target       = spec.new_target,
        flow_matching    = flow_matching,
        strategy         = strategy,
        visualizer       = visualizer,
        config           = spec.train_cfg,
        method           = method,
        device           = device,
    )

    trainer.train(
        retrain          = retrain,
        save             = model_dir,
        save_checkpoints = model_dir if save_models else None,
    )

    # --- Post-training ----------------------------------------------------
    visualizer.make_timelapse_all(fps=3)
    save_metrics_plot(spec.output_dir)
    wandb_run.finish()
    logger.info("  Finished -> %s", spec.output_dir)


# ---------------------------------------------------------------------------
# Metrics plotting
# ---------------------------------------------------------------------------

def save_metrics_plot(output_dir: str) -> None:
    """Read ``metrics.csv`` and write a comprehensive summary dashboard to *output_dir*."""
    csv_path = os.path.join(output_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    iters = [int(r["iteration"]) for r in rows]

    def _get(key):
        vals = []
        for r in rows:
            v = r.get(key, "")
            vals.append(float(v) if v not in ("", "N/A") else None)
        return vals

    def _plot(ax, keys_colors_labels, title, ylabel="", log=False):
        for key, color, label in keys_colors_labels:
            vals = _get(key)
            xs   = [i for i, v in zip(iters, vals) if v is not None]
            ys   = [v for v in vals if v is not None]
            if xs:
                ax.plot(xs, ys, color=color, label=label, linewidth=1.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=7)
        if log:
            try:
                ax.set_yscale("log")
            except Exception:
                pass
        ax.spines[["top", "right"]].set_visible(False)

    # 3×2 comprehensive dashboard
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    fig.patch.set_facecolor("#f8f9fa")
    for ax in axes.flat:
        ax.set_facecolor("#ffffff")

    # Row 0: SW marginal distances
    _plot(axes[0, 0],
          [("W1_mu", "#4878d0", "SW₁ source"), ("W2_mu", "#1a3a6e", "SW₂ source")],
          "Source marginal (SW distance)", "SW")
    _plot(axes[0, 1],
          [("W1_nu", "#ee854a", "SW₁ target"), ("W2_nu", "#c0392b", "SW₂ target")],
          "Target marginal (SW distance)", "SW")

    # Row 1: distributional quality + map errors
    _plot(axes[1, 0],
          [("mmd_nu",         "#6acc65", "MMD"),
           ("energy_dist_nu", "#956cb4", "Energy dist"),
           ("sinkhorn_div",   "#d65f5f", "Sinkhorn div")],
          "Target distribution quality", log=True)
    _plot(axes[1, 1],
          [("map_err_fwd", "#0173b2", "Map err fwd  √E‖T(x)−ŷ‖²"),
           ("map_err_bwd", "#de8f05", "Map err bwd  √E‖T⁻¹(y)−x̂‖²")],
          "Transport map error (ground truth)", "RMSE")

    # Row 2: importance weights
    w_mean = _get("weight_mean")
    w_std  = _get("weight_std")
    w_ess  = _get("weight_ess")
    xs_w   = [i for i, v in zip(iters, w_mean) if v is not None]
    ys_m   = [v for v in w_mean if v is not None]
    ys_s   = [v for v in w_std  if v is not None]

    axes[2, 0].plot(xs_w, ys_m, color="#023eff", linewidth=1.8, label="mean(w)")
    if ys_s:
        lo = [m - s for m, s in zip(ys_m, ys_s)]
        hi = [m + s for m, s in zip(ys_m, ys_s)]
        axes[2, 0].fill_between(xs_w, lo, hi, alpha=0.2, color="#023eff")
    axes[2, 0].axhline(1.0, color="#ff7c00", linestyle="--", linewidth=1, label="ideal (1.0)")
    axes[2, 0].set_title("Importance weight mean ± std", fontsize=10, fontweight="bold")
    axes[2, 0].set_xlabel("Iteration", fontsize=8)
    axes[2, 0].legend(fontsize=7)
    axes[2, 0].grid(True, alpha=0.25, linestyle="--")
    axes[2, 0].tick_params(labelsize=7)
    axes[2, 0].spines[["top", "right"]].set_visible(False)

    axes[2, 1].plot(
        [i for i, v in zip(iters, w_ess) if v is not None],
        [v for v in w_ess if v is not None],
        color="#3cb44b", linewidth=1.8, label="ESS (normalised)"
    )
    axes[2, 1].axhline(1.0, color="#ff7c00", linestyle="--", linewidth=1, label="ideal (1.0)")
    axes[2, 1].set_ylim(0, 1.05)
    axes[2, 1].set_title("Effective Sample Size", fontsize=10, fontweight="bold")
    axes[2, 1].set_xlabel("Iteration", fontsize=8)
    axes[2, 1].legend(fontsize=7)
    axes[2, 1].grid(True, alpha=0.25, linestyle="--")
    axes[2, 1].tick_params(labelsize=7)
    axes[2, 1].spines[["top", "right"]].set_visible(False)

    exp_name = os.path.basename(output_dir.rstrip("/\\"))
    fig.suptitle(f"Training Dashboard — {exp_name}", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
