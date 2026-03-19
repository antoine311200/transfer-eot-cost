# DualWCFM Codebase Overview

## What the Project Does

DualWCFM learns a **transport dynamic** from a reference coupling π (a joint distribution over source/target pairs) and transfers that dynamic to new marginals (μ', ν'). The core idea is entropic optimal transport via iterative refinement: starting from π₀, the algorithm evolves the coupling forward in time through a sequence of Euler steps, each consisting of a dual potential update and a conditional flow matching step.

The algorithm can be applied to:
- 2D/3D toy distributions (Gaussian blobs, GMMs, rotated shapes)
- Image-to-image transport (MNIST digit transfer, FashionMNIST)

---

## Package Layout

```
src/dualwcfm/
├── config.py                  # All dataclass configs (TrainingConfig, FeedForwardConfig, VisualizeConfig)
├── trainer.py                 # Main training loop
├── flow_matching.py           # ConditionalFlowMatching: interpolation, vector fields, ODE integration
├── strategy.py                # Strategy / BidirectionalStrategy: model containers + weight computation
├── mlp.py                     # MLP and torch_wrapper (time-varying feedforward network)
├── logging_visualizer.py      # LoggingVisualizer{,3D,Image}: per-iteration plots, metrics, timelapse
├── visualization.py           # Low-level plot helpers
├── utils.py                   # is_image_tensor and other helpers
├── distributions/
│   ├── distribution.py        # Distribution ABC, GMM variants, Simple2DGaussianDistribution
│   ├── coupling.py            # CouplingDistribution, FixedSizeCouplingDistribution, DiskCouplingDistribution
│   └── scaled.py              # ScaledDistribution, ScaledCouplingDistribution (normalization wrappers)
├── losses/
│   ├── base.py                # Loss ABC
│   ├── weight_loss.py         # Importance-weighted CFM loss
│   ├── explicit_loss.py       # Explicit (primal) OT loss
│   └── implicit_loss.py       # Implicit (dual) OT loss
├── methods/
│   ├── base.py                # Method ABC + MethodConfig base dataclass
│   ├── standard.py            # StandardMethod / StandardConfig (baseline)
│   ├── mixture.py             # MixtureMethod / MixtureConfig (+ particle mixing)
│   └── score.py               # ScoreMethod / ScoreConfig (+ score matching)
├── models/unet/               # UNet implementation for image experiments
└── experiments/
    ├── base.py                # ExperimentSpec, run_experiment(), METHOD_REGISTRY
    ├── exp_2d.py              # 2D experiment catalogue
    ├── exp_3d.py              # 3D experiment catalogue
    ├── exp_image.py           # Image experiment catalogue + UNet wrappers
    ├── ablation.py            # AblationSuite, AblationVariant, AblationRunner
    └── run.py                 # CLI entry point
```

---

## Core Abstractions

### Trainer

`Trainer` owns the entire training state and delegates each phase to a `Method`. On construction it:
1. Computes a normalization scale from the four marginals and wraps everything in `ScaledDistribution` / `ScaledCouplingDistribution`.
2. Builds the fixed-size (or disk-backed) coupling dataset from the initial coupling.

The `train(retrain, save)` method orchestrates:
```
pretrain  →  [outer iteration loop]
               step_preprocess
               step_duals
               step_flow_matching
               step_postprocess
               log metrics / checkpoints
```

### Method

`Method` is the strategy pattern for the training algorithm. It defines five abstract hooks called by `Trainer`:

| Hook | Responsibility |
|---|---|
| `pretrain` | Train marginal bridges (μ-model, ν-model) and the initial flow |
| `step_preprocess` | Build iteration context (time t, direction, is_image) |
| `step_duals` | Update dual potentials f, g (Kantorovich duals) |
| `step_flow_matching` | Train the conditional flow model with importance weights |
| `step_postprocess` | Advance the coupling dataset one Euler step |

Three concrete methods are provided:

- **StandardMethod** — baseline, trains duals then CFM with the log-weight from the current strategy.
- **MixtureMethod** — extends Standard with *particle mixing*: at each iteration a fraction (`particle_ratio`) of coupling pairs are replaced by fresh samples from the new marginals, keeping the dataset diverse.
- **ScoreMethod** — extends Standard with a *score model* for ∇ log π_t. The score is learned via denoising score matching during pretrain, then jointly updated with the flow model each iteration using the full change-of-measure weight (includes divergence and drift correction terms).

### Strategy

`Strategy` holds the neural models and computes importance weights:
- `get_duals()` — returns f, g (Kantorovich potential models)
- `get_bridges()` — returns μ-model, ν-model (marginal bridge velocity fields)
- `get_flow()` — returns the current iteration's CFM model and its optimizer
- `log_weight()` — log(w) = f(x) + g(y) for a coupling pair (x, y)
- `to_device()` — moves all models to the target device

`BidirectionalStrategy` alternates direction every iteration (forward on even, backward on odd), effectively training both μ→ν and ν→μ directions.

To use a custom architecture (e.g., UNet), subclass `Strategy` or `BidirectionalStrategy` and override `_create_models()`. **Important:** the override must start with `self._device = None` to ensure `to_device()` actually moves the new models.

### ConditionalFlowMatching

Provides the interpolation and target computation for CFM:
- `interpolate(x0, x1, t, sigma)` — stochastic interpolant x_t = (1-t)x0 + t·x1 + σ·ε
- `vector_field(...)` — target velocity u_t = x1 - x0 (+ optional noise conditioning)
- `evolve(x, model, ...)` — Euler ODE integration from t=0 to t=1

### Distribution System

All distributions implement a `sample(n)` interface. Couplings additionally provide:
- `coupling_sample(n)` — sample (x, y) pairs
- `get_marginals()` — return (source_distribution, target_distribution)

**FixedSizeCouplingDistribution**: stores the full dataset in memory as two tensors. Supports `update_chunk()` for in-place Euler advances.

**DiskCouplingDistribution**: chunks the dataset to disk as `.pt` files with an LRU cache for loading. Auto-selected for image experiments when `dataset_size > 1000`. Configured via `TrainingConfig.coupling_save_dir`, `coupling_chunk_size`, `use_disk_coupling`.

The **Scaled** wrappers divide all samples by a normalization constant at construction time, ensuring every tensor lives in [-1, 1] during training regardless of marginal spread.

---

## Training Loop in Detail

### Pretrain Phase

Runs once unless `retrain=False`:
1. Train marginal flow models: μ-model learns ∇ log μ' and ν-model learns ∇ log ν' via CFM.
2. Train dual potentials on the initial coupling π₀.
3. Train the initial CFM model (flow model at t=0).
4. *(ScoreMethod only)* Train the score model s_θ ≈ ∇ log π₀ via denoising score matching.

### Outer Iteration Loop

For iteration i = 0 … T-1 (T = `iterations` in `TrainingConfig`):

1. **step_preprocess**: set t = i/T, determine direction, build context dict.
2. **step_duals**: optimize f, g on the *current* coupling π_t (not π₀).
3. **step_flow_matching**: train CFM model with importance-weighted loss.
   - Weight: w(x,y) = exp(f(x) + g(y))
   - Loss: E[w · ‖u_θ(s, x_s, x) - target‖²]
4. **step_postprocess**: Euler-advance every coupling pair:
   - x ← x + Δ · μ-model(t, x)
   - y ← y + Δ · ν-model(t, y)
5. Compute Sliced-Wasserstein distances (W1, W2 to new source/target), log to `metrics.csv`.

Each outer iteration is an independent optimization — models are re-initialized (or warm-started for duals), and the LR is reset to the configured value.

---

## Image Support

For image experiments, the standard MLP-based strategy is replaced by `BidirectionalUnetStrategy` (defined in `exp_image.py`):

- **UNetWrapper**: adapts the UNet to the `(t, x[, condition])` calling convention used by CFM. When a condition is provided, it concatenates `[x, condition]` along the channel dimension (doubling input channels).
- **DualUNetWrapper**: time-independent wrapper for dual potentials. Outputs a scalar `(N, 1, 1, 1)` tensor using `stop_middle=True` to extract a bottleneck feature.

`TrainingConfig.image_shape = (C, H, W)` enables all image-specific code paths (DiskCoupling auto-selection, visualizer choice, CFM `extend=True` for 2C channels).

---

## Config System

All hyperparameters live in dataclasses in `config.py`:

- **`TrainingConfig`**: dataset_size, epoch counts per phase, iterations, batch_size, LR values (dual_lr, mu/nu_lr, flow_lr), normalization, image_shape, coupling storage options.
- **`FeedForwardConfig`**: dimension, hidden_size, image_shape, is_image_model.
- **`VisualizeConfig`**: show_each, n_trajectories, trajectory_stride.

Method-specific hyperparameters live in `MethodConfig` subclasses:
- `StandardConfig`: sigma, sigma2, loss type, regularization terms, LR schedule flags.
- `MixtureConfig(StandardConfig)`: adds `particle_ratio`.
- `ScoreConfig(StandardConfig)`: adds score_epochs, score_sigma, score_lr.

---

## Visualization and Logging

`LoggingVisualizer` (and its 3D/Image variants) hooks into the training loop at:
- `at_flow_matching_start` / `at_flow_matching_end` — trajectory plots, quiver fields
- `at_postprocess_end` — coupling scatter after Euler step
- `make_timelapse_all` — assembles per-iteration PNGs into GIFs

All metrics (W1_mu, W2_mu, W1_nu, W2_nu, weight_mean, weight_std) are logged to `metrics.csv` and plotted by `save_metrics_plot()` in `base.py`.
