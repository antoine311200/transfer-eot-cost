# TACO: Transport Adaptation via Cost-free Optimization

**NeurIPS 2026 Submission — Code Repository**

TACO (Transport Adaptation via Cost-free Optimization) is a method for transferring entropic optimal transport (EOT) / Schrödinger bridge solutions across marginal distributions without access to the ground-truth transport cost. Given a reference coupling π between source µ and target ν, TACO adapts the learned transport to a new pair of marginals (µ', ν') by alternating between dual-potential tilting and conditional flow-matching updates.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Repository Structure](#repository-structure)
5. [Data](#data)
6. [Running Experiments](#running-experiments)
   - [2D Synthetic Experiments](#2d-synthetic-experiments)
   - [Nonlinear 2D Experiments](#nonlinear-2d-experiments)
   - [Scalability Experiments](#scalability-experiments)
   - [Single-Cell Drug Response Transfer](#single-cell-drug-response-transfer)
7. [Running Baselines](#running-baselines)
8. [Generating Figures](#generating-figures)
9. [Configuration Reference](#configuration-reference)
10. [Metrics](#metrics)
11. [Project Structure (detailed)](#project-structure-detailed)

---

## Overview

TACO solves the following problem: given a *reference coupling* π on (µ, ν) learned by any EOT solver, transfer it to new marginals (µ', ν') **without** access to the underlying cost function.

The algorithm alternates between:
1. **Dual tilting** — refit Kantorovich potentials (f, g) to re-weight the coupling toward the new marginals via an importance-weighted dual objective.
2. **CFM update** — retrain the conditional flow-matching model β_s on the updated coupling, extending it to the new distribution support.

Three loss variants are available (all produce the same fixed point):
- **Weighted** (`--loss weighted`): direct importance-weighted regression on the EOT cost.
- **Implicit** (`--loss implicit`): covariance-based loss without explicit cost evaluation.
- **Explicit** (`--loss explicit`): explicit reconstruction loss with a learned covariance term.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA (recommended; CPU works but is slow for single-cell experiments)

Core dependencies:
```
torch torchvision
numpy scipy matplotlib
torchdiffeq
geomloss          # Sinkhorn divergence
pot               # Python Optimal Transport
scikit-learn
anndata scanpy    # single-cell data
pertpy            # for auto-downloading Sci-Plex3 (optional)
umap-learn        # optional, for UMAP projections in figures
wandb             # optional, for experiment tracking
```

Install all at once (example with conda):
```bash
conda create -n taco python=3.11
conda activate taco
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib scikit-learn geomloss pot torchdiffeq
pip install anndata scanpy pertpy umap-learn wandb
```

---

## Installation

```bash
git clone <repo>
cd dualwcfm
pip install -e .
```

The package installs as `dualwcfm`. All experiments are launched via `python -m dualwcfm.experiments.run`.

---

## Repository Structure

```
dualwcfm/
├── src/dualwcfm/               # Core library
│   ├── config.py               # TrainingConfig, FeedForwardConfig, etc.
│   ├── trainer.py              # Main Trainer class (outer TACO loop)
│   ├── flow_matching.py        # CFM / OT-CFM training utilities
│   ├── strategy.py             # BidirectionalStrategy (MLP / SIREN)
│   ├── metrics.py              # SW2, MMD, Sinkhorn, energy distance, etc.
│   ├── visualization.py        # Logging & plotting helpers
│   ├── logging_visualizer.py   # Per-iteration checkpoint visualizer
│   ├── mlp.py                  # MLP and torch_wrapper
│   ├── methods/
│   │   ├── mixture.py          # MixtureConfig + MixtureMethod (default)
│   │   ├── standard.py         # StandardMethod
│   │   └── score.py            # ScoreMethod
│   ├── losses/
│   │   ├── weight_loss.py      # Weighted TACO loss
│   │   ├── implicit_loss.py    # Implicit covariance loss
│   │   └── explicit_loss.py    # Explicit covariance loss
│   ├── distributions/
│   │   ├── distribution.py     # Base distribution interface
│   │   ├── coupling.py         # FixedSizeCouplingDistribution, DiskCouplingDistribution
│   │   └── scaled.py           # ScaledDistribution wrapper
│   ├── models/
│   │   ├── siren.py            # SirenMLP (sinusoidal networks)
│   │   └── unet/               # Conditional UNet for image experiments
│   ├── data/
│   │   └── sciplexdata.py      # Sci-Plex3 loader (single-cell)
│   ├── baselines/
│   │   ├── run.py              # Baseline experiment runner
│   │   ├── otcfm.py            # OT-CFM and CFM baselines
│   │   ├── lightsb.py          # LightSB baseline
│   │   ├── dsb.py              # DSB/DSBM baseline
│   │   ├── genot.py            # GENOT baseline
│   │   └── pretrained.py       # Zero-shot pretrained baseline
│   └── experiments/
│       ├── run.py              # Unified CLI entry point
│       ├── base.py             # ExperimentSpec dataclass
│       ├── exp_2d.py           # 2D synthetic experiments
│       ├── exp_3d.py           # 3D synthetic experiments
│       ├── exp_singlecell.py   # Single-cell drug response experiments
│       ├── exp_scalability.py  # High-dimensional scalability experiments
│       ├── exp_gushchin.py     # Gushchin benchmark experiments
│       └── exp_image.py        # Image transport experiments
│
├── make_2d_transport_figure.py          # Figure: 3-panel transport visualisation
├── make_2d_coupling_evolution_figure.py # Figure: coupling evolution over iterations
├── make_convergence_figure.py           # Figure: map-error convergence vs N (wandb)
├── make_far_convergence_figure.py       # Figure: O(1/N) convergence on 2D-Far
├── make_scalability_figure.py           # Figure: scalability (wandb, 3-panel)
├── make_scalability_figure_v2.py        # Figure: scalability dual-axis (from cache)
├── make_singlecell_figure.py            # Figure: single-cell 3-panel transport
├── make_singlecell_paper_figure.py      # Figure: full single-cell paper figure suite
├── make_baseline_tables.py             # Generate LaTeX baseline comparison tables
├── paper_theme.py                      # Shared matplotlib style + paper palette
│
├── plots/                              # Generated LaTeX tables
│   └── baseline_tables/
│       └── per_experiment/             # One .tex file per experiment
├── figures/                            # Generated PDF/PNG figures (gitignored)
├── results/                            # Experiment outputs (gitignored)
├── results_cell/                       # Single-cell outputs (gitignored)
│
├── data/                               # Data directory (gitignored)
│   └── SrivatsanTrapnell2020_sciplex3.h5ad
│
├── EXPERIMENTS.md                      # Detailed experiment argument reference
├── BENCHMARK.md                        # Baseline reproduction notes
└── pyproject.toml
```

---

## Data

### 2D / 3D / Scalability

All synthetic distributions are generated on the fly — no data files needed.

### Single-Cell (Sci-Plex3)

Download the Sci-Plex3 h5ad file (~2.5 GB) from Zenodo:

```
https://zenodo.org/records/13350497
File: SrivatsanTrapnell2020_sciplex3.h5ad
```

Place it anywhere and pass `--data-path /path/to/SrivatsanTrapnell2020_sciplex3.h5ad` to all single-cell commands.

Alternatively, if `pertpy` is installed with the dataset cached locally, the path can be omitted.

**Preprocessing** (done automatically inside `SciplexLoader`):
- Filter to Vehicle + Givinostat conditions only
- Select 1000 highly variable genes
- Log-normalize (already done in the h5ad)
- PCA to 50 components
- Build nearest-neighbour coupling within each condition (CellOT protocol)

---

## Running Experiments

All experiments are run via the unified CLI:

```bash
python -m dualwcfm.experiments.run [OPTIONS]
```

Results are saved to `results/<experiment_name>/` by default.
Each run saves: `models/forward_model.pth`, `models/initial_forward_model.pth`, dual potentials (`f_dual.pth`, `g_dual.pth`, `h_dual.pth`), marginal models, and `metrics.csv`.

### 2D Synthetic Experiments

Three difficulty levels × two variants (standard / perturbed):

```bash
# Standard variants
python -m dualwcfm.experiments.run --mode 2d --level simple
python -m dualwcfm.experiments.run --mode 2d --level medium
python -m dualwcfm.experiments.run --mode 2d --level complex
python -m dualwcfm.experiments.run --mode 2d --level moon

# Perturbed variants (rotated/shifted new target)
python -m dualwcfm.experiments.run --mode 2d --level simple  --perturbed
python -m dualwcfm.experiments.run --mode 2d --level medium  --perturbed
python -m dualwcfm.experiments.run --mode 2d --level complex --perturbed
python -m dualwcfm.experiments.run --mode 2d --level moon    --perturbed

# Run all 2D experiments at once
python -m dualwcfm.experiments.run --mode 2d
```

**Loss function variants** (run each level with all three):
```bash
python -m dualwcfm.experiments.run --mode 2d --level simple --loss weighted
python -m dualwcfm.experiments.run --mode 2d --level simple --loss implicit
python -m dualwcfm.experiments.run --mode 2d --level simple --loss explicit
```

**Key flags:**
```
--iterations N       Override number of outer TACO iterations (default: 100)
--sigma FLOAT        Override noise level σ (default: from config)
--particle-ratio R   Particle mix ratio α (default: 0.25)
--batch-size B       Override batch size
--output PATH        Override output directory
--load-pretrained    Skip pretraining, load existing initial_forward_model.pth
--no-perturbed       Run only non-perturbed variants when --mode 2d is given
```

### Nonlinear 2D Experiments

Radial warp, polar twist, and spiral experiments:

```bash
python -m dualwcfm.experiments.run --mode 2d --level radial
python -m dualwcfm.experiments.run --mode 2d --level polar
python -m dualwcfm.experiments.run --mode 2d --level spiral
```

### Scalability Experiments

Anti-correlation benchmark T(x) = −x across dimensions d ∈ {8, 16, 32, 64, 128}:

```bash
python -m dualwcfm.experiments.run --mode scalability --dims 8 16 32 64 128
```

### Single-Cell Drug Response Transfer

Three experimental settings (A549 → K562/MCF7, Givinostat):

```bash
DATA=data/SrivatsanTrapnell2020_sciplex3.h5ad

# Standard: A549→K562, Givinostat 1µM
python -m dualwcfm.experiments.run \
    --mode singlecell --no-perturbed \
    --data-path $DATA \
    --output results_cell/singlecell

# Perturbed: A549→MCF7, Givinostat 0.1µM (distributional shift)
python -m dualwcfm.experiments.run \
    --mode singlecell --perturbed \
    --data-path $DATA \
    --output results_cell/singlecell_perturbed

# New: A549→MCF7, Givinostat 1µM (standard dose on MCF7)
python -m dualwcfm.experiments.run \
    --mode singlecell-extra \
    --data-path $DATA \
    --output results_cell/singlecell_mcf7_std
```

---

## Running Baselines

Baselines (CFM, OT-CFM, LightSB, DSB, GENOT) are run via a separate CLI that requires access to the ground-truth cost (for coupled variants) or runs independently (for independent variants):

```bash
# Run all baselines on a given experiment
python -m dualwcfm.baselines.run --mode 2d --level simple

# Run only coupled OT-CFM baseline
python -m dualwcfm.baselines.run --mode 2d --level complex --baseline ot-cfm --coupled

# Run all baselines on all 2D experiments and generate LaTeX tables
python -m dualwcfm.baselines.run --mode 2d --all --table
```

Baseline outputs are saved to `results/baselines/` and tables to `plots/baseline_tables/`.

See `BENCHMARK.md` for per-baseline reproduction notes and hyperparameter choices.

---

## Generating Figures

All figure scripts are in the repository root and use the shared `paper_theme.py` palette (matching the paper's TikZ color scheme).

### Prerequisites

```bash
# Build trained models first (or use --load-pretrained)
# Then run figure generation commands below
```

### Transport figures (3-panel: reference π | pretrained | TACO)

```bash
# 2D simple
python make_2d_transport_figure.py \
    --mode 2d --level simple \
    --exp-dir results/2d_simple \
    --output figures/2d_simple_transport

# 2D medium
python make_2d_transport_figure.py \
    --mode 2d --level medium \
    --exp-dir results/2d_medium \
    --output figures/2d_medium_transport

# 2D complex
python make_2d_transport_figure.py \
    --mode 2d --level complex \
    --exp-dir results/2d_complex \
    --n-particles 400 \
    --output figures/2d_complex_transport

# 2D complex (perturbed)
python make_2d_transport_figure.py \
    --mode 2d --level complex --perturbed \
    --exp-dir results/2d_complex_perturbed \
    --output figures/2d_complex_perturbed_transport

# 2D moon
python make_2d_transport_figure.py \
    --mode 2d --level moon \
    --exp-dir results/2d_moon \
    --output figures/2d_moon_transport
```

### Coupling evolution figure

```bash
# 2D complex (perturbed) — evolution across 10 checkpoints
python make_2d_coupling_evolution_figure.py \
    --mode 2d --level complex --perturbed \
    --exp-dir results/2d_complex_perturbed \
    --step 10 --n-particles 400 \
    --output figures/2d_complex_perturbed_coupling_evolution

# 2D moon
python make_2d_coupling_evolution_figure.py \
    --mode 2d --level moon \
    --exp-dir results/2d_moon \
    --step 10 --n-particles 500 \
    --output figures/2d_moon_coupling_evolution
```

### Convergence figures

```bash
# O(1/N) convergence on 2D-Far (reads from results_final_2d/2d_far/)
python make_far_convergence_figure.py

# Map-error vs N (reads from W&B; requires WANDB_API_KEY)
python make_convergence_figure.py
```

### Scalability figure

```bash
# From W&B (requires wandb login + correct run IDs in the script)
python make_scalability_figure.py

# From cached JSON (images/scalability/wandb_history.json)
python make_scalability_figure_v2.py
```

### Single-cell figures

```bash
DATA=data/SrivatsanTrapnell2020_sciplex3.h5ad

# Full single-cell figure suite (transport + convergence + evolution + metrics bar)
python make_singlecell_paper_figure.py \
    --data-path $DATA \
    --std-dir  results_cell/singlecell \
    --pert-dir results_cell/singlecell_perturbed \
    --figure all \
    --output-dir figures/singlecell_mlp

# Standard variant only (3-panel transport)
python make_singlecell_paper_figure.py \
    --data-path $DATA \
    --std-dir results_cell/singlecell \
    --figure main_transport \
    --output-dir figures/singlecell_mlp

# Perturbed variant
python make_singlecell_paper_figure.py \
    --data-path $DATA \
    --std-dir  results_cell/singlecell \
    --pert-dir results_cell/singlecell_perturbed \
    --figure perturbed \
    --output-dir figures/singlecell_mlp

# 2×3 dual figure (standard + perturbed in one grid)
python make_singlecell_paper_figure.py \
    --data-path $DATA \
    --std-dir  results_cell/singlecell \
    --pert-dir results_cell/singlecell_perturbed \
    --figure dual_transport \
    --output-dir figures/singlecell_mlp

# Convergence curves (standard vs perturbed, W1 over TACO iterations)
python make_singlecell_paper_figure.py \
    --data-path $DATA \
    --std-dir  results_cell/singlecell \
    --pert-dir results_cell/singlecell_perturbed \
    --figure convergence \
    --output-dir figures/singlecell_mlp

# Simpler 3-panel figure (no metrics overlay, for supplement)
python make_singlecell_figure.py \
    --exp-dir results_cell/singlecell \
    --data-path $DATA \
    --output figures/singlecell_mlp/transport_simple
```

### Baseline tables

```bash
# Regenerate all per-experiment LaTeX tables
python make_baseline_tables.py
```

---

## Configuration Reference

### `TrainingConfig` (src/dualwcfm/config.py)

| Field | Default | Description |
|---|---|---|
| `iterations` | 100 | Number of outer TACO iterations |
| `dual_epochs` | 100 | Dual potential training steps per iteration |
| `cfm_epochs` | 500 | CFM model training steps per iteration |
| `marginal_epochs` | 3000 | Bridge training steps per iteration |
| `pretrain_epochs` | 8000 | Initial CFM pretraining steps |
| `advance_steps` | 100 | Bridge advance steps |
| `batch_size` | 2048 | Training batch size |
| `dataset_size` | 100000 | Coupling dataset size |
| `scale_factor` | 50 | Input normalization scale (0 = auto-detect) |
| `sigma` | 0.05 | Interpolation noise σ |
| `particle_ratio` | 0.20 | Fraction of fresh-particle pairs per iteration |
| `save_model_checkpoints` | False | Save per-iteration model snapshots |
| `save_coupling_checkpoints` | False | Save per-iteration coupling datasets |

### `FeedForwardConfig` (src/dualwcfm/config.py)

| Field | Default | Description |
|---|---|---|
| `dimension` | 2 | Input data dimensionality |
| `hidden_size` | 64 | MLP hidden layer width |
| `num_layers` | 2 | Number of hidden layers |
| `flow_learning_rate` | 2e-4 | CFM model learning rate |
| `dual_learning_rate` | 1e-4 | Dual potential learning rate |
| `mu_learning_rate` | 2e-4 | Source bridge learning rate |
| `nu_learning_rate` | 2e-4 | Target bridge learning rate |

### `MixtureConfig` (src/dualwcfm/methods/mixture.py)

| Field | Default | Description |
|---|---|---|
| `sigma` | 0.05 | EOT noise level for coupling construction |
| `sigma2` | 0.05 | Secondary noise level (bridge regularization) |
| `loss` | `"weighted"` | Loss variant: `weighted`, `implicit`, `explicit` |
| `particle_ratio` | 0.20 | Fresh-particle fraction α |
| `l2_reg` | 5e-3 | L2 regularization on dual potentials |
| `lip_reg` | 1.0 | Lipschitz regularization strength |
| `dual_warmup_epochs` | 20 | Warm-up steps before dual training |

---

## Metrics

All metrics are lower-is-better. Evaluated in `src/dualwcfm/metrics.py`.

| Metric | Symbol | Description |
|---|---|---|
| Sliced-Wasserstein-2 (target) | SW₂(ν̂) | SW₂ between predicted and true target marginals |
| Sliced-Wasserstein-1 (target) | SW₁(ν̂) | SW₁ variant |
| Sliced-Wasserstein-2 (source) | SW₂(µ̂) | Verifies source marginal is preserved |
| Sliced-Wasserstein-1 (source) | SW₁(µ̂) | SW₁ variant |
| Map error (forward) | MapErr | RMSE ‖T(x) − ŷ‖ (when ground-truth T is known) |
| Sinkhorn divergence | Sε | Debiased entropic OT cost between predicted and true targets |
| Maximum Mean Discrepancy | MMD | Kernel-based distributional divergence (RBF kernel) |
| Energy distance | E-dist | Euclidean-distance-based distributional metric |
| Bures-Wasserstein distance | BW₂ | Distance to ground-truth Gaussian coupling (scalability only) |
| Effective Sample Size | ESS | Importance weight entropy; measures coupling collapse |

---

## Project Structure (detailed)

### Core Algorithm (`src/dualwcfm/trainer.py`)

The `Trainer` class implements the TACO outer loop:

```
for k = 1..N:
    1. Sample coupling pairs (x, y) ∼ π_k (weighted by dual potentials)
    2. Dual step: update (f_k, g_k, h_k) by maximizing the weighted dual objective
    3. CFM step: retrain β_s on the updated coupling π_{k+1}
    4. Bridge step: update marginal bridge models (u_t, v_t)
    5. Evaluate metrics; optionally save checkpoint
```

### Methods (`src/dualwcfm/methods/`)

- `MixtureMethod` — default; mixes particle-propagated pairs with importance-weighted pairs to prevent circularity collapse.
- `StandardMethod` — pure dual-tilting without particle mixing.
- `ScoreMethod` — score-matching variant for stochastic dynamics.

### Loss Functions (`src/dualwcfm/losses/`)

- `WeightedLoss` — direct importance-weighted regression; requires evaluating exp(f+g-c).
- `ImplicitLoss` — covariance-based objective derived from the Bregman projection structure; avoids explicit cost evaluation.
- `ExplicitLoss` — explicit reconstruction with a learned covariance correction term.

### Distributions (`src/dualwcfm/distributions/`)

- `FixedSizeCouplingDistribution` — in-memory coupling; suited for 2D/small experiments.
- `DiskCouplingDistribution` — disk-backed coupling; automatically used for image datasets (> 1000 samples with `image_shape` set).

### Baselines (`src/dualwcfm/baselines/`)

All baselines run with access to the true quadratic cost (coupled variants) or independently (independent variants):

| Baseline | Reference |
|---|---|
| CFM (indep.) | Lipman et al., 2022 |
| OT-CFM (indep./coupl.) | Tong et al., 2023 |
| LightSB (indep./coupl.) | Korotin et al., 2023 |
| DSB (indep./coupl.) | De Bortoli et al., 2021 |
| GENOT (coupl.) | Klein et al., 2023 |

---

## Reproducibility Notes

- All random seeds are fixed at 42 by default (override with `--seed N`).
- Model checkpoints are saved as `.pth` files; architecture is inferred automatically at load time from the state dict shape.
- Metrics are logged to `metrics.csv` in the experiment output directory, one row per TACO iteration.
- W&B logging is optional; disable with `--no-wandb` or by not setting `WANDB_API_KEY`.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{taco2026,
  title     = {TACO: Transport Adaptation via Cost-free Optimization},
  author    = {[authors]},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```
