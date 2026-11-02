# Running Experiments

All experiments are launched via the unified CLI:

```bash
python -m dualwcfm.experiments.run [OPTIONS]
```

Output goes to `results/<name>/` by default. Override with `--output PATH`.

---

## Quick Start

```bash
# Run the 2D simple experiment with the mixture method
python -m dualwcfm.experiments.run --mode 2d --level simple

# Run the 3D medium experiment with the score method
python -m dualwcfm.experiments.run --mode 3d --level medium --method score

# Run the image simple experiment
python -m dualwcfm.experiments.run --mode image --level simple
```

---

## Argument Reference

### Experiment Selection

| Argument | Values | Description |
|---|---|---|
| `--mode` | `2d`, `3d`, `image` | Experiment dimensionality. Required unless `--all` is given. |
| `--level` | `simple`, `medium`, `complex` | Difficulty level. Omit to run all levels for the chosen mode. |
| `--perturbed` | flag | Use the perturbed variant (shifted/rotated new target ν'). |
| `--all` | flag | Run every experiment across all modes sequentially. |

**2D experiments:**
- `simple` — anticorrelation map x ↦ −x, Gaussian blobs at (10, 10)
- `medium` — 60° rotation, 2-blob GMM → 4-blob square GMM
- `complex` — 45° rotation, 5-blob cross → 3× larger cross

**3D experiments:**
- `simple` — 90° z-axis rotation of a 2-blob GMM
- `medium` — 120° rotation, triangular 3-blob arrangement
- `complex` — 60° rotation, hexagonal 6-blob arrangement

**Image experiments:**
- `simple` — MNIST digit 3 → digit 7 with horizontal flip
- `medium` — FashionMNIST shirt → sneaker with flip + 10° perturbation

### Method

| Argument | Values | Default | Description |
|---|---|---|---|
| `--method` | `standard`, `mixture`, `score` | `mixture` | Training method for single runs. |
| `--sigma` | float | from config | Override the noise level σ for interpolation and score matching. |
| `--particle-ratio` | float | 0.25 | Fraction of coupling pairs replaced by fresh samples each iteration (MixtureMethod only). |

**Method summary:**
- `standard` — baseline importance-weighted CFM
- `mixture` — adds particle mixing to keep the coupling diverse
- `score` — adds a score model for ∇ log π_t, uses full change-of-measure weight

### Training Overrides

| Argument | Default | Description |
|---|---|---|
| `--iterations N` | from config | Number of outer Euler iterations. |
| `--batch-size N` | from config | Mini-batch size for all training phases. |
| `--no-save-models` | (save by default) | Do not write `.pth` checkpoints to disk. |

### Loading Pre-trained Models

Pretraining (marginal bridges + initial flow) is the most expensive phase. Two options to skip it:

**Option 1 — Load from a previous run's model directory:**
```bash
python -m dualwcfm.experiments.run --mode 2d --level simple --load-pretrained
```
Looks for `.pth` files in `<output>/<name>/models/` (must exist from a prior run with `--save-models`).

**Option 2 — Provide an explicit path:**
```bash
python -m dualwcfm.experiments.run --mode 2d --level simple \
    --pretrain-dir /path/to/pretrained_weights/
```
Copies all `.pth` files from that directory into the experiment's own `models/` folder, then proceeds with the outer iteration loop. The source directory is left untouched. This option takes precedence over `--load-pretrained`.

### Output

| Argument | Default | Description |
|---|---|---|
| `--output PATH` | `results` | Root directory for all experiment output. |

---

## Ablation Studies

An ablation study runs multiple method variants on the same experiment and produces comparison plots and tables.

```bash
# Compare all three methods on the 2D simple experiment
python -m dualwcfm.experiments.run --ablation --mode 2d --level simple

# Compare only standard and mixture
python -m dualwcfm.experiments.run --ablation --mode 2d --level simple \
    --methods standard mixture

# Seed sweep over 3 seeds
python -m dualwcfm.experiments.run --ablation --mode 2d --level simple \
    --methods mixture --seeds 0 1 2

# Custom output name for the suite
python -m dualwcfm.experiments.run --ablation --mode 2d --level simple \
    --ablation-name my_comparison
```

### Ablation Arguments

| Argument | Default | Description |
|---|---|---|
| `--ablation` | flag | Enable ablation mode. |
| `--methods` | all three | Space-separated list of methods to compare. |
| `--seeds` | `0` | Space-separated list of random seeds. |
| `--ablation-name` | auto | Name for the output suite directory. |

### Run Everything

```bash
# Run every experiment (all modes × levels × perturbed) sequentially
python -m dualwcfm.experiments.run --all

# Run an ablation across every experiment
python -m dualwcfm.experiments.run --all --ablation --methods standard mixture score
```

---

## Output Structure

### Single Experiment

```
results/<name>/
├── training.log              # Full training log
├── metrics.csv               # W1/W2 distances and weight stats per iteration
├── metrics.png               # Convergence curves (source + target SW distances)
├── weights_evolution.png     # Importance weight mean ± std over iterations
├── models/
│   ├── initial_forward_model.pth
│   ├── initial_mu_model.pth
│   ├── initial_nu_model.pth
│   └── (iteration checkpoints if save_model_checkpoints=True)
└── plots/
    ├── iter_000/             # Per-iteration trajectory and coupling plots
    ├── iter_001/
    └── ...
    ├── timelapse_*.gif       # Animated evolution
```

### Ablation Suite

```
results/<suite_name>/
├── <variant_1>/              # Full single-experiment output for each variant
├── <variant_2>/
├── ...
├── figures/
│   ├── convergence.png       # W2 curves for all variants on one axes
│   ├── weight_evolution.png  # Importance weight evolution per variant
│   ├── final_distributions.png  # Final coupling scatter per variant
│   ├── checkpoint_evolution.png # Coupling snapshots across iterations (2D only)
│   └── seed_aggregate_W2_nu.png # Mean ± std across seeds (multi-seed runs)
└── tables/
    ├── ablation_results.tex  # LaTeX booktabs table (bold = best per metric)
    └── ablation_results.csv  # Same data as CSV
```

---

## Reading the Outputs

**`metrics.csv`** — one row per outer iteration with columns:
- `W1_mu`, `W2_mu` — Sliced-Wasserstein-1/2 distance between generated source and μ'
- `W1_nu`, `W2_nu` — same for target ν'
- `weight_mean`, `weight_std` — statistics of the importance weights w(x,y) = exp(f(x)+g(y)); ideal mean ≈ 1.0

**Convergence plots** — lower is better for all SW metrics. A well-converging run shows monotonically decreasing curves that flatten near the end.

**Weight evolution** — the shaded band shows mean ± std. A band that stays close to the `y=1` reference line indicates the dual potentials are well-calibrated. A drifting mean or growing variance signals dual training instability.

**`checkpoint_evolution.png`** (ablation, 2D only) — shows the coupling dataset as a scatter plot at each Euler iteration, one column per variant. Useful for checking that the transport is being correctly transferred.

---

## Examples

```bash
# Fast prototype run (few iterations, smaller batch)
python -m dualwcfm.experiments.run --mode 2d --level simple \
    --iterations 10 --batch-size 512

# Full 3D complex experiment with checkpoints
python -m dualwcfm.experiments.run --mode 3d --level complex \
    --method mixture --iterations 150

# Image experiment, reusing pretrained marginals from a previous run
python -m dualwcfm.experiments.run --mode image --level simple \
    --pretrain-dir results/image_simple_prev/models/ \
    --iterations 30

# Ablation with seeds, saving to a custom directory
python -m dualwcfm.experiments.run \
    --ablation --mode 2d --level medium \
    --methods standard mixture score \
    --seeds 0 1 2 3 \
    --ablation-name medium_full_ablation \
    --output paper_results
```

---

## Entropic Gaussian transfer (tilting diagnostics)

```bash
PYTHONPATH=src python -m dualwcfm.experiments.exp_gaussian_eot --variants all --scale 2.0
```

The reference coupling is a non-degenerate joint Gaussian, i.e. a genuine entropic OT plan
(bilinear latent cost). For new isotropic Gaussian marginals the transferred plan is known in
closed form, and every tilt preserves the cross block `q` of the precision matrix. Scaling the
new marginals changes the target correlation (`rho*`), whereas pairs transported along the OT
bridges keep the reference correlation, so this experiment separates tilting from particle
transport. Per-iteration statistics are written to `results/gaussian_eot/<variant>.csv`:
`q_ratio` (latent-cost content of the current coupling, 1 = exact), `rho_hat` (empirical
correlation), `rho_own` (EOT correlation for the coupling's own marginals) and `rho_path`
(exact plan on the marginal path).

### New method options (all opt-in; defaults reproduce previous behaviour)

| Option | Default | Effect |
|---|---|---|
| `source_std` | `0.0` | Std of the noise source of the coupling flow (original units). With `0.0` the conditional flow starts at the condition itself (Dirac source): generated couplings are graphs of maps and weighted re-training cannot tilt them. Use `> 0` for entropic couplings. |
| `lip_reg_relative` | `False` | Multiply the dual gradient penalty by the batch variance of the coupling samples. An absolute penalty `lip_reg * E|grad a|^2` dominates the coupling term `E[(a+b)^2] ~ Var(x)|grad a|^2` by `lip_reg / Var(x)` (about `1e4` for the 2D configs with `lip_reg=1`, scale 50) and shrinks the importance weights accordingly. |
| `disable_tilt` | `False` | Re-train the coupling flow without importance weights (ablation of the tilt). |
| `use_bridge_marginals` | `False` | (existing) Sample marginals by integrating the OT bridges, consistently with the bridge velocities used in the dual energy. With `False`, marginal samples come from independent linear interpolation while the velocities come from OT-paired bridges. |
