# DualWCFM — Complete Benchmark Commands

All commands are run from the `dualwcfm/` root directory.
Replace `antoine311200` / `"Data-Based EOT"` with your W&B credentials, or drop the W&B flags entirely to disable logging.

---

## 0. Setup

```bash
cd dualwcfm

# Install in editable mode (once)
pip install -e .

# W&B login (once)
wandb login
```

---

## 1. E1 — Gushchin Gaussian EOT Benchmark

Isotropic and GMM Gaussian pairs with analytical ground-truth BW₂.
Covers `eps ∈ {0.1, 0.5, 2.0}` for iso, `eps ∈ {0.5, 2.0}` for GMM, and a high-dimensional `R⁸` scaling experiment.

```bash
# All Gushchin experiments at once
python -m dualwcfm.experiments.run --mode gushchin --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

# Individual epsilon levels (iso 2D)
python -m dualwcfm.experiments.run --mode gushchin --level eps0p1 \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode gushchin --level eps0p5 \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode gushchin --level eps2p0 \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results
```

---

## 2. E2 — 2-D Toy Experiments

Anticorrelation (simple), 60° rotation (medium), 45° rotation (complex).
Each has a standard and perturbed variant.

```bash
# All 2D (standard + perturbed)
python -m dualwcfm.experiments.run \
    --mode 2d \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

# Individual levels
python -m dualwcfm.experiments.run --mode 2d --level simple \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode 2d --level medium \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode 2d --level complex \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

# Perturbed variants
python -m dualwcfm.experiments.run --mode 2d --level simple --perturbed \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode 2d --level medium --perturbed \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode 2d --level complex --perturbed \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results
```

---

## 3. E3 — 3-D Toy Experiments

Z-axis rotations (90°, 120°, 60°) generalised to new 3-D blob configurations.

```bash
# All 3D (standard + perturbed)
python -m dualwcfm.experiments.run \
    --mode 3d \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

# Individual levels
python -m dualwcfm.experiments.run --mode 3d --level simple \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode 3d --level medium \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode 3d --level complex \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

# Perturbed variants
python -m dualwcfm.experiments.run --mode 3d --level simple --perturbed \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode 3d --level medium --perturbed \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode 3d --level complex --perturbed \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results
```

---

## 4. E4 — Image Experiments (MNIST / FashionMNIST)

Grayscale 32×32. Reference transport = horizontal flip (simple) or flip + 10° rotation (medium).

```bash
# All image experiments
python -m dualwcfm.experiments.run \
    --mode image \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

# Individual
python -m dualwcfm.experiments.run --mode image --level simple \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode image --level medium \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

# Perturbed
python -m dualwcfm.experiments.run --mode image --level simple --perturbed \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode image --level medium --perturbed \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results
```

---

## 5. E5 — Color-Domain Transfer (MNIST RGB / CIFAR-10)

RGB 3-channel experiments; tests adaptation to orthogonal color-channel shifts.

```bash
# All color experiments
python -m dualwcfm.experiments.run \
    --mode color \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

# Individual
python -m dualwcfm.experiments.run --mode color --level simple \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode color --level medium \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results

python -m dualwcfm.experiments.run --mode color --level complex \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 --output results
```

---

## 6. Baseline Comparison (all methods)

Trains OT-CFM and random-CFM from scratch, and evaluates the pretrained-only ablation.
Results land in `results/baselines_summary.csv` and auto-generate LaTeX tables.

### 2-D baselines

```bash
# OT-CFM (Sinkhorn-paired) — primary quantitative baseline
python -m dualwcfm.baselines.run \
    --baseline otcfm --mode 2d \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

# Random-paired CFM (no OT coupling)
python -m dualwcfm.baselines.run \
    --baseline cfm --mode 2d \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

# Pretrained-only DualWCFM (no dual adaptation) — ablation baseline
python -m dualwcfm.baselines.run \
    --baseline pretrained --mode 2d \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

# All at once
python -m dualwcfm.baselines.run \
    --baseline all --mode 2d \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results
```

### 3-D baselines

```bash
python -m dualwcfm.baselines.run \
    --baseline all --mode 3d \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results
```

### Gushchin baselines

```bash
python -m dualwcfm.baselines.run \
    --baseline all --mode gushchin \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results
```

### DSB — Diffusion Schrödinger Bridge (De Bortoli et al., NeurIPS 2021)

```bash
python -m dualwcfm.baselines.run \
    --baseline dsb --mode 2d \
    --dsb-iters 10 --epochs 2000 \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

# Stochastic SDE variant (sigma > 0)
python -m dualwcfm.baselines.run \
    --baseline dsb --mode 2d --dsb-sigma 0.1 \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

python -m dualwcfm.baselines.run \
    --baseline dsb --mode 3d \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

python -m dualwcfm.baselines.run \
    --baseline dsb --mode gushchin \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results
```

### LightSB (Korotin et al., ICML 2023)

```bash
python -m dualwcfm.baselines.run \
    --baseline lightsb --mode 2d \
    --lightsb-eps 0.1 --epochs 5000 \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

# Sweep regularisation epsilon
python -m dualwcfm.baselines.run \
    --baseline lightsb --mode 2d --lightsb-eps 0.5 \
    --output results

python -m dualwcfm.baselines.run \
    --baseline lightsb --mode gushchin \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results
```

### MetaOT / Neural-W₂ OT (Amos et al., NeurIPS 2022)

```bash
python -m dualwcfm.baselines.run \
    --baseline metaot --mode 2d \
    --epochs 10000 \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

python -m dualwcfm.baselines.run \
    --baseline metaot --mode 3d \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

python -m dualwcfm.baselines.run \
    --baseline metaot --mode gushchin \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results
```

### Filter to a specific experiment level

```bash
python -m dualwcfm.baselines.run \
    --baseline otcfm --mode 2d --level simple \
    --output results

python -m dualwcfm.baselines.run \
    --baseline otcfm --mode 2d --level medium --perturbed \
    --output results
```

---

## 7. E7 — Ablation Studies

### E7a — Particle ratio α

```bash
python -m dualwcfm.experiments.run \
    --ablation --ablation-type particle_ratio --mode 2d --level medium \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results
```

### E7b — Noise level σ

```bash
python -m dualwcfm.experiments.run \
    --ablation --ablation-type sigma --mode 2d --level medium \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results
```

### E7c — Number of outer iterations N

```bash
python -m dualwcfm.experiments.run \
    --ablation --ablation-type iterations --mode 2d --level medium \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results
```

### E7d — Method comparison (Standard / Mixture / Score)

```bash
# On 2D simple
python -m dualwcfm.experiments.run \
    --ablation --ablation-type method --mode 2d --level simple \
    --methods standard mixture score \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

# On 2D medium
python -m dualwcfm.experiments.run \
    --ablation --ablation-type method --mode 2d --level medium \
    --methods standard mixture score \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results
```

### E7f — Multi-seed stability (seeds 0–4)

```bash
python -m dualwcfm.experiments.run \
    --ablation --ablation-type seeds --mode 2d --level medium \
    --seeds 0 1 2 3 4 \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results
```

### All ablations at once

```bash
python -m dualwcfm.experiments.run \
    --ablation --ablation-type all --mode 2d --level medium \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results
```

---

## 8. Full Benchmark (sequential, all experiments + baselines)

```bash
# Step 1 — Train DualWCFM on every experiment
python -m dualwcfm.experiments.run \
    --all \
    --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
    --output results

# Step 2 — Run all baselines on all modes
# "all" covers: otcfm, cfm, pretrained, dsb, lightsb, metaot
for MODE in 2d 3d gushchin; do
    python -m dualwcfm.baselines.run \
        --baseline all --mode $MODE \
        --wandb-project "Data-Based EOT" --wandb-entity antoine311200 \
        --output results
done

# Step 3 — Generate comparison tables (no retraining)
python -m dualwcfm.baselines.run --table --output results --metric W2_nu
python -m dualwcfm.baselines.run --table --output results --metric map_err_fwd
python -m dualwcfm.baselines.run --table --output results --metric bw2
python -m dualwcfm.baselines.run --table --output results --metric fid

# Or all metrics at once
python -m dualwcfm.baselines.run --table --output results --metric all
```

---

## 9. Load Pre-trained Models (skip marginal + pretrain phases)

Useful for re-running the adaptation phase with a different method or hyperparameter.

```bash
# Re-use saved weights from a previous run
python -m dualwcfm.experiments.run \
    --mode 2d --level medium \
    --load-pretrained \
    --output results

# Point to an explicit checkpoint directory
python -m dualwcfm.experiments.run \
    --mode 2d --level medium \
    --pretrain-dir results/2d_medium/models \
    --output results
```

---

## 10. Output Structure

```
results/
├── baselines_summary.csv          # all baseline metrics, one row per (exp, method)
├── comparison_W2_nu.csv/.tex      # pivot table: rows=experiments, cols=methods
├── comparison_map_err_fwd.csv/.tex
├── comparison_bw2.csv/.tex
├── comparison_fid.csv/.tex
│
├── 2d_simple/
│   ├── metrics.csv                # per-iteration: W1/W2, MMD, sinkhorn_div,
│   │                              #   map_err_fwd, map_err_bwd, ESS, weights
│   ├── metrics.png
│   ├── weights_evolution.png
│   ├── timelapse_coupling.gif
│   ├── timelapse_marginals.gif
│   ├── timelapse_weights.gif
│   └── models/
│       ├── initial_forward_model.pth
│       ├── initial_backward_model.pth
│       ├── initial_mu_model.pth
│       └── initial_nu_model.pth
│
├── 2d_simple/baselines/
│   ├── otcfm/result.csv
│   ├── cfm/result.csv
│   └── pretrained_only/result.csv
│
└── ...  (same structure for every experiment)
```

---

## Metrics Reference

| Metric | Applies to | Description |
|--------|-----------|-------------|
| `W1_mu`, `W2_mu` | all | Sliced-Wasserstein SW₁/SW₂ on the source marginal |
| `W1_nu`, `W2_nu` | all | Sliced-Wasserstein SW₁/SW₂ on the target marginal |
| `mmd_nu` | all | Maximum Mean Discrepancy (RBF, median bandwidth) |
| `energy_dist_nu` | all | Energy distance (kernel-free) |
| `sinkhorn_div` | low-dim | Debiased Sinkhorn divergence (smooth W₂ proxy) |
| `map_err_fwd` | 2d/3d/image | `sqrt(E[‖T(x)−y‖²])` — forward transport map RMSE |
| `map_err_bwd` | 2d/3d/image | `sqrt(E[‖T⁻¹(y)−x‖²])` — backward transport map RMSE |
| `bw2` | gushchin | Bures-Wasserstein₂ to the analytical Gaussian EOT plan |
| `mean_err_nu` | gushchin | L2 error of empirical mean vs. ground-truth |
| `cov_frob_nu` | gushchin | Frobenius error of empirical covariance |
| `fid` | image/color | Fréchet Inception Distance |
| `kid` | image/color | Kernel Inception Distance |
| `weight_ess` | all | Effective Sample Size of importance weights (∈ [0,1]) |
