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

### Description

Follows the spirit of Gushchin et al. (NeurIPS 2023) *"Building the Bridge of Schrödinger: A Continuous EOT Benchmark"*. All source/target pairs are Gaussian or Gaussian-mixture distributions for which the ground-truth entropic OT plan (with quadratic cost and regularisation ε) is known, enabling exact quantitative comparison via the **Bures–Wasserstein distance (BW₂)** between the learned coupling and the ground truth.

Three experiment families are included:

| Sub-experiment | Distributions | Dim | ε levels | Reference coupling |
|---|---|---|---|---|
| **gaussian_iso_2d** | μ = N([0,0], I₂) → ν = N([3,2], [[2,0.5],[0.5,1]]) | R² | 0.1, 0.5, 2.0 | Closed-form Gaussian EOT plan (Janati et al., 2020) |
| **gaussian_gmm_2d** | 4-component GMM: src centers [±2, ±2], tgt centers [4,0],[0,4],[−4,0],[0,−4] (blob σ=0.4) | R² | 0.5, 2.0 | Sinkhorn oracle coupling (POT library) |
| **gaussian_iso_8d** | μ = N(0, I₈) → ν = N(1.5·**1**, 2·I₈) | R⁸ | 0.5 | Closed-form Gaussian EOT plan |

**New marginals** (unseen at pretraining): same distribution families shifted to a different region of space (shift [5,5] for iso-2d, [8,8] for gmm-2d, [5,…,5] for 8d), testing whether the learned transport structure generalises to new supports.

The noise level is set to σ = ε/4 in all cases.

**Primary metric:** BW₂ to the ground-truth or Sinkhorn-oracle coupling.  Secondary metrics: `mean_err_nu`, `cov_frob_nu`.

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

### Description

Each experiment encodes a **known point-transport map** on Gaussian-mixture blobs in R².  The reference coupling π is built from blobs at positions S₀; the new source μ′ and target ν′ live at **different** positions S₁ but follow the **same** transport dynamic — testing the algorithm's ability to generalise the learned map to unseen supports.

Three difficulty levels:

**simple — Anticorrelation map (x ↦ −x)**
- Reference: x ~ N(0, 0.5²·I₂), y = −x.
- New marginals: μ′ = N([10,10], 0.5²·I₂), ν′ = N([−10,−10], 0.5²·I₂).
- Ground-truth map: T(x) = −x (self-inverse).
- Perturbed variant: target shifted by [0, 2], so T(x) = −x + [0, 2].

**medium — 60° z-rotation, 2-blob → 4-blob GMM**
- Reference coupling: 2 blobs at [±2, 0], transport = R(60°).
- New marginals: 4-blob square at [±5, ±5]; target = R(60°) applied to source.
- Perturbed variant: target rotated by 70° instead of 60° (extra 10° mismatch).

**complex — 45° rotation, 5-blob cross → 3× larger cross**
- Reference coupling: 5-blob cross at radius r₀ = 1.5 (blob σ = 0.15), transport = R(45°).
- New marginals: 5-blob cross at radius r₁ = 4.5 (3× scale, blob σ = 0.4); target = R(45°) of source.
- Perturbed variant: target rotated by 55° (extra 10° mismatch).

**Metrics:** `map_err_fwd`, `map_err_bwd` (transport-map RMSE), `W2_nu`, `sinkhorn_div`.

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

### Description

Same design principle as E2, extended to R³.  The reference coupling encodes a **z-axis rotation** applied to Gaussian blobs; the new marginals use a **different number and arrangement of blobs** at different positions.  Perturbed variants add an extra **30° tilt around the x-axis** on the new target, probing robustness to out-of-plane displacement.

Three difficulty levels:

**simple — 90° z-rotation, 2 blobs → 3 blobs**
- Reference: 2 blobs at [±1.5, 0, 0], transport = R_z(90°).
- New marginals: 3 blobs at [3,±1,1] and [−3,0,1]; target = R_z(90°) of source.
- Perturbed: target additionally tilted R_x(30°).

**medium — 120° z-rotation, equilateral triangle → 4-blob rectangle**
- Reference: 3 blobs at the vertices of an equilateral triangle (r = 2 in z = 0), transport = R_z(120°).  Each vertex maps cyclically to the next.
- New marginals: 4-blob rectangle at [±4, ±2, z=1]; target = R_z(120°) of source.
- Perturbed: target additionally tilted R_x(30°).

**complex — 60° z-rotation, 4-blob square → 6-blob hexagon**
- Reference: 4-blob axis-aligned square at r = 2 in z = 0, transport = R_z(60°).
- New marginals: 6-blob regular hexagon at r = 4, z = 2; target = R_z(60°) of source.
- Perturbed: target additionally tilted R_x(30°).

**Metrics:** `map_err_fwd`, `map_err_bwd`, `W2_nu`.

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

### Description

Tests generalisation of a learned image-space transport dynamic to an **unseen digit/garment class**.  Images are grayscale, resized to 32×32, normalised to [−1, 1].  The model architecture switches from MLP to **UNet** (4 resolution levels, channel multiplier (1,2,3,4)).  The coupling is stored on disk via `DiskCouplingDistribution` due to the large image dataset size.

Two experiments:

**simple — MNIST digit 3 → digit 7, horizontal flip**
- Reference coupling: (x, flip(x)) for 2 000 class-3 images.  T(x) = flip(x) (self-inverse).
- New marginals: 2 000 class-7 images; target = flip of class-7.
- Perturbed: new target additionally rotated by 10°.
- Architecture: UNet with 32 base channels, 1 res-block per level; `dual_epochs=100`, `cfm_epochs=1000`, `iterations=10`.

**medium — FashionMNIST shirt (class 6) → sneaker (class 7), flip + 10° rotation**
- Reference coupling: (x, rotate(flip(x), 10°)) for 2 000 shirt images.  T = flip ∘ rotate(10°), T⁻¹ = rotate(−10°) ∘ flip.
- New marginals: 2 000 sneaker images; target = flip + 10° rotation of sneaker.
- Perturbed: new target additionally rotated by 10° (effective target rotation 20°).
- Architecture: UNet with 256 base channels.

**Metrics:** `map_err_fwd`, `map_err_bwd`, `fid`, `kid`, `W2_nu`.

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

### Description

Tests DualWCFM's ability to adapt a pretrained transport to **different image content while preserving the underlying transport structure**.  The core design invariant is `Supp(T(μ')) = Supp(ν')`: the new source and new target are crafted so that exactly the same transport map T that defines the reference coupling also maps the new source support to the new target support — but on different digit classes, making the task non-trivial.

All images are 3-channel RGB, 32×32, normalised to [−1, 1].  The same `BidirectionalUnetStrategy` UNet from E4 is reused with C = 3 input channels.

#### Transport map T_cyclic

The MNIST sub-experiments are built around a single cyclic RGB channel permutation:

```
T_cyclic(r, g, b) = (g, b, r)    # cyclic left-shift of channels
T_cyclic⁻¹(r, g, b) = (b, r, g)  # cyclic right-shift (inverse)
```

This induces two three-element orbits under repeated application:

| Orbit | Color 1 | Color 2 | Color 3 |
|-------|---------|---------|---------|
| Orbit A | RED (1,0,0) | BLUE (0,0,1) | GREEN (0,1,0) |
| Orbit B | PURPLE (1,0,1) | CYAN (0,1,1) | YELLOW (1,1,0) |

All pairs within each orbit have **equal quadratic transport cost** 2v² per pixel (where v is the colourisation intensity), so T_cyclic is a valid equal-cost transport for all three transitions in each orbit.

The reference coupling uses one step of Orbit A: **RED → BLUE** on digit 3.  The new marginals use one step of a different orbit or a different position in Orbit A on a different digit class.

#### Standard vs. Perturbed variants

- **Standard**: new source and target satisfy `Supp(T_cyclic(μ')) = Supp(ν')` exactly — the same transport applies perfectly.
- **Perturbed**: the new target is additionally rotated 10° in image space, breaking the exact support condition and testing robustness of the dual adaptation to slight distribution mismatch.

#### Sub-experiments

**mnist_rgb_simple — MNIST digit 3 (RED→BLUE) transferred to digit 7 (PURPLE→CYAN)**
- Reference coupling: (RED_digit3, BLUE_digit3) with T_cyclic on digit 3 (10 000 images per class).
- New marginals: source = PURPLE_digit7 (1,0,1), target = CYAN_digit7 (0,1,1).
- Why `Supp(T(μ')) = Supp(ν')`: T_cyclic(1,0,1) = (0,1,1) = CYAN ✓ (Orbit B, step 1).
- The pretrained model has seen only digit 3 with Orbit A colours; digit 7 with Orbit B colours is genuinely out-of-distribution, requiring adaptation.
- Perturbed: CYAN_digit7 additionally rotated 10°.

**mnist_rgb_complex — MNIST digit 3 (RED→BLUE) transferred to digit 7 (YELLOW→PURPLE)**
- Same reference coupling (RED→BLUE on digit 3).
- New marginals: source = YELLOW_digit7 (1,1,0), target = PURPLE_digit7 (1,0,1).
- Why `Supp(T(μ')) = Supp(ν')`: T_cyclic(1,1,0) = (1,0,1) = PURPLE ✓ (Orbit B, step 2).
- Harder than simple: YELLOW is further from RED in the orbit, and PURPLE is further from BLUE, requiring the model to generalise the cyclic rule two steps.
- Perturbed: PURPLE_digit7 additionally rotated 10°.

**mnist_rgb_flip_simple — MNIST digit 3 (RED→flip(BLUE)) transferred to digit 7 (PURPLE→flip(CYAN))**
- Reference coupling: (RED_digit3, hflip(BLUE_digit3)) — T_flip = hflip ∘ T_cyclic on digit 3.
- New marginals: source = PURPLE_digit7, target = hflip(CYAN_digit7).
- Why `Supp(T_flip(μ')) = Supp(ν')`: T_flip(PURPLE_digit7) = hflip(T_cyclic(PURPLE_digit7)) = hflip(CYAN_digit7) ✓.
- Adds a spatial challenge on top of the colour mismatch: the pretrained model has seen only non-flipped RED→BLUE; PURPLE inputs already confuse it, and the target is also spatially reflected.
- Perturbed: hflip(CYAN_digit7) additionally rotated 10°.

**mnist_rgb_flip_complex — MNIST digit 3 (RED→flip(BLUE)) transferred to digit 7 (YELLOW→flip(PURPLE))**
- Same reference coupling as mnist_rgb_flip_simple.
- New marginals: source = YELLOW_digit7 (1,1,0), target = hflip(PURPLE_digit7).
- Why `Supp(T_flip(μ')) = Supp(ν')`: T_flip(YELLOW_digit7) = hflip(T_cyclic(YELLOW_digit7)) = hflip(PURPLE_digit7) ✓.
- Hardest MNIST variant: combines the two-channel source mismatch of mnist_rgb_complex with the spatial flip, maximising the pretrained model's transport error before adaptation.
- Perturbed: hflip(PURPLE_digit7) additionally rotated 10°.

**cifar_hue_simple — CIFAR-10 automobile (+0.3 hue) transferred to airplane (+0.3 hue)**
- Reference coupling: (automobile, hue_rotated_automobile) with hue factor +0.3, using 5 000 images.
- New marginals: airplane images (class 0); target = same +0.3 hue rotation applied to airplane.
- Why `Supp(T(μ')) = Supp(ν')`: T = hue rotation by +0.3; the same T is applied to the new class, so the invariant holds exactly.
- Why interesting: automobiles (red-heavy) and airplanes (blue-heavy) have very different intrinsic colour profiles; the reference coupling was trained on one profile.
- Perturbed: new target additionally rotated 10° in image space.

**cifar_hue_medium — CIFAR-10 automobile (+0.3 hue) transferred to ship (+0.3 hue)**
- Same design as cifar_hue_simple but new class = ship (class 8, blue-grey dominant).
- Perturbed: new target additionally rotated 10° in image space.

**Metrics:** `fid`, `kid`, `W2_nu`, `mmd_nu`.

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

### Description

All ablations are run on the **2D medium** experiment (60° rotation, 2-blob → 4-blob GMM) unless otherwise noted, using it as a canonical test bed.  Each ablation sweeps one hyperparameter while keeping all others at their default values.  Results are reported in terms of `map_err_fwd` and `W2_nu` convergence curves.

Six ablation families:

| ID | Parameter | Values swept | Purpose |
|---|---|---|---|
| E7a | Particle ratio α | {0, 0.1, 0.25, 0.5, 1.0} | Fraction of new-marginal samples mixed into the CFM loss; α=0 causes weight-ESS collapse |
| E7b | Noise level σ | {0.001, 0.01, 0.05, 0.1, 0.5} | Regularisation of the Mixture transport; too small → over-fitting; too large → blurring |
| E7c | Outer iterations N | {10, 25, 50, 100, 200} | Number of dual–CFM alternation rounds; tests convergence rate |
| E7d | Method | Standard / Mixture / Score | Compares three dual-potential parameterisations on 2D simple & medium |
| E7e | Reference noise σ_ref | {0, 0.1, 0.5, 1.0} | Quality of the reference coupling; tests robustness to noisy/approximate references |
| E7f | Random seed | 0, 1, 2, 3, 4 | Multi-seed stability on 2D medium |

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
