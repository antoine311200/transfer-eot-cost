"""dualwcfm.experiments.exp_ablations
=====================================
Pre-defined ablation suites for the NeurIPS paper (E7 in the experiment plan).

Each function returns an :class:`~dualwcfm.experiments.ablation.AblationSuite`
that can be passed directly to :class:`~dualwcfm.experiments.ablation.AblationRunner`.

Ablation families
-----------------
particle_ratio  (E7a) – α ∈ {0, 0.1, 0.25, 0.5, 1.0} on 2D medium
sigma           (E7b) – σ ∈ {0.001, 0.01, 0.05, 0.1, 0.5} on 2D medium
iterations      (E7c) – N ∈ {10, 25, 50, 100, 200} on 2D medium
method          (E7d) – Standard vs. Mixture vs. Score on 2D simple & medium
reference_noise (E7e) – reference coupling quality: σ_ref ∈ {0, 0.1, 0.5, 1.0}
seeds           (E7f) – multi-seed stability (seeds 0-4) on 2D medium

Usage::

    from dualwcfm.experiments.exp_ablations import make_all_ablation_suites
    from dualwcfm.experiments.ablation import AblationRunner

    for suite in make_all_ablation_suites(output_root="ablation_results"):
        runner = AblationRunner(suite, output_root="ablation_results")
        runner.run()
        runner.plot_convergence()
        runner.export_latex_table()
"""

from __future__ import annotations

from typing import List

from dualwcfm.experiments.ablation import AblationSuite, AblationVariant
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.methods.standard import StandardConfig
from dualwcfm.methods.score import ScoreConfig


# ---------------------------------------------------------------------------
# Shared default for all 2D ablations
# ---------------------------------------------------------------------------

def _base_method_cfg(**overrides) -> MixtureConfig:
    defaults = dict(
        sigma           = 0.01,
        sigma2          = 0.01,
        loss            = "weighted",
        l2_reg          = 0,
        lip_reg         = 0,
        mean_reg        = 0,
        use_noisy_field = False,
        particle_ratio  = 0.25,
    )
    defaults.update(overrides)
    return MixtureConfig(**defaults)


def _get_2d_medium(output_root: str):
    from dualwcfm.experiments.exp_2d import make_2d_experiments
    specs = make_2d_experiments(output_root)
    return next(s for s in specs if s.name == "2d_medium")


def _get_2d_simple(output_root: str):
    from dualwcfm.experiments.exp_2d import make_2d_experiments
    specs = make_2d_experiments(output_root)
    return next(s for s in specs if s.name == "2d_simple")


# ---------------------------------------------------------------------------
# E7a — Particle ratio
# ---------------------------------------------------------------------------

def make_particle_ratio_ablation(output_root: str = "ablation_results") -> AblationSuite:
    """E7a: sweep particle_ratio ∈ {0, 0.1, 0.25, 0.5, 1.0} on 2D medium.

    Expected finding:
    - ratio=0.0  → weight ESS collapses (circularity); slow/no convergence
    - ratio=0.25 → best balance (our default)
    - ratio=1.0  → slightly higher SW₂ due to accumulated Euler error
    """
    base_spec = _get_2d_medium(output_root)

    variants = [
        AblationVariant(
            name       = f"ratio_{r}".replace(".", "p"),
            method_cfg = _base_method_cfg(particle_ratio=r),
            seed       = 0,
            label      = f"α = {r}",
        )
        for r in [0.0, 0.1, 0.25, 0.5, 1.0]
    ]

    return AblationSuite(
        name      = "ablation_particle_ratio",
        base_spec = base_spec,
        variants  = variants,
    )


# ---------------------------------------------------------------------------
# E7b — Sigma (entropic regularisation)
# ---------------------------------------------------------------------------

def make_sigma_ablation(output_root: str = "ablation_results") -> AblationSuite:
    """E7b: sweep σ ∈ {0.001, 0.01, 0.05, 0.1, 0.5} on 2D medium.

    Expected finding:
    - Very small σ → sharp but numerically unstable coupling
    - σ ≈ 0.01    → good default for 2D (normalised space)
    - Large σ     → over-smoothed coupling, missing fine structure
    """
    base_spec = _get_2d_medium(output_root)

    variants = [
        AblationVariant(
            name       = f"sigma_{s}".replace(".", "p"),
            method_cfg = _base_method_cfg(sigma=s, sigma2=s, particle_ratio=0.1),
            seed       = 0,
            label      = f"σ = {s}",
        )
        for s in [0.001, 0.01, 0.05, 0.1, 0.5]
    ]

    return AblationSuite(
        name      = "ablation_sigma",
        base_spec = base_spec,
        variants  = variants,
    )


# ---------------------------------------------------------------------------
# E7c — Number of iterations
# ---------------------------------------------------------------------------

def make_iterations_ablation(output_root: str = "ablation_results") -> AblationSuite:
    """E7c: sweep N ∈ {10, 25, 50, 100, 200} on 2D medium.

    Expected finding:
    - N < 25  → insufficient convergence
    - N = 50-100 → plateau in quality
    - N = 200 → no further improvement (shows convergence saturation)
    """
    base_spec = _get_2d_medium(output_root)

    variants = [
        AblationVariant(
            name           = f"iters_{n}",
            method_cfg     = _base_method_cfg(particle_ratio=0.25),
            train_overrides= {"iterations": n},
            seed           = 0,
            label          = f"N = {n}",
        )
        for n in [10, 25, 50, 100, 200]
    ]

    return AblationSuite(
        name      = "ablation_iterations",
        base_spec = base_spec,
        variants  = variants,
    )


# ---------------------------------------------------------------------------
# E7d — Method comparison
# ---------------------------------------------------------------------------

def make_method_ablation(output_root: str = "ablation_results") -> List[AblationSuite]:
    """E7d: compare Standard vs. Mixture vs. Score on 2D simple and medium.

    Produces two AblationSuites (one per experiment level).
    """
    shared_kw = dict(
        sigma           = 0.01,
        sigma2          = 0.01,
        loss            = "weighted",
        l2_reg          = 0,
        lip_reg         = 0,
        mean_reg        = 0,
        use_noisy_field = False,
    )

    variants = [
        AblationVariant(
            name       = "standard",
            method_cfg = StandardConfig(**shared_kw),
            seed       = 0,
            label      = "Standard (no mixing)",
        ),
        AblationVariant(
            name       = "mixture_10",
            method_cfg = MixtureConfig(**shared_kw, particle_ratio=0.10),
            seed       = 0,
            label      = "Mixture α=0.10",
        ),
        AblationVariant(
            name       = "mixture_25",
            method_cfg = MixtureConfig(**shared_kw, particle_ratio=0.25),
            seed       = 0,
            label      = "Mixture α=0.25 (default)",
        ),
        AblationVariant(
            name       = "score",
            method_cfg = ScoreConfig(**shared_kw),
            seed       = 0,
            label      = "Score matching",
        ),
    ]

    return [
        AblationSuite(
            name      = "ablation_method_simple",
            base_spec = _get_2d_simple(output_root),
            variants  = variants,
        ),
        AblationSuite(
            name      = "ablation_method_medium",
            base_spec = _get_2d_medium(output_root),
            variants  = variants,
        ),
    ]


# ---------------------------------------------------------------------------
# E7e — Reference coupling quality
# ---------------------------------------------------------------------------

def make_reference_noise_ablation(output_root: str = "ablation_results") -> AblationSuite:
    """E7e: degrade reference coupling quality with additive noise.

    Tests sensitivity to imperfect reference couplings — important for
    practical applications where the reference is estimated from data.

    Variants inject Gaussian noise (std = σ_ref) into the target component
    of the reference coupling before training.
    """
    import torch
    from dualwcfm.distributions.coupling import FunctionalCouplingDistribution

    base_spec = _get_2d_simple(output_root)

    # Rebuild the anticorrelation coupling with configurable target noise.
    import copy

    def _make_noisy_spec(noise_level: float) -> AblationVariant:
        def _noisy_anticorrelate(n: int):
            x = torch.randn(n, 2) * 0.5
            y = -x + torch.randn(n, 2) * noise_level
            return x, y

        spec_copy = copy.deepcopy(base_spec)
        spec_copy.initial_coupling = FunctionalCouplingDistribution(_noisy_anticorrelate)
        name = f"ref_noise_{noise_level}".replace(".", "p")
        return AblationVariant(
            name       = name,
            method_cfg = _base_method_cfg(particle_ratio=0.25),
            seed       = 0,
            label      = f"σ_ref = {noise_level}",
        )

    # Note: for this ablation the initial_coupling in base_spec is overridden
    # per-variant.  The AblationRunner deepcopies the spec, so we can safely
    # mutate per-variant initial_coupling via a custom variant.
    #
    # Since AblationVariant doesn't support coupling overrides directly, we
    # store the noise level in train_overrides as a no-op placeholder and
    # override the coupling externally before calling runner.run().
    # In practice, create separate ExperimentSpecs via the helper below.

    variants = [
        AblationVariant(
            name       = f"ref_noise_{nl}".replace(".", "p"),
            method_cfg = _base_method_cfg(particle_ratio=0.25),
            train_overrides = {"_ref_noise": nl},   # marker for external override
            seed       = 0,
            label      = f"σ_ref = {nl}",
        )
        for nl in [0.0, 0.1, 0.5, 1.0, 2.0]
    ]

    return AblationSuite(
        name      = "ablation_reference_noise",
        base_spec = base_spec,
        variants  = variants,
    )


# ---------------------------------------------------------------------------
# E7f — Multi-seed stability
# ---------------------------------------------------------------------------

def make_seed_ablation(output_root: str = "ablation_results") -> AblationSuite:
    """E7f: run 2D medium experiment with 5 random seeds.

    Reports mean ± std over seeds to verify reproducibility.
    Use runner.plot_seed_aggregate(metric='W2_nu') to visualise.
    """
    base_spec = _get_2d_medium(output_root)

    variants = [
        AblationVariant(
            name       = f"mixture_seed{seed}",
            method_cfg = _base_method_cfg(particle_ratio=0.25),
            seed       = seed,
            label      = f"Seed {seed}",
        )
        for seed in range(5)
    ]

    return AblationSuite(
        name      = "ablation_seeds",
        base_spec = base_spec,
        variants  = variants,
    )


# ---------------------------------------------------------------------------
# Combined catalogue
# ---------------------------------------------------------------------------

def make_all_ablation_suites(output_root: str = "ablation_results") -> List[AblationSuite]:
    """Return all ablation suites for the NeurIPS paper.

    Excludes E7e (reference noise) which requires coupling override logic not
    yet integrated into AblationRunner.
    """
    suites: List[AblationSuite] = [
        make_particle_ratio_ablation(output_root),
        make_sigma_ablation(output_root),
        make_iterations_ablation(output_root),
        make_seed_ablation(output_root),
    ]
    suites.extend(make_method_ablation(output_root))
    return suites
