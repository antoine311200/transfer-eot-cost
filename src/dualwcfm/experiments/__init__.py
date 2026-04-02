"""dualwcfm.experiments
======================
Experiment definitions, runners, and ablation tools for generative EOT transfer.

Package layout
--------------
base             – ExperimentSpec, run_experiment(), setup_logging(), save_metrics_plot()
exp_2d           – 2-D toy-distribution experiment catalogue
exp_3d           – 3-D toy-distribution experiment catalogue (GMM blobs)
exp_3d_manifold  – 3-D manifold experiments (Swiss Roll, Helices, Sphere, Rings→Helix)
exp_image        – Image (MNIST / FashionMNIST) experiment catalogue
exp_color        – Color-domain image transfer experiments (E5 — non-trivial cost)
exp_gushchin     – Gushchin-style Gaussian EOT benchmark with BW₂ metric (E1)
exp_ablations    – Pre-defined ablation suites (E7a-f)
ablation         – AblationSuite + AblationRunner for paper-quality comparisons
run              – CLI entry point  (python -m dualwcfm.experiments.run --help)
"""

from dualwcfm.experiments.base import ExperimentSpec, get_device, run_experiment, setup_logging
from dualwcfm.experiments.exp_2d import make_2d_experiments
from dualwcfm.experiments.exp_3d import make_3d_experiments
from dualwcfm.experiments.exp_3d_manifold import make_3d_manifold_experiments
from dualwcfm.experiments.exp_image import make_image_experiments
from dualwcfm.experiments.exp_color import make_color_experiments
from dualwcfm.experiments.exp_gushchin import make_gushchin_experiments
from dualwcfm.experiments.exp_ablations import (
    make_all_ablation_suites,
    make_particle_ratio_ablation,
    make_sigma_ablation,
    make_iterations_ablation,
    make_method_ablation,
    make_seed_ablation,
)

__all__ = [
    "ExperimentSpec",
    "get_device",
    "run_experiment",
    "setup_logging",
    "make_2d_experiments",
    "make_3d_experiments",
    "make_3d_manifold_experiments",
    "make_image_experiments",
    "make_color_experiments",
    "make_gushchin_experiments",
    "make_all_ablation_suites",
    "make_particle_ratio_ablation",
    "make_sigma_ablation",
    "make_iterations_ablation",
    "make_method_ablation",
    "make_seed_ablation",
]
