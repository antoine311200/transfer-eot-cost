"""dualwcfm.experiments
======================
Experiment definitions, runners, and ablation tools for generative EOT transfer.

Package layout
--------------
base        – ExperimentSpec, run_experiment(), setup_logging(), save_metrics_plot()
exp_2d      – 2-D toy-distribution experiment catalogue
exp_3d      – 3-D toy-distribution experiment catalogue
exp_image   – Image (MNIST / FashionMNIST) experiment catalogue
ablation    – AblationSuite + AblationRunner for paper-quality comparisons
run         – CLI entry point  (python -m dualwcfm.experiments.run --help)
"""

from dualwcfm.experiments.base import ExperimentSpec, get_device, run_experiment, setup_logging
from dualwcfm.experiments.exp_2d import make_2d_experiments
from dualwcfm.experiments.exp_3d import make_3d_experiments
from dualwcfm.experiments.exp_image import make_image_experiments

__all__ = [
    "ExperimentSpec",
    "get_device",
    "run_experiment",
    "setup_logging",
    "make_2d_experiments",
    "make_3d_experiments",
    "make_image_experiments",
]
