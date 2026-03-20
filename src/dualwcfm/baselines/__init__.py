"""dualwcfm.baselines
=====================
From-scratch baseline runners for paper comparison tables.

Package layout
--------------
base       -- BaselineSpec, BaselineRunner, compute_transport_metrics, sliced_wasserstein
otcfm      -- OT-CFM trained from scratch on (mu', nu') — primary quantitative baseline
pretrained -- DualWCFM pretrained-only (no dual-potential adaptation) — ablation baseline
table      -- Aggregate results into LaTeX / CSV comparison tables
run        -- CLI entry point  (python -m dualwcfm.baselines.run --help)

All baselines share the same metric structure as the main experiments
(W1_mu, W2_mu, W1_nu, W2_nu, transport_cost) and write to the same W&B
project so results appear in a unified dashboard.
"""

from dualwcfm.baselines.base import (
    BaselineSpec,
    BaselineRunner,
    evaluate_metrics,
    append_summary_csv,
)
from dualwcfm.metrics import sliced_wasserstein
from dualwcfm.baselines.otcfm import OTCFMRunner, OTCFMConfig
from dualwcfm.baselines.pretrained import PretrainedOnlyRunner
from dualwcfm.baselines.table import build_comparison_table, build_all_tables

__all__ = [
    "BaselineSpec",
    "BaselineRunner",
    "evaluate_metrics",
    "sliced_wasserstein",
    "append_summary_csv",
    "OTCFMRunner",
    "OTCFMConfig",
    "PretrainedOnlyRunner",
    "build_comparison_table",
    "build_all_tables",
]
