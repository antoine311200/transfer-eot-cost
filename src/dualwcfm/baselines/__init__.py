"""dualwcfm.baselines
=====================
From-scratch baseline runners for paper comparison tables.

Package layout
--------------
base       -- BaselineSpec, BaselineRunner, evaluate_metrics, append_summary_csv
otcfm      -- OT-CFM trained from scratch on (mu', nu') — primary quantitative baseline
pretrained -- DualWCFM pretrained-only (no dual-potential adaptation) — ablation baseline
dsb        -- Diffusion Schrödinger Bridge via IPF (De Bortoli et al., NeurIPS 2021)
lightsb    -- LightSB via dual potentials + stochastic interpolant (Korotin et al., ICML 2023)
metaot     -- Neural W₂-OT map via ICNN potential (inner loop of MetaOT, Amos et al., NeurIPS 2022)
genot      -- GENOT: Generalized Neural OT via conditional flow on mini-batch plans (Bunne et al., NeurIPS 2023)
table      -- Aggregate results into LaTeX / CSV comparison tables
run        -- CLI entry point  (python -m dualwcfm.baselines.run --help)

All baselines share the same metric structure as the main experiments
(W1_mu, W2_mu, W1_nu, W2_nu, map_err_fwd, …) and write to the same W&B
project so results appear in a unified dashboard.
"""

from dualwcfm.baselines.base import (
    BaselineSpec,
    BaselineRunner,
    evaluate_metrics,
    append_summary_csv,
)
from dualwcfm.metrics import sliced_wasserstein
from dualwcfm.baselines.otcfm    import OTCFMRunner, OTCFMConfig
from dualwcfm.baselines.pretrained import PretrainedOnlyRunner
from dualwcfm.baselines.dsb      import DSBRunner, DSBConfig
from dualwcfm.baselines.lightsb  import LightSBRunner, LightSBConfig
from dualwcfm.baselines.metaot   import MetaOTRunner, MetaOTConfig
from dualwcfm.baselines.genot    import GENOTRunner, GENOTConfig
from dualwcfm.baselines.table    import build_comparison_table, build_all_tables

__all__ = [
    "BaselineSpec",
    "BaselineRunner",
    "evaluate_metrics",
    "sliced_wasserstein",
    "append_summary_csv",
    # Runners
    "OTCFMRunner",     "OTCFMConfig",
    "PretrainedOnlyRunner",
    "DSBRunner",       "DSBConfig",
    "LightSBRunner",   "LightSBConfig",
    "MetaOTRunner",    "MetaOTConfig",
    "GENOTRunner",     "GENOTConfig",
    # Tables
    "build_comparison_table",
    "build_all_tables",
]
