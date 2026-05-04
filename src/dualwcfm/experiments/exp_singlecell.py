"""dualwcfm.experiments.exp_singlecell
=======================================
Single-cell drug response transfer experiment using the Sci-Plex3 dataset
(Srivatsan et al., Science 2020).

Experimental design
-------------------
We apply DualWCFM to the problem of cross-cell-line drug response transfer —
a canonical challenge in computational biology addressed by CellOT (Bunne et al., NeurIPS 2022).

Reference coupling (source cell line, e.g. A549):
    Source x: control cells (Vehicle) in PCA space
    Target y: drug-treated cells (Givinostat 1 μM) in PCA space
    Coupling: NN-matched pairs in PCA space (≈ Monge map, CellOT protocol)

New marginals (target cell line, e.g. K562):
    Source μ: control cells of the new cell line
    Target ν: drug-treated cells of the new cell line (same drug, same dose)
              ["standard" variant]
          OR drug-treated at a different dose
              ["perturbed" variant — dose mismatch as distributional shift]

No analytical transport map is available (real biological data),
so evaluation uses Wasserstein distance / MMD between predicted and true targets.

Preprocessing follows CellOT:
    adata.X (already normalized + log1p) → 1000 HVGs → PCA 50 dims

Data
----
Dataset : Sci-Plex3 h5ad from Zenodo (https://zenodo.org/records/13350497)
          ~2.5 GB, file: SrivatsanTrapnell2020_sciplex3.h5ad
          OR loaded automatically via ``pertpy.data.sciplex3_raw()``.

Usage
-----
    from dualwcfm.experiments.exp_singlecell import make_singlecell_experiments
    specs = make_singlecell_experiments(data_path="path/to/sciplex3.h5ad")
"""
from __future__ import annotations

from typing import List, Optional

from dualwcfm.config import FeedForwardConfig, TrainingConfig, VisualizeConfig
from dualwcfm.methods.mixture import MixtureConfig
from dualwcfm.experiments.base import ExperimentSpec, get_device
from dualwcfm.data.sciplexdata import SciplexLoader, build_sciplex_condition


# ---------------------------------------------------------------------------
# Experimental constants — real Sci-Plex3 conditions
# ---------------------------------------------------------------------------

#: Reference cell line: A549 (lung adenocarcinoma)
_REF_CELL_LINE   = "A549"

#: Drug used in reference and transfer experiments (HDAC inhibitor)
#: Givinostat is a hold-out drug in the CellOT benchmark.
_DRUG_NAME       = "Givinostat"
_DOSE_STANDARD   = 1.0    # μM — standard dose used in most experiments
_DOSE_PERTURBED  = 0.1    # μM — lower dose as distributional shift

#: Transfer cell lines
_NEW_CELL_LINE_STD  = "K562"   # standard transfer: same drug, same dose
_NEW_CELL_LINE_PERT = "MCF7"   # perturbed transfer: different cell line

#: PCA dimensionality (matches CellOT)
_N_PCS  = 50
_N_HVG  = 1000


# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

def _default_train_cfg(device, n_pcs: int) -> TrainingConfig:
    return TrainingConfig(
        device                    = device,
        dataset_size              = 100_000,
        marginal_epochs           = 3_000,
        pretrain_epochs           = 8_000,
        dual_epochs               = 100,
        cfm_epochs                = 500,
        advance_steps             = 50,
        iterations                = 100,
        num_rectified_flow        = 0,
        scale_factor              = 20,
        batch_size                = 2048,
        coupling_chunk_size       = 10_000,
        save_coupling_checkpoints = False,
        save_model_checkpoints    = False,
    )


def _siren_train_cfg(device, n_pcs: int) -> TrainingConfig:
    return TrainingConfig(
        device                    = device,
        dataset_size              = 100_000,
        marginal_epochs           = 5_000,
        pretrain_epochs           = 12_000,
        dual_epochs               = 150,
        cfm_epochs                = 750,
        advance_steps             = 50,
        iterations                = 100,
        num_rectified_flow        = 0,
        scale_factor              = 20,
        batch_size                = 2048,
        coupling_chunk_size       = 10_000,
        save_coupling_checkpoints = False,
        save_model_checkpoints    = False,
    )


def _default_method_cfg() -> MixtureConfig:
    return MixtureConfig(
        sigma              = 0.05,
        sigma2             = 0.05,
        loss               = "weighted",
        l2_reg             = 5e-3,
        lip_reg            = 1.0,
        mean_reg           = 0,
        use_noisy_field    = False,
        particle_ratio     = 0.2,
        lr_cosine_decay    = False,
        lr_min_factor      = 0.5,
        dual_warmup_epochs = 20,
    )


def _default_model_cfg(n_pcs: int) -> FeedForwardConfig:
    return FeedForwardConfig(
        dimension          = n_pcs,
        hidden_size        = 256,
        num_layers         = 4,
        flow_learning_rate = 1e-3,
        nu_learning_rate   = 1e-3,
        mu_learning_rate   = 1e-3,
        dual_learning_rate = 1e-3,
        dual_l2_reg        = 0,
    )


def _siren_model_cfg(n_pcs: int) -> FeedForwardConfig:
    return FeedForwardConfig(
        dimension          = n_pcs,
        hidden_size        = 512,
        num_layers         = 5,
        flow_learning_rate = 5e-4,
        nu_learning_rate   = 5e-4,
        mu_learning_rate   = 5e-4,
        dual_learning_rate = 5e-4,
        dual_l2_reg        = 0,
        use_siren          = True,
        siren_omega_0      = 30.0,
        siren_hidden_omega = 30.0,
    )


# ---------------------------------------------------------------------------
# Experiment factory
# ---------------------------------------------------------------------------

def _singlecell(
    loader: SciplexLoader,
    perturbed: bool,
    output_root: str,
    use_siren: bool = False,
) -> ExperimentSpec:
    """Build one ExperimentSpec from a fitted SciplexLoader.

    Standard (perturbed=False):
        Reference: A549,  Vehicle → Givinostat 1 μM
        New source: K562, Vehicle
        New target: K562, Givinostat 1 μM

    Perturbed (perturbed=True):
        Reference: A549,  Vehicle → Givinostat 1 μM   (same)
        New source: MCF7, Vehicle
        New target: MCF7, Givinostat 0.1 μM           (lower dose = distributional shift)
    """
    device = get_device()

    new_cell_line = _NEW_CELL_LINE_PERT if perturbed else _NEW_CELL_LINE_STD
    new_dose      = _DOSE_PERTURBED     if perturbed else _DOSE_STANDARD

    # Reference coupling — A549, Givinostat 1 μM
    ref_coupling, ref_ctrl, ref_drug = build_sciplex_condition(
        loader,
        cell_line    = _REF_CELL_LINE,
        perturbation = _DRUG_NAME,
        dose_value   = _DOSE_STANDARD,
        device       = device,
    )

    # New marginals — transfer cell line
    _, new_src, new_tgt = build_sciplex_condition(
        loader,
        cell_line    = new_cell_line,
        perturbation = _DRUG_NAME,
        dose_value   = new_dose,
        device       = device,
    )

    n_pcs = loader.n_pcs_actual

    train_cfg  = _siren_train_cfg(device, n_pcs)  if use_siren else _default_train_cfg(device, n_pcs)
    method_cfg = _default_method_cfg()
    model_cfg  = _siren_model_cfg(n_pcs)           if use_siren else _default_model_cfg(n_pcs)

    vis_cfg = VisualizeConfig(
        show_each          = 5,
        n_trajectories     = 800,
        trajectory_stride  = 2,
        show_each_marginal = 100,
        show_each_pretrain = 500,
    )

    siren_suffix = "_siren" if use_siren else ""
    suffix = "_perturbed" if perturbed else ""
    return ExperimentSpec(
        name                  = f"singlecell{siren_suffix}{suffix}",
        mode                  = "singlecell",
        initial_coupling      = ref_coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = None,   # no analytical ground truth with real data
        inverse_transport_map = None,
        method_cfg            = method_cfg,
        output_root           = output_root,
        visualizer_kwargs     = {
            "ref_cell_line":  _REF_CELL_LINE,
            "new_cell_line":  new_cell_line,
            "drug_name":      _DRUG_NAME,
            "ref_drug_emb":   ref_drug,     # SciplexMarginalDistribution (true ref target)
            "new_tgt_dist":   new_tgt,      # SciplexMarginalDistribution (true new target)
            "n_pcs":          n_pcs,
        },
    )


def _singlecell_extra(
    loader: SciplexLoader,
    output_root: str,
) -> ExperimentSpec:
    """MCF7 at standard dose (1 µM) — same cell line as perturbed but at the standard dose.

    Reference: A549, Vehicle → Givinostat 1 µM  (same as always)
    New source: MCF7, Vehicle
    New target: MCF7, Givinostat 1 µM
    """
    device = get_device()

    ref_coupling, _, _ = build_sciplex_condition(
        loader,
        cell_line    = _REF_CELL_LINE,
        perturbation = _DRUG_NAME,
        dose_value   = _DOSE_STANDARD,
        device       = device,
    )
    _, new_src, new_tgt = build_sciplex_condition(
        loader,
        cell_line    = _NEW_CELL_LINE_PERT,   # MCF7
        perturbation = _DRUG_NAME,
        dose_value   = _DOSE_STANDARD,        # 1 µM (standard dose)
        device       = device,
    )

    n_pcs      = loader.n_pcs_actual
    train_cfg  = _default_train_cfg(device, n_pcs)
    method_cfg = _default_method_cfg()
    model_cfg  = _default_model_cfg(n_pcs)

    vis_cfg = VisualizeConfig(
        show_each          = 5,
        n_trajectories     = 800,
        trajectory_stride  = 2,
        show_each_marginal = 100,
        show_each_pretrain = 500,
    )

    return ExperimentSpec(
        name                  = "singlecell_mcf7_standard",
        mode                  = "singlecell",
        initial_coupling      = ref_coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = None,
        inverse_transport_map = None,
        method_cfg            = method_cfg,
        output_root           = output_root,
        visualizer_kwargs     = {
            "ref_cell_line": _REF_CELL_LINE,
            "new_cell_line": _NEW_CELL_LINE_PERT,
            "drug_name":     _DRUG_NAME,
            "ref_drug_emb":  None,
            "new_tgt_dist":  new_tgt,
            "n_pcs":         n_pcs,
        },
    )


def make_singlecell_experiments(
    data_path: Optional[str] = None,
    output_root: str = "results",
    n_hvg: int = _N_HVG,
    n_pcs: int = _N_PCS,
) -> List[ExperimentSpec]:
    """Return both single-cell experiment specs (standard + perturbed).

    Parameters
    ----------
    data_path:
        Path to ``SrivatsanTrapnell2020_sciplex3.h5ad``.
        If None, loads via ``pertpy.data.sciplex3_raw()``.
    output_root:
        Root directory for experiment outputs.
    n_hvg:
        Number of highly variable genes (default 1000, matching CellOT).
    n_pcs:
        Number of PCA dimensions (default 50, matching CellOT).
    """
    print(f"Loading Sci-Plex3 dataset (n_hvg={n_hvg}, n_pcs={n_pcs}) …")
    loader = SciplexLoader(
        data_path=data_path,
        n_hvg=n_hvg,
        n_pcs=n_pcs,
        perturbations=[_DRUG_NAME],  # keep only Vehicle + Givinostat to avoid OOM
    )
    print("  Done. Building experiment specs …")

    return [
        _singlecell(loader, False, output_root),
        _singlecell(loader, True,  output_root),
    ]


def _singlecell_otcfm_baseline(
    loader: SciplexLoader,
    cell_line: str,
    dose_value: float,
    output_root: str,
) -> ExperimentSpec:
    """OT-CFM baseline trained from scratch on a single cell line/dose.

    Uses the K562 (or MCF7) coupling directly as the reference coupling,
    then runs ZERO adaptation iterations.  The resulting model is trained
    from scratch on the new marginals using OT mini-batch pairing —
    equivalent to running OT-CFM with the quadratic cost on the new task.

    This serves as the cost-aware baseline required by reviewer I1.
    """
    device = get_device()

    # Use the new cell line's own coupling as the 'reference'
    ref_coupling, new_src, new_tgt = build_sciplex_condition(
        loader,
        cell_line    = cell_line,
        perturbation = _DRUG_NAME,
        dose_value   = dose_value,
        device       = device,
    )

    n_pcs = loader.n_pcs_actual

    train_cfg = TrainingConfig(
        device                    = device,
        dataset_size              = 100_000,
        marginal_epochs           = 0,        # No marginal bridge (same source/target)
        pretrain_epochs           = 8_000,    # Train CFM directly on K562/MCF7 pairs
        dual_epochs               = 0,        # No adaptation
        cfm_epochs                = 0,
        advance_steps             = 100,
        iterations                = 0,        # Zero adaptation iterations
        num_rectified_flow        = 0,
        scale_factor              = 20,
        batch_size                = 2048,
        coupling_chunk_size       = 10_000,
        save_coupling_checkpoints = False,
        save_model_checkpoints    = False,
    )
    method_cfg = _default_method_cfg()
    model_cfg  = _default_model_cfg(n_pcs)

    vis_cfg = VisualizeConfig(
        show_each=5, n_trajectories=800, trajectory_stride=2,
        show_each_marginal=100, show_each_pretrain=500,
    )

    cell_tag = cell_line.lower()
    dose_tag = str(dose_value).replace(".", "p")
    return ExperimentSpec(
        name                  = f"singlecell_otcfm_{cell_tag}_{dose_tag}um",
        mode                  = "singlecell",
        initial_coupling      = ref_coupling,
        new_source            = new_src,
        new_target            = new_tgt,
        train_cfg             = train_cfg,
        model_cfg             = model_cfg,
        vis_cfg               = vis_cfg,
        transport_map         = None,
        inverse_transport_map = None,
        method_cfg            = method_cfg,
        output_root           = output_root,
        visualizer_kwargs     = {
            "ref_cell_line": cell_line,
            "new_cell_line": cell_line,
            "drug_name":     _DRUG_NAME,
            "ref_drug_emb":  None,
            "new_tgt_dist":  new_tgt,
            "n_pcs":         n_pcs,
        },
    )


def make_singlecell_otcfm_baselines(
    data_path: Optional[str] = None,
    output_root: str = "results",
    n_hvg: int = _N_HVG,
    n_pcs: int = _N_PCS,
) -> List[ExperimentSpec]:
    """OT-CFM from-scratch baselines for all three transfer settings.

    Trains a flow matching model directly on each cell line's own coupling with
    OT mini-batch pairing (quadratic cost†).  Covers K562 1µM (standard),
    MCF7 1µM (extra), and MCF7 0.1µM (perturbed).
    """
    print(f"Loading Sci-Plex3 dataset (n_hvg={n_hvg}, n_pcs={n_pcs}) …")
    loader = SciplexLoader(
        data_path=data_path,
        n_hvg=n_hvg,
        n_pcs=n_pcs,
        perturbations=[_DRUG_NAME],
    )
    return [
        _singlecell_otcfm_baseline(loader, _NEW_CELL_LINE_STD,  _DOSE_STANDARD,  output_root),
        _singlecell_otcfm_baseline(loader, _NEW_CELL_LINE_PERT,  _DOSE_STANDARD,  output_root),
        _singlecell_otcfm_baseline(loader, _NEW_CELL_LINE_PERT,  _DOSE_PERTURBED, output_root),
    ]


def make_singlecell_extra_experiments(
    data_path: Optional[str] = None,
    output_root: str = "results",
    n_hvg: int = _N_HVG,
    n_pcs: int = _N_PCS,
) -> List[ExperimentSpec]:
    """MCF7 standard-dose (1 µM) experiment — new transfer target.

    This complements the existing perturbed (MCF7 0.1 µM) experiment by
    running the standard dose on the same MCF7 cell line.
    """
    print(f"Loading Sci-Plex3 dataset (n_hvg={n_hvg}, n_pcs={n_pcs}) …")
    loader = SciplexLoader(
        data_path=data_path,
        n_hvg=n_hvg,
        n_pcs=n_pcs,
        perturbations=[_DRUG_NAME],
    )
    print("  Done. Building MCF7-standard experiment spec …")
    return [_singlecell_extra(loader, output_root)]


def make_singlecell_siren_experiments(
    data_path: Optional[str] = None,
    output_root: str = "results",
    n_hvg: int = _N_HVG,
    n_pcs: int = _N_PCS,
) -> List[ExperimentSpec]:
    """Like make_singlecell_experiments but with SIREN networks and more training epochs."""
    print(f"Loading Sci-Plex3 dataset (n_hvg={n_hvg}, n_pcs={n_pcs}) ...")
    loader = SciplexLoader(
        data_path=data_path,
        n_hvg=n_hvg,
        n_pcs=n_pcs,
        perturbations=[_DRUG_NAME],
    )
    print("  Done. Building SIREN experiment specs ...")

    return [
        _singlecell(loader, False, output_root, use_siren=True),
        _singlecell(loader, True,  output_root, use_siren=True),
    ]
