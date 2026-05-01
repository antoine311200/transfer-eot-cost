"""
make_singlecell_paper_figure.py
================================
Publication-quality figures for the DualWCFM single-cell drug-response experiment
(Sci-Plex3 dataset — Srivatsan et al., Science 2020).

Generated figures
-----------------
  main_transport  — 3-panel PCA figure: reference π | pretrained | DualWCFM adapted
  perturbed       — Same 3-panel figure for the MCF7 perturbed variant
  dual_transport  — 2×3 combined grid (both variants)
  convergence     — Training convergence: SW₂, MMD, ESS, Sinkhorn over iterations
  evolution       — Checkpoint evolution showing improvement per DualWCFM iteration
  coupling        — Coupling-arrow plots coloured by iteration
  metrics_bar     — Bar chart of final quantitative metrics
  all             — All of the above

Usage
-----
  # Standard variant only
  python make_singlecell_paper_figure.py \\
      --data-path /path/to/SrivatsanTrapnell2020_sciplex3.h5ad \\
      --std-dir results_cell/singlecell \\
      --figure main_transport

  # Generate all figures with SIREN comparison
  python make_singlecell_paper_figure.py \\
      --data-path /path/to/SrivatsanTrapnell2020_sciplex3.h5ad \\
      --std-dir  results_cell/singlecell \\
      --siren-dir results_cell_siren/singlecell_siren \\
      --figure all --output-dir figures/singlecell

  # Use UMAP projection instead of PCA (requires umap-learn)
  python make_singlecell_paper_figure.py \\
      --data-path /path/to/SrivatsanTrapnell2020_sciplex3.h5ad \\
      --std-dir results_cell/singlecell --umap
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(_SCRIPT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dualwcfm.mlp import MLP, torch_wrapper


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size":          9,
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "xtick.labelsize":    7.5,
        "ytick.labelsize":    7.5,
        "legend.fontsize":    7.5,
        "figure.dpi":         200,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.7,
        "grid.linewidth":     0.4,
        "grid.alpha":         0.30,
        "lines.linewidth":    1.5,
        "patch.linewidth":    0.5,
    })


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_COLOR_SRC    = "#4878d0"   # control / source  — blue
_COLOR_TGT    = "#d65f5f"   # true treated       — red
_COLOR_PRED   = "#6acc65"   # DualWCFM predicted — green
_COLOR_TRUE   = "#ffcf3e"   # true new target overlay — amber
_COLOR_SIREN  = "#956cb4"   # SIREN variant line — purple

_CMAP_DENSITY = LinearSegmentedColormap.from_list(
    "cell_density",
    ["#0d0221", "#1a0a4e", "#3a1278", "#6a2d9f", "#9c4bc7",
     "#c873d8", "#e6a8e0", "#f8d8f0", "#ffffff"],
    N=256,
)


# ---------------------------------------------------------------------------
# PCA / UMAP helpers
# ---------------------------------------------------------------------------

def fit_pca(X: np.ndarray, n_components: int = 2):
    """Fit 2-component PCA on X (N, d) → (components (n,d), mean (d,))."""
    mean = X.mean(0)
    Xc   = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:n_components], mean


def apply_pca(X: np.ndarray, comps: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return (X - mean) @ comps.T


def fit_umap(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.3):
    """Fit UMAP reducer (requires umap-learn).  Returns the fitted reducer."""
    from umap import UMAP
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors,
                   min_dist=min_dist, random_state=42)
    reducer.fit(X)
    return reducer


class Projector:
    """Unified PCA / UMAP projector fitted on a reference set."""

    def __init__(self, X_ref: np.ndarray, use_umap: bool = False):
        self._use_umap = use_umap
        if use_umap:
            print("  Fitting UMAP …")
            self._reducer = fit_umap(X_ref)
            self._comps = self._mean = None
        else:
            self._comps, self._mean = fit_pca(X_ref)
            self._reducer = None
        self.x_label = "UMAP 1" if use_umap else "PC 1"
        self.y_label = "UMAP 2" if use_umap else "PC 2"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self._use_umap:
            return self._reducer.transform(X)
        return apply_pca(X, self._comps, self._mean)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _infer_arch(state: dict):
    is_siren = any("model.final.weight" in k or ".linear.weight" in k for k in state)
    if is_siren:
        hidden  = state["model.net.0.linear.weight"].shape[0]
        n_layer = sum(1 for k in state if k.startswith("model.net.") and k.endswith(".linear.weight"))
        return hidden, n_layer, True
    linear_idx = []
    for key in state:
        parts = key.split(".")
        if len(parts) == 4 and parts[0] == "model" and parts[1] == "net" and parts[3] == "weight":
            linear_idx.append(int(parts[2]))
    n_layers = max(linear_idx) // 2 if linear_idx else 4
    hidden   = state["model.net.0.weight"].shape[0] if "model.net.0.weight" in state else 256
    return hidden, n_layers, False


def load_model(path: str, dim: int, device: torch.device) -> Optional[torch.nn.Module]:
    if not os.path.isfile(path):
        return None
    state = torch.load(path, map_location=device, weights_only=True)
    hidden, n_layers, is_siren = _infer_arch(state)
    if is_siren:
        from dualwcfm.models.siren import SirenMLP
        net = SirenMLP(dim=2*dim, out_dim=2*dim, w=hidden,
                       n_layers=n_layers, time_varying=True)
    else:
        net = MLP(dim=2*dim, out_dim=2*dim, w=hidden,
                  n_layers=n_layers, time_varying=True)
    model = torch_wrapper(net).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def find_latest_checkpoint(models_dir: str, direction: str = "forward") -> Optional[str]:
    """Return the highest-iteration checkpoint path, or None."""
    ckpt_dir = os.path.join(models_dir, "checkpoints", "models")
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = [
        f for f in os.listdir(ckpt_dir)
        if f.endswith(f"_{direction}.pth") and f.startswith("iter_")
    ]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(ckpt_dir, candidates[-1])


def find_all_checkpoints(models_dir: str, direction: str = "forward") -> list[str]:
    ckpt_dir = os.path.join(models_dir, "checkpoints", "models")
    if not os.path.isdir(ckpt_dir):
        return []
    candidates = sorted(
        f for f in os.listdir(ckpt_dir)
        if f.endswith(f"_{direction}.pth") and f.startswith("iter_")
    )
    return [os.path.join(ckpt_dir, f) for f in candidates]


# ---------------------------------------------------------------------------
# Euler integration
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_trajectory(
    x0_norm:   torch.Tensor,
    model:     torch.nn.Module,
    n_steps:   int = 120,
    scale:     float = 1.0,
    n_capture: int  = 30,
) -> list[np.ndarray]:
    """Return n_capture+1 snapshot arrays in data space (not normalised)."""
    N, d   = x0_norm.shape
    device = x0_norm.device
    dt     = 1.0 / n_steps
    xt     = torch.cat([x0_norm, x0_norm], dim=1)

    capture_at = {round(i * n_steps / n_capture) for i in range(n_capture + 1)}
    snaps: list[np.ndarray] = []
    if 0 in capture_at:
        snaps.append(x0_norm.cpu().numpy() * scale)

    for i in range(n_steps):
        t  = torch.full((N, 1), i * dt, device=device)
        vt = model(t, xt)
        xt = torch.cat([xt[:, :d] + vt[:, :d] * dt, xt[:, d:]], dim=1)
        if (i + 1) in capture_at:
            snaps.append(xt[:, :d].cpu().numpy() * scale)

    return snaps


@torch.no_grad()
def run_model(
    x0_norm:   torch.Tensor,
    model:     torch.nn.Module,
    n_steps:   int   = 120,
    scale:     float = 1.0,
    n_capture: int   = 30,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (final_np, snapshots_list)."""
    snaps = euler_trajectory(x0_norm, model, n_steps, scale, n_capture)
    return snaps[-1], snaps


# ---------------------------------------------------------------------------
# Density / KDE background
# ---------------------------------------------------------------------------

def _kde_background(ax, pts: np.ndarray, xlim, ylim, res: int = 200,
                    cmap=_CMAP_DENSITY, alpha: float = 0.75, bw: str = "scott"):
    try:
        from scipy.stats import gaussian_kde
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], res),
                             np.linspace(ylim[0], ylim[1], res))
        kde = gaussian_kde(pts.T, bw_method=bw)
        z   = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(res, res)
        if z.max() > 0:
            z /= z.max()
        ax.imshow(z, extent=[*xlim, *ylim], origin="lower",
                  cmap=cmap, aspect="auto", alpha=alpha, zorder=0)
    except Exception:
        pass


def _auto_lim(arrays: list[np.ndarray], pad: float = 0.6):
    pts = np.vstack(arrays)
    return (pts[:, 0].min() - pad, pts[:, 0].max() + pad), \
           (pts[:, 1].min() - pad, pts[:, 1].max() + pad)


# ---------------------------------------------------------------------------
# Core panel drawing
# ---------------------------------------------------------------------------

def draw_transport_panel(
    ax,
    proj:          Projector,
    src_raw:       np.ndarray,
    tgt_raw:       np.ndarray,
    title:         str,
    src_label:     str = "Control",
    tgt_label:     str = "Predicted",
    tgt_color:     str = _COLOR_PRED,
    true_tgt_raw:  Optional[np.ndarray] = None,
    true_tgt_label: str = "True target",
    snaps_raw:     Optional[list[np.ndarray]] = None,
    n_traj_lines:  int   = 70,
    pad:           float = 0.6,
    dot_size:      int   = 8,
    dot_alpha:     float = 0.60,
    line_alpha:    float = 0.10,
    show_metrics:  bool  = True,
    metrics:       Optional[dict] = None,
) -> None:
    src_2d  = proj(src_raw)
    tgt_2d  = proj(tgt_raw)
    pts_2d  = [src_2d, tgt_2d]
    if true_tgt_raw is not None:
        pts_2d.append(proj(true_tgt_raw))
    xlim, ylim = _auto_lim(pts_2d, pad=pad)

    all_kde = np.vstack([src_2d, tgt_2d])
    _kde_background(ax, all_kde, xlim, ylim)

    if snaps_raw is not None:
        snaps_2d = [proj(s) for s in snaps_raw]
        N        = len(src_2d)
        step     = max(1, N // n_traj_lines)
        for i in range(0, N, step):
            xs = [s[i, 0] for s in snaps_2d if i < len(s)]
            ys = [s[i, 1] for s in snaps_2d if i < len(s)]
            ax.plot(xs, ys, c="white", alpha=line_alpha, lw=0.45,
                    solid_capstyle="round", zorder=1)

    if true_tgt_raw is not None:
        tt_2d = proj(true_tgt_raw)
        ax.scatter(tt_2d[:, 0], tt_2d[:, 1],
                   s=dot_size * 0.75, c=_COLOR_TRUE, alpha=0.60,
                   linewidths=0.3, edgecolors="#b89300",
                   zorder=2, label=true_tgt_label, marker="D")

    kw = dict(linewidths=0, zorder=3, alpha=dot_alpha)
    ax.scatter(src_2d[:, 0], src_2d[:, 1], c=_COLOR_SRC, s=dot_size,
               marker="o", label=src_label, **kw)
    ax.scatter(tgt_2d[:, 0], tgt_2d[:, 1], c=tgt_color, s=dot_size,
               marker="^", label=tgt_label, **kw)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(proj.x_label, fontsize=8)
    ax.set_ylabel(proj.y_label, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_aspect("equal")
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.7,
              facecolor="#fafafa", edgecolor="#cccccc",
              handletextpad=0.4, borderpad=0.5)

    if show_metrics and metrics:
        lines = [f"{k} = {v:.3f}" for k, v in metrics.items()]
        ax.text(0.02, 0.02, "\n".join(lines),
                transform=ax.transAxes, fontsize=6,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.75, edgecolor="#cccccc"))


# ---------------------------------------------------------------------------
# Build main transport figure (3 panels)
# ---------------------------------------------------------------------------

def build_main_transport_figure(
    spec,
    models_dir:     str,
    device:         torch.device,
    n_particles:    int   = 500,
    n_steps:        int   = 120,
    n_traj_frames:  int   = 30,
    n_traj_lines:   int   = 80,
    pad:            float = 0.6,
    dot_size:       int   = 8,
    dot_alpha:      float = 0.65,
    use_umap:       bool  = False,
    output_stem:    str   = "singlecell_main",
    separate:       bool  = False,
) -> None:
    def _np(t):
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    scale = float(getattr(spec.train_cfg, "scale_factor", 20))
    vkw   = getattr(spec, "visualizer_kwargs", {}) or {}
    ref_cell = vkw.get("ref_cell_line", "A549")
    new_cell = vkw.get("new_cell_line", "K562")
    drug     = vkw.get("drug_name",    "Drug")
    new_tgt_dist = vkw.get("new_tgt_dist", None)

    print(f"  Sampling distributions (n={n_particles}) …")
    _ref_x, _ref_y = spec.initial_coupling.coupling_sample(n_particles)
    x_ref_np = _np(_ref_x)
    y_ref_np = _np(_ref_y)
    x_new_np = _np(spec.new_source.sample(n_particles))

    true_tgt_np: Optional[np.ndarray] = None
    if new_tgt_dist is not None:
        try:
            true_tgt_np = _np(new_tgt_dist.sample(min(n_particles, 400)))
        except Exception:
            pass
    if true_tgt_np is None:
        true_tgt_np = _np(spec.new_target.sample(n_particles))

    print("  Fitting projector …")
    all_raw = np.vstack([x_ref_np, y_ref_np, x_new_np, true_tgt_np])
    proj    = Projector(all_raw, use_umap=use_umap)

    dim = x_new_np.shape[1]
    x_new_norm = torch.tensor(x_new_np / scale, dtype=torch.float32, device=device)

    init_path  = os.path.join(models_dir, "initial_forward_model.pth")
    final_path = os.path.join(models_dir, "forward_model.pth")
    if not os.path.isfile(final_path):
        final_path = find_latest_checkpoint(models_dir, "forward") or ""

    print(f"  Loading pretrained model: {init_path}")
    model_init = load_model(init_path, dim, device)
    init_final_np = init_snaps = None
    if model_init is not None:
        print(f"    Integrating ({n_steps} steps) …")
        init_final_np, init_snaps = run_model(x_new_norm, model_init, n_steps, scale, n_traj_frames)
        del model_init

    print(f"  Loading adapted model:   {final_path}")
    model_final = load_model(final_path, dim, device)
    final_final_np = final_snaps = None
    if model_final is not None:
        print(f"    Integrating ({n_steps} steps) …")
        final_final_np, final_snaps = run_model(x_new_norm, model_final, n_steps, scale, n_traj_frames)
        del model_final

    def _metrics(pred_np, true_np):
        if pred_np is None or true_np is None:
            return {}
        from dualwcfm.metrics import sliced_wasserstein, mmd as _mmd
        p = torch.tensor(pred_np).float()
        t = torch.tensor(true_np).float()
        return {
            "SW₂": sliced_wasserstein(p, t, n_proj=200, p=2),
            "MMD": _mmd(p, t, max_samples=400),
        }

    panels = [
        dict(
            src_raw=x_ref_np, tgt_raw=y_ref_np, snaps_raw=[x_ref_np, y_ref_np],
            title=f"Reference coupling π\n({ref_cell}: control → {drug} 1 μM)",
            src_label=f"{ref_cell} control", tgt_label=f"{ref_cell} treated (true)",
            tgt_color=_COLOR_TGT, true_tgt_raw=None, metrics={},
        ),
        dict(
            src_raw=x_new_np, tgt_raw=init_final_np if init_final_np is not None else x_new_np,
            snaps_raw=init_snaps,
            title=f"Pretrained (before adaptation)\n({new_cell}: control → predicted)",
            src_label=f"{new_cell} control", tgt_label="Predicted (pretrained)",
            tgt_color=_COLOR_PRED, true_tgt_raw=true_tgt_np,
            true_tgt_label=f"{new_cell} treated (true)",
            metrics=_metrics(init_final_np, true_tgt_np),
        ),
        dict(
            src_raw=x_new_np, tgt_raw=final_final_np if final_final_np is not None else x_new_np,
            snaps_raw=final_snaps,
            title=f"DualWCFM (adapted)\n({new_cell}: control → predicted)",
            src_label=f"{new_cell} control", tgt_label="Predicted (DualWCFM)",
            tgt_color=_COLOR_PRED, true_tgt_raw=true_tgt_np,
            true_tgt_label=f"{new_cell} treated (true)",
            metrics=_metrics(final_final_np, true_tgt_np),
        ),
    ]

    _panel_kw = dict(proj=proj, pad=pad, dot_size=dot_size, dot_alpha=dot_alpha,
                     n_traj_lines=n_traj_lines, show_metrics=True)
    _stems    = ["reference", "pretrained", "adapted"]

    if separate:
        for stem, p in zip(_stems, panels):
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            draw_transport_panel(ax, **{k: v for k, v in p.items() if k != "metrics"},
                                 **_panel_kw, metrics=p.get("metrics"))
            _save(fig, f"{output_stem}_{stem}")
            plt.close(fig)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
        for ax, p in zip(axes, panels):
            draw_transport_panel(ax, **{k: v for k, v in p.items() if k != "metrics"},
                                 **_panel_kw, metrics=p.get("metrics"))
        fig.suptitle(
            f"DualWCFM — Single-Cell Drug Response Transfer ({drug})",
            fontsize=11, fontweight="bold", y=1.01,
        )
        plt.tight_layout(w_pad=1.2)
        _save(fig, output_stem)
        plt.close(fig)
    print(f"  Saved: {output_stem}.png / .pdf")


# ---------------------------------------------------------------------------
# 2×3 dual-variant figure (standard + perturbed)
# ---------------------------------------------------------------------------

def build_dual_transport_figure(
    spec_std,   models_dir_std:   str,
    spec_pert,  models_dir_pert:  str,
    device:     torch.device,
    n_particles: int   = 400,
    n_steps:     int   = 120,
    n_traj_frames: int = 30,
    use_umap:    bool  = False,
    output_stem: str   = "singlecell_dual",
) -> None:
    """2 rows × 3 columns: top=standard (K562), bottom=perturbed (MCF7)."""

    def _collect(spec, mdir):
        def _np(t):
            return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        sc   = float(getattr(spec.train_cfg, "scale_factor", 20))
        vkw  = getattr(spec, "visualizer_kwargs", {}) or {}
        new_cell     = vkw.get("new_cell_line",  "K562")
        drug         = vkw.get("drug_name",       "Drug")
        new_tgt_dist = vkw.get("new_tgt_dist",   None)

        _cx, _cy = spec.initial_coupling.coupling_sample(n_particles)
        x_ref = _np(_cx)
        y_ref = _np(_cy)
        x_new = _np(spec.new_source.sample(n_particles))
        true_tgt = (_np(new_tgt_dist.sample(min(n_particles, 300)))
                    if new_tgt_dist is not None else _np(spec.new_target.sample(n_particles)))
        dim = x_new.shape[1]
        x_norm = torch.tensor(x_new / sc, dtype=torch.float32, device=device)

        init_path  = os.path.join(mdir, "initial_forward_model.pth")
        final_path = os.path.join(mdir, "forward_model.pth")
        if not os.path.isfile(final_path):
            final_path = find_latest_checkpoint(mdir, "forward") or ""

        m_init  = load_model(init_path,  dim, device)
        m_final = load_model(final_path, dim, device)

        init_np = init_snaps = None
        if m_init is not None:
            init_np, init_snaps = run_model(x_norm, m_init, n_steps, sc, n_traj_frames)
            del m_init

        final_np = final_snaps = None
        if m_final is not None:
            final_np, final_snaps = run_model(x_norm, m_final, n_steps, sc, n_traj_frames)
            del m_final

        return dict(
            x_ref=x_ref, y_ref=y_ref, x_new=x_new, true_tgt=true_tgt,
            init_np=init_np, init_snaps=init_snaps,
            final_np=final_np, final_snaps=final_snaps,
            new_cell=new_cell, drug=drug, scale=sc,
        )

    print("  Collecting standard variant …")
    std  = _collect(spec_std,  models_dir_std)
    print("  Collecting perturbed variant …")
    pert = _collect(spec_pert, models_dir_pert)

    all_raw = np.vstack([
        std["x_ref"],  std["y_ref"],  std["x_new"],  std["true_tgt"],
        pert["x_ref"], pert["y_ref"], pert["x_new"], pert["true_tgt"],
    ])
    proj = Projector(all_raw, use_umap=use_umap)

    fig = plt.figure(figsize=(17, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.28)

    row_labels = [
        f"Standard ({std['new_cell']}, {std['drug']} 1 μM)",
        f"Perturbed ({pert['new_cell']}, {pert['drug']} 0.1 μM)",
    ]

    col_titles = [
        "Reference coupling π\n(A549 control → treated)",
        "Pretrained (before adaptation)",
        "DualWCFM (after adaptation)",
    ]

    for row_idx, dat in enumerate([std, pert]):
        row_tag = row_labels[row_idx]
        for col_idx in range(3):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            if col_idx == 0:
                draw_transport_panel(ax, proj, dat["x_ref"], dat["y_ref"],
                    title=col_titles[0] if row_idx == 0 else "",
                    src_label="A549 control", tgt_label="A549 treated",
                    tgt_color=_COLOR_TGT, true_tgt_raw=None,
                    snaps_raw=[dat["x_ref"], dat["y_ref"]],
                    show_metrics=False)
                ax.set_ylabel(f"{row_tag}\n{proj.y_label}", fontsize=7.5)

            elif col_idx == 1:
                draw_transport_panel(ax, proj,
                    dat["x_new"],
                    dat["init_np"] if dat["init_np"] is not None else dat["x_new"],
                    title=col_titles[1] if row_idx == 0 else "",
                    src_label=f"{dat['new_cell']} control", tgt_label="Pretrained",
                    tgt_color=_COLOR_PRED, true_tgt_raw=dat["true_tgt"],
                    true_tgt_label=f"{dat['new_cell']} treated (true)",
                    snaps_raw=dat["init_snaps"], show_metrics=True)
                ax.set_ylabel(proj.y_label, fontsize=8)

            else:
                draw_transport_panel(ax, proj,
                    dat["x_new"],
                    dat["final_np"] if dat["final_np"] is not None else dat["x_new"],
                    title=col_titles[2] if row_idx == 0 else "",
                    src_label=f"{dat['new_cell']} control", tgt_label="DualWCFM",
                    tgt_color=_COLOR_PRED, true_tgt_raw=dat["true_tgt"],
                    true_tgt_label=f"{dat['new_cell']} treated (true)",
                    snaps_raw=dat["final_snaps"], show_metrics=True)
                ax.set_ylabel(proj.y_label, fontsize=8)

    fig.suptitle(
        f"DualWCFM — Cross-Cell-Line Drug Response Transfer ({std['drug']})",
        fontsize=12, fontweight="bold", y=1.01,
    )
    _save(fig, output_stem)
    plt.close(fig)
    print(f"  Saved: {output_stem}.png / .pdf")


# ---------------------------------------------------------------------------
# Convergence figure
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> tuple[list[int], dict[str, list]]:
    if not os.path.isfile(path):
        return [], {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], {}
    iters = [int(r["iteration"]) for r in rows]
    data  = {}
    for key in rows[0]:
        if key == "iteration":
            continue
        vals = []
        for r in rows:
            raw = r.get(key, "")
            try:
                vals.append(float(raw) if raw not in ("", "N/A", "nan") else None)
            except ValueError:
                vals.append(None)
        data[key] = vals
    return iters, data


def build_convergence_figure(
    std_dir:    Optional[str],
    siren_dir:  Optional[str] = None,
    output_stem: str = "singlecell_convergence",
) -> None:
    """6-panel convergence dashboard: SW₂, MMD, Energy, Sinkhorn, ESS, SW₂_source."""
    iters_std,  data_std  = _read_csv(os.path.join(std_dir or "", "metrics.csv"))
    iters_sir,  data_sir  = _read_csv(os.path.join(siren_dir or "", "metrics.csv"))

    if not iters_std and not iters_sir:
        print("  No metrics.csv found; skipping convergence figure.")
        return

    panels = [
        ("W2_nu",        "SW₂ (target marginal)",   "SW₂",  False),
        ("mmd_nu",       "MMD (target marginal)",    "MMD",  True),
        ("energy_dist_nu","Energy distance",         "E-dist",False),
        ("sinkhorn_div", "Sinkhorn divergence",      "Sink.", True),
        ("weight_ess",   "Effective Sample Size",    "ESS",  False),
        ("W2_mu",        "SW₂ (source marginal)",   "SW₂",  False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    axes_flat = axes.ravel()

    def _plot_series(ax, key, iters, data, color, label, logy):
        if key not in data:
            return
        xs = [i for i, v in zip(iters, data[key]) if v is not None]
        ys = [v for v in data[key] if v is not None]
        if not xs:
            return
        ax.plot(xs, ys, color=color, label=label, lw=1.8, zorder=3)
        ax.scatter(xs[::max(1, len(xs)//10)], ys[::max(1, len(ys)//10)],
                   c=color, s=16, zorder=4, linewidths=0)
        if logy:
            try:
                ax.set_yscale("log")
            except Exception:
                pass

    for ax, (key, title, ylabel, logy) in zip(axes_flat, panels):
        if iters_std:
            _plot_series(ax, key, iters_std, data_std, "#4878d0", "MLP", logy)
        if iters_sir:
            _plot_series(ax, key, iters_sir, data_sir, _COLOR_SIREN, "SIREN", logy)

        if key == "weight_ess":
            ax.axhline(1.0, color="#ff8c00", lw=1.0, ls="--", label="ideal (1.0)", zorder=2)
            ax.set_ylim(0, 1.05)

        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("DualWCFM iteration", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.25, linestyle="--", zorder=0)

    fig.suptitle("DualWCFM Training Convergence — Single-Cell Experiment",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, output_stem)
    plt.close(fig)
    print(f"  Saved: {output_stem}.png / .pdf")


# ---------------------------------------------------------------------------
# Checkpoint evolution figure
# ---------------------------------------------------------------------------

def build_evolution_figure(
    spec,
    models_dir:    str,
    device:        torch.device,
    n_particles:   int   = 250,
    n_steps:       int   = 100,
    use_umap:      bool  = False,
    output_stem:   str   = "singlecell_evolution",
    max_panels:    int   = 8,
) -> None:
    """Show improvement of the adapted model at each saved checkpoint."""
    ckpts = find_all_checkpoints(models_dir, "forward")
    # Also include pretrained as iteration -1 and final forward if present
    init_path  = os.path.join(models_dir, "initial_forward_model.pth")
    final_path = os.path.join(models_dir, "forward_model.pth")

    paths_and_labels: list[tuple[str, str]] = []
    if os.path.isfile(init_path):
        paths_and_labels.append((init_path, "Pretrained\n(before)"))
    for ckpt in ckpts:
        stem  = os.path.splitext(os.path.basename(ckpt))[0]
        label = stem.replace("iter_", "Iter ").replace("_forward", "")
        paths_and_labels.append((ckpt, label))
    if os.path.isfile(final_path) and final_path not in [p for p, _ in paths_and_labels]:
        paths_and_labels.append((final_path, "Final"))

    if not paths_and_labels:
        print("  No checkpoints found; skipping evolution figure.")
        return

    if len(paths_and_labels) > max_panels:
        indices = np.round(np.linspace(0, len(paths_and_labels) - 1, max_panels)).astype(int)
        paths_and_labels = [paths_and_labels[i] for i in indices]

    def _np(t):
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    scale = float(getattr(spec.train_cfg, "scale_factor", 20))
    vkw   = getattr(spec, "visualizer_kwargs", {}) or {}
    new_cell     = vkw.get("new_cell_line",  "K562")
    drug         = vkw.get("drug_name",       "Drug")
    new_tgt_dist = vkw.get("new_tgt_dist",   None)

    x_new_np = _np(spec.new_source.sample(n_particles))
    true_tgt_np = (
        _np(new_tgt_dist.sample(min(n_particles, 300)))
        if new_tgt_dist is not None
        else _np(spec.new_target.sample(n_particles))
    )
    dim    = x_new_np.shape[1]
    x_norm = torch.tensor(x_new_np / scale, dtype=torch.float32, device=device)

    all_raw = np.vstack([x_new_np, true_tgt_np])
    proj    = Projector(all_raw, use_umap=use_umap)

    n_cols = min(4, len(paths_and_labels))
    n_rows = (len(paths_and_labels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.8 * n_cols, 4.5 * n_rows))
    axes_flat = np.array(axes).ravel()

    for idx, (path, label) in enumerate(paths_and_labels):
        ax    = axes_flat[idx]
        model = load_model(path, dim, device)
        if model is None:
            ax.set_visible(False)
            continue
        pred_np, snaps = run_model(x_norm, model, n_steps, scale, 20)
        del model
        draw_transport_panel(
            ax, proj, x_new_np, pred_np,
            title=label,
            src_label=f"{new_cell} ctrl", tgt_label="Predicted",
            true_tgt_raw=true_tgt_np,
            true_tgt_label=f"{new_cell} true",
            snaps_raw=snaps, n_traj_lines=50,
            show_metrics=True,
        )

    for ax in axes_flat[len(paths_and_labels):]:
        ax.set_visible(False)

    fig.suptitle(
        f"DualWCFM Checkpoint Evolution — {new_cell}: {drug} response",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, output_stem)
    plt.close(fig)
    print(f"  Saved: {output_stem}.png / .pdf")


# ---------------------------------------------------------------------------
# Coupling arrows figure
# ---------------------------------------------------------------------------

def build_coupling_arrows_figure(
    spec,
    models_dir:  str,
    device:      torch.device,
    n_arrows:    int   = 200,
    n_steps:     int   = 100,
    use_umap:    bool  = False,
    output_stem: str   = "singlecell_coupling",
) -> None:
    """Arrow plot from source to predicted target coloured by source density."""

    def _np(t):
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    scale = float(getattr(spec.train_cfg, "scale_factor", 20))
    vkw   = getattr(spec, "visualizer_kwargs", {}) or {}
    new_cell     = vkw.get("new_cell_line",  "K562")
    drug         = vkw.get("drug_name",       "Drug")
    new_tgt_dist = vkw.get("new_tgt_dist",   None)

    x_new_np = _np(spec.new_source.sample(n_arrows))
    true_tgt  = (
        _np(new_tgt_dist.sample(min(n_arrows, 300)))
        if new_tgt_dist is not None
        else _np(spec.new_target.sample(n_arrows))
    )
    dim    = x_new_np.shape[1]
    x_norm = torch.tensor(x_new_np / scale, dtype=torch.float32, device=device)

    all_raw = np.vstack([x_new_np, true_tgt])
    proj    = Projector(all_raw, use_umap=use_umap)

    model_paths = [
        ("Pretrained", os.path.join(models_dir, "initial_forward_model.pth")),
    ]
    final = os.path.join(models_dir, "forward_model.pth")
    if not os.path.isfile(final):
        final = find_latest_checkpoint(models_dir, "forward") or ""
    if final:
        model_paths.append(("DualWCFM", final))

    n_cols = len(model_paths)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]

    src_2d   = proj(x_new_np)
    true_2d  = proj(true_tgt)

    for ax, (label, path) in zip(axes, model_paths):
        model = load_model(path, dim, device)
        if model is None:
            ax.set_visible(False)
            continue
        pred_np, _ = run_model(x_norm, model, n_steps, scale, n_capture=1)
        del model
        pred_2d = proj(pred_np)

        _kde_background(ax, np.vstack([src_2d, pred_2d]),
                        (src_2d[:, 0].min() - 0.6, src_2d[:, 0].max() + 0.6),
                        (src_2d[:, 1].min() - 0.6, src_2d[:, 1].max() + 0.6))

        ax.scatter(true_2d[:, 0], true_2d[:, 1], c=_COLOR_TRUE, s=6,
                   alpha=0.45, linewidths=0, label=f"{new_cell} true", zorder=2)

        for i in range(len(src_2d)):
            ax.annotate(
                "", xy=pred_2d[i], xytext=src_2d[i],
                arrowprops=dict(
                    arrowstyle="-|>", color=_COLOR_PRED,
                    alpha=0.45, lw=0.6, mutation_scale=5,
                ),
                zorder=3,
            )

        ax.scatter(src_2d[:, 0], src_2d[:, 1], c=_COLOR_SRC, s=7,
                   alpha=0.8, linewidths=0, label=f"{new_cell} control", zorder=4)
        ax.scatter(pred_2d[:, 0], pred_2d[:, 1], c=_COLOR_PRED, s=7,
                   alpha=0.8, linewidths=0, label="Predicted", zorder=4)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel(proj.x_label); ax.set_ylabel(proj.y_label)
        ax.legend(fontsize=7, framealpha=0.7)

    fig.suptitle(f"Coupling Arrows — {new_cell}: {drug} ({new_cell})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_stem)
    plt.close(fig)
    print(f"  Saved: {output_stem}.png / .pdf")


# ---------------------------------------------------------------------------
# Metrics bar chart
# ---------------------------------------------------------------------------

def build_metrics_bar_figure(
    spec,
    models_dir:     str,
    device:         torch.device,
    n_particles:    int  = 600,
    n_steps:        int  = 120,
    siren_dir:      Optional[str] = None,
    output_stem:    str  = "singlecell_metrics_bar",
) -> None:
    """Bar chart comparing pretrained vs DualWCFM (and optionally SIREN) on final metrics."""
    from dualwcfm.metrics import sliced_wasserstein, mmd as _mmd, energy_distance, sinkhorn_divergence

    def _np(t):
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    scale = float(getattr(spec.train_cfg, "scale_factor", 20))
    vkw   = getattr(spec, "visualizer_kwargs", {}) or {}
    new_tgt_dist = vkw.get("new_tgt_dist", None)

    x_new_np = _np(spec.new_source.sample(n_particles))
    true_tgt_np = (
        _np(new_tgt_dist.sample(min(n_particles, n_particles)))
        if new_tgt_dist is not None
        else _np(spec.new_target.sample(n_particles))
    )
    dim    = x_new_np.shape[1]
    x_norm = torch.tensor(x_new_np / scale, dtype=torch.float32, device=device)
    y_true = torch.tensor(true_tgt_np).float()

    def _eval_path(path, label):
        m = load_model(path, dim, device)
        if m is None:
            return None
        with torch.no_grad():
            pred_np, _ = run_model(x_norm, m, n_steps, scale, n_capture=1)
        del m
        y_pred = torch.tensor(pred_np).float()
        return {
            "label":  label,
            "SW₂":    sliced_wasserstein(y_pred, y_true, n_proj=300, p=2),
            "MMD":    _mmd(y_pred, y_true, max_samples=400),
            "E-dist": energy_distance(y_pred, y_true, max_samples=400),
            "Sink.":  sinkhorn_divergence(y_pred, y_true, max_samples=200),
        }

    results = []
    p_init  = os.path.join(models_dir, "initial_forward_model.pth")
    p_final = os.path.join(models_dir, "forward_model.pth")
    if not os.path.isfile(p_final):
        p_final = find_latest_checkpoint(models_dir, "forward") or ""

    if os.path.isfile(p_init):
        r = _eval_path(p_init, "Pretrained")
        if r:
            results.append(r)
    if p_final and os.path.isfile(p_final):
        r = _eval_path(p_final, "DualWCFM (MLP)")
        if r:
            results.append(r)
    if siren_dir:
        sp = os.path.join(siren_dir, "forward_model.pth")
        if not os.path.isfile(sp):
            sp = find_latest_checkpoint(siren_dir, "forward") or ""
        if sp and os.path.isfile(sp):
            r = _eval_path(sp, "DualWCFM (SIREN)")
            if r:
                results.append(r)

    if not results:
        print("  No models to evaluate; skipping metrics bar figure.")
        return

    metric_keys = ["SW₂", "MMD", "E-dist", "Sink."]
    n_metrics   = len(metric_keys)
    n_models    = len(results)
    bar_colors  = [_COLOR_PRED, "#4878d0", _COLOR_SIREN, "#ee854a"][:n_models]

    x    = np.arange(n_metrics)
    wid  = 0.75 / n_models
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for j, (res, col) in enumerate(zip(results, bar_colors)):
        vals   = [res[k] for k in metric_keys]
        offset = (j - (n_models - 1) / 2) * wid
        bars   = ax.bar(x + offset, vals, width=wid * 0.9,
                        color=col, label=res["label"], alpha=0.85, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, color="#333333")

    ax.set_xticks(x); ax.set_xticklabels(metric_keys, fontsize=9)
    ax.set_ylabel("Metric value (lower = better)", fontsize=8.5)
    ax.set_title("DualWCFM — Final metric comparison", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)
    plt.tight_layout()
    _save(fig, output_stem)
    plt.close(fig)
    print(f"  Saved: {output_stem}.png / .pdf")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, stem: str) -> None:
    for ext in (".png", ".pdf"):
        path = stem + ext
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fig.savefig(path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data = p.add_argument_group("data")
    data.add_argument("--data-path", default=None, dest="data_path", metavar="PATH",
                      help="Path to SrivatsanTrapnell2020_sciplex3.h5ad")

    dirs = p.add_argument_group("experiment directories")
    dirs.add_argument("--std-dir",   default="results_cell/singlecell",       dest="std_dir")
    dirs.add_argument("--siren-dir", default="results_cell_siren/singlecell_siren", dest="siren_dir")
    dirs.add_argument("--pert-dir",  default=None, dest="pert_dir",
                      help="Optional: separate dir for perturbed variant (MCF7).")
    dirs.add_argument("--output-dir", default="figures/singlecell", dest="output_dir")

    figs = p.add_argument_group("figures to generate")
    figs.add_argument("--figure", choices=[
        "main_transport", "perturbed", "dual_transport",
        "convergence", "evolution", "coupling", "metrics_bar", "all",
    ], default="all", dest="figure")

    viz = p.add_argument_group("visualisation")
    viz.add_argument("--n-particles",   type=int,   default=400,  dest="n_particles")
    viz.add_argument("--n-steps",       type=int,   default=120,  dest="n_steps")
    viz.add_argument("--n-traj-frames", type=int,   default=30,   dest="n_traj_frames")
    viz.add_argument("--n-traj-lines",  type=int,   default=80,   dest="n_traj_lines")
    viz.add_argument("--dot-size",      type=int,   default=8,    dest="dot_size")
    viz.add_argument("--dot-alpha",     type=float, default=0.65, dest="dot_alpha")
    viz.add_argument("--pad",           type=float, default=0.6)
    viz.add_argument("--umap",          action="store_true",
                     help="Use UMAP projection (requires umap-learn).")
    viz.add_argument("--separate",      action="store_true",
                     help="Save each panel as a separate file.")
    return p


def main() -> None:
    _apply_style()
    args   = _build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    do = args.figure
    do_transport  = do in ("main_transport", "all")
    do_perturbed  = do in ("perturbed", "all")
    do_dual       = do in ("dual_transport", "all")
    do_conv       = do in ("convergence", "all")
    do_evolution  = do in ("evolution", "all")
    do_coupling   = do in ("coupling", "all")
    do_bars       = do in ("metrics_bar", "all")

    std_dir   = args.std_dir
    siren_dir = args.siren_dir if os.path.isdir(args.siren_dir or "") else None

    # Load experiment specs (requires data)
    needs_spec = do_transport or do_perturbed or do_dual or do_evolution or do_coupling or do_bars
    spec_std = spec_pert = None

    if needs_spec:
        print("Loading Sci-Plex3 experiment specs …")
        from dualwcfm.experiments.exp_singlecell import make_singlecell_experiments
        specs = make_singlecell_experiments(
            data_path   = args.data_path,
            output_root = std_dir or "results_cell",
        )
        for s in specs:
            if "perturbed" in s.name:
                spec_pert = s
            else:
                spec_std  = s
        print(f"  Standard: {spec_std.name if spec_std else 'N/A'}")
        print(f"  Perturbed: {spec_pert.name if spec_pert else 'N/A'}")

    models_std   = os.path.join(std_dir or "", "models")
    pert_dir_arg = args.pert_dir
    # Perturbed model: fallback to std_dir (same experiment root, different spec)
    models_pert  = (os.path.join(pert_dir_arg, "models")
                    if pert_dir_arg else models_std)
    models_siren = os.path.join(siren_dir or "", "models") if siren_dir else None

    kw = dict(device=device, n_particles=args.n_particles,
              n_steps=args.n_steps, use_umap=args.umap)

    # ── Main transport figure ─────────────────────────────────────────────────
    if do_transport and spec_std and os.path.isdir(models_std):
        print("\n[1/7] Main transport figure …")
        build_main_transport_figure(
            spec_std, models_std, **kw,
            n_traj_frames = args.n_traj_frames,
            n_traj_lines  = args.n_traj_lines,
            pad           = args.pad,
            dot_size      = args.dot_size,
            dot_alpha     = args.dot_alpha,
            output_stem   = os.path.join(out, "singlecell_main_transport"),
            separate      = args.separate,
        )

    # ── Perturbed transport figure ────────────────────────────────────────────
    if do_perturbed and spec_pert and os.path.isdir(models_pert):
        print("\n[2/7] Perturbed transport figure (MCF7) …")
        build_main_transport_figure(
            spec_pert, models_pert, **kw,
            n_traj_frames = args.n_traj_frames,
            n_traj_lines  = args.n_traj_lines,
            pad           = args.pad,
            dot_size      = args.dot_size,
            dot_alpha     = args.dot_alpha,
            output_stem   = os.path.join(out, "singlecell_perturbed_transport"),
            separate      = args.separate,
        )

    # ── Dual (2×3) combined figure ────────────────────────────────────────────
    if do_dual and spec_std and spec_pert and os.path.isdir(models_std):
        print("\n[3/7] Dual 2×3 combined figure …")
        build_dual_transport_figure(
            spec_std,  models_std,
            spec_pert, models_pert,
            **kw,
            n_traj_frames = args.n_traj_frames,
            output_stem   = os.path.join(out, "singlecell_dual_transport"),
        )

    # ── Convergence ───────────────────────────────────────────────────────────
    if do_conv:
        print("\n[4/7] Convergence figure …")
        build_convergence_figure(
            std_dir   = std_dir,
            siren_dir = siren_dir,
            output_stem = os.path.join(out, "singlecell_convergence"),
        )

    # ── Checkpoint evolution ──────────────────────────────────────────────────
    if do_evolution and spec_std and os.path.isdir(models_std):
        print("\n[5/7] Checkpoint evolution figure …")
        build_evolution_figure(
            spec_std, models_std, device,
            n_particles = args.n_particles,
            n_steps     = args.n_steps,
            use_umap    = args.umap,
            output_stem = os.path.join(out, "singlecell_evolution"),
        )

    # ── Coupling arrows ───────────────────────────────────────────────────────
    if do_coupling and spec_std and os.path.isdir(models_std):
        print("\n[6/7] Coupling arrows figure …")
        build_coupling_arrows_figure(
            spec_std, models_std, device,
            n_arrows    = min(args.n_particles, 200),
            n_steps     = args.n_steps,
            use_umap    = args.umap,
            output_stem = os.path.join(out, "singlecell_coupling_arrows"),
        )

    # ── Metrics bar chart ─────────────────────────────────────────────────────
    if do_bars and spec_std and os.path.isdir(models_std):
        print("\n[7/7] Metrics bar chart …")
        build_metrics_bar_figure(
            spec_std, models_std, device,
            n_particles = args.n_particles,
            n_steps     = args.n_steps,
            siren_dir   = models_siren,
            output_stem = os.path.join(out, "singlecell_metrics_bar"),
        )

    print(f"\nAll done.  Figures saved to: {out}/")


if __name__ == "__main__":
    main()
