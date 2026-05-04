"""
make_singlecell_figure.py
--------------------------
Paper-quality transport visualisation for the single-cell DualWCFM experiment
using real Sci-Plex3 data (Srivatsan et al., Science 2020).

Three-panel figure in PCA space:
  Panel 1 — Reference coupling π  (A549: control → Givinostat-treated)
  Panel 2 — Pretrained model on new cell line  (before adaptation)
  Panel 3 — Adapted model on new cell line     (after DualWCFM)

Usage
~~~~~
  python make_singlecell_figure.py \\
      --exp-dir results/singlecell \\
      --data-path /path/to/SrivatsanTrapnell2020_sciplex3.h5ad

  # Perturbed variant (MCF7, lower dose)
  python make_singlecell_figure.py \\
      --exp-dir results/singlecell_perturbed --perturbed \\
      --data-path /path/to/SrivatsanTrapnell2020_sciplex3.h5ad

  # One file per panel
  python make_singlecell_figure.py --exp-dir results/singlecell --separate \\
      --data-path /path/to/SrivatsanTrapnell2020_sciplex3.h5ad

  # Also generate metrics convergence dashboard
  python make_singlecell_figure.py --exp-dir results/singlecell --metrics \\
      --data-path /path/to/SrivatsanTrapnell2020_sciplex3.h5ad
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(_SCRIPT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _SCRIPT_DIR)

import paper_theme
paper_theme.setup_style()

from dualwcfm.mlp import MLP, torch_wrapper


# ---------------------------------------------------------------------------
# Colour scheme — paper palette
# ---------------------------------------------------------------------------

_COLOR_SRC  = paper_theme.TEAL    # control / source
_COLOR_TGT  = paper_theme.PINK    # true treated / target
_COLOR_PRED = paper_theme.PURPLE  # model prediction
_COLOR_TRUE = paper_theme.BLUE    # reference true target overlay


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------

def fit_pca(X: np.ndarray, n_components: int = 2):
    """Fit PCA on X (N, d).  Returns (components (n,d), mean (d,))."""
    mean = X.mean(0)
    Xc   = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:n_components], mean


def apply_pca(X: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Project (N, d) → (N, 2)."""
    return (X - mean) @ components.T


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _infer_arch(state: dict) -> tuple[int, int, bool]:
    """Infer (hidden_size, n_layers, is_siren) from a torch_wrapper state dict."""
    # SIREN models have keys like "model.net.0.linear.weight" and "model.final.weight"
    is_siren = any("model.final.weight" in k or ".linear.weight" in k for k in state)
    if is_siren:
        hidden = state["model.net.0.linear.weight"].shape[0]
        n_siren = sum(1 for k in state if k.startswith("model.net.") and k.endswith(".linear.weight"))
        return hidden, n_siren, True
    # MLP models: "model.net.0.weight", "model.net.2.weight", ... (every other index)
    linear_idx = []
    for key in state:
        parts = key.split(".")
        if (len(parts) == 4 and parts[0] == "model" and
                parts[1] == "net" and parts[3] == "weight"):
            linear_idx.append(int(parts[2]))
    if not linear_idx:
        return 256, 4, False
    n_layers = max(linear_idx) // 2
    hidden   = state["model.net.0.weight"].shape[0]
    return hidden, n_layers, False


def load_model(path: str, dim: int, device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location=device, weights_only=True)
    hidden, n_layers, is_siren = _infer_arch(state)
    if is_siren:
        from dualwcfm.models.siren import SirenMLP
        net   = SirenMLP(dim=2 * dim, out_dim=2 * dim, w=hidden,
                         n_layers=n_layers, time_varying=True)
    else:
        net   = MLP(dim=2 * dim, out_dim=2 * dim, w=hidden,
                    n_layers=n_layers, time_varying=True)
    model = torch_wrapper(net).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Euler integration
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_trajectory(
    x0_norm:   torch.Tensor,
    model:     torch.nn.Module,
    n_steps:   int,
    scale:     float,
    n_capture: int = 30,
) -> list[np.ndarray]:
    """Euler-integrate the forward model; return n_capture+1 snapshots in data space."""
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


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------

def _detect_scale(spec, explicit: float | None) -> float:
    if explicit is not None and explicit > 0:
        return float(explicit)
    sf = getattr(spec.train_cfg, "scale_factor", -1)
    if sf is not None and sf > 0:
        return float(sf)
    probe = 500
    with torch.no_grad():
        x0p, x1p = spec.initial_coupling.coupling_sample(probe)
        sp = spec.new_source.sample(probe)
        tp = spec.new_target.sample(probe)
    return float(max(
        x0p.abs().max(), x1p.abs().max(),
        sp.abs().max(),  tp.abs().max(),
    ))


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def _draw_panel(
    ax,
    src_2d:       np.ndarray,
    tgt_2d:       np.ndarray,
    trajs_2d:     list[np.ndarray] | None,
    title:        str,
    src_label:    str = "Control",
    tgt_label:    str = "Predicted",
    true_tgt_2d:  np.ndarray | None = None,
    true_tgt_label: str = r"True $\nu'$",
    pad:          float = 0.5,
    dot_size:     int   = 7,
    dot_alpha:    float = 0.60,
    line_alpha:   float = 0.12,
    n_lines:      int   = 60,
) -> None:
    all_pts = np.vstack(
        [src_2d, tgt_2d] + ([true_tgt_2d] if true_tgt_2d is not None else [])
    )
    xmin = all_pts[:, 0].min() - pad;  xmax = all_pts[:, 0].max() + pad
    ymin = all_pts[:, 1].min() - pad;  ymax = all_pts[:, 1].max() + pad

    # KDE background
    try:
        from scipy.stats import gaussian_kde
        res  = 220
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, res),
                             np.linspace(ymin, ymax, res))
        grid    = np.vstack([xx.ravel(), yy.ravel()])
        pts_kde = np.vstack([src_2d, tgt_2d])
        kde     = gaussian_kde(pts_kde.T, bw_method="scott")
        z       = kde(grid).reshape(res, res)
        if z.max() > 0:
            z /= z.max()
        ax.imshow(z, extent=[xmin, xmax, ymin, ymax], origin="lower",
                  cmap=paper_theme.CMAP_CELL, aspect="auto", alpha=0.55, zorder=0)
    except Exception:
        pass

    # Trajectory lines
    if trajs_2d is not None:
        N    = len(src_2d)
        step = max(1, N // n_lines)
        for i in range(0, N, step):
            xs = [snap[i, 0] for snap in trajs_2d if i < len(snap)]
            ys = [snap[i, 1] for snap in trajs_2d if i < len(snap)]
            ax.plot(xs, ys, color=paper_theme.LBLUE, alpha=line_alpha, lw=0.5,
                    solid_capstyle="round", zorder=1)

    # True target overlay
    if true_tgt_2d is not None:
        ax.scatter(true_tgt_2d[:, 0], true_tgt_2d[:, 1],
                   s=dot_size * 0.8, c=_COLOR_TGT, alpha=0.40,
                   linewidths=0, zorder=2, label=true_tgt_label)

    tgt_c = _COLOR_PRED if "True" not in tgt_label else _COLOR_TGT
    kw = dict(linewidths=0, zorder=3, alpha=dot_alpha, s=dot_size)
    ax.scatter(src_2d[:, 0], src_2d[:, 1], c=_COLOR_SRC,
               label=src_label, marker="o", **kw)
    ax.scatter(tgt_2d[:, 0], tgt_2d[:, 1], c=tgt_c,
               label=tgt_label, marker="^", **kw)

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", **paper_theme.legend_kw(fontsize=7))
    paper_theme.style_axes(ax)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def build_figure(
    spec,
    initial_model_path: str,
    final_model_path:   str,
    device:             torch.device,
    scale:              float | None = None,
    n_particles:        int   = 300,
    n_steps:            int   = 100,
    n_traj_frames:      int   = 30,
    output_stem:        str   = "singlecell_transport_figure",
    separate:           bool  = False,
    pad:                float = 0.5,
    dot_size:           int   = 7,
    dot_alpha:          float = 0.60,
) -> None:
    """Generate and save the three-panel single-cell transport figure."""
    def _np(t) -> np.ndarray:
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    sc = _detect_scale(spec, scale)
    print(f"Normalisation scale: {sc:.4f}")

    # Metadata from visualizer_kwargs
    vkw          = getattr(spec, "visualizer_kwargs", {}) or {}
    ref_cell     = vkw.get("ref_cell_line",  "A549")
    new_cell     = vkw.get("new_cell_line",  "K562")
    drug_name    = vkw.get("drug_name",      "Drug")
    new_tgt_dist = vkw.get("new_tgt_dist",   None)

    # ── 1. Reference coupling samples ────────────────────────────────────────
    print("Sampling reference coupling …")
    x_ref_t, y_ref_t = spec.initial_coupling.coupling_sample(n_particles)
    x_ref_np = _np(x_ref_t)
    y_ref_np = _np(y_ref_t)

    # ── 2. New source + true target ───────────────────────────────────────────
    print("Sampling new source / true target …")
    x_new_np = _np(spec.new_source.sample(n_particles))

    true_tgt_np: np.ndarray | None = None
    if new_tgt_dist is not None:
        try:
            true_tgt_np = _np(new_tgt_dist.sample(min(n_particles, 300)))
        except Exception:
            pass
    if true_tgt_np is None:
        true_tgt_np = _np(spec.new_target.sample(n_particles))

    # ── Fit shared PCA on all data ────────────────────────────────────────────
    print("Fitting PCA …")
    all_raw = np.vstack([x_ref_np, y_ref_np, x_new_np, true_tgt_np])
    pca_comps, pca_mean = fit_pca(all_raw, n_components=2)

    def proj(X):
        return apply_pca(X, pca_comps, pca_mean)

    ref_src_2d  = proj(x_ref_np)
    ref_tgt_2d  = proj(y_ref_np)
    new_src_2d  = proj(x_new_np)
    true_tgt_2d = proj(true_tgt_np)

    # ── 3. Pretrained model ───────────────────────────────────────────────────
    dim = x_new_np.shape[1]
    print(f"Loading initial model: {initial_model_path}")
    model_init  = load_model(initial_model_path, dim=dim, device=device)
    x_new_norm  = torch.tensor(x_new_np / sc, dtype=torch.float32, device=device)
    print(f"  Integrating ({n_steps} steps) …")
    init_snaps  = euler_trajectory(x_new_norm, model_init, n_steps, sc, n_traj_frames)
    x_init_np   = init_snaps[-1]
    init_traj_2d = [proj(s) for s in init_snaps]
    del model_init

    # ── 4. Final adapted model ────────────────────────────────────────────────
    print(f"Loading final model: {final_model_path}")
    model_final = load_model(final_model_path, dim=dim, device=device)
    print(f"  Integrating ({n_steps} steps) …")
    final_snaps  = euler_trajectory(x_new_norm, model_final, n_steps, sc, n_traj_frames)
    x_final_np   = final_snaps[-1]
    final_traj_2d = [proj(s) for s in final_snaps]
    del model_final

    # ── Panels ────────────────────────────────────────────────────────────────
    panels = [
        dict(
            src_2d      = ref_src_2d,
            tgt_2d      = ref_tgt_2d,
            trajs       = [ref_src_2d, ref_tgt_2d],
            title       = f"Reference coupling π\n({ref_cell}: control → {drug_name})",
            src_label   = f"{ref_cell} control",
            tgt_label   = f"{ref_cell} treated (true)",
            true_tgt_2d = None,
        ),
        dict(
            src_2d      = new_src_2d,
            tgt_2d      = proj(x_init_np),
            trajs       = init_traj_2d,
            title       = f"Pretrained (before adaptation)\n({new_cell}: control → predicted)",
            src_label   = f"{new_cell} control",
            tgt_label   = "Predicted (pretrained)",
            true_tgt_2d = true_tgt_2d,
            true_tgt_label = f"{new_cell} treated (true)",
        ),
        dict(
            src_2d      = new_src_2d,
            tgt_2d      = proj(x_final_np),
            trajs       = final_traj_2d,
            title       = f"DualWCFM (adapted)\n({new_cell}: control → predicted)",
            src_label   = f"{new_cell} control",
            tgt_label   = "Predicted (adapted)",
            true_tgt_2d = true_tgt_2d,
            true_tgt_label = f"{new_cell} treated (true)",
        ),
    ]

    _panel_kw = dict(pad=pad, dot_size=dot_size, dot_alpha=dot_alpha)
    _stems    = ["reference_coupling", "pretrained", "adapted"]

    if separate:
        for stem, p in zip(_stems, panels):
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            fig.patch.set_facecolor("white")
            _draw_panel(
                ax, p["src_2d"], p["tgt_2d"], p["trajs"], p["title"],
                src_label=p["src_label"], tgt_label=p["tgt_label"],
                true_tgt_2d=p.get("true_tgt_2d"),
                true_tgt_label=p.get("true_tgt_label", r"True $\nu'$"),
                **_panel_kw,
            )
            for ext in (".png", ".pdf"):
                out = f"{output_stem}_{stem}{ext}"
                fig.savefig(out, bbox_inches="tight")
                print(f"Saved: {out}")
            plt.close(fig)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        fig.patch.set_facecolor("white")
        fig.suptitle(
            f"Single-cell drug response transfer ({drug_name}) — TACO",
            color=paper_theme.DARK, fontsize=12, fontweight="bold", y=1.02,
        )
        for ax, p in zip(axes, panels):
            _draw_panel(
                ax, p["src_2d"], p["tgt_2d"], p["trajs"], p["title"],
                src_label=p["src_label"], tgt_label=p["tgt_label"],
                true_tgt_2d=p.get("true_tgt_2d"),
                true_tgt_label=p.get("true_tgt_label", r"True $\nu'$"),
                **_panel_kw,
            )
        plt.tight_layout(pad=0.5)
        for ext in (".png", ".pdf"):
            out = f"{output_stem}{ext}"
            fig.savefig(out, bbox_inches="tight")
            print(f"Saved: {out}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Convergence / metrics dashboard (supplement)
# ---------------------------------------------------------------------------

def build_metrics_figure(exp_dir: str, output_stem: str | None = None) -> None:
    """Read metrics.csv and plot convergence curves (SW₂, MMD, ESS)."""
    import csv

    csv_path = os.path.join(exp_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        print(f"metrics.csv not found at {csv_path}")
        return

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    iters = [int(r["iteration"]) for r in rows]

    def _get(key):
        return [float(r[key]) if r.get(key, "") not in ("", "N/A") else None
                for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training Convergence — {os.path.basename(exp_dir)}", fontsize=12)

    def _plot(ax, series, title, ylabel="", logy=False):
        for key, col, lab in series:
            vals = _get(key)
            xs = [i for i, v in zip(iters, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            if xs:
                ax.plot(xs, ys, color=col, label=lab, lw=1.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        if logy:
            try: ax.set_yscale("log")
            except Exception: pass

    _plot(axes[0, 0],
          [("W2_nu", paper_theme.PINK, "SW₂ target"),
           ("W2_mu", paper_theme.TEAL, "SW₂ source")],
          "Sliced Wasserstein (marginals)", "SW₂")
    _plot(axes[0, 1],
          [("mmd_nu", paper_theme.PURPLE, "MMD target"),
           ("mmd_mu", paper_theme.TEAL, "MMD source")],
          "MMD (distribution quality)", "MMD", logy=True)
    _plot(axes[1, 0],
          [("sinkhorn_div", paper_theme.PINK, "Sinkhorn div")],
          "Sinkhorn divergence", logy=True)

    ess = _get("weight_ess")
    xs_e = [i for i, v in zip(iters, ess) if v is not None]
    ys_e = [v for v in ess if v is not None]
    axes[1, 1].plot(xs_e, ys_e, color="#3cb44b", lw=1.8)
    axes[1, 1].axhline(1.0, color="#ff7c00", lw=1, linestyle="--", label="ideal")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Effective Sample Size", fontsize=10, fontweight="bold")
    axes[1, 1].set_xlabel("Iteration"); axes[1, 1].set_ylabel("ESS (norm.)")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.25, linestyle="--")
    axes[1, 1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    stem = output_stem or os.path.join(exp_dir, "convergence_figure")
    for ext in (".png", ".pdf"):
        out = f"{stem}{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    exp = p.add_argument_group("experiment")
    exp.add_argument("--perturbed", action="store_true",
                     help="Use the perturbed variant (MCF7, lower dose).")
    exp.add_argument("--results-root", default="results", dest="results_root",
                     help="Root results directory (default: results).")
    exp.add_argument(
        "--data-path", default=None, dest="data_path", metavar="PATH",
        help=(
            "Path to SrivatsanTrapnell2020_sciplex3.h5ad. "
            "Required unless pertpy is installed with the dataset cached."
        ),
    )

    paths = p.add_argument_group("paths")
    paths.add_argument("--exp-dir", default=None, dest="exp_dir",
                       help="Experiment output directory (contains models/).")
    paths.add_argument("--models-dir", default=None, dest="models_dir",
                       help="Directory containing model .pth files.")
    paths.add_argument("--output", default=None,
                       help="Output stem (no extension). Defaults to <exp_dir>/transport_figure.")

    viz = p.add_argument_group("visualisation")
    viz.add_argument("--n-particles",   type=int,   default=300,  dest="n_particles")
    viz.add_argument("--n-steps",       type=int,   default=100,  dest="n_steps")
    viz.add_argument("--n-traj-frames", type=int,   default=30,   dest="n_traj_frames")
    viz.add_argument("--dot-size",      type=int,   default=7,    dest="dot_size")
    viz.add_argument("--dot-alpha",     type=float, default=0.60, dest="dot_alpha")
    viz.add_argument("--pad",           type=float, default=0.5)
    viz.add_argument("--scale",         type=float, default=None)
    viz.add_argument("--separate",      action="store_true",
                     help="Write each panel as a separate file.")
    viz.add_argument("--metrics",       action="store_true",
                     help="Also generate the convergence metrics figure.")

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve model paths
    models_dir = args.models_dir
    if models_dir is None:
        if args.exp_dir:
            models_dir = os.path.join(args.exp_dir, "models")
        else:
            parser.error("Provide --exp-dir or --models-dir.")
    if not os.path.isdir(models_dir):
        parser.error(f"models_dir not found: {models_dir}")

    init_path  = os.path.join(models_dir, "initial_forward_model.pth")
    final_path = os.path.join(models_dir, "forward_model.pth")
    for chk in (init_path, final_path):
        if not os.path.isfile(chk):
            parser.error(f"Model file not found: {chk}")

    output_stem = args.output
    if output_stem is None:
        base = args.exp_dir or os.path.dirname(models_dir)
        output_stem = os.path.join(base, "transport_figure")

    # Load experiment spec (this loads Sci-Plex3 data)
    from dualwcfm.experiments.exp_singlecell import make_singlecell_experiments
    catalogue = make_singlecell_experiments(
        data_path   = args.data_path,
        output_root = args.results_root,
    )
    matching = [s for s in catalogue if ("perturbed" in s.name) == args.perturbed]
    if not matching:
        parser.error("No matching experiment spec found.")
    spec = matching[0]
    print(f"Experiment: {spec.name}")

    build_figure(
        spec               = spec,
        initial_model_path = init_path,
        final_model_path   = final_path,
        device             = device,
        scale              = args.scale,
        n_particles        = args.n_particles,
        n_steps            = args.n_steps,
        n_traj_frames      = args.n_traj_frames,
        output_stem        = output_stem,
        separate           = args.separate,
        pad                = args.pad,
        dot_size           = args.dot_size,
        dot_alpha          = args.dot_alpha,
    )

    if args.metrics and args.exp_dir:
        build_metrics_figure(
            exp_dir     = args.exp_dir,
            output_stem = os.path.join(args.exp_dir, "convergence_figure"),
        )


if __name__ == "__main__":
    main()
