"""
make_2d_coupling_evolution_figure.py
-------------------------------------
Paper-quality figure showing the evolution of the current coupling
across training iterations for a 2-D DualWCFM experiment.

Generates a 1 × N grid where each panel shows the state of the coupling
at a given checkpoint iteration.  No quiver arrows.  Each panel contains:
  • KDE density heatmap (source + transported combined)
  • Source scatter (blue)
  • Transported scatter (red) — the marginal pushed through the forward model
  • Coupling lines — thin lines connecting each source point to its
    transported counterpart (a random subset for legibility)
  • New-target scatter (green, small, faint) for reference

Style matches make_2d_transport_figure.py: light background, viridis KDE,
blue/red/green palette.

Usage
~~~~~
  # Show every 10th checkpoint across 100 iterations (11 panels):
  python make_2d_coupling_evolution_figure.py \\
      --mode 2d --level complex --perturbed \\
      --exp-dir "E:/Antoine/IEOT/results_2d_perturbed/2d_complex_perturbed" \\
      --output "E:/Antoine/IEOT/results_2d_perturbed/2d_complex_perturbed/coupling_evolution" \\
      --n-steps-model 100 --n-particles 400 --step 10

  # Show a sparser selection (every 20 iterations):
  python make_2d_coupling_evolution_figure.py \\
      --mode 2d --level complex --perturbed \\
      --exp-dir "E:/Antoine/IEOT/results_2d_perturbed/2d_complex_perturbed" \\
      --step 20 --n-panels 6

  # Moon experiment:
  python make_2d_coupling_evolution_figure.py \\
      --mode 2d --level moon --perturbed \\
      --exp-dir "E:/Antoine/IEOT/results_2d_perturbed/2d_moon_perturbed" \\
      --step 10 --n-particles 500
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":   "dejavuserif",
    "axes.titlesize":     10,
    "axes.labelsize":     8,
    "legend.fontsize":    7,
})
import numpy as np
import torch

_CMAP_WB = LinearSegmentedColormap.from_list(
    "white_blue",
    [(1.00, 1.00, 1.00),
     (0.88, 0.93, 0.98),
     (0.55, 0.75, 0.90),
     (0.22, 0.52, 0.78),
     (0.07, 0.28, 0.58)],
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(_SCRIPT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dualwcfm.mlp import MLP, torch_wrapper


# ---------------------------------------------------------------------------
# Palette (matches make_2d_transport_figure.py)
# ---------------------------------------------------------------------------

_C_SRC       = "#2C6FAC"   # deep blue     – source μ'
_C_SRC_EDGE  = "#1a4d7a"
_C_TRN       = "#C14D6B"   # rose          – transported
_C_TRN_EDGE  = "#8f2d45"
_C_TGT       = "#D4A835"   # warm gold     – new target ν' (reference)
_C_LINE      = "#b8c8d8"   # blue-grey coupling lines
_C_TITLE     = "#1a1e2a"


# ---------------------------------------------------------------------------
# Model helpers (identical to make_2d_transport_figure.py)
# ---------------------------------------------------------------------------

def _is_siren(state: dict) -> bool:
    return any(k.startswith("model.net.") and ".linear.weight" in k for k in state)


def _infer_arch_from_state(state: dict) -> tuple[int, int]:
    """Infer (hidden_size, n_layers) from a torch_wrapper(MLP or SirenMLP) state dict."""
    if _is_siren(state):
        siren_indices = [
            int(k.split(".")[2])
            for k in state
            if k.startswith("model.net.") and k.endswith(".linear.weight")
        ]
        n_layers = max(siren_indices) + 1
        hidden   = state["model.net.0.linear.weight"].shape[0]
        return hidden, n_layers
    linear_indices = [
        int(k.split(".")[2])
        for k in state
        if k.startswith("model.net.") and k.endswith(".weight")
    ]
    if not linear_indices:
        return 64, 2
    max_idx  = max(linear_indices)
    n_layers = max_idx // 2
    hidden   = state["model.net.0.weight"].shape[0]
    return hidden, n_layers


def _build_model(dim: int, hidden: int, n_layers: int, device: torch.device, siren: bool = False):
    if siren:
        from dualwcfm.models.siren import SirenMLP
        mlp = SirenMLP(dim=2 * dim, out_dim=2 * dim, w=hidden, n_layers=n_layers, time_varying=True)
    else:
        mlp = MLP(dim=2 * dim, out_dim=2 * dim, w=hidden, n_layers=n_layers, time_varying=True)
    model = torch_wrapper(mlp).to(device)
    return model


def _load_model(path: str, dim: int, hidden: int, n_layers: int, device: torch.device):
    state = torch.load(path, map_location=device, weights_only=True)
    siren = _is_siren(state)
    ckpt_hidden, ckpt_layers = _infer_arch_from_state(state)
    if (ckpt_hidden, ckpt_layers) != (hidden, n_layers):
        print(
            f"  [arch] checkpoint: hidden={ckpt_hidden} n_layers={ckpt_layers} "
            f"(overrides CLI: hidden={hidden} n_layers={n_layers})"
        )
        hidden, n_layers = ckpt_hidden, ckpt_layers
    if siren:
        print(f"  [SIREN] sinusoidal model detected  (hidden={hidden}, n_layers={n_layers})")
    model = _build_model(dim, hidden, n_layers, device, siren=siren)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Euler transport (same as make_2d_transport_figure.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _transport(
    x_norm: torch.Tensor,
    model:  torch.nn.Module,
    n_steps: int,
) -> np.ndarray:
    """Transport x_norm through model and return final positions (unnormalized)."""
    N, d = x_norm.shape
    dt = 1.0 / n_steps
    xt = torch.cat([x_norm, x_norm], dim=1)
    device = x_norm.device
    for i in range(n_steps):
        t  = torch.full((N, 1), i * dt, device=device)
        vt = model(t, xt)
        xt = torch.cat([xt[:, :d] + vt[:, :d] * dt, xt[:, d:]], dim=1)
    return xt[:, :d].cpu().numpy()


# ---------------------------------------------------------------------------
# KDE helper (identical to make_2d_transport_figure.py)
# ---------------------------------------------------------------------------

def _kde(points: np.ndarray, extent: tuple, res: int = 250, bw="scott") -> np.ndarray:
    from scipy.stats import gaussian_kde
    xmin, xmax, ymin, ymax = extent
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, res),
                          np.linspace(ymin, ymax, res))
    kde = gaussian_kde(points.T, bw_method=bw)
    z   = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(res, res)
    return z


# ---------------------------------------------------------------------------
# Single panel renderer
# ---------------------------------------------------------------------------

def _panel(
    ax:           plt.Axes,
    x_src:        np.ndarray,          # (N, 2)  source in data space
    x_trn:        np.ndarray,          # (N, 2)  transported
    x_tgt:        np.ndarray,          # (N, 2)  new target ν'
    title:        str,
    pad:          float = 1.2,
    n_lines:      int   = 80,
    dot_size:     int   = 7,
    dot_alpha:    float = 0.55,
    line_alpha:   float = 0.25,
    line_width:   float = 0.7,
    kde_res:      int   = 250,
    show_target:  bool  = True,
    show_legend:  bool  = False,
):
    """Render one coupling snapshot on *ax*."""

    # ── Axis limits ─────────────────────────────────────────────────────────
    all_pts = np.vstack([x_src, x_trn, x_tgt])
    xmin = all_pts[:, 0].min() - pad
    xmax = all_pts[:, 0].max() + pad
    ymin = all_pts[:, 1].min() - pad
    ymax = all_pts[:, 1].max() + pad
    extent = (xmin, xmax, ymin, ymax)

    ax.set_facecolor("white")

    # ── Axis crosshairs through origin ───────────────────────────────────────
    ax.axhline(0, color="#c8d4de", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#c8d4de", linewidth=0.7, zorder=0)

    # ── KDE — density of transported (current model output) ─────────────────
    z_src = _kde(x_src, extent, kde_res)
    z_trn = _kde(x_trn, extent, kde_res)
    z     = z_src + z_trn
    if z.max() > 0:
        z /= z.max()
    ax.imshow(
        z, extent=[xmin, xmax, ymin, ymax], origin="lower",
        cmap=_CMAP_WB, aspect="auto", alpha=0.82, zorder=0,
    )

    # ── Coupling lines — random subset src → transported ────────────────────
    idx = np.random.choice(len(x_src), size=min(n_lines, len(x_src)), replace=False)
    for i in idx:
        ax.plot(
            [x_src[i, 0], x_trn[i, 0]],
            [x_src[i, 1], x_trn[i, 1]],
            color=_C_LINE, alpha=line_alpha, linewidth=line_width,
            solid_capstyle="round", zorder=1,
        )

    # ── New target ν' (faint reference) ─────────────────────────────────────
    if show_target:
        sc_trg = ax.scatter(
            x_tgt[:, 0], x_tgt[:, 1],
            s=dot_size * 0.6, c=_C_TGT, alpha=0.30,
            linewidths=0, zorder=2,
            label=r"Target $\nu'$",
        )

    # ── Transported scatter ──────────────────────────────────────────────────
    sc_trn = ax.scatter(
        x_trn[:, 0], x_trn[:, 1],
        s=dot_size, c=_C_TRN, alpha=dot_alpha,
        linewidths=0.3, edgecolors=_C_TRN_EDGE,
        zorder=4, label="Transported",
    )

    # ── Source scatter ───────────────────────────────────────────────────────
    sc_src = ax.scatter(
        x_src[:, 0], x_src[:, 1],
        s=dot_size, c=_C_SRC, alpha=dot_alpha,
        linewidths=0.3, edgecolors=_C_SRC_EDGE,
        zorder=4, label=r"Source $\mu'$",
    )

    if show_legend:
        handles = [sc_src, sc_trn]
        if show_target:
            handles = [sc_src, sc_trn, sc_trg]
        ax.legend(
            handles=handles,
            loc="upper left", fontsize=7,
            framealpha=0.88, facecolor="white", edgecolor="#d0d8e4",
            labelcolor=_C_TITLE, handletextpad=0.4, borderpad=0.4,
        )

    ax.set_title(title, color=_C_TITLE, fontsize=10, fontweight="bold", pad=5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color("#d0d8e4")
        spine.set_linewidth(0.6)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def build_figure(
    exp_name:      str,
    spec,
    ckpt_dir:      str,
    dim:           int   = 2,
    hidden:        int   = 128,
    n_layers:      int   = 3,
    device:        torch.device = None,
    scale:         float | None = None,
    n_particles:   int   = 300,
    n_steps_model: int   = 100,
    step:          int   = 10,
    n_panels:      int   | None = None,
    n_lines:       int   = 80,
    output_stem:   str   = "coupling_evolution",
    dot_size:      int   = 7,
    dot_alpha:     float = 0.55,
    pad:           float = 1.2,
) -> None:
    """Build and save the coupling-evolution grid figure.

    Parameters
    ----------
    spec         : ExperimentSpec — used to sample source / target / scale.
    ckpt_dir     : Directory containing ``iter_XXXX_forward.pth`` files.
    step         : Show every *step* checkpoints (e.g. 10 → every 10 iters).
    n_panels     : Maximum number of panels.  If None, determined from ckpts.
    output_stem  : Path stem (without extension) for saved PDF + PNG.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Resolve scale ────────────────────────────────────────────────────────
    if scale is None or scale <= 0:
        sf = getattr(spec.train_cfg, "scale_factor", -1)
        if sf is not None and sf > 0:
            scale = float(sf)
            print(f"Scale from train_cfg.scale_factor = {scale}")
        else:
            with torch.no_grad():
                x0p, x1p = spec.initial_coupling.coupling_sample(500)
                sp = spec.new_source.sample(500)
                tp = spec.new_target.sample(500)
            scale = max(
                float(x0p.abs().max()), float(x1p.abs().max()),
                float(sp.abs().max()),  float(tp.abs().max()),
            )
            print(f"Scale auto-detected = {scale:.4f}")

    # ── Enumerate checkpoints ────────────────────────────────────────────────
    pattern = os.path.join(ckpt_dir, "iter_*_forward.pth")
    ckpt_files = sorted(glob.glob(pattern))
    if not ckpt_files:
        raise FileNotFoundError(
            f"No iter_*_forward.pth files found in: {ckpt_dir}\n"
            f"(pattern: {pattern})"
        )

    # Extract iteration numbers
    ckpt_iters = []
    for f in ckpt_files:
        base = os.path.basename(f)  # iter_0009_forward.pth
        try:
            it = int(base.split("_")[1])
        except (IndexError, ValueError):
            continue
        ckpt_iters.append((it, f))
    ckpt_iters.sort(key=lambda x: x[0])

    # Select every `step`-th file from the sorted list (index-based, not by
    # iteration value), always keeping the first and last checkpoint.
    # e.g. 11 checkpoints [0,9,19,...,99] with step=1 → all 11
    #                                      with step=2 → [0,19,39,59,79,99]
    indices  = list(range(0, len(ckpt_iters), step))
    last_idx = len(ckpt_iters) - 1
    if last_idx not in indices:
        indices.append(last_idx)
    selected = [ckpt_iters[i] for i in sorted(set(indices))]

    if n_panels is not None and len(selected) > n_panels:
        idx = np.round(np.linspace(0, len(selected) - 1, n_panels)).astype(int)
        selected = [selected[i] for i in sorted(set(idx))]

    print(f"Panels ({len(selected)}): iterations {[it for it, _ in selected]}")

    # ── Shared data ──────────────────────────────────────────────────────────
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    np.random.seed(0)
    torch.manual_seed(0)

    # Reference source μ₀ (t=0) and new source μ₁ (t=1).
    # At outer iteration k the algorithm uses the linearly interpolated source
    #   μ_t  with  t = k / N_total,
    # i.e. x_src(t) = (1 - t) * x_ref_src + t * x_new_src.
    x_ref_src, _ = spec.initial_coupling.coupling_sample(n_particles)
    x_ref_src    = _to_np(x_ref_src)                        # μ₀  (t = 0)
    x_new_src    = _to_np(spec.new_source.sample(n_particles))  # μ₁  (t = 1)
    x_tgt_raw    = _to_np(spec.new_target.sample(n_particles))  # ν'  (reference)

    # Total iterations inferred from the last checkpoint index (0-based).
    # ckpt_iters[-1][0] is the iteration number of the last checkpoint (e.g. 99
    # for a 100-iteration run), so N_total = last_iter + 1.
    N_total = ckpt_iters[-1][0] + 1

    # ── Compute transported points for each checkpoint ───────────────────────
    panels_data = []
    for it, ckpt_path in selected:
        t_k = it / N_total                                    # interpolation time ∈ [0, 1)
        # μ_{t_k}: linearly interpolated source particles
        x_src_k   = (1.0 - t_k) * x_ref_src + t_k * x_new_src
        x_src_norm = torch.tensor(x_src_k / scale, dtype=torch.float32, device=device)

        print(f"  iter {it:04d}  t={t_k:.3f}  loading {ckpt_path}")
        model       = _load_model(ckpt_path, dim, hidden, n_layers, device)
        transported = _transport(x_src_norm, model, n_steps_model) * scale
        del model
        panels_data.append((it, t_k, x_src_k, transported))
    print("Transport done.")

    # ── Figure ───────────────────────────────────────────────────────────────
    n = len(panels_data)
    fig_w = max(2.5 * n, 6.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 2.8), dpi=200)
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Coupling Evolution: {exp_name}",
        color=_C_TITLE, fontsize=12, fontweight="bold", y=1.03,
    )

    for ax_i, (ax, (it, t_k, x_src_k, x_trn)) in enumerate(zip(axes, panels_data)):
        if it == ckpt_iters[0][0]:
            panel_title = "Initial"
        elif it == ckpt_iters[-1][0]:
            panel_title = "Final"
        else:
            panel_title = f"Iteration {it + 1}"

        _panel(
            ax, x_src_k, x_trn, x_tgt_raw,
            title=panel_title,
            pad=pad, n_lines=n_lines,
            dot_size=dot_size, dot_alpha=dot_alpha,
            show_target=True,
            show_legend=(ax_i == 0),
        )

    plt.tight_layout(pad=0.3, w_pad=0.2)

    for ext in (".pdf", ".png"):
        out = output_stem + ext
        dpi = 200 if ext == ".png" else None
        kw = {"dpi": dpi} if dpi else {}
        fig.savefig(out, bbox_inches="tight", facecolor="white", **kw)
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
    exp.add_argument("--mode", choices=["2d", "3d", "gushchin"],
                     default="2d")
    exp.add_argument("--level", default=None,
                     help="Name filter: simple / medium / complex / moon.")
    exp.add_argument("--perturbed", action="store_true")
    exp.add_argument("--results-root", default="results", dest="results_root")

    paths = p.add_argument_group("paths")
    paths.add_argument("--exp-dir", default=None, dest="exp_dir",
                       help="Experiment directory (contains models/checkpoints/models/).")
    paths.add_argument("--ckpt-dir", default=None, dest="ckpt_dir",
                       help="Checkpoint directory (default: <exp-dir>/models/checkpoints/models).")
    paths.add_argument("--output", default=None,
                       help="Output stem (no extension).  Default: <exp-dir>/coupling_evolution.")

    arch = p.add_argument_group("model architecture")
    arch.add_argument("--hidden",   type=int, default=None)
    arch.add_argument("--n-layers", type=int, default=None, dest="n_layers")
    arch.add_argument("--dim",      type=int, default=2)

    viz = p.add_argument_group("visualisation")
    viz.add_argument("--n-particles",   type=int,   default=400,  dest="n_particles")
    viz.add_argument("--n-steps-model", type=int,   default=100,  dest="n_steps_model",
                     help="Euler steps for the ODE integration (default: 100).")
    viz.add_argument("--step",          type=int,   default=10,
                     help="Show every STEP-th checkpoint iteration (default: 10).")
    viz.add_argument("--n-panels",      type=int,   default=None, dest="n_panels",
                     help="Max number of panels to show (sub-sample if needed).")
    viz.add_argument("--n-lines",       type=int,   default=80,   dest="n_lines",
                     help="Number of coupling lines per panel (default: 80).")
    viz.add_argument("--dot-size",      type=int,   default=7,    dest="dot_size")
    viz.add_argument("--dot-alpha",     type=float, default=0.55, dest="dot_alpha")
    viz.add_argument("--pad",           type=float, default=1.2)
    viz.add_argument("--scale",         type=float, default=None)

    return p


def _infer_arch(level: str | None) -> tuple[int, int]:
    level = (level or "").lower()
    if "complex" in level or "moon" in level:
        return 128, 3
    if "medium" in level:
        return 64, 2
    return 32, 2


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Resolve paths ────────────────────────────────────────────────────────
    if args.exp_dir is None:
        parser.error("--exp-dir is required.")
    if not os.path.isdir(args.exp_dir):
        parser.error(f"exp_dir not found: {args.exp_dir}")

    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        ckpt_dir = os.path.join(args.exp_dir, "models", "checkpoints", "models")
    if not os.path.isdir(ckpt_dir):
        parser.error(f"Checkpoint directory not found: {ckpt_dir}")

    output_stem = args.output
    if output_stem is None:
        output_stem = os.path.join(args.exp_dir, "coupling_evolution")

    # ── Load experiment spec ─────────────────────────────────────────────────
    from dualwcfm.experiments.exp_2d       import make_2d_experiments
    from dualwcfm.experiments.exp_3d       import make_3d_experiments
    from dualwcfm.experiments.exp_gushchin import make_gushchin_experiments

    factories = {
        "2d":       make_2d_experiments,
        "3d":       make_3d_experiments,
        "gushchin": make_gushchin_experiments,
    }
    catalogue = factories[args.mode](args.results_root)

    matching = catalogue
    if args.level:
        matching = [s for s in matching if args.level in s.name]
    if args.perturbed:
        matching = [s for s in matching if "perturbed" in s.name]
    else:
        matching = [s for s in matching if "perturbed" not in s.name]

    if not matching:
        parser.error(
            f"No experiments found: mode={args.mode!r} level={args.level!r} "
            f"perturbed={args.perturbed}"
        )
    spec = matching[0]
    print(f"Experiment: {spec.name}")

    # ── Architecture ─────────────────────────────────────────────────────────
    hidden_default, layers_default = _infer_arch(args.level)
    hidden   = args.hidden   if args.hidden   is not None else hidden_default
    n_layers = args.n_layers if args.n_layers is not None else layers_default
    dim      = args.dim
    print(f"Architecture: dim={dim}  hidden={hidden}  n_layers={n_layers}")

    # ── Build figure ─────────────────────────────────────────────────────────
    build_figure(
        exp_name      = spec.name,
        spec          = spec,
        ckpt_dir      = ckpt_dir,
        dim           = dim,
        hidden        = hidden,
        n_layers      = n_layers,
        device        = device,
        scale         = args.scale,
        n_particles   = args.n_particles,
        n_steps_model = args.n_steps_model,
        step          = args.step,
        n_panels      = args.n_panels,
        n_lines       = args.n_lines,
        output_stem   = output_stem,
        dot_size      = args.dot_size,
        dot_alpha     = args.dot_alpha,
        pad           = args.pad,
    )


if __name__ == "__main__":
    main()
