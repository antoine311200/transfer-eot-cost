"""
make_2d_transport_figure.py
---------------------------
Paper-quality particle-transport visualisation for 2D DualWCFM experiments.

Generates three figures side-by-side (or individually):
  1. reference_coupling  — samples drawn from the reference coupling π (training data)
  2. pretrained          — initial model (before adaptation) applied to new marginals
  3. adapted             — final adapted model applied to new marginals

Style
~~~~~
  Dark background · KDE heatmap (viridis) · cyan source dots · red target dots
  · semi-transparent white trajectory lines (matching the paper figures).

Usage
~~~~~
  # All three panels into one PNG (default)
  python make_2d_transport_figure.py --mode 2d --level simple \\
      --exp-dir results/2d_simple

  # Override paths explicitly
  python make_2d_transport_figure.py --mode 2d --level simple \\
      --models-dir results/2d_simple/models \\
      --output figs/2d_simple_transport

  # More particles / integration steps
  python make_2d_transport_figure.py --mode 2d --level complex \\
      --exp-dir results/2d_complex --n-particles 400 --n-steps 100 \\
      --hidden 128 --n-layers 3

  # Write each panel as a separate file
  python make_2d_transport_figure.py --mode 2d --level simple \\
      --exp-dir results/2d_simple --separate
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

from dualwcfm.mlp import MLP, torch_wrapper


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _infer_arch_from_state(state: dict) -> tuple[int, int]:
    """Infer (hidden_size, n_layers) from a torch_wrapper(MLP) state dict.

    MLP.net layout (no norm, time_varying=True):
      index 0 : Linear(2*dim+1, w)   ← net.0
      index 1 : SELU
      index 2 : Linear(w, w)         ← net.2  (repeated n_layers-1 times)
      index 3 : SELU
      ...
      index 2*n_layers : Linear(w, out_dim)   ← last linear = output

    So  n_layers = max_linear_index / 2
    and hidden_size = net.0.weight shape[0].
    """
    linear_indices = []
    for key in state:
        # keys: "model.net.N.weight"
        parts = key.split(".")
        if len(parts) == 4 and parts[0] == "model" and parts[1] == "net" and parts[3] == "weight":
            linear_indices.append(int(parts[2]))

    if not linear_indices:
        return 64, 2  # safe fallback

    max_idx   = max(linear_indices)
    n_layers  = max_idx // 2          # e.g. max_idx=4 → n_layers=2; max_idx=6 → n_layers=3
    # hidden = output size of the first linear layer
    hidden    = state["model.net.0.weight"].shape[0]
    return hidden, n_layers


def _build_forward_model(
    dim: int, hidden_size: int, n_layers: int, device: torch.device
) -> torch.nn.Module:
    """Reconstruct the forward model architecture (matches Strategy._create_models)."""
    mlp   = MLP(dim=2 * dim, out_dim=2 * dim, w=hidden_size, n_layers=n_layers, time_varying=True)
    model = torch_wrapper(mlp).to(device)
    return model


def _load_model(path: str, dim: int, hidden_size: int, n_layers: int,
                device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location=device, weights_only=True)
    # Auto-detect architecture from checkpoint; override CLI args if they differ.
    ckpt_hidden, ckpt_layers = _infer_arch_from_state(state)
    if (ckpt_hidden, ckpt_layers) != (hidden_size, n_layers):
        print(
            f"  [arch mismatch] CLI says hidden={hidden_size} n_layers={n_layers}; "
            f"checkpoint has hidden={ckpt_hidden} n_layers={ckpt_layers} — using checkpoint."
        )
        hidden_size, n_layers = ckpt_hidden, ckpt_layers
    model = _build_forward_model(dim, hidden_size, n_layers, device)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Scale detection (mirrors Trainer.__init__ normalization logic)
# ---------------------------------------------------------------------------

def _detect_scale(spec, explicit_scale: float | None) -> float:
    """Return the normalization scale used during training.

    Priority:
      1. *explicit_scale* if provided via CLI.
      2. spec.train_cfg.scale_factor if > 0.
      3. Auto-compute as max absolute value across 1000 probe samples
         (same formula as Trainer.__init__).
    """
    if explicit_scale is not None and explicit_scale > 0:
        return float(explicit_scale)

    sf = getattr(spec.train_cfg, "scale_factor", -1)
    if sf is not None and sf > 0:
        print(f"Scale from train_cfg.scale_factor = {sf}")
        return float(sf)

    # Auto-detect from data (mirrors Trainer logic)
    import torch
    probe_n = 1000
    with torch.no_grad():
        x0p, x1p = spec.initial_coupling.sample(probe_n)
        sp = spec.new_source.sample(probe_n)
        tp = spec.new_target.sample(probe_n)
    scale = max(
        float(x0p.abs().max()), float(x1p.abs().max()),
        float(sp.abs().max()),  float(tp.abs().max()),
    )
    print(f"Scale auto-detected = {scale:.4f}")
    return scale


# ---------------------------------------------------------------------------
# Euler integration
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_trajectory(
    x0_norm: torch.Tensor,   # (N, d) — already normalized by scale
    model:   torch.nn.Module,
    n_steps: int,
    scale:   float,
    n_capture: int = 30,
) -> list[np.ndarray]:
    """Integrate the forward model in normalized space; return snapshots in data space.

    *x0_norm* must be pre-divided by *scale* (the trainer's normalization constant).
    Snapshots are multiplied back by *scale* before returning so the caller works
    entirely in the original data coordinate system.

    Returns a list of (n_capture+1) arrays, each of shape (N, d), at times
    t = 0, 1/n_capture, …, 1, in **data space**.
    """
    N, d = x0_norm.shape
    device = x0_norm.device
    dt = 1.0 / n_steps

    # Extended state: [x_evolved | x0_frozen] — both in normalized space
    xt = torch.cat([x0_norm, x0_norm], dim=1)

    capture_at = {round(i * n_steps / n_capture) for i in range(n_capture + 1)}
    snapshots  = []

    if 0 in capture_at:
        snapshots.append((x0_norm.cpu().numpy() * scale))

    for i in range(n_steps):
        t  = torch.full((N, 1), i * dt, device=device)
        vt = model(t, xt)
        xt = torch.cat([xt[:, :d] + vt[:, :d] * dt, xt[:, d:]], dim=1)
        if (i + 1) in capture_at:
            snapshots.append(xt[:, :d].cpu().numpy() * scale)   # back to data space

    return snapshots   # list of (n_capture+1) arrays in data space, shape (N, d)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _kde_image(
    points: np.ndarray,   # (N, 2)
    extent: tuple,        # (xmin, xmax, ymin, ymax)
    resolution: int = 300,
    bw: float | str = "scott",
) -> np.ndarray:
    """Return a (resolution, resolution) KDE density image for *points*."""
    from scipy.stats import gaussian_kde
    xmin, xmax, ymin, ymax = extent
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, resolution),
        np.linspace(ymin, ymax, resolution),
    )
    grid = np.vstack([xx.ravel(), yy.ravel()])
    kde  = gaussian_kde(points.T, bw_method=bw)
    z    = kde(grid).reshape(resolution, resolution)
    return z


def _panel(
    ax:           plt.Axes,
    x_src:        np.ndarray,              # (N, 2) source particles (blue)
    x_tgt:        np.ndarray,              # (N, 2) transported/target particles (red)
    trajs:        list[np.ndarray] | None, # trajectory snapshots, or None
    title:        str,
    src_label:    str   = r"Source $\mu'$",
    tgt_label:    str   = r"Target $\nu'$",
    x_true_tgt:   np.ndarray | None = None,  # (N, 2) true target ν' overlay (green)
    true_tgt_label: str = r"True target $\nu'$",
    pad:          float = 1.2,
    kde_res:      int   = 300,
    dot_size:     int   = 9,
    dot_alpha:    float = 0.55,
    line_alpha:   float = 0.18,
    line_width:   float = 0.6,
):
    """Draw one transport panel onto *ax*.

    *x_true_tgt*: when provided (e.g. for the pretrained panel), draws the
    true new-target distribution ν' as small semi-transparent green scatter
    points so the reader can see how far the pretrained transport deviates
    from the desired output.
    """

    # ── Axis limits — expand to include true target if present ──────────────
    all_pts = np.vstack([x_src, x_tgt] + ([x_true_tgt] if x_true_tgt is not None else []))
    xmin = all_pts[:, 0].min() - pad
    xmax = all_pts[:, 0].max() + pad
    ymin = all_pts[:, 1].min() - pad
    ymax = all_pts[:, 1].max() + pad
    extent = (xmin, xmax, ymin, ymax)

    # ── KDE background — viridis density of source + transported combined ───
    z_src = _kde_image(x_src, extent, kde_res)
    z_tgt = _kde_image(x_tgt, extent, kde_res)
    z     = z_src + z_tgt
    if z.max() > 0:
        z /= z.max()

    ax.imshow(
        z, extent=[xmin, xmax, ymin, ymax], origin="lower",
        cmap="viridis", aspect="auto", alpha=1.0, zorder=0,
    )

    # ── Trajectory lines ────────────────────────────────────────────────────
    if trajs is not None:
        N = x_src.shape[0]
        for i in range(N):
            xs = [snap[i, 0] for snap in trajs]
            ys = [snap[i, 1] for snap in trajs]
            ax.plot(xs, ys,
                    color="#cccccc", alpha=line_alpha, linewidth=line_width,
                    solid_capstyle="round", zorder=1)

    # ── True target ν' overlay (green, faint) — drawn first so it sits under ─
    legend_handles = []
    if x_true_tgt is not None:
        sc_true = ax.scatter(
            x_true_tgt[:, 0], x_true_tgt[:, 1],
            s=dot_size * 0.9, c="#ffe033", alpha=0.65,
            linewidths=0.3, edgecolors="#ccaa00",
            zorder=2, label=true_tgt_label,
        )
        legend_handles.append(sc_true)

    # ── Scatter: source (blue) and transported/target (red) ─────────────────
    sc_src = ax.scatter(x_src[:, 0], x_src[:, 1],
                        s=dot_size, c="#00cfff", alpha=dot_alpha,
                        linewidths=0.4, edgecolors="#007aaa",
                        zorder=3, label=src_label)
    sc_tgt = ax.scatter(x_tgt[:, 0], x_tgt[:, 1],
                        s=dot_size, c="#ff4422", alpha=dot_alpha,
                        linewidths=0.4, edgecolors="#cc2200",
                        zorder=3, label=tgt_label)
    legend_handles += [sc_src, sc_tgt]

    # ── Legend ───────────────────────────────────────────────────────────────
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left", fontsize=8,
        framealpha=0.5, facecolor="white", edgecolor="#aaaaaa",
        labelcolor="black", handletextpad=0.4, borderpad=0.5,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, color="black", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="black", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#aaaaaa")


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def build_figure(
    exp_name:    str,
    spec,                       # ExperimentSpec — used for distributions + scale detection
    initial_model_path: str,
    final_model_path:   str,
    dim:         int,
    hidden_size: int,
    n_layers:    int,
    device:      torch.device,
    scale:       float | None = None,
    n_particles: int   = 100,
    n_steps:     int   = 100,
    n_traj_frames: int = 30,
    dot_size:    int   = 9,
    dot_alpha:   float = 0.55,
    output_stem: str   = "transport_figure",
    separate:    bool  = False,
    pad:         float = 1.2,
):
    """Build and save the three-panel transport figure.

    Parameters
    ----------
    initial_coupling:
        Reference coupling π — must support `sample(n)` returning a (x, y) tuple.
    new_source / new_target:
        New marginals μ', ν' — each must support `sample(n)` returning a tensor.
    initial_model_path:
        Path to ``initial_forward_model.pth``.
    final_model_path:
        Path to ``forward_model.pth``.
    n_traj_frames:
        Number of trajectory waypoints stored for drawing lines (higher = smoother).
    separate:
        If True, write each panel as a separate PNG instead of one combined figure.
    """

    def _to_np(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # ── Resolve normalization scale ──────────────────────────────────────────
    sc = _detect_scale(spec, scale)
    print(f"Normalization scale: {sc:.4f}  (source at +/-{sc:.1f} -> model sees +/-1)")

    # ── 1. Reference coupling ────────────────────────────────────────────────
    print("Sampling reference coupling ...", flush=True)
    x_ref, y_ref = spec.initial_coupling.coupling_sample(n_particles)
    x_ref_np = _to_np(x_ref)
    y_ref_np  = _to_np(y_ref)
    # Straight lines: no model, just pair source → target
    ref_trajs = [x_ref_np, y_ref_np]

    # ── Shared normalized source + true target for panels 2 & 3 ────────────
    x_new_raw    = _to_np(spec.new_source.sample(n_particles))   # data space
    x_new_np     = x_new_raw
    x_new_norm   = torch.tensor(x_new_raw / sc, dtype=torch.float32, device=device)
    x_new_tgt_np = _to_np(spec.new_target.sample(n_particles))   # true ν' for overlay

    # ── 2. Pretrained model on new source ────────────────────────────────────
    print(f"Loading initial model: {initial_model_path}", flush=True)
    model_init = _load_model(initial_model_path, dim, hidden_size, n_layers, device)
    print(f"  Integrating pretrained model ({n_steps} Euler steps) ...", flush=True)
    init_trajs = euler_trajectory(x_new_norm, model_init, n_steps, sc, n_traj_frames)
    x_init_transported = init_trajs[-1]   # data space
    del model_init

    # ── 3. Final adapted model on new source ─────────────────────────────────
    print(f"Loading final model: {final_model_path}", flush=True)
    model_final = _load_model(final_model_path, dim, hidden_size, n_layers, device)
    print(f"  Integrating final model ({n_steps} Euler steps) ...", flush=True)
    final_trajs = euler_trajectory(x_new_norm, model_final, n_steps, sc, n_traj_frames)
    x_final_transported = final_trajs[-1]   # data space
    del model_final

    # ── Figure ───────────────────────────────────────────────────────────────
    # Each entry: (x_src, x_tgt, trajs, title, src_label, tgt_label, x_true_tgt_overlay)
    panels = [
        (x_ref_np,  y_ref_np,            ref_trajs,
         "Reference Coupling $\\pi$",
         r"Source $\mu$",  r"Target $\nu$",
         None),
        (x_new_np,  x_init_transported,  init_trajs,
         "Pretrained (before adaptation)",
         r"Source $\mu'$", r"Transported (pretrained)",
         x_new_tgt_np),   # ← true ν' shown in green so the gap is visible
        (x_new_np,  x_final_transported, final_trajs,
         "Adapted (final model)",
         r"Source $\mu'$", r"Transported (adapted)",
         None),
    ]

    _panel_kw = dict(pad=pad, dot_size=dot_size, dot_alpha=dot_alpha)

    if separate:
        for stem, (x_s, x_t, trajs, title, slbl, tlbl, x_ov) in zip(
            ["reference_coupling", "pretrained", "adapted"], panels
        ):
            fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
            _panel(ax, x_s, x_t, trajs, title, src_label=slbl, tgt_label=tlbl,
                   x_true_tgt=x_ov, **_panel_kw)
            out = f"{output_stem}_{stem}.png"
            fig.savefig(out, bbox_inches="tight", dpi=200, facecolor=fig.get_facecolor())
            print(f"Saved: {out}")
            out_pdf = f"{output_stem}_{stem}.pdf"
            fig.savefig(out_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"Saved: {out_pdf}")
            plt.close(fig)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)
        fig.suptitle(
            f"Transfer Learning of Data-based EOT — {exp_name}",
            color="black", fontsize=14, fontweight="bold", y=1.02,
        )
        for ax, (x_s, x_t, trajs, title, slbl, tlbl, x_ov) in zip(axes, panels):
            _panel(ax, x_s, x_t, trajs, title, src_label=slbl, tgt_label=tlbl,
                   x_true_tgt=x_ov, **_panel_kw)
        plt.tight_layout(pad=0.5)
        out = f"{output_stem}.png"
        fig.savefig(out, bbox_inches="tight", dpi=200, facecolor=fig.get_facecolor())
        print(f"Saved: {out}")
        # also PDF
        out_pdf = f"{output_stem}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight", dpi=200, facecolor=fig.get_facecolor())
        print(f"Saved: {out_pdf}")
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
    exp.add_argument(
        "--mode", choices=["2d", "3d", "gushchin", "scalability"],
        default="2d", help="Experiment mode (default: 2d).",
    )
    exp.add_argument(
        "--level", default=None,
        help="Name filter: simple / medium / complex (default: first match).",
    )
    exp.add_argument(
        "--perturbed", action="store_true",
        help="Use the perturbed variant.",
    )

    paths = p.add_argument_group("paths")
    paths.add_argument(
        "--exp-dir", default=None, dest="exp_dir",
        help=(
            "Experiment output directory (the folder that contains 'models/'). "
            "Sets --models-dir automatically; explicit --models-dir takes precedence."
        ),
    )
    paths.add_argument(
        "--models-dir", default=None, dest="models_dir",
        help="Directory containing initial_forward_model.pth and forward_model.pth.",
    )
    paths.add_argument(
        "--output", default=None,
        help=(
            "Output file stem (without extension). "
            "Defaults to <exp_dir>/transport_figure."
        ),
    )
    paths.add_argument(
        "--results-root", default="results", dest="results_root",
        help="Root results directory for experiment catalogue lookup (default: results).",
    )

    arch = p.add_argument_group("model architecture")
    arch.add_argument("--hidden",   type=int, default=None,
                      help="MLP hidden size (default: inferred from level).")
    arch.add_argument("--n-layers", type=int, default=None, dest="n_layers",
                      help="MLP layers (default: inferred from level).")
    arch.add_argument("--dim",      type=int, default=2,
                      help="Data dimensionality (default: 2).")

    viz = p.add_argument_group("visualisation")
    viz.add_argument("--n-particles",  type=int,   default=300,  dest="n_particles")
    viz.add_argument("--n-steps",      type=int,   default=100,  dest="n_steps",
                     help="Euler integration steps (default: 100).")
    viz.add_argument("--n-traj-frames",type=int,   default=30,   dest="n_traj_frames",
                     help="Trajectory waypoints for line drawing (default: 30).")
    viz.add_argument("--dot-size",     type=int,   default=9,    dest="dot_size",
                     help="Scatter dot size in pts² (default: 9).")
    viz.add_argument("--dot-alpha",    type=float, default=0.55, dest="dot_alpha",
                     help="Dot opacity 0–1 (default: 0.55).")
    viz.add_argument("--pad",          type=float, default=1.2,
                     help="Axis padding around particle cloud (default: 1.2).")
    viz.add_argument("--separate",     action="store_true",
                     help="Write each panel as a separate file instead of one combined figure.")
    viz.add_argument("--scale",        type=float, default=None,
                     help=(
                         "Normalization scale used during training (divides model inputs). "
                         "Auto-detected from spec.train_cfg.scale_factor when omitted."
                     ))

    return p


def _infer_arch(level: str | None) -> tuple[int, int]:
    """Return (hidden_size, n_layers) defaults based on experiment level."""
    level = (level or "").lower()
    if "complex" in level:
        return 128, 3
    if "medium" in level:
        return 64, 2
    return 32, 2   # simple (default)


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Resolve paths ────────────────────────────────────────────────────────
    models_dir = args.models_dir
    if models_dir is None:
        if args.exp_dir:
            models_dir = os.path.join(args.exp_dir, "models")
        else:
            parser.error("Provide --exp-dir or --models-dir.")

    if not os.path.isdir(models_dir):
        parser.error(f"models_dir not found: {models_dir}")

    initial_model_path = os.path.join(models_dir, "initial_forward_model.pth")
    final_model_path   = os.path.join(models_dir, "forward_model.pth")

    for p_check in (initial_model_path, final_model_path):
        if not os.path.isfile(p_check):
            parser.error(f"Model file not found: {p_check}")

    output_stem = args.output
    if output_stem is None:
        base = args.exp_dir or os.path.dirname(models_dir)
        output_stem = os.path.join(base, "transport_figure")

    # ── Load experiment spec for distributions ───────────────────────────────
    from dualwcfm.experiments.exp_2d       import make_2d_experiments
    from dualwcfm.experiments.exp_3d       import make_3d_experiments
    from dualwcfm.experiments.exp_gushchin import make_gushchin_experiments
    from dualwcfm.experiments.exp_scalability import make_scalability_experiments

    catalogue_factories = {
        "2d":          make_2d_experiments,
        "3d":          make_3d_experiments,
        "gushchin":    make_gushchin_experiments,
        "scalability": make_scalability_experiments,
    }
    catalogue = catalogue_factories[args.mode](args.results_root)

    # Filter by level / perturbed
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
    hidden_size = args.hidden   if args.hidden   is not None else hidden_default
    n_layers    = args.n_layers if args.n_layers is not None else layers_default
    dim         = args.dim
    print(f"Architecture: dim={dim}  hidden={hidden_size}  n_layers={n_layers}")

    # ── Build figure ─────────────────────────────────────────────────────────
    build_figure(
        exp_name            = spec.name,
        spec                = spec,
        initial_model_path  = initial_model_path,
        final_model_path    = final_model_path,
        dim                 = dim,
        hidden_size         = hidden_size,
        n_layers            = n_layers,
        device              = device,
        scale               = args.scale,
        n_particles         = args.n_particles,
        n_steps             = args.n_steps,
        n_traj_frames       = args.n_traj_frames,
        output_stem         = output_stem,
        separate            = args.separate,
        pad                 = args.pad,
    )


if __name__ == "__main__":
    main()
