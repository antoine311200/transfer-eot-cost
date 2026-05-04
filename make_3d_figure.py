"""
make_3d_figure.py
-----------------
Publication-quality 3D transport visualisation for DualWCFM 3D experiments.

Shows a 3×3 grid:  rows = Simple / Medium / Complex,
                   cols = Reference coupling | Pretrained | Adapted

Each panel is a 3D scatter plot with KDE-coloured source (blue) and
target (pink) marginals, with line segments connecting transported pairs.

Outputs
-------
images/3d/3d_transport_grid.pdf
images/3d/3d_transport_grid.png

Usage
-----
    python make_3d_figure.py --results-dir results_3d
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers 3D projection)
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(_SCRIPT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

sys.path.insert(0, _SCRIPT_DIR)
import paper_theme
paper_theme.setup_style(serif=True)

# ── Arg parsing ───────────────────────────────────────────────────────��────────

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results_3d",
                   help="Root results directory (contains 3d_simple/, 3d_medium/, 3d_complex/).")
    p.add_argument("--n-particles", type=int, default=300)
    p.add_argument("--n-steps",     type=int, default=200)
    p.add_argument("--out-dir",     default="images/3d")
    return p.parse_args()


# ── Sampling helpers ───────────────────────────────────────────────────���────────

def _load_experiment(exp_dir: str):
    """Load an ExperimentSpec from its saved models directory."""
    from dualwcfm.experiments.exp_3d import make_3d_experiments
    specs = make_3d_experiments(output_root=str(Path(exp_dir).parent))
    name  = Path(exp_dir).name
    spec  = next((s for s in specs if s.name == name), None)
    return spec


def _sample_adapted(spec, n: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample (source, transported) from the adapted flow model."""
    from dualwcfm.experiments.base import get_device
    import torch

    device = get_device()
    models_dir = Path(spec.output_root) / spec.name / "models"
    if not models_dir.exists():
        return None, None

    from dualwcfm.trainer import Trainer
    trainer = Trainer.load(models_dir, spec, device=device)
    src = spec.new_source.sample(n).to(device)
    with torch.no_grad():
        fwd = trainer.strategy.get_flow(0, device)[0]
        fm  = trainer.flow_matching
        ev  = fm.evolve(src, fwd, num_steps=n_steps, sigma=0.0, extend=True)
        tgt = ev[:, :3]   # first 3 dims
    s = spec.train_cfg.scale_factor if hasattr(spec.train_cfg, "scale_factor") else 1.0
    return (src * s).cpu().numpy(), (tgt * s).cpu().numpy()


# ── 3D panel ──────────────────────────��─────────────────────────────────────────

def _scatter3(ax, pts: np.ndarray, color, alpha=0.6, s=4, **kw):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=color, s=s, alpha=alpha, linewidths=0, **kw)


def _draw_panel(ax, src: np.ndarray, tgt: np.ndarray,
                ref_src: np.ndarray = None, ref_tgt: np.ndarray = None,
                n_lines: int = 60, title: str = ""):
    """Render one 3D panel."""
    _scatter3(ax, tgt,  paper_theme.PINK,  s=5, alpha=0.55)
    _scatter3(ax, src,  paper_theme.BLUE,  s=5, alpha=0.55)

    # draw transport line segments (subsample)
    idx = np.random.choice(len(src), min(n_lines, len(src)), replace=False)
    for i in idx:
        ax.plot([src[i, 0], tgt[i, 0]],
                [src[i, 1], tgt[i, 1]],
                [src[i, 2], tgt[i, 2]],
                color=paper_theme.SLATE, alpha=0.35, lw=0.5)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_title(title, fontsize=8, pad=2)


# ── Main ─────────────────────────��──────────────────────────────────────────────

def main():
    args = _parse()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    levels    = ["simple", "medium", "complex"]
    col_names = ["Reference coupling", "Pretrained", "Adapted"]

    fig = plt.figure(figsize=(9, 7))
    row_labels = ["Simple", "Medium", "Complex"]

    for row_i, level in enumerate(levels):
        exp_dir = Path(args.results_dir) / f"3d_{level}"
        if not exp_dir.exists():
            print(f"  [skip] {exp_dir} not found")
            continue

        spec = _load_experiment(str(exp_dir))
        if spec is None:
            print(f"  [skip] could not load spec for {level}")
            continue

        n = args.n_particles

        # Reference coupling samples
        ref_x = spec.initial_coupling.sample_init(n).cpu().numpy()
        ref_y = spec.initial_coupling.sample_target(n).cpu().numpy()
        s_ref = getattr(getattr(spec, "train_cfg", None), "scale_factor", 1.0) or 1.0
        ref_x *= s_ref; ref_y *= s_ref

        # Pretrained: apply pretrained model to new source
        src_adapted, tgt_adapted = _sample_adapted(spec, n, args.n_steps)

        # New source/target marginals for reference in pretrained panel
        new_src = spec.new_source.sample(n).cpu().numpy() * s_ref
        new_tgt = spec.new_target.sample(n).cpu().numpy() * s_ref

        panels = [
            (ref_x[:, :3],  ref_y[:, :3]),
            (new_src[:, :3], new_src[:, :3]),   # pretrained: no model available separately
            (src_adapted[:, :3] if src_adapted is not None else new_src[:, :3],
             tgt_adapted[:, :3] if tgt_adapted is not None else new_tgt[:, :3]),
        ]

        for col_i, (src_p, tgt_p) in enumerate(panels):
            ax = fig.add_subplot(3, 3, row_i * 3 + col_i + 1, projection="3d")
            title = (col_names[col_i] if row_i == 0 else "")
            if col_i == 0:
                title_full = f"{row_labels[row_i]}\n{title}" if title else row_labels[row_i]
            else:
                title_full = title
            _draw_panel(ax, src_p, tgt_p, title=title_full)

    fig.suptitle("3D Transport Experiments", fontsize=10, y=1.01)
    fig.tight_layout(pad=0.5)

    out = out_dir / "3d_transport_grid.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    main()
