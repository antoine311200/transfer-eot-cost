"""
make_grid_figure.py
-------------------
Creates a paper-ready grid figure for the color_mnist_rgb_simple experiment.

Grid layout
~~~~~~~~~~~
  Columns (left to right) : training iterations
      Initial | Iter 10 | Iter 20 | Iter 30 | Iter 40 | Final

  Rows (top to bottom) : ODE trajectory snapshots via Euler integration
      t = 0.00  (source / PURPLE digit 7)
      t = 0.25
      t = 0.50
      t = 0.75
      t = 1.00  (transported result)

Directional axis annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  -------------------------------------------------> Training iterations
  v  ODE transport  t

Usage
~~~~~
  python make_grid_figure.py                    # default sample index 0
  python make_grid_figure.py --sample 3         # pick another digit-7 image
  python make_grid_figure.py --steps 3          # use 3 Euler steps (4 rows)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make the dualwcfm package importable from any CWD
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(_SCRIPT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dualwcfm.models.unet import UNetModel                  # UNetModelWrapper
from dualwcfm.experiments.exp_image import UNetWrapper      # (t, x[, cond]) → v

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_EXP_SLUG = (
    "color_mnist_rgb_simple__as100_bs128_ce250_de100_it51_lrdF_pr2e-1_sc1e0_s5e-2_s5e-2_usF"
)
_EXP_DIR         = os.path.join(_SCRIPT_DIR, "results_grid_color", _EXP_SLUG)
_MODELS_DIR      = os.path.join(_EXP_DIR,    "models")
_CKPT_DIR        = os.path.join(_MODELS_DIR, "checkpoints", "models")
_DATA_ROOT       = os.path.join(_SCRIPT_DIR, "data")

# ---------------------------------------------------------------------------
# Architecture constants (must match BidirectionalUnetStrategy._create_models)
# ---------------------------------------------------------------------------
C, H, W          = 3, 32, 32        # RGB, 32×32 (MNIST resized)
_NUM_CHANNELS    = 32               # FeedForwardConfig.hidden_size
_NUM_RES_BLOCKS  = 1
_CHANNEL_MULT    = (1, 2, 3, 4)
_PURPLE          = (1.0, 0.0, 1.0)  # source colour mask for the simple variant

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _build_forward_model(device: torch.device) -> torch.nn.Module:
    """Instantiate the forward-flow UNet wrapper (architecture only, no weights)."""
    unet = UNetModel(
        dim          = (2 * C, H, W),       # extended state: [x_t ‖ x_0]
        num_channels = _NUM_CHANNELS,
        num_res_blocks = _NUM_RES_BLOCKS,
        channel_mult = _CHANNEL_MULT,
        out_channels = 2 * C,               # velocity for both halves (only first C used)
    )
    return UNetWrapper(unet).to(device)


def load_forward_model(path: str, device: torch.device) -> torch.nn.Module:
    model = _build_forward_model(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_purple_digit7(n: int = 100, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Load first *n* MNIST digit-7 images with PURPLE colour mask applied.

    Returns tensor of shape (n, 3, 32, 32) in [-1, 1].
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root=_DATA_ROOT, train=True, download=True, transform=transform)
    indices  = [i for i, (_, lbl) in enumerate(dataset) if lbl == 7][:n]
    imgs     = torch.stack([dataset[i][0] for i in indices]).to(device)  # (N, 1, 32, 32)

    # Grayscale → 3-channel RGB then apply PURPLE mask  (r=1, g=0, b=1)
    imgs = imgs.expand(-1, 3, -1, -1).clone()
    imgs[:, 1, :, :] *= 0.0  # zero the green channel
    return imgs


# ---------------------------------------------------------------------------
# ODE integration
# ---------------------------------------------------------------------------

def euler_trajectory(
    x0: torch.Tensor,
    model: torch.nn.Module,
    n_display: int,
    device: torch.device,
    n_integration: int = 200,
) -> list:
    """Euler integration from x0, returning n_display evenly-spaced snapshots.

    Integrates with n_integration fine steps (matching the eval convention of
    200 steps) and captures frames at evenly-spaced indices so the displayed
    trajectory is smooth and artefact-free.

    The forward model expects the extended state [x_t || x_0] (2C channels).
    Only the first C channels evolve; the second half stays frozen as x_0.

    Returns a list of n_display + 1 tensors of shape (1, C, H, W) in [-1, 1],
    corresponding to t = 0, 1/n_display, ..., 1.
    """
    dt = 1.0 / n_integration
    xt = torch.cat([x0, x0], dim=1)   # (1, 2C, H, W)

    # Indices of integration steps at which to capture a snapshot
    capture = {round(i * n_integration / n_display) for i in range(n_display + 1)}

    states = []
    with torch.no_grad():
        if 0 in capture:
            states.append(x0.clone())
        for i in range(n_integration):
            t  = torch.full((xt.shape[0], 1), i * dt, device=device)
            vt = model(t, xt)
            first_half = xt[:, :C] + vt[:, :C] * dt
            xt = torch.cat([first_half, xt[:, C:]], dim=1)
            if (i + 1) in capture:
                states.append(xt[:, :C].clone())

    return states  # list of (n_display+1) tensors, shape (1, C, H, W)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _to_rgb(t: torch.Tensor) -> np.ndarray:
    """Convert (1, C, H, W) tensor in [-1, 1] to (H, W, C) uint8 array.

    The colour masks used in this experiment (e.g. PURPLE = (1, 0, 1)) set
    masked channels to 0 rather than -1, so the background becomes
    (R=-1, G=0, B=-1).  Clipping to [0, 1] — as the eval visualiser does —
    maps every sub-zero value to 0 (black), giving the expected appearance.
    Using (x+1)/2 would instead lift G=0 to 0.5, producing a green background.
    """
    img = t[0].permute(1, 2, 0).cpu().float().numpy()   # (H, W, C) in [-1, 1]
    img = np.clip(img, 0, 1)                             # match eval convention
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def _resolve_dirs(ckpt_dir: str, models_dir: str):
    """Resolve models_dir from ckpt_dir when not explicitly given.

    The standard layout is:
        <exp>/models/                     ← models_dir  (initial + final)
        <exp>/models/checkpoints/models/  ← ckpt_dir    (per-iteration)

    So models_dir = os.path.dirname(os.path.dirname(ckpt_dir)).
    """
    ckpt_dir = ckpt_dir or _CKPT_DIR
    if models_dir is None:
        # Walk up two levels: .../checkpoints/models → .../checkpoints → ...
        models_dir = os.path.dirname(os.path.dirname(ckpt_dir))
        # Sanity-check: fall back to the hardcoded default if it doesn't look right
        if not os.path.isfile(os.path.join(models_dir, "forward_model.pth")):
            models_dir = _MODELS_DIR
    return models_dir, ckpt_dir


def _pick_checkpoints(ckpt_dir: str, n_middle: int = 4):
    """Auto-detect available iter_XXXX_forward.pth files and pick n_middle
    evenly-spaced ones (excluding iteration 0, which is covered by
    initial_forward_model.pth).

    Returns a list of (label, path) pairs.
    """
    import re
    files = sorted(
        f for f in os.listdir(ckpt_dir) if re.match(r"iter_\d+_forward\.pth", f)
    )
    # Extract iteration numbers and skip iter 0
    iters = []
    for f in files:
        m = re.match(r"iter_(\d+)_forward\.pth", f)
        if m:
            n = int(m.group(1))
            if n > 0:
                iters.append(n)
    iters.sort()

    if len(iters) == 0:
        return []

    # Drop the last checkpoint — it duplicates forward_model.pth (Final column)
    if len(iters) > 1:
        iters = iters[:-1]

    # Pick n_middle evenly-spaced indices
    if len(iters) <= n_middle:
        chosen = iters
    else:
        indices = [round(i * (len(iters) - 1) / (n_middle - 1)) for i in range(n_middle)]
        chosen  = [iters[i] for i in indices]

    # Labels are 1-indexed (iter_0004 = iteration 5 in 1-based counting)
    return [
        (f"Iter {it + 1}", os.path.join(ckpt_dir, f"iter_{it:04d}_forward.pth"))
        for it in chosen
    ]


def build_figure(
    sample_idx: int = 0,
    n_display: int = 4,
    n_integration: int = 200,
    n_middle: int = 4,
    output_stem: str = "paper_grid_figure",
    models_dir: str = None,
    ckpt_dir: str = None,
):
    """Build the grid figure.

    Parameters
    ----------
    n_display:
        Number of ODE snapshot intervals (rows = n_display + 1, default 4 -> 5 rows).
    n_integration:
        Number of fine Euler steps for integration (default 200, matching eval).
    n_middle:
        Number of intermediate checkpoint columns between Initial and Final
        (default 4, giving 6 columns total).
    models_dir:
        Directory containing ``initial_forward_model.pth`` and
        ``forward_model.pth``.  When omitted, inferred as two levels above
        ``ckpt_dir`` (the standard layout).
    ckpt_dir:
        Directory containing the per-iteration checkpoint files
        (``iter_XXXX_forward.pth``).  Defaults to ``_CKPT_DIR``.
    """
    models_dir, ckpt_dir = _resolve_dirs(ckpt_dir, models_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Models dir : {models_dir}")
    print(f"Checkpoints: {ckpt_dir}")

    # ------------------------------------------------------------------ data
    print("Loading PURPLE digit-7 images ...")
    all_imgs = load_purple_digit7(n=200, device=device)
    x0       = all_imgs[sample_idx : sample_idx + 1]   # (1, 3, 32, 32)
    print(f"  sample {sample_idx}: shape={x0.shape}, "
          f"range=[{x0.min():.2f}, {x0.max():.2f}]")

    # ----------------------------------------------------------------- models
    middle_cols = _pick_checkpoints(ckpt_dir, n_middle=n_middle)
    columns = (
        [("Initial", os.path.join(models_dir, "initial_forward_model.pth"))]
        + middle_cols
        + [("Final",  os.path.join(models_dir, "forward_model.pth"))]
    )
    print("Columns:")
    for lbl, path in columns:
        print(f"  {lbl:12s} {path}")

    n_cols = len(columns)
    n_rows = n_display + 1
    t_vals = [i / n_display for i in range(n_rows)]

    print(f"Integration: {n_integration} Euler steps, displaying {n_rows} snapshots")

    # ---------------------------------------------------- compute trajectories
    all_traj: list[list[np.ndarray]] = []  # [col][row] → (H, W, C) uint8
    for col_label, model_path in columns:
        print(f"  Loading model: {col_label} ({os.path.basename(model_path)}) ...", flush=True)
        model  = load_forward_model(model_path, device)
        frames = euler_trajectory(x0, model, n_display, device, n_integration=n_integration)
        all_traj.append([_to_rgb(f) for f in frames])
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ------------------------------------------------------------------ figure
    img_pt   = 1.5          # size of each image cell in inches
    pad_left = 1.0          # inches: row-label + left-arrow room
    pad_bot  = 0.75         # inches: bottom-arrow room
    pad_top  = 0.60         # inches: column-header room
    pad_right = 0.25        # inches: small right margin

    fig_w = pad_left  + n_cols * img_pt + pad_right
    fig_h = pad_top   + n_rows * img_pt + pad_bot

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor("white")

    # grid left/right/top/bottom as figure fractions
    gl = pad_left  / fig_w
    gr = 1 - pad_right / fig_w
    gt = 1 - pad_top   / fig_h
    gb = pad_bot  / fig_h

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        left=gl, right=gr, top=gt, bottom=gb,
        hspace=0.035, wspace=0.035,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]

    # ----------------------------------------------------------- fill cells
    for c in range(n_cols):
        for r in range(n_rows):
            ax  = axes[r][c]
            img = all_traj[c][r]
            ax.imshow(img, interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#cccccc")

    # ----------------------------------------------------- column headers
    for c, (label, _) in enumerate(columns):
        axes[0][c].set_title(label, fontsize=8.5, fontweight="bold", pad=5,
                             fontfamily="serif")

    # --------------------------------------------------------- row labels
    for r, tv in enumerate(t_vals):
        ax = axes[r][0]
        ax.set_ylabel(
            f"$t = {tv:.2f}$",
            fontsize=8, rotation=0,
            labelpad=34, va="center",
            fontfamily="serif",
        )

    # ------------------------------------------------------ overlay axes
    # An axes spanning the full figure for drawing arrows
    ax_ov = fig.add_axes([0, 0, 1, 1], facecolor="none", zorder=10)
    ax_ov.set_xlim(0, 1)
    ax_ov.set_ylim(0, 1)
    ax_ov.axis("off")

    arrow_kw = dict(
        arrowstyle    = "->",
        color         = "#222222",
        lw            = 1.4,
        mutation_scale = 10,
    )

    # ---- horizontal arrow  (left → right, Training iterations)
    h_y   = gb - 0.045
    ax_ov.annotate(
        "",
        xy      = (gr, h_y), xytext = (gl, h_y),
        xycoords = "figure fraction", textcoords = "figure fraction",
        arrowprops = arrow_kw,
    )
    ax_ov.text(
        (gl + gr) / 2, h_y - 0.022,
        "Training iterations",
        ha="center", va="top",
        fontsize=9, color="#222222",
        fontfamily="serif",
        transform=fig.transFigure,
    )

    # ---- vertical arrow  (top → bottom, ODE transport t)
    v_x   = gl - 0.055
    ax_ov.annotate(
        "",
        xy      = (v_x, gb), xytext = (v_x, gt),
        xycoords = "figure fraction", textcoords = "figure fraction",
        arrowprops = arrow_kw,
    )
    ax_ov.text(
        v_x - 0.022, (gb + gt) / 2,
        "ODE transport $t$",
        ha="right", va="center",
        fontsize=9, color="#222222",
        fontfamily="serif",
        rotation=90,
        transform=fig.transFigure,
    )

    # ---------------------------------------------------------------- save
    for ext in ("pdf", "png"):
        out = os.path.join(_SCRIPT_DIR, f"{output_stem}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample",   type=int, default=0,
                        help="Index of the digit-7 image to use (default: 0)")
    parser.add_argument("--steps",         type=int, default=4,
                        help="Number of ODE display rows minus 1 (default: 4 -> 5 rows)")
    parser.add_argument("--n-integration", type=int, default=200,
                        help="Fine Euler steps for integration (default: 200, matches eval)")
    parser.add_argument("--n-middle", type=int, default=4,
                        help="Number of intermediate checkpoint columns (default: 4)")
    parser.add_argument("--output",   type=str, default="paper_grid_figure",
                        help="Output filename stem (default: paper_grid_figure)")

    # Path overrides — useful when running on a different experiment folder
    parser.add_argument(
        "--exp-dir", type=str, default=None,
        help=(
            "Root of an experiment directory (the folder that contains a "
            "'models/' sub-folder).  Sets both --models-dir and --ckpt-dir "
            "automatically; individual overrides take precedence."
        ),
    )
    parser.add_argument(
        "--models-dir", type=str, default=None,
        help=(
            "Directory containing initial_forward_model.pth and "
            "forward_model.pth.  Overrides --exp-dir for the pretrained models."
        ),
    )
    parser.add_argument(
        "--ckpt-dir", type=str, default=None,
        help=(
            "Directory containing iter_XXXX_forward.pth checkpoint files.  "
            "Overrides --exp-dir for the checkpoints."
        ),
    )

    args = parser.parse_args()

    # Resolve paths: --exp-dir sets defaults; explicit flags override it
    models_dir = None
    ckpt_dir   = None
    if args.exp_dir:
        models_dir = os.path.join(args.exp_dir, "models")
        ckpt_dir   = os.path.join(args.exp_dir, "models", "checkpoints", "models")
    if args.models_dir:
        models_dir = args.models_dir
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir

    build_figure(
        sample_idx  = args.sample,
        n_display     = args.steps,
        n_integration = args.n_integration,
        n_middle    = args.n_middle,
        output_stem = args.output,
        models_dir  = models_dir,
        ckpt_dir    = ckpt_dir,
    )
