"""
make_scalability_figure_v2.py
-----------------------------
Reads cached wandb data from images/scalability/wandb_history.json and
produces a clean publication-quality dual-axis scalability figure:
  - left  axis (blue): W2(nu_hat, nu)  -- stays flat with d
  - right axis (orange): BW2           -- grows with d
  - O(sqrt(d)) reference dashed line on the BW2 axis

Saved to:
  images/scalability/scalability.pdf   (replaces old dark-background figure)
  images/scalability/scalability.jpg
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
JSON_PATH = Path("images/scalability/wandb_history.json")
with open(JSON_PATH) as f:
    raw = json.load(f)

DIMS = [8, 16, 32, 64, 128]

# Take the last entry (step 98) for each dimension
def _final(d: int, key: str) -> float:
    rows = raw[str(d)]
    return float(rows[-1][key])

w2_vals  = np.array([_final(d, "W2_nu") for d in DIMS])
bw2_vals = np.array([_final(d, "bw2")   for d in DIMS])
dims_arr = np.array(DIMS, dtype=float)

print("Final values:")
for d, w, b in zip(DIMS, w2_vals, bw2_vals):
    print(f"  d={d:3d}  W2_nu={w:.4f}  BW2={b:.4f}")

# O(sqrt(d)) reference: fit C in log space with slope fixed at 0.5
log_C = float(np.mean(np.log(bw2_vals) - 0.5 * np.log(dims_arr)))
C_ref = np.exp(log_C)
print(f"\nO(sqrt(d)) reference C = {C_ref:.4f}")

# ---------------------------------------------------------------------------
# Style — matches make_far_convergence_figure.py
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "text.usetex":          False,
    "mathtext.fontset":     "cm",
    "font.family":          "serif",
    "font.size":            12,
    "axes.labelsize":       13,
    "axes.titlesize":       12,
    "legend.fontsize":      10.5,
    "xtick.labelsize":      11,
    "ytick.labelsize":      11,
    "axes.linewidth":       0.85,
    "lines.linewidth":      1.8,
    "grid.linewidth":       0.45,
    "grid.alpha":           0.30,
    "figure.dpi":           150,
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "savefig.facecolor":    "white",
})

BLUE   = "#1D4ED8"    # W2_nu (left axis)
ORANGE = "#C2410C"    # BW2   (right axis)
GRAY   = "#6B7280"    # reference line / annotations

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(5.8, 4.2))
ax2 = ax1.twinx()

# --- BW2 on right axis (drawn first so W2 points sit on top) ---
d_dense = np.logspace(np.log2(6), np.log2(160), 300, base=2)
bw2_ref = C_ref * np.sqrt(d_dense)

ax2.plot(d_dense, bw2_ref, color=ORANGE, linestyle="--", linewidth=1.6,
         alpha=0.70, label=rf"$C\sqrt{{d}}$  ($C\approx{C_ref:.2f}$)", zorder=2)
ax2.plot(dims_arr, bw2_vals, color=ORANGE, linewidth=0.8, alpha=0.4, zorder=3)
ax2.scatter(dims_arr, bw2_vals, color=ORANGE, s=72, zorder=5,
            label=r"$\mathrm{BW}_2$ (final)")

# annotate BW2 values
bw2_offsets = {8: (5.5, 0.20), 16: (11, 0.42), 32: (22, 0.81),
               64: (43, 1.65), 128: (86, 2.8)}
for d, bw in zip(DIMS, bw2_vals):
    xt, yt = bw2_offsets[d]
    ax2.annotate(f"${bw:.3g}$", xy=(d, bw), xytext=(xt, yt),
                 fontsize=8.0, color=ORANGE, ha="right", va="top",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.5,
                                 shrinkA=2, shrinkB=2))

# --- W2_nu on left axis ---
ax1.plot(dims_arr, w2_vals, color=BLUE, linewidth=0.8, alpha=0.4, zorder=3)
ax1.scatter(dims_arr, w2_vals, color=BLUE, s=72, zorder=5,
            label=r"$W_2(\hat\nu,\nu)$ (final)")

# annotate W2 values
w2_offsets = {8: (5.5, 0.068), 16: (11, 0.072), 32: (22, 0.068),
              64: (43, 0.107), 128: (86, 0.098)}
for d, w in zip(DIMS, w2_vals):
    xt, yt = w2_offsets[d]
    ax1.annotate(f"${w:.3f}$", xy=(d, w), xytext=(xt, yt),
                 fontsize=8.0, color=BLUE, ha="right", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5,
                                 shrinkA=2, shrinkB=2))

# --- axes setup ---
ax1.set_xscale("log", base=2)
ax1.set_xticks(dims_arr)
ax1.get_xaxis().set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${int(x)}$"))
ax1.tick_params(axis="x", which="minor", bottom=False)

ax1.set_xlim(5.5, 175)

# Left axis (blue W2)
w2_pad = 0.015
ax1.set_ylim(min(w2_vals) - w2_pad, max(w2_vals) + w2_pad + 0.04)
ax1.set_ylabel(r"$W_2(\hat\nu,\,\nu)$", color=BLUE)
ax1.tick_params(axis="y", colors=BLUE)
ax1.spines["left"].set_edgecolor(BLUE)
ax1.spines["left"].set_linewidth(1.2)

# Right axis (orange BW2)
bw2_pad = 0.3
ax2.set_ylim(0, max(bw2_vals) + bw2_pad + 0.5)
ax2.set_ylabel(r"$\mathrm{BW}_2$", color=ORANGE)
ax2.tick_params(axis="y", colors=ORANGE)
ax2.spines["right"].set_edgecolor(ORANGE)
ax2.spines["right"].set_linewidth(1.2)

ax1.set_xlabel(r"Dimension $d$")
ax1.grid(True, which="major", linestyle=":", color="#aaaaaa", alpha=0.4)

# --- title ---
ax1.set_title(
    r"Scalability  —  Anti-correlation benchmark  $T(x)=-x$",
    pad=9, fontweight="bold",
)

# --- combined legend ---
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2,
           loc="upper left", framealpha=0.93,
           edgecolor="#cccccc", borderpad=0.75)

fig.tight_layout(pad=0.7)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = Path("images/scalability")
out_dir.mkdir(exist_ok=True)

pdf_path = out_dir / "scalability.pdf"
jpg_path = out_dir / "scalability.jpg"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(jpg_path, bbox_inches="tight", dpi=220)
plt.close(fig)

print(f"\nSaved  {pdf_path}")
print(f"Saved  {jpg_path}")
