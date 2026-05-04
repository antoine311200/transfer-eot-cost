"""
make_far_convergence_figure.py
------------------------------
Reads the final map_err_fwd from results_final_2d/2d_far/iter_N/metrics.csv
for N in {11, 25, 51, 101, 251, 1001} and produces a log-log convergence
figure with a C/N least-squares fit, saved to images/ as PDF and JPG.
"""
from __future__ import annotations

import csv
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import paper_theme

# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
RESULTS_ROOT = Path("results_final_2d/2d_far")
ITERS        = [11, 25, 51, 101, 251, 1001]

def _read_last_map_err(n: int) -> float:
    csv_path = RESULTS_ROOT / f"iter_{n}" / "metrics.csv"
    last_row = None
    with open(csv_path, newline="") as f:
        for last_row in csv.DictReader(f):
            pass
    if last_row is None:
        raise ValueError(f"Empty CSV: {csv_path}")
    val = last_row["map_err_fwd"]
    if not val:
        raise ValueError(f"map_err_fwd is empty in final row of {csv_path}")
    return float(val)

# raw command-line iteration counts → last iteration index in the CSV
N_cmd   = np.array(ITERS, dtype=float)            # 11, 25, 51, 101, 251, 1001
N_disp  = np.where(N_cmd % 10 == 1, N_cmd - 1, N_cmd)  # 10, 25, 50, 100, 250, 1000
map_err = np.array([_read_last_map_err(n) for n in ITERS])

# ---------------------------------------------------------------------------
# C/N fit in log-space  (slope fixed at -1, solve for intercept)
# Use displayed N so the C constant corresponds to the axis labels
# ---------------------------------------------------------------------------
log_C = float(np.mean(np.log(N_disp * map_err)))
C_fit = np.exp(log_C)

X_MIN, X_MAX = 7.0,   1900.0
Y_MIN, Y_MAX = 2.8,   2200.0

N_dense = np.logspace(np.log10(X_MIN), np.log10(X_MAX), 500)
y_fit   = C_fit / N_dense

print("Data (displayed N, map_err):")
for nd, ni, mi in zip(N_disp, N_cmd, map_err):
    print(f"  N={int(nd):5d}  (cmd {int(ni)})  map_err={mi:.4f}  N*map_err={nd*mi:.1f}")
print(f"\nFitted C = {C_fit:.1f}  (delta = 1/N)")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
paper_theme.setup_style()
matplotlib.rcParams.update({
    "axes.labelsize":   11,
    "axes.titlesize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.linewidth":   0.85,
    "lines.linewidth":  1.8,
})

BLUE   = paper_theme.BLUE
ORANGE = paper_theme.PINK
GRAY   = paper_theme.SLATE

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 4.2))

# — fit curve —
ax.plot(N_dense, y_fit, color=ORANGE, linestyle="--", linewidth=2.0,
        label=rf"$C/N$   ($C \approx {C_fit:.0f}$)", zorder=2)

# — data —
ax.plot(N_disp, map_err, color=BLUE, linewidth=0.9, alpha=0.45, zorder=3)
ax.scatter(N_disp, map_err, color=BLUE, s=72, zorder=5,
           label=r"Map error (final iteration)")

# — annotate each point: N value + RMSE value, two lines —
# N=10 and N=1000 are placed directly next to their points (no arrow stub)
label_cfg = {
    10:   dict(xt=12.5, yt=600,  ha="left",  va="top",    arrow=False),
    25:   dict(xt=34,   yt=330,  ha="left",  va="bottom",  arrow=True),
    50:   dict(xt=67,   yt=119,  ha="left",  va="bottom",  arrow=True),
    100:  dict(xt=137,  yt=64,   ha="left",  va="bottom",  arrow=True),
    250:  dict(xt=332,  yt=25,   ha="left",  va="bottom",  arrow=True),
    1000: dict(xt=700,  yt=7.8,  ha="right", va="bottom",  arrow=False),
}
for nd, mi in zip(N_disp, map_err):
    cfg = label_cfg[int(nd)]
    txt = f"$N\\!=\\!{int(nd)}$\nRMSE $= {mi:.3g}$"
    kw = dict(
        fontsize=8.0, color=BLUE,
        ha=cfg["ha"], va=cfg["va"],
        multialignment=cfg["ha"],
    )
    if cfg["arrow"]:
        kw["arrowprops"] = dict(arrowstyle="-", color=BLUE, lw=0.55,
                                shrinkA=3, shrinkB=3)
    ax.annotate(txt, xy=(nd, mi), xytext=(cfg["xt"], cfg["yt"]), **kw)

# — slope reference right-triangle: right angle at BOTTOM-LEFT —
# vertical leg on the LEFT, horizontal leg at BOTTOM
# x0=13 → x1=41  (factor ≈3.15, log10 ratio ≈ 0.50 decades)
tri_x0, tri_x1 = 13.0, 41.0
tri_y_low       = 30.0
tri_y_high      = tri_y_low * (tri_x1 / tri_x0)     # ≈ 94.6

x_tri_c = (tri_x0 * tri_x1) ** 0.5                  # ≈ 23.1
y_tri_c = (tri_y_low * tri_y_high) ** 0.5            # ≈ 53.3

ax.plot([tri_x0, tri_x0], [tri_y_low,  tri_y_high], color=GRAY, lw=1.0)  # vertical left
ax.plot([tri_x0, tri_x1], [tri_y_low,  tri_y_low],  color=GRAY, lw=1.0)  # horizontal bottom
ax.plot([tri_x0, tri_x1], [tri_y_high, tri_y_low],  color=GRAY, lw=0.7,
        linestyle=":")  # hypotenuse (slope −1 in log space)

# leg labels: left of vertical leg, below horizontal leg
ax.text(tri_x0 * 0.91, y_tri_c,
        r"$\log_{10} 3$", ha="right", va="center", fontsize=7.5, color=GRAY,
        rotation=90)
ax.text(x_tri_c, tri_y_low * 0.73,
        r"$\log_{10} 3$", ha="center", va="top", fontsize=7.5, color=GRAY)

# — axes —
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_xlabel(r"Number of iterations $N$   ($\delta = 1/N$)")
ax.set_ylabel(r"Map error (RMSE)")
ax.set_title(
    r"Empirical $\mathcal{O}(\delta)$ convergence  —  2D-Far experiment",
    pad=9, fontweight="bold",
)

# — ticks: show only the experiment N values —
ax.set_xticks(N_disp)
ax.get_xaxis().set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${int(x)}$"))
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda y, _: f"${y:.4g}$"))
ax.tick_params(axis="x", which="minor", bottom=False)

ax.grid(True, which="major", linestyle=":", color=paper_theme.BORDER)
ax.grid(True, which="minor", linestyle=":", color=paper_theme.BORDER, linewidth=0.3)
paper_theme.style_axes(ax)

# — legend —
legend = ax.legend(loc="upper right", **paper_theme.legend_kw(borderpad=0.75))

fig.tight_layout(pad=0.7)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = Path("images")
out_dir.mkdir(exist_ok=True)

pdf_path = out_dir / "2d_far_convergence.pdf"
jpg_path = out_dir / "2d_far_convergence.jpg"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(jpg_path, bbox_inches="tight", dpi=220)
plt.close(fig)

print(f"\nSaved  {pdf_path}")
print(f"Saved  {jpg_path}")
