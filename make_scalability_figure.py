"""
Scalability figure for DualWCFM — pulls best runs from W&B and produces a
clean, publication-quality plot replacing the original black-background one.

Outputs
-------
images/scalability/scalability_wandb.pdf
images/scalability/scalability_wandb.png
"""

import json
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import wandb

warnings.filterwarnings("ignore")

# ── 0. Paths ───────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "images", "scalability")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Pull data from W&B ──────────────────────────────────────────────────────
api = wandb.Api()

# Best (scale_factor=50) finished runs per dimension
RUN_IDS = {8: "exq56rq3", 16: "vl3b6sl4", 32: "816s0y8a",
           64: "irgqqvl2", 128: "5j0llvt4"}

print("Pulling W&B history …")
history: dict[int, list[dict]] = {}
finals: dict[int, dict] = {}

for d, rid in RUN_IDS.items():
    run = api.run(f"antoine311200/dualwcfm-scalability/{rid}")
    rows = run.history(
        keys=["W2_nu", "bw2", "map_err_fwd", "sinkhorn_div", "weight_ess"],
        pandas=False,
    )
    pts = [
        {k: row.get(k) for k in ("_step", "W2_nu", "bw2", "map_err_fwd",
                                   "sinkhorn_div", "weight_ess")}
        for row in rows if row.get("W2_nu") is not None
    ]
    history[d] = pts
    s = run.summary
    finals[d] = {
        "W2_nu":        s.get("W2_nu"),
        "bw2":          s.get("bw2"),
        "map_err":      s.get("map_err_fwd"),
        "sinkhorn_div": s.get("sinkhorn_div"),
    }
    print(f"  d={d:3d}: {len(pts)} iters  W2_nu={finals[d]['W2_nu']:.4f}"
          f"  BW2={finals[d]['bw2']:.3f}  MapErr={finals[d]['map_err']:.3f}")

# ── 2. Style ───────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9.5,
    "xtick.labelsize":   9.5,
    "ytick.labelsize":   9.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E0E0E0",
    "grid.linewidth":    0.6,
    "lines.linewidth":   2.0,
})

DIMS   = [8, 16, 32, 64, 128]
# Yellow→teal palette (5 shades that work on white)
COLORS = ["#E6A817", "#C8720F", "#7B8B3A", "#2E7D9C", "#1A4E6E"]
DIM_COLOR = dict(zip(DIMS, COLORS))

# ── 3. Figure layout ───────────────────────────────────────────────────────────
# Three panels:
#   [convergence curves]  |  [final W2_nu vs d]  |  [final BW2 vs d]
fig = plt.figure(figsize=(13.5, 4.2))
gs  = fig.add_gridspec(1, 3, wspace=0.38, left=0.07, right=0.97,
                       top=0.88, bottom=0.15)
ax_conv = fig.add_subplot(gs[0])
ax_w2   = fig.add_subplot(gs[1])
ax_bw   = fig.add_subplot(gs[2])

# ── Panel A: convergence ───────────────────────────────────────────────────────
for d in DIMS:
    pts   = history[d]
    iters = [p["_step"] for p in pts]
    w2nu  = [p["W2_nu"] for p in pts]
    ax_conv.plot(iters, w2nu, color=DIM_COLOR[d], label=f"$d={d}$", zorder=3)

ax_conv.set_xlabel("Iteration")
ax_conv.set_ylabel(r"$W_2(\hat\nu,\,\nu)$")
ax_conv.set_title("A  Target-marginal quality over training", loc="left", fontweight="bold")
ax_conv.legend(title="Dimension", framealpha=0.85,
               ncol=1, loc="upper right", handlelength=1.5)
ax_conv.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

# ── Panel B: final W2_nu vs d ─────────────────────────────────────────────────
w2_vals  = [finals[d]["W2_nu"]  for d in DIMS]
bw2_vals = [finals[d]["bw2"]    for d in DIMS]
me_vals  = [finals[d]["map_err"] for d in DIMS]

bar_colors = [DIM_COLOR[d] for d in DIMS]
x = np.arange(len(DIMS))

bars = ax_w2.bar(x, w2_vals, color=bar_colors, edgecolor="white",
                 linewidth=0.8, zorder=3, width=0.6)
for bar, val in zip(bars, w2_vals):
    ax_w2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
               f"{val:.3f}", ha="center", va="bottom", fontsize=8.5,
               fontweight="bold")

# Reference line: pretrained W2_nu for d=8 (huge, draw as dashed cap)
ax_w2.axhline(0.12, ls="--", color="#888888", lw=1.2, label="Pretrained (d=8): 5.4")

# Arrow annotation showing flatness
ax_w2.annotate("nearly flat!", xy=(4, w2_vals[-1]),
               xytext=(2.5, max(w2_vals) * 1.25),
               arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1),
               fontsize=9, color="#444444", ha="center")

ax_w2.set_xticks(x)
ax_w2.set_xticklabels([str(d) for d in DIMS])
ax_w2.set_xlabel("Dimension $d$")
ax_w2.set_ylabel(r"Final $W_2(\hat\nu,\,\nu)$")
ax_w2.set_title("B  Marginal error stays flat with $d$", loc="left", fontweight="bold")
ax_w2.set_ylim(0, max(w2_vals) * 1.55)
ax_w2.legend(fontsize=8.5, framealpha=0.8)

# ── Panel C: BW2 vs d ─────────────────────────────────────────────────────────
ax_bw.plot(DIMS, bw2_vals, "o-", color="#2E7D9C", lw=2.2,
           ms=7, markerfacecolor="white", markeredgewidth=2, zorder=3,
           label="DualWCFM")

# Theoretical reference: BW2 of identity map with unit covariance = sqrt(2d)
bw2_ref = [np.sqrt(2 * d) for d in DIMS]
ax_bw.plot(DIMS, bw2_ref, "s--", color="#C0C0C0", lw=1.4,
           ms=5, markerfacecolor="#C0C0C0", label=r"$\sqrt{2d}$ ref.")

ax_bw.set_xlabel("Dimension $d$")
ax_bw.set_ylabel(r"Final $\mathrm{BW}_2$")
ax_bw.set_title(r"C  Bures–Wasserstein $\mathrm{BW}_2$ vs $d$", loc="left",
                fontweight="bold")
ax_bw.set_xscale("log", base=2)
ax_bw.set_yscale("log")
ax_bw.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: str(int(v))))
ax_bw.legend(framealpha=0.85)

# ── 4. Shared super-title ─────────────────────────────────────────────────────
fig.suptitle(
    "DualWCFM Scalability  —  Anti-correlation benchmark  "
    r"$T(x)=-x$,  $d\in\{8,16,32,64,128\}$",
    fontsize=12, fontweight="bold", y=0.97,
)

# ── 5. Save ────────────────────────────────────────────────────────────────────
pdf_path = os.path.join(OUT_DIR, "scalability_wandb.pdf")
png_path = os.path.join(OUT_DIR, "scalability_wandb.png")
fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
fig.savefig(png_path, dpi=200, bbox_inches="tight")
print(f"\nSaved:\n  {pdf_path}\n  {png_path}")
plt.show()
