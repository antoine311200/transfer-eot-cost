"""
make_convergence_figure.py
--------------------------
Convergence figure from exp_2d_simple wandb runs.
Fixed parametrisation: sigma=sigma2=0.01, particle_ratio=1.
One run per N (10, 25, 50, 100, 250 …) — no averaging.
"""
from __future__ import annotations
import os, re, base64, json
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
ENTITY   = "antoine311200"
PROJECT  = "exp_2d_simple"
SIGMA    = 0.01
SIGMA2   = 0.01
PR       = 1.0           # particle_ratio=1 gives cleanest monotone decrease
OUT_DIR  = Path("plots")

CRANE_YELLOW = "#FFD200"
COLOR_EDGE   = "#8B6D00"
COLOR_REF    = "#E05000"
FIGSIZE      = (4.8, 3.2)

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Wandb API helpers ─────────────────────────────────────────────────────────
def _api_key() -> str:
    for path in [os.path.expanduser("~/.netrc"), os.path.expanduser("~/_netrc")]:
        if os.path.exists(path):
            m = re.search(r"machine api\.wandb\.ai.*?password (\S+)", open(path).read(), re.DOTALL)
            if m:
                return m.group(1)
    raise RuntimeError("No wandb API key found in ~/.netrc")


def fetch_runs(entity: str, project: str) -> list[dict]:
    key = _api_key()
    headers = {"Authorization": "Basic " + base64.b64encode(f"api:{key}".encode()).decode()}
    q = (
        '{project(name:"%s",entityName:"%s")'
        '{runs(first:500){edges{node{state config summaryMetrics}}}}}'
        % (project, entity)
    )
    resp = requests.post("https://api.wandb.ai/graphql", headers=headers, json={"query": q})
    resp.raise_for_status()
    return resp.json()["data"]["project"]["runs"]["edges"]


def gv(cfg: dict, key: str):
    v = cfg.get(key)
    if v is None:
        return None
    return v.get("value", v) if isinstance(v, dict) else v


def parse_runs(edges: list) -> list[dict]:
    rows = []
    for edge in edges:
        n = edge["node"]
        if n["state"] != "finished":
            continue
        cfg  = json.loads(n.get("config", "{}"))
        smry = json.loads(n.get("summaryMetrics", "{}"))
        iters = gv(cfg, "iterations")
        pr    = gv(cfg, "particle_ratio")
        sig   = gv(cfg, "sigma")
        sig2  = gv(cfg, "sigma2")
        if any(x is None for x in [iters, pr, sig, sig2]):
            continue
        try:
            iters = int(iters)
            pr    = round(float(pr), 4)
            sig   = round(float(sig), 6)
            sig2  = round(float(sig2), 6)
        except (ValueError, TypeError):
            continue
        map_err = smry.get("map_err_fwd", smry.get("map_err_bwd"))
        if map_err is None:
            continue
        rows.append({"iterations": iters, "pr": pr, "sigma": sig, "sigma2": sig2,
                     "map_err": float(map_err)})
    return rows


# ── Select runs: fixed sigma/sigma2/pr, one per N ────────────────────────────
def select_runs(rows: list[dict]) -> dict[int, float]:
    """Return {N: map_err} for the target parametrisation."""
    result: dict[int, float] = {}
    for r in rows:
        if abs(r["sigma"] - SIGMA) > 1e-9:
            continue
        if abs(r["sigma2"] - SIGMA2) > 1e-9:
            continue
        if abs(r["pr"] - PR) > 1e-4:
            continue
        n = r["iterations"]
        # keep the run; if duplicates exist (shouldn't), take smallest map_err
        if n not in result or r["map_err"] < result[n]:
            result[n] = r["map_err"]
    return dict(sorted(result.items()))


# ── Plotting ──────────────────────────────────────────────────────────────────
def bar_plot(data: dict[int, float], ax: plt.Axes, show_odelta: bool = False) -> None:
    ns      = sorted(data)
    errs    = np.array([data[n] for n in ns])
    x       = np.arange(len(ns))

    ax.bar(x, errs, color=CRANE_YELLOW, edgecolor=COLOR_EDGE, linewidth=0.8, zorder=3)

    if show_odelta:
        deltas = np.array([1.0 / n for n in ns])
        C      = errs[0] / deltas[0]
        ref    = C * deltas
        ax.plot(x, ref, color=COLOR_REF, linestyle="--", linewidth=1.4,
                label=r"$\mathcal{O}(\delta) = \mathcal{O}(1/N)$  reference",
                zorder=4)
        ax.legend(frameon=False, loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"$N={n}$\n$\\delta=1/{n}$" for n in ns], fontsize=7.5)
    ax.set_ylabel(r"Final map error  $\sqrt{\mathbb{E}\|T(x)-\hat{y}\|^2}$")
    ax.set_xlabel("Number of outer iterations $N$")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Fetching runs from wandb …")
    edges = fetch_runs(ENTITY, PROJECT)
    rows  = parse_runs(edges)
    print(f"  {len(rows)} valid finished runs")

    data = select_runs(rows)
    if not data:
        raise RuntimeError(
            f"No runs found for sigma={SIGMA}, sigma2={SIGMA2}, pr={PR}. "
            "Check config key names or parameter values."
        )
    print(f"  Selected runs (sigma={SIGMA}, sigma2={SIGMA2}, pr={PR}):")
    for n, e in data.items():
        print(f"    N={n:4d}  map_err={e:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: plain bars
    fig1, ax1 = plt.subplots(figsize=FIGSIZE)
    bar_plot(data, ax1, show_odelta=False)
    ax1.set_title(
        "Convergence of map error on 2D-Simple\n"
        r"($\sigma=\sigma_2=" + f"{SIGMA}" + r"$, $\alpha=" + f"{PR}" + r"$, weighted loss)",
        pad=6,
    )
    fig1.tight_layout()
    p1 = OUT_DIR / "convergence_map_err.pdf"
    fig1.savefig(p1, bbox_inches="tight")
    print(f"  Saved {p1}")

    # Figure 2: bars + O(delta) reference
    fig2, ax2 = plt.subplots(figsize=FIGSIZE)
    bar_plot(data, ax2, show_odelta=True)
    ax2.set_title(
        "Convergence of map error on 2D-Simple\n"
        r"($\sigma=\sigma_2=" + f"{SIGMA}" + r"$, $\alpha=" + f"{PR}" + r"$) + $\mathcal{O}(\delta)$ bound",
        pad=6,
    )
    fig2.tight_layout()
    p2 = OUT_DIR / "convergence_map_err_odelta.pdf"
    fig2.savefig(p2, bbox_inches="tight")
    print(f"  Saved {p2}")

    plt.close("all")
    print("Done.")


if __name__ == "__main__":
    main()
