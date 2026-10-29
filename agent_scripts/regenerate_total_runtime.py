"""Regenerate every runtime-bearing figure with ONE consistent T_compute definition:

    T_compute = compute time + preprocessing
      BvN   : bvn_runtime              (no preprocessing)
      Radix : {pre}_{b}_time           (digit extraction is ~0 / untimed)
      Euler : depth_time + depth_split  (max-leaf parallel + split phase)

Produces (paper/plots_pdf/):
  euler_tradeoff_corrected.pdf      (Euler, depths d1-d4, TOTAL runtime)
  design_space_projections.pdf      (Radix + Euler trajectories, TOTAL runtime)
"""
from __future__ import annotations
import glob, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8})

OUT = "C:/Users/ofekc/Desktop/Msc/Thesis/paper/plots_pdf"
ENG = [("Hungarian", "max", "maximum", "#0072B2", "o"),
       ("GW Dynamic", "heavy", "heavy", "#D55E00", "s"),
       ("GW Static", "heavy_static", "heavy_static", "#009E73", "^"),
       ("WFA", "wfa", "wfa", "#CC79A7", "D")]

def rerun(e): return sorted(glob.glob(f"run/rerun_2026/{e}/*/results_stats.csv"))[-1]
def euler(p): return sorted(glob.glob(f"euler_exp/eng_{p}_dense_d4/*/results_stats.csv"))[-1]

def points(name, re, pre):
    """Return list of (label, N, C, T_compute_ms) with TOTAL runtime."""
    dr = pd.read_csv(rerun(re)).iloc[5:]
    de = pd.read_csv(euler(pre)).iloc[5:]
    bvn = ("BvN", dr["bvn_perms"].mean(), dr["bvn_cycle"].mean(), dr["bvn_runtime"].mean()*1000)
    radix = [bvn] + [(f"B{b}", dr[f"{pre}_{b}_perms"].mean(), dr[f"{pre}_{b}_cycle"].mean(),
                      dr[f"{pre}_{b}_time"].mean()*1000) for b in [16, 8, 4, 2]]
    eb = ("BvN", de["bvn_perms"].mean(), de["bvn_cycle"].mean(), de["bvn_runtime"].mean()*1000)
    eul = [eb] + [(f"d{d}", de[f"euler_heuristic_depth{d}_perms"].mean(),
                   de[f"euler_heuristic_depth{d}_cycle"].mean(),
                   (de[f"euler_heuristic_depth{d}_time"].mean()
                    + de[f"euler_heuristic_depth{d}_split"].mean())*1000) for d in [1, 2, 3, 4]]
    return radix, eul

# ---------- FIG 1: Euler trade-off (TOTAL runtime) ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
S_ref = None
for name, re, pre, col, mk in ENG:
    _, eul = points(name, re, pre)
    N = [p[1] for p in eul]; rt = [p[3] for p in eul]
    C = [p[2] for p in eul]; depth = [0, 1, 2, 3, 4]
    if name == "Hungarian":
        S_ref = eul[0][2]  # strong-engine BvN cycle = S (the optimal cycle length)
    ax1.plot(N, rt, "-", color=col, lw=1.5, alpha=.85)
    ax1.plot(N, rt, mk, color=col, ms=8, markeredgecolor="white", markeredgewidth=.7, label=name)
    for lbl, n, c, r in eul:
        ax1.annotate(lbl, (n, r), fontsize=7.5, color=col, xytext=(4, 3), textcoords="offset points")
    # right panel: RUNTIME vs CYCLE LENGTH (companion to the runtime-vs-N view) — strong
    # engines drop straight DOWN at C = S (cycle unchanged as depth grows, runtime falls);
    # only WFA drifts right as its cycle inflates.
    ax2.plot(C, rt, "-", color=col, lw=1.5, alpha=.85)
    ax2.plot(C, rt, mk, color=col, ms=8, markeredgecolor="white", markeredgewidth=.7, label=name)
    for lbl, n, c, r in eul:
        ax2.annotate(lbl, (c, r), fontsize=7.5, color=col, xytext=(4, 3), textcoords="offset points")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("Permutation count  N  (log)")
ax1.set_ylabel("TOTAL compute time  [ms]  (split + max-leaf, log)")
ax1.set_title("(a) Euler trade-off (TOTAL runtime = split + max-leaf)\n(cycle C = S held constant throughout)")
ax1.grid(True, which="both", ls=":", lw=.4, alpha=.6); ax1.legend(title="Leaf engine", fontsize=9, framealpha=.95)
ax2.set_yscale("log")
if S_ref is not None:
    ax2.axvline(S_ref, color="gray", ls="--", lw=1.2, zorder=1,
                label=f"$C=S={S_ref:.0f}$ (optimal)")
ax2.set_xlabel("Cycle length  C")
ax2.set_ylabel("TOTAL compute time  [ms]  (split + max-leaf, log)")
ax2.set_title("(b) Runtime vs cycle length\n(strong engines drop straight down at $C=S$; only WFA drifts right)")
ax2.grid(True, which="both", ls=":", lw=.4, alpha=.6); ax2.legend(fontsize=9, framealpha=.95)
fig.suptitle("Euler Splitting — TOTAL-runtime trade-off (depths BvN-d4)    |    N=256, k=256, Wmax=64, dense",
             fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/euler_tradeoff_corrected.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/euler_tradeoff_corrected.png", dpi=150, bbox_inches="tight")
plt.close(fig); print("saved euler_tradeoff_corrected (TOTAL runtime)")

# ---------- FIG 2: design-space projections (TOTAL runtime) ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
proj = [("N (configs)", "C (cycle)", 1, 2, "log", None),
        ("N (configs)", "T_compute (ms, total)", 1, 3, "log", "log"),
        ("C (cycle)", "T_compute (ms, total)", 2, 3, None, "log")]
data = {name: (col, mk, *points(name, re, pre)) for name, re, pre, col, mk in ENG}
for ax, (xl, yl, ix, iy, xs, ys) in zip(axes, proj):
    for name, (col, mk, radix, eul) in data.items():
        ax.plot([p[ix] for p in radix], [p[iy] for p in radix], "-o", color=col, lw=1.4, ms=6,
                markeredgecolor="white", markeredgewidth=.6)
        ax.plot([p[ix] for p in eul], [p[iy] for p in eul], "--^", color=col, lw=1.4, ms=6,
                markeredgecolor="white", markeredgewidth=.6)
        ax.scatter([radix[0][ix]], [radix[0][iy]], color=col, marker="*", s=240, edgecolor="k", lw=.6, zorder=6)
    if xs: ax.set_xscale(xs)
    if ys: ax.set_yscale(ys)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, which="both", ls=":", lw=.4, alpha=.6)
leg = [Line2D([], [], color=c, marker="*", ls="", ms=12, label=n) for n, (c, *_) in data.items()]
leg += [Line2D([], [], color="gray", ls="-", marker="o", label="Radix bases (B16->B2)"),
        Line2D([], [], color="gray", ls="--", marker="^", label="Euler depths (d1->d4)"),
        Line2D([], [], color="gray", marker="*", ls="", ms=12, label="BvN baseline")]
axes[0].legend(handles=leg, fontsize=8, loc="upper left", framealpha=.95)
fig.suptitle("Design space (N, C, T_compute) — TOTAL runtime (Euler = split + max-leaf)   |   "
             "N=256, k=256, Wmax=64, dense", fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/design_space_projections.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/design_space_projections.png", dpi=150, bbox_inches="tight")
plt.close(fig); print("saved design_space_projections (TOTAL runtime)")
