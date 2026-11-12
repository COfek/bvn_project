"""Wide two-column trade-off figures for the paper.

Both are single-column (not figure*) so LaTeX can place them inside the subsection.
Each has two side-by-side panels giving computation time against BOTH other metrics,
scaled to fit within a single column horizontally.

  scaled_tradeoff.pdf       Radix(B,c), scaled only  -- for "Radix across engines"
  euler_tradeoff_paper.pdf  Euler depths             -- for "Euler splitting across engines"
"""
from __future__ import annotations
from _paths import PLOTS, REPO  # repository-relative paths
import os, glob
import numpy as np
import pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from radix_extraction import correct_summary, correct_scale_results  # noqa: E402
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# same campaign as Chapter 6's tables and Chapter 7's figures
from variance_data import CAMPAIGN, radix_frame, euler_frame
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif"})
ROOT = str(REPO)
PLOTS = str(PLOTS)
ENG = [("Hungarian", "#0072B2", "o"), ("GW Dynamic", "#D55E00", "s"),
       ("GW Static", "#009E73", "^"), ("WFA", "#CC79A7", "D")]
# the workloads of a rerun live under BVN_RUN_ROOT (default run/), as their campaign scripts write them
RUN = os.environ.get("BVN_RUN_ROOT", "run")
if CAMPAIGN != "variance_2026":   # the November rerun: WWFA is the weak engine, in place of WFA
    ENG = [("WWFA", c, m) if n == "WFA" else (n, c, m) for n, c, m in ENG]
BASES = [16, 8, 4, 2]
DEPTHS = [1, 2, 3, 4]


def _style(ax, xlab, ylab, logx=False, logy=True):
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlab, fontsize=5.5, labelpad=1)
    ax.set_ylabel(ylab, fontsize=5.5, labelpad=1)
    ax.tick_params(labelsize=5, pad=1)
    ax.grid(True, which="both", ls=":", lw=.35, alpha=.6)


# ============================== RADIX (scaled) ==============================
# a rerun (BVN_RUN_ROOT) times the digit extraction itself; only run/ needs it added
res = pd.read_csv(os.path.join(ROOT, RUN, "radix_scale", "scale_results.csv"))
if RUN == "run":
    res = correct_scale_results(res)
res["S_original"] = res.S_scaled / res.scale
res["C_orig"] = res.C_over_S * res.S_original
bvn = pd.read_csv(os.path.join(ROOT, RUN, "tm_workload", "tm_summary.csv"))
r = res[res.regime == "Uniform"]
b = bvn[bvn.regime == "Uniform (thesis)"]

# stacked, one column wide, so the figure can be pinned where the text needs it
fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.6))
# C = S line at the mean demand of the matrices the Radix points average over (before, the
# first matrix's S alone)
S0 = float(r.groupby("matrix").S_original.first().mean())
for name, col, mk in ENG:
    # The tight scale depends on the matrix's largest entry, so each matrix has its own: pick,
    # per matrix, the scale with the shortest cycle (c=1 or one of the two tight scales), then
    # average over the matrices, so every point uses all of them. (Picking the (base, scale)
    # group with the lowest mean cycle selected single matrices, those with the smallest S.)
    e = r[r.engine == name]
    best = e.loc[e.groupby(["base", "matrix"]).C_over_S.idxmin()]
    tu = (best.groupby("base")
          .agg(C=("C_orig", "mean"), N=("N", "mean"), T=("runtime_ms", "mean")).reindex(BASES))
    bv = b[(b.engine == name) & (b.config == "BvN")]
    bC, bN, bT = float(bv.C.iloc[0]), float(bv.N.iloc[0]), float(bv.runtime_ms.iloc[0])
    Cs, Ns, Ts = [bC] + tu.C.tolist(), [bN] + tu.N.tolist(), [bT] + tu["T"].tolist()
    # (a) time against cycle, (b) time against perms, (c) the two against each other
    for ax, xs, ys in ((axes[0], Cs, Ts), (axes[1], Ns, Ts), (axes[2], Ns, Cs)):
        ax.plot(xs, ys, "-", color=col, lw=1.3, alpha=.95)
        ax.plot(xs, ys, mk, color=col, ms=4.5, markeredgecolor="white",
                markeredgewidth=.5, label=name)
    # the unscaled counterpart, scale = 1, drawn faint so the correction the text
    # claims is visible rather than asserted. Without it the reader is asked to
    # compare against something the figure does not show.
    gu = (r[(r.engine == name) & (r.scale == 1)].groupby('base')
          .agg(C=('C_orig', 'mean'), N=('N', 'mean'), T=('runtime_ms', 'mean'))
          .reindex(BASES))
    Cu, Nu, Tu_ = [bC] + gu.C.tolist(), [bN] + gu.N.tolist(), [bT] + gu['T'].tolist()
    for ax, xs_, ys_ in ((axes[0], Cu, Tu_), (axes[1], Nu, Tu_), (axes[2], Nu, Cu)):
        ax.plot(xs_, ys_, '--', color=col, lw=0.9, alpha=.35, zorder=1)
_style(axes[0], "cycle length $C$", "$T_{compute}$ [ms]")
_style(axes[1], "permutation count $N$", "$T_{compute}$ [ms]", logx=True)
_style(axes[2], "permutation count $N$", "cycle length $C$", logx=True, logy=False)
axes[0].axvline(S0, color="gray", ls=":", lw=1.0)
axes[2].axhline(S0, color="gray", ls=":", lw=1.0)
axes[0].set_title("(a) against cycle", fontsize=6.5)
axes[1].set_title("(b) against perms", fontsize=6.5)
axes[2].set_title("(c) cycle against perms", fontsize=6.5)
# Legend in panel (b), lower right: that corner is empty, whereas panel (a) upper
# right sits on top of the Hungarian and GW Dynamic trajectories.
axes[1].legend(fontsize=5.2, framealpha=.95, loc="lower right")
fig.tight_layout(pad=0.2)
for e in ("pdf", "png"):
    fig.savefig(os.path.join(PLOTS, f"scaled_tradeoff.{e}"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved scaled_tradeoff (single column, 2 panels)")

# ============================== EULER ==============================
def rerun(e): return sorted(glob.glob(f"{ROOT}/run/rerun_2026/{e}/*/results_stats.csv"))[-1]
def euler(p): return sorted(glob.glob(f"{ROOT}/euler_exp/eng_{p}_dense_d4/*/results_stats.csv"))[-1]
KEYS = [("Hungarian", "max", "maximum"), ("GW Dynamic", "heavy", "heavy"),
        ("GW Static", "heavy_static", "heavy_static"), ("WFA", "wfa", "wfa")]
COL = {n: c for n, c, _ in ENG}
MK = {n: m for n, _, m in ENG}
if CAMPAIGN != "variance_2026":
    KEYS = [("WWFA", "wwfa", "wwfa") if k[1] == "wfa" else k for k in KEYS]

fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8))
fig.subplots_adjust(wspace=0.35, bottom=0.25)
S_ref = None
for name, re_, pre in KEYS:
    de = euler_frame(pre)
    N = [de["bvn_perms"].mean()] + [de[f"euler_heuristic_ruined_depth{d}_perms"].mean() for d in DEPTHS]
    C = [de["bvn_cycle"].mean()] + [de[f"euler_heuristic_ruined_depth{d}_cycle"].mean() for d in DEPTHS]
    T = [de["bvn_runtime"].mean()*1000] + [
        (de[f"euler_heuristic_ruined_depth{d}_time"].mean()
         + de[f"euler_heuristic_ruined_depth{d}_split"].mean())*1000 for d in DEPTHS]
    if name == "Hungarian":
        S_ref = C[0]
    for ax, xs in ((axes[0], N), (axes[1], C)):
        ax.plot(xs, T, "-", color=COL[name], lw=1.3, alpha=.95)
        ax.plot(xs, T, MK[name], color=COL[name], ms=4.5,
                markeredgecolor="white", markeredgewidth=.5, label=name)
_style(axes[0], "permutation count $N$", "$T_{compute}$ [ms]", logx=True)
_style(axes[1], "cycle length $C$", "$T_{compute}$ [ms]")
axes[1].axvline(S_ref, color="gray", ls=":", lw=1.0)
axes[0].set_title("(a) against perms", fontsize=6)
axes[1].set_title("(b) against cycle", fontsize=6)
axes[1].legend(fontsize=4.5, framealpha=.95, loc="lower right")
fig.tight_layout(pad=0.2)
for e in ("pdf", "png"):
    fig.savefig(os.path.join(PLOTS, f"euler_tradeoff_paper.{e}"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"saved euler_tradeoff_paper (single column, 2 panels)  C=S={S_ref:.0f}")
