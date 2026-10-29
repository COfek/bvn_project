"""Winner maps sensitivity sweep:
  - y axis: ratio r = T_unit/T_config starting at 5 (up to 1000)
  - x axis: T_config over realistic optical-switch values (1 us .. 1 ms)
  - three panels: T_compute full, /8 and /64, enough to show the trend; the
    finer sweep (/4 .. /128 and 0) is still printed to stdout for the text
Configs: all 4 engines x {BvN, B16..B2, Euler d1..d4}; Euler T_compute split-inclusive.
"""
from _paths import PLOTS, REPO  # repository-relative paths
import glob
import numpy as np
import pandas as pd
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# Chapter 6's tables and these figures must agree, so both read the
# three-sweep variance campaign rather than the older rerun_2026.
from variance_data import radix_frame, euler_frame, EULER_WINDOW
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
def _tfmt(v, _p):
    if v >= 1: return f"{v:g} ms"
    if v >= 1e-3: return f"{v*1e3:g} µs"
    return f"{v*1e6:g} ns"
TFMT = FuncFormatter(_tfmt)

plt.rcParams.update({"font.family": "serif", "font.size": 11})
ROOT = str(REPO)
OUT = str(PLOTS)
ENG = [("Hungarian", "max", "maximum"), ("WFA", "wfa", "wfa"),
       ("GW Dynamic", "heavy", "heavy"), ("GW Static", "heavy_static", "heavy_static")]

def rerun(e): return sorted(glob.glob(f"{ROOT}/run/rerun_2026/{e}/*/results_stats.csv"))[-1]
def euler(p): return sorted(glob.glob(f"{ROOT}/euler_exp/eng_{p}_dense_d4/*/results_stats.csv"))[-1]

configs = {}
for name, re_, pre in ENG:
    dr = radix_frame(re_, rows=EULER_WINDOW)
    for b in [16, 8, 4, 2]:
        configs[f"{name} B{b}"] = (dr[f"{pre}_{b}_perms"].mean(), dr[f"{pre}_{b}_cycle"].mean(), dr[f"{pre}_{b}_time"].mean()*1000)
    de = euler_frame(pre)
    # BvN from the SAME run as Euler (same matrices) so BvN-vs-Euler is artifact-free
    configs[f"{name} BvN"] = (de["bvn_perms"].mean(), de["bvn_cycle"].mean(), de["bvn_runtime"].mean()*1000)
    for d in [1, 2, 3, 4]:
        tc = (de[f"euler_heuristic_depth{d}_time"].mean() + de[f"euler_heuristic_depth{d}_split"].mean())*1000
        configs[f"{name} Euler d{d}"] = (de[f"euler_heuristic_depth{d}_perms"].mean(), de[f"euler_heuristic_depth{d}_cycle"].mean(), tc)
labels = list(configs)
NCT = np.array([configs[l] for l in labels])

tcfg = np.logspace(-3, 0, 220)                 # 1 us .. 1 ms
rgrid = np.logspace(np.log10(5), 3, 220)       # ratio 5 .. 1000
TC, R = np.meshgrid(tcfg, rgrid)

# Fine sweep of the computation-time budget; only the PANELS are drawn, the rest
# are printed so the shares quoted in the text can be checked.
PANELS = (1.0, 0.125, 1/64.)
SCALES = [
    (1.0,     r"$T_{compute}$ (full)"),
    (0.25,    r"$T_{compute}/4$"),
    (0.125,   r"$T_{compute}/8$"),
    (1/16.,   r"$T_{compute}/16$"),
    (1/32.,   r"$T_{compute}/32$"),
    (1/64.,   r"$T_{compute}/64$"),
    (1/128.,  r"$T_{compute}/128$"),
    (0.0,     r"$T_{compute}=0$ (prescheduled)"),
]
bvn_i = labels.index("Hungarian BvN")
wins = []
for s, _ in SCALES:
    allD = np.stack([NCT[i, 2]*s + TC*(NCT[i, 0] + NCT[i, 1]*R) for i in range(len(labels))])
    wins.append(allD.argmin(axis=0))

winners_all = sorted(set(np.concatenate([w.ravel() for w in wins]).tolist()))
cmap_cols = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999", "#000000", "#882255"]
remap = {w: i for i, w in enumerate(winners_all)}
cmap = ListedColormap(cmap_cols[:len(winners_all)])
norm = BoundaryNorm(np.arange(-0.5, len(winners_all)+0.5), cmap.N)

for w, (s, ttl) in zip(wins, SCALES):
    share = {labels[x]: (w == x).mean()*100 for x in sorted(set(w.ravel().tolist()))}
    print(ttl, {k: f"{v:.0f}%" for k, v in share.items()})

keep = [i for i, (s, _) in enumerate(SCALES) if s in PANELS]
wins = [wins[i] for i in keep]
SCALES = [SCALES[i] for i in keep]
winners_all = sorted(set(np.concatenate([w.ravel() for w in wins]).tolist()))
remap = {w: i for i, w in enumerate(winners_all)}
cmap = ListedColormap(cmap_cols[:len(winners_all)])
norm = BoundaryNorm(np.arange(-0.5, len(winners_all)+0.5), cmap.N)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
for ax, w, (s, ttl) in zip(axes, wins, SCALES):
    ax.pcolormesh(TC, R, np.vectorize(remap.get)(w), cmap=cmap, norm=norm, shading="auto")
    ax.axhline(10, color="k", ls="--", lw=1.1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(TFMT)
    bvn_pct = (w == bvn_i).mean() * 100
    ax.set_title(f"{ttl}\nHungarian BvN wins {bvn_pct:.0f}%", fontsize=10.5,
                 color=("#0072B2" if bvn_pct > 0.5 else "black"),
                 fontweight=("bold" if bvn_pct > 0.5 else "normal"))
axes[0].set_ylabel("ratio  $r = T_{unit}/T_{config}$")
for ax in axes:
    ax.set_xlabel("$T_{config}$")
handles = [Patch(color=cmap_cols[remap[w]], label=labels[w]) for w in winners_all] \
        + [plt.Line2D([], [], color="k", ls="--", lw=1.1, label="$r=10$")]
fig.legend(handles=handles, title="Winning configuration", ncol=min(len(handles), 5),
           loc="lower center", fontsize=9, title_fontsize=10, framealpha=0.95,
           bbox_to_anchor=(0.5, -0.02))
# [point 27] title moved to caption: fig.suptitle("Winner map vs the computation-time budget ($T_{config}$: 1$\\mu$s--1ms, $r\\geq5$) --- "
# [point 27] title moved to caption: "the plain Hungarian BvN region emerges gradually as $T_{compute}$ shrinks\n"
# [point 27] title moved to caption: "Dense, n=256, k=256, $W_{max}$=64; BvN cycle-consistent with Euler (same run, $C=S$)",
# [point 27] title moved to caption: fontsize=13, y=1.005)
fig.tight_layout(rect=[0, 0.1, 1, 0.99])
fig.savefig(f"{OUT}/winner_map_tcalc_sensitivity.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/winner_map_tcalc_sensitivity.png", dpi=140, bbox_inches="tight")
print("saved winner_map_tcalc_sensitivity")
