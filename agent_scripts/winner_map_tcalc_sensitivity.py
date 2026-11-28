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
from variance_data import CAMPAIGN, radix_frame, euler_frame, EULER_WINDOW
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from winner_colors import color_of, label_regions
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
if CAMPAIGN != "variance_2026":   # the November rerun: WWFA is the weak engine, in place of WFA
    ENG = [("WWFA", "wwfa", "wwfa") if e[1] == "wfa" else e for e in ENG]

def rerun(e): return sorted(glob.glob(f"{ROOT}/run/rerun_2026/{e}/*/results_stats.csv"))[-1]
def euler(p): return sorted(glob.glob(f"{ROOT}/euler_exp/eng_{p}_dense_d4/*/results_stats.csv"))[-1]

configs = {}
for name, re_, pre in ENG:
    dr = radix_frame(re_, rows=EULER_WINDOW)
    for b in [16, 8, 4, 2]:
        configs[f"{name} B{b}"] = (dr[f"{pre}_{b}_perms"].mean(), dr[f"{pre}_{b}_cycle"].mean(), dr[f"{pre}_{b}_time"].mean()*1000)
    de = euler_frame(pre)
    # BvN from the SAME run as Euler (same matrices) so BvN-vs-Euler is artifact-free
    configs[f"{name} unsplit"] = (de["bvn_perms"].mean(), de["bvn_cycle"].mean(), de["bvn_runtime"].mean()*1000)
    for d in [1, 2, 3, 4]:
        tc = (de[f"euler_heuristic_ruined_depth{d}_time"].mean() + de[f"euler_heuristic_ruined_depth{d}_split"].mean())*1000
        configs[f"{name} Euler d{d}"] = (de[f"euler_heuristic_ruined_depth{d}_perms"].mean(), de[f"euler_heuristic_ruined_depth{d}_cycle"].mean(), tc)
labels = list(configs)
NCT = np.array([configs[l] for l in labels])

tcfg = np.logspace(-3, 0, 220)                 # 1 us .. 1 ms
rgrid = np.logspace(np.log10(5), 3, 220)       # ratio 5 .. 1000
TC, R = np.meshgrid(tcfg, rgrid)

# Fine sweep of the computation-time budget; only the PANELS are drawn, the rest
# are printed so the shares quoted in the text can be checked.
PANELS = (1.0, 0.125, 1/64.)
SCALES = [
    (1.0,     r"$T_{\mathrm{parallel}}$ (full)"),
    (0.25,    r"$T_{\mathrm{parallel}}/4$"),
    (0.125,   r"$T_{\mathrm{parallel}}/8$"),
    (1/16.,   r"$T_{\mathrm{parallel}}/16$"),
    (1/32.,   r"$T_{\mathrm{parallel}}/32$"),
    (1/64.,   r"$T_{\mathrm{parallel}}/64$"),
    (1/128.,  r"$T_{\mathrm{parallel}}/128$"),
    (0.0,     r"$T_{\mathrm{parallel}}=0$ (prescheduled)"),
]
bvn_i = labels.index("Hungarian unsplit")
wins = []
for s, _ in SCALES:
    allD = np.stack([NCT[i, 2]*s + TC*(NCT[i, 0] + NCT[i, 1]*R) for i in range(len(labels))])
    wins.append(allD.argmin(axis=0))

winners_all = sorted(set(np.concatenate([w.ravel() for w in wins]).tolist()))
remap = {w: i for i, w in enumerate(winners_all)}
cmap = ListedColormap([color_of(labels[w]) for w in winners_all])
norm = BoundaryNorm(np.arange(-0.5, len(winners_all)+0.5), cmap.N)

for w, (s, ttl) in zip(wins, SCALES):
    share = {labels[x]: (w == x).mean()*100 for x in sorted(set(w.ravel().tolist()))}
    print(ttl, {k: f"{v:.0f}%" for k, v in share.items()})

keep = [i for i, (s, _) in enumerate(SCALES) if s in PANELS]
wins = [wins[i] for i in keep]
SCALES = [SCALES[i] for i in keep]
winners_all = sorted(set(np.concatenate([w.ravel() for w in wins]).tolist()))
remap = {w: i for i, w in enumerate(winners_all)}
cmap = ListedColormap([color_of(labels[w]) for w in winners_all])
norm = BoundaryNorm(np.arange(-0.5, len(winners_all)+0.5), cmap.N)

# stacked, one column wide, so the figure fits the column under its subsection heading
# Drawn on a finer grid than the 220 x 220 one the shares are computed on, as one raster image
# without antialiasing between cells: at 220 cells the boundaries showed as steps and the cell
# seams as faint white lines when zoomed in (Ofek, Chen K15).
FINE = 1200
TCf, Rf = np.meshgrid(np.logspace(-3, 0, FINE), np.logspace(np.log10(5), 3, FINE))
# In the style of Figures 3-7 and 8 (2026-11-28, Ofek: every figure in the same format): LaTeX
# text in the paper's font at 9 pt, printed at its own size (3.33 in wide, fixed margins, no bbox
# cropping), setups named as the paper names them, titles all in the same weight.
from winner_colors import PAPER_RC, paper_name, tex_time, COLORS
plt.rcParams.update(PAPER_RC)
SCALES = [(s, {1.0: r"measured $T_{\mathrm{parallel}}$", 0.125: r"$T_{\mathrm{parallel}}/8$",
               1/64.: r"$T_{\mathrm{parallel}}/64$"}[s]) for s, _ in SCALES]
W_IN, L_IN, R_IN, MAP_H, GAP_IN, TOP_IN, BOT_IN = 3.33, 0.62, 0.16, 1.2, 0.33, 0.28, 0.42
H_IN = TOP_IN + 3 * MAP_H + 2 * GAP_IN + BOT_IN
fig = plt.figure(figsize=(W_IN, H_IN))
axes = []
for j in range(3):
    b = BOT_IN + (2 - j) * (MAP_H + GAP_IN)
    axes.append(fig.add_axes([L_IN / W_IN, b / H_IN, (W_IN - L_IN - R_IN) / W_IN, MAP_H / H_IN],
                             sharex=axes[0] if axes else None, sharey=axes[0] if axes else None))
for ax, w, (s, ttl) in zip(axes, wins, SCALES):
    wf = np.stack([NCT[i, 2]*s + TCf*(NCT[i, 0] + NCT[i, 1]*Rf)
                   for i in range(len(labels))]).argmin(axis=0)
    assert set(wf.ravel().tolist()) <= set(winners_all), "the fine grid finds another winner"
    print("fine grid (drawing only)", ttl, {labels[x]: f"{(wf == x).mean()*100:.1f}%"
                                            for x in sorted(set(wf.ravel().tolist()))})
    ax.pcolormesh(TCf, Rf, np.vectorize(remap.get)(wf), cmap=cmap, norm=norm, shading="auto",
                  rasterized=True, antialiased=False, linewidth=0)
    label_regions(ax, np.vectorize(remap.get)(w), TC, R,
                  {remap[x]: labels[x] for x in winners_all}, fontsize=6.5, min_share=0.015,
                  short=paper_name)
    ax.axhline(10, color="k", ls="--", lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(tex_time))
    ax.tick_params(pad=1.5)
    if ax is not axes[-1]:
        ax.tick_params(labelbottom=False)
    bvn_pct = (w == bvn_i).mean() * 100
    ax.set_title(f"{ttl}: unsplit Hungarian wins {bvn_pct:.0f}\\%", pad=3)
axes[-1].set_xlabel(r"$T_{\mathrm{config}}$", labelpad=1)
fig.text(0.08 / W_IN, (BOT_IN + 1.5 * MAP_H + GAP_IN) / H_IN,
         r"ratio $r = T_{\mathrm{unit}}/T_{\mathrm{config}}$", rotation=90, ha="left", va="center")
# no legend: every region carries its setup's name, and the caption says what the dashed line is
# [point 27] title moved to caption: fig.suptitle("Winner map vs the computation-time budget ($T_{config}$: 1$\\mu$s--1ms, $r\\geq5$) --- "
# [point 27] title moved to caption: "the plain Hungarian BvN region emerges gradually as $T_{compute}$ shrinks\n"
# [point 27] title moved to caption: "Dense, n=256, k=256, $W_{max}$=64; BvN cycle-consistent with Euler (same run, $C=S$)",
# [point 27] title moved to caption: fontsize=13, y=1.005)
fig.savefig(f"{OUT}/winner_map_tcalc_sensitivity.pdf", dpi=600)
fig.savefig(f"{OUT}/winner_map_tcalc_sensitivity.png", dpi=140)
print("saved winner_map_tcalc_sensitivity")
