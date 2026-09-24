"""DCT parameter-space analysis, rebuilt from the FRESH complete data.

DCT = T_compute + N*T_config + C*T_unit   (all in ms)

Candidate configurations: for every engine {Hungarian, WFA, GW Dynamic, GW Static}
  - BvN baseline
  - Radix B16, B8, B4, B2
  - Euler depths d1, d2, d3
each carrying its own (N, C, T_compute) from run/rerun_2026 (radix+BvN) and
euler_exp (Euler, dense).

Outputs (paper/plots_pdf/):
  dct_param_space_2d.pdf   : ratio heatmap + winner map  (T_config x T_unit)
  dct_3d_surfaces.pdf      : two-surface crossover + minimum-DCT surface
"""
from __future__ import annotations
import glob, numpy as np, pandas as pd
import sys, os as _os
_os.sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# Chapter 6's tables and these figures must agree, so both read the
# three-sweep variance campaign rather than the older rerun_2026.
from variance_data import radix_frame, euler_frame, EULER_WINDOW
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter

def _tfmt(v, _p):
    if v >= 1: return f"{v:g} ms"
    if v >= 1e-3: return f"{v*1e3:g} µs"
    return f"{v*1e6:g} ns"
TFMT = FuncFormatter(_tfmt)

ROOT = "C:/Users/ofekc/Desktop/Msc/Thesis/bvn_project"
OUT = "C:/Users/ofekc/Desktop/Msc/Thesis/paper/plots_pdf"
ENG = [("Hungarian", "max", "maximum"), ("WFA", "wfa", "wfa"),
       ("GW Dynamic", "heavy", "heavy"), ("GW Static", "heavy_static", "heavy_static")]

def rerun(e): return sorted(glob.glob(f"{ROOT}/run/rerun_2026/{e}/*/results_stats.csv"))[-1]
def euler(p): return sorted(glob.glob(f"{ROOT}/euler_exp/eng_{p}_dense_d4/*/results_stats.csv"))[-1]
EULER_DEPTHS = [1, 2, 3, 4]

# ---- gather candidate configs: label -> (N, C, T_compute_ms) ----
configs = {}
for name, re, pre in ENG:
    dr = radix_frame(re, rows=EULER_WINDOW)
    configs[f"{name} BvN"] = (dr["bvn_perms"].mean(), dr["bvn_cycle"].mean(), dr["bvn_runtime"].mean()*1000)
    for b in [16, 8, 4, 2]:
        configs[f"{name} B{b}"] = (dr[f"{pre}_{b}_perms"].mean(), dr[f"{pre}_{b}_cycle"].mean(), dr[f"{pre}_{b}_time"].mean()*1000)
    de = euler_frame(pre)
    for d in EULER_DEPTHS:
        # T_compute for Euler = split phase + max-leaf parallel extraction (the
        # split is NOT free and grows with depth; Radix's digit extraction is ~0).
        tcalc = (de[f"euler_heuristic_depth{d}_time"].mean()
                 + de[f"euler_heuristic_depth{d}_split"].mean()) * 1000
        configs[f"{name} Euler d{d}"] = (de[f"euler_heuristic_depth{d}_perms"].mean(),
                                          de[f"euler_heuristic_depth{d}_cycle"].mean(),
                                          tcalc)
labels = list(configs)
NCT = np.array([configs[l] for l in labels])  # (M,3): N, C, Tcalc

# ---- parameter grid ----
tcfg = np.logspace(-4, 0, 240)   # ms
tunit = np.logspace(-3, 1, 240)  # ms
TC, TU = np.meshgrid(tcfg, tunit)

def dct(idx):
    N, C, Tcalc = NCT[idx]
    return Tcalc + N*TC + C*TU

# ---- winner at each grid point ----
allD = np.stack([dct(i) for i in range(len(labels))], axis=0)  # (M, ny, nx)
win = allD.argmin(axis=0)
winners = sorted(set(win.ravel().tolist()))
print("Winning configs across the grid:")
for w in winners:
    frac = (win == w).mean()
    print(f"  {labels[w]:24s}  wins {frac*100:5.1f}% of the plane")

# pick the two dominant winners for the ratio / two-surface panels
areas = [( (win==w).mean(), w ) for w in winners]
areas.sort(reverse=True)
A = areas[0][1]; B = areas[1][1]
print(f"\nDominant pair: A={labels[A]}  vs  B={labels[B]}")

# ================= FIGURE 1: 2-D ratio + winner map =================
fig, (axr, axw) = plt.subplots(1, 2, figsize=(15, 6))
ratio = dct(A) / dct(B)
pc = axr.pcolormesh(TC, TU, ratio, cmap="RdYlGn_r", vmin=0.5, vmax=2.0, shading="auto")
axr.contour(TC, TU, ratio, levels=[1.0], colors="k", linewidths=1.2)
fig.colorbar(pc, ax=axr, label=f"DCT ratio  ({labels[A]} / {labels[B]})")
axr.set_xscale("log"); axr.set_yscale("log")
axr.set_xlabel("$T_{config}$"); axr.set_ylabel("$T_{unit}$")
# [point 27] panel title moved to caption: axr.set_title(f"Ratio: {labels[A]} vs {labels[B]}\n(green = {labels[A]} wins, red = {labels[B]} wins)")

# winner map (only dominant winners, discrete colors)
cmap_cols = ["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7","#999999","#000000","#882255"]
remap = {w:i for i,w in enumerate(winners)}
winimg = np.vectorize(remap.get)(win)
cmap = ListedColormap(cmap_cols[:len(winners)])
norm = BoundaryNorm(np.arange(-0.5, len(winners)+0.5), cmap.N)
axw.pcolormesh(TC, TU, winimg, cmap=cmap, norm=norm, shading="auto")
axw.set_xscale("log"); axw.set_yscale("log")
axw.set_xlabel("$T_{config}$"); axw.set_ylabel("$T_{unit}$")
# [point 27] panel title moved to caption: axw.set_title("Winner map (lowest-DCT configuration)")
from matplotlib.patches import Patch
axw.legend(handles=[Patch(color=cmap_cols[remap[w]], label=labels[w]) for w in winners],
           fontsize=8, loc="upper left", framealpha=0.95)

for ax in (axr, axw):
    ax.plot(tcfg, 10*tcfg, "k--", lw=1.4, label="$T_{unit}=10\\,T_{config}$")
    ax.axhline(0.01, color="navy", ls=":", lw=1.4, label="Paper: $T_{unit}=0.01$ ms")
    ax.set_xlim(tcfg.min(), tcfg.max()); ax.set_ylim(tunit.min(), tunit.max())
from matplotlib.lines import Line2D
axr.legend(handles=[
    Line2D([], [], color="k", ls="-", lw=1.2, label="break-even (DCT ratio = 1)"),
    Line2D([], [], color="k", ls="--", lw=1.4, label="$T_{unit}=10\\,T_{config}$"),
    Line2D([], [], color="navy", ls=":", lw=1.4, label="Paper: $T_{unit}=0.01$ ms"),
], fontsize=8, loc="lower right", framealpha=0.9)
# [point 27] title moved to caption: fig.suptitle("DCT Parameter Space — Dense (n=256, k=256, $W_{max}$=64)   |   "
# [point 27] title moved to caption: "DCT = $T_{compute}+N\\,T_{config}+C\\,T_{unit}$", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/dct_param_space_2d.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/dct_param_space_2d.png", dpi=140, bbox_inches="tight")
print("saved dct_param_space_2d")

# ===== FIGURE 1b: y-axis re-parameterised as ratio r = T_unit / T_config =====
# At (T_config, r) the actual T_unit = r * T_config, so
#   DCT = T_compute + N*T_config + C*(r*T_config) = T_compute + T_config*(N + C*r)
rgrid = np.logspace(np.log10(5), 3, 240)   # ratio 5 .. 1000 (config-dominated r<5 excluded)
tcfgr = np.logspace(-3, 0, 240)            # realistic T_config: 1 us .. 1 ms
TCr, R = np.meshgrid(tcfgr, rgrid)
def dct_r(idx):
    N, C, Tcalc = NCT[idx]
    return Tcalc + TCr * (N + C * R)
allDr = np.stack([dct_r(i) for i in range(len(labels))], axis=0)
winr = allDr.argmin(axis=0)
winners_r = sorted(set(winr.ravel().tolist()))
print("\nWinners in the ratio-parameterised plane (r = T_unit/T_config up to 1000):")
for w in winners_r:
    print(f"  {labels[w]:24s}  wins {(winr==w).mean()*100:5.1f}%")

figR, (axrr, axrw) = plt.subplots(1, 2, figsize=(15, 6))
ratioR = dct_r(A) / dct_r(B)
pcr = axrr.pcolormesh(TCr, R, ratioR, cmap="RdYlGn_r", vmin=0.5, vmax=2.0, shading="auto")
axrr.contour(TCr, R, ratioR, levels=[1.0], colors="k", linewidths=1.2)
figR.colorbar(pcr, ax=axrr, label=f"DCT ratio  ({labels[A]} / {labels[B]})")
# [point 27] panel title moved to caption: axrr.set_title(f"Ratio: {labels[A]} vs {labels[B]}\n(green = {labels[A]} wins, red = {labels[B]} wins)")

remap_r = {w: i for i, w in enumerate(winners_r)}
cmap_r = ListedColormap(cmap_cols[:len(winners_r)])
norm_r = BoundaryNorm(np.arange(-0.5, len(winners_r)+0.5), cmap_r.N)
axrw.pcolormesh(TCr, R, np.vectorize(remap_r.get)(winr), cmap=cmap_r, norm=norm_r, shading="auto")
# [point 27] panel title moved to caption: axrw.set_title("Winner map (lowest-DCT configuration)")
for ax in (axrr, axrw):
    ax.axhline(10, color="k", ls="--", lw=1.4, label="$T_{unit}=10\\,T_{config}$")
    ax.plot(tcfgr, 0.01/tcfgr, "navy", ls=":", lw=1.4, label="Paper: $T_{unit}=0.01$ ms")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$T_{config}$"); ax.set_ylabel("ratio  $r = T_{unit}/T_{config}$")
    ax.set_xlim(tcfgr.min(), tcfgr.max()); ax.set_ylim(rgrid.min(), rgrid.max())
    ax.xaxis.set_major_formatter(TFMT)
axrr.legend(fontsize=8, loc="upper left", framealpha=0.9)
axrw.legend(handles=[Patch(color=cmap_cols[remap_r[w]], label=labels[w]) for w in winners_r]
                   + [Line2D([], [], color="k", ls="--", lw=1.4, label="$T_{unit}=10\\,T_{config}$"),
                      Line2D([], [], color="navy", ls=":", lw=1.4, label="Paper: $T_{unit}=0.01$ ms")],
            title="Winning configuration", title_fontsize=9,
            fontsize=8, loc="upper left", framealpha=0.95)
# [point 27] title moved to caption: figR.suptitle("DCT Parameter Space — y-axis as ratio $r=T_{unit}/T_{config}$ (up to 1000)   |   "
# [point 27] title moved to caption: "Dense (n=256, k=256, $W_{max}$=64)", fontsize=13, y=1.02)
figR.tight_layout()
figR.savefig(f"{OUT}/dct_param_space_ratio.pdf", bbox_inches="tight")
figR.savefig(f"{OUT}/dct_param_space_ratio.png", dpi=140, bbox_inches="tight")
print("saved dct_param_space_ratio")

# ===== FIGURE 1c: the winner map alone, sized for a single column =====
# Same data as the right-hand panel above. The ratio panel is dropped: it compares
# one pair of configurations, which the winner map already subsumes.
figW, axW = plt.subplots(figsize=(3.4, 2.7))
axW.pcolormesh(TCr, R, np.vectorize(remap_r.get)(winr), cmap=cmap_r, norm=norm_r, shading="auto")
axW.axhline(10, color="k", ls="--", lw=1.0)
axW.plot(tcfgr, 0.01 / tcfgr, "navy", ls=":", lw=1.0)
axW.set_xscale("log"); axW.set_yscale("log")
axW.set_xlabel("$T_{config}$", fontsize=7)
axW.set_ylabel("ratio  $r = T_{unit}/T_{config}$", fontsize=7)
axW.set_xlim(tcfgr.min(), tcfgr.max()); axW.set_ylim(rgrid.min(), rgrid.max())
axW.xaxis.set_major_formatter(TFMT)
axW.tick_params(labelsize=6)
axW.legend(handles=[Patch(color=cmap_cols[remap_r[w]], label=labels[w]) for w in winners_r]
                  + [Line2D([], [], color="k", ls="--", lw=1.0, label="$T_{unit}=10\\,T_{config}$"),
                     Line2D([], [], color="navy", ls=":", lw=1.0, label="Paper: $T_{unit}=0.01$ ms")],
           fontsize=5.2, loc="upper left", framealpha=0.95, handlelength=1.4, borderpad=0.3)
figW.tight_layout()
figW.savefig(f"{OUT}/dct_winner_map.pdf", bbox_inches="tight")
figW.savefig(f"{OUT}/dct_winner_map.png", dpi=200, bbox_inches="tight")
print("saved dct_winner_map (single column)")


# ================= FIGURE 2: 3-D surfaces =================
# Axes MATCH the ratio winner map (dct_param_space_ratio.pdf):
#   x = T_config (1 us .. 1 ms),  y = ratio r = T_unit/T_config (5 .. 1000),  z = log10 DCT.
from matplotlib.colors import to_rgba
# dominant pair in the SAME ratio plane as the winner map
areas_r = sorted([((winr == w).mean(), w) for w in winners_r], reverse=True)
Ar = areas_r[0][1]; Br = areas_r[1][1]
print(f"Ratio-plane dominant pair: A={labels[Ar]}  vs  B={labels[Br]}")
LTCr, LR = np.log10(TCr), np.log10(R)
_xt = [-3, -2, -1, 0]; _xtl = ["1 µs", "10 µs", "100 µs", "1 ms"]
_yt = [np.log10(5), 1, 2, 3]; _ytl = ["5", "10", "100", "1000"]

# Single panel: the minimum-DCT surface coloured by the winner map (winr / remap_r).
# (The two-surface d4-vs-d3 crossover panel was dropped per author request.)
fig = plt.figure(figsize=(9, 7))
ax2 = fig.add_subplot(111, projection="3d")
minZr = np.log10(allDr.min(axis=0))
facecol = np.empty(winr.shape + (4,))
for w in winners_r:
    facecol[winr == w] = to_rgba(cmap_cols[remap_r[w]])
ax2.plot_surface(LTCr, LR, minZr, facecolors=facecol, alpha=1.0, linewidth=0, antialiased=False, shade=False)
ax2.set_xticks(_xt); ax2.set_xticklabels(_xtl, fontsize=9)
ax2.set_yticks(_yt); ax2.set_yticklabels(_ytl, fontsize=9)
ax2.set_xlabel("$T_{config}$", labelpad=10); ax2.set_ylabel("ratio $r=T_{unit}/T_{config}$", labelpad=10)
ax2.set_zlabel("log10 DCT", labelpad=6)
ax2.view_init(elev=22, azim=-52)
ax2.legend(handles=[Patch(color=cmap_cols[remap_r[w]], label=labels[w]) for w in winners_r],
           title="Winning configuration", title_fontsize=9,
           fontsize=9, loc="upper left", framealpha=0.95)
# [point 27] title moved to caption: fig.suptitle("Minimum-DCT surface, coloured by winning configuration\n"
# [point 27] title moved to caption: "($T_{config}$: 1$\\mu$s--1ms, $r=T_{unit}/T_{config}$: 5--1000)   |   "
# [point 27] title moved to caption: "Dense (n=256, k=256, $W_{max}$=64)", fontsize=12, y=0.98)
fig.tight_layout()
fig.savefig(f"{OUT}/dct_3d_surfaces.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/dct_3d_surfaces.png", dpi=140, bbox_inches="tight")
print("saved dct_3d_surfaces")

# ===== FIGURE 3: extended-T_config winner map — WHERE does plain Hungarian BvN win? =====
# The realistic window (T_config <= 1 ms) is dominated by Euler; plain BvN only wins once
# T_config is large enough that its low permutation count N pays off. We extend T_config
# well past the realistic limit to expose that boundary.
#
# CYCLE-CONSISTENT config set: BvN and Euler are sourced from the SAME euler run (same
# matrices, same S) so BvN is not penalised by the cross-run S mismatch (8174 vs 8211);
# Radix stays from rerun_2026 (it never wins this regime, so the small mismatch is moot).
configs_c = {}
for _name, _re, _pre in ENG:
    _dr = radix_frame(_re, rows=EULER_WINDOW)
    _de = euler_frame(_pre)
    configs_c[f"{_name} BvN"] = (_de["bvn_perms"].mean(), _de["bvn_cycle"].mean(), _de["bvn_runtime"].mean()*1000)
    for _b in [16, 8, 4, 2]:
        configs_c[f"{_name} B{_b}"] = (_dr[f"{_pre}_{_b}_perms"].mean(), _dr[f"{_pre}_{_b}_cycle"].mean(), _dr[f"{_pre}_{_b}_time"].mean()*1000)
    for _d in EULER_DEPTHS:
        _tc = (_de[f"euler_heuristic_depth{_d}_time"].mean() + _de[f"euler_heuristic_depth{_d}_split"].mean())*1000
        configs_c[f"{_name} Euler d{_d}"] = (_de[f"euler_heuristic_depth{_d}_perms"].mean(), _de[f"euler_heuristic_depth{_d}_cycle"].mean(), _tc)
labels = list(configs_c)                       # shadow for this figure onward
NCT = np.array([configs_c[l] for l in labels])

tcfg_ext = np.logspace(-3, 1.7, 320)          # 1 us .. ~50 ms  (realistic limit = 1 ms)
rgrid2 = np.logspace(np.log10(5), 3, 320)     # r = 5 .. 1000
TCe, Re = np.meshgrid(tcfg_ext, rgrid2)
def dct_e(idx):
    N, C, Tcalc = NCT[idx]
    return Tcalc + TCe * (N + C * Re)
allDe = np.stack([dct_e(i) for i in range(len(labels))])
wine = allDe.argmin(axis=0)
winners_e = sorted(set(wine.ravel().tolist()))
print("\nExtended plane (T_config up to ~50 ms) winners [cycle-consistent BvN]:")
for w in winners_e:
    print(f"  {labels[w]:24s}  wins {(wine==w).mean()*100:5.1f}%")
bvn_idx = labels.index("Hungarian BvN")
if (wine == bvn_idx).any():
    # smallest T_config (column) at which Hungarian BvN wins anywhere in the column
    cols = np.where((wine == bvn_idx).any(axis=0))[0]
    tc_first = tcfg_ext[cols.min()]
    print(f"Hungarian BvN first wins at T_config = {tc_first*1e3:.0f} us "
          f"({tc_first:.3g} ms); it wins {(wine==bvn_idx).mean()*100:.1f}% of the extended plane.")
else:
    tc_first = None
    print("Hungarian BvN never wins even in the extended plane.")

figE, axE = plt.subplots(figsize=(9, 6))
remap_e = {w: i for i, w in enumerate(winners_e)}
cmap_e = ListedColormap(cmap_cols[:len(winners_e)])
norm_e = BoundaryNorm(np.arange(-0.5, len(winners_e)+0.5), cmap_e.N)
axE.pcolormesh(TCe, Re, np.vectorize(remap_e.get)(wine), cmap=cmap_e, norm=norm_e, shading="auto")
axE.axvline(1.0, color="k", ls="-", lw=2.0)      # realistic hardware ceiling (1 ms)
axE.text(1.0, rgrid2.max()*0.62, " realistic\n hardware\n limit (1 ms)", fontsize=9, va="top")
if tc_first is not None:
    axE.axvline(tc_first, color="k", ls=":", lw=1.6)
    axE.text(tc_first, rgrid2.min()*1.25, f" BvN starts\n {tc_first:.2g} ms",
             fontsize=8.5, va="bottom")
axE.set_xscale("log"); axE.set_yscale("log")
axE.set_xlabel("$T_{config}$"); axE.set_ylabel("ratio  $r = T_{unit}/T_{config}$")
axE.xaxis.set_major_formatter(TFMT)
axE.set_xlim(tcfg_ext.min(), tcfg_ext.max()); axE.set_ylim(rgrid2.min(), rgrid2.max())
axE.legend(handles=[Patch(color=cmap_cols[remap_e[w]], label=labels[w]) for w in winners_e]
                   + [Line2D([], [], color="k", ls="-", lw=2.0, label="realistic limit (1 ms)")],
           title="Winning configuration", title_fontsize=9, fontsize=8,
           loc="upper right", framealpha=0.95)
# [point 27] panel title moved to caption: axE.set_title("Winner map, extended $T_{config}$ — where plain Hungarian BvN takes over\n"
# [point 27] panel title moved to caption: "(few-permutation regime; realistic hardware stays left of the 1 ms line)", fontsize=11)
figE.tight_layout()
figE.savefig(f"{OUT}/winner_map_bvn_regime.pdf", bbox_inches="tight")
figE.savefig(f"{OUT}/winner_map_bvn_regime.png", dpi=140, bbox_inches="tight")
print("saved winner_map_bvn_regime")
