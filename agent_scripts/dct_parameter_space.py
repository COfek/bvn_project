"""DCT parameter-space analysis, rebuilt from the FRESH complete data.

DCT = T_calc + N*T_config + C*T_unit   (all in ms)

Candidate configurations: for every engine {Hungarian, WFA, GW Dynamic, GW Static}
  - BvN baseline
  - Radix B16, B8, B4, B2
  - Euler depths d1, d2, d3
each carrying its own (N, C, T_calc) from run/rerun_2026 (radix+BvN) and
euler_exp (Euler, dense).

Outputs (paper/plots_pdf/):
  dct_param_space_2d.pdf   : ratio heatmap + winner map  (T_config x T_unit)
  dct_3d_surfaces.pdf      : two-surface crossover + minimum-DCT surface
"""
from __future__ import annotations
import glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = "C:/Users/ofekc/Desktop/Msc/Thesis/bvn_project"
OUT = "C:/Users/ofekc/Desktop/Msc/Thesis/paper/plots_pdf"
ENG = [("Hungarian", "max", "maximum"), ("WFA", "wfa", "wfa"),
       ("GW Dynamic", "heavy", "heavy"), ("GW Static", "heavy_static", "heavy_static")]

def rerun(e): return sorted(glob.glob(f"{ROOT}/run/rerun_2026/{e}/*/results_stats.csv"))[-1]
def euler(p): return sorted(glob.glob(f"{ROOT}/euler_exp/eng_{p}_dense_d4/*/results_stats.csv"))[-1]
EULER_DEPTHS = [1, 2, 3, 4]

# ---- gather candidate configs: label -> (N, C, T_calc_ms) ----
configs = {}
for name, re, pre in ENG:
    dr = pd.read_csv(rerun(re)).iloc[5:]
    configs[f"{name} BvN"] = (dr["bvn_perms"].mean(), dr["bvn_cycle"].mean(), dr["bvn_runtime"].mean()*1000)
    for b in [16, 8, 4, 2]:
        configs[f"{name} B{b}"] = (dr[f"{pre}_{b}_perms"].mean(), dr[f"{pre}_{b}_cycle"].mean(), dr[f"{pre}_{b}_time"].mean()*1000)
    de = pd.read_csv(euler(pre)).iloc[5:]
    for d in EULER_DEPTHS:
        # T_calc for Euler = split phase + max-leaf parallel extraction (the
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
axr.set_xlabel("$T_{config}$ (ms)"); axr.set_ylabel("$T_{unit}$ (ms)")
axr.set_title(f"Ratio: {labels[A]} vs {labels[B]}\n(green = {labels[A]} wins, red = {labels[B]} wins)")

# winner map (only dominant winners, discrete colors)
cmap_cols = ["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7","#999999","#000000","#882255"]
remap = {w:i for i,w in enumerate(winners)}
winimg = np.vectorize(remap.get)(win)
cmap = ListedColormap(cmap_cols[:len(winners)])
norm = BoundaryNorm(np.arange(-0.5, len(winners)+0.5), cmap.N)
axw.pcolormesh(TC, TU, winimg, cmap=cmap, norm=norm, shading="auto")
axw.set_xscale("log"); axw.set_yscale("log")
axw.set_xlabel("$T_{config}$ (ms)"); axw.set_ylabel("$T_{unit}$ (ms)")
axw.set_title("Winner map (lowest-DCT configuration)")
from matplotlib.patches import Patch
axw.legend(handles=[Patch(color=cmap_cols[remap[w]], label=labels[w]) for w in winners],
           fontsize=8, loc="upper left", framealpha=0.95)

for ax in (axr, axw):
    ax.plot(tcfg, 10*tcfg, "k--", lw=1.4, label="$T_{unit}=10\\,T_{config}$")
    ax.axhline(0.01, color="navy", ls=":", lw=1.4, label="Paper: $T_{unit}=0.01$ ms")
    ax.set_xlim(tcfg.min(), tcfg.max()); ax.set_ylim(tunit.min(), tunit.max())
fig.suptitle("DCT Parameter Space — Dense (n=256, k=256, $W_{max}$=64)   |   "
             "DCT = $T_{calc}+N\\,T_{config}+C\\,T_{unit}$", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/dct_param_space_2d.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/dct_param_space_2d.png", dpi=140, bbox_inches="tight")
print("saved dct_param_space_2d")

# ===== FIGURE 1b: y-axis re-parameterised as ratio r = T_unit / T_config =====
# At (T_config, r) the actual T_unit = r * T_config, so
#   DCT = T_calc + N*T_config + C*(r*T_config) = T_calc + T_config*(N + C*r)
rgrid = np.logspace(-1, 3, 240)          # ratio up to 1000
TCr, R = np.meshgrid(tcfg, rgrid)
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
axrr.set_title(f"Ratio: {labels[A]} vs {labels[B]}\n(green = {labels[A]} wins, red = {labels[B]} wins)")

remap_r = {w: i for i, w in enumerate(winners_r)}
cmap_r = ListedColormap(cmap_cols[:len(winners_r)])
norm_r = BoundaryNorm(np.arange(-0.5, len(winners_r)+0.5), cmap_r.N)
axrw.pcolormesh(TCr, R, np.vectorize(remap_r.get)(winr), cmap=cmap_r, norm=norm_r, shading="auto")
axrw.set_title("Winner map (lowest-DCT configuration)")
axrw.legend(handles=[Patch(color=cmap_cols[remap_r[w]], label=labels[w]) for w in winners_r],
            fontsize=8, loc="upper left", framealpha=0.95)
for ax in (axrr, axrw):
    ax.axhline(10, color="k", ls="--", lw=1.4, label="$T_{unit}=10\\,T_{config}$ (lecturer)")
    ax.plot(tcfg, 0.01/tcfg, "navy", ls=":", lw=1.4, label="Paper: $T_{unit}=0.01$ ms")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$T_{config}$ (ms)"); ax.set_ylabel("ratio  $r = T_{unit}/T_{config}$")
    ax.set_xlim(tcfg.min(), tcfg.max()); ax.set_ylim(rgrid.min(), rgrid.max())
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
figR.suptitle("DCT Parameter Space — y-axis as ratio $r=T_{unit}/T_{config}$ (up to 1000)   |   "
              "Dense (n=256, k=256, $W_{max}$=64)", fontsize=13, y=1.02)
figR.tight_layout()
figR.savefig(f"{OUT}/dct_param_space_ratio.pdf", bbox_inches="tight")
figR.savefig(f"{OUT}/dct_param_space_ratio.png", dpi=140, bbox_inches="tight")
print("saved dct_param_space_ratio")

# ================= FIGURE 2: 3-D surfaces =================
LTC, LTU = np.log10(TC), np.log10(TU)
fig = plt.figure(figsize=(15, 6.5))
# left: two surfaces intersecting
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(LTC, LTU, np.log10(dct(A)), color="#56B4E9", alpha=0.65, label=labels[A])
ax1.plot_surface(LTC, LTU, np.log10(dct(B)), color="#E69F00", alpha=0.65)
ax1.set_xlabel("log10 $T_{config}$"); ax1.set_ylabel("log10 $T_{unit}$"); ax1.set_zlabel("log10 DCT")
ax1.set_title(f"{labels[A]} vs {labels[B]}\n(surfaces intersect = crossover)")
ax1.view_init(elev=22, azim=-52)
# right: minimum-DCT surface colored by winner
ax2 = fig.add_subplot(122, projection="3d")
minZ = np.log10(allD.min(axis=0))
facecol = np.empty(win.shape + (4,))
from matplotlib.colors import to_rgba
for w in winners:
    facecol[win == w] = to_rgba(cmap_cols[remap[w]])
ax2.plot_surface(LTC, LTU, minZ, facecolors=facecol, alpha=1.0, linewidth=0, antialiased=False, shade=False)
ax2.set_xlabel("log10 $T_{config}$"); ax2.set_ylabel("log10 $T_{unit}$"); ax2.set_zlabel("log10 DCT")
ax2.set_title("Minimum-DCT surface\n(coloured by winning configuration)")
ax2.view_init(elev=22, azim=-52)
ax2.legend(handles=[Patch(color=cmap_cols[remap[w]], label=labels[w]) for w in winners],
           fontsize=7.5, loc="upper left")
fig.suptitle("DCT Parameter Space (3-D) — Dense (n=256, k=256, $W_{max}$=64)", fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/dct_3d_surfaces.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/dct_3d_surfaces.png", dpi=140, bbox_inches="tight")
print("saved dct_3d_surfaces")
