"""Winner maps per the advisor's spec:
  - y axis: ratio r = T_unit/T_config starting at 5 (up to 1000)
  - x axis: T_config over realistic optical-switch values (1 us .. 1 ms)
  - three panels: T_calc x1 (full), x1/2, x0 (fully amortised / prescheduled)
Configs: all 4 engines x {BvN, B16..B2, Euler d1..d4}; Euler T_calc split-inclusive.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

plt.rcParams.update({"font.family": "serif", "font.size": 11})
ROOT = "C:/Users/ofekc/Desktop/Msc/Thesis/bvn_project"
OUT = "C:/Users/ofekc/Desktop/Msc/Thesis/paper/plots_pdf"
ENG = [("Hungarian", "max", "maximum"), ("WFA", "wfa", "wfa"),
       ("GW Dynamic", "heavy", "heavy"), ("GW Static", "heavy_static", "heavy_static")]

def rerun(e): return sorted(glob.glob(f"{ROOT}/run/rerun_2026/{e}/*/results_stats.csv"))[-1]
def euler(p): return sorted(glob.glob(f"{ROOT}/euler_exp/eng_{p}_dense_d4/*/results_stats.csv"))[-1]

configs = {}
for name, re_, pre in ENG:
    dr = pd.read_csv(rerun(re_)).iloc[5:]
    for b in [16, 8, 4, 2]:
        configs[f"{name} B{b}"] = (dr[f"{pre}_{b}_perms"].mean(), dr[f"{pre}_{b}_cycle"].mean(), dr[f"{pre}_{b}_time"].mean()*1000)
    de = pd.read_csv(euler(pre)).iloc[5:]
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

SCALES = [(1.0, r"$T_{calc}$ (full)"), (0.5, r"$T_{calc}/2$"), (0.0, r"$T_{calc}=0$ (prescheduled)")]
wins = []
for s, _ in SCALES:
    allD = np.stack([NCT[i, 2]*s + TC*(NCT[i, 0] + NCT[i, 1]*R) for i in range(len(labels))])
    wins.append(allD.argmin(axis=0))

winners_all = sorted(set(np.concatenate([w.ravel() for w in wins]).tolist()))
cmap_cols = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
remap = {w: i for i, w in enumerate(winners_all)}
cmap = ListedColormap(cmap_cols[:len(winners_all)])
norm = BoundaryNorm(np.arange(-0.5, len(winners_all)+0.5), cmap.N)

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharey=True)
for ax, w, (s, ttl) in zip(axes, wins, SCALES):
    ax.pcolormesh(TC, R, np.vectorize(remap.get)(w), cmap=cmap, norm=norm, shading="auto")
    ax.axhline(10, color="k", ls="--", lw=1.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$T_{config}$ (ms)")
    ax.set_title(ttl, fontsize=11)
    share = {labels[x]: (w == x).mean()*100 for x in sorted(set(w.ravel().tolist()))}
    print(ttl, {k: f"{v:.0f}%" for k, v in share.items()})
axes[0].set_ylabel("ratio  $r = T_{unit}/T_{config}$")
axes[2].legend(handles=[Patch(color=cmap_cols[remap[w]], label=labels[w]) for w in winners_all]
                       + [plt.Line2D([], [], color="k", ls="--", lw=1.3, label="$r=10$ (lecturer)")],
               title="Winning configuration", fontsize=8.5, title_fontsize=9,
               loc="lower right", framealpha=0.95)
fig.suptitle("Winner map at realistic hardware ($T_{config}$: 1$\\mu$s--1ms, $r\\geq5$) --- sensitivity to computation time\n"
             "Dense, n=256, k=256, Wmax=64; Euler $T_{calc}$ split-inclusive", fontsize=12.5, y=1.04)
fig.tight_layout()
fig.savefig(f"{OUT}/winner_map_tcalc_sensitivity.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/winner_map_tcalc_sensitivity.png", dpi=140, bbox_inches="tight")
print("saved winner_map_tcalc_sensitivity")
