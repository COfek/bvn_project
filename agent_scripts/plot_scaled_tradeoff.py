"""Runtime vs cycle-length trade-off, with the scale-tuned Radix points.

Solid = Radix with the per-base tuned scale; faint dashed = the original (untuned)
trajectory. Cycle is reported in the ORIGINAL matrix's units (C_scaled / c), so all
points describe decompositions of the same demand matrix and are directly comparable.
"""
from __future__ import annotations
from _paths import PLOTS, REPO  # repository-relative paths
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8})
ROOT = str(REPO)
PLOTS = str(PLOTS)

res = pd.read_csv(os.path.join(ROOT, "run", "radix_scale", "scale_results.csv"))
res["S_original"] = res.S_scaled / res.scale
res["C_orig"] = res.C_over_S * res.S_original

bvn = pd.read_csv(os.path.join(ROOT, "run", "tm_workload", "tm_summary.csv"))
BVN_MAP = {"TM dense": "TM dense", "Uniform": "Uniform (thesis)"}

ENG = [("Hungarian", "#0072B2", "o"), ("GW Dynamic", "#D55E00", "s"),
       ("GW Static", "#009E73", "^"), ("WFA", "#CC79A7", "D")]
BASES = [16, 8, 4, 2]

# Single panel, primary (generated) workload only, SCALED trajectories only.
# The unscaled comparison was dropped at the author's request, and the second workload
# is handled separately in the workload-sensitivity section. Sized for ONE column.
REGIME = "Uniform"
fig, ax = plt.subplots(figsize=(3.4, 3.1))
r = res[res.regime == REGIME]
b = bvn[bvn.regime == BVN_MAP[REGIME]]
for name, col, mk in ENG:
    re_ = r[r.engine == name]
    g = (re_.groupby(["base", "scale"])
         .agg(C=("C_orig", "mean"), Tms=("runtime_ms", "mean")).reset_index())
    tu = g.loc[g.groupby("base").C.idxmin()].set_index("base").reindex(BASES)

    bv = b[(b.engine == name) & (b.config == "BvN")]
    bC = float(bv.C.iloc[0]); bT = float(bv.runtime_ms.iloc[0])

    ax.plot([bC] + tu.C.tolist(), [bT] + tu.Tms.tolist(), "-", color=col, lw=1.4, alpha=.95)
    ax.plot([bC] + tu.C.tolist(), [bT] + tu.Tms.tolist(), mk, color=col, ms=5,
            markeredgecolor="white", markeredgewidth=.5, label=name)
    ax.annotate("BvN", (bC, bT), fontsize=5.5, color=col,
                xytext=(3, 3), textcoords="offset points")
    for base, row in tu.iterrows():
        ax.annotate(f"B{base}", (row.C, row.Tms), fontsize=5.5, color=col,
                    xytext=(3, 2), textcoords="offset points")

S0 = float(r.S_original.iloc[0])
ax.axvline(S0, color="gray", ls=":", lw=1.0)
ax.set_yscale("log")
ax.set_xlabel("cycle length $C$", fontsize=7.5)
ax.set_ylabel("$T_{compute}$ [ms] (log)", fontsize=7.5)
ax.tick_params(labelsize=6.5)
ax.grid(True, which="both", ls=":", lw=.35, alpha=.6)
ax.legend(fontsize=6, framealpha=.95, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "scaled_tradeoff.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(PLOTS, "scaled_tradeoff.png"), dpi=145, bbox_inches="tight")
print("saved scaled_tradeoff")
