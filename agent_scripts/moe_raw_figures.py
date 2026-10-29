"""Figure and LaTeX table for section 5.8 (measured MoE traffic, no regularisation).

  moe_raw.pdf          single column, two panels:
                         (a) port utilization per trace, traces ordered by column skew
                         (b) cycle inflation C/S by Radix base
  tab_moe_raw.tex      per-engine digest, one column wide
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif"})
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = r"C:\Users\ofekc\Desktop\Msc\Thesis\paper\plots_pdf"
TABLES = r"C:\Users\ofekc\Desktop\Msc\Thesis\paper"

ENG = [("Hungarian", "#0072B2", "o"), ("GW Dynamic", "#D55E00", "s"),
       ("GW Static", "#009E73", "^"), ("WFA", "#CC79A7", "D")]
BASES = [2, 4, 8, 16]

df = pd.read_csv(os.path.join(ROOT, "run", "moe_raw", "moe_raw_results.csv"))
bvn = df[df.framework == "BvN"]

fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.85))
fig.subplots_adjust(wspace=0.35)

# ---- (a) utilization per trace, ordered by how skewed the trace is -------------
order = (bvn.groupby("trace").col_skew.first().sort_values().index.tolist())
x = np.arange(len(order))
ax = axes[0]
for name, col, mk in ENG:
    sub = bvn[bvn.engine == name].set_index("trace").reindex(order)
    ax.plot(x, sub.eta, "-", color=col, lw=1.0, alpha=.9)
    ax.plot(x, sub.eta, mk, color=col, ms=3.2, markeredgecolor="white",
            markeredgewidth=.4, label=name)
ax.axhline(1.0, color="gray", ls=":", lw=1.0)
ax.set_ylim(0.84, 1.012)
ax.set_xlabel("trace (ordered by column skew)", fontsize=5.5, labelpad=1)
ax.set_ylabel(r"$\eta = S/C$", fontsize=5.5, labelpad=1)
ax.set_xticks([])
ax.tick_params(labelsize=5, pad=1)
ax.grid(True, which="both", ls=":", lw=.35, alpha=.6)
ax.set_title("(a) utilization, BvN", fontsize=6)
ax.legend(fontsize=4.2, framealpha=.95, loc="lower left", ncol=2)

# ---- (b) cycle inflation by Radix base ----------------------------------------
ax = axes[1]
for name, col, mk in ENG:
    sub = df[(df.engine == name) & (df.framework != "BvN")]
    g = sub.groupby("base").C_over_S.mean().reindex(BASES)
    ax.plot(BASES, g.values, "-", color=col, lw=1.0, alpha=.9)
    ax.plot(BASES, g.values, mk, color=col, ms=3.2, markeredgecolor="white",
            markeredgewidth=.4)
ax.axhline(1.0, color="gray", ls=":", lw=1.0)
ax.set_xscale("log", base=2)
ax.set_xticks(BASES)
ax.set_xticklabels([str(b) for b in BASES])
ax.set_xlabel("radix base $B$", fontsize=5.5, labelpad=1)
ax.set_ylabel("$C/S$", fontsize=5.5, labelpad=1)
ax.tick_params(labelsize=5, pad=1)
ax.grid(True, which="both", ls=":", lw=.35, alpha=.6)
ax.set_title("(b) cycle inflation, Radix", fontsize=6)

fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(PLOTS, f"moe_raw.{ext}"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved moe_raw.pdf")

# ---- LaTeX table ---------------------------------------------------------------
rows = []
for name, _, _ in ENG:
    b = bvn[bvn.engine == name]
    r8 = df[(df.engine == name) & (df.base == 8)]
    rows.append((name, b.N.mean(), b.C_over_S.mean(), b.eta.mean(), b.eta.min(),
                 (b.C_over_S <= 1 + 1e-9).sum(), len(b), b.frac_full.mean(),
                 b.runtime_ms.mean(), r8.C_over_S.mean()))

lines = [
    r"\begin{table}[t]", r"  \centering", r"  \small",
    r"  \caption{Measured MoE traffic decomposed as recorded, without regularisation."
    r"  $S=\max_i\max(\sum_j D_{ij}, \sum_j D_{ji})$ is the cycle-length lower bound."
    r"  \emph{opt} counts the traces (of 12) reaching $C=S$; \emph{full} is the share of"
    r"  extracted matchings that are complete permutations.}",
    r"  \label{tab:moeraw}",
    r"  \begin{tabular}{lrrrrrr}", r"    \toprule",
    r"    & \multicolumn{5}{c}{BvN} & Radix \\",
    r"    \cmidrule(lr){2-6}\cmidrule(lr){7-7}",
    r"    engine & $N$ & $C/S$ & $\eta$ & opt & full & $C/S$ ($B{=}8$) \\",
    r"    \midrule",
]
for (name, N, cs, eta, emin, opt, tot, full, ms, r8) in rows:
    lines.append(f"    {name} & {N:.1f} & {cs:.3f} & {eta:.3f} & {opt}/{tot} "
                 f"& {full:.0%} & {r8:.3f} \\\\".replace("%", r"\%"))
lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]

path = os.path.join(TABLES, "tab_moe_raw.tex")
with open(path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("wrote", path)
print("\n".join(lines))
