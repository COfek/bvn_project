"""Six standalone thesis-candidate figures, each saved as its own PDF+PNG in
paper/plots_pdf/. Built from the fresh run/rerun_2026 CSVs.

  cost_cycle_inflation      (1) cost of Radix: cycle inflation vs base
  trade_speedup_vs_cost     (2) runtime speed-up bought per unit cycle cost
  benefit_perm_collapse     (3) permutation-count collapse vs base
  normalized_tradeoff       (4) each engine normalized to its own BvN
  parallel_coords           (5) all 20 engine x base configs, 3 metrics
  dct_crossover             (6) BvN vs Radix B2 DCT curves per engine
"""
from __future__ import annotations
import glob, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "C:/Users/ofekc/Desktop/Msc/Thesis/bvn_project"
OUT = "C:/Users/ofekc/Desktop/Msc/Thesis/paper/plots_pdf"
RUNS = [("Hungarian", "max", "maximum", "#4C72B0"),
        ("GW Dynamic", "heavy", "heavy", "#55A868"),
        ("GW Static", "heavy_static", "heavy_static", "#C44E52"),
        ("WFA", "wfa", "wfa", "#8172B3")]
ORDER = ["BvN", "B16", "B8", "B4", "B2"]
MK = {"BvN": "o", "B16": "s", "B8": "^", "B4": "D", "B2": "v"}

def latest(e):
    return sorted(glob.glob(f"{ROOT}/run/rerun_2026/{e}/*/results_stats.csv"))[-1]

def load():
    data = {}
    for name, eng, pre, col in RUNS:
        df = pd.read_csv(latest(eng)).iloc[5:]
        d = {"BvN": (df["bvn_cycle"].mean(), df["bvn_runtime"].mean() * 1000, df["bvn_perms"].mean())}
        for b in [16, 8, 4, 2]:
            d[f"B{b}"] = (df[f"{pre}_{b}_cycle"].mean(), df[f"{pre}_{b}_time"].mean() * 1000, df[f"{pre}_{b}_perms"].mean())
        data[name] = (col, d)
    S = pd.read_csv(latest("heavy")).iloc[5:]["bvn_cycle"].mean()
    return data, S

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

def main():
    data, S = load()

    # (1) cost
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for name, (col, d) in data.items():
        ax.plot(range(5), [100 * (d[k][0] / S - 1) for k in ORDER], "-o", color=col, label=name)
    ax.set_xticks(range(5)); ax.set_xticklabels(ORDER); ax.axhline(0, color="gray", ls=":", lw=.6)
    ax.set_xlabel("Decomposition"); ax.set_ylabel("Cycle inflation over $S$  (%)")
    ax.set_title("The cost of Radix: cycle grows as the base shrinks")
    ax.legend(frameon=False); ax.grid(alpha=.3); save(fig, "cost_cycle_inflation")

    # (2) trade
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for name, (col, d) in data.items():
        rb = d["BvN"][1]
        for k in ["B16", "B8", "B4", "B2"]:
            ax.scatter(rb / d[k][1], 100 * (d[k][0] / S - 1), s=70, color=col, edgecolor="k", lw=.4)
            ax.annotate(k[1:], (rb / d[k][1], 100 * (d[k][0] / S - 1)), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.plot([], [], "o", color=col, label=name)
    ax.set_xscale("log"); ax.set_xlabel("Runtime speed-up over BvN  ($\\times$, log)")
    ax.set_ylabel("Cycle inflation  (%)")
    ax.set_title("What you trade: speed-up bought per unit of cycle cost")
    ax.legend(frameon=False); ax.grid(alpha=.3, which="both"); save(fig, "trade_speedup_vs_cost")

    # (3) benefit
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for name, (col, d) in data.items():
        ax.plot(range(5), [d[k][2] for k in ORDER], "-o", color=col, label=name)
    ax.set_xticks(range(5)); ax.set_xticklabels(ORDER); ax.set_yscale("log")
    ax.set_xlabel("Decomposition"); ax.set_ylabel("Permutation count (log)")
    ax.set_title("The benefit: Radix collapses the permutation count")
    ax.legend(frameon=False); ax.grid(alpha=.3, which="both"); save(fig, "benefit_perm_collapse")

    # (4) normalized
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for name, (col, d) in data.items():
        c0, r0 = d["BvN"][0], d["BvN"][1]
        ax.plot([d[k][0] / c0 for k in ORDER], [d[k][1] / r0 for k in ORDER], "-", color=col, alpha=.4)
        for k in ORDER:
            ax.scatter(d[k][0] / c0, d[k][1] / r0, s=45, marker=MK[k], color=col, edgecolor="k", lw=.4)
        ax.plot([], [], "-o", color=col, label=name)
    ax.axhline(1, color="gray", ls=":", lw=.6); ax.axvline(1, color="gray", ls=":", lw=.6)
    ax.set_xlabel("Cycle / BvN"); ax.set_ylabel("Runtime / BvN")
    ax.set_title("Normalized trade-off: a universal shape across engines")
    ax.legend(frameon=False); ax.grid(alpha=.3); save(fig, "normalized_tradeoff")

    # (5) parallel coordinates
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    allc = [(name, col, k, d[k][1], d[k][0], d[k][2]) for name, (col, d) in data.items() for k in ORDER]
    arr = np.array([[c[3], c[4], c[5]] for c in allc], float)
    norm = (arr - arr.min(0)) / (arr.max(0) - arr.min(0))
    for i, c in enumerate(allc):
        ax.plot([0, 1, 2], norm[i], "-", color=c[1], alpha=.55, lw=1.2, marker=MK[c[2]], ms=5)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Runtime (ms)", "Cycle", "Permutations"])
    ax.set_ylabel("normalized  (0 = best, 1 = worst)")
    ax.set_title("Design space: all 20 engine $\\times$ base configurations")
    ax.grid(alpha=.3, axis="x")
    ax.legend(handles=[Line2D([], [], color=c, label=n) for n, _, _, c in RUNS], frameon=False)
    save(fig, "parallel_coords")

    # (6) DCT crossover
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    T_UNIT = 0.01; t = np.logspace(-6, 0, 600)
    for name, (col, d) in data.items():
        for k, ls in [("BvN", "-"), ("B2", "--")]:
            C, Tc, N = d[k]
            ax.plot(t, Tc + N * t + C * T_UNIT, ls, color=col, lw=1.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$T_{\\mathrm{config}}$  [ms, log]"); ax.set_ylabel("DCT  [ms, log]")
    ax.set_title("DCT crossover:  solid = BvN,  dashed = Radix $B{=}2$")
    ax.legend(handles=[Line2D([], [], color=c, label=n) for n, _, _, c in RUNS], frameon=False)
    ax.grid(alpha=.3, which="both"); save(fig, "dct_crossover")

if __name__ == "__main__":
    main()
