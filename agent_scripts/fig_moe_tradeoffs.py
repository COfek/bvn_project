"""MoE traffic at n = 256 as a figure, every setup of every engine (Chen K19, K16, K18).

The moe256 campaign (moe256_campaign.py: each of the 12 measured traces upsampled to 256 x 256,
three samples each, 18 matrices per family), the unsplit decomposition and Radix B = 16, 8, 4, 2
for each of the four engines (no Euler: the matrices are not regular), drawn as Figure 3 is
(LaTeX text, 9 pt, one column wide, fixed margins):
  moe_tradeoff_a.pdf  throughput traces: T_parallel against C/S
  moe_tradeoff_b.pdf  MMLU traces:       T_parallel against C/S
C/S rather than C, since the matrices of a family differ in S (up to five-fold); each point is
the mean over the 18 matrices of the per-matrix C/S and T_parallel, and the dashed lines and the
15% shade mark one standard deviation of T_parallel between the three samples of a trace,
pooled over the six traces.
Also prints, per engine and family, the unit time above which the unsplit decomposition has a
shorter DCT than every Radix base (the break-even of the DCT, with T_config = 10 us).
Run with BVN_CAMPAIGN=2026_11 BVN_RUN_ROOT=run/rerun_2026_11e.
"""
from __future__ import annotations
from _paths import PLOTS, REPO
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter, NullFormatter

RUN = os.environ.get("BVN_RUN_ROOT", "run")
TCFG_MS = 0.010
ENG = [("Hungarian", "#0072B2", "o"), ("GW Dynamic", "#D55E00", "s"),
       ("GW Static", "#009E73", "^"), ("WWFA", "#CC79A7", "D")]
CFGS = ["BvN", "B16", "B8", "B4", "B2"]
SHORT = {"BvN": "unsplit", "B16": "16", "B8": "8", "B4": "4", "B2": "2"}
FAMILIES = [("MoE throughput", "a"), ("MoE mmlu", "b")]
T_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

FS = 9
plt.rcParams.update({
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{libertine}"
                           r"\usepackage[libertine]{newtxmath}"})
ML, MR, MB, MT, LEG_H = 0.50, 0.06, 0.40, 0.06, 0.30   # inches, as Figure 3

res = pd.read_csv(os.path.join(str(REPO), RUN, "moe256", "moe256_results.csv"))
res["matrix"] = res.trace + " / " + res["sample"].astype(str)
res = res[res.engine.isin([e for e, _, _ in ENG])]
g = res.groupby(["regime", "engine", "config"])
mean = g[["N", "C", "C_over_S", "runtime_ms"]].mean()
# the band: the spread between the three samples of one trace (the generator's randomness),
# pooled over the traces of a family; the spread across all 18 matrices would mostly measure the
# difference between models, whose S differs up to 2.4-fold
v = res.groupby(["regime", "engine", "config", "trace"]).runtime_ms.var()
sd = np.sqrt(v.groupby(["regime", "engine", "config"]).mean())

# (panel, engine, label) -> offset in points; "r" left of the marker, "c" centred below
OFFSET = {}


def place_labels(ax, labs):
    """As in Figure 3: one dark-grey label where the same text of several engines nearly meets."""
    placed = []
    toax = ax.transAxes.inverted()
    for x, y, t, col, off in labs:
        q = toax.transform(ax.transData.transform((x, y)))
        near = [a for p_, t_, a in placed if t_ == t and np.hypot(*(q - p_)) < 0.06]
        if near:
            near[0].set_color("0.25")
            continue
        mode = off[2] if len(off) == 3 else ""
        placed.append((q, t, ax.annotate(
            t, (x, y), xytext=off[:2], textcoords="offset points", fontsize=FS, color=col,
            ha={"r": "right", "c": "center", "c+": "center"}.get(mode, "left"),
            va={"r": "center", "c": "top", "c+": "bottom"}.get(mode, "baseline"),
            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])))


for fam, tag in FAMILIES:
    fig, ax = plt.subplots(figsize=(3.33, 2.05 + (LEG_H if tag == "a" else 0)))
    labs = []
    S = res[res.regime == fam].groupby("matrix").S.first()
    print(f"{fam}: {len(S)} matrices, S from {S.min():.0f} to {S.max():.0f}")
    for name, col, mk in ENG:
        X = [mean.loc[(fam, name, c), "C_over_S"] for c in CFGS]
        T = [mean.loc[(fam, name, c), "runtime_ms"] for c in CFGS]
        Tsd = [sd.loc[(fam, name, c)] for c in CFGS]
        up, lo = [t + s for t, s in zip(T, Tsd)], [max(t - s, 1.0) for t, s in zip(T, Tsd)]
        ax.fill(X + X[::-1], up + lo[::-1], color=col, alpha=.15, lw=0)
        for ys_ in (up, lo):
            ax.plot(X, ys_, "--", color=col, lw=.7, alpha=.8, dashes=(3, 2))
        ax.plot(X, T, "-", color=col, lw=1.3, alpha=.95)
        ax.plot(X, T, mk, color=col, ms=5, markeredgecolor="white", markeredgewidth=.5, label=name)
        for x, y, c in zip(X, T, CFGS):
            labs.append((x, y, SHORT[c], col, OFFSET.get((tag, name, SHORT[c]), (3, 3))))
        print(f"  {name:10s} T", [f"{t:.0f}" for t in T], "C/S", [f"{x:.3f}" for x in X])
        # DCT break-even: unit time (us per token) above which unsplit beats every Radix base
        m0 = mean.loc[(fam, name, "BvN")]
        be = []
        for c in CFGS[1:]:
            m = mean.loc[(fam, name, c)]
            dT = m0.runtime_ms - m.runtime_ms + (m0.N - m.N) * TCFG_MS   # what Radix saves
            dC = m.C - m0.C                                                # what it costs, tokens
            be.append(1000 * dT / dC if dC > 0 else np.inf)
        print(f"  {'':10s} DCT: unsplit shortest for T_unit above "
              f"{max(be):.3f} us per token (per base: {[f'{b:.3f}' for b in be]})")
    ax.set_yscale("log")
    ax.set_xlabel("cycle length $C/S$", labelpad=1)
    ax.set_ylabel(r"$T_{\mathrm{parallel}}$ [ms]", labelpad=1)
    ax.tick_params(pad=1.5)
    ax.grid(True, which="both", ls=":", lw=.35, alpha=.6)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.5)
    lo_, hi_ = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator([v for v in T_TICKS if lo_ <= v <= hi_]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axvline(1, color="gray", ls=":", lw=1.0)
    l, h = ax.get_xlim()
    ax.set_xlim(left=1 - 0.15 * (h - 1))   # room for the unsplit labels
    place_labels(ax, labs)
    if tag == "a":
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncol=4, frameon=False, handlelength=0.8, handletextpad=0.25, columnspacing=0.7,
                   borderaxespad=0.1)
    w, h = fig.get_size_inches()
    top = MT + (LEG_H if tag == "a" else 0)
    fig.subplots_adjust(left=ML / w, right=1 - MR / w, bottom=MB / h, top=1 - top / h)
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(str(PLOTS), f"moe_tradeoff_{tag}.{e}"), dpi=200)
    plt.close(fig)
print("saved moe_tradeoff_a/b")
