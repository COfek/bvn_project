"""The traffic-model (TM) workload as figures, every setup of every engine (Chen K19, K16).

Replaces Table 5 in the body: the TM dense regime of the tm_workload campaign, all nine setups
for each of the four engines, drawn as Figures 3 and 4 are (LaTeX text, 9 pt, one column wide,
fixed margins, one standard deviation of T_parallel across the campaign's 10 matrices as dashed
lines with a 15% shade between them):
  tm_radix_a.pdf / tm_radix_b.pdf  Radix, B = 16, 8, 4, 2: T_parallel against C, against N
  tm_euler_a.pdf / tm_euler_b.pdf  Euler, depth 1-4:        T_parallel against C, against N
and prints the DCT of every setup at T_config = 10 us, T_unit = 100 us (Table 4's point).
Each line starts at the engine's unsplit point. As in Figure 4, the strong engines' Euler points
in (a) all sit on C = S, so there only WWFA's depths are labelled; the others are read from (b).
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
REGIME = "TM dense"
TCFG_MS, TUNIT_MS = 0.010, 0.100          # the hardware point of Table 4's DCT column
ENG = [("Hungarian", "#0072B2", "o"), ("GW Dynamic", "#D55E00", "s"),
       ("GW Static", "#009E73", "^"), ("WWFA", "#CC79A7", "D")]
RADIX = ["B16", "B8", "B4", "B2"]
EULER = ["Euler d1", "Euler d2", "Euler d3", "Euler d4"]
SHORT = {"BvN": "unsplit", **{b: b[1:] for b in RADIX}, **{e: e[-1] for e in EULER}}
N_TICKS = [500, 700, 1000, 1500, 3000, 5000]
T_TICKS = [20, 50, 100, 200, 500, 1000, 2000]

FS = 9
plt.rcParams.update({
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{libertine}"
                           r"\usepackage[libertine]{newtxmath}"})
ML, MR, MB, MT, LEG_H = 0.50, 0.06, 0.40, 0.06, 0.30   # inches, as Figures 3 and 4

res = pd.read_csv(os.path.join(str(REPO), RUN, "tm_workload", "tm_results.csv"))
res = res[res.regime == REGIME]
g = res.groupby(["engine", "config"])
mean, sd = g[["N", "C", "runtime_ms"]].mean(), g["runtime_ms"].std()
S = res.groupby("matrix").S.first().mean()
print(f"{REGIME}: {res.matrix.nunique()} matrices, mean S = {S:.1f}")


def style(ax, xlab, logx):
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlab, labelpad=1)
    ax.set_ylabel(r"$T_{\mathrm{parallel}}$ [ms]", labelpad=1)
    ax.tick_params(pad=1.5)
    ax.grid(True, which="both", ls=":", lw=.35, alpha=.6)


def ticks(axis, values, lo, hi):
    axis.set_major_locator(FixedLocator([v for v in values if lo <= v <= hi]))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def place_labels(ax, labs):
    """As in Figure 3: where the same text of several engines falls closer than 6% of the panel,
    one dark-grey label stands for all of them. Offset mode "r": left of the marker, "c":
    centred below it, "c+": centred above it."""
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


def figure(branch, tag, offsets, label_a, n_ticks):
    """Two panels, tm_<tag>_a (T against C) and tm_<tag>_b (T against N)."""
    figs = [plt.subplots(figsize=(3.33, 2.05 + (LEG_H if i == 0 else 0))) for i in range(2)]
    axes = [ax for _, ax in figs]
    labels = [[], []]
    cfgs = ["BvN"] + branch
    for name, col, mk in ENG:
        N = [mean.loc[(name, c), "N"] for c in cfgs]
        C = [mean.loc[(name, c), "C"] for c in cfgs]
        T = [mean.loc[(name, c), "runtime_ms"] for c in cfgs]
        Tsd = [sd.loc[(name, c)] for c in cfgs]
        print(tag, name, [f"{t:.0f}" for t in T], [f"{c / S:.3f}" for c in C],
              [f"{n:.0f}" for n in N])
        for k, (ax, xs) in enumerate(((axes[0], C), (axes[1], N))):
            up, lo = [t + s for t, s in zip(T, Tsd)], [t - s for t, s in zip(T, Tsd)]
            ax.fill(xs + xs[::-1], up + lo[::-1], color=col, alpha=.15, lw=0)
            for ys_ in (up, lo):
                ax.plot(xs, ys_, "--", color=col, lw=.7, alpha=.8, dashes=(3, 2))
            ax.plot(xs, T, "-", color=col, lw=1.3, alpha=.95)
            ax.plot(xs, T, mk, color=col, ms=5, markeredgecolor="white", markeredgewidth=.5,
                    label=name)
            if k == 1 or label_a(name):
                for x, y, c in zip(xs, T, cfgs):
                    labels[k].append((x, y, SHORT[c], col,
                                      offsets.get((k, name, SHORT[c]), (3, 3))))
    style(axes[0], "cycle length $C$", logx=False)
    style(axes[1], "number of configurations $N$", logx=True)
    for ax in axes:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.5)
        ticks(ax.yaxis, T_TICKS, *ax.get_ylim())
    ticks(axes[1].xaxis, n_ticks, *axes[1].get_xlim())
    axes[0].axvline(S, color="gray", ls=":", lw=1.0)
    axes[0].annotate("$S$", (S, 1), xycoords=("data", "axes fraction"), xytext=(3, -12),
                     textcoords="offset points", color="gray")
    lo, hi = axes[0].get_xlim()
    axes[0].set_xlim(left=S - 0.2 * (hi - S))        # room for the unsplit labels
    for ax, labs in zip(axes, labels):
        place_labels(ax, labs)
    figs[0][0].legend(*axes[0].get_legend_handles_labels(), loc="upper center",
                      bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, handlelength=0.8,
                      handletextpad=0.25, columnspacing=0.7, borderaxespad=0.1)
    for (fig, _), p in zip(figs, "ab"):
        w, h = fig.get_size_inches()
        top = MT + (LEG_H if p == "a" else 0)
        fig.subplots_adjust(left=ML / w, right=1 - MR / w, bottom=MB / h, top=1 - top / h)
        for e in ("pdf", "png"):
            fig.savefig(os.path.join(str(PLOTS), f"tm_{tag}_{p}.{e}"), dpi=200)
        plt.close(fig)


# (panel, engine, label) -> offset in points where the default (3, 3) collides; a third element
# sets the mode (see place_labels)
RADIX_OFFSET = {(0, "Hungarian", "unsplit"): (-4, 0, "r"), (0, "GW Static", "unsplit"): (-4, 0, "r"),
                (0, "GW Dynamic", "unsplit"): (-4, 0, "r"), (0, "WWFA", "unsplit"): (-4, 0, "r"),
                (0, "Hungarian", "4"): (3, -9), (0, "GW Static", "16"): (-4, 4, "r"),
                (0, "WWFA", "8"): (-4, 0, "r"),
                (1, "GW Static", "unsplit"): (-4, -6, "r"), (1, "WWFA", "unsplit"): (0, 5, "c+")}
EULER_OFFSET = {(0, "WWFA", "unsplit"): (-4, 0, "r"), (0, "WWFA", "4"): (0, -6, "c"),
                (1, "GW Static", "unsplit"): (-4, 0, "r"), (1, "WWFA", "unsplit"): (-4, 0, "r"),
                (1, "GW Static", "1"): (-4, 0, "r"), (1, "GW Static", "2"): (-4, 0, "r"),
                (1, "GW Static", "3"): (-4, 4, "r"), (1, "GW Static", "4"): (-4, -4, "r")}
figure(RADIX, "radix", RADIX_OFFSET, lambda name: True, N_TICKS)
figure(EULER, "euler", EULER_OFFSET, lambda name: name == "WWFA", N_TICKS + [10000])

# the DCT at Table 4's hardware point, per setup, printed for the text (no panel: Ofek 2026-11-26)
order = ["BvN"] + RADIX + EULER
for name, _, _ in ENG:
    dct = {c: mean.loc[(name, c), "runtime_ms"] + mean.loc[(name, c), "N"] * TCFG_MS
           + mean.loc[(name, c), "C"] * TUNIT_MS for c in order}
    best = min(dct, key=dct.get)
    print(f"{name} DCT [ms]:", {c: round(v) for c, v in dct.items()}, "best:", best)
print("saved tm_radix_a/b, tm_euler_a/b")
