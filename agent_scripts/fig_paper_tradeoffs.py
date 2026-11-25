"""Wide two-column trade-off figures for the paper.

Both are single-column (not figure*) so LaTeX can place them inside the subsection.
Each has two side-by-side panels giving computation time against BOTH other metrics,
scaled to fit within a single column horizontally.

  scaled_tradeoff_{a,b,c}.pdf  Radix across engines, the data of Table 2, unscaled (Figure 3;
                               the file name is historical: the scale was removed on 2026-11-24)
  euler_tradeoff_paper.pdf  Euler depths             -- for "Euler splitting across engines"
"""
from __future__ import annotations
from _paths import PLOTS, REPO  # repository-relative paths
import os, glob
import numpy as np
import pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from radix_extraction import correct_summary, correct_scale_results  # noqa: E402
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# same campaign as Chapter 6's tables and Chapter 7's figures
from variance_data import CAMPAIGN, radix_frame, euler_frame
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif"})
ROOT = str(REPO)
PLOTS = str(PLOTS)
ENG = [("Hungarian", "#0072B2", "o"), ("GW Dynamic", "#D55E00", "s"),
       ("GW Static", "#009E73", "^"), ("WFA", "#CC79A7", "D")]
# the workloads of a rerun live under BVN_RUN_ROOT (default run/), as their campaign scripts write them
RUN = os.environ.get("BVN_RUN_ROOT", "run")
if CAMPAIGN != "variance_2026":   # the November rerun: WWFA is the weak engine, in place of WFA
    ENG = [("WWFA", c, m) if n == "WFA" else (n, c, m) for n, c, m in ENG]
BASES = [16, 8, 4, 2]
DEPTHS = [1, 2, 3, 4]


def _style(ax, xlab, ylab, logx=False, logy=True):
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlab, fontsize=5.5, labelpad=1)
    ax.set_ylabel(ylab, fontsize=5.5, labelpad=1)
    ax.tick_params(labelsize=5, pad=1)
    ax.grid(True, which="both", ls=":", lw=.35, alpha=.6)


# ============================== RADIX ==============================
# Figure 3 draws Table 2: the main Radix campaign (all matrices, mean over the three sweeps),
# unscaled, every engine from its BvN baseline down to B=2 (Ofek, 2026-11-24: the scale is
# removed from the paper). Same frames as make_paper_tables.py table_radix().
RADIX_KEYS = {"Hungarian": ("max", "maximum"), "WWFA": ("wwfa", "wwfa"), "WFA": ("wfa", "wfa"),
              "GW Dynamic": ("heavy", "heavy"), "GW Static": ("heavy_static", "heavy_static")}

# Three separate panels, scaled_tradeoff_{a,b,c}.pdf, each one column wide, so the paper sets the
# panel labels (a)-(c) as LaTeX subcaptions below each panel. All text is the paper's body font
# (Linux Libertine, 9 pt) at the size it is printed: the PDF is as wide as the column (3.33 in),
# so \includegraphics[width=\columnwidth] does not rescale it.
from matplotlib import patheffects
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter, NullFormatter
FS = 9
# Text is set by LaTeX with the fonts acmart uses (Linux Libertine, newtxmath), so it is the
# paper's own font. (Matplotlib draws the Libertine OpenType outlines wrongly.)
rc = {"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
      "legend.fontsize": FS, "text.usetex": True, "font.family": "serif",
      "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{libertine}"
                             r"\usepackage[libertine]{newtxmath}"}
LABELS = ["unsplit"] + [str(B) for B in BASES]
N_TICKS = [500, 700, 1000, 1500, 3000, 5000]   # Chen K3: more numbers on the axes
T_TICKS = [20, 50, 100, 200, 500, 1000]        # ms
# where a base label sits, in points from its marker, when the default (3, 3) would hit a point,
# a line or another label: (panel 0-2, engine, base) -> offset
# a third element "r" sets the label right-aligned and centred on its marker, to its left
LABEL_OFFSET = {(0, "Hungarian", "unsplit"): (-4, 0, "r"), (0, "GW Static", "unsplit"): (-4, 0, "r"),
                (0, "GW Dynamic", "unsplit"): (-4, 0, "r"),
                (0, "WWFA", "unsplit"): (-4, 0, "r"), (0, "Hungarian", "8"): (3, -11),
                (0, "Hungarian", "4"): (4, -11),
                (1, "WWFA", "unsplit"): (-4, 3, "r"), (1, "GW Static", "unsplit"): (0, -5, "c"),
                (1, "Hungarian", "8"): (-11, -2), (1, "Hungarian", "4"): (-11, -2),
                (1, "Hungarian", "2"): (-10, -2),
                (2, "Hungarian", "unsplit"): (-6, 6), (2, "Hungarian", "2"): (6, -4),
                (2, "GW Dynamic", "8"): (5, -12), (2, "GW Dynamic", "16"): (3, -11)}
plt.rcParams.update(rc)
figs = [plt.subplots(figsize=(3.33, 2.05)) for _ in range(3)]
axes = [ax for _, ax in figs]
BASE_LABELS = [[], [], []]   # (x, y, base, colour, offset) per panel, placed after the axes
S_ALL = []
for name, col, mk in ENG:
    key, pre = RADIX_KEYS[name]
    d = radix_frame(key)
    Cs = [d["bvn_cycle"].mean()] + [d[f"{pre}_{B}_cycle"].mean() for B in BASES]
    Ns = [d["bvn_perms"].mean()] + [d[f"{pre}_{B}_perms"].mean() for B in BASES]
    Ts = [d["bvn_runtime"].mean() * 1000] + [d[f"{pre}_{B}_time"].mean() * 1000 for B in BASES]
    # standard deviation across the matrices: dashed lines at T +- sd, the band between them
    # shaded at 15% (Chen K10, Ofek 2026-11-26)
    Tsd = [d["bvn_runtime"].std() * 1000] + [d[f"{pre}_{B}_time"].std() * 1000 for B in BASES]
    if name != "WWFA":                       # the strong engines' baseline cycle is S
        S_ALL.append(Cs[0])
    print(name, [f"{x:.1f}" for x in Ts], [f"{x:.0f}" for x in Cs], [f"{x:.0f}" for x in Ns])
    # (a) time against cycle, (b) time against N, (c) cycle against N
    for k, (ax, xs, ys) in enumerate(((axes[0], Cs, Ts), (axes[1], Ns, Ts), (axes[2], Ns, Cs))):
        if k < 2:
            up, lo = [y + s for y, s in zip(Ts, Tsd)], [y - s for y, s in zip(Ts, Tsd)]
            ax.fill(list(xs) + list(xs)[::-1], up + lo[::-1], color=col, alpha=.15, lw=0)
            for ys_ in (up, lo):
                ax.plot(xs, ys_, "--", color=col, lw=.7, alpha=.8, dashes=(3, 2))
        ax.plot(xs, ys, "-", color=col, lw=1.3, alpha=.95)
        ax.plot(xs, ys, mk, color=col, ms=5, markeredgecolor="white",
                markeredgewidth=.5, label=name)
        for x, y, t in zip(xs, ys, LABELS):   # the base of each point, labelled below
            BASE_LABELS[k].append((x, y, t, col, LABEL_OFFSET.get((k, name, t), (3, 3))))
# C = S line: the strong engines' baseline cycle, S of the campaign's matrices
S0 = float(np.mean(S_ALL))


def _style9(ax, xlab, ylab, logx=False, logy=True):
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlab, labelpad=1)
    ax.set_ylabel(ylab, labelpad=1)
    ax.tick_params(pad=1.5)
    ax.grid(True, which="both", ls=":", lw=.35, alpha=.6)


NC = "number of configurations $N$"
_style9(axes[0], "cycle length $C$", r"$T_{\mathrm{parallel}}$ [ms]")
_style9(axes[1], NC, r"$T_{\mathrm{parallel}}$ [ms]", logx=True)
_style9(axes[2], NC, "cycle length $C$", logx=True, logy=False)
for ax in (axes[1], axes[2]):   # labelled ticks on the N axis
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator([t for t in N_TICKS if lo <= t <= hi]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
# cycle-length ticks every 1000: a label at 11500 would stand past the right edge of (a)
axes[0].xaxis.set_major_locator(FixedLocator([8000, 9000, 10000, 11000]))
# room left of C = S for the unsplit labels, which sit to the left of their markers
axes[0].set_xlim(left=7650)
for ax in (axes[0], axes[1]):   # headroom, so the labels of the top points stay inside the frame
    ax.set_ylim(top=ax.get_ylim()[1] * 1.5)
for ax in (axes[0], axes[1]):   # labelled time ticks in plain ms, not only the decades (Chen K3)
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator([v for v in T_TICKS if lo <= v <= hi]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
# the dotted line C = S, labelled S
axes[0].axvline(S0, color="gray", ls=":", lw=1.0)
axes[0].annotate("$S$", (S0, 0), xycoords=("data", "axes fraction"), xytext=(3, 3),
                 textcoords="offset points", color="gray")
axes[2].axhline(S0, color="gray", ls=":", lw=1.0)
axes[2].annotate("$S$", (0.62, S0), xycoords=("axes fraction", "data"), xytext=(0, 2),
                 textcoords="offset points", color="gray", ha="center")
def place_labels(ax, labs):
    """Label each point (x, y, text, colour, offset). Where the same text of several engines
    falls at nearly the same place (closer than 6% of the panel), one label in dark grey
    stands for all of them."""
    placed = []   # (axes-fraction position, text, annotation)
    toax = ax.transAxes.inverted()
    for x, y, t, col, off in labs:
        q = toax.transform(ax.transData.transform((x, y)))
        near = [a for p_, t_, a in placed if t_ == t and np.hypot(*(q - p_)) < 0.06]
        if near:
            near[0].set_color("0.25")
            continue
        mode = off[2] if len(off) == 3 else ""   # "r": left of the marker; "c": centred below it
        placed.append((q, t, ax.annotate(
            t, (x, y), xytext=off[:2], textcoords="offset points", fontsize=FS, color=col,
            ha={"r": "right", "c": "center"}.get(mode, "left"), va="center" if mode == "r" else ("top" if mode == "c" else "baseline"),
            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])))


# the base of every point
for ax, labs in zip(axes, BASE_LABELS):
    place_labels(ax, labs)
# legend in one row above (a) (Chen K2); panel (c) is no longer in the paper (Chen K8, Ofek
# 2026-11-25: dropped rather than added to Figure 4 as well), but is still written
LEG_H = 0.30                                   # inches added on top of (a) for the legend
figs[0][0].set_size_inches(3.33, 2.05 + LEG_H)
figs[0][0].legend(*axes[0].get_legend_handles_labels(), loc="upper center",
                  bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, handlelength=0.8,
                  handletextpad=0.25, columnspacing=0.7, borderaxespad=0.1)
# the same margins in inches for every panel, so the plotting areas of (a) and (b) line up
# when stacked (tight_layout fitted each panel to its own tick labels, and they did not)
ML, MR, MB, MT = 0.50, 0.06, 0.40, 0.06
for (fig, _), tag in zip(figs, "abc"):
    w, h = fig.get_size_inches()
    top = MT + (LEG_H if tag == "a" else 0)
    fig.subplots_adjust(left=ML / w, right=1 - MR / w, bottom=MB / h, top=1 - top / h)
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(PLOTS, f"scaled_tradeoff_{tag}.{e}"), dpi=200)
    plt.close(fig)
print("saved scaled_tradeoff_a/b/c (three panels, one column wide each)")

# ============================== EULER ==============================
def rerun(e): return sorted(glob.glob(f"{ROOT}/run/rerun_2026/{e}/*/results_stats.csv"))[-1]
def euler(p): return sorted(glob.glob(f"{ROOT}/euler_exp/eng_{p}_dense_d4/*/results_stats.csv"))[-1]
KEYS = [("Hungarian", "max", "maximum"), ("GW Dynamic", "heavy", "heavy"),
        ("GW Static", "heavy_static", "heavy_static"), ("WFA", "wfa", "wfa")]
COL = {n: c for n, c, _ in ENG}
MK = {n: m for n, _, m in ENG}
if CAMPAIGN != "variance_2026":
    KEYS = [("WWFA", "wwfa", "wwfa") if k[1] == "wfa" else k for k in KEYS]

# Two separate panels, euler_tradeoff_{a,b}.pdf, one column wide, set as LaTeX subfigures; text
# as in Figure 3 (LaTeX, Linux Libertine, 9 pt), and in Figure 3's order (Chen K9): (a) time
# against cycle length, (b) time against N. Every point in (b) carries its depth (unsplit, 1-4).
# In (a) the three strong engines sit on the line C = S at overlapping heights, so there only
# WWFA's points are labelled; the strong engines' depths are read from (b), at the same height.
DLABELS = ["unsplit"] + [str(d) for d in DEPTHS]
# (panel, engine, depth) -> offset in points, where the default (3, 3) hits a point or a line
# GW Static's points in (b) stand just left of WWFA's, so its labels go on the left
DEPTH_OFFSET = {(1, "GW Static", "unsplit"): (-4, 0, "r"),
                (1, "WWFA", "unsplit"): (-5, 0, "r")}   # inside the right edge
DEPTH_OFFSET.update({(1, "GW Static", str(d)): (-9, -3) for d in DEPTHS})
efigs = [plt.subplots(figsize=(3.33, 2.05)) for _ in range(2)]
eaxes = [ax for _, ax in efigs]
DEPTH_LABELS = [[], []]
S_ref = None
for name, re_, pre in KEYS:
    de = euler_frame(pre)
    N = [de["bvn_perms"].mean()] + [de[f"euler_heuristic_ruined_depth{d}_perms"].mean() for d in DEPTHS]
    C = [de["bvn_cycle"].mean()] + [de[f"euler_heuristic_ruined_depth{d}_cycle"].mean() for d in DEPTHS]
    T = [de["bvn_runtime"].mean()*1000] + [
        (de[f"euler_heuristic_ruined_depth{d}_time"].mean()
         + de[f"euler_heuristic_ruined_depth{d}_split"].mean())*1000 for d in DEPTHS]
    # standard deviation across the matrices of split + slowest component (Chen K10)
    Tsd = [de["bvn_runtime"].std()*1000] + [
        (de[f"euler_heuristic_ruined_depth{d}_time"]
         + de[f"euler_heuristic_ruined_depth{d}_split"]).std()*1000 for d in DEPTHS]
    if name == "Hungarian":
        S_ref = C[0]
    for k, (ax, xs) in enumerate(((eaxes[0], C), (eaxes[1], N))):
        up, lo = [y + s for y, s in zip(T, Tsd)], [y - s for y, s in zip(T, Tsd)]
        ax.fill(list(xs) + list(xs)[::-1], up + lo[::-1], color=COL[name], alpha=.15, lw=0)
        for ys_ in (up, lo):
            ax.plot(xs, ys_, "--", color=COL[name], lw=.7, alpha=.8, dashes=(3, 2))
        ax.plot(xs, T, "-", color=COL[name], lw=1.3, alpha=.95)
        ax.plot(xs, T, MK[name], color=COL[name], ms=5,
                markeredgecolor="white", markeredgewidth=.5, label=name)
        if k == 1 or name == "WWFA":
            for x, y, t in zip(xs, T, DLABELS):
                DEPTH_LABELS[k].append((x, y, t, COL[name], DEPTH_OFFSET.get((k, name, t), (3, 3))))
_style9(eaxes[0], "cycle length $C$", r"$T_{\mathrm{parallel}}$ [ms]")
_style9(eaxes[1], NC, r"$T_{\mathrm{parallel}}$ [ms]", logx=True)
lo, hi = eaxes[1].get_xlim()
eaxes[1].xaxis.set_major_locator(FixedLocator([t for t in N_TICKS + [10000] if lo <= t <= hi]))
eaxes[1].xaxis.set_minor_locator(NullLocator())
eaxes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
for ax in eaxes:
    ax.set_ylim(top=ax.get_ylim()[1] * 1.5)
    lo, hi = ax.get_ylim()   # the same plain-ms time ticks as Figure 3 (Chen K3)
    ax.yaxis.set_major_locator(FixedLocator([v for v in T_TICKS if lo <= v <= hi]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
eaxes[0].axvline(S_ref, color="gray", ls=":", lw=1.0)
eaxes[0].annotate("$S$", (S_ref, 1), xycoords=("data", "axes fraction"), xytext=(3, -12),
                  textcoords="offset points", color="gray")
for ax, labs in zip(eaxes, DEPTH_LABELS):
    place_labels(ax, labs)
# legend in one row above (a) and the same fixed margins as Figure 3, so the panels of both
# figures line up (Chen K9)
efigs[0][0].set_size_inches(3.33, 2.05 + LEG_H)
efigs[0][0].legend(*eaxes[0].get_legend_handles_labels(), loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, handlelength=0.8,
                   handletextpad=0.25, columnspacing=0.7, borderaxespad=0.1)
for (fig, _), tag in zip(efigs, "ab"):
    w, h = fig.get_size_inches()
    top = MT + (LEG_H if tag == "a" else 0)
    fig.subplots_adjust(left=ML / w, right=1 - MR / w, bottom=MB / h, top=1 - top / h)
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(PLOTS, f"euler_tradeoff_{tag}.{e}"), dpi=200)
    plt.close(fig)
print(f"saved euler_tradeoff_a/b (two panels, one column wide each)  C=S={S_ref:.0f}")
