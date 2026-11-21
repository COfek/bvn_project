"""Wide two-column trade-off figures for the paper.

Both are single-column (not figure*) so LaTeX can place them inside the subsection.
Each has two side-by-side panels giving computation time against BOTH other metrics,
scaled to fit within a single column horizontally.

  scaled_tradeoff.pdf       Radix(B,c), scaled only  -- for "Radix across engines"
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


# ============================== RADIX (scaled) ==============================
# a rerun (BVN_RUN_ROOT) times the digit extraction itself; only run/ needs it added
res = pd.read_csv(os.path.join(ROOT, RUN, "radix_scale", "scale_results.csv"))
if RUN == "run":
    res = correct_scale_results(res)
res["S_original"] = res.S_scaled / res.scale
res["C_orig"] = res.C_over_S * res.S_original
bvn = pd.read_csv(os.path.join(ROOT, RUN, "tm_workload", "tm_summary.csv"))
r = res[res.regime == "Uniform"]
b = bvn[bvn.regime == "Uniform (thesis)"]

# Three separate panels, scaled_tradeoff_{a,b,c}.pdf, each one column wide, so the paper sets the
# panel labels (a)-(c) as LaTeX subcaptions below each panel. All text is the paper's body font
# (Linux Libertine, 9 pt) at the size it is printed: the PDF is as wide as the column (3.33 in),
# so \includegraphics[width=\columnwidth] does not rescale it.
from matplotlib import patheffects
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
FS = 9
# Text is set by LaTeX with the fonts acmart uses (Linux Libertine, newtxmath), so it is the
# paper's own font. (Matplotlib draws the Libertine OpenType outlines wrongly.)
rc = {"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
      "legend.fontsize": FS, "text.usetex": True, "font.family": "serif",
      "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{libertine}"
                             r"\usepackage[libertine]{newtxmath}"}
LABELS = ["BvN"] + [str(B) for B in BASES]
N_TICKS = [500, 1000, 1500, 5000]
# where a base label sits, in points from its marker, when the default (3, 3) would hit a point,
# a line or another label: (panel 0-2, engine, base) -> offset
LABEL_OFFSET = {(0, "Hungarian", "BvN"): (3, -12), (0, "GW Static", "BvN"): (-8, -13),
                (0, "WWFA", "BvN"): (-2, 7), (0, "Hungarian", "8"): (3, -11),
                (0, "Hungarian", "4"): (4, -11),
                (1, "Hungarian", "8"): (-11, -2), (1, "Hungarian", "4"): (-11, -2),
                (1, "Hungarian", "2"): (-10, -2),
                (2, "Hungarian", "BvN"): (-6, 6), (2, "Hungarian", "2"): (6, -4),
                (2, "GW Dynamic", "8"): (5, -12), (2, "GW Dynamic", "16"): (3, -11)}
plt.rcParams.update(rc)
figs = [plt.subplots(figsize=(3.33, 2.05)) for _ in range(3)]
axes = [ax for _, ax in figs]
BASE_LABELS = [[], [], []]   # (x, y, base, colour, offset) per panel, placed after the axes
# C = S line at the mean demand of the matrices the Radix points average over
S0 = float(r.groupby("matrix").S_original.first().mean())
for name, col, mk in ENG:
    # The tight scale depends on the matrix's largest entry, so each matrix has its own: pick,
    # per matrix, the scale with the shortest cycle (c=1 or one of the two tight scales), then
    # average over the matrices, so every point uses all of them.
    e = r[r.engine == name]
    best = e.loc[e.groupby(["base", "matrix"]).C_over_S.idxmin()]
    tu = (best.groupby("base")
          .agg(C=("C_orig", "mean"), N=("N", "mean"), T=("runtime_ms", "mean")).reindex(BASES))
    bv = b[(b.engine == name) & (b.config == "BvN")]
    bC, bN, bT = float(bv.C.iloc[0]), float(bv.N.iloc[0]), float(bv.runtime_ms.iloc[0])
    Cs, Ns, Ts = [bC] + tu.C.tolist(), [bN] + tu.N.tolist(), [bT] + tu["T"].tolist()
    # (a) time against cycle, (b) time against N, (c) cycle against N
    for k, (ax, xs, ys) in enumerate(((axes[0], Cs, Ts), (axes[1], Ns, Ts), (axes[2], Ns, Cs))):
        ax.plot(xs, ys, "-", color=col, lw=1.3, alpha=.95)
        ax.plot(xs, ys, mk, color=col, ms=5, markeredgecolor="white",
                markeredgewidth=.5, label=name)
        for x, y, t in zip(xs, ys, LABELS):   # the base of each point, labelled below
            BASE_LABELS[k].append((x, y, t, col, LABEL_OFFSET.get((k, name, t), (3, 3))))
    # the unscaled counterpart, scale = 1, drawn faint and dashed
    gu = (r[(r.engine == name) & (r.scale == 1)].groupby('base')
          .agg(C=('C_orig', 'mean'), N=('N', 'mean'), T=('runtime_ms', 'mean'))
          .reindex(BASES))
    Cu, Nu, Tu_ = [bC] + gu.C.tolist(), [bN] + gu.N.tolist(), [bT] + gu['T'].tolist()
    for ax, xs_, ys_ in ((axes[0], Cu, Tu_), (axes[1], Nu, Tu_), (axes[2], Nu, Cu)):
        ax.plot(xs_, ys_, '--', color=col, lw=0.9, alpha=.35, zorder=1)


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
_style9(axes[0], "cycle length $C$", r"$T_{\mathrm{compute}}$ [ms]")
_style9(axes[1], NC, r"$T_{\mathrm{compute}}$ [ms]", logx=True)
_style9(axes[2], NC, "cycle length $C$", logx=True, logy=False)
for ax in (axes[1], axes[2]):   # labelled ticks on the N axis
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator([t for t in N_TICKS if lo <= t <= hi]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
for ax in (axes[0], axes[1]):   # headroom, so the labels of the top points stay inside the frame
    ax.set_ylim(top=ax.get_ylim()[1] * 1.5)
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
        placed.append((q, t, ax.annotate(
            t, (x, y), xytext=off, textcoords="offset points", fontsize=FS, color=col,
            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])))


# the base of every point
for ax, labs in zip(axes, BASE_LABELS):
    place_labels(ax, labs)
# legend in the empty upper right of (c), clear of every point and label
axes[2].legend(framealpha=.95, loc="upper right")
for (fig, _), tag in zip(figs, "abc"):
    fig.tight_layout(pad=0.2)
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
# as in Figure 3 (LaTeX, Linux Libertine, 9 pt). Every point carries its depth (BvN, 1-4). In (b)
# the three strong engines sit on the line C = S at overlapping heights, so there only WWFA's
# points are labelled; the strong engines' depths are read from (a), at the same height.
DLABELS = ["BvN"] + [str(d) for d in DEPTHS]
# (panel, engine, depth) -> offset in points, where the default (3, 3) hits a point or a line
# GW Static's points in (a) stand just left of WWFA's, so its labels go on the left
DEPTH_OFFSET = {(0, "GW Static", "BvN"): (-21, -3)}
DEPTH_OFFSET.update({(0, "GW Static", str(d)): (-9, -3) for d in DEPTHS})
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
    if name == "Hungarian":
        S_ref = C[0]
    for k, (ax, xs) in enumerate(((eaxes[0], N), (eaxes[1], C))):
        ax.plot(xs, T, "-", color=COL[name], lw=1.3, alpha=.95)
        ax.plot(xs, T, MK[name], color=COL[name], ms=5,
                markeredgecolor="white", markeredgewidth=.5, label=name)
        if k == 0 or name == "WWFA":
            for x, y, t in zip(xs, T, DLABELS):
                DEPTH_LABELS[k].append((x, y, t, COL[name], DEPTH_OFFSET.get((k, name, t), (3, 3))))
_style9(eaxes[0], NC, r"$T_{\mathrm{compute}}$ [ms]", logx=True)
_style9(eaxes[1], "cycle length $C$", r"$T_{\mathrm{compute}}$ [ms]")
lo, hi = eaxes[0].get_xlim()
eaxes[0].xaxis.set_major_locator(FixedLocator([t for t in N_TICKS + [10000] if lo <= t <= hi]))
eaxes[0].xaxis.set_minor_locator(NullLocator())
eaxes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
for ax in eaxes:
    ax.set_ylim(top=ax.get_ylim()[1] * 1.5)
eaxes[1].axvline(S_ref, color="gray", ls=":", lw=1.0)
eaxes[1].annotate("$S$", (S_ref, 1), xycoords=("data", "axes fraction"), xytext=(3, -12),
                  textcoords="offset points", color="gray")
for ax, labs in zip(eaxes, DEPTH_LABELS):
    place_labels(ax, labs)
eaxes[1].legend(framealpha=.95, loc="upper right")
for (fig, _), tag in zip(efigs, "ab"):
    fig.tight_layout(pad=0.2)
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(PLOTS, f"euler_tradeoff_{tag}.{e}"), dpi=200)
    plt.close(fig)
print(f"saved euler_tradeoff_a/b (two panels, one column wide each)  C=S={S_ref:.0f}")
