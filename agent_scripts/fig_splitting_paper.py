"""Visualization of the two splittings, for the framework chapter (Figure 3).

Four columns, one per splitting, all applied to the SAME demand matrix; each
column reads top-down as the arithmetic it represents (Chen, 2026-10-12: vertical,
panels named (a)-(d) in the caption):

    (a) radix-split(D,2)   D = 2^2*Delta_2 + 2^1*Delta_1 + 2^0*Delta_0   NOT regular
    (b) radix-split(D,4)   D = 4^1*Delta_1 + 4^0*Delta_0                 NOT regular
    (c) euler-split(D,1)   D = red + blue                    both halves (S/2)-regular
    (d) euler-split(D,1)   same-direction heuristic          also regular, and sparser

Layout notes:
  * each column is titled by its panel letter and, below it, the operator name of
    Section 5, so the figure and the text name the same things;
  * each Radix plane carries its weight as a left-hand label, written as a power of
    the base, which shows where the weight comes from. Euler halves carry no label:
    they are summed with coefficient 1;
  * no row or column sums are drawn -- the regularity claim is left to the caption;
  * the figure is one text column wide (3.4 in); a full-width version
    (figsize 7.0 x 1.45*nrow+0.5) is pending Chen's view.

Output: paper/plots_pdf/splitting_overview.pdf  (vector, one column wide)
"""
from __future__ import annotations
from _paths import PLOTS, REPO  # repository-relative paths

import os
import sys


import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.algorithms.euler_splitting import euler_split_once
from src.utils.matrix_generator import generate_matrix

PLOTS = str(PLOTS)
N = 6


def is_regular(M):
    return len(set(M.sum(1))) == 1 and len(set(M.sum(0))) == 1


def planes_of(M, base):
    """Digit planes, most significant first."""
    out, tmp = [], M.copy()
    while tmp.max() > 0:
        out.append(tmp % base)
        tmp //= base
    return out[::-1]


def draw(ax, M):
    """One matrix, cells square, value in each cell, a dot for zero."""
    n = M.shape[0]
    vmax = max(1, int(M.max()))
    ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="equal")
    for i in range(n):
        for j in range(n):
            v = int(M[i, j])
            ax.text(j, i, str(v) if v else "\u00b7", ha="center", va="center",
                    fontsize=6.4, color="black" if v < 0.65 * vmax else "white")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def coeff(base, n_planes, i):
    return f"${base}^{{{n_planes - 1 - i}}}\\!\\cdot$"


rng = np.random.default_rng(40)
D = generate_matrix(N, k=N, max_weight=4.0, rng=rng).astype(np.int64)
S = int(D.sum(1)[0])
assert S % 2 == 0, f"S={S} must be even for a clean depth-1 split"

p2, p4 = planes_of(D, 2), planes_of(D, 4)
red, blue = (x.astype(int) for x in euler_split_once(D.astype(float)))
red_h, blue_h = (x.astype(int) for x in
                 euler_split_once(D.astype(float), split_method="heuristic"))

cols = [
    ("(a)\nradix-split$(D,2)$", [(coeff(2, len(p2), i), m) for i, m in enumerate(p2)]),
    ("(b)\nradix-split$(D,4)$", [(coeff(4, len(p4), i), m) for i, m in enumerate(p4)]),
    ("(c)\neuler-split$(D,1)$", [("", red), ("", blue)]),
    ("(d)\neuler-split$(D,1)$\nheuristic", [("", red_h), ("", blue_h)]),
]
nrow = 1 + max(len(parts) for _, parts in cols)

fig, axes = plt.subplots(nrow, len(cols), figsize=(3.4, 0.86 * nrow + 0.35))
fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.005,
                    wspace=0.55, hspace=0.42)

for c, (title, parts) in enumerate(cols):
    draw(axes[0][c], D)
    # va="bottom" keeps the top matrices level whatever the title's line count
    axes[0][c].set_title(title, fontsize=6.6, pad=3, va="bottom", linespacing=1.15)
    for r in range(1, nrow):
        ax = axes[r][c]
        if r - 1 < len(parts):
            lab, M = parts[r - 1]
            draw(ax, M)
            if lab:
                ax.set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center",
                              labelpad=4)
        else:
            ax.axis("off")
    # "=" below D, "+" between later parts, centred in the vertical gap
    for k in range(len(parts)):
        a, b = axes[k][c].get_position(), axes[k + 1][c].get_position()
        fig.text((a.x0 + a.x1) / 2, (a.y0 + b.y1) / 2, "=" if k == 0 else "+",
                 ha="center", va="center", fontsize=8)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(PLOTS, f"splitting_overview.{ext}"), dpi=240,
                bbox_inches="tight")

print(f"saved splitting_overview  (n={N}, S={S}, max entry {D.max()})")
print("  B=2 planes:", len(p2), "regular?", [is_regular(x) for x in p2])
print("  B=4 planes:", len(p4), "regular?", [is_regular(x) for x in p4])
print("  Euler halves regular?", is_regular(red), is_regular(blue),
      " nonzeros:", int((red > 0).sum()), int((blue > 0).sum()))
print("  Heuristic halves regular?", is_regular(red_h), is_regular(blue_h),
      " nonzeros:", int((red_h > 0).sum()), int((blue_h > 0).sum()))
