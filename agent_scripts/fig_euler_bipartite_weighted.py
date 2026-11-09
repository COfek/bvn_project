"""Figure 2 (chosen by Ofek, 2026-11-10): one line per communication pair on Figure 1's matrix.

Same matrix and colourings as fig_euler_bipartite_paper.py ((a) alternating colouring,
(b) sparse Euler splitting, four tiers), but each communication pair (i,j) is ONE straight
line of width proportional to D_ij instead of D_ij parallel edges:
    consistent pair      one colour, width D_ij
    non-consistent pair  two side-by-side strips, red of width (red edges), blue of width
                         (blue edges), together D_ij
Widths are drawn as polygons in data units on an equal-aspect axis, so a width of w edges is
exactly w times one edge's width everywhere.

Output: paper/plots_pdf/euler_bipartite_paper.pdf|png (the name the paper includes)
"""
from __future__ import annotations
from _paths import PLOTS, REPO  # repository-relative paths
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from src.algorithms.euler_splitting import euler_split_once
from src.utils.matrix_generator import generate_matrix

OUT = str(PLOTS)
plt.rcParams.update({"font.family": "serif"})

N, SEED, MAXW = 6, 40, 4.0          # the matrix of fig_splitting_paper.py
C_RED, C_BLUE = "#D62728", "#0072B2"
X_ROW, X_COL, DY = 0.0, 3.0, 1.0    # vertex columns and row spacing (data units)
UNIT = 0.040                       # width of one edge (data units)


def strip(ax, p, q, off0, off1, color, z, edge=None, lw=0.0):
    """Quadrilateral between offsets off0 and off1 (perpendicular to p->q)."""
    d = q - p
    nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
    pts = [p + nrm * off0, q + nrm * off0, q + nrm * off1, p + nrm * off1]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=edge or color, lw=lw,
                         zorder=z))


def draw(ax, red, blue, title):
    ys = {i: (N - 1 - i) * DY for i in range(N)}
    consistent = total = 0
    for i in range(N):
        for j in range(N):
            r, b = int(red[i, j]), int(blue[i, j])
            if r + b == 0:
                continue
            total += 1
            consistent += (r == 0) or (b == 0)
            p, q = np.array([X_ROW, ys[i]]), np.array([X_COL, ys[j]])
            w = (r + b) * UNIT
            # red on one side of the centre line, blue on the other, together width D_ij
            # a white outline around the whole pair line (not between its two colours)
            # separates it from the lines it crosses
            z = 3 + 0.01 * total
            lo = -w / 2
            strip(ax, p, q, lo, w / 2, "white", z, edge="white", lw=1.2)
            if r:
                strip(ax, p, q, lo, lo + r * UNIT, C_RED, z + 0.001, lw=0.2)
            if b:
                strip(ax, p, q, lo + r * UNIT, w / 2, C_BLUE, z + 0.001, lw=0.2)
    for x, tag, col in ((X_ROW, "r", "#444444"), (X_COL, "c", "#888888")):
        for v in range(N):
            ax.plot(x, ys[v], "o", color=col, ms=17, zorder=6)
            ax.text(x, ys[v], f"{tag}{v}", ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold", zorder=7)
    ax.set_xlim(X_ROW - 0.35, X_COL + 0.35)
    ax.set_ylim(-0.6, (N - 1) * DY + 0.4)
    ax.set_aspect("equal")
    ax.text(0.5, -0.01, f"{title}\n{consistent} consistent, {total - consistent} non-consistent",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax.axis("off")
    return consistent, total


rng = np.random.default_rng(SEED)
D = generate_matrix(N, k=N, max_weight=MAXW, rng=rng).astype(np.int64)
r_e, b_e = euler_split_once(D, split_method="euler")
r_h, b_h = euler_split_once(D, split_method="heuristic_ruined")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6))
ca = draw(axes[0], r_e, b_e, "(a) alternating colouring")
cb = draw(axes[1], r_h, b_h, "(b) sparse Euler splitting")

# legend: colours, and a width key (1 and 4 edges)
from matplotlib.patches import Patch
handles = [Patch(color=C_RED, label="red edges"), Patch(color=C_BLUE, label="blue edges")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False,
           bbox_to_anchor=(.5, -.02))
fig.tight_layout(rect=[0, .06, 1, 1])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"euler_bipartite_paper.{ext}"), dpi=200,
                bbox_inches="tight")
plt.close(fig)
print(f"weighted: consistent (a) {ca[0]} of {ca[1]}, (b) {cb[0]} of {cb[1]}; "
      f"max D_ij {int(D.max())}")
