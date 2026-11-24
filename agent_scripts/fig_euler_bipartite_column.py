"""Figure 2 of the paper, one column wide (Ofek, 2026-11-24). It replaces the two-column drawing
of fig_euler_bipartite_weighted.py.

The same drawing as fig_euler_bipartite_weighted.py (Figure 1's matrix, (a) alternating
colouring, (b) sparse Euler splitting, one line per communication pair of width D_ij), set at
the column width (3.33 in) with the text in the paper's font (LaTeX, Linux Libertine, 9 pt),
as Figures 3 and 4. The two panels stay side by side. Non-consistent pairs are drawn at half
opacity behind every consistent pair, so the consistent ones stand out (Ofek, 2026-11-24).

Output: <paper>/plots_pdf/euler_bipartite_paper.pdf|png (the name the paper includes)
"""
from __future__ import annotations
from _paths import PLOTS, REPO  # repository-relative paths
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon

from src.algorithms.euler_splitting import euler_split_once
from src.utils.matrix_generator import generate_matrix

OUT = PLOTS
FS = 9
plt.rcParams.update({
    "font.size": FS, "legend.fontsize": FS, "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{libertine}"
                           r"\usepackage[libertine]{newtxmath}"})

N, SEED, MAXW = 6, 40, 4.0          # the matrix of fig_splitting_paper.py
C_RED, C_BLUE = "#D62728", "#0072B2"
X_ROW, X_COL, DY = 0.0, 3.0, 1.0    # vertex columns and row spacing (data units)
UNIT = 0.050                       # width of one edge (data units)


def strip(ax, p, q, off0, off1, color, z, edge=None, lw=0.0, alpha=1.0):
    """Quadrilateral between offsets off0 and off1 (perpendicular to p->q)."""
    d = q - p
    nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
    pts = [p + nrm * off0, q + nrm * off0, q + nrm * off1, p + nrm * off1]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=edge or color, lw=lw,
                         zorder=z, alpha=alpha))


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
            # non-consistent pairs (both colours) at half opacity, behind every consistent one
            mixed = r > 0 and b > 0
            a = 0.5 if mixed else 1.0
            z = (2 if mixed else 4) + 0.01 * total
            lo = -w / 2
            strip(ax, p, q, lo, w / 2, "white", z, edge="white", lw=0.8, alpha=a)
            if r:
                strip(ax, p, q, lo, lo + r * UNIT, C_RED, z + 0.001, lw=0.15, alpha=a)
            if b:
                strip(ax, p, q, lo + r * UNIT, w / 2, C_BLUE, z + 0.001, lw=0.15, alpha=a)
    for x, tag, col in ((X_ROW, "r", "#444444"), (X_COL, "c", "#888888")):
        for v in range(N):
            ax.plot(x, ys[v], "o", color=col, ms=14, zorder=6)
            ax.text(x, ys[v], rf"\textbf{{{tag}{v}}}", ha="center", va="center",
                    fontsize=FS - 1, color="white", zorder=7)
    ax.set_xlim(X_ROW - 0.3, X_COL + 0.3)
    ax.set_ylim(-0.45, (N - 1) * DY + 0.45)
    ax.set_aspect("equal")
    ax.text(0.5, -0.01, f"{title}\n{consistent} consistent,\n{total - consistent} non-consistent",
            transform=ax.transAxes, ha="center", va="top", fontsize=FS)
    ax.axis("off")
    return consistent, total


rng = np.random.default_rng(SEED)
D = generate_matrix(N, k=N, max_weight=MAXW, rng=rng).astype(np.int64)
r_e, b_e = euler_split_once(D, split_method="euler")
r_h, b_h = euler_split_once(D, split_method="heuristic_ruined")

fig, axes = plt.subplots(1, 2, figsize=(3.57, 3.8))
ca = draw(axes[0], r_e, b_e, "(a) alternating colouring")
cb = draw(axes[1], r_h, b_h, "(b) sparse Euler splitting")
handles = [Patch(color=C_RED, label="red edges"), Patch(color=C_BLUE, label="blue edges")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, handlelength=1.5,
           bbox_to_anchor=(.5, -.02))
fig.tight_layout(rect=[0, .07, 1, 1], w_pad=0.6)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"euler_bipartite_paper.{ext}"), dpi=300, bbox_inches="tight",
                pad_inches=0.02)
plt.close(fig)
print(f"column: consistent (a) {ca[0]} of {ca[1]}, (b) {cb[0]} of {cb[1]}; max D_ij {int(D.max())}")
