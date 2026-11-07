"""Figure 2: the colouring itself, on Figure 1's matrix (Chen P1, P11).

Every parallel edge drawn, under (a) the alternating colouring and (b) sparse Euler splitting
(four tiers). The colouring always comes from splitting the WHOLE matrix of
fig_splitting_paper.py; an option with fewer rows and columns is only a smaller drawing of it.

Three options to show Chen, each saved under its own name; the one chosen becomes
euler_bipartite_paper.pdf, the name the paper includes:
    whole   all 6 rows and columns
    small   r1, r3 and c0, c2
    medium  r1, r3, r4 and c0, c2, c5

Output: paper/plots_pdf/euler_bipartite_option_<name>.pdf|png
"""
from __future__ import annotations
from _paths import PLOTS, REPO  # repository-relative paths
import sys, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.algorithms.euler_splitting import euler_split_once
from src.utils.matrix_generator import generate_matrix

OUT = str(PLOTS)
plt.rcParams.update({"font.family": "serif"})

N, SEED, MAXW = 6, 40, 4.0          # the matrix of fig_splitting_paper.py
# The two colours also differ by line style, so the figure reads in greyscale (Chen M5)
C_RED, C_BLUE = "#D62728", "#0072B2"
LS_RED, LS_BLUE = "-", (0, (3, 1.2))

#          rows                columns             edge gap, width, figure size
OPTIONS = {
    "whole":  (tuple(range(6)), tuple(range(6)), 0.018, 0.9, (7.2, 4.2)),
    "small":  ((1, 3),          (0, 2),          0.085, 1.7, (6.4, 2.7)),
    "medium": ((1, 3, 4),       (0, 2, 5),       0.050, 1.4, (6.8, 3.3)),
}


def draw(ax, red, blue, title, rows, cols, gap, lw):
    """One curve per parallel edge, fanned out between its two vertices; red before blue."""
    ys = dict(enumerate(np.linspace(1, 0, len(rows))))
    consistent = total = 0
    t = np.linspace(0, 1, 60)
    for a, i in enumerate(rows):
        for b, j in enumerate(cols):
            r, bl = int(red[i, j]), int(blue[i, j])
            k = r + bl
            if k == 0:
                continue
            total += 1
            consistent += (r == 0) or (bl == 0)
            for e in range(k):
                off = (e - (k - 1) / 2) * gap
                y = ys[a] + (ys[b] - ys[a]) * t + off * 4 * t * (1 - t)
                is_red = e < r
                ax.plot(t, y, color=C_RED if is_red else C_BLUE, ls=LS_RED if is_red else LS_BLUE,
                        lw=lw, zorder=3, solid_capstyle="round")
    for side, verts, tag, col in ((0, rows, "r", "#444444"), (1, cols, "c", "#888888")):
        for a, v in enumerate(verts):
            ax.plot(side, ys[a], "o", color=col, ms=17, zorder=6)
            ax.text(side, ys[a], f"{tag}{v}", ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold", zorder=7)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.2, 1.2)
    ax.text(0.5, -0.01, f"{title}\n{consistent} consistent, {total - consistent} non-consistent",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax.axis("off")
    return consistent, total


rng = np.random.default_rng(SEED)
D = generate_matrix(N, k=N, max_weight=MAXW, rng=rng).astype(np.int64)
r_e, b_e = euler_split_once(D, split_method="euler")
r_h, b_h = euler_split_once(D, split_method="heuristic_ruined")

for name, (rows, cols, gap, lw, size) in OPTIONS.items():
    fig, axes = plt.subplots(1, 2, figsize=size)
    ca = draw(axes[0], r_e, b_e, "(a) alternating colouring", rows, cols, gap, lw)
    cb = draw(axes[1], r_h, b_h, "(b) sparse Euler splitting", rows, cols, gap, lw)
    fig.legend(handles=[
        Line2D([], [], color=C_RED, ls=LS_RED, lw=2, label="red edge"),
        Line2D([], [], color=C_BLUE, ls=LS_BLUE, lw=2, label="blue edge")],
        handlelength=3, loc="lower center", ncol=2, fontsize=8.5, frameon=False,
        bbox_to_anchor=(.5, -.03))
    fig.tight_layout(rect=[0, .08, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"euler_bipartite_option_{name}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    edges = sum(int(D[i, j]) for i in rows for j in cols)
    print(f"{name}: {edges} edges; consistent (a) {ca[0]} of {ca[1]}, (b) {cb[0]} of {cb[1]}")

pairs = int(np.count_nonzero(D))
print(f"whole matrix: {int(np.sum((r_e > 0) & (b_e > 0)))} of {pairs} non-consistent in (a), "
      f"{int(np.sum((r_h > 0) & (b_h > 0)))} of {pairs} in (b)")
