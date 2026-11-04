"""Paper-shaped version of the bipartite split figure (Figure Y).

The test-suite figure (tests/.../bipartite_seed00.png) is 3427x1647 px and carries six
heatmaps alongside the two graphs; at full text width in a two-column layout its edge
labels render at ~4pt. This version keeps ONLY the two bipartite graphs, sized so that the
labels remain legible after scaling to \\linewidth in a two-column paper, and leaves the
matrix-level view to the companion heatmap figure.

Output: paper/plots_pdf/euler_bipartite_paper.pdf|png
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
from tests.test_euler_splitting import _make_matrix

OUT = str(PLOTS)
plt.rcParams.update({"font.family": "serif"})

N, K, SEED, MAXW = 6, 3, 0, 6
# Okabe-Ito colours, safe under the common colour-vision deficiencies, and a different line
# style per class, so the figure also reads in greyscale (Chen M5)
C_RED, C_BLUE, C_SPLIT = "#D55E00", "#0072B2", "#000000"
LS_RED, LS_BLUE, LS_SPLIT = "-", (0, (4, 1.5)), (0, (1, 1.2))


def draw(ax, matrix, red, blue, title, max_labels=1):
    """Only non-consistent (purple) pairs are labelled, and at most `max_labels` of them:
    colour already distinguishes consistent from non-consistent, the panel title carries the counts,
    and a label on every arc makes the figure unreadable at column scale."""
    n = matrix.shape[0]
    row_ys = np.linspace(n - 1, 0, n)
    col_ys = np.linspace(n - 1, 0, n)

    # choose which split pairs get a label: the heaviest, as exemplars of the format
    splits = [(int(matrix[i, j]), i, j) for i in range(n) for j in range(n)
              if matrix[i, j] > 0 and red[i, j] > 0 and blue[i, j] > 0]
    labelled = {(i, j) for _, i, j in sorted(splits, reverse=True)[:max_labels]}

    for i in range(n):
        for j in range(n):
            m = int(matrix[i, j])
            if m == 0:
                continue
            r, b = int(red[i, j]), int(blue[i, j])
            ec, ls = ((C_RED, LS_RED) if (r > 0 and b == 0) else
                      (C_BLUE, LS_BLUE) if (b > 0 and r == 0) else (C_SPLIT, LS_SPLIT))
            lw = 1.0 + m ** 0.55
            sign = 1 if (i + j) % 2 == 0 else -1
            rad = sign * (0.12 + 0.22 * abs(i - j) / max(n - 1, 1))
            ax.annotate("", xy=(1.0, col_ys[j]), xytext=(0.0, row_ys[i]),
                        arrowprops=dict(arrowstyle="-", color=ec, lw=lw, alpha=.85, linestyle=ls,
                                        connectionstyle=f"arc3,rad={rad:.3f}"), zorder=2)
            if (i, j) in labelled:
                ax.text(0.5, (row_ys[i] + col_ys[j]) / 2 + rad * 0.65, f"{r}R+{b}B",
                        ha="center", va="center", fontsize=6.5, color=ec,
                        fontweight="bold", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=ec,
                                  linewidth=0.5, alpha=.95))
    for i in range(n):
        ax.plot(0, row_ys[i], "o", color="#444444", ms=15, zorder=6)
        ax.text(0, row_ys[i], f"r{i}", ha="center", va="center", fontsize=6.5,
                color="white", fontweight="bold", zorder=7)
        ax.plot(1, col_ys[i], "o", color="#888888", ms=15, zorder=6)
        ax.text(1, col_ys[i], f"c{i}", ha="center", va="center", fontsize=6.5,
                color="white", fontweight="bold", zorder=7)
    clean = int(np.sum((red > 0) ^ (blue > 0)))
    nsplit = int(np.sum((red > 0) & (blue > 0)))
    ax.set_xlim(-0.22, 1.22); ax.set_ylim(-0.55, n - 0.45)
    ax.set_title(f"{title}\ncommunication pairs: {clean} consistent, {nsplit} non-consistent",
                 fontsize=8.5, pad=6)
    ax.axis("off")


mat = _make_matrix(N, K, SEED, max_weight=MAXW).astype(np.int64)
r_e, b_e = euler_split_once(mat, split_method="euler")
r_h, b_h = euler_split_once(mat, split_method="heuristic")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
draw(axes[0], mat, r_e, b_e, "(a) alternating colouring")
draw(axes[1], mat, r_h, b_h, "(b) sparse Euler splitting")
fig.legend(handles=[
    Line2D([], [], color=C_RED, ls=LS_RED, lw=2, label="consistent comm. pair, red"),
    Line2D([], [], color=C_BLUE, ls=LS_BLUE, lw=2, label="consistent comm. pair, blue"),
    Line2D([], [], color=C_SPLIT, ls=LS_SPLIT, lw=2,
           label="non-consistent comm. pair, in both ($r$R+$b$B)")],
    handlelength=3,
    loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(.5, -.02))
fig.tight_layout(rect=[0, .06, 1, 1])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"euler_bipartite_paper.{ext}"),
                dpi=200, bbox_inches="tight")
print("saved euler_bipartite_paper")
print(f"  alternating: whole={int(np.sum((r_e>0)^(b_e>0)))} split={int(np.sum((r_e>0)&(b_e>0)))}"
      f"  red nnz={np.count_nonzero(r_e)} blue nnz={np.count_nonzero(b_e)}")
print(f"  heuristic  : whole={int(np.sum((r_h>0)^(b_h>0)))} split={int(np.sum((r_h>0)&(b_h>0)))}"
      f"  red nnz={np.count_nonzero(r_h)} blue nnz={np.count_nonzero(b_h)}")
print(f"  original nnz={np.count_nonzero(mat)}  S={int(mat.sum(1)[0])}")
