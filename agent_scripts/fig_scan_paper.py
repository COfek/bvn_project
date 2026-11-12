"""Figure: the Scan procedure of Section 5.2.2 (Chen, meeting of 2026-11-09, M15).

All on the matrix D of Figure 1 (n=6, S=10), so the reader meets no new matrix:

    (a) GW Dynamic, 4th extraction: the residual R (4-regular) scanned largest entry first.
        Each non-zero entry carries its position in the scan order; the scan accepts an
        entry when its row and its column are both free. It leaves row 5 unmatched.
    (b) The same R after Augment: the alternating path r5 -> c1 -> r1 -> c5 gives c1 to
        row 5 and moves row 1 to c5, so the matching becomes perfect.
    (c) WFA on D: the cells in order of i+j (anti-diagonals, top-left first), no repair.
        It leaves row 4 unmatched, a partial matching.

The scan order is the engines' own (np.argsort(-vals) over np.nonzero, as in
src/algorithms/sorted_array_matching.py), and the matchings are those the engines return
(sorted_array_matching_noaug, sorted_array_matching, wfa.wavefront_matching); the script
asserts both.

Output: paper/plots_pdf/scan_paper.{pdf,png}
"""
from __future__ import annotations
from _paths import PLOTS, REPO  # repository-relative paths

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(REPO))
from src.algorithms.sorted_array_matching import (  # noqa: E402
    sorted_array_matching, sorted_array_matching_noaug)
from src.algorithms.wfa import wavefront_matching  # noqa: E402

PLOTS = str(PLOTS)
plt.rcParams.update({"font.family": "serif", "font.size": 7})
C_ACC = "#1B7837"    # accepted / matched
C_MISS = "#D62728"   # unmatched row
C_SKIP = "0.55"      # visited, skipped

D = np.array([[0, 6, 0, 2, 0, 2], [3, 2, 2, 0, 0, 3], [0, 0, 3, 2, 3, 2],
              [2, 0, 3, 0, 2, 3], [3, 0, 2, 0, 5, 0], [2, 2, 0, 6, 0, 0]], float)
N = D.shape[0]


def gw_dynamic_residual(M, extraction):
    """The residual GW Dynamic scans at the given extraction (1-based)."""
    R = M.copy()
    for _ in range(extraction - 1):
        a = sorted_array_matching(R)
        lam = min(R[i, j] for i, j in a)
        for i, j in a:
            R[i, j] -= lam
    return R


def gw_order(M):
    rows, cols = np.nonzero(M)
    idx = np.argsort(-M[rows, cols])
    return [(int(rows[k]), int(cols[k])) for k in idx]


def wfa_order(M):
    return [(i, k - i) for k in range(2 * N - 1) for i in range(N)
            if 0 <= k - i < N and M[i, k - i] > 0]


def scan(order):
    rf, cf, out = set(), set(), []
    for i, j in order:
        if i not in rf and j not in cf:
            out.append((i, j)); rf.add(i); cf.add(j)
    return out


def grid(ax, M, title):
    vmax = M.max()
    ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=vmax * 1.6, aspect="equal")
    for i in range(N):
        for j in range(N):
            if M[i, j] == 0:
                ax.text(j, i, "·", ha="center", va="center", color="0.5")
    ax.set_xticks(range(N), [f"c{j}" for j in range(N)])
    ax.set_yticks(range(N), [f"r{i}" for i in range(N)])
    ax.tick_params(length=0, pad=2)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.set_title(title, y=-0.2, va="top", fontsize=7.5)


def value(ax, i, j, M, color="black", weight="normal"):
    ax.text(j, i + 0.05, f"{int(M[i, j])}", ha="center", va="center", fontsize=7.5,
            color=color, fontweight=weight)


def order_tag(ax, i, j, k, color):
    ax.text(j - 0.42, i - 0.40, str(k), ha="left", va="top", fontsize=5.4, color=color)


def box(ax, i, j, color, ls="-", lw=1.4):
    ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False, ec=color, lw=lw,
                           ls=ls, zorder=3))


def miss(ax, rows):
    for i in rows:
        ax.text(N - 0.35, i, "  unmatched", ha="left", va="center", color=C_MISS,
                fontsize=6.5, clip_on=False)
        ax.add_patch(Rectangle((-0.5, i - 0.5), N, 1, fill=False, ec=C_MISS, lw=0.9,
                               ls=(0, (2, 1.5)), zorder=2))


def panel_scan(ax, M, order, title):
    grid(ax, M, title)
    acc = scan(order)
    for k, (i, j) in enumerate(order, 1):
        ok = (i, j) in acc
        value(ax, i, j, M, "black" if ok else C_SKIP)
        order_tag(ax, i, j, k, "black" if ok else C_SKIP)
        if ok:
            box(ax, i, j, C_ACC)
    miss(ax, sorted(set(range(N)) - {i for i, _ in acc}))
    return acc


R = gw_dynamic_residual(D, 4)
gw_scan = scan(gw_order(R))
gw_full = sorted(map(tuple, sorted_array_matching(R)))
wfa = scan(wfa_order(D))
# the drawing must show what the engines return
assert sorted(gw_scan) == sorted(map(tuple, sorted_array_matching_noaug(R)))
assert sorted(wfa) == sorted(map(tuple, wavefront_matching(D > 0)))
assert len(gw_scan) == N - 1 and len(gw_full) == N and len(wfa) == N - 1
removed = sorted(set(gw_scan) - set(gw_full))
added = sorted(set(gw_full) - set(gw_scan))

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75))
panel_scan(axes[0], R, gw_order(R),
           "(a) GW Dynamic, 4th extraction:\nscan of $R$, largest entry first")

# (b) after Augment
ax = axes[1]
grid(ax, R, "(b) the same $R$ after Augment:\nan alternating path frees a column")
for i, j in zip(*np.nonzero(R)):
    on = (i, j) in gw_full
    value(ax, i, j, R, "black" if on else C_SKIP)
    if on:
        box(ax, i, j, C_ACC)
for i, j in removed:
    box(ax, i, j, C_MISS, ls=(0, (2, 1.5)), lw=1.1)
# the path: unmatched row -> its added cell -> the removed cell in that column -> next added
r0 = [i for i in range(N) if i not in {r for r, _ in gw_scan}][0]
path, cur = [], r0
while True:
    c = [j for i, j in added if i == cur][0]
    path.append((cur, c))
    nxt = [i for i, j in removed if j == c]
    if not nxt:
        break
    path.append((nxt[0], c))
    cur = nxt[0]
for (i1, j1), (i2, j2) in zip(path, path[1:]):
    ax.annotate("", xy=(j2, i2), xytext=(j1, i1), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=C_ACC, lw=1.0, shrinkA=6, shrinkB=6,
                                mutation_scale=7))
ax.text(N - 0.35, r0, "  matched", ha="left", va="center", color=C_ACC, fontsize=6.5,
        clip_on=False)

panel_scan(axes[2], D, wfa_order(D),
           "(c) WFA on $D$: anti-diagonals $i{+}j$,\ntop-left first, no repair")

fig.subplots_adjust(left=0.035, right=0.905, top=0.90, bottom=0.17, wspace=0.62)
os.makedirs(PLOTS, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(PLOTS, f"scan_paper.{ext}"), dpi=300)
print("scan:", sorted(gw_scan), "\naugment:", gw_full, "\npath:", path, "\nwfa:", sorted(wfa))
