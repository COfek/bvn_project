"""Paper-shaped matrix-level view of the Euler split (Figure X), as vector PDF.

Single strip: Original | alternating Red/Blue | heuristic Red/Blue, with the non-zero count
in each title. The point is that the alternating split leaves BOTH halves with the full
support of D (each is ~D/2), whereas the heuristic yields genuinely sparser halves.

Output: paper/plots_pdf/euler_split_heatmaps.pdf|png
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, r"C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.algorithms.euler_splitting import euler_split_once
from tests.test_euler_splitting import _make_matrix

OUT = r"C:\Users\ofekc\Desktop\Msc\Thesis\paper\plots_pdf"
plt.rcParams.update({"font.family": "serif"})

# seed 5 gives the cleanest demonstration: the alternating colouring leaves BOTH halves with
# the full support of D (nnz 21 = 21 = 21), while the heuristic reduces them to 16 and 17.
N, K, SEED, MAXW = 8, 4, 5, 8
mat = _make_matrix(N, K, SEED, max_weight=MAXW).astype(np.int64)
r_e, b_e = euler_split_once(mat, split_method="euler")
r_h, b_h = euler_split_once(mat, split_method="heuristic")
S = int(mat.sum(axis=1)[0])
vmax = int(mat.max())

panels = [
    (mat, "Original"),
    (r_e, "alternating: red"),
    (b_e, "alternating: blue"),
    (r_h, "heuristic: red"),
    (b_h, "heuristic: blue"),
]

fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.85))
for ax, (M, name) in zip(axes, panels):
    ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=vmax)
    for i in range(N):
        for j in range(N):
            if M[i, j] > 0:
                ax.text(j, i, str(int(M[i, j])), ha="center", va="center",
                        fontsize=4.4, color="black")
    ax.set_title(f"{name}\nnnz = {np.count_nonzero(M)}", fontsize=7, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)

# [point 27] title moved to caption: fig.suptitle(f"One Euler split of an $n{{=}}8$, $S{{=}}{S}$ matrix — "
# [point 27] title moved to caption: "the alternating colouring leaves both halves with the full support of $D$",
# [point 27] title moved to caption: fontsize=8, y=1.06)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"euler_split_heatmaps.{ext}"), dpi=200,
                bbox_inches="tight")
print("saved euler_split_heatmaps")
print(f"  original nnz={np.count_nonzero(mat)}  S={S}  max={vmax}")
print(f"  alternating: red nnz={np.count_nonzero(r_e)}  blue nnz={np.count_nonzero(b_e)}")
print(f"  heuristic  : red nnz={np.count_nonzero(r_h)}  blue nnz={np.count_nonzero(b_h)}")
print(f"  identical halves under alternating? {np.array_equal(r_e, b_e)}")
