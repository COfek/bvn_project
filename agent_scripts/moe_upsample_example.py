"""For Chen (K18): one measured 8x8 MoE trace next to a 256x256 matrix sampled from it.

The sampling is src.utils.moe_trace.upsample (reading 1, Ofek 2026-11-26): each GPU of the
trace becomes 32 GPUs that each send the original GPU's row total; every token picks a
destination group with the trace's measured probability, then one of its 32 GPUs uniformly.
Writes Thesis/for_chen/moe_upsample_example.{pdf,png} and prints the checks (row and column
sums, block shares, S*). Not a paper figure.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
import moe_trace_experiment as mt
from src.utils.moe_trace import load_all, upsample

OUT = HERE.parents[2] / "for_chen"
FACTOR, SEED = 32, 0
tr = [x for x in load_all(mt.TRACES) if "mmlu" not in x.dataset][0]
M = tr.matrix.astype(np.int64)
U = upsample(M, FACTOR, np.random.default_rng(SEED))
n = M.shape[0]
G = U.reshape(n, FACTOR, n, FACTOR).sum(axis=(1, 3)) / FACTOR
print(tr.label)
print("row sums: trace", sorted(set(M.sum(1))), " sample", U.sum(1).min(), "-", U.sum(1).max())
print("column sums: trace", M.sum(0), "\n  sample, mean per group", U.sum(0).reshape(n, FACTOR).mean(1).round(0))
print(f"largest deviation of a block from the trace: {100 * np.abs(G - M).max() / M.max():.2f}%")
print(f"sample entries: mean {U.mean():.0f}, max {U.max()}; S* {max(U.sum(0).max(), U.sum(1).max())}"
      f" (trace {max(M.sum(0).max(), M.sum(1).max())})")

fig, ax = plt.subplots(1, 2, figsize=(9, 4.4))
v = M.max()
ax[0].imshow(M, cmap="viridis", vmin=0, vmax=v)
ax[0].set_title(f"measured trace, {n}x{n}\n{tr.label}", fontsize=9)
for i in range(n):
    for j in range(n):
        ax[0].text(j, i, M[i, j], ha="center", va="center", fontsize=6,
                   color="w" if M[i, j] < v * .6 else "k")
ax[1].imshow(U * FACTOR, cmap="viridis", vmin=0, vmax=v)
ax[1].set_title(f"one sample, {n * FACTOR}x{n * FACTOR} (each GPU split into {FACTOR})\n"
                f"entries x{FACTOR}, same colour scale", fontsize=9)
for k in range(1, n):
    ax[1].axhline(FACTOR * k - .5, color="w", lw=.3)
    ax[1].axvline(FACTOR * k - .5, color="w", lw=.3)
for a in ax:
    a.set_xlabel("destination GPU")
    a.set_ylabel("source GPU")
fig.text(0.5, 0.01, "Each new GPU sends its original GPU's total; every token picks a destination "
         "group with the trace's probability, then one of its 32 GPUs uniformly.",
         ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
OUT.mkdir(exist_ok=True)
for e in ("pdf", "png"):
    fig.savefig(OUT / f"moe_upsample_example.{e}", dpi=200)
print("saved", OUT / "moe_upsample_example.pdf")
