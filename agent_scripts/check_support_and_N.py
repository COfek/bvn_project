"""Why GW Static's N stays flat with Euler depth (Chapter 6 review, item R9, 2026-10-29).

Hypothesis: GW Static's number of configurations follows the support of the matrix (it needs
roughly one configuration per handful of non-zero entries), and the same-direction Euler split
divides the non-zeros between the two halves rather than copying them, so their total, and with
it N, barely changes with depth. Hungarian and GW Dynamic are not support-bound, so their N grows.

For two generated matrices (n=256, k=256, W_max=64; seeds 42 and 43) and depths 0-2, writes the
total N over the leaves, the total number of non-zero entries over the leaves, and their ratio.

Output: run/checks/support_and_N.csv
"""
import os

import numpy as np
import pandas as pd

from _paths import REPO  # repository-relative paths
from src.algorithms.euler_splitting import _leaf_bvn, euler_split_once
from src.utils.matrix_generator import generate_matrix


def leaves(D, depth):
    out = [D]
    for _ in range(depth):
        out = [h for M in out for h in euler_split_once(M.astype(float), split_method="heuristic")]
    return [m.astype(np.int64) for m in out]


rows = []
for seed in (42, 43):
    D = generate_matrix(256, k=256, max_weight=64, rng=np.random.default_rng(seed)).astype(np.int64)
    for engine in ("heavy_static", "maximum", "heavy"):
        for depth in (0, 1, 2):
            L = leaves(D, depth)
            N = sum(len(_leaf_bvn(m.astype(float), engine)[0]) for m in L)
            nnz = sum(int((m > 0).sum()) for m in L)
            rows.append(dict(seed=seed, engine=engine, depth=depth, N=N, nnz=nnz, N_per_nnz=N / nnz))

out = REPO / "run" / "checks"
os.makedirs(out, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out / "support_and_N.csv", index=False)
print(df.to_string(index=False))
