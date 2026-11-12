"""Does wrapped WFA (WWFA) give the ~1.25 cycle ratio the BhaVaN dissertation reports?

Renganathan (PhD dissertation, Stanford, 2026), Fig. 4.5: on 1000 "uniformly random" matrices
(size not stated), a maximal-matching decomposition takes about 1.25 times the minimum number
of time slots (the largest line sum), and BhaVaN (bit planes, WWFA, unit decrements) about 1.1.

Untimed: only cycle length (sum of weights) and configuration count are measured, so the
machine's load does not matter. Engines are in src/algorithms/wwfa.py:
    orders  WFA (anti-diagonals), WWFA fixed start, WWFA rotating start
    steps   min (our WFA engine) and unit (the dissertation's)
Ratio = cycle / largest line sum of the matrix.

Workloads
  ours     the campaign's 256-port matrices (n=256, k=256, max_weight=64, seed 42 + i),
           matrices 5.. (the paper discards the first 5); whole matrix, and BhaVaN bit planes
  uniform  n x n, entries i.i.d. uniform integers in [0, 15] (not regular), several n
  regular  the campaign generator at small n (k = n permutations, weights 0..15)

Usage: .venv/Scripts/python.exe agent_scripts/wwfa_check.py [n_ours] [n_small]
Output: euler_exp/wwfa_check/results.csv and a printed summary.
"""
from __future__ import annotations
from _paths import REPO  # repository-relative paths
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(REPO))
from src.algorithms.wwfa import decompose_counts  # noqa: E402
from src.utils.matrix_generator import generate_matrix  # noqa: E402

OUT = os.path.join(str(REPO), "euler_exp", "wwfa_check")
ORDERS = {0: "WFA", 1: "WWFA fixed", 2: "WWFA rotating"}
STEPS = {0: "min", 1: "unit"}
N_OURS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_SMALL = int(sys.argv[2]) if len(sys.argv) > 2 else 200


def line_max(M):
    return float(max(M.sum(axis=0).max(), M.sum(axis=1).max()))


def whole(M, order, step):
    return decompose_counts(M.astype(np.float64).copy(), order, step)


def bitplanes(M, order, step):
    """BhaVaN: binary plane b of D, decomposed alone, every matching weighted 2^b."""
    Mi = M.astype(np.int64)
    cyc = cnt = 0.0
    b = 0
    while (Mi >> b).any():
        plane = ((Mi >> b) & 1).astype(np.float64)
        c, k = decompose_counts(plane, order, step)
        cyc += c * 2 ** b
        cnt += k
        b += 1
    return cyc, cnt


rows = []


def run(workload, idx, M, methods=("whole", "bitplanes")):
    L = line_max(M)
    for meth in methods:
        f = whole if meth == "whole" else bitplanes
        for o in ORDERS:
            for s in STEPS:
                if meth == "bitplanes" and s == 0:
                    continue        # a binary plane: min step and unit step coincide
                cyc, cnt = f(M, o, s)
                rows.append(dict(workload=workload, n=M.shape[0], matrix=idx, method=meth,
                                 order=ORDERS[o], step=STEPS[s], cycle=cyc, configs=cnt,
                                 line_max=L, ratio=cyc / L))


# ours: the campaign matrices
for i in range(5, 5 + N_OURS):
    M = generate_matrix(256, k=256, max_weight=64.0, rng=np.random.default_rng(42 + i))
    run("ours", i, M)
    print(f"ours matrix {i} done", flush=True)

rng = np.random.default_rng(2026)
for n in (4, 8, 16, 32, 64):
    for i in range(N_SMALL):
        run("uniform", i, rng.integers(0, 16, size=(n, n)).astype(np.float64))
        run("regular", i, generate_matrix(n, k=n, max_weight=15.0, rng=rng))
    print(f"small n={n} done", flush=True)

os.makedirs(OUT, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "results.csv"), index=False)
summary = (df.groupby(["workload", "n", "method", "order", "step"])["ratio"]
             .agg(["mean", "min", "max", "count"]).round(4))
summary.to_csv(os.path.join(OUT, "summary.csv"))
pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 500)
print(summary)
