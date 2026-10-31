"""Paired comparison of the Euler split heuristics "heuristic" (three tiers, the paper's)
and "heuristic_ruined" (a fourth tier that holds already-ruined pairs back).

Same matrices as the canonical Euler campaign (euler_exp/eng_*_dense_d4): matrix i is
generate_matrix(256, k=256, max_weight=64, rng=default_rng(42 + i)), i = 0..104; the
tables discard i < 5. For every matrix, depth 1-4 and leaf engine, both methods run back
to back through decompose_euler_framework (the call the campaign makes), in alternating
order (heuristic first on even i, heuristic_ruined first on odd i), so the timing
comparison is paired and does not see the between-run spread of separate campaigns.

N, C and the leaf non-zeros are deterministic; only the times are measured.

Usage: python agent_scripts/euler_ruined_tier_compare.py [matrices=105] [tag]

Outputs (euler_exp/ruined_tier/, each name followed by the optional tag):
  paired_results.csv  per (matrix, engine, depth, method): N, C, split_ms, leaf_ms, leaves
  paired_splits.csv   per (matrix, method, depth): non-zeros over all leaves, the largest
                      leaf's non-zeros, and the pairs ruined at that level
  environment.json    library versions, CPU count, start and end time
"""
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime

import numba
import numpy as np
import scipy

from _paths import REPO  # repository-relative paths
from src.algorithms.euler_splitting import decompose_euler_framework, euler_split_once
from src.utils.matrix_generator import generate_matrix

N_DIM, K, W, SEED = 256, 256, 64.0, 42
N_MATRICES = int(sys.argv[1]) if len(sys.argv) > 1 else 105
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
DEPTHS = (1, 2, 3, 4)
METHODS = ("heuristic", "heuristic_ruined")
ENGINES = ("maximum", "heavy", "heavy_static", "wfa")
OUT = REPO / "euler_exp" / "ruined_tier"


def matrix(i):
    return generate_matrix(n=N_DIM, k=K, max_weight=W, rng=np.random.default_rng(SEED + i))


def split_stats(M, method):
    """Non-zeros of the leaves at every depth, and the pairs each level ruins."""
    rows = []
    leaves = [np.round(M).astype(np.int64)]
    for d in DEPTHS:
        nxt, ruined = [], 0
        for L in leaves:
            if int(L.sum(axis=1).max()) <= 1:
                nxt.append(L)
                continue
            red, blue = euler_split_once(L, split_method=method)
            ruined += int(np.count_nonzero(red)) + int(np.count_nonzero(blue)) - int(np.count_nonzero(L))
            nxt += [red, blue]
        leaves = nxt
        nnz = [int(np.count_nonzero(L)) for L in leaves if np.any(L > 0)]
        rows.append(dict(depth=d, leaves=len(nnz), nnz_total=sum(nnz), nnz_max=max(nnz),
                         ruined_at_level=ruined))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    env = dict(start=datetime.now().isoformat(timespec="seconds"), python=platform.python_version(),
               numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
               cpus=os.cpu_count(), machine=platform.machine(), system=platform.platform(),
               n=N_DIM, k=K, max_weight=W, seed=SEED, matrices=N_MATRICES)

    # warm-up: compile every kernel both methods and all engines touch
    Mw = generate_matrix(n=N_DIM, k=K, max_weight=W, rng=np.random.default_rng(10_000))
    for eng in ENGINES:
        for method in METHODS:
            decompose_euler_framework(Mw, matching_method=eng, depth=2, split_method=method)
    print("warm-up done", flush=True)

    f_res = open(OUT / f"paired_results{TAG}.csv", "w", newline="")
    f_spl = open(OUT / f"paired_splits{TAG}.csv", "w", newline="")
    w_res = csv.DictWriter(f_res, ["matrix", "engine", "depth", "method", "order", "N", "C",
                                   "S", "split_ms", "leaf_ms", "leaves"])
    w_spl = csv.DictWriter(f_spl, ["matrix", "method", "depth", "leaves", "nnz_total",
                                   "nnz_max", "ruined_at_level", "nnz_D"])
    w_res.writeheader(); w_spl.writeheader()

    t_start = time.perf_counter()
    for i in range(N_MATRICES):
        t_i = time.perf_counter()
        M = matrix(i)
        S = float(M.sum(axis=1)[0])
        order = METHODS if i % 2 == 0 else METHODS[::-1]
        for method in METHODS:
            for r in split_stats(M, method):
                w_spl.writerow(dict(matrix=i, method=method, nnz_D=int(np.count_nonzero(M)), **r))
        for eng in ENGINES:
            for d in DEPTHS:
                for pos, method in enumerate(order):
                    comps, split_s, leaf_s, n_leaves = decompose_euler_framework(
                        M, matching_method=eng, depth=d, split_method=method)
                    w_res.writerow(dict(matrix=i, engine=eng, depth=d, method=method, order=pos,
                                        N=len(comps), C=float(sum(c.weight for c in comps)), S=S,
                                        split_ms=split_s * 1000, leaf_ms=leaf_s * 1000,
                                        leaves=n_leaves))
        f_res.flush(); f_spl.flush()
        el = time.perf_counter() - t_start
        eta = el / (i + 1) * (N_MATRICES - i - 1)
        print(f"[{i + 1}/{N_MATRICES}] {time.perf_counter() - t_i:.1f} s  "
              f"elapsed {el / 60:.1f} min  eta {eta / 60:.1f} min", flush=True)

    f_res.close(); f_spl.close()
    env["end"] = datetime.now().isoformat(timespec="seconds")
    (OUT / f"environment{TAG}.json").write_text(json.dumps(env, indent=2))
    print("PAIRED COMPARISON COMPLETE", flush=True)


if __name__ == "__main__":
    main()
