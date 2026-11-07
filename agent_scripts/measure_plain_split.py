"""How much does an Euler split without a heuristic keep of D's support? (paper 5.1.2, M9/M10)

For the paper's Euler matrices (n=256, k=256, max weight 64, seed 42 + i, i = 5..104),
one split at depth 1 by
  plain   the alternating colouring as implemented (split_method="euler")
  random  an arbitrary walk: Hierholzer's algorithm taking, at every vertex, a uniformly
          random unused edge (each vertex's edges shuffled once, then taken from the end)
  heur    the same-direction heuristic (split_method="heuristic"), for comparison
and records, per matrix: non-zeros of D, of each component matrix, the pairs in both
component matrices (non-consistent), and the same among pairs with D_ij >= 2.
Odd S: one perfect matching is taken out first and added to red, as in euler_split_once.

Usage: python agent_scripts/measure_plain_split.py [matrices=105] [random_seed=0]
Output: euler_exp/plain_split/plain_split.csv and a printout.
"""
import sys

import numpy as np
import pandas as pd

from _paths import REPO  # repository-relative paths
from src.algorithms.euler_splitting import _extract_one_perm, euler_split_once
from src.utils.matrix_generator import generate_matrix

OUT = REPO / "euler_exp" / "plain_split"


def random_walk_split(M, rng):
    """Euler split with a uniformly random unused edge chosen at every vertex."""
    M = np.round(M).astype(np.int64)
    n = M.shape[0]
    extra = None
    if int(M.sum(axis=1).max()) % 2 == 1:
        extra, M = _extract_one_perm(M)
    # adjacency: every unit edge appears in both endpoints' lists, with an edge id
    eid_i, eid_j = [], []
    for i, j in zip(*np.nonzero(M)):
        eid_i += [i] * int(M[i, j]); eid_j += [j] * int(M[i, j])
    eid_i, eid_j = np.array(eid_i), np.array(eid_j)
    E = len(eid_i)
    adj = [[] for _ in range(2 * n)]
    for e in range(E):
        adj[eid_i[e]].append(e); adj[n + eid_j[e]].append(e)
    for lst in adj:
        rng.shuffle(lst)
    used = np.zeros(E, dtype=bool)
    red = np.zeros((n, n), dtype=np.int64); blue = np.zeros((n, n), dtype=np.int64)
    for start in range(2 * n):
        stack = [start]
        while stack:
            v = stack[-1]
            lst = adj[v]
            while lst and used[lst[-1]]:
                lst.pop()
            if not lst:
                stack.pop(); continue
            e = lst.pop(); used[e] = True
            i, j = eid_i[e], eid_j[e]
            if v < n:                       # row -> column: red
                red[i, j] += 1; stack.append(n + j)
            else:                           # column -> row: blue
                blue[i, j] += 1; stack.append(i)
    if extra is not None:
        red = red + extra
    return red, blue


def stats(D, red, blue, method, i):
    both = (red > 0) & (blue > 0)
    multi = D >= 2
    return dict(matrix=i, method=method, nnz_D=int(np.count_nonzero(D)),
                nnz_red=int(np.count_nonzero(red)), nnz_blue=int(np.count_nonzero(blue)),
                pairs_both=int(both.sum()), pairs_multi=int(multi.sum()),
                pairs_multi_both=int((both & multi).sum()))


def main():
    n_mat = int(sys.argv[1]) if len(sys.argv) > 1 else 105
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    rows = []
    for i in range(5, n_mat):
        D = np.round(generate_matrix(n=256, k=256, max_weight=64.0,
                                     rng=np.random.default_rng(42 + i))).astype(np.int64)
        for method in ("plain", "heur"):
            r, b = euler_split_once(D.astype(float), split_method="euler" if method == "plain" else "heuristic")
            r, b = np.asarray(r).astype(np.int64), np.asarray(b).astype(np.int64)
            assert np.array_equal(r + b, D)
            rows.append(stats(D, r, b, method, i))
        r, b = random_walk_split(D, rng)
        assert np.array_equal(r + b, D) and len(set(r.sum(1)) | set(r.sum(0))) == 1
        rows.append(stats(D, r, b, "random", i))
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "plain_split.csv", index=False)
    print(f"{df.matrix.nunique()} matrices; means:")
    for m, s in df.groupby("method", sort=False):
        print(f"  {m:6s} nnz(D) {s.nnz_D.mean():8.0f}  each component {((s.nnz_red + s.nnz_blue) / 2).mean():8.0f}"
              f" ({100 * ((s.nnz_red + s.nnz_blue) / 2 / s.nnz_D).mean():5.1f}% of nnz(D))  "
              f"pairs in both {100 * (s.pairs_both / s.nnz_D).mean():5.1f}%  "
              f"of pairs with D_ij>=2 {100 * (s.pairs_multi_both / s.pairs_multi).mean():5.1f}%")


if __name__ == "__main__":
    main()
