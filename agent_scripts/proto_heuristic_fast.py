"""
Prototype: same-direction Euler heuristic with O(1) amortized edge selection.

Bottleneck in _euler_split_heuristic_jit: every one of the sum(M) ~ n*S unit
edge traversals calls _choose_next_jit, which scans all n potential
neighbours to pick the best tier. This prototype replaces the scan with
lazy per-node candidate stacks:

  tier-1 stack: pairs whose locked direction matches this node's departure
                direction (push when the pair gets locked "aligned" to v)
  tier-2 stack: pairs not yet locked (initial adjacency)
  tier-3 stack: pairs locked against this node's direction (forced ruin)

Selection pops from t1, then t2, then t3, discarding stale entries (count
exhausted / tier changed since push). Each pair enters each stack at most
once => amortized O(1) per traversal step => total O(n*S + nnz).

Trade-off vs the original: "smallest M[i,j] first" tie-breaking inside each
tier is replaced by stack order (t2 is pre-sorted ascending once, so first
visits still prefer small entries; t1 order is traversal order). The test
below measures whether this costs any sparsity.

Run:  python -m agent_scripts.proto_heuristic_fast
"""
from __future__ import annotations

import time

import numpy as np
from numba import jit

from src.algorithms.euler_splitting import _euler_split_heuristic
from src.utils.matrix_generator import generate_matrix


# ---------------------------------------------------------------------------
# JIT implementation
# ---------------------------------------------------------------------------

@jit(nopython=True, nogil=True, cache=True)
def _heuristic_fast_jit(M, n):
    """
    Same-direction Euler split with O(1) amortized neighbour selection.

    Direction encoding in dir_arr: 0 = free, 1 = red (row->col), 2 = blue.
    Node ids: rows 0..n-1, cols n..2n-1.

    Per-node stacks hold *pair partner* node ids. Stale entries (pair count
    already zero, or tier changed) are discarded lazily on pop.
    """
    # ---- counts: adjacency matrix of remaining copies per pair ----
    cnt = np.zeros((n, n), dtype=np.int64)          # cnt[i, j] = remaining copies
    deg = np.zeros(2 * n, dtype=np.int64)           # remaining unit-degree per node
    nnz_per_row = np.zeros(n, dtype=np.int64)
    nnz_per_col = np.zeros(n, dtype=np.int64)
    total_edges = np.int64(0)
    for i in range(n):
        for j in range(n):
            k = M[i, j]
            if k > 0:
                cnt[i, j] = k
                deg[i] += k
                deg[n + j] += k
                nnz_per_row[i] += 1
                nnz_per_col[j] += 1
                total_edges += k

    dir_arr = np.zeros((n, n), dtype=np.int8)
    red = np.zeros((n, n), dtype=np.int64)
    blue = np.zeros((n, n), dtype=np.int64)
    if total_edges == 0:
        return red, blue

    # ---- per-node stacks (CSR-style storage) ----
    # t2 (free pairs): exactly the initial adjacency, sorted ascending by
    # M value so first visits prefer small entries (mirrors "smallest first").
    # t1 / t3: a pair is pushed at most once per side (on lock), so capacity
    # = nnz per node side.
    max_nnz_row = np.int64(0)
    max_nnz_col = np.int64(0)
    for i in range(n):
        if nnz_per_row[i] > max_nnz_row:
            max_nnz_row = nnz_per_row[i]
    for j in range(n):
        if nnz_per_col[j] > max_nnz_col:
            max_nnz_col = nnz_per_col[j]
    cap_r = max_nnz_row
    cap_c = max_nnz_col

    # stacks store the partner index (col j for row nodes, row i for col nodes)
    t1_r = np.empty((n, cap_r), dtype=np.int32); t1_r_top = np.zeros(n, dtype=np.int64)
    t2_r = np.empty((n, cap_r), dtype=np.int32); t2_r_top = np.zeros(n, dtype=np.int64)
    t3_r = np.empty((n, cap_r), dtype=np.int32); t3_r_top = np.zeros(n, dtype=np.int64)
    t1_c = np.empty((n, cap_c), dtype=np.int32); t1_c_top = np.zeros(n, dtype=np.int64)
    t2_c = np.empty((n, cap_c), dtype=np.int32); t2_c_top = np.zeros(n, dtype=np.int64)
    t3_c = np.empty((n, cap_c), dtype=np.int32); t3_c_top = np.zeros(n, dtype=np.int64)

    # Fill t2 stacks sorted DESCENDING by value so pops (LIFO) see the
    # SMALLEST values first - mirroring the original "smallest first" rule.
    vals_buf = np.empty(n, dtype=np.int64)
    idx_buf = np.empty(n, dtype=np.int32)
    for i in range(n):
        m = 0
        for j in range(n):
            if cnt[i, j] > 0:
                vals_buf[m] = -M[i, j]   # ascending sort of negative = descending value
                idx_buf[m] = j
                m += 1
        order = np.argsort(vals_buf[:m])
        for q in range(m):
            t2_r[i, q] = idx_buf[order[q]]
        t2_r_top[i] = m
    for j in range(n):
        m = 0
        for i in range(n):
            if cnt[i, j] > 0:
                vals_buf[m] = -M[i, j]
                idx_buf[m] = np.int32(i)
                m += 1
        order = np.argsort(vals_buf[:m])
        for q in range(m):
            t2_c[j, q] = idx_buf[order[q]]
        t2_c_top[j] = m

    # ---- traversal stack (unit edges) ----
    walk = np.empty(total_edges + 2, dtype=np.int64)

    for start in range(2 * n):
        if deg[start] == 0:
            continue
        walk[0] = start
        top = np.int64(1)

        while top > 0:
            v = walk[top - 1]
            if deg[v] == 0:
                top -= 1
                continue

            # ---- O(1) amortized selection ----
            u = np.int64(-1)
            if v < n:
                i = v
                # tier 1: locked red (aligned with row departure)
                while t1_r_top[i] > 0:
                    j = np.int64(t1_r[i, t1_r_top[i] - 1])
                    if cnt[i, j] > 0 and dir_arr[i, j] == 1:
                        u = n + j; break
                    t1_r_top[i] -= 1     # stale -> discard
                if u == -1:
                    # tier 2: free pairs
                    while t2_r_top[i] > 0:
                        j = np.int64(t2_r[i, t2_r_top[i] - 1])
                        if cnt[i, j] > 0 and dir_arr[i, j] == 0:
                            u = n + j; break
                        t2_r_top[i] -= 1
                if u == -1:
                    # tier 3: locked blue (conflict)
                    while t3_r_top[i] > 0:
                        j = np.int64(t3_r[i, t3_r_top[i] - 1])
                        if cnt[i, j] > 0:
                            u = n + j; break
                        t3_r_top[i] -= 1
            else:
                j = v - n
                while t1_c_top[j] > 0:
                    i2 = np.int64(t1_c[j, t1_c_top[j] - 1])
                    if cnt[i2, j] > 0 and dir_arr[i2, j] == 2:
                        u = i2; break
                    t1_c_top[j] -= 1
                if u == -1:
                    while t2_c_top[j] > 0:
                        i2 = np.int64(t2_c[j, t2_c_top[j] - 1])
                        if cnt[i2, j] > 0 and dir_arr[i2, j] == 0:
                            u = i2; break
                        t2_c_top[j] -= 1
                if u == -1:
                    while t3_c_top[j] > 0:
                        i2 = np.int64(t3_c[j, t3_c_top[j] - 1])
                        if cnt[i2, j] > 0:
                            u = i2; break
                        t3_c_top[j] -= 1

            if u == -1:
                # deg said edges remain but stacks are exhausted -> should
                # never happen; guard to avoid infinite loop.
                deg[v] = 0
                top -= 1
                continue

            # ---- traverse v -> u, colour by departure side ----
            if v < n:
                i = v; j = u - n
                red[i, j] += 1
                if dir_arr[i, j] == 0:
                    dir_arr[i, j] = 1
                    # aligned for row side (red), conflicting for col side
                    t1_r[i, t1_r_top[i]] = np.int32(j); t1_r_top[i] += 1
                    t3_c[j, t3_c_top[j]] = np.int32(i); t3_c_top[j] += 1
            else:
                i = u; j = v - n
                blue[i, j] += 1
                if dir_arr[i, j] == 0:
                    dir_arr[i, j] = 2
                    t1_c[j, t1_c_top[j]] = np.int32(i); t1_c_top[j] += 1
                    t3_r[i, t3_r_top[i]] = np.int32(j); t3_r_top[i] += 1

            cnt[i, j] -= 1
            deg[v] -= 1
            deg[u] -= 1
            walk[top] = u
            top += 1

    return red, blue


def euler_split_heuristic_fast(int_matrix: np.ndarray):
    n = int_matrix.shape[0]
    M = np.round(int_matrix).astype(np.int64)
    return _heuristic_fast_jit(M, n)


# ---------------------------------------------------------------------------
# Validation + benchmark
# ---------------------------------------------------------------------------

def _check(M, red, blue, label):
    S = int(M.sum(axis=1).max())
    half = S // 2
    assert np.all(red >= 0) and np.all(blue >= 0), f"{label}: negative entries"
    assert np.array_equal(red + blue, M), f"{label}: not lossless"
    assert np.all(red.sum(axis=1) == half) and np.all(red.sum(axis=0) == half), \
        f"{label}: red not {half}-regular (rows {red.sum(axis=1).min()}..{red.sum(axis=1).max()})"
    assert np.all(blue.sum(axis=1) == half) and np.all(blue.sum(axis=0) == half), \
        f"{label}: blue not {half}-regular"


def main():
    # ---- correctness fuzz (even S only; odd handled by caller in real code) ----
    print("Fuzz: ", end="")
    checked = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(4, 33))
        k = int(rng.integers(2, 17))
        mw = int(rng.integers(1, 9))
        M = generate_matrix(n=n, k=k, max_weight=mw, rng=rng).astype(np.int64)
        if int(M.sum(axis=1).max()) % 2 == 1:
            continue
        red, blue = euler_split_heuristic_fast(M)
        _check(M, red, blue, f"seed {seed}")
        checked += 1
    print(f"{checked} even-S matrices passed (lossless + doubly stochastic)")

    # ---- sparsity + speed vs original heuristic at paper scale ----
    print(f"\n{'seed':>5} {'method':<12} {'split ms':>9} {'red nnz':>8} {'split pairs':>12}")
    seeds, s = [], 10
    while len(seeds) < 5:   # collect 5 even-S matrices (internal split needs even S)
        rng = np.random.default_rng(s)
        if int(generate_matrix(n=256, k=256, max_weight=64, rng=rng).sum(axis=0)[0]) % 2 == 0:
            seeds.append(s)
        s += 1
    for seed in seeds:
        rng = np.random.default_rng(seed)
        M = generate_matrix(n=256, k=256, max_weight=64, rng=rng).astype(np.int64)
        for label, fn in (("heuristic", _euler_split_heuristic),
                          ("fast-proto", euler_split_heuristic_fast)):
            best = float("inf")
            for _ in range(3):
                t0 = time.perf_counter()
                red, blue = fn(M)
                best = min(best, time.perf_counter() - t0)
            _check(M, red, blue, label)
            split_pairs = int(np.sum((red > 0) & (blue > 0)))
            print(f"{seed:>5} {label:<12} {best*1e3:>9.1f} "
                  f"{np.count_nonzero(red):>8} {split_pairs:>12}")

    # ---- scaling check on the big config ----
    print("\nScaling (n=256, k=256):")
    for mw in [16, 64, 256]:
        s2 = 99
        while True:   # find an even-S matrix for this max_weight
            rng = np.random.default_rng(s2)
            M = generate_matrix(n=256, k=256, max_weight=mw, rng=rng).astype(np.int64)
            if int(M.sum(axis=1).max()) % 2 == 0:
                break
            s2 += 1
        t0 = time.perf_counter(); _euler_split_heuristic(M); t_old = time.perf_counter() - t0
        t0 = time.perf_counter(); euler_split_heuristic_fast(M); t_new = time.perf_counter() - t0
        S = int(M.sum(axis=1).max())
        print(f"  mw={mw:>4} S={S:>6}: heuristic {t_old*1e3:8.1f} ms -> fast {t_new*1e3:7.1f} ms "
              f"({t_old/t_new:5.1f}x)")


if __name__ == "__main__":
    main()
