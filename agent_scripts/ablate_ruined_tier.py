"""Ablation of the four-tier Euler split: which of its changes does the work?

The four-tier rule (split_method="heuristic_ruined") differs from the paper's three-tier
rule ("heuristic") in two ways:
  (a) where a pair is locked to the opposite direction, pairs already split are taken
      before whole ones (the three-tier rule mixes them, most recently locked first);
  (b) on its lock side, a split pair leaves tier 1 and waits with the other split pairs.
It also ranks split pairs below the pairs not yet locked. Seven variants of one kernel, a
copy of _euler_split_heuristic_fast_jit with five switches, separate the pieces:
  3tier       the paper's rule                          (= "heuristic")
  4tier       (a) + (b), split pairs after free pairs   (= "heuristic_ruined")
  a_only      (a) only: a split pair stays tier 1 on its lock side
  b_only      (b) only: on the other side a split pair stays among the opposite pairs,
              most recently locked first, as in the three-tier rule
  4tier_swap  (a) + (b), split pairs before free pairs
  a_swap      (a) only, split pairs before free pairs
  b_last      b_only with split pairs after the opposite pairs too, i.e. last
The copy is checked against the production kernel first: 3tier and 4tier must reproduce
"heuristic" and "heuristic_ruined" exactly, leaf by leaf, on every matrix and depth.

Expected, and checked here leaf by leaf: a_only, 4tier_swap and a_swap split exactly as
3tier, and b_only exactly as 4tier. Why:
- Under the three-tier rule a row gains a new red lock only from tier 2, which it reaches
  only when no red-locked pair of the row has copies left; so each row (and, for blue,
  each column) has at most one locked pair with copies at any time.
- A pair is split only at a forced pick, at a vertex with no free pair left; no pair at
  that vertex is locked afterwards, so the split pair stays its top forced candidate.
- Under three tiers the split pair is also the only tier-1 candidate at its lock-side
  vertex, so the walk alternates between its two vertices until its copies are used up:
  no split pair ever has copies outside that exchange. Hence (a) never applies, and
  ranking split pairs above free pairs reproduces the exchange.
- Ranking split pairs below free pairs at their lock-side vertex (b) is the one change
  that leaves a split pair with copies for later; at its other vertex it is reached first
  anyway, so the extra split tier there (a) changes nothing even with (b).
Three read-only reviews (2026-11-01) checked this argument against the code and with an
exact transliteration of the kernel, and found no counterexample. b_last is the one other
ranking of (b) that splits differently; small fuzz matrices suggested it might split fewer
pairs than 4tier, which is why it is here.

On the paper's workload (n=256, k=256, max weight 64, matrices i = 5..104 from seed 42),
for every variant and depth 1-4: the leaf non-zeros (all leaves, and the largest leaf), the
pairs in both halves created at that level (split_at_level = non-zeros of the two halves
minus those of the parent, summed over the level; at an odd-S level this includes pairs
that are in both halves only because the matching taken out first is added to red), and
whether the leaves equal those of 3tier and of 4tier. With
N=1, N and the cycle length C for each engine are also counted, for the variants in
N_VARIANTS and for 3tier and 4tier on the first N_CHECK matrices (a check against
paired_results.csv), through the same engine calls the Euler framework makes (_leaf_bvn)
but without building the dense configurations. That is only needed for a variant whose
leaves differ from both 3tier and 4tier. Nothing is timed, so the run may use many
processes.

Usage: python agent_scripts/ablate_ruined_tier.py [matrices=105] [workers=16] [N=1]
  N=0 skips the decompositions (split statistics only, a few minutes).
Outputs (euler_exp/ruined_tier/): ablation_splits.csv, and ablation_N.csv with N=1
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from numba import jit

from _paths import REPO  # repository-relative paths
from src.algorithms.bvn import bvn_decomposition
from src.algorithms.euler_splitting import _extract_one_perm, euler_split_once
from src.algorithms.sorted_array_matching import (_jit_decompose_sorted_dynamic,
                                                  _jit_decompose_sorted_static)
from src.algorithms.wfa import _jit_decompose_wfa
from src.utils.matrix_generator import generate_matrix

N_DIM, K, W, SEED, SKIP = 256, 256, 64.0, 42, 5
DEPTHS = (1, 2, 3, 4)
ENGINES = ("maximum", "heavy", "heavy_static", "wfa")
# variant: (split_tier, split_leaves_t1, split_before_free, split_tier_opposite, split_last)
VARIANTS = {"3tier": (0, 0, 0, 0, 0), "4tier": (1, 1, 0, 1, 0), "a_only": (1, 0, 0, 1, 0),
            "b_only": (1, 1, 0, 0, 0), "4tier_swap": (1, 1, 1, 1, 0), "a_swap": (1, 0, 1, 1, 0),
            "b_last": (1, 1, 0, 0, 1)}
PRODUCTION = {"3tier": "heuristic", "4tier": "heuristic_ruined"}
N_VARIANTS, N_CHECK = ("b_last",), 5
OUT = REPO / "euler_exp" / "ruined_tier"


@jit(nopython=True, nogil=True, cache=True)
def _split_variant_jit(M, n, split_tier, split_leaves_t1, split_before_free, split_tier_opposite,
                       split_last):
    """_euler_split_heuristic_fast_jit with switches (see the module docstring).
    dir_arr: 0 free, 1 red, 2 blue, 3 split (only when split_leaves_t1). A pair split
    enters the split list of its lock-side vertex, and of the other vertex too when
    split_tier_opposite; otherwise it is found there among the opposite pairs (t3).
    The split list is read before the free pairs (split_before_free), after them, or
    after the opposite pairs too (split_last)."""
    cnt = np.zeros((n, n), dtype=np.int64)
    deg = np.zeros(2 * n, dtype=np.int64)
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
    is_split = np.zeros((n, n), dtype=np.int8)
    red = np.zeros((n, n), dtype=np.int64)
    blue = np.zeros((n, n), dtype=np.int64)
    if total_edges == 0:
        return red, blue

    cap_r = np.int64(0)
    cap_c = np.int64(0)
    for i in range(n):
        if nnz_per_row[i] > cap_r:
            cap_r = nnz_per_row[i]
    for j in range(n):
        if nnz_per_col[j] > cap_c:
            cap_c = nnz_per_col[j]

    t1_r = np.empty((n, cap_r), dtype=np.int32); t1_r_top = np.zeros(n, dtype=np.int64)
    t2_r = np.empty((n, cap_r), dtype=np.int32); t2_r_top = np.zeros(n, dtype=np.int64)
    t3_r = np.empty((n, cap_r), dtype=np.int32); t3_r_top = np.zeros(n, dtype=np.int64)
    t1_c = np.empty((n, cap_c), dtype=np.int32); t1_c_top = np.zeros(n, dtype=np.int64)
    t2_c = np.empty((n, cap_c), dtype=np.int32); t2_c_top = np.zeros(n, dtype=np.int64)
    t3_c = np.empty((n, cap_c), dtype=np.int32); t3_c_top = np.zeros(n, dtype=np.int64)
    tr_r = np.empty((n, cap_r), dtype=np.int32); tr_r_top = np.zeros(n, dtype=np.int64)
    tr_c = np.empty((n, cap_c), dtype=np.int32); tr_c_top = np.zeros(n, dtype=np.int64)

    vals_buf = np.empty(n, dtype=np.int64)
    idx_buf = np.empty(n, dtype=np.int32)
    for i in range(n):
        m = 0
        for j in range(n):
            if cnt[i, j] > 0:
                vals_buf[m] = -M[i, j]
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

            u = np.int64(-1)
            if v < n:
                i = v
                while t1_r_top[i] > 0:
                    j = np.int64(t1_r[i, t1_r_top[i] - 1])
                    if cnt[i, j] > 0 and dir_arr[i, j] == 1:
                        u = n + j; break
                    t1_r_top[i] -= 1
                if u == -1 and split_tier != 0 and split_before_free != 0:
                    while tr_r_top[i] > 0:
                        j = np.int64(tr_r[i, tr_r_top[i] - 1])
                        if cnt[i, j] > 0:
                            u = n + j; break
                        tr_r_top[i] -= 1
                if u == -1:
                    while t2_r_top[i] > 0:
                        j = np.int64(t2_r[i, t2_r_top[i] - 1])
                        if cnt[i, j] > 0 and dir_arr[i, j] == 0:
                            u = n + j; break
                        t2_r_top[i] -= 1
                if u == -1 and split_tier != 0 and split_before_free == 0 and split_last == 0:
                    while tr_r_top[i] > 0:
                        j = np.int64(tr_r[i, tr_r_top[i] - 1])
                        if cnt[i, j] > 0:
                            u = n + j; break
                        tr_r_top[i] -= 1
                if u == -1:
                    while t3_r_top[i] > 0:
                        j = np.int64(t3_r[i, t3_r_top[i] - 1])
                        if cnt[i, j] > 0:
                            u = n + j; break
                        t3_r_top[i] -= 1
                if u == -1 and split_tier != 0 and split_last != 0:
                    while tr_r_top[i] > 0:
                        j = np.int64(tr_r[i, tr_r_top[i] - 1])
                        if cnt[i, j] > 0:
                            u = n + j; break
                        tr_r_top[i] -= 1
            else:
                j = v - n
                while t1_c_top[j] > 0:
                    i2 = np.int64(t1_c[j, t1_c_top[j] - 1])
                    if cnt[i2, j] > 0 and dir_arr[i2, j] == 2:
                        u = i2; break
                    t1_c_top[j] -= 1
                if u == -1 and split_tier != 0 and split_before_free != 0:
                    while tr_c_top[j] > 0:
                        i2 = np.int64(tr_c[j, tr_c_top[j] - 1])
                        if cnt[i2, j] > 0:
                            u = i2; break
                        tr_c_top[j] -= 1
                if u == -1:
                    while t2_c_top[j] > 0:
                        i2 = np.int64(t2_c[j, t2_c_top[j] - 1])
                        if cnt[i2, j] > 0 and dir_arr[i2, j] == 0:
                            u = i2; break
                        t2_c_top[j] -= 1
                if u == -1 and split_tier != 0 and split_before_free == 0 and split_last == 0:
                    while tr_c_top[j] > 0:
                        i2 = np.int64(tr_c[j, tr_c_top[j] - 1])
                        if cnt[i2, j] > 0:
                            u = i2; break
                        tr_c_top[j] -= 1
                if u == -1:
                    while t3_c_top[j] > 0:
                        i2 = np.int64(t3_c[j, t3_c_top[j] - 1])
                        if cnt[i2, j] > 0:
                            u = i2; break
                        t3_c_top[j] -= 1
                if u == -1 and split_tier != 0 and split_last != 0:
                    while tr_c_top[j] > 0:
                        i2 = np.int64(tr_c[j, tr_c_top[j] - 1])
                        if cnt[i2, j] > 0:
                            u = i2; break
                        tr_c_top[j] -= 1

            if u == -1:
                deg[v] = 0
                top -= 1
                continue

            if v < n:
                i = v; j = u - n
                red[i, j] += 1
                if dir_arr[i, j] == 0:
                    dir_arr[i, j] = 1
                    t1_r[i, t1_r_top[i]] = np.int32(j); t1_r_top[i] += 1
                    t3_c[j, t3_c_top[j]] = np.int32(i); t3_c_top[j] += 1
                elif split_tier != 0 and dir_arr[i, j] == 2 and is_split[i, j] == 0:
                    # a blue-locked pair traversed red: its lock side is column j
                    is_split[i, j] = 1
                    if split_leaves_t1 != 0:
                        dir_arr[i, j] = 3
                    tr_c[j, tr_c_top[j]] = np.int32(i); tr_c_top[j] += 1
                    if split_tier_opposite != 0:
                        tr_r[i, tr_r_top[i]] = np.int32(j); tr_r_top[i] += 1
            else:
                i = u; j = v - n
                blue[i, j] += 1
                if dir_arr[i, j] == 0:
                    dir_arr[i, j] = 2
                    t1_c[j, t1_c_top[j]] = np.int32(i); t1_c_top[j] += 1
                    t3_r[i, t3_r_top[i]] = np.int32(j); t3_r_top[i] += 1
                elif split_tier != 0 and dir_arr[i, j] == 1 and is_split[i, j] == 0:
                    # a red-locked pair traversed blue: its lock side is row i
                    is_split[i, j] = 1
                    if split_leaves_t1 != 0:
                        dir_arr[i, j] = 3
                    tr_r[i, tr_r_top[i]] = np.int32(j); tr_r_top[i] += 1
                    if split_tier_opposite != 0:
                        tr_c[j, tr_c_top[j]] = np.int32(i); tr_c_top[j] += 1

            cnt[i, j] -= 1
            deg[v] -= 1
            deg[u] -= 1
            walk[top] = u
            top += 1

    return red, blue


def split_once(L, variant):
    """euler_split_once with the variant kernel: odd S takes one matching out first."""
    M = np.round(L).astype(np.int64)
    extra = None
    if int(M.sum(axis=1).max()) % 2 == 1:
        extra, M = _extract_one_perm(M)
    red, blue = _split_variant_jit(M, M.shape[0], *VARIANTS[variant])
    if extra is not None:
        red = red + extra
    return red, blue


def leaf_tree(D, split):
    """Leaves at depths 1..4 and the pairs split at each level, for split(L) -> (red, blue)."""
    levels, leaves = [], [np.round(D).astype(np.int64)]
    for _ in DEPTHS:
        nxt, split_pairs = [], 0
        for L in leaves:
            if int(L.sum(axis=1).max()) <= 1:
                nxt.append(L)
                continue
            red, blue = split(L)
            split_pairs += int(np.count_nonzero(red)) + int(np.count_nonzero(blue)) - int(np.count_nonzero(L))
            nxt += [red, blue]
        leaves = nxt
        levels.append((leaves, split_pairs))
    return levels


def count_N(leaf, engine):
    """N and C of one leaf, through the calls _leaf_bvn makes (strategy min, tol 1e-9)."""
    f = leaf.astype(np.float64)
    if engine == "maximum":
        comps = bvn_decomposition(f.copy(), matching_algorithm="maximum", step_strategy="min")
        return len(comps), float(sum(c.weight for c in comps))
    if engine == "heavy":
        w, _ = _jit_decompose_sorted_dynamic(f.copy(), 0, 1e-9)
    elif engine == "heavy_static":
        w, _ = _jit_decompose_sorted_static(f.copy(), 0, 1e-9)
    else:
        w, _ = _jit_decompose_wfa(f.copy(), f.shape[0], 1e-9, 0)
    return len(w), float(sum(w))


def matrix(i):
    return generate_matrix(n=N_DIM, k=K, max_weight=W, rng=np.random.default_rng(SEED + i))


def work(args):
    """One matrix, every variant: split statistics, whether the leaves equal those of
    3tier, and, if with_N, N and C per engine and depth for the variants that need them."""
    i, with_N = args
    D = matrix(i)
    trees = {v: leaf_tree(D, lambda L, v=v: split_once(L, v)) for v in VARIANTS}
    srows, nrows = [], []
    def same(x, y):
        return len(x) == len(y) and all(np.array_equal(a, b) for a, b in zip(x, y))

    for v, levels in trees.items():
        need_N = with_N and (v in N_VARIANTS or (v in PRODUCTION and i < SKIP + N_CHECK))
        for d, (leaves, split_pairs), (l3, _), (l4, _) in zip(DEPTHS, levels, trees["3tier"], trees["4tier"]):
            nnz = [int(np.count_nonzero(L)) for L in leaves]
            srows.append(dict(matrix=i, variant=v, depth=d, leaves=len(leaves), nnz_total=sum(nnz),
                              nnz_max=max(nnz), split_at_level=split_pairs,
                              nnz_D=int(np.count_nonzero(D)), same_as_3tier=same(leaves, l3),
                              same_as_4tier=same(leaves, l4)))
            if need_N:
                for eng in ENGINES:
                    NC = [count_N(L, eng) for L in leaves if np.any(L > 0)]
                    nrows.append(dict(matrix=i, variant=v, depth=d, engine=eng,
                                      N=sum(x[0] for x in NC), C=sum(x[1] for x in NC),
                                      S=float(D.sum(axis=1)[0])))
    return srows, nrows


def check_against_production(n_matrices):
    """3tier and 4tier must equal the production splits, leaf by leaf (main process)."""
    for i in range(n_matrices):
        D = matrix(i)
        for variant, method in PRODUCTION.items():
            mine = leaf_tree(D, lambda L: split_once(L, variant))
            prod = leaf_tree(D, lambda L: euler_split_once(L, split_method=method))
            for (lm, sm), (lp, sp) in zip(mine, prod):
                assert sm == sp and all(np.array_equal(a, b) for a, b in zip(lm, lp)), (i, variant)
    print(f"variant kernel reproduces heuristic and heuristic_ruined on matrices 0..{n_matrices - 1}, "
          f"every leaf of depths 1-4", flush=True)


def main():
    n_mat = int(sys.argv[1]) if len(sys.argv) > 1 else 105
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    with_N = (sys.argv[3] != "0") if len(sys.argv) > 3 else True
    t0 = time.perf_counter()
    check_against_production(n_mat)
    jobs = [(i, with_N) for i in range(SKIP, n_mat)]
    srows, nrows = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, (s, r) in enumerate(pool.map(work, jobs, chunksize=1), 1):
            srows += s; nrows += r
            if k % 10 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    spl = pd.DataFrame(srows)
    spl.to_csv(OUT / "ablation_splits.csv", index=False)
    if with_N:
        pd.DataFrame(nrows).to_csv(OUT / "ablation_N.csv", index=False)
    for ref in ("3tier", "4tier"):
        same = spl.groupby("variant")[f"same_as_{ref}"].agg(["sum", "count"])
        print(f"leaves equal to {ref} (matrix x depth):",
              {v: f"{int(r['sum'])}/{int(r['count'])}" for v, r in same.iterrows()}, flush=True)
    print("ABLATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
