"""
Task 3 — Where is the splitting bottleneck, and what is cheaper?

Part 1: scaling probe of the heuristic split.
  Theory: traversal walks sum(M) ~ n*S unit edges, each step scanning O(n)
  neighbours in _choose_next_jit => total O(n^2 * S).
  Probe: vary n at fixed S, and S at fixed n; fit the exponent.

Part 2: cheap-alternative check at paper scale.
  The existing "greedy" split is O(nnz log nnz) and also targets sparsity
  (whole-entry assignment, Euler fallback). Compare against "heuristic":
    - split wall time
    - leaf sparsity (nnz of each half)
    - whether greedy actually fell back to Euler
    - downstream max-leaf BvN extraction time (heavy_static)
"""
from __future__ import annotations

import time

import numpy as np

from src.algorithms.euler_splitting import (
    euler_split_once,
    _euler_split,
    _euler_split_heuristic,
    _greedy_split,
    _leaf_bvn,
)
from src.utils.matrix_generator import generate_matrix


def _gen(n, k, mw, seed):
    rng = np.random.default_rng(seed)
    return generate_matrix(n=n, k=k, max_weight=mw, rng=rng).astype(np.int64)


def _t(fn, *args, reps=3):
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, out


def part1_scaling():
    print("=" * 70)
    print("PART 1: heuristic split scaling  (times are best-of-3, ms)")
    print("=" * 70)

    # Warm up JIT
    _euler_split_heuristic(_gen(32, 8, 4, 0))

    print("\n-- vary n, hold S ~ const (k=32, max_weight=8, S~128) --")
    print(f"{'n':>5} {'S':>7} {'sum(M)':>9} {'nnz':>7} {'time ms':>9} {'time/(n*S) us':>14} {'time/(n^2*S) ns':>16}")
    for n in [32, 64, 128, 256, 512]:
        M = _gen(n, 32, 8, 1)
        S = int(M.sum(axis=1).max())
        tm, _ = _t(_euler_split_heuristic, M)
        e = n * S
        print(f"{n:>5} {S:>7} {int(M.sum()):>9} {np.count_nonzero(M):>7} "
              f"{tm*1e3:>9.2f} {tm/e*1e6:>14.4f} {tm/(n*e)*1e9:>16.3f}")

    print("\n-- vary S via k, hold n=256 (max_weight=8) --")
    print(f"{'k':>5} {'S':>7} {'sum(M)':>9} {'nnz':>7} {'time ms':>9} {'time/(n*S) us':>14}")
    for k in [16, 64, 256, 1024]:
        M = _gen(256, k, 8, 2)
        S = int(M.sum(axis=1).max())
        tm, _ = _t(_euler_split_heuristic, M)
        print(f"{k:>5} {S:>7} {int(M.sum()):>9} {np.count_nonzero(M):>7} "
              f"{tm*1e3:>9.2f} {tm/(256*S)*1e6:>14.4f}")

    print("\n-- vary S via max_weight, hold n=256, k=256 (denser entries) --")
    print(f"{'mw':>5} {'S':>7} {'sum(M)':>9} {'nnz':>7} {'time ms':>9} {'time/(n*S) us':>14}")
    for mw in [4, 16, 64]:
        M = _gen(256, 256, mw, 3)
        S = int(M.sum(axis=1).max())
        tm, _ = _t(_euler_split_heuristic, M)
        print(f"{mw:>5} {S:>7} {int(M.sum()):>9} {np.count_nonzero(M):>7} "
              f"{tm*1e3:>9.2f} {tm/(256*S)*1e6:>14.4f}")


def part2_greedy_vs_heuristic():
    print()
    print("=" * 70)
    print("PART 2: greedy vs heuristic at paper scale (n=256, k=256, mw=64)")
    print("=" * 70)
    print(f"\n{'seed':>5} {'method':<10} {'split ms':>9} {'red nnz':>8} {'blue nnz':>9} "
          f"{'fallback':>9} {'max-leaf BvN ms':>16}")

    for seed in [10, 11, 12]:
        M = _gen(256, 256, 64, seed)
        S = int(M.sum(axis=1).max())
        if S % 2 == 1:
            M[0, :] = M[0, :]  # keep as-is; euler_split_once handles odd S

        for method in ("heuristic", "greedy"):
            t0 = time.perf_counter()
            red, blue = euler_split_once(M, split_method=method)
            split_t = time.perf_counter() - t0

            # Detect greedy fallback: re-run _greedy_split's validity check
            fellback = ""
            if method == "greedy":
                half = (S // 2)
                # greedy returns euler fallback when its own red fails validity.
                # Heuristic test: greedy's intended output has many "whole"
                # entries; the euler fallback splits nearly everything.
                whole = int(np.sum(((red == M) | (red == 0)) & (M > 0)))
                frac = whole / max(1, np.count_nonzero(M))
                fellback = f"{frac:.2f} whole"

            # Downstream extraction: max leaf BvN time, heavy_static
            _, rt_red = _leaf_bvn(red.astype(np.float64), "heavy_static")
            _, rt_blue = _leaf_bvn(blue.astype(np.float64), "heavy_static")
            max_leaf = max(rt_red, rt_blue)

            print(f"{seed:>5} {method:<10} {split_t*1e3:>9.1f} "
                  f"{np.count_nonzero(red):>8} {np.count_nonzero(blue):>9} "
                  f"{fellback:>9} {max_leaf*1e3:>16.1f}")

        # Reference: plain euler split sparsity
        red_e, blue_e = euler_split_once(M, split_method="euler_grouped")
        print(f"{seed:>5} {'grouped':<10} {'-':>9} {np.count_nonzero(red_e):>8} "
              f"{np.count_nonzero(blue_e):>9} {'':>9} {'(reference)':>16}")


if __name__ == "__main__":
    part1_scaling()
    part2_greedy_vs_heuristic()
