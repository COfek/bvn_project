"""Time-To-First-Permutation (TTFP) experiment.

Metric: how long until the switch has its FIRST usable configuration?
  TTFP = (preprocessing required before any extraction can start)
       + (time to extract ONE matching on the relevant unit)

Per framework:
  BvN      : one matching extraction on the full matrix (no preprocessing).
  Radix B  : digit-plane extraction (timed) + min over planes of one
             extraction (planes run in parallel -> first available = fastest).
  Euler d  : split wall-clock (levels sequential, splits within a level
             parallel -> wall = sum over levels of max split in level)
             + min over the 2^d leaves of one extraction.

Engines (first iteration only):
  Hungarian : scipy linear_sum_assignment (maximise weight)
  WFA       : wavefront_matching_vectorized (maximal matching)
  GW        : sorted_array_matching (sort + greedy + augment).
              NOTE: GW Dynamic and GW Static are IDENTICAL on the first
              iteration (both sort fresh at t=0), so one GW row covers both.

Dense benchmark: n=256, k=256, Wmax=64, seed 42+i, N_MATRICES matrices.
Output: run/ttfp/ttfp_results.csv + printed summary. No paper changes.
"""
import os, time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, "C:/Users/ofekc/Desktop/Msc/Thesis/bvn_project")
from src.utils.matrix_generator import generate_matrix
from src.algorithms.wfa import wavefront_matching_vectorized
from src.algorithms.sorted_array_matching import sorted_array_matching
from src.algorithms.euler_splitting import euler_split_once

N_MATRICES = 30
BASES = [2, 4, 8, 16]
DEPTHS = [1, 2, 3, 4]

def t_hungarian(M):
    t0 = time.perf_counter()
    linear_sum_assignment(M, maximize=True)
    return time.perf_counter() - t0

def t_wfa(M):
    t0 = time.perf_counter()
    wavefront_matching_vectorized(M)
    return time.perf_counter() - t0

def t_gw(M):
    t0 = time.perf_counter()
    sorted_array_matching(M)
    return time.perf_counter() - t0

MATCHERS = {"Hungarian": t_hungarian, "WFA": t_wfa, "GW (dyn=stat)": t_gw}

def digit_planes(D, B):
    t0 = time.perf_counter()
    planes, temp = [], D.copy()
    while temp.max() > 0:
        planes.append((temp % B).astype(np.float64))
        temp //= B
    return planes, time.perf_counter() - t0

def euler_leaves(D, depth):
    """Split to 2^depth leaves; wall = sum over levels of max split within level."""
    level = [D.astype(np.int64)]
    wall = 0.0
    for _ in range(depth):
        nxt, times = [], []
        for M in level:
            t0 = time.perf_counter()
            r, b = euler_split_once(M, "heuristic")
            times.append(time.perf_counter() - t0)
            nxt += [r, b]
        wall += max(times)
        level = nxt
    return [m.astype(np.float64) for m in level], wall

# JIT / solver warm-up (not timed)
_w = generate_matrix(n=256, k=256, max_weight=64, rng=np.random.default_rng(0)).astype(np.float64)
for f in MATCHERS.values():
    f(_w)
euler_split_once(_w.astype(np.int64), "heuristic")

rows = []
for i in range(N_MATRICES):
    D = generate_matrix(n=256, k=256, max_weight=64, rng=np.random.default_rng(42 + i)).astype(np.float64)
    for eng, match in MATCHERS.items():
        rows.append(dict(matrix=i, engine=eng, framework="BvN", ttfp_ms=match(D) * 1000))
        for B in BASES:
            planes, t_ext = digit_planes(D.astype(np.int64), B)
            first = min(match(p) for p in planes)
            rows.append(dict(matrix=i, engine=eng, framework=f"Radix B{B}",
                             ttfp_ms=(t_ext + first) * 1000))
        for d in DEPTHS:
            leaves, t_split = euler_leaves(D, d)
            first = min(match(l) for l in leaves)
            rows.append(dict(matrix=i, engine=eng, framework=f"Euler d{d}",
                             ttfp_ms=(t_split + first) * 1000))
    if (i + 1) % 5 == 0:
        print(f"progress {i+1}/{N_MATRICES}", flush=True)

df = pd.DataFrame(rows)
os.makedirs("run/ttfp", exist_ok=True)
df.to_csv("run/ttfp/ttfp_results.csv", index=False)

print("\n=== TTFP (ms), mean over", N_MATRICES, "matrices ===")
pivot = df.pivot_table(index="framework", columns="engine", values="ttfp_ms", aggfunc="mean")
order = ["BvN"] + [f"Radix B{b}" for b in BASES] + [f"Euler d{d}" for d in DEPTHS]
print(pivot.reindex(order).round(2).to_string())
print("\nsaved run/ttfp/ttfp_results.csv")
