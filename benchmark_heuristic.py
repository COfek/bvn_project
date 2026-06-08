"""
Benchmark: Heuristic Euler split vs plain BvN vs Radix B=2

For each matrix:
  1. Plain BvN (GW Dynamic)               → T_bvn, N_bvn, C_bvn
  2. Radix B=2 (GW Dynamic, best Radix)   → T_radix, N_radix, C_radix
  3. Heuristic framework (split + BvN):
       a. Euler split (baseline)          → T_split_euler
       b. Heuristic split                 → T_split_heuristic
       c. BvN on Red leaf                 → T_red, N_red, C_red
       d. BvN on Blue leaf               → T_blue, N_blue, C_blue
       e. Simulated parallel total        → T_split + max(T_red, T_blue)
          (two processors run Red and Blue concurrently)

The "viable?" question comes down to:
  T_heuristic_total  vs  T_bvn  vs  T_radix

Also reports N_total = N_red + N_blue and C_total = C_red + C_blue
to show whether quality (cycle length) is preserved.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.algorithms.sorted_array_matching import _jit_decompose_sorted_dynamic
from src.algorithms.radix_decomposition import decompose_radix
from src.algorithms.euler_splitting import euler_split_once
from src.utils.matrix_generator import generate_matrix

# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Warming up JIT... ", end="", flush=True)
_d = np.ones((4, 4), dtype=np.float64)
_jit_decompose_sorted_dynamic(_d.copy(), 0)
euler_split_once(generate_matrix(4, 2, 2, np.random.default_rng(0)).astype(np.int64),
                 "heuristic")
print("done.\n")

# ── helpers ───────────────────────────────────────────────────────────────────

def bvn_time(M: np.ndarray):
    """Run GW Dynamic BvN on M. Returns (time_ms, N, C)."""
    work = M.astype(np.float64).copy()
    t0 = time.perf_counter()
    weights, _ = _jit_decompose_sorted_dynamic(work, strategy_int=0, tol=1e-9)
    ms = (time.perf_counter() - t0) * 1000
    C  = sum(weights)
    N  = len(weights)
    return ms, N, C

def radix_time(M: np.ndarray, base: int = 2):
    """Run Radix B=base (GW Dynamic) on M. Returns (time_ms, N, C)."""
    t0 = time.perf_counter()
    comps, _, _ = decompose_radix(M, base=base, matching_method="heavy",
                                  step_strategy="min")
    ms = (time.perf_counter() - t0) * 1000
    C  = sum(c.weight for c in comps)
    N  = len(comps)
    return ms, N, C

def split_and_bvn(M: np.ndarray, method: str):
    """Split M, run BvN on each half in simulated parallel. Returns full stats."""
    t0 = time.perf_counter()
    red, blue = euler_split_once(M, split_method=method)
    t_split = (time.perf_counter() - t0) * 1000

    t_red,  N_red,  C_red  = bvn_time(red.astype(np.float64))
    t_blue, N_blue, C_blue = bvn_time(blue.astype(np.float64))

    t_parallel = t_split + max(t_red, t_blue)   # split sequential, BvN parallel
    N_total = N_red + N_blue
    C_total = C_red + C_blue

    return {
        "t_split": t_split,
        "t_red": t_red, "t_blue": t_blue,
        "t_parallel": t_parallel,
        "N": N_total, "C": C_total,
    }

# ── benchmark ─────────────────────────────────────────────────────────────────
N_MATRICES = 5
SEEDS      = range(N_MATRICES)

print(f"{'='*84}")
print(f"Benchmark: n=256, k=256, max_w=64  ({N_MATRICES} matrices)")
print(f"{'='*84}")
hdr = (f"{'seed':>4}  {'S':>5}  "
       f"{'BvN':>8}  {'Radix-B2':>10}  "
       f"{'Euler-split':>12}  {'H-split':>9}  "
       f"{'H-BvN-Red':>10}  {'H-BvN-Blu':>10}  "
       f"{'H-total':>9}  "
       f"{'N_bvn':>6}  {'N_radix':>7}  {'N_heur':>7}  "
       f"{'C_bvn':>7}  {'C_radix':>7}  {'C_heur':>7}")
print(hdr)
print("-" * 140)

rows = []
for seed in SEEDS:
    rng = np.random.default_rng(42 + seed)
    M   = generate_matrix(n=256, k=256, max_weight=64, rng=rng)
    S   = float(M.sum(axis=1).max())
    Mi  = np.round(M).astype(np.int64)

    t_bvn,   N_bvn,   C_bvn   = bvn_time(M)
    t_radix, N_radix, C_radix = radix_time(Mi, base=2)
    e = split_and_bvn(Mi, "euler")
    h = split_and_bvn(Mi, "heuristic")

    rows.append((seed, S, t_bvn, t_radix, e, h,
                 N_bvn, N_radix, C_bvn, C_radix))

    print(f"{seed:>4}  {S:>5.0f}  "
          f"{t_bvn:>8.1f}  {t_radix:>10.1f}  "
          f"{e['t_split']:>12.1f}  {h['t_split']:>9.1f}  "
          f"{h['t_red']:>10.1f}  {h['t_blue']:>10.1f}  "
          f"{h['t_parallel']:>9.1f}  "
          f"{N_bvn:>6}  {N_radix:>7}  {h['N']:>7}  "
          f"{C_bvn:>7.0f}  {C_radix:>7.0f}  {h['C']:>7.0f}")

# ── summary ───────────────────────────────────────────────────────────────────
print("=" * 140)
avg = lambda col: np.mean([col(r) for r in rows])

print(f"\nAverages over {N_MATRICES} matrices:")
print(f"  Plain BvN          : {avg(lambda r: r[2]):>8.1f} ms   N={avg(lambda r: r[6]):.0f}   C={avg(lambda r: r[8]):.0f}")
print(f"  Radix B=2          : {avg(lambda r: r[3]):>8.1f} ms   N={avg(lambda r: r[7]):.0f}   C={avg(lambda r: r[9]):.0f}")
print(f"  Euler split total  : {avg(lambda r: r[4]['t_parallel']):>8.1f} ms   N={avg(lambda r: r[4]['N']):.0f}   C={avg(lambda r: r[4]['C']):.0f}")
print(f"  Heuristic split t  : {avg(lambda r: r[5]['t_split']):>8.1f} ms   (split step only)")
print(f"  Heuristic total    : {avg(lambda r: r[5]['t_parallel']):>8.1f} ms   N={avg(lambda r: r[5]['N']):.0f}   C={avg(lambda r: r[5]['C']):.0f}")
print()
print(f"  Heuristic split / BvN baseline  = {avg(lambda r: r[5]['t_split']) / avg(lambda r: r[2]):.1f}x")
print(f"  Heuristic total  / BvN baseline = {avg(lambda r: r[5]['t_parallel']) / avg(lambda r: r[2]):.1f}x")
print(f"  Heuristic total  / Radix B=2    = {avg(lambda r: r[5]['t_parallel']) / avg(lambda r: r[3]):.1f}x")
