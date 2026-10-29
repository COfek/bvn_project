"""
Diagnostic: compare step sizes and iteration counts for BVN vs B=16 digit planes,
using both maximum (Hungarian) and WFA matching.

Run from project root:
    .venv\Scripts\python.exe agent_scripts\diagnose_step_sizes.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from scipy.optimize import linear_sum_assignment
from src.utils.matrix_generator import generate_matrix
from src.algorithms.wfa import _jit_wfa_kernel, _jit_decompose_wfa

N = 256
K = 10
MAX_WEIGHT = 15.0
SAMPLES = 3
rng = np.random.default_rng(42)


def hungarian_step_stats(matrix):
    """Run BVN with Hungarian matching, return (n_calls, avg_step, total_sum)."""
    work = matrix.astype(np.float64).copy()
    steps = []
    tol = 1e-9
    while np.any(work > tol):
        active_rows = np.where(work.max(axis=1) > tol)[0]
        if len(active_rows) == 0:
            break
        sub = work[active_rows, :]
        row_ind, col_ind = linear_sum_assignment(sub, maximize=True)
        matches = [(active_rows[r], c) for r, c in zip(row_ind, col_ind) if sub[r, c] > tol]
        if not matches:
            break
        rows, cols = zip(*matches)
        vals = work[np.array(rows), np.array(cols)]
        lam = float(np.min(vals))
        if lam < 1e-12:
            break
        steps.append(lam)
        work[np.array(rows), np.array(cols)] -= lam
        work[work < tol] = 0.0
    total_sum = float(matrix.sum())
    avg_step = float(np.mean(steps)) if steps else 0.0
    return len(steps), avg_step, total_sum


def wfa_step_stats(matrix):
    """Run BVN with WFA matching, return (n_calls, avg_step, total_sum)."""
    work = matrix.astype(np.float64).copy()
    total_sum = float(matrix.sum())
    weights, _ = _jit_decompose_wfa(work, N, 1e-9)
    steps = list(weights)
    avg_step = float(np.mean(steps)) if steps else 0.0
    return len(steps), avg_step, total_sum


def run_and_print(label, stats_fn, matrix):
    calls, avg_step, total_sum = stats_fn(matrix)
    predicted = total_sum / (N * avg_step) if avg_step > 0 else 0
    print(f"  {label:35s} calls={calls:>5}  avg_step={avg_step:>8.3f}  sum={total_sum:>10.0f}  predicted={predicted:>6.1f}")


# Warm up Numba JIT
_dummy = np.ones((4, 4), dtype=np.float64)
_jit_decompose_wfa(_dummy, 4, 1e-9)

print(f"N={N}, K={K}, MAX_WEIGHT={MAX_WEIGHT}")
print("=" * 85)

for i in range(SAMPLES):
    matrix = generate_matrix(n=N, k=K, max_weight=MAX_WEIGHT, rng=rng)
    print(f"\nSample {i}  (S={matrix.sum(axis=1)[0]:.1f})")
    print(f"  {'Algorithm + Context':<35} {'calls':>7}  {'avg_step':>10}  {'sum':>12}  {'predicted':>10}")
    print("  " + "-" * 75)

    run_and_print("Hungarian / BVN full matrix", hungarian_step_stats, matrix)
    run_and_print("WFA       / BVN full matrix", wfa_step_stats, matrix)

    # B=16 digit planes
    temp = matrix.copy().astype(np.int64)
    plane_idx = 0
    while np.any(temp > 0):
        digit_plane = (temp % 16).astype(np.float64)
        unit = 16 ** plane_idx
        if np.any(digit_plane > 0):
            run_and_print(f"Hungarian / plane {plane_idx} (x{unit})", hungarian_step_stats, digit_plane)
            run_and_print(f"WFA       / plane {plane_idx} (x{unit})", wfa_step_stats, digit_plane)
        temp //= 16
        plane_idx += 1
        if np.all(temp == 0):
            break
