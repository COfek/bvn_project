"""
Diagnostic: count actual Hungarian calls for BVN vs B=16 radix.
Run from project root:
    .venv\Scripts\python.exe agent_scripts\diagnose_b16_vs_bvn.py
"""
import sys, time
sys.path.insert(0, ".")

import numpy as np
from scipy.optimize import linear_sum_assignment
from src.utils.matrix_generator import generate_matrix
from src.algorithms.bvn import bvn_decomposition

# --- Patch linear_sum_assignment to count calls ---
import scipy.optimize as _sopt
_orig_lsa = _sopt.linear_sum_assignment
_call_counts = {"bvn": 0, "radix": 0}
_active_ctx = ["bvn"]

def _counting_lsa(cost, maximize=False):
    _call_counts[_active_ctx[0]] += 1
    return _orig_lsa(cost, maximize=maximize)

_sopt.linear_sum_assignment = _counting_lsa
# Also patch in the modules that imported it already
import src.algorithms.bvn as _bvn_mod
import src.algorithms.radix_decomposition as _radix_mod
_bvn_mod.linear_sum_assignment = _counting_lsa
_radix_mod.linear_sum_assignment = _counting_lsa

from src.algorithms.radix_decomposition import decompose_radix

N = 256
K = 256
MAX_WEIGHT = 100
SAMPLES = 10
rng = np.random.default_rng(42)

bvn_iters, b16_max_plane_iters, b16_total_iters = [], [], []

for i in range(SAMPLES):
    matrix = generate_matrix(n=N, k=K, max_weight=MAX_WEIGHT, rng=rng)

    # BVN
    _call_counts["bvn"] = 0
    _active_ctx[0] = "bvn"
    t0 = time.perf_counter()
    comps_bvn = bvn_decomposition(matrix=matrix, matching_algorithm="maximum")
    t_bvn = time.perf_counter() - t0
    bvn_calls = _call_counts["bvn"]

    # B=16 radix (we need per-plane counts, so patch _decompose_digit_plane)
    plane_call_counts = []
    orig_ddp = _radix_mod._decompose_digit_plane

    def _counting_ddp(plane, unit_weight, strategy="min", matching_method="heavy"):
        before = _call_counts["radix"]
        result = orig_ddp(plane, unit_weight, strategy, matching_method)
        plane_call_counts.append(_call_counts["radix"] - before)
        return result

    _radix_mod._decompose_digit_plane = _counting_ddp
    _call_counts["radix"] = 0
    _active_ctx[0] = "radix"

    t0 = time.perf_counter()
    comps_b16, _, num_planes = decompose_radix(matrix=matrix, base=16,
                                               matching_method="maximum")
    t_b16 = time.perf_counter() - t0

    _radix_mod._decompose_digit_plane = orig_ddp

    total_radix = _call_counts["radix"]
    max_plane = max(plane_call_counts) if plane_call_counts else 0

    bvn_iters.append(bvn_calls)
    b16_max_plane_iters.append(max_plane)
    b16_total_iters.append(total_radix)

    print(f"Sample {i:2d} | "
          f"BVN: {bvn_calls:4d} calls, {t_bvn:.3f}s | "
          f"B16 planes={num_planes}, max_plane={max_plane:3d} calls, "
          f"total={total_radix:4d} calls, {t_b16:.3f}s")

print()
print(f"Averages over {SAMPLES} samples:")
print(f"  BVN Hungarian calls:        {np.mean(bvn_iters):.1f}")
print(f"  B=16 max-plane calls:       {np.mean(b16_max_plane_iters):.1f}")
print(f"  B=16 total calls (all planes): {np.mean(b16_total_iters):.1f}")
print()
print("If BVN calls >> B=16 max-plane calls: B=16 SHOULD be faster → overhead is the bug.")
print("If BVN calls ~= B=16 max-plane calls: iteration counts are similar → overhead explains the gap.")
print("If B=16 max-plane calls >> B-1=15:    partial matching is inflating B=16 iterations.")
