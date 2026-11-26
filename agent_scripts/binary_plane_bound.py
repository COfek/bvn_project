"""Radix at B = 2 reaches the fewest configurations a binary plane allows (Section 6.2).

In a binary digit plane every entry is 0 or 1, so a configuration is held for one unit and
carries at most one unit through each port: a plane needs at least as many configurations as the
load of its busiest port, S*(plane). This script rebuilds the 995 matrices of the Radix campaign
(n = 256, k = 256, W_max = 64, matrix i seeded 42 + i, the first 5 discarded) and compares, per
engine, the campaign's mean N at B = 2 with the mean of sum over planes of S*(plane); at B = 4
the same sum is printed too, where it is not a bound (a configuration can carry up to 3 units).
Run with BVN_CAMPAIGN=2026_11 BVN_RUN_ROOT=run/rerun_2026_11e.
"""
from __future__ import annotations
from _paths import REPO
import sys
import numpy as np
sys.path.insert(0, str(REPO))
from src.utils.matrix_generator import generate_matrix
from src.algorithms.radix_decomposition import extract_digit_planes
import variance_data as v

WARMUP, COUNT = 5, 1000
bound = {2: [], 4: []}
for i in range(WARMUP, COUNT):
    D = generate_matrix(256, 256, 64, np.random.default_rng(42 + i))
    for B in bound:
        bound[B].append(sum(max(P.sum(0).max(), P.sum(1).max()) for _, P in extract_digit_planes(D, B)))
print(f"{len(bound[2])} matrices")
for B in (2, 4):
    print(f"B = {B}: mean sum over planes of S*(plane) = {np.mean(bound[B]):.1f}")
for name, (key, pre) in {"Hungarian": ("max", "maximum"), "GW Dynamic": ("heavy", "heavy"),
                         "GW Static": ("heavy_static", "heavy_static"), "WWFA": ("wwfa", "wwfa")}.items():
    d = v.radix_frame(key)
    print(f"{name:10s} mean N: B=2 {d[f'{pre}_2_perms'].mean():.1f}   B=4 {d[f'{pre}_4_perms'].mean():.1f}"
          f"   (matrices: {len(d)})")
