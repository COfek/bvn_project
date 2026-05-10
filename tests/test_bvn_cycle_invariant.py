"""
Invariant: for a scaled doubly stochastic matrix M with common row/column
sum S, every valid BVN decomposition must satisfy

    sum(phi_i  for every component_i) == S

within numerical tolerance. This file pytest-checks that property across
all four matching algorithms, several matrix shapes, and a handful of seeds
so silent correctness drift in the peeling loop or in any matching engine
is caught early. Pure test-suite code; not invoked by the main runner.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.bvn import bvn_decomposition
from src.utils.matrix_generator import generate_matrix

MATCHING_ALGORITHMS = ["wfa", "heavy", "heavy_static", "maximum"]
TOL = 1e-6


def _scale_factor(matrix: np.ndarray) -> float:
    """Returns the common row/col sum S; fails the test if the matrix
    isn't scaled doubly stochastic in the first place."""
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    S = float(row_sums[0])
    assert np.allclose(row_sums, S, atol=TOL), (
        f"row sums are not all equal: "
        f"min={row_sums.min():.6g}, max={row_sums.max():.6g}"
    )
    assert np.allclose(col_sums, S, atol=TOL), (
        f"column sums differ from row sums: "
        f"S={S:.6g}, col_min={col_sums.min():.6g}, col_max={col_sums.max():.6g}"
    )
    return S


@pytest.mark.parametrize("matching", MATCHING_ALGORITHMS)
@pytest.mark.parametrize("seed", [42, 7, 123, 2024])
@pytest.mark.parametrize(
    "n,k,max_weight",
    [
        (16,   5, 1),
        (32,  10, 4),
        (64,  32, 8),
        (16, 100, 1),  # high-k, unit-weight regime
    ],
)
def test_bvn_cycle_equals_scale_factor_integer(matching, seed, n, k, max_weight):
    """Integer-weighted matrices: BVN cycle length == S."""
    rng = np.random.default_rng(seed)
    M = generate_matrix(n=n, k=k, max_weight=max_weight, rng=rng)

    S = _scale_factor(M)

    components = bvn_decomposition(matrix=M, matching_algorithm=matching)
    cycle = float(sum(c.weight for c in components))

    assert np.isclose(cycle, S, atol=TOL), (
        f"BVN({matching}) cycle length {cycle:.6f} != scale factor S={S:.6f} "
        f"(diff={abs(cycle - S):.2e}, "
        f"n={n}, k={k}, max_weight={max_weight}, seed={seed})"
    )


@pytest.mark.parametrize("matching", MATCHING_ALGORITHMS)
@pytest.mark.parametrize("seed", [42, 7, 2024])
def test_bvn_cycle_equals_scale_factor_float(matching, seed):
    """Float-weighted matrices (Sinkhorn-style): BVN cycle length == S
    within float64 tolerance. Slightly looser tol because the peeling
    loop terminates on values < 1e-9, which can accumulate over many
    iterations."""
    rng = np.random.default_rng(seed)
    M = generate_matrix(n=32, k=20, max_weight=1.0, rng=rng, float_weights=True)

    S = _scale_factor(M)

    components = bvn_decomposition(matrix=M, matching_algorithm=matching)
    cycle = float(sum(c.weight for c in components))

    # Float-mode tol: residual clip in BVN is 1e-9 per cell and there are
    # N*N cells, so worst-case drift is ~N*N*1e-9.
    float_tol = max(TOL, 1e-7 * M.size)

    assert np.isclose(cycle, S, atol=float_tol), (
        f"BVN({matching}) float cycle length {cycle:.6f} != S={S:.6f} "
        f"(diff={abs(cycle - S):.2e}, seed={seed}, tol={float_tol:.2e})"
    )
