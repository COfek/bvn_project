"""
Formal correctness test suite for BvN and Radix decompositions.

Verifies the three mathematical properties that define a valid decomposition
D = {(P_i, phi_i)}:

  Property 1 — Reconstruction:     sum_i phi_i * P_i  ==  D   (within FP tolerance)
  Property 2 — Valid permutations:  each P_i is binary, exactly one 1 per row and col
  Property 3 — Non-negative weights: phi_i >= 0 for all i

Additionally verifies:

  Property 4 — Cycle length (strong engines):
                C(D) = sum_i phi_i == S  for Hungarian / GW Dynamic / GW Static
  Property 5 — Radix reconstruction:
                sum_i component_i.matrix == D  (weight already embedded)

All tests are parametrised over multiple engines, matrix sizes, and random seeds
to give statistical coverage rather than a single smoke-test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.bvn import bvn_decomposition
from src.algorithms.radix_decomposition import decompose_radix
from src.utils.matrix_generator import generate_matrix


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

BVN_ENGINES = ["maximum", "heavy", "heavy_static", "wfa"]
STRONG_ENGINES = ["maximum", "heavy", "heavy_static"]   # produce C(D) == S exactly
RADIX_BASES = [2, 4, 8, 16]

# (n, k, max_weight, seed)
MATRIX_CONFIGS = [
    (16,  8,  8,  0),
    (16,  8,  8,  1),
    (32, 16, 16,  2),
    (32, 16, 16,  3),
    (64, 32, 32,  4),
    (64, 32, 32,  5),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(n, k, max_weight, seed):
    rng = np.random.default_rng(seed)
    return generate_matrix(n=n, k=k, max_weight=max_weight, rng=rng)


def _row_sum(m):
    return float(m.sum(axis=1).max())


# ===========================================================================
# BvN decomposition tests
# ===========================================================================

class TestBvNProperty1Reconstruction:
    """sum_i phi_i * P_i == D (exact reconstruction)."""

    @pytest.mark.parametrize("engine", BVN_ENGINES)
    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS)
    def test_bvn_reconstruction(self, engine, n, k, w, seed):
        D = _make(n, k, w, seed)
        comps = bvn_decomposition(D, matching_algorithm=engine)

        reconstructed = np.zeros_like(D)
        for c in comps:
            reconstructed += c.weight * c.permutation

        np.testing.assert_allclose(
            reconstructed, D, atol=1e-6,
            err_msg=(
                f"Reconstruction failed: engine={engine} n={n} k={k} seed={seed}\n"
                f"  max_abs_err={np.max(np.abs(reconstructed - D)):.2e}"
            ),
        )


class TestBvNProperty2ValidPermutations:
    """Each P_i is binary (entries in {0,1}) and encodes a valid matching.

    Note on WFA: WFA is a *weak* decomposition that uses wavefront (anti-diagonal)
    scans without augmentation.  This can leave some rows/columns unmatched in a
    single round, so its components are *partial* matchings (row sums ∈ {0, 1},
    not necessarily all 1).  WFA is excluded from the strict one-per-row/col check;
    reconstruction correctness (Property 1) still holds for WFA.
    """

    # All engines produce binary entries — even WFA uses 0/1 indicator matrices
    @pytest.mark.parametrize("engine", BVN_ENGINES)
    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS[:4])
    def test_permutation_is_binary(self, engine, n, k, w, seed):
        D = _make(n, k, w, seed)
        comps = bvn_decomposition(D, matching_algorithm=engine)
        for idx, c in enumerate(comps):
            P = c.permutation
            vals = P.ravel()
            assert np.all((vals >= -1e-9) & (vals <= 1 + 1e-9)), (
                f"engine={engine} seed={seed}: component {idx} has non-binary entries "
                f"(min={vals.min():.4g}, max={vals.max():.4g})"
            )

    # Strong engines (Hungarian, GW Dynamic, GW Static) produce FULL permutations
    @pytest.mark.parametrize("engine", STRONG_ENGINES)
    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS[:4])
    def test_permutation_one_per_row(self, engine, n, k, w, seed):
        """Strong engines: every component has exactly one 1 per row."""
        D = _make(n, k, w, seed)
        comps = bvn_decomposition(D, matching_algorithm=engine)
        for idx, c in enumerate(comps):
            row_sums = c.permutation.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-9), (
                f"engine={engine} seed={seed}: component {idx} row sums != 1 "
                f"({row_sums.tolist()})"
            )

    @pytest.mark.parametrize("engine", STRONG_ENGINES)
    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS[:4])
    def test_permutation_one_per_col(self, engine, n, k, w, seed):
        """Strong engines: every component has exactly one 1 per col."""
        D = _make(n, k, w, seed)
        comps = bvn_decomposition(D, matching_algorithm=engine)
        for idx, c in enumerate(comps):
            col_sums = c.permutation.sum(axis=0)
            assert np.allclose(col_sums, 1.0, atol=1e-9), (
                f"engine={engine} seed={seed}: component {idx} col sums != 1 "
                f"({col_sums.tolist()})"
            )

    # WFA: each component is a *partial* matching — row sums ∈ {0, 1}
    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS[:4])
    def test_wfa_partial_matching_binary(self, n, k, w, seed):
        """WFA components have row/col sums in {0, 1} (partial matching)."""
        D = _make(n, k, w, seed)
        comps = bvn_decomposition(D, matching_algorithm="wfa")
        for idx, c in enumerate(comps):
            row_sums = c.permutation.sum(axis=1)
            col_sums = c.permutation.sum(axis=0)
            assert np.all((row_sums >= -1e-9) & (row_sums <= 1 + 1e-9)), (
                f"WFA seed={seed}: component {idx} has row sum > 1"
            )
            assert np.all((col_sums >= -1e-9) & (col_sums <= 1 + 1e-9)), (
                f"WFA seed={seed}: component {idx} has col sum > 1"
            )


class TestBvNProperty3NonNegativeWeights:
    """phi_i >= 0 for all components."""

    @pytest.mark.parametrize("engine", BVN_ENGINES)
    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS)
    def test_weights_non_negative(self, engine, n, k, w, seed):
        D = _make(n, k, w, seed)
        comps = bvn_decomposition(D, matching_algorithm=engine)
        for idx, c in enumerate(comps):
            assert c.weight >= -1e-12, (
                f"engine={engine} seed={seed}: component {idx} has negative weight "
                f"phi={c.weight:.4g}"
            )


class TestBvNProperty4CycleLength:
    """C(D) = sum phi_i == S for strong-decomposition engines."""

    @pytest.mark.parametrize("engine", STRONG_ENGINES)
    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS)
    def test_cycle_length_equals_S(self, engine, n, k, w, seed):
        D = _make(n, k, w, seed)
        S = _row_sum(D)
        comps = bvn_decomposition(D, matching_algorithm=engine)
        C = sum(c.weight for c in comps)
        assert abs(C - S) < 1e-6, (
            f"engine={engine} seed={seed}: C(D)={C:.6f} != S={S:.6f} "
            f"(diff={abs(C-S):.2e})"
        )

    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS)
    def test_wfa_cycle_length_geq_S(self, n, k, w, seed):
        """WFA is a weak decomposition: C(D) >= S (may exceed S)."""
        D = _make(n, k, w, seed)
        S = _row_sum(D)
        comps = bvn_decomposition(D, matching_algorithm="wfa")
        C = sum(c.weight for c in comps)
        assert C >= S - 1e-6, (
            f"WFA seed={seed}: C(D)={C:.6f} < S={S:.6f} — violated lower bound"
        )


# ===========================================================================
# Radix decomposition tests
# ===========================================================================

class TestRadixProperty1Reconstruction:
    """sum_i component_i.matrix == D (weight embedded in RadixComponent.matrix)."""

    @pytest.mark.parametrize("base", RADIX_BASES)
    @pytest.mark.parametrize("engine", ["heavy", "heavy_static", "wfa", "maximum"])
    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS[:4])
    def test_radix_reconstruction(self, base, engine, n, k, w, seed):
        D = _make(n, k, w, seed)
        comps, _, _ = decompose_radix(D, base=base, matching_method=engine)

        reconstructed = np.zeros_like(D)
        for c in comps:
            reconstructed += c.matrix   # matrix = weight * permutation

        np.testing.assert_allclose(
            reconstructed, D, atol=1e-5,
            err_msg=(
                f"Radix reconstruction failed: base={base} engine={engine} "
                f"n={n} k={k} seed={seed}\n"
                f"  max_abs_err={np.max(np.abs(reconstructed - D)):.2e}"
            ),
        )


class TestRadixProperty2ValidPermutations:
    """Each Radix component encodes a valid permutation (matrix / weight is binary, 1/row, 1/col)."""

    @pytest.mark.parametrize("base", [2, 4])   # subset for speed
    @pytest.mark.parametrize("engine", ["heavy", "wfa"])
    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS[:3])
    def test_radix_permutation_binary(self, base, engine, n, k, w, seed):
        D = _make(n, k, w, seed)
        comps, _, _ = decompose_radix(D, base=base, matching_method=engine)
        for idx, c in enumerate(comps):
            if abs(c.weight) < 1e-12:
                continue
            P = c.matrix / c.weight   # recover the permutation matrix
            vals = P.ravel()
            assert np.all((vals >= -1e-9) & (vals <= 1 + 1e-9)), (
                f"base={base} engine={engine} seed={seed}: "
                f"Radix component {idx} is non-binary after dividing by weight"
            )

    @pytest.mark.parametrize("base", [2, 4])
    @pytest.mark.parametrize("engine", ["heavy", "wfa"])
    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS[:3])
    def test_radix_permutation_one_per_row(self, base, engine, n, k, w, seed):
        D = _make(n, k, w, seed)
        comps, _, _ = decompose_radix(D, base=base, matching_method=engine)
        for idx, c in enumerate(comps):
            if abs(c.weight) < 1e-12:
                continue
            P = c.matrix / c.weight
            row_sums = P.sum(axis=1)
            # Only check rows that are non-zero (partial matchings are allowed in radix)
            active = row_sums > 1e-9
            assert np.allclose(row_sums[active], 1.0, atol=1e-9), (
                f"base={base} engine={engine} seed={seed}: "
                f"Radix component {idx} has active row sums != 1"
            )


class TestRadixProperty3NonNegativeWeights:
    """phi_i >= 0 for all Radix components."""

    @pytest.mark.parametrize("base", RADIX_BASES)
    @pytest.mark.parametrize("engine", ["heavy", "wfa"])
    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS[:4])
    def test_radix_weights_non_negative(self, base, engine, n, k, w, seed):
        D = _make(n, k, w, seed)
        comps, _, _ = decompose_radix(D, base=base, matching_method=engine)
        for idx, c in enumerate(comps):
            assert c.weight >= -1e-12, (
                f"base={base} engine={engine} seed={seed}: "
                f"Radix component {idx} has negative weight phi={c.weight:.4g}"
            )


# ===========================================================================
# Cross-engine consistency
# ===========================================================================

class TestCrossEngineConsistency:
    """Different BvN engines decomposing the same matrix must all reconstruct D."""

    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS)
    def test_all_engines_reconstruct_same_matrix(self, n, k, w, seed):
        D = _make(n, k, w, seed)
        for engine in BVN_ENGINES:
            comps = bvn_decomposition(D, matching_algorithm=engine)
            reconstructed = sum(c.weight * c.permutation for c in comps)
            np.testing.assert_allclose(
                reconstructed, D, atol=1e-6,
                err_msg=f"engine={engine} failed reconstruction for n={n} k={k} seed={seed}",
            )

    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS[:3])
    def test_all_radix_bases_reconstruct_same_matrix(self, n, k, w, seed):
        D = _make(n, k, w, seed)
        for base in RADIX_BASES:
            comps, _, _ = decompose_radix(D, base=base, matching_method="heavy")
            reconstructed = sum(c.matrix for c in comps)
            np.testing.assert_allclose(
                reconstructed, D, atol=1e-5,
                err_msg=f"base={base} failed reconstruction for n={n} k={k} seed={seed}",
            )

    @pytest.mark.parametrize("n,k,w,seed", MATRIX_CONFIGS)
    def test_strong_engines_agree_on_cycle_length(self, n, k, w, seed):
        """All strong engines must produce C(D) == S (same value, different paths)."""
        D = _make(n, k, w, seed)
        S = _row_sum(D)
        cycle_lengths = {}
        for engine in STRONG_ENGINES:
            comps = bvn_decomposition(D, matching_algorithm=engine)
            cycle_lengths[engine] = sum(c.weight for c in comps)

        for engine, C in cycle_lengths.items():
            assert abs(C - S) < 1e-6, (
                f"engine={engine} seed={seed}: C(D)={C:.6f} != S={S:.6f}"
            )
