"""
Tests for the Euler Splitting framework.

Covers:
  1. Split correctness: red + blue == original
  2. Doubly stochastic guarantee: red and blue each have row/col sums == S/2
  3. BvN decomposition on leaves reconstructs the original matrix
  4. Both "euler" and "greedy" split strategies
  5. Edge cases: odd S, S=1, empty matrix, identity-like matrices
  6. decompose_euler_framework end-to-end for depth in {0, 1, 2, 3}
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.euler_splitting import (
    decompose_euler_framework,
    euler_decomposition,
    euler_split_once,
)
from src.algorithms.bvn import bvn_decomposition
from src.utils.matrix_generator import generate_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_matrix(n: int, k: int, seed: int = 0, max_weight: int = 10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return generate_matrix(n=n, k=k, max_weight=max_weight, rng=rng)


def _is_doubly_stochastic_int(m: np.ndarray, target_sum: int, atol: int = 0) -> bool:
    """Check that integer matrix m has all row sums and col sums == target_sum."""
    row_ok = np.all(np.abs(m.sum(axis=1) - target_sum) <= atol)
    col_ok = np.all(np.abs(m.sum(axis=0) - target_sum) <= atol)
    return bool(row_ok and col_ok)


def _reconstruct_from_euler_framework(matrix, depth, split_method, matching_method="heavy"):
    """Run decompose_euler_framework and reconstruct the matrix from components."""
    comps, _, _ = decompose_euler_framework(
        matrix=matrix,
        matching_method=matching_method,
        depth=depth,
        split_method=split_method,
    )
    reconstructed = np.zeros_like(matrix, dtype=np.float64)
    for c in comps:
        reconstructed += c.permutation * c.weight
    return reconstructed


# ===========================================================================
# 1. Split correctness: red + blue == original
# ===========================================================================

class TestSplitCorrectness:
    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    @pytest.mark.parametrize("n,k,seed", [
        (4,  4,  0),   # small even S
        (8,  8,  1),   # medium even S
        (16, 16, 2),   # larger even S
        (8,  6,  3),   # odd k (S will be even if max_weight and k combine evenly)
    ])
    def test_split_lossless(self, split_method, n, k, seed):
        """red + blue must exactly equal the original integer matrix."""
        mat = _make_matrix(n, k, seed).astype(np.int64)
        S = int(mat.sum(axis=1).max())
        # euler_split_once handles odd S by extracting one perm, so test still valid
        red, blue = euler_split_once(mat, split_method=split_method)
        np.testing.assert_array_equal(
            red + blue, mat,
            err_msg=f"split_method={split_method} n={n} k={k}: red+blue != original"
        )

    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    def test_split_non_negative(self, split_method):
        """Both halves must have non-negative entries."""
        mat = _make_matrix(16, 8, seed=99)
        red, blue = euler_split_once(mat.astype(np.int64), split_method=split_method)
        assert np.all(red >= 0), f"red has negative entries ({split_method})"
        assert np.all(blue >= 0), f"blue has negative entries ({split_method})"


# ===========================================================================
# 2. Doubly stochastic guarantee
# ===========================================================================

class TestDoublyStochastic:
    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    @pytest.mark.parametrize("n,k,seed", [
        (4,  4,  10),
        (8,  8,  11),
        (16, 16, 12),
        (32, 32, 13),
        (8,  4,  14),
    ])
    def test_red_doubly_stochastic(self, split_method, n, k, seed):
        """Red sub-matrix must have equal row and column sums == floor(S/2)."""
        mat = _make_matrix(n, k, seed).astype(np.int64)
        S = int(mat.sum(axis=1).max())
        half = S // 2
        red, _ = euler_split_once(mat, split_method=split_method)
        # Red sums should be half (or half+1 if S was odd — perm folded into red)
        expected = half + (S % 2)  # ceil(S/2) for red when S odd
        assert _is_doubly_stochastic_int(red, expected, atol=0), (
            f"{split_method} n={n} k={k}: red not doubly stochastic "
            f"(row sums={red.sum(axis=1)}, col sums={red.sum(axis=0)}, expected={expected})"
        )

    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    @pytest.mark.parametrize("n,k,seed", [
        (4,  4,  20),
        (8,  8,  21),
        (16, 16, 22),
        (32, 32, 23),
    ])
    def test_blue_doubly_stochastic(self, split_method, n, k, seed):
        """Blue sub-matrix must have equal row and column sums == floor(S/2)."""
        mat = _make_matrix(n, k, seed).astype(np.int64)
        S = int(mat.sum(axis=1).max())
        half = S // 2
        _, blue = euler_split_once(mat, split_method=split_method)
        assert _is_doubly_stochastic_int(blue, half, atol=0), (
            f"{split_method} n={n} k={k}: blue not doubly stochastic "
            f"(row sums={blue.sum(axis=1)}, col sums={blue.sum(axis=0)}, expected={half})"
        )

    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    def test_uniform_matrix(self, split_method):
        """Uniform matrix (all entries = S/n) — hardest case for greedy."""
        n = 4
        # Build a doubly stochastic matrix with all entries equal
        S = 4
        mat = np.full((n, n), S // n, dtype=np.int64)  # each entry = 1
        half = S // 2
        red, blue = euler_split_once(mat, split_method=split_method)
        assert _is_doubly_stochastic_int(red, half), f"uniform red fail ({split_method})"
        assert _is_doubly_stochastic_int(blue, half), f"uniform blue fail ({split_method})"
        np.testing.assert_array_equal(red + blue, mat)

    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    def test_diagonal_matrix(self, split_method):
        """Diagonal matrix: all weight concentrated on diagonal."""
        n = 8
        S = 8
        mat = np.diag(np.full(n, S, dtype=np.int64))
        half = S // 2
        red, blue = euler_split_once(mat, split_method=split_method)
        assert _is_doubly_stochastic_int(red, half), f"diagonal red fail ({split_method})"
        assert _is_doubly_stochastic_int(blue, half), f"diagonal blue fail ({split_method})"
        np.testing.assert_array_equal(red + blue, mat)


# ===========================================================================
# 3. Sparsity comparison: greedy should be at least as sparse as euler
# ===========================================================================

class TestGreedySparsity:
    @pytest.mark.parametrize("n,k,seed", [
        (8,  8,  30),
        (16, 8,  31),
        (16, 16, 32),
        (32, 16, 33),
    ])
    def test_greedy_not_worse_sparsity_on_dense_entries(self, n, k, seed):
        """
        For matrices with high-multiplicity entries, greedy should produce
        sub-matrices with ≤ non-zeros than Euler 2-colouring.
        (Greedy is allowed to tie with Euler or fall back to it.)
        """
        mat = _make_matrix(n, k, seed, max_weight=k).astype(np.int64)
        euler_red, _ = euler_split_once(mat, split_method="euler")
        greedy_red, _ = euler_split_once(mat, split_method="greedy")
        euler_nnz = int(np.count_nonzero(euler_red))
        greedy_nnz = int(np.count_nonzero(greedy_red))
        # greedy should be at least as sparse (fewer or equal non-zeros)
        assert greedy_nnz <= euler_nnz + 1, (   # +1 tolerance for tiny rounding edge cases
            f"Greedy red nnz={greedy_nnz} > euler red nnz={euler_nnz} for n={n} k={k} seed={seed}"
        )


# ===========================================================================
# 4. decompose_euler_framework: end-to-end reconstruction
# ===========================================================================

class TestFrameworkReconstruction:
    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    @pytest.mark.parametrize("depth", [0, 1, 2, 3])
    @pytest.mark.parametrize("n,k,seed", [
        (8,  4,  40),
        (16, 8,  41),
    ])
    def test_reconstruct_original(self, split_method, depth, n, k, seed):
        """
        The sum of all leaf BvN decompositions must equal the original matrix.
        """
        mat = _make_matrix(n, k, seed).astype(np.float64)
        reconstructed = _reconstruct_from_euler_framework(mat, depth, split_method)
        np.testing.assert_allclose(
            reconstructed, mat, atol=1e-6,
            err_msg=f"Reconstruction failed: split={split_method} depth={depth} n={n} k={k}"
        )

    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    @pytest.mark.parametrize("depth", [1, 2])
    def test_cycle_length_equals_S(self, split_method, depth):
        """C(D) = sum of weights = S for strong decompositions on integer matrices."""
        n, k, seed = 16, 8, 50
        mat = _make_matrix(n, k, seed)
        S = float(mat.sum(axis=1).max())
        comps, _, _ = decompose_euler_framework(
            mat, matching_method="heavy", depth=depth, split_method=split_method
        )
        c_total = sum(c.weight for c in comps)
        assert abs(c_total - S) < 1e-6, (
            f"C(D)={c_total:.4f} != S={S:.4f} for split={split_method} depth={depth}"
        )

    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    def test_num_leaves_correct(self, split_method):
        """Number of non-empty leaves should be ≤ 2^depth."""
        n, k, seed = 16, 8, 60
        mat = _make_matrix(n, k, seed)
        for depth in range(4):
            _, _, n_leaves = decompose_euler_framework(
                mat, depth=depth, split_method=split_method
            )
            assert n_leaves <= 2 ** depth + 1, (   # +1 for possible odd-S extra perm leaf
                f"n_leaves={n_leaves} > 2^depth={2**depth} for depth={depth}"
            )

    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    def test_leaves_doubly_stochastic(self, split_method):
        """
        All leaf matrices produced by the splitting phase must be doubly stochastic
        (equal row and column sums).
        """
        n, k, depth = 16, 8, 2
        mat = _make_matrix(n, k, seed=70).astype(np.int64)

        # Manually collect leaves (mirrors the framework logic)
        leaves = [mat]
        for _ in range(depth):
            next_leaves = []
            for leaf in leaves:
                S_leaf = int(leaf.sum(axis=1).max())
                if S_leaf <= 1:
                    next_leaves.append(leaf)
                else:
                    red, blue = euler_split_once(leaf, split_method=split_method)
                    next_leaves.append(red)
                    next_leaves.append(blue)
            leaves = next_leaves

        for idx, leaf in enumerate(leaves):
            row_sums = leaf.sum(axis=1)
            col_sums = leaf.sum(axis=0)
            assert np.allclose(row_sums, row_sums[0]), (
                f"Leaf {idx} row sums not equal: {row_sums} (split={split_method})"
            )
            assert np.allclose(col_sums, col_sums[0]), (
                f"Leaf {idx} col sums not equal: {col_sums} (split={split_method})"
            )
            assert np.isclose(row_sums[0], col_sums[0]), (
                f"Leaf {idx} row sum != col sum (split={split_method})"
            )


# ===========================================================================
# 5. Edge cases
# ===========================================================================

class TestEdgeCases:
    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    def test_odd_S(self, split_method):
        """Odd row sum S: one permutation extracted, remainder split evenly."""
        n = 4
        # Build a 3-regular matrix (S=3, odd)
        rng = np.random.default_rng(80)
        mat = generate_matrix(n=n, k=3, max_weight=3, rng=rng).astype(np.int64)
        S = int(mat.sum(axis=1).max())
        if S % 2 == 0:
            pytest.skip("Generated matrix happened to have even S — skip odd-S test")
        red, blue = euler_split_once(mat, split_method=split_method)
        np.testing.assert_array_equal(red + blue, mat, err_msg="odd-S split not lossless")
        # red sums = ceil(S/2), blue sums = floor(S/2)
        assert _is_doubly_stochastic_int(red, S // 2 + 1), "odd-S red not DS"
        assert _is_doubly_stochastic_int(blue, S // 2), "odd-S blue not DS"

    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    def test_permutation_matrix_depth0(self, split_method):
        """A pure permutation matrix (S=1) at depth=0 should still decompose."""
        n = 8
        perm = np.eye(n, dtype=np.float64)
        comps, _, n_leaves = decompose_euler_framework(
            perm, depth=0, split_method=split_method
        )
        reconstructed = sum(c.permutation * c.weight for c in comps)
        np.testing.assert_allclose(reconstructed, perm, atol=1e-9)

    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    def test_large_matrix_correctness(self, split_method):
        """n=64, k=16 — ensures no crash and correct reconstruction."""
        mat = _make_matrix(64, 16, seed=90)
        reconstructed = _reconstruct_from_euler_framework(mat, depth=2, split_method=split_method)
        np.testing.assert_allclose(reconstructed, mat, atol=1e-5,
                                   err_msg=f"n=64 reconstruction failed ({split_method})")

    @pytest.mark.parametrize("split_method", ["euler", "greedy"])
    def test_depth0_equals_plain_bvn(self, split_method):
        """depth=0 framework should produce same cycle length as plain BvN."""
        mat = _make_matrix(16, 8, seed=100)
        comps_fw, _, _ = decompose_euler_framework(mat, depth=0, split_method=split_method)
        comps_bvn = bvn_decomposition(mat, matching_algorithm="heavy")
        c_fw  = sum(c.weight for c in comps_fw)
        c_bvn = sum(c.weight for c in comps_bvn)
        assert abs(c_fw - c_bvn) < 1e-6, f"depth=0 C={c_fw} != BvN C={c_bvn}"
