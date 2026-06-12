"""
Tests for the Euler Splitting framework.

Covers:
1. Split correctness: red + blue == original
2. Doubly stochastic guarantee: red and blue each have row/col sums == S/2
3. BvN decomposition on leaves reconstructs the original matrix
4. Both "euler" and "greedy" split strategies
5. Edge cases: odd S, S=1, empty matrix, identity-like matrices
6. decompose_euler_framework end-to-end for depth in {0, 1, 2, 3}
7. Fuzz: 50 random (n, k, max_weight) combinations
8. Visualization: heatmaps of splits and leaf trees saved to tests/visual_output/

Run only visualisation tests:
    pytest tests/test_euler_splitting.py -m visual -v
"""

from __future__ import annotations

import os
from pathlib import Path

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


def check_split_invariants(
    original: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    split_method: str = "",
) -> None:
    """
    Assert all mathematical invariants that must hold after any valid split.

    For even S:   red sums == blue sums == S//2
    For odd  S:   red sums == S//2 + 1,  blue sums == S//2
                  (the extra permutation extracted for odd S is folded into red)
    """
    S = int(np.round(original).astype(np.int64).sum(axis=1).max())
    half = S // 2
    label = f"[{split_method}]" if split_method else ""

    assert np.all(red >= 0),   f"{label} red has negative entries"
    assert np.all(blue >= 0),  f"{label} blue has negative entries"
    np.testing.assert_array_equal(
        red + blue,
        np.round(original).astype(np.int64),
        err_msg=f"{label} red + blue != original",
    )

    expected_red = half + (S % 2)   # ceil(S/2)
    expected_blue = half             # floor(S/2)

    assert _is_doubly_stochastic_int(red, expected_red), (
        f"{label} red not doubly stochastic: "
        f"row={red.sum(axis=1).tolist()} col={red.sum(axis=0).tolist()} expected={expected_red}"
    )
    assert _is_doubly_stochastic_int(blue, expected_blue), (
        f"{label} blue not doubly stochastic: "
        f"row={blue.sum(axis=1).tolist()} col={blue.sum(axis=0).tolist()} expected={expected_blue}"
    )


def _reconstruct_from_euler_framework(matrix, depth, split_method, matching_method="heavy"):
    """Run decompose_euler_framework and reconstruct the matrix from components."""
    comps, _, _, _ = decompose_euler_framework(
        matrix=matrix,
        matching_method=matching_method,
        depth=depth,
        split_method=split_method,
    )
    reconstructed = np.zeros_like(matrix, dtype=np.float64)
    for c in comps:
        reconstructed += c.permutation * c.weight
    return reconstructed


def _collect_leaves(matrix: np.ndarray, depth: int, split_method: str):
    """Manually replicate the splitting loop from decompose_euler_framework."""
    leaves = [np.round(matrix).astype(np.int64)]
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
    return leaves


# ===========================================================================
# 1. Split correctness: red + blue == original
# ===========================================================================

class TestSplitCorrectness:
    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
    @pytest.mark.parametrize("n,k,seed", [
        (4,  4,  0),
        (8,  8,  1),
        (16, 16, 2),
        (8,  6,  3),
    ])
    def test_split_lossless(self, split_method, n, k, seed):
        """red + blue must exactly equal the original integer matrix."""
        mat = _make_matrix(n, k, seed).astype(np.int64)
        red, blue = euler_split_once(mat, split_method=split_method)
        np.testing.assert_array_equal(
            red + blue, mat,
            err_msg=f"split_method={split_method} n={n} k={k}: red+blue != original"
        )

    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
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
    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
    @pytest.mark.parametrize("n,k,seed", [
        (4,  4,  10),
        (8,  8,  11),
        (16, 16, 12),
        (32, 32, 13),
        (8,  4,  14),
    ])
    def test_red_doubly_stochastic(self, split_method, n, k, seed):
        """Red sub-matrix must have equal row and column sums == ceil(S/2)."""
        mat = _make_matrix(n, k, seed).astype(np.int64)
        S = int(mat.sum(axis=1).max())
        red, _ = euler_split_once(mat, split_method=split_method)
        expected = S // 2 + (S % 2)   # ceil(S/2)
        assert _is_doubly_stochastic_int(red, expected, atol=0), (
            f"{split_method} n={n} k={k}: red not doubly stochastic "
            f"(row sums={red.sum(axis=1)}, col sums={red.sum(axis=0)}, expected={expected})"
        )

    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
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

    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
    def test_uniform_matrix(self, split_method):
        """Uniform matrix (all entries = S/n) — hardest case for greedy (forces fallback)."""
        n, S = 4, 4
        mat = np.full((n, n), S // n, dtype=np.int64)
        check_split_invariants(mat, *euler_split_once(mat, split_method), split_method)

    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
    def test_diagonal_matrix(self, split_method):
        """Diagonal matrix: all weight concentrated on diagonal."""
        n, S = 8, 8
        mat = np.diag(np.full(n, S, dtype=np.int64))
        check_split_invariants(mat, *euler_split_once(mat, split_method), split_method)


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
    def test_greedy_never_worse_than_euler(self, n, k, seed):
        """
        Greedy either wins (fewer non-zeros) or ties (fell back to Euler).
        It should NEVER produce more non-zeros than the Euler 2-coloring.
        """
        mat = _make_matrix(n, k, seed, max_weight=k).astype(np.int64)
        euler_red, _ = euler_split_once(mat, split_method="euler")
        greedy_red, _ = euler_split_once(mat, split_method="greedy")
        euler_nnz  = int(np.count_nonzero(euler_red))
        greedy_nnz = int(np.count_nonzero(greedy_red))
        assert greedy_nnz <= euler_nnz, (
            f"Greedy red nnz={greedy_nnz} > euler red nnz={euler_nnz} "
            f"for n={n} k={k} seed={seed} — greedy must never be worse"
        )

    def test_greedy_strictly_sparser_on_concentrated(self):
        """
        On matrices with at least one dominant entry (>= half of row sum),
        greedy can assign the whole block to one side → fewer non-zeros than
        Euler 2-coloring, which splits every edge along the tour.
        Over 100 random concentrated matrices, greedy should win on at least some.
        """
        rng = np.random.default_rng(42)
        wins = 0
        for seed in range(100):
            # high max_weight → many entries will be >= half(S)
            mat = generate_matrix(n=8, k=4, max_weight=16, rng=rng).astype(np.int64)
            if mat.sum(axis=1).max() == 0:
                continue
            euler_red, _  = euler_split_once(mat, split_method="euler")
            greedy_red, _ = euler_split_once(mat, split_method="greedy")
            nnz_e = int(np.count_nonzero(euler_red))
            nnz_g = int(np.count_nonzero(greedy_red))
            # greedy must never be worse
            assert nnz_g <= nnz_e, (
                f"seed={seed}: greedy nnz={nnz_g} > euler nnz={nnz_e}"
            )
            if nnz_g < nnz_e:
                wins += 1

        assert wins > 0, (
            "Greedy never beat Euler over 100 concentrated matrices — "
            "greedy may be silently always falling back to Euler"
        )


# ===========================================================================
# 4. decompose_euler_framework: end-to-end reconstruction
# ===========================================================================

class TestFrameworkReconstruction:
    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
    @pytest.mark.parametrize("depth", [0, 1, 2, 3])
    @pytest.mark.parametrize("n,k,seed", [
        (8,  4,  40),
        (16, 8,  41),
    ])
    def test_reconstruct_original(self, split_method, depth, n, k, seed):
        """The sum of all leaf BvN decompositions must equal the original matrix."""
        mat = _make_matrix(n, k, seed).astype(np.float64)
        reconstructed = _reconstruct_from_euler_framework(mat, depth, split_method)
        np.testing.assert_allclose(
            reconstructed, mat, atol=1e-6,
            err_msg=f"Reconstruction failed: split={split_method} depth={depth} n={n} k={k}"
        )

    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
    @pytest.mark.parametrize("depth", [1, 2])
    def test_cycle_length_equals_S(self, split_method, depth):
        """C(D) = sum of weights = S for strong decompositions on integer matrices."""
        n, k, seed = 16, 8, 50
        mat = _make_matrix(n, k, seed)
        S = float(mat.sum(axis=1).max())
        comps, _, _, _ = decompose_euler_framework(
            mat, matching_method="heavy", depth=depth, split_method=split_method
        )
        c_total = sum(c.weight for c in comps)
        assert abs(c_total - S) < 1e-6, (
            f"C(D)={c_total:.4f} != S={S:.4f} for split={split_method} depth={depth}"
        )

    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
    def test_num_leaves_correct(self, split_method):
        """Number of non-empty leaves should be <= 2^depth."""
        n, k, seed = 16, 8, 60
        mat = _make_matrix(n, k, seed)
        for depth in range(4):
            _, _, _, n_leaves = decompose_euler_framework(
                mat, depth=depth, split_method=split_method
            )
            assert n_leaves <= 2 ** depth + 1, (
                f"n_leaves={n_leaves} > 2^depth={2**depth} for depth={depth}"
            )

    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
    def test_leaves_doubly_stochastic(self, split_method):
        """All leaf matrices must have equal row and column sums."""
        n, k, depth = 16, 8, 2
        mat = _make_matrix(n, k, seed=70).astype(np.int64)
        leaves = _collect_leaves(mat, depth, split_method)
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
    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
    def test_odd_S(self, split_method):
        """Odd row sum S: guaranteed by summing exactly 3 unit-weight permutations."""
        n = 8
        rng = np.random.default_rng(80)
        # Sum 3 permutations with weight 1 each → S = 3 (always odd), no skip needed
        mat = np.zeros((n, n), dtype=np.int64)
        for _ in range(3):
            p = rng.permutation(n)
            mat[np.arange(n), p] += 1
        S = int(mat.sum(axis=1).max())
        assert S % 2 == 1, "Bug in test setup: S should be odd"
        red, blue = euler_split_once(mat, split_method=split_method)
        np.testing.assert_array_equal(red + blue, mat, err_msg="odd-S split not lossless")
        assert _is_doubly_stochastic_int(red,  S // 2 + 1), f"odd-S red not DS ({split_method})"
        assert _is_doubly_stochastic_int(blue, S // 2),     f"odd-S blue not DS ({split_method})"

    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
    def test_permutation_matrix_depth0(self, split_method):
        """A pure permutation matrix (S=1) at depth=0 should still decompose."""
        n = 8
        perm = np.eye(n, dtype=np.float64)
        comps, _, _, _ = decompose_euler_framework(perm, depth=0, split_method=split_method)
        reconstructed = sum(c.permutation * c.weight for c in comps)
        np.testing.assert_allclose(reconstructed, perm, atol=1e-9)

    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
    def test_large_matrix_correctness(self, split_method):
        """n=64, k=16 — no crash and correct reconstruction."""
        mat = _make_matrix(64, 16, seed=90)
        reconstructed = _reconstruct_from_euler_framework(mat, depth=2, split_method=split_method)
        np.testing.assert_allclose(reconstructed, mat, atol=1e-5,
                                   err_msg=f"n=64 reconstruction failed ({split_method})")

    @pytest.mark.parametrize("split_method", ["euler", "euler_grouped", "greedy"])
    def test_depth0_equals_plain_bvn(self, split_method):
        """depth=0 framework should produce same cycle length as plain BvN."""
        mat = _make_matrix(16, 8, seed=100)
        comps_fw, _, _, _ = decompose_euler_framework(mat, depth=0, split_method=split_method)
        comps_bvn = bvn_decomposition(mat, matching_algorithm="heavy")
        c_fw  = sum(c.weight for c in comps_fw)
        c_bvn = sum(c.weight for c in comps_bvn)
        assert abs(c_fw - c_bvn) < 1e-6, f"depth=0 C={c_fw} != BvN C={c_bvn}"


# ===========================================================================
# 6. Fuzz: randomised sweep over 50 seeds
# ===========================================================================

class TestFuzz:
    @pytest.mark.parametrize("seed", range(50))
    def test_fuzz_split_invariants(self, seed):
        """
        Random (n, k, max_weight) for each seed.
        Both strategies must satisfy all split invariants every time.
        This catches rare budget-arithmetic bugs in the greedy path.
        """
        rng = np.random.default_rng(seed)
        n          = int(rng.integers(4, 17))
        k          = int(rng.integers(2, 13))
        max_weight = int(rng.integers(1, 9))
        mat = generate_matrix(n=n, k=k, max_weight=max_weight, rng=rng).astype(np.int64)

        for method in ("euler", "greedy"):
            red, blue = euler_split_once(mat, split_method=method)
            check_split_invariants(mat, red, blue, method)


# ===========================================================================
# 7. Heuristic split correctness
# ===========================================================================

class TestHeuristicCorrectness:
    """
    Two invariants must hold for EVERY heuristic split:

      1. Lossless — red + blue == original matrix (no weight created or lost)
      2. Doubly stochastic — both halves have row sums = col sums = S/2
         (for odd S, red gets ceil(S/2) and blue gets floor(S/2))
    """

    # ── small matrices, many seeds ───────────────────────────────────────
    @pytest.mark.parametrize("seed", range(20))
    @pytest.mark.parametrize("n,k,max_w", [
        (4,  2,  4),
        (8,  4,  8),
        (16, 8, 15),
    ])
    def test_lossless_small(self, n, k, max_w, seed):
        """red + blue must exactly equal the original matrix."""
        rng = np.random.default_rng(seed)
        M   = generate_matrix(n=n, k=k, max_weight=max_w, rng=rng).astype(np.int64)
        red, blue = euler_split_once(M, split_method="heuristic")
        np.testing.assert_array_equal(
            red + blue, M,
            err_msg=f"lossless failed: n={n} k={k} seed={seed}",
        )

    @pytest.mark.parametrize("seed", range(20))
    @pytest.mark.parametrize("n,k,max_w", [
        (4,  2,  4),
        (8,  4,  8),
        (16, 8, 15),
    ])
    def test_doubly_stochastic_small(self, n, k, max_w, seed):
        """Both halves must have equal row and column sums (S/2 each)."""
        rng = np.random.default_rng(seed)
        M   = generate_matrix(n=n, k=k, max_weight=max_w, rng=rng).astype(np.int64)
        S   = int(M.sum(axis=1).max())
        red, blue = euler_split_once(M, split_method="heuristic")

        half_hi = S // 2 + S % 2   # ceil(S/2) — red gets this when S is odd
        half_lo = S // 2            # floor(S/2) — blue always gets this

        assert np.all(red.sum(axis=1)  == half_hi), \
            f"red row sums != {half_hi}: n={n} k={k} seed={seed}"
        assert np.all(red.sum(axis=0)  == half_hi), \
            f"red col sums != {half_hi}: n={n} k={k} seed={seed}"
        assert np.all(blue.sum(axis=1) == half_lo), \
            f"blue row sums != {half_lo}: n={n} k={k} seed={seed}"
        assert np.all(blue.sum(axis=0) == half_lo), \
            f"blue col sums != {half_lo}: n={n} k={k} seed={seed}"

    # ── fuzz: random n, k, max_w ─────────────────────────────────────────
    @pytest.mark.parametrize("seed", range(50))
    def test_fuzz_heuristic(self, seed):
        """Full invariant check on random matrices — reuses check_split_invariants."""
        rng = np.random.default_rng(seed + 1000)   # different seed space from TestFuzz
        n   = int(rng.integers(4, 17))
        k   = int(rng.integers(2, 13))
        mw  = int(rng.integers(1, 9))
        mat = generate_matrix(n=n, k=k, max_weight=mw, rng=rng).astype(np.int64)
        red, blue = euler_split_once(mat, split_method="heuristic")
        check_split_invariants(mat, red, blue, "heuristic")

    # ── 100 matrices: non-negativity, reconstruction, cycle length ───────

    @pytest.mark.parametrize("seed", range(100))
    def test_non_negative_100(self, seed):
        """red[i,j] >= 0 and blue[i,j] >= 0 everywhere (100 matrices)."""
        rng = np.random.default_rng(seed + 5000)
        M   = generate_matrix(n=32, k=16, max_weight=8, rng=rng).astype(np.int64)
        red, blue = euler_split_once(M, split_method="heuristic")
        assert np.all(red  >= 0), f"red has negative entries  (seed={seed})"
        assert np.all(blue >= 0), f"blue has negative entries (seed={seed})"

    @pytest.mark.parametrize("seed", range(100))
    def test_reconstruction_100(self, seed):
        """
        Full pipeline: split with heuristic → BvN on each leaf →
        sum of all phi_i * P_i must exactly reconstruct M.
        """
        rng = np.random.default_rng(seed + 6000)
        M   = generate_matrix(n=32, k=16, max_weight=8, rng=rng).astype(np.float64)
        comps, _, _, _ = decompose_euler_framework(
            M, matching_method="heavy", depth=1, split_method="heuristic"
        )
        reconstructed = sum(c.weight * c.permutation for c in comps)
        np.testing.assert_allclose(
            reconstructed, M, atol=1e-6,
            err_msg=f"reconstruction failed (seed={seed})",
        )

    @pytest.mark.parametrize("seed", range(100))
    def test_cycle_length_100(self, seed):
        """
        C = sum(phi_i) must equal S after the full heuristic-split decomposition.
        Each leaf is (S/2)-regular, so BvN on each leaf gives C_leaf = S/2, and
        the two leaves together give C_total = S.
        """
        rng = np.random.default_rng(seed + 7000)
        M   = generate_matrix(n=32, k=16, max_weight=8, rng=rng).astype(np.float64)
        S   = float(M.sum(axis=1).max())
        comps, _, _, _ = decompose_euler_framework(
            M, matching_method="heavy", depth=1, split_method="heuristic"
        )
        C = float(sum(c.weight for c in comps))
        assert abs(C - S) < 1e-6, (
            f"cycle length C={C:.4f} != S={S:.4f}  (seed={seed})"
        )

    # ── paper parameters (n=256) ─────────────────────────────────────────
    @pytest.mark.parametrize("seed", [42, 7, 123])
    def test_lossless_paper_params(self, seed):
        """Lossless at paper scale: n=256, k=256, max_weight=64."""
        rng = np.random.default_rng(seed)
        M   = generate_matrix(n=256, k=256, max_weight=64, rng=rng).astype(np.int64)
        red, blue = euler_split_once(M, split_method="heuristic")
        np.testing.assert_array_equal(
            red + blue, M,
            err_msg=f"lossless failed at paper params seed={seed}",
        )

    @pytest.mark.parametrize("seed", [42, 7, 123])
    def test_doubly_stochastic_paper_params(self, seed):
        """Both halves are S/2-regular at paper scale: n=256, k=256, max_weight=64."""
        rng = np.random.default_rng(seed)
        M   = generate_matrix(n=256, k=256, max_weight=64, rng=rng).astype(np.int64)
        S   = int(M.sum(axis=1).max())
        red, blue = euler_split_once(M, split_method="heuristic")

        half_hi = S // 2 + S % 2   # ceil(S/2) — red gets this when S is odd
        half_lo = S // 2            # floor(S/2) — blue always gets this

        assert np.all(red.sum(axis=1)  == half_hi), \
            f"red row sums != {half_hi} (paper params seed={seed}, S={S})"
        assert np.all(red.sum(axis=0)  == half_hi), \
            f"red col sums != {half_hi} (paper params seed={seed}, S={S})"
        assert np.all(blue.sum(axis=1) == half_lo), \
            f"blue row sums != {half_lo} (paper params seed={seed}, S={S})"
        assert np.all(blue.sum(axis=0) == half_lo), \
            f"blue col sums != {half_lo} (paper params seed={seed}, S={S})"


# ===========================================================================
# 8. Visualisation: heatmaps saved to tests/visual_output/
#    Run with:  pytest tests/test_euler_splitting.py -m visual -v
# ===========================================================================

def _heatmap(ax, data: np.ndarray, title: str, vmin: float, vmax: float) -> None:
    """Draw an annotated integer heatmap on ax."""
    im = ax.imshow(data, aspect="equal", vmin=vmin, vmax=vmax,
                   cmap="YlOrRd", interpolation="nearest")
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    n = data.shape[0]
    # Annotate cells only for small matrices (readability)
    if n <= 12:
        for i in range(n):
            for j in range(n):
                v = int(data[i, j])
                if v > 0:
                    ax.text(j, i, str(v), ha="center", va="center",
                            fontsize=max(5, 8 - n // 4), color="black")
    return im


def _out_dir() -> Path:
    out = Path(__file__).parent / "visual_output"
    out.mkdir(exist_ok=True)
    return out


@pytest.mark.visual
class TestVisualization:
    """
    Each test saves one PNG to tests/visual_output/.
    These tests always pass (they are for inspection, not assertion).
    """

    def test_vis_single_split(self):
        """
        Show original matrix alongside its Euler and Greedy splits.
        Uses k=2 permutations with high max_weight so one permutation dominates —
        this makes entries >= S/2 common, giving greedy real structural work to do
        (assign whole dominant blocks to one side rather than splitting everything).

        Layout (2 rows × 3 cols):
          Row 0: original | euler-red  | euler-blue
          Row 1: original | greedy-red | greedy-blue
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # k=2: two permutations, high weights → one will likely dominate rows.
        # Euler halves every entry in place; greedy concentrates the dominant perm
        # entirely into red and the weak perm into blue → sparser leaves.
        mat = _make_matrix(n=8, k=2, max_weight=20, seed=3).astype(np.int64)
        S   = int(mat.sum(axis=1).max())
        vmax = int(mat.max())

        fig, axes = plt.subplots(2, 3, figsize=(10, 7))
        fig.suptitle(
            f"One-level split comparison  (n=8, k=2, max_w=20, S={S})\n"
            f"Euler halves every entry; Greedy assigns dominant blocks whole to one side",
            fontsize=10,
        )

        for row, method in enumerate(("euler", "greedy")):
            red, blue = euler_split_once(mat, split_method=method)
            S_red  = int(red.sum(axis=1).max())
            S_blue = int(blue.sum(axis=1).max())
            im = _heatmap(axes[row, 0], mat,
                          f"Original  S={S}  nnz={np.count_nonzero(mat)}", 0, vmax)
            _heatmap(axes[row, 1], red,
                     f"{method.capitalize()} red   S={S_red}  nnz={np.count_nonzero(red)}",
                     0, vmax)
            _heatmap(axes[row, 2], blue,
                     f"{method.capitalize()} blue  S={S_blue}  nnz={np.count_nonzero(blue)}",
                     0, vmax)
            axes[row, 0].set_ylabel(method, fontsize=10, fontweight="bold")

        # Shared colorbar on the right
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        sm = cm.ScalarMappable(cmap="YlOrRd",
                               norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Entry value")

        out = _out_dir() / "split_euler_vs_greedy.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[visual] saved -> {out}")

    def test_vis_depth_tree(self):
        """
        Show the leaf matrices produced at each depth (0–3) for both strategies.

        Uses k=4, max_weight=12 so the original matrix has concentrated entries
        (few large values per row) — the halving of S at each level is meaningful
        and greedy can produce sparser leaves than Euler.

        Layout: one figure per strategy.
          Rows  = depth levels (0, 1, 2, 3)
          Cols  = individual leaf matrices at that depth
        Each cell shows S_leaf and nnz so the tree-pruning effect is visible.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # k=4, high max_weight: each row has 1-4 concentrated non-zero entries.
        # Greedy can assign dominant entries whole to one side → sparser leaves.
        n, k, max_w, seed = 8, 4, 12, 42
        mat = _make_matrix(n=n, k=k, seed=seed, max_weight=max_w).astype(np.int64)
        S_orig = int(mat.sum(axis=1).max())
        vmax   = int(mat.max())

        for method in ("euler", "greedy"):
            max_depth = 3
            leaves_by_depth = [_collect_leaves(mat, d, method)
                                for d in range(max_depth + 1)]
            max_leaves = max(len(lvs) for lvs in leaves_by_depth)

            fig, axes = plt.subplots(
                max_depth + 1, max_leaves,
                figsize=(max(6, max_leaves * 2.4), (max_depth + 1) * 2.4),
                squeeze=False,
            )
            fig.suptitle(
                f"Leaf tree — {method}  "
                f"(n={n}, k={k}, max_w={max_w}, S={S_orig})\n"
                f"Each level splits S in half; greedy concentrates mass to produce "
                f"sparser leaves",
                fontsize=9,
            )

            for d, leaves in enumerate(leaves_by_depth):
                for col in range(max_leaves):
                    ax = axes[d, col]
                    if col < len(leaves):
                        leaf   = leaves[col]
                        S_leaf = int(leaf.sum(axis=1).max()) if leaf.any() else 0
                        nnz    = int(np.count_nonzero(leaf))
                        _heatmap(ax, leaf,
                                 f"d={d}  leaf {col}\nS={S_leaf}  nnz={nnz}",
                                 0, vmax)
                    else:
                        ax.axis("off")
                # Row label on left-most cell
                axes[d, 0].set_ylabel(f"depth {d}", fontsize=8, labelpad=4)

            # Aggregate nnz info in a text box
            for d, leaves in enumerate(leaves_by_depth):
                total_nnz = sum(int(np.count_nonzero(lf)) for lf in leaves)
                axes[d, 0].set_title(
                    axes[d, 0].get_title() +
                    f"\n[total nnz={total_nnz}]",
                    fontsize=7,
                )

            # Shared colorbar
            fig.subplots_adjust(right=0.88)
            cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
            sm = cm.ScalarMappable(cmap="YlOrRd",
                                   norm=plt.Normalize(vmin=0, vmax=vmax))
            sm.set_array([])
            fig.colorbar(sm, cax=cbar_ax, label="Entry value")

            out = _out_dir() / f"depth_tree_{method}.png"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"\n[visual] saved -> {out}")

    def test_vis_greedy_sparsity_sweep(self):
        """
        Two-panel chart comparing greedy vs Euler sparsity over 20 matrices.

        Top panel  — grouped bars: Euler (blue) and Greedy (orange), fixed colours.
                     A star (*) marks cases where greedy strictly wins.
        Bottom panel — savings bar: euler_nnz - greedy_nnz.
                     Green = greedy wins (fewer non-zeros), grey = tie (greedy
                     fell back to Euler).  Zero savings means the matrix had no
                     entry >= S/2 so greedy could not outperform Euler.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        import time

        # n=32 gives meaningful split runtimes (microsecond range).
        # max_weight=16 ensures entries often exceed S/2, giving greedy real wins.
        N_MATRICES = 20
        rng = np.random.default_rng(99)
        matrices = [
            generate_matrix(n=32, k=3, max_weight=16, rng=rng).astype(np.int64)
            for _ in range(N_MATRICES)
        ]

        # Warmup: one call each so Python/import overhead doesn't skew index 0
        euler_split_once(matrices[0], "euler")
        euler_split_once(matrices[0], "greedy")

        euler_nnzs, greedy_nnzs = [], []
        euler_times_us, greedy_times_us = [], []
        s_values = []
        print()
        for i, mat in enumerate(matrices):
            t0 = time.perf_counter()
            euler_red, _  = euler_split_once(mat, "euler")
            euler_times_us.append((time.perf_counter() - t0) * 1e6)

            t0 = time.perf_counter()
            greedy_red, _ = euler_split_once(mat, "greedy")
            greedy_times_us.append((time.perf_counter() - t0) * 1e6)

            euler_nnzs.append(int(np.count_nonzero(euler_red)))
            greedy_nnzs.append(int(np.count_nonzero(greedy_red)))
            S = int(mat.sum(axis=1).max())
            s_values.append(S)
            print(f"  matrix {i:2d}: S={S:3d}  nnz={np.count_nonzero(mat):3d}"
                  f"  euler={euler_times_us[-1]:6.1f}us"
                  f"  greedy={greedy_times_us[-1]:6.1f}us")

        x      = np.arange(20)
        savings = [e - g for e, g in zip(euler_nnzs, greedy_nnzs)]
        wins    = sum(s > 0 for s in savings)

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(13, 7),
            gridspec_kw={"height_ratios": [3, 1.5]},
        )
        fig.suptitle(
            f"Greedy vs Euler — sparsity (top) and split runtime (bottom)  "
            f"(n=32, k=3, max_weight=16)\n"
            f"Greedy wins on {wins}/20 matrices  |  * = greedy strictly sparser",
            fontsize=10,
        )

        # --- Top panel: grouped bars, fixed colours ---
        ax_top.bar(x - 0.2, euler_nnzs,  width=0.38, label="Euler red  nnz",
                   color="steelblue", alpha=0.85)
        ax_top.bar(x + 0.2, greedy_nnzs, width=0.38, label="Greedy red nnz",
                   color="darkorange", alpha=0.85)
        # Mark greedy wins with a star above the greedy bar
        for i, s in enumerate(savings):
            if s > 0:
                ax_top.text(x[i] + 0.2, greedy_nnzs[i] + 0.3, "*",
                            ha="center", va="bottom", fontsize=14,
                            color="darkgreen", fontweight="bold")
        ax_top.set_ylabel("Non-zeros in red half")
        ax_top.set_xticks(x)
        ax_top.set_xticklabels([])
        ax_top.legend(loc="upper right")
        ax_top.set_ylim(bottom=0)

        # --- Bottom panel: split runtime per matrix ---
        # Cap y-axis at 1.3× the 95th percentile so OS-scheduler outliers
        # (random thread preemptions) don't collapse all real bars to zero.
        all_times = euler_times_us + greedy_times_us
        y_cap = float(np.percentile(all_times, 95)) * 1.3

        ax_bot.bar(x - 0.2, euler_times_us,  width=0.38, label="Euler time",
                   color="steelblue", alpha=0.85)
        ax_bot.bar(x + 0.2, greedy_times_us, width=0.38, label="Greedy time",
                   color="darkorange", alpha=0.85)

        # Annotate any bar that exceeds the cap with its actual value
        for i, (et, gt) in enumerate(zip(euler_times_us, greedy_times_us)):
            if et > y_cap:
                ax_bot.text(x[i] - 0.2, y_cap * 0.97, f"{et/1000:.0f}ms",
                            ha="center", va="top", fontsize=6, color="steelblue",
                            rotation=90)
            if gt > y_cap:
                ax_bot.text(x[i] + 0.2, y_cap * 0.97, f"{gt/1000:.0f}ms",
                            ha="center", va="top", fontsize=6, color="darkorange",
                            rotation=90)

        ax_bot.set_ylabel("Split time (us)")
        ax_bot.set_xlabel("Matrix index  (S = row sum)")
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([f"{i}\nS={s_values[i]}" for i in x], fontsize=7)
        ax_bot.set_ylim(0, y_cap)
        ax_bot.legend(loc="upper right")

        plt.tight_layout()
        out = _out_dir() / "greedy_vs_euler_nnz.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[visual] saved -> {out}  (greedy won {wins}/20 cases)")

    # ------------------------------------------------------------------
    # Heuristic visual tests (Euler vs Heuristic only)
    # ------------------------------------------------------------------

    def test_vis_euler_vs_heuristic(self):
        """
        Two-way heatmap: Euler (baseline) vs Heuristic split side by side.

        Layout (2 rows × 3 cols):
          Row 0: original | euler-red   | euler-blue
          Row 1: original | heuristic-red | heuristic-blue

        Annotates each sub-matrix with nnz and how many (i,j) pairs are
        "clean" (entirely in one half) vs "split" (appear in both halves).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        mat = _make_matrix(n=8, k=4, max_weight=8, seed=5).astype(np.int64)
        S    = int(mat.sum(axis=1).max())
        vmax = int(mat.max())

        METHODS = ("euler", "heuristic")
        COLOURS = {"euler": "steelblue", "heuristic": "seagreen"}

        fig, axes = plt.subplots(2, 3, figsize=(11, 8))
        fig.suptitle(
            f"Euler vs Heuristic split  (n=8, k=4, max_w=8, S={S})\n"
            "Heuristic aims to keep all M[i,j] copies in the same half",
            fontsize=11,
        )

        for row, method in enumerate(METHODS):
            red, blue = euler_split_once(mat, split_method=method)
            clean = int(np.sum((red > 0) ^ (blue > 0)))
            split = int(np.sum((red > 0) & (blue > 0)))

            _heatmap(axes[row, 0], mat,
                     f"Original  S={S}  nnz={np.count_nonzero(mat)}", 0, vmax)
            _heatmap(axes[row, 1], red,
                     f"{method} RED   nnz={np.count_nonzero(red)}", 0, vmax)
            _heatmap(axes[row, 2], blue,
                     f"{method} BLUE  nnz={np.count_nonzero(blue)}", 0, vmax)
            axes[row, 0].set_ylabel(
                f"{method}\nclean={clean}  split={split}",
                fontsize=10, fontweight="bold", color=COLOURS[method],
            )

        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
        sm = cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Entry value")

        out = _out_dir() / "euler_vs_heuristic_heatmap.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[visual] saved -> {out}")

    def test_vis_heuristic_sparsity_sweep(self):
        """
        Euler vs Heuristic sparsity and runtime comparison over 30 matrices.

        Top panel    — grouped bars: Euler (blue) vs Heuristic (green).
                       Stars mark matrices where heuristic strictly wins.
        Bottom panel — nnz saved by heuristic vs Euler baseline.
                       Green = heuristic wins, grey = tie or Euler wins.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import time

        N = 30
        rng = np.random.default_rng(77)
        matrices = [
            generate_matrix(n=16, k=4, max_weight=6, rng=rng).astype(np.int64)
            for _ in range(N)
        ]

        euler_split_once(matrices[0], "euler")
        euler_split_once(matrices[0], "heuristic")

        euler_nnz, heuristic_nnz = [], []
        euler_t, heuristic_t = [], []
        s_values = []

        print()
        for i, mat in enumerate(matrices):
            S = int(mat.sum(axis=1).max())
            s_values.append(S)

            t0 = time.perf_counter()
            r, _ = euler_split_once(mat, "euler")
            euler_t.append((time.perf_counter() - t0) * 1e6)
            euler_nnz.append(int(np.count_nonzero(r)))

            t0 = time.perf_counter()
            r, _ = euler_split_once(mat, "heuristic")
            heuristic_t.append((time.perf_counter() - t0) * 1e6)
            heuristic_nnz.append(int(np.count_nonzero(r)))

            print(f"  mat {i:2d}: S={S:3d}  euler={euler_nnz[-1]:3d}  "
                  f"heuristic={heuristic_nnz[-1]:3d}  t_h={heuristic_t[-1]:5.1f}us")

        x = np.arange(N)
        savings = [euler_nnz[i] - heuristic_nnz[i] for i in range(N)]
        h_wins  = sum(s > 0 for s in savings)

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(15, 7),
            gridspec_kw={"height_ratios": [3, 1.5]},
        )
        fig.suptitle(
            f"Euler vs Heuristic — n=16, k=4, max_w=6, {N} matrices\n"
            f"Heuristic strictly sparser on {h_wins}/{N}  |  * = heuristic wins",
            fontsize=10,
        )

        ax_top.bar(x - 0.2, euler_nnz,     width=0.38, label="Euler",     color="steelblue", alpha=0.85)
        ax_top.bar(x + 0.2, heuristic_nnz, width=0.38, label="Heuristic", color="seagreen",  alpha=0.85)
        for i in range(N):
            if savings[i] > 0:
                ax_top.text(x[i] + 0.2, heuristic_nnz[i] + 0.2, "*",
                            ha="center", va="bottom", fontsize=13,
                            color="darkgreen", fontweight="bold")
        ax_top.set_ylabel("Non-zeros in red half")
        ax_top.set_xticks(x)
        ax_top.set_xticklabels([])
        ax_top.legend(loc="upper right")
        ax_top.set_ylim(bottom=0)

        bar_colors = ["seagreen" if s > 0 else "lightgray" for s in savings]
        ax_bot.bar(x, savings, color=bar_colors, alpha=0.9, label="nnz saved by heuristic")
        ax_bot.axhline(0, color="black", linewidth=0.8)
        ax_bot.set_ylabel("nnz saved vs Euler")
        ax_bot.set_xlabel("Matrix index")
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([f"{i}\nS={s_values[i]}" for i in x], fontsize=6)

        plt.tight_layout()
        out = _out_dir() / "heuristic_sparsity_sweep.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[visual] saved -> {out}  (heuristic won {h_wins}/{N})")

    def test_vis_bipartite_graph(self):
        """
        Bipartite multigraph visualisation — one large figure per matrix.

        Saves to tests/visual_output/bipartite_graphs/ (folder is cleaned first).
        Each file: one matrix, left panel = Euler split, right panel = Heuristic.

        GRAPH LAYOUT
        ============
        Row nodes (blue circles, left side):   r0 … r{n-1}
        Col nodes (green circles, right side): c0 … c{n-1}
        Each (i,j) pair is a curved arc.  Arc width ∝ M[i,j].

        EDGE COLOURS
        ============
          Red    — all copies went to the Red sub-matrix (row→col direction)
          Blue   — all copies went to the Blue sub-matrix (col→row direction)
          Purple — pair split: some Red, some Blue  (format: "rR + bB")

        ARC CURVATURE
        =============
        Each (i,j) pair gets a unique curvature so overlapping arcs can be
        distinguished.  Curvature sign alternates with (i+j) parity; magnitude
        scales with the row-distance |i-j| so far-apart pairs curve more.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import shutil

        # ── Clean / create output subfolder ─────────────────────────────
        graph_dir = _out_dir() / "bipartite_graphs"
        if graph_dir.exists():
            shutil.rmtree(graph_dir)
        graph_dir.mkdir()

        # ── Graph drawing helper ─────────────────────────────────────────
        def _draw_graph(ax, matrix, red, blue, title, n):
            row_ys = np.linspace(n - 1, 0, n)
            col_ys = np.linspace(n - 1, 0, n)

            C_RED   = "#CC2222"
            C_BLUE  = "#1144BB"
            C_SPLIT = "#882299"

            for i in range(n):
                for j in range(n):
                    m = int(matrix[i, j])
                    if m == 0:
                        continue
                    r = int(red[i, j])
                    b = int(blue[i, j])

                    if r > 0 and b == 0:
                        ec  = C_RED
                        lbl = str(r)
                    elif b > 0 and r == 0:
                        ec  = C_BLUE
                        lbl = str(b)
                    else:
                        ec  = C_SPLIT
                        lbl = f"{r}R+{b}B"   # e.g. "2R+4B" — clear breakdown

                    lw = 1.5 + m ** 0.65

                    # Unique curvature per (i,j): sign alternates, magnitude
                    # grows with row-distance so far-apart arcs spread out more
                    sign = 1 if (i + j) % 2 == 0 else -1
                    dist = abs(i - j) / max(n - 1, 1)
                    rad  = sign * (0.12 + 0.22 * dist)

                    ax.annotate(
                        "", xy=(1.0, col_ys[j]), xytext=(0.0, row_ys[i]),
                        arrowprops=dict(
                            arrowstyle="-",
                            color=ec, lw=lw, alpha=0.80,
                            connectionstyle=f"arc3,rad={rad:.3f}",
                        ),
                        zorder=2,
                    )

                    # Label at arc midpoint (shifted perpendicular to the arc)
                    mx = 0.5
                    my = (row_ys[i] + col_ys[j]) / 2 + rad * 0.65
                    ax.text(mx, my, lbl,
                            ha="center", va="center",
                            fontsize=9, color=ec, fontweight="bold", zorder=4,
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", ec=ec, linewidth=0.6,
                                      alpha=0.92))

            # Row nodes (blue circles)
            for i in range(n):
                ax.plot(0, row_ys[i], "o", color="#4472C4", ms=26, zorder=6)
                ax.text(0, row_ys[i], f"r{i}",
                        ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold", zorder=7)

            # Col nodes (green circles)
            for j in range(n):
                ax.plot(1, col_ys[j], "o", color="#548235", ms=26, zorder=6)
                ax.text(1, col_ys[j], f"c{j}",
                        ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold", zorder=7)

            clean  = int(np.sum((red > 0) ^ (blue > 0)))
            nsplit = int(np.sum((red > 0) & (blue > 0)))
            nnz    = int(np.count_nonzero(red))
            ax.set_xlim(-0.28, 1.28)
            ax.set_ylim(-0.7, n - 0.3)
            ax.set_title(
                f"{title}\nnnz={nnz}   clean pairs={clean}   split pairs={nsplit}",
                fontsize=11, pad=10,
            )
            ax.axis("off")

        # ── One figure per matrix ────────────────────────────────────────
        # Layout per figure (wide):
        #
        #   ┌─────────────────────┬──────────┐   ┌─────────────────────┬──────────┐
        #   │                     │ Original │   │                     │ Original │
        #   │   Euler graph       │──────────│   │   Heuristic graph   │──────────│
        #   │   (bipartite arcs)  │ Red half │   │   (bipartite arcs)  │ Red half │
        #   │                     │──────────│   │                     │──────────│
        #   │                     │Blue half │   │                     │Blue half │
        #   └─────────────────────┴──────────┘   └─────────────────────┴──────────┘
        #        graph (3/4 width)  maps (1/4)          graph (3/4 width) maps(1/4)
        #
        # The heatmaps on the right of each panel show exactly what the bipartite
        # graph is encoding: Original M → how it was split into Red and Blue.

        SEEDS  = [0, 1, 5, 7]
        n, k, max_w = 6, 3, 6

        legend_handles = [
            mpatches.Patch(color="#CC2222", label="Red half only  (all copies row→col)"),
            mpatches.Patch(color="#1144BB", label="Blue half only  (all copies col→row)"),
            mpatches.Patch(color="#882299", label="Split  (format: rR+bB)"),
        ]

        for seed in SEEDS:
            mat = _make_matrix(n=n, k=k, seed=seed, max_weight=max_w).astype(np.int64)
            S    = int(mat.sum(axis=1).max())
            vmax = int(mat.max())

            r_e, b_e = euler_split_once(mat, "euler")
            r_h, b_h = euler_split_once(mat, "heuristic")

            # ── GridSpec: 2 main sections (Euler | Heuristic), each with
            #    a wide graph column and a narrow heatmap column (3 stacked).
            #    Total cols: graph_e, maps_e, graph_h, maps_h  with ratios 3:1:3:1
            fig = plt.figure(figsize=(26, 12))
            fig.suptitle(
                f"Bipartite graph + split heatmaps — seed={seed}  "
                f"n={n}  k={k}  max_w={max_w}  S={S}\n"
                "Red = Red half only   |   Blue = Blue half only   "
                "|   Purple = split across both (rR+bB)",
                fontsize=12,
            )

            import matplotlib.gridspec as gridspec
            gs = gridspec.GridSpec(
                3, 4,
                width_ratios=[3, 1, 3, 1],
                hspace=0.35, wspace=0.08,
                left=0.03, right=0.97, top=0.88, bottom=0.09,
            )

            ax_eg  = fig.add_subplot(gs[:, 0])   # Euler graph  (spans all 3 rows)
            ax_eo  = fig.add_subplot(gs[0, 1])   # Euler — original heatmap
            ax_er  = fig.add_subplot(gs[1, 1])   # Euler — Red heatmap
            ax_eb  = fig.add_subplot(gs[2, 1])   # Euler — Blue heatmap

            ax_hg  = fig.add_subplot(gs[:, 2])   # Heuristic graph
            ax_ho  = fig.add_subplot(gs[0, 3])   # Heuristic — original
            ax_hr  = fig.add_subplot(gs[1, 3])   # Heuristic — Red
            ax_hb  = fig.add_subplot(gs[2, 3])   # Heuristic — Blue

            # Bipartite graphs
            _draw_graph(ax_eg, mat, r_e, b_e, "Euler  (baseline)", n)
            _draw_graph(ax_hg, mat, r_h, b_h, "Heuristic  (same-direction)", n)

            # Heatmaps alongside each graph
            for ax_o, ax_r, ax_b, red, blue, tag in [
                (ax_eo, ax_er, ax_eb, r_e, b_e, "Euler"),
                (ax_ho, ax_hr, ax_hb, r_h, b_h, "Heuristic"),
            ]:
                _heatmap(ax_o, mat,  f"Original  nnz={np.count_nonzero(mat)}", 0, vmax)
                _heatmap(ax_r, red,  f"{tag} Red  nnz={np.count_nonzero(red)}", 0, vmax)
                _heatmap(ax_b, blue, f"{tag} Blue nnz={np.count_nonzero(blue)}", 0, vmax)

            fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                       fontsize=10, framealpha=0.95,
                       bbox_to_anchor=(0.5, 0.01))

            out = graph_dir / f"bipartite_seed{seed:02d}.png"
            plt.savefig(out, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"\n[visual] saved -> {out}")

    def test_vis_realistic_paper_params(self):
        """
        Heuristic split on the actual paper parameters: n=256, k=256, max_weight=64.

        At this scale the bipartite graph has 512 nodes and is unreadable, so we
        show only the sparsity heatmaps.  Two figures are saved:

          realistic_full_matrix.png  — the full n=256 matrix split
          realistic_digit_plane.png  — one B=4 digit plane extracted from it

        Layout for each figure (2 rows × 3 cols):
          Row 0  Euler:      Original  |  Red half   |  Blue half
          Row 1  Heuristic:  Original  |  Red half   |  Blue half

        Titles include nnz, clean-pair count, and split-pair count so the
        sparsity improvement of the heuristic is numerically visible even
        when the matrix is too dense to see individual cells.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import time

        graph_dir = _out_dir() / "bipartite_graphs"
        graph_dir.mkdir(exist_ok=True)

        rng = np.random.default_rng(42)
        M   = generate_matrix(n=256, k=256, max_weight=64, rng=rng).astype(np.int64)
        S   = int(M.sum(axis=1).max())
        print(f"\n  Paper params: n=256 k=256 max_w=64  S={S}  nnz={np.count_nonzero(M)}")

        def _split_stats(r, b, tag):
            clean  = int(np.sum((r > 0) ^ (b > 0)))
            nsplit = int(np.sum((r > 0) & (b > 0)))
            return (f"{tag}  nnz_red={np.count_nonzero(r)}"
                    f"  clean={clean}  split={nsplit}")

        def _make_fig(mat, label):
            vmax = int(mat.max())
            S_   = int(mat.sum(axis=1).max())
            nnz0 = np.count_nonzero(mat)

            t0 = time.perf_counter()
            r_e, b_e = euler_split_once(mat, "euler")
            t_e = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            r_h, b_h = euler_split_once(mat, "heuristic")
            t_h = (time.perf_counter() - t0) * 1000

            print(f"  {label}: euler {t_e:.1f}ms  heuristic {t_h:.1f}ms")
            print(f"    Euler:     {_split_stats(r_e, b_e, '')}")
            print(f"    Heuristic: {_split_stats(r_h, b_h, '')}")

            fig, axes = plt.subplots(2, 3, figsize=(14, 10))
            fig.suptitle(
                f"{label}  (n={mat.shape[0]}, S={S_}, nnz={nnz0})\n"
                f"Euler:     {_split_stats(r_e, b_e, '')}\n"
                f"Heuristic: {_split_stats(r_h, b_h, '')}",
                fontsize=9, y=1.01,
            )

            for row, (red, blue, tag) in enumerate([
                (r_e, b_e, "Euler"),
                (r_h, b_h, "Heuristic"),
            ]):
                _heatmap(axes[row, 0], mat,
                         f"Original  S={S_}  nnz={nnz0}", 0, vmax)
                _heatmap(axes[row, 1], red,
                         f"{tag} RED   nnz={np.count_nonzero(red)}", 0, vmax)
                _heatmap(axes[row, 2], blue,
                         f"{tag} BLUE  nnz={np.count_nonzero(blue)}", 0, vmax)
                axes[row, 0].set_ylabel(tag, fontsize=10, fontweight="bold",
                                        color="steelblue" if tag == "Euler" else "seagreen")

            fig.subplots_adjust(right=0.87, hspace=0.3, wspace=0.1)
            cbar_ax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
            sm = cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=0, vmax=vmax))
            sm.set_array([])
            fig.colorbar(sm, cax=cbar_ax, label="Entry value")
            return fig

        # ── Figure 1: full n=256 matrix ──────────────────────────────────
        fig = _make_fig(M, "Full matrix  n=256 k=256 max_w=64")
        out = graph_dir / "realistic_full_matrix.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[visual] saved -> {out}")

        # ── Figure 2: B=4 digit plane d=0 from the same matrix ───────────
        M_int   = np.round(M).astype(np.int64)
        plane   = ((M_int // 4**0) % 4).astype(np.float64)
        plane_i = plane.astype(np.int64)
        fig = _make_fig(plane_i, "Digit plane  B=4  d=0  (entries in [0,3])")
        out = graph_dir / "realistic_digit_plane.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[visual] saved -> {out}")

    def test_vis_assignment_map(self):
        """
        Pair-assignment colour map for n=256 paper parameters.

        Each cell (i,j) is coloured by where M[i,j] ended up after the split:
          White  — M[i,j] = 0  (no entry)
          Red    — all copies went to the Red half   (clean)
          Blue   — all copies went to the Blue half  (clean)
          Purple — copies split across BOTH halves   (not clean)

        Layout: 2 rows × 2 cols
          Row 0: Full matrix    — Euler | Heuristic
          Row 1: Digit plane B=4 d=0 — Euler | Heuristic

        Expected result: Euler rows are almost entirely purple.
                         Heuristic rows are almost entirely red and blue.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        graph_dir = _out_dir() / "bipartite_graphs"
        graph_dir.mkdir(exist_ok=True)

        def _colour_map(matrix, red, blue):
            """Build an (n, n, 3) RGB image from the split assignment."""
            n   = matrix.shape[0]
            img = np.ones((n, n, 3), dtype=np.float32)  # white background

            clean_red  = (red  > 0) & (blue == 0)
            clean_blue = (red == 0) & (blue  > 0)
            split_both = (red  > 0) & (blue  > 0)

            img[clean_red]  = [0.85, 0.08, 0.08]   # vivid red
            img[clean_blue] = [0.08, 0.22, 0.88]   # vivid blue
            img[split_both] = [0.58, 0.08, 0.62]   # purple

            n_clean = int(clean_red.sum() + clean_blue.sum())
            n_split = int(split_both.sum())
            return img, n_clean, n_split

        def _panel(ax, img, n_clean, n_split, title):
            ax.imshow(img, interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            total = n_clean + n_split
            pct   = 100.0 * n_clean / max(total, 1)
            ax.set_title(
                f"{title}\n"
                f"clean={n_clean:,} ({pct:.1f}%)   split={n_split:,}",
                fontsize=9,
            )

        # ── Generate 3 doubly stochastic matrices (different seeds) ─────
        SEEDS = [42, 7, 123]
        rng_list = [np.random.default_rng(s) for s in SEEDS]
        matrices = [
            generate_matrix(n=256, k=256, max_weight=64, rng=r).astype(np.int64)
            for r in rng_list
        ]

        # ── One figure: 3 rows × 2 cols (one row per matrix) ────────────
        fig, axes = plt.subplots(len(SEEDS), 2,
                                 figsize=(14, 7 * len(SEEDS)))
        fig.suptitle(
            "Pair-assignment colour map  —  k-regular doubly stochastic matrix  "
            "(n=256, k=256, max_w=64)\n"
            "Red = clean Red half   |   Blue = clean Blue half   "
            "|   Purple = split across both   |   White = zero",
            fontsize=11,
        )

        for row, (M, seed) in enumerate(zip(matrices, SEEDS)):
            S = int(M.sum(axis=1).max())

            r_e, b_e = euler_split_once(M, "euler")
            r_h, b_h = euler_split_once(M, "heuristic")

            img_e, nc_e, ns_e = _colour_map(M, r_e, b_e)
            img_h, nc_h, ns_h = _colour_map(M, r_h, b_h)

            _panel(axes[row, 0], img_e, nc_e, ns_e,
                   f"Euler  (seed={seed}, S={S})")
            _panel(axes[row, 1], img_h, nc_h, ns_h,
                   f"Heuristic  (seed={seed}, S={S})")

        legend_handles = [
            mpatches.Patch(color="#D91414", label="Clean — Red half only"),
            mpatches.Patch(color="#1438E0", label="Clean — Blue half only"),
            mpatches.Patch(color="#940F9E", label="Split across both halves"),
            mpatches.Patch(facecolor="white", edgecolor="lightgray",
                           label="Zero entry"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                   fontsize=9, framealpha=0.95,
                   bbox_to_anchor=(0.5, 0.005))
        fig.subplots_adjust(bottom=0.04, hspace=0.18, wspace=0.06)

        out = graph_dir / "assignment_map.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[visual] saved -> {out}")

    def test_vis_heuristic_depth_tree(self):
        """
        Depth tree showing how sparsity evolves across recursive splits.
        Compares Euler vs Heuristic at depths 0–3 on the same matrix.

        Layout: two figures (one per method), each with:
          Rows = depth levels  (0 = original, 1 = 2 leaves, 2 = 4 leaves, ...)
          Cols = individual leaf matrices
          Each cell annotated with S_leaf and nnz.

        At the bottom of each depth row the total nnz across all leaves is shown
        so you can see whether the heuristic produces cumulatively sparser trees.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        n, k, max_w, seed = 8, 4, 8, 42
        mat = _make_matrix(n=n, k=k, seed=seed, max_weight=max_w).astype(np.int64)
        S_orig = int(mat.sum(axis=1).max())
        vmax   = int(mat.max())

        METHODS = ("euler", "heuristic")
        COLOURS = {"euler": "steelblue", "heuristic": "seagreen"}

        for method in METHODS:
            max_depth = 3
            leaves_by_depth = [_collect_leaves(mat, d, method)
                                for d in range(max_depth + 1)]
            max_leaves = max(len(lvs) for lvs in leaves_by_depth)

            fig, axes = plt.subplots(
                max_depth + 1, max_leaves,
                figsize=(max(6, max_leaves * 2.5), (max_depth + 1) * 2.5),
                squeeze=False,
            )
            fig.suptitle(
                f"Recursive split depth tree — {method}  "
                f"(n={n}, k={k}, max_w={max_w}, S={S_orig})\n"
                f"S halves at each level; heuristic targets sparser leaves",
                fontsize=9, color=COLOURS[method],
            )

            for d, leaves in enumerate(leaves_by_depth):
                total_nnz = sum(int(np.count_nonzero(lf)) for lf in leaves)
                for col in range(max_leaves):
                    ax = axes[d, col]
                    if col < len(leaves):
                        leaf   = leaves[col]
                        S_leaf = int(leaf.sum(axis=1).max()) if leaf.any() else 0
                        nnz    = int(np.count_nonzero(leaf))
                        _heatmap(ax, leaf,
                                 f"d={d} leaf {col}\nS={S_leaf}  nnz={nnz}",
                                 0, vmax)
                    else:
                        ax.axis("off")
                axes[d, 0].set_ylabel(f"depth {d}\ntotal nnz={total_nnz}",
                                      fontsize=8, labelpad=4)

            fig.subplots_adjust(right=0.88)
            cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
            sm = cm.ScalarMappable(cmap="YlOrRd",
                                   norm=plt.Normalize(vmin=0, vmax=vmax))
            sm.set_array([])
            fig.colorbar(sm, cax=cbar_ax, label="Entry value")

            out = _out_dir() / f"heuristic_depth_tree_{method}.png"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"\n[visual] saved -> {out}")
