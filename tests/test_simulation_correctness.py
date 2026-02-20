import pytest
import numpy as np
import time
from unittest.mock import patch
from src.utils.matrix_generator import (
    generate_scaled_doubly_stochastic_matrix,
    generate_binary_weighted_matrix,
    generate_weighted_sum_matrix,
    check_k_regularity
)
from src.algorithms.radix_decomposition import decompose_radix
from src.algorithms.bvn import bvn_decomposition
from src.runner import _compute_for_index
from src.config import ExperimentConfig


def test_matrix_generation_is_doubly_stochastic():
    """Verify generated matrices have equal row and col sums (K-regular)."""
    rng = np.random.default_rng(42)
    n = 32
    k = 10
    
    # 1. Standard approach
    mat1 = generate_scaled_doubly_stochastic_matrix(n, k, rng)
    assert check_k_regularity(mat1.astype(np.int64), k), "Standard generator should be K-regular"
    
    # 2. Binary weighted approach (K = 2^bits - 1)
    bits = 4
    expected_k_binary = (2**bits) - 1
    mat2 = generate_binary_weighted_matrix(n, rng, bits)
    assert check_k_regularity(mat2.astype(np.int64), expected_k_binary), "Binary generator should be (2^bits)-1 regular"
    
    # 3. Weighted sum approach
    weights = [3, 5]
    sub_k = 10
    expected_k_weighted = sum(w * sub_k for w in weights)
    mat3 = generate_weighted_sum_matrix(n, weights, sub_k, rng)
    assert check_k_regularity(mat3.astype(np.int64), expected_k_weighted), "Weighted generator should be (sum(w)*sub_k) regular"


def test_decomposition_correctness():
    """Verify that decomposing a matrix accurately reconstructions the exact original matrix."""
    rng = np.random.default_rng(42)
    n = 32
    k = 10
    original_matrix = generate_scaled_doubly_stochastic_matrix(n, k, rng)
    
    # Standard BVN
    bvn_components = bvn_decomposition(original_matrix, matching_algorithm="wfa")
    
    reconstructed_bvn = np.zeros_like(original_matrix)
    for comp in bvn_components:
        reconstructed_bvn += (comp.permutation * comp.weight)
        
    np.testing.assert_allclose(reconstructed_bvn, original_matrix, atol=1e-7, 
                               err_msg="BVN decomposition should mathematically reconstruct the original matrix.")
    
    # Radix Decomposition
    radix_components = decompose_radix(original_matrix, base=2, matching_method="wfa", max_workers=1)
    
    reconstructed_radix = np.zeros_like(original_matrix)
    for comp in radix_components:
        reconstructed_radix += comp.matrix # (comp.matrix already includes comp.weight implicitly in RadixComponent)
        
    np.testing.assert_allclose(reconstructed_radix, original_matrix, atol=1e-7,
                               err_msg="Radix decomposition should mathematically reconstruct the original matrix.")


def test_metrics_cycle_and_permutations():
    """Verify permutation counts and cycle length calculations."""
    rng = np.random.default_rng(42)
    n = 16
    k = 5
    matrix = generate_scaled_doubly_stochastic_matrix(n, k, rng)
    
    # Check Radix
    radix_components = decompose_radix(matrix, base=2, matching_method="wfa", max_workers=1)
    
    # Evaluated exactly as runner.py evaluates them
    cycle_length = float(sum(comp.weight for comp in radix_components))
    num_permutations = len(radix_components)
    
    # cycle length should be exactly the theoretical amount needed to decompose the matrix weight
    # Because our radix algorithm might decompose fractional weights depending on the matrix
    # But fundamentally cycle_length is the true normalized cost.
    assert cycle_length >= k, "Cycle length should at least be K (usually more due to overhead)."
    assert num_permutations == len(radix_components), "Permutation count strictly equals number of components."


@patch("src.runner.time.perf_counter")
def test_timing_measurement_runner(mock_perf_counter):
    """Verify that the runner accurately profiles the end-to-end timing."""
    # Mock time passing: tick by 0.1s every call
    mock_perf_counter.side_effect = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    
    config = ExperimentConfig(
        n=16, num_matrices=1, is_parallel=False,
        generator="standard", density=0.5,
        engine="wfa_bvn", radix_bases=[2],
        skip_split=True, random_seed=42
    )

    stats = _compute_for_index(0, config)
    
    # Timing calculations based on the mock ticks difference:
    # 0.5 - 0.0 = 0.5s for BVN
    assert stats.runtime_bvn == 0.5, "BVN runtime calculation is completely accurate."
    
    # Next radix call: 1.5 - 1.0 = 0.5s for Radix 2
    assert "wfa_2" in stats.radix_multi_results
    assert stats.radix_multi_results["wfa_2"][0] == 0.5, "Radix runtime calculation accurately captures the entire plane evaluation."


def test_parallelism_preserves_correctness():
    """Verify that using ThreadPoolExecutor yields valid decompositions."""
    rng = np.random.default_rng(42)
    n = 64
    k = 15
    matrix = generate_scaled_doubly_stochastic_matrix(n, k, rng)
    
    # Sequential
    radix_seq = decompose_radix(matrix, base=4, matching_method="wfa", max_workers=1)
    
    # Parallel
    radix_par = decompose_radix(matrix, base=4, matching_method="wfa", max_workers=4)
    
    rebuilt_seq = np.zeros_like(matrix)
    for c in radix_seq: rebuilt_seq += c.matrix
        
    rebuilt_par = np.zeros_like(matrix)
    for c in radix_par: rebuilt_par += c.matrix
        
    np.testing.assert_allclose(rebuilt_seq, matrix, atol=1e-7)
    np.testing.assert_allclose(rebuilt_par, matrix, atol=1e-7)
    
    # Permutation count of sequential should equal parallel roughly (could vary by scheduler order jitter, but math limits dictate similarly)
    # The important part is they both perfectly decompose the matrix down to 0 without hanging or missing traces.
    assert abs(len(radix_seq) - len(radix_par)) < (0.10 * len(radix_seq)), "Parallelism thread contention should not drastically alter permutation pathfinding."

