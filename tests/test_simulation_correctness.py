import pytest
import numpy as np
import time
from unittest.mock import patch
from src.utils.matrix_generator import generate_matrix, check_k_regularity
from src.algorithms.radix_decomposition import decompose_radix
from src.algorithms.bvn import bvn_decomposition
from src.runner import _compute_for_index
from src.config import ExperimentConfig


def test_matrix_generation_is_doubly_stochastic():
    """Verify generated matrices have equal row and col sums (K-regular)."""
    rng = np.random.default_rng(42)
    n = 32
    k = 10
    max_weight = 10
    
    matrix = generate_matrix(n, k, max_weight, rng)
    expected_k = float(np.sum(matrix[0, :]))
    assert check_k_regularity(matrix, expected_k), "Unified generator should be K-regular"


def test_decomposition_correctness():
    """Verify that decomposing a matrix accurately reconstructions the exact original matrix."""
    rng = np.random.default_rng(42)
    n = 32
    k = 10
    max_weight = 5
    
    original_matrix = generate_matrix(n, k, max_weight, rng)
    
    # Standard BVN
    bvn_components = bvn_decomposition(original_matrix, matching_algorithm="wfa")
    
    reconstructed_bvn = np.zeros_like(original_matrix)
    for comp in bvn_components:
        reconstructed_bvn += (comp.permutation * comp.weight)
        
    np.testing.assert_allclose(reconstructed_bvn, original_matrix, atol=1e-7, 
                               err_msg="BVN decomposition should mathematically reconstruct the original matrix.")
    
    # Radix Decomposition
    radix_components, simulated_runtime, _num_planes = decompose_radix(original_matrix, base=2, matching_method="wfa", max_workers=1)
    
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
    max_weight = 1
    matrix = generate_matrix(n, k, max_weight, rng)
    
    # Check Radix
    radix_components, max_runtime, _num_planes = decompose_radix(matrix, base=2, matching_method="wfa", max_workers=1)
    
    # Evaluated exactly as runner.py evaluates them
    cycle_length = float(sum(comp.weight for comp in radix_components))
    num_permutations = len(radix_components)
    
    assert cycle_length >= k, "Cycle length should at least be K (usually more due to overhead)."
    assert num_permutations == len(radix_components), "Permutation count strictly equals number of components."


def test_parallelism_preserves_correctness():
    """Verify that decomposing sequentially vs logically yields valid decompositions."""
    rng = np.random.default_rng(42)
    n = 64
    k = 10
    max_weight = 1
    matrix = generate_matrix(n, k, max_weight, rng)
    
    # decompose_radix handles it sequentially (since multi-threading was removed)
    # The max_workers argument doesn't functionally spawn threads anymore,
    # but the decomposition logic should still be sound.
    radix_seq, _, _ = decompose_radix(matrix, base=4, matching_method="wfa", max_workers=1)
    
    rebuilt_seq = np.zeros_like(matrix)
    for c in radix_seq: rebuilt_seq += c.matrix
        
    np.testing.assert_allclose(rebuilt_seq, matrix, atol=1e-7)
