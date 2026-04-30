import numpy as np
import pytest
from src.utils.matrix_generator import generate_matrix, check_k_regularity

def test_generate_matrix_regularity():
    n = 20
    k = 30
    max_weight = 10
    rng = np.random.default_rng(42)
    
    matrix = generate_matrix(n, k, max_weight, rng)
    
    # Since we select exactly k permutations, and each has a specific weight,
    # the sum of all columns per row is the theoretical exact sum of the randomly chosen weights.
    expected_sum = float(np.sum(matrix[0, :]))
    
    # check_k_regularity verify ALL rows and columns sum to expected_sum
    assert check_k_regularity(matrix, expected_sum)
    assert np.all(matrix >= 0)

def test_generate_matrix_reproducibility():
    n = 10
    k = 15
    max_weight = 10
    
    rng1 = np.random.default_rng(123)
    m1 = generate_matrix(n, k, max_weight, rng1)
    
    rng2 = np.random.default_rng(123)
    m2 = generate_matrix(n, k, max_weight, rng2)
    
    np.testing.assert_array_equal(m1, m2)

def test_generate_matrix_default_weights():
    n = 10
    k = 5
    rng = np.random.default_rng(42)
    
    # If we pass max_weight 0, it defaults to weight of 0 but bounded to 0.
    matrix = generate_matrix(n, k, 0, rng)
    assert check_k_regularity(matrix, 0)

def test_generate_matrix_float_weights():
    n = 10
    k = 5
    max_weight = 10.0
    rng = np.random.default_rng(42)
    
    matrix = generate_matrix(n, k, max_weight, rng, float_weights=True)
    
    expected_sum = float(np.sum(matrix[0, :]))
    assert check_k_regularity(matrix, expected_sum)
    assert np.all(matrix >= 0)
    assert matrix.dtype == np.float64
    
    # Check that there are actually non-integer values
    assert not np.all(matrix == matrix.astype(np.int64))
