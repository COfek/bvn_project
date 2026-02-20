
import numpy as np
import pytest
from src.utils.matrix_generator import generate_weighted_sum_matrix, check_k_regularity

def test_generate_weighted_sum_matrix_regularity():
    n = 20
    weights = [1, 16]
    sub_k = 3 # Each layer has sum 3
    rng = np.random.default_rng(42)
    
    matrix = generate_weighted_sum_matrix(n, weights, sub_k, rng)
    
    # Expected total sum = 1*3 + 16*3 = 51
    expected_k = 1*3 + 16*3
    
    assert check_k_regularity(matrix.astype(np.int64), expected_k)
    assert np.all(matrix >= 0)

def test_generate_weighted_sum_matrix_list_sub_k():
    n = 20
    weights = [1, 8]
    sub_k = [2, 5] # Layer 0 sum 2, Layer 1 sum 5
    rng = np.random.default_rng(42)
    
    matrix = generate_weighted_sum_matrix(n, weights, sub_k, rng)
    
    expected_k = 1*2 + 8*5 # 2 + 40 = 42
    
    assert check_k_regularity(matrix.astype(np.int64), expected_k)

def test_generate_weighted_sum_matrix_reproducibility():
    n = 10
    weights = [1, 10]
    sub_k = 2
    
    rng1 = np.random.default_rng(123)
    m1 = generate_weighted_sum_matrix(n, weights, sub_k, rng1)
    
    rng2 = np.random.default_rng(123)
    m2 = generate_weighted_sum_matrix(n, weights, sub_k, rng2)
    
    np.testing.assert_array_equal(m1, m2)

def test_generate_weighted_sum_matrix_bad_input():
    n = 10
    weights = [1, 2]
    sub_k = [1, 2, 3] # Mismatch length
    rng = np.random.default_rng(1)
    
    with pytest.raises(ValueError, match="Length of sub_k"):
        generate_weighted_sum_matrix(n, weights, sub_k, rng)

if __name__ == "__main__":
    # verification
    rng = np.random.default_rng(42)
    m = generate_weighted_sum_matrix(10, [1, 16], 3, rng)
    print("Generated 10x10 matrix with weights=[1, 16], sub_k=3")
    print(m)
    print("Row sums:", m.sum(axis=1))
    print("Col sums:", m.sum(axis=0))
