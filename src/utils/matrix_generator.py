from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_matrix(
    n: int, 
    k: int, 
    max_weight: float, 
    rng: np.random.Generator,
    float_weights: bool = False
) -> NDArray[np.float64]:
    r"""
    Generates an n x n matrix by summing 'k' weighted random permutations.
    For each permutation, a random weight is chosen uniformly in the range [0, max_weight].
    Can optionally pick real numbers (float_weights=True) instead of integers.
    
    Formula: M = \sum_{i=1}^{k} P_i * Random(0, W)
    
    Args:
        n: Dimension of the matrix (n x n)
        k: The number of permutations to sum
        max_weight: The maximum weight W (inclusive for ints, exclusive for floats) to randomly sample from.
        rng: NumPy random generator
        float_weights: If True, weights are drawn uniformly from [0, max_weight) as floats.
        
    Returns:
        A float64 matrix.
    """
    if max_weight < 0:
        max_weight = 0
        
    if float_weights:
        result = np.zeros((n, n), dtype=np.float64)
    else:
        result = np.zeros((n, n), dtype=np.int64)
    
    for _ in range(int(k)):
        if float_weights:
            w = rng.uniform(0, max_weight)
        else:
            w = int(rng.integers(0, int(max_weight) + 1))
            
        p = rng.permutation(n)
        result[np.arange(n), p] += w
        
    return result.astype(np.float64)


def check_k_regularity(matrix: NDArray[np.float64], expected_k: float) -> bool:
    """
    Checks if all rows and columns sum exactly to expected_k.
    Note: expected_k should be the sum of all chosen weights.
    """
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)

    rows_ok = np.allclose(row_sums, expected_k)
    cols_ok = np.allclose(col_sums, expected_k)

    return bool(rows_ok and cols_ok)
