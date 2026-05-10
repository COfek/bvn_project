from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_matrix(
    n: int,
    k: int,
    max_weight: float,
    rng: np.random.Generator,
    float_weights: bool = False,
    unit_weight: float = 1.0,
) -> NDArray[np.float64]:
    r"""
    Generates an n x n scaled doubly-stochastic matrix as a sum of k weighted
    random permutations, then scales the result by unit_weight.

    Each permutation draws one weight uniformly from [0, max_weight] (integer or
    float).  Setting max_weight to B-1 (e.g. 15 for base-16) keeps every entry
    in {0, ..., k*(B-1)}, while unit_weight = B^d shifts the significance level
    without changing the digit structure.

    Args:
        n:            Dimension of the n x n matrix.
        k:            Number of permutations to sum.
        max_weight:   Upper bound for each permutation's weight.
        rng:          NumPy random generator.
        float_weights: Draw weights from [0, max_weight) as floats instead of ints.
        unit_weight:  Scalar applied to the whole matrix after generation.
                      Use powers of the radix base (1, 16, 256, ...) to place the
                      matrix at a specific digit-plane significance level.

    Returns:
        A float64 matrix with row/col sums = (sum of chosen weights) * unit_weight.
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

    return result.astype(np.float64) * unit_weight


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
