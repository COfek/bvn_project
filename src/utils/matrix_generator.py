from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]

def generate_scaled_doubly_stochastic_matrix(
        n: int,
        k: int,
        rng: np.random.Generator
) -> np.ndarray:
    """
    Generates a K-regular matrix by summing exactly K permutations.
    Each row and column will sum exactly to K.
    """
    # Initialize as integers to ensure exact sums
    result = np.zeros((n, n), dtype=np.int64)

    # The loop MUST run exactly 'k' times total
    # If k=63, we should have 63 permutations in the sum
    for _ in range(int(k)):
        p = rng.permutation(n)
        result[np.arange(n), p] += 1

    # Return as float64 to prevent the UFunc Casting Errors we saw earlier
    return result.astype(np.float64)

# --- The fixed validation function ---
def check_k_regularity(matrix: NDArray[np.int64], expected_k: int) -> bool:
    """
    Checks if all rows and columns sum exactly to expected_k.
    """
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)

    rows_ok = np.all(row_sums == expected_k)
    cols_ok = np.all(col_sums == expected_k)

    return rows_ok and cols_ok
