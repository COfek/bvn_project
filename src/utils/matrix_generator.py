from __future__ import annotations

import numpy as np
from typing import Optional

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]


def random_sparse_doubly_stochastic(
    n: int,
    density: float = 0.9,
    iters: int = 200,
    eps: float = 1e-12,
    rng: Optional[np.random.Generator] = None,
) -> FloatMatrix:
    """
    Generate an n x n sparse doubly stochastic matrix using Sinkhorn scaling.

    Args:
        n: Matrix dimension (n x n).
        density: Fraction in (0, 1] controlling sparsity (1.0 = dense).
        iters: Number of Sinkhorn iterations.
        eps: Small positive constant to avoid division by zero.
        rng: Optional numpy random generator.

    Returns:
        A (numerically) doubly stochastic matrix of shape (n, n).
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = rng.random((n, n)) < density
    matrix = rng.random((n, n)) * mask

    # Ensure no empty rows
    for i in range(n):
        if np.sum(mask[i]) == 0:
            j = rng.integers(0, n)
            mask[i, j] = True
            matrix[i, j] = float(rng.random() + eps)

    # Ensure no empty columns
    for j in range(n):
        if np.sum(mask[:, j]) == 0:
            i = rng.integers(0, n)
            mask[i, j] = True
            matrix[i, j] = float(rng.random() + eps)

    for _ in range(iters):
        # Normalize rows
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / (row_sums + eps)

        # Normalize columns
        col_sums = matrix.sum(axis=0, keepdims=True)
        matrix = matrix / (col_sums + eps)

        # Re-impose sparsity
        matrix *= mask

    return matrix.astype(np.float64)


def generate_sinkhorn_matrix(
    n: int,
    k: int,
    rng: np.random.Generator,
    density: float = 0.5
) -> np.ndarray:
    """
    Generates a matrix by creating a doubly stochastic matrix via Sinkhorn
    and scaling it by K. Returns a float matrix.
    """
    ds_matrix = random_sparse_doubly_stochastic(
        n=n,
        density=density,
        rng=rng
    )
    # Scale by K
    return ds_matrix * k

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
