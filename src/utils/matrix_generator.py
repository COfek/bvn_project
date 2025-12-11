from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]


def random_sparse_doubly_stochastic(
    n: int,
    density: float = 0.2,
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


def is_doubly_stochastic(matrix: FloatMatrix, tol: float = 1e-9) -> bool:
    """
    Check whether a matrix is numerically doubly stochastic.

    Conditions:
      1. All entries >= 0
      2. Each row sums to 1
      3. Each column sums to 1

    Args:
        matrix: Input matrix.
        tol: Numerical tolerance.

    Returns:
        True if the matrix is (approximately) doubly stochastic, False otherwise.
    """
    arr = matrix.astype(float)

    if np.any(arr < -tol):
        return False

    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        return False

    col_sums = arr.sum(axis=0)
    if not np.allclose(col_sums, 1.0, atol=tol):
        return False

    return True
