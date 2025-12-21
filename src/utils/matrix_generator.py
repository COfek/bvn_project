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


def split_tree_friendly_matrix(
    n: int,
    num_blocks: int = 4,
    tail_alpha: float = 1.5,
    spike_fraction: float = 0.05,
    sinkhorn_iters: int = 500,
    seed: int | None = None,
):
    """
    Generate a doubly stochastic matrix that favors split-tree over bit-plane.

    Properties:
    - heavy-tailed weights (pareto)
    - block / hierarchical structure
    - few large spikes, many tiny residuals
    """
    assert n % num_blocks == 0
    rng = np.random.default_rng(seed)
    block_size = n // num_blocks

    a = np.zeros((n, n))

    # ----------------------------------------------------
    # 1. block-structured heavy-tailed mass
    # ----------------------------------------------------
    for b in range(num_blocks):
        i0 = b * block_size
        i1 = (b + 1) * block_size

        block = rng.pareto(tail_alpha, size=(block_size, block_size))
        a[i0:i1, i0:i1] = block

    # ----------------------------------------------------
    # 2. inject sparse large spikes (worst for bit-plane)
    # ----------------------------------------------------
    num_spikes = int(spike_fraction * n * n)
    for _ in range(num_spikes):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        a[i, j] += rng.uniform(50, 200)

    # ----------------------------------------------------
    # 3. small background noise everywhere
    # ----------------------------------------------------
    a += 1e-6 * rng.random((n, n))

    # ----------------------------------------------------
    # 4. sinkhorn normalization
    # ----------------------------------------------------
    for _ in range(sinkhorn_iters):
        a /= a.sum(axis=1, keepdims=True)
        a /= a.sum(axis=0, keepdims=True)

    return a
