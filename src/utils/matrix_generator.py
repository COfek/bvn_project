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


def generate_binary_weighted_matrix(
    n: int,
    rng: np.random.Generator,
    bits: int = 8
) -> np.ndarray:
    """
    Generates a matrix by summing weighted permutations:
    M = 1*P0 + 2*P1 + ... + 2^(bits-1)*P(bits-1)
    
    This results in a K=(2^bits - 1) matrix (if unscaled) with entries in range [0, 2^bits - 1].
    NOTE: float64 has 53 bits of precision. bits > 53 will lose lower planes.
    """
    # Use object type for intermediate calculation if bits > 63 to avoid overflow before cast?
    # But ultimately we return float64, so >53 is lossy regardless.
    # We stick to int64 accumulator which is safe up to 63 bits.
    result = np.zeros((n, n), dtype=np.int64)
    
    for i in range(bits):
        weight = 2**i
        p = rng.permutation(n)
        result[np.arange(n), p] += weight
        
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


def generate_weighted_sum_matrix(
    n: int,
    weights: list[int],
    sub_k: int | list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generates a matrix by summing weighted K-regular layers.
    M = w0 * Layer0 + w1 * Layer1 + ...
    
    Each Layer_i is a sum of sub_k[i] (or sub_k if int) random permutations.
    This creates a matrix where decomposition in the basis of `weights` yields
    dense layers of weight sub_k[i], rather than trivial single permutations.
    
    Args:
        n: Matrix dimension.
        weights: List of weights for each layer (e.g. [1, 16, 256] for base 16).
        sub_k: Number of permutations per layer. Can be a single int or a list
               matching len(weights).
        rng: Random generator.
        
    Returns:
        A float64 matrix.
    """
    if isinstance(sub_k, int):
        sub_k_list = [sub_k] * len(weights)
    else:
        sub_k_list = sub_k
        if len(sub_k_list) != len(weights):
            raise ValueError("Length of sub_k list must match length of weights")
            
    result = np.zeros((n, n), dtype=np.int64)
    
    for w, k in zip(weights, sub_k_list):
        # Generate a layer with sum k
        layer = generate_scaled_doubly_stochastic_matrix(n, k, rng)
        # Add to result with weight w
        # layer is already int64 inside generate_scaled_doubly_stochastic_matrix (before cast to float)
        # But the function returns float64, so we cast back to int to keep precision
        # or just use the float result. Since we want exact integers, let's cast.
        result += (layer.astype(np.int64) * w)
        
    return result.astype(np.float64)

