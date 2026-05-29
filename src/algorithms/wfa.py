from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from numba import jit

BoolMatrix = NDArray[np.bool_]


def wavefront_matching(mask: BoolMatrix) -> List[Tuple[int, int]]:
    """
    Wavefront Arbiter maximal matching (hardware-style diagonal sweep).
    Produces a GREEDY maximal matching (not maximum).
    """
    n = mask.shape[0]

    row_free = np.ones(n, dtype=bool)
    col_free = np.ones(n, dtype=bool)
    matches: List[Tuple[int, int]] = []
    # Sweep diagonals k = i + j
    for k in range(2 * n - 1):
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                if row_free[i] and col_free[j] and mask[i, j]:
                    matches.append((i, j))
                    row_free[i] = False
                    col_free[j] = False
    return matches


from numba import jit


@jit(nopython=True, nogil=True)
def _jit_wfa_kernel(
    matrix: np.ndarray, # Numeric matrix
    n: int,
    tol: float
) -> List[Tuple[int, int]]:
    """
    Numba-optimized Wavefront Arbiter Kernel.
    
    Iterates through the diagonals of the matrix (k = i + j) to find a maximal matching.
    This mimics the hardware wavefront propagation.
    
    Args:
        matrix (np.ndarray): The N x N float matrix.
        n (int): Matrix dimension.
        tol (float): Tolerance check for edges.
        
    Returns:
        List[Tuple[int, int]]: A list of (row, col) indices representing the matching.
    """
    row_free = np.ones(n, dtype=np.int8)
    col_free = np.ones(n, dtype=np.int8)
    matches = []
    
    # Sweep diagonals k = i + j
    # Max k is 2*n - 2 (indices 0..n-1)
    for k in range(2 * n - 1):
        # Calculate range for i
        start_i = max(0, k - n + 1)
        end_i = min(k + 1, n)
        
        # Iterate diagonal elements
        for i in range(start_i, end_i):
            j = k - i
            
            # Logic: If free and eligible
            if row_free[i] == 1 and col_free[j] == 1:
                if matrix[i, j] > tol:
                    matches.append((i, j))
                    row_free[i] = 0
                    col_free[j] = 0
                    
    return matches


@jit(nopython=True, nogil=True)
def _jit_decompose_wfa(matrix: np.ndarray, n: int, tol: float, strategy_int: int = 0) -> Tuple[List[float], List[List[Tuple[int, int]]]]:
    """
    Perform the entire iterative decomposition in Numba to release the GIL.

    Args:
        matrix (np.ndarray): The N x N float matrix to decompose. Modified in-place.
        n (int): Dimension of the matrix.
        tol (float): Tolerance for treating values as zero.
        strategy_int (int): Step-size strategy: 0=min (standard), 1=median, 2=max.

    Returns:
        Tuple[List[float], List[List[Tuple[int, int]]]]:
            - A list of weights (lambda values).
            - A list of matching lists (list of (row, col) tuples).
    """
    weights = []
    all_matches = []

    while True:
        matches = _jit_wfa_kernel(matrix, n, tol)

        if len(matches) == 0:
            break

        # Find step λ according to strategy
        m = len(matches)
        matched_w = np.empty(m, dtype=np.float64)
        for k in range(m):
            i, j = matches[k]
            matched_w[k] = matrix[i, j]

        if strategy_int == 2:  # max
            lam = 0.0
            for k in range(m):
                if matched_w[k] > lam:
                    lam = matched_w[k]
        elif strategy_int == 1:  # median
            sorted_w = np.sort(matched_w)
            lam = sorted_w[m // 2]
        else:  # min (default, strategy_int == 0)
            lam = np.inf
            for k in range(m):
                if matched_w[k] < lam:
                    lam = matched_w[k]

        if lam <= tol:
            break

        # Subtract λ (clip at 0 for safety with max/median over-serving)
        for k in range(m):
            i, j = matches[k]
            matrix[i, j] = max(0.0, matrix[i, j] - lam)

        weights.append(lam)
        all_matches.append(matches)

    return weights, all_matches


def wavefront_matching_vectorized(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    JIT-Accelerated Wavefront Matching (Single Step).
    Designed to be compatible with radix decomposition wrapper.
    Accepts a numeric matrix, treats > 0 as edges.
    """
    n = matrix.shape[0]
    return _jit_wfa_kernel(matrix, n, 1e-9)


