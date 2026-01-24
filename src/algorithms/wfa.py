from __future__ import annotations
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

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
    mask: np.ndarray, # Boolean or numeric mask
    n: int
) -> List[Tuple[int, int]]:
    row_free = np.ones(n, dtype=np.int8)
    col_free = np.ones(n, dtype=np.int8)
    matches = []
    
    # Sweep diagonals k = i + j
    # Max k is 2*n - 2 (indices 0..n-1)
    for k in range(2 * n - 1):
        # Calculate range for i
        # i + j = k => j = k - i
        # 0 <= i < n AND 0 <= j < n
        # => 0 <= k - i < n => i <= k AND i > k - n
        
        start_i = max(0, k - n + 1)
        end_i = min(k + 1, n)
        
        # Iterate diagonal elements
        for i in range(start_i, end_i):
            j = k - i
            
            # Logic: If free and eligible
            # Note: Assuming mask[i, j] checks > 0 logic externally or implicit bool
            if row_free[i] == 1 and col_free[j] == 1:
                # Numba handles non-zero check on numeric types as Truthy usually,
                # but let's be explicit if possible. But inputs vary.
                if mask[i, j]:  # Works for bool and non-zero numbers
                    matches.append((i, j))
                    row_free[i] = 0
                    col_free[j] = 0
                    
    return matches


def wavefront_matching_vectorized(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    JIT-Accelerated Wavefront Matching.
    """
    n = matrix.shape[0]
    
    # 1. Internal Conversion: Numba works best with typed arrays.
    # We can pass the matrix directly. If float/int, it treats non-zero as True.
    return _jit_wfa_kernel(matrix, n)
