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


def wavefront_matching_vectorized(matrix: NDArray) -> List[Tuple[int, int]]:
    """
    Vectorized Wavefront matching. Now accepts numeric matrices
    to support a universal algorithm interface.
    """
    # 1. Internal Conversion: Treat any non-zero value as 'True'
    # This ensures WFA logic remains the same even if passed numeric weights
    if matrix.dtype != bool:
        mask = matrix > 0
    else:
        mask = matrix

    n = mask.shape[0]
    row_free = np.ones(n, dtype=bool)
    col_free = np.ones(n, dtype=bool)
    matches = []

    for k in range(2 * n - 1):
        i_indices = np.arange(max(0, k - n + 1), min(k + 1, n))
        j_indices = k - i_indices

        # Diagonal evaluation
        eligible = mask[i_indices, j_indices] & row_free[i_indices] & col_free[j_indices]

        if np.any(eligible):
            for idx in np.where(eligible)[0]:
                row_idx = i_indices[idx]
                col_idx = j_indices[idx]
                matches.append((row_idx, col_idx))
                row_free[row_idx] = False
                col_free[col_idx] = False

    return matches
