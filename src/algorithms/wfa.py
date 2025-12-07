from __future__ import annotations
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

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

def wfa_randomized(mask: BoolMatrix) -> List[Tuple[int, int]]:
    n = mask.shape[0]
    start = np.random.randint(0, 2 * n - 1)

    row_free = np.ones(n, dtype=bool)
    col_free = np.ones(n, dtype=bool)
    matches: List[Tuple[int, int]] = []

    for d in range(start, start + 2 * n - 1):
        k = d % (2 * n - 1)
        for i in range(n):
            j = k - i
            if 0 <= j < n and row_free[i] and col_free[j] and mask[i, j]:
                matches.append((i, j))
                row_free[i] = False
                col_free[j] = False

    return matches


def wfa_weighted(mask: BoolMatrix, weights: NDArray[np.float_]) -> List[Tuple[int, int]]:
    n = mask.shape[0]
    row_free = np.ones(n, dtype=bool)
    col_free = np.ones(n, dtype=bool)
    matches: List[Tuple[int, int]] = []

    # Precompute diagonals with weighted ordering
    diagonals = []
    for k in range(2 * n - 1):
        diag_entries = []
        for i in range(n):
            j = k - i
            if 0 <= j < n and mask[i, j]:
                diag_entries.append((weights[i, j], i, j))
        diag_entries.sort(reverse=True)  # highest weight first
        diagonals.append(diag_entries)

    # Sweep with priority
    for diag in diagonals:
        for w, i, j in diag:
            if row_free[i] and col_free[j]:
                matches.append((i, j))
                row_free[i] = False
                col_free[j] = False
                break

    return matches


def wfa_rotating(mask: BoolMatrix, round_idx: int = 0) -> List[Tuple[int, int]]:
    n = mask.shape[0]

    # Rotate priority each iteration
    row_order = np.roll(np.arange(n), shift=round_idx)
    col_order = np.roll(np.arange(n), shift=round_idx)

    row_free = np.ones(n, dtype=bool)
    col_free = np.ones(n, dtype=bool)
    matches: List[Tuple[int, int]] = []

    for k in range(2 * n - 1):
        for ri in row_order:
            j = k - ri
            if 0 <= j < n:
                cj = col_order[j]
                if row_free[ri] and col_free[cj] and mask[ri, cj]:
                    matches.append((ri, cj))
                    row_free[ri] = False
                    col_free[cj] = False

    return matches


def wfa_early_exit(mask: BoolMatrix) -> List[Tuple[int, int]]:
    n = mask.shape[0]
    row_free = np.ones(n, dtype=bool)
    col_free = np.ones(n, dtype=bool)
    matches: List[Tuple[int, int]] = []

    for k in range(2 * n - 1):
        # Skip if no free rows OR no free cols can possibly be used
        if not row_free.any() or not col_free.any():
            break

        for i in np.where(row_free)[0]:
            j = k - i
            if 0 <= j < n and col_free[j] and mask[i, j]:
                matches.append((i, j))
                row_free[i] = False
                col_free[j] = False

    return matches


from scipy.optimize import linear_sum_assignment

def maximum_matching(mask: BoolMatrix) -> List[Tuple[int, int]]:
    n = mask.shape[0]
    cost = np.where(mask, 0, 1e9)
    row_ind, col_ind = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(row_ind, col_ind) if mask[i, j]]

def wfa_hybrid(mask: BoolMatrix, min_factor: float = 0.7) -> List[Tuple[int, int]]:
    # Step 1: WFA
    wfa_result = wavefront_matching(mask)

    # Step 2: Compute maximum matching
    max_result = maximum_matching(mask)
    max_size = len(max_result)

    # Step 3: If WFA is too small, use maximum matching instead
    if len(wfa_result) < min_factor * max_size:
        return max_result
    return wfa_result