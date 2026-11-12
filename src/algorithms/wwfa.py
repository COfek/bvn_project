"""Wrapped Wavefront Arbiter (WWFA) and a common decomposition loop for the WFA family.

Plain WFA (wfa.py) sweeps the 2n-1 anti-diagonals k = i + j, so it always starts at the
top-left cell and favours low rows and columns. WWFA (Tamir and Chi, 1993) sweeps the n
*wrapped* diagonals d = (j - i) mod n instead: each holds n cells, no two of which share a row
or a column, so a sweep takes n steps. The hardware rotates which wrapped diagonal goes first
from one time slot to the next, for fairness; with `rotate` the start diagonal advances by one
for every matching extracted, otherwise it is always diagonal 0.

The engine "wwfa" (bvn.py, radix_decomposition.py, euler_splitting.py) is
_jit_decompose_wwfa: rotating start, step chosen as in the WFA engine (min / median / max).
wfa.py is not changed. decompose_counts is the counting loop used by the check of the
cycle-length ratio reported for maximal matching in the BhaVaN dissertation
(agent_scripts/wwfa_check.py).

Orders:   0 = plain WFA (anti-diagonals i + j), 1 = WWFA fixed start, 2 = WWFA rotating start
Steps:    0 = min step (subtract the smallest matched entry; weight = that entry)
          1 = unit step (subtract 1 from every matched entry; weight 1), as in the dissertation
"""
from __future__ import annotations

import numpy as np
from numba import jit


@jit(nopython=True, nogil=True, cache=True)
def _kernel(matrix, n, tol, order, start):
    """One maximal matching on the support of `matrix` (entries > tol)."""
    row_free = np.ones(n, dtype=np.int8)
    col_free = np.ones(n, dtype=np.int8)
    rows = np.empty(n, dtype=np.int64)
    cols = np.empty(n, dtype=np.int64)
    m = 0
    if order == 0:
        for k in range(2 * n - 1):
            for i in range(max(0, k - n + 1), min(k + 1, n)):
                j = k - i
                if row_free[i] == 1 and col_free[j] == 1 and matrix[i, j] > tol:
                    rows[m] = i
                    cols[m] = j
                    m += 1
                    row_free[i] = 0
                    col_free[j] = 0
    else:
        for step in range(n):
            d = (start + step) % n
            for i in range(n):
                j = (i + d) % n
                if row_free[i] == 1 and col_free[j] == 1 and matrix[i, j] > tol:
                    rows[m] = i
                    cols[m] = j
                    m += 1
                    row_free[i] = 0
                    col_free[j] = 0
    return rows[:m], cols[:m]


@jit(nopython=True, nogil=True, cache=True)
def decompose_counts(matrix, order, step):
    """Decompose `matrix` (modified in place) and return (cycle, configurations).

    cycle = sum of the weights; configurations = number of matchings extracted.
    Only the counts are returned: this is for measuring cycle length, not for scheduling.
    """
    n = matrix.shape[0]
    tol = 1e-9
    cycle = 0.0
    count = 0
    t = 0
    while True:
        start = t % n if order == 2 else 0
        rows, cols = _kernel(matrix, n, tol, order, start)
        m = rows.shape[0]
        if m == 0:
            break
        if step == 1:
            lam = 1.0
        else:
            lam = np.inf
            for k in range(m):
                v = matrix[rows[k], cols[k]]
                if v < lam:
                    lam = v
        for k in range(m):
            v = matrix[rows[k], cols[k]] - lam
            matrix[rows[k], cols[k]] = v if v > 0.0 else 0.0
        cycle += lam
        count += 1
        t += 1
    return cycle, count


@jit(nopython=True, nogil=True, cache=True)
def _jit_decompose_wwfa(matrix, n, tol, strategy_int=0):
    """The WWFA engine: rotating start, the same interface as wfa._jit_decompose_wfa.

    `matrix` is modified in place. The start diagonal is 0 for the first matching and
    advances by one for every matching extracted. strategy_int: 0 = min, 1 = median, 2 = max.
    Returns (weights, matchings), each matching a list of (row, col) tuples.
    """
    weights = []
    all_matches = []
    t = 0
    while True:
        rows, cols = _kernel(matrix, n, tol, 2, t % n)
        m = rows.shape[0]
        if m == 0:
            break
        matched_w = np.empty(m, dtype=np.float64)
        for k in range(m):
            matched_w[k] = matrix[rows[k], cols[k]]
        if strategy_int == 2:
            lam = matched_w.max()
        elif strategy_int == 1:
            lam = np.sort(matched_w)[m // 2]
        else:
            lam = matched_w.min()
        if lam <= tol:
            break
        matches = []
        for k in range(m):
            v = matrix[rows[k], cols[k]] - lam
            matrix[rows[k], cols[k]] = v if v > 0.0 else 0.0
            matches.append((rows[k], cols[k]))
        weights.append(lam)
        all_matches.append(matches)
        t += 1
    return weights, all_matches
