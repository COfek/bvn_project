from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from .sorted_array_matching import (
    _jit_decompose_sorted_static,
    _jit_decompose_sorted_dynamic,
    _jit_decompose_sorted_dynamic_noaug,
    _jit_decompose_sorted_static_noaug,
)
from .wfa import _jit_decompose_wfa
from .wwfa import _jit_decompose_wwfa

FloatMatrix = NDArray[np.float64]

STRATEGY_INT = {"min": 0, "median": 1, "max": 2}
JIT_ENGINES = ("heavy", "heavy_static", "wfa", "wwfa", "heavy_noaug", "heavy_static_noaug")


@dataclass
class DecompositionComponent:
    """
    Single component in a decomposition.
    """
    permutation: FloatMatrix | Dict[int, int]
    weight: float


def strategy_index(step_strategy: str) -> int:
    """The JIT kernels' code for a step strategy: 0 = min (classic BvN), 1 = median, 2 = max."""
    try:
        return STRATEGY_INT[step_strategy]
    except KeyError:
        raise ValueError(f"Unknown step strategy {step_strategy!r}; "
                         f"expected one of {sorted(STRATEGY_INT)}") from None


def _drop_empty_cells(work, row_ind, col_ind, tol):
    """Keep only the assigned cells that carry positive residual demand.

    `linear_sum_assignment` returns a *complete* assignment of the active rows.
    When the residual has no perfect matching on its positive entries -- which
    happens as soon as the input is not regular -- some rows are assigned to zero
    cells via the large-cost fallback. Recording those cells in the permutation
    would inflate the reconstruction: the cell contributes lambda to the sum while
    the residual there stays clipped at zero, so sum(lambda_i P_i) > D.

    Dropping them turns the peel into a partial matching, i.e. a weak
    decomposition step, which is what a non-regular residual actually admits.
    This mirrors the filter already applied in radix_decomposition's Hungarian
    wrappers, where non-regular digit planes are the norm.

    On regular input this is a no-op: a positive perfect matching exists at every
    peel, and any fallback cell costs more than every positive cell, so the
    optimal assignment never selects one.
    """
    keep = work[row_ind, col_ind] > tol
    return row_ind[keep], col_ind[keep]


def bvn_decomposition(
    matrix: FloatMatrix,
    matching_algorithm: str = "maximum",
    max_iters: Optional[int] = None,
    step_strategy: str = "min",
) -> List[DecompositionComponent]:
    """
    Perform an exact Birkhoff-von Neumann decomposition on a K-regular integer matrix.

    Args:
        matrix: The input matrix (must be Doubly Stochastic / K-regular).
        matching_algorithm: 'maximum' / 'minimum' (Hungarian), 'heavy' / 'heavy_static'
            (GW Dynamic / GW Static), 'wfa', 'wwfa' (wrapped WFA), or the weak '*_noaug' variants.
        max_iters: Optional limit on iterations (Hungarian engines only).
        step_strategy: 'min', 'median' or 'max' ('*_noaug' support 'min' only).

    Because the input is a sum of permutations, this algorithm is guaranteed
    to terminate when the residual matrix is exactly zero.
    """
    return bvn_decomposition_timed(matrix, matching_algorithm, max_iters, step_strategy)[0]


def bvn_decomposition_timed(
    matrix: FloatMatrix,
    matching_algorithm: str = "maximum",
    max_iters: Optional[int] = None,
    step_strategy: str = "min",
) -> tuple[List[DecompositionComponent], float]:
    """bvn_decomposition, plus the time it took to find the decomposition.

    The clock covers finding the matchings and their weights. Turning each matching
    into a dense n x n matrix comes after it: that costs O(N n^2) memory traffic,
    which grows with the number of configurations rather than with the matching
    work, and before 2026-11-04 it made up about half of the time recorded for the
    engines with many configurations (docs/code_review_2026-11-04.md, A1).
    """
    t0 = time.perf_counter()
    weights, matches = _bvn_plan(matrix, matching_algorithm, max_iters, step_strategy)
    seconds = time.perf_counter() - t0
    n = np.shape(matrix)[0]
    return materialise(weights, matches, n, skip_below=1e-12), seconds


def materialise(weights, matches, n: int, skip_below: Optional[float] = None
                ) -> List[DecompositionComponent]:
    """Turn (weight, matching) pairs into components with dense permutation matrices.

    A matching is either a sequence of (row, col) pairs, as the JIT kernels return
    it, or a (rows, cols) tuple of index arrays. Weights at or below skip_below are
    dropped.
    """
    components: List[DecompositionComponent] = []
    for w, m in zip(weights, matches):
        if skip_below is not None and w <= skip_below:
            continue
        permutation = np.zeros((n, n), dtype=np.float64)
        if isinstance(m, tuple):
            permutation[m[0], m[1]] = 1.0
        elif len(m):
            rows, cols = zip(*m)
            permutation[list(rows), list(cols)] = 1.0
        components.append(DecompositionComponent(permutation=permutation, weight=float(w)))
    return components


def _bvn_plan(matrix, matching_algorithm, max_iters, step_strategy):
    """The decomposition as (weights, matchings), without building any matrix."""
    strategy_int = strategy_index(step_strategy)
    # Work on a copy to avoid modifying the original matrix
    work = np.array(matrix, dtype=np.float64, copy=True)
    n = work.shape[0]

    if matching_algorithm in JIT_ENGINES:
        if max_iters is not None:
            raise ValueError(f"max_iters is not supported by the JIT engine {matching_algorithm!r}")
        if matching_algorithm == "heavy":
            return _jit_decompose_sorted_dynamic(work, strategy_int, 1e-9)
        if matching_algorithm == "heavy_static":
            return _jit_decompose_sorted_static(work, strategy_int, 1e-9)
        if matching_algorithm == "wfa":
            return _jit_decompose_wfa(work, n, 1e-9, strategy_int)
        if matching_algorithm == "wwfa":
            return _jit_decompose_wwfa(work, n, 1e-9, strategy_int)
        # GW without augmenting paths: weak (partial-matching) decompositions, min step only
        if strategy_int != 0:
            raise ValueError(f"{matching_algorithm!r} supports step_strategy='min' only")
        if matching_algorithm == "heavy_noaug":
            return _jit_decompose_sorted_dynamic_noaug(work, 1e-9)
        return _jit_decompose_sorted_static_noaug(work, 1e-9)

    if matching_algorithm not in ("maximum", "minimum"):
        raise ValueError(f"Unknown matching algorithm: {matching_algorithm}")

    weights: List[float] = []
    matches: list = []
    iteration = 0
    tol = 1e-9
    # In integer space, we continue until the matrix is purely zero
    while np.any(work > tol):
        iteration += 1
        # Only consider rows that still have at least one positive entry.
        active_rows = np.where(work.max(axis=1) > tol)[0]
        if len(active_rows) == 0:
            break
        sub = work[np.ix_(active_rows, np.arange(n))]
        if matching_algorithm == "maximum":
            # Zero entries get a large positive cost (= avoid in minimisation of
            # -weight), so every active row gets assigned to a positive cell.
            big = float(sub.max() + 1.0) * n   # penalty > any real weight
            cost = np.where(sub > tol, -sub, big)  # minimise -> prefer large weights
        else:
            # Minimum-weight perfect matching over positive edges only; zero entries
            # get a large finite penalty so the problem stays feasible.
            big = float(np.max(sub) * 1e6 + 1.0)
            cost = np.where(sub > tol, sub, big)
        sub_row_ind, col_ind = linear_sum_assignment(cost)
        row_ind, col_ind = _drop_empty_cells(work, active_rows[sub_row_ind], col_ind, tol)
        if len(row_ind) == 0:
            break

        # lambda over POSITIVE selected values only: a median/max over zeros would
        # stall the drain and truncate the decomposition.
        selected_values = work[row_ind, col_ind]
        positive_values = selected_values[selected_values > 1e-12]
        if len(positive_values) == 0:
            break
        if strategy_int == 2:
            lambda_value = float(np.max(positive_values))
        elif strategy_int == 1:
            lambda_value = float(np.sort(positive_values)[len(positive_values) // 2])
        else:
            lambda_value = float(np.min(positive_values))
        if lambda_value <= 1e-12:
            break

        weights.append(lambda_value)
        matches.append((row_ind.copy(), col_ind.copy()))
        # Subtract the weight (clip at 0: median/max strategies overdraw cells
        # below lambda, same as the JIT kernels), then clean up small residuals.
        work[row_ind, col_ind] = np.maximum(work[row_ind, col_ind] - lambda_value, 0.0)
        work[work < 1e-12] = 0.0

        if max_iters is not None and iteration >= max_iters:
            break

    return weights, matches
