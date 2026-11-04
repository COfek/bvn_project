from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .sorted_array_matching import (
    sorted_array_matching,
    sorted_array_matching_noaug,
    _jit_decompose_sorted_static,
    _jit_decompose_sorted_dynamic,
    _jit_decompose_sorted_dynamic_noaug,
    _jit_decompose_sorted_static_noaug,
)
from .wfa import wavefront_matching_vectorized, _jit_decompose_wfa
from .bvn import materialise, strategy_index


@dataclass
class RadixComponent:
    matrix: np.ndarray
    weight: float


def minimum_matching_wrapper(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Wrapper for scipy's Hungarian algorithm using *minimize* cost
    (maximize=False, the scipy default).  Finds the minimum-weight
    perfect matching over the active rows of the digit plane.
    """
    tol = 1e-9
    active_rows = np.where(matrix.max(axis=1) > tol)[0]
    if len(active_rows) == 0:
        return []

    sub = matrix[active_rows, :]
    # Use a large finite penalty for zero entries so the assignment is always
    # feasible; zero-weight matches are filtered out afterwards.
    big = float(np.max(sub) * 1e6 + 1.0)
    cost = np.where(sub > tol, sub, big)
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if sub[r, c] > tol:
            matches.append((int(active_rows[r]), int(c)))
    return matches


def maximum_matching_wrapper(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Wrapper for Scipy's Hungarian algorithm restricted to active rows.
    Running n x n Hungarian when only m rows are non-zero wastes O(n^3) work
    each iteration. By extracting the m x n submatrix of active rows we pay
    O(m * n^2) instead, which is a meaningful saving when m < n (e.g. as rows
    are progressively drained during digit-plane decomposition).
    """
    tol = 1e-9
    active_rows = np.where(matrix.max(axis=1) > tol)[0]
    if len(active_rows) == 0:
        return []

    sub = matrix[active_rows, :]
    row_ind, col_ind = linear_sum_assignment(sub, maximize=True)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if sub[r, c] > tol:
            matches.append((int(active_rows[r]), int(c)))
    return matches


# Global registry of available matching algorithms
MATCHING_ALGORITHMS: Dict[str, Callable[[np.ndarray], List[Tuple[int, int]]]] = {
    "heavy": sorted_array_matching,
    "heavy_noaug": sorted_array_matching_noaug,
    "heavy_static_noaug": sorted_array_matching_noaug,  # static sort only matters inside JIT fast-path
    "wfa": wavefront_matching_vectorized,
    "maximum": maximum_matching_wrapper,
    "minimum": minimum_matching_wrapper,
}


def extract_digit_planes(matrix: np.ndarray, base: int) -> List[Tuple[float, np.ndarray]]:
    """Write every entry of `matrix` in base `base` and return the non-empty digit
    planes as (weight, plane) pairs. Split out of decompose_radix so that its cost
    can be timed on its own and charged to Radix's computation time.

    Digits are extracted until nothing is left, not up to a digit count computed
    with logarithms: log(b**e) / log(b) can round to just below e (1000 in base 10,
    243 in base 3), and the float mode scales the largest entry to exactly 2**16, so
    a count of ceil(16 / log2(b)) planes dropped every largest entry for b = 2, 4, 16.
    """
    if base < 2:
        raise ValueError(f"Radix base must be at least 2, got {base}")
    max_val = np.max(matrix)
    if max_val == 0:
        return []

    is_float = np.issubdtype(matrix.dtype, np.floating)
    # Check if a float matrix is secretly just an integer matrix.
    # We use a tiny tolerance (1e-9) in case of floating point drift during generation.
    is_functionally_integer = is_float and np.allclose(matrix, np.round(matrix), rtol=0, atol=1e-9)

    if is_float and not is_functionally_integer:
        scaling_factor = 2 ** 16                 # 16 bits of precision
        temp_matrix = np.round((matrix / max_val) * scaling_factor).astype(np.int64)
        unit = max_val / scaling_factor
    else:
        # Standard integer mode (sum of permutations); round, so that drift such as
        # 2.9999999999 is read as 3 rather than truncated to 2.
        temp_matrix = np.round(matrix).astype(np.int64)
        unit = 1.0

    planes: List[Tuple[float, np.ndarray]] = []
    d = 0
    while temp_matrix.any():
        digit_plane = temp_matrix % base
        if digit_plane.any():
            planes.append(((base ** d) * unit, digit_plane.astype(np.float64)))
        temp_matrix //= base
        d += 1
    return planes


PLANE_ENGINES = ("heavy", "heavy_static", "heavy_noaug", "heavy_static_noaug",
                 "wfa", "maximum", "minimum")


def decompose_radix(
        matrix: np.ndarray,
        base: int = 3,
        max_workers: int | None = None,
        step_strategy: str = "min",
        matching_method: str = "heavy"
) -> Tuple[List[RadixComponent], float, int]:
    """
    Decomposes an integer matrix into weighted permutations using a Radix approach.
    Simulates hardware parallelism by computing planes sequentially and returning
    the maximum execution time of any single plane.

    Args:
        matrix: The K-regular integer matrix (float matrices are quantised to 16 bits).
        base: The radix base (e.g., 2 for bitplane).
        max_workers: Ignored: the planes are timed one after another, alone.
        step_strategy: "min", "max", or "median" for weight selection.
        matching_method: one of PLANE_ENGINES.

    Returns:
        Tuple of (components, extraction_time + simulated_max_plane_runtime,
        num_non_empty_planes).
        num_non_empty_planes is the count of digit planes that contained
        at least one non-zero entry and were therefore actually decomposed.

    Each plane's time covers finding its matchings and weights; its components are
    built as dense matrices after the clock stops (see bvn.bvn_decomposition_timed),
    so while a plane is timed only the earlier planes' matchings are held.
    """
    if matching_method not in PLANE_ENGINES:
        raise ValueError(f"Unknown matching method {matching_method!r}; "
                         f"expected one of {PLANE_ENGINES}")
    strategy_int = strategy_index(step_strategy)

    t_extract = time.perf_counter()
    planes = extract_digit_planes(matrix, base)
    extraction_time = time.perf_counter() - t_extract
    if not planes:
        return [], extraction_time, 0

    max_plane_runtime = 0.0
    plans = []
    # Sequential hardware simulation: the simulated runtime is the slowest plane
    for unit_weight, plane_matrix in planes:
        t_start = time.perf_counter()
        weights, matches = _plane_plan(plane_matrix, strategy_int, matching_method)
        max_plane_runtime = max(max_plane_runtime, time.perf_counter() - t_start)
        plans.append((unit_weight, weights, matches))

    n = matrix.shape[0]
    all_components: List[RadixComponent] = []
    for unit_weight, weights, matches in plans:
        for comp in materialise([w * unit_weight for w in weights], matches, n):
            all_components.append(RadixComponent(matrix=comp.weight * comp.permutation,
                                                 weight=comp.weight))

    # The extraction runs once before the planes are decomposed in parallel, so it
    # adds to the slowest plane's time (charged since 2026-10-04; earlier campaigns
    # recorded the slowest plane alone and are corrected by radix_extraction.py).
    return all_components, extraction_time + max_plane_runtime, len(planes)


def _plane_plan(plane: np.ndarray, strategy_int: int, matching_method: str):
    """One digit plane's decomposition as (weights, matchings), weights in plane units."""
    # Work on a float copy for precision
    x = plane.astype(np.float64)
    tol = 1e-9

    # JIT fast path: heavy / heavy_static / wfa for all three strategies
    if matching_method == "heavy":
        return _jit_decompose_sorted_dynamic(x, strategy_int=strategy_int, tol=tol)
    if matching_method == "heavy_static":
        return _jit_decompose_sorted_static(x, strategy_int=strategy_int, tol=tol)
    if matching_method == "wfa":
        return _jit_decompose_wfa(x, x.shape[0], tol=tol, strategy_int=strategy_int)
    if matching_method in ("heavy_noaug", "heavy_static_noaug"):
        if strategy_int != 0:
            raise ValueError(f"{matching_method!r} supports step_strategy='min' only")
        if matching_method == "heavy_noaug":
            return _jit_decompose_sorted_dynamic_noaug(x, tol=tol)
        return _jit_decompose_sorted_static_noaug(x, tol=tol)

    # Slow path: Python loop for Hungarian (maximum/minimum), which has no JIT version
    match_func = MATCHING_ALGORITHMS[matching_method]
    epsilon = 1e-9
    weights: List[float] = []
    matches_out: list = []
    while np.any(x > epsilon):
        matches = match_func(x)
        if not matches:
            break
        match_values = np.sort([x[i, j] for (i, j) in matches])
        if strategy_int == 2:
            digit_step = float(match_values[-1])
        elif strategy_int == 1:
            # the upper median, as bvn.py and the JIT kernels take it: an entry of
            # the matching, so the step stays integral on an integral plane
            digit_step = float(match_values[len(match_values) // 2])
        else:
            digit_step = float(match_values[0])
        if digit_step < 1e-12:
            break

        rows_arr, cols_arr = (np.array(a) for a in zip(*matches))
        x[rows_arr, cols_arr] = np.maximum(0.0, x[rows_arr, cols_arr] - digit_step)
        weights.append(digit_step)
        matches_out.append((rows_arr, cols_arr))
    return weights, matches_out
