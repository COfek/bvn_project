from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Local imports - ensuring paths match your project structure
from .sorted_array_matching import (
    sorted_array_matching,
    sorted_array_matching_noaug,
    _jit_decompose_sorted_static,
    _jit_decompose_sorted_dynamic,
    _jit_decompose_sorted_dynamic_noaug,
    _jit_decompose_sorted_static_noaug,
)
from .wfa import wavefront_matching_vectorized, _jit_decompose_wfa


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
        matrix: The K-regular integer matrix.
        base: The radix base (e.g., 2 for bitplane).
        max_workers: Ignored (forced sequential for hardware simulation).
        step_strategy: "min", "max", or "median" for weight selection.
        matching_method: "heavy", "wfa", or "maximum".
        float_precision_bits: Bits of precision for Sinkhorn/Float matrices.
        normalize_input: If True, scale float matrix to int space (Sinkhorn mode).

    Returns:
        Tuple of (components, simulated_max_plane_runtime, num_non_empty_planes).
        num_non_empty_planes is the count of digit planes that contained
        at least one non-zero entry and were therefore actually decomposed.
    """
    max_val = np.max(matrix)
    if max_val == 0:
        return [], 0.0, 0

    planes: List[Tuple[float, np.ndarray]] = []

    is_float = np.issubdtype(matrix.dtype, np.floating)
    
    # Check if a float matrix is secretly just an integer matrix.
    # We use a tiny tolerance (1e-9) in case of floating point drift during generation.
    is_functionally_integer = is_float and np.allclose(matrix, np.round(matrix), rtol=0, atol=1e-9)

    if is_float and not is_functionally_integer:
        precision_bits = 16
        scaling_factor = 2 ** precision_bits
        scaled_matrix = np.round((matrix / max_val) * scaling_factor).astype(np.int64)
        num_planes = int(np.ceil(np.log(scaling_factor) / np.log(base)))
        temp_matrix = scaled_matrix.copy()

        for d in range(num_planes):
            unit_weight = (base ** d) * (max_val / scaling_factor)
            digit_plane = temp_matrix % base
            if np.any(digit_plane > 0):
                planes.append((unit_weight, digit_plane.astype(np.float64)))
            temp_matrix //= base
            if np.all(temp_matrix == 0):
                break
    else:
        # Standard Integer Mode (Permutation Sum)
        num_planes = int(np.floor(np.log(max_val) / np.log(base))) + 1
        temp_matrix = matrix.copy().astype(np.int64)

        for d in range(num_planes):
            unit_weight = float(base ** d)
            digit_plane = temp_matrix % base
            if np.any(digit_plane > 0):
                planes.append((unit_weight, digit_plane.astype(np.float64)))
            temp_matrix //= base
            if np.all(temp_matrix == 0):
                break

    all_components: List[RadixComponent] = []
    max_plane_runtime = 0.0
    import time

    # Sequential hardware simulation
    for weight, plane_matrix in planes:
        try:
            t_start = time.perf_counter()
            comps = _decompose_digit_plane(
                plane_matrix,
                weight,
                step_strategy,
                matching_method
            )
            plane_runtime = time.perf_counter() - t_start
            
            # Simulated hardware runtime is the max of any independent plane
            max_plane_runtime = max(max_plane_runtime, plane_runtime)
            all_components.extend(comps)
        except Exception as exc:
            print(f"Radix plane sequential worker failed: {exc}")

    return all_components, max_plane_runtime, len(planes)


def _decompose_digit_plane(
        plane: np.ndarray,
        unit_weight: float,
        strategy: str = "min",
        matching_method: str = "heavy"
) -> List[RadixComponent]:
    # Work on a float copy for precision
    x = plane.astype(np.float64)
    components: List[RadixComponent] = []
    tol = 1e-9

    # Map strategy string → int for JIT functions (0=min, 1=median, 2=max)
    _strategy_map = {"min": 0, "median": 1, "max": 2}
    strategy_int = _strategy_map.get(strategy, 0)

    # JIT fast path: covers heavy / heavy_static / wfa for all three strategies
    if matching_method in ["heavy", "heavy_static", "heavy_noaug", "heavy_static_noaug"]:
        if matching_method == "heavy":
            weights, all_matches_list = _jit_decompose_sorted_dynamic(
                x, strategy_int=strategy_int, tol=tol)
        elif matching_method == "heavy_static":
            weights, all_matches_list = _jit_decompose_sorted_static(
                x, strategy_int=strategy_int, tol=tol)
        elif matching_method == "heavy_noaug":
            weights, all_matches_list = _jit_decompose_sorted_dynamic_noaug(x, tol=tol)
        else:  # heavy_static_noaug
            weights, all_matches_list = _jit_decompose_sorted_static_noaug(x, tol=tol)

        for w, matches in zip(weights, all_matches_list):
            actual_weight = w * unit_weight
            p = np.zeros_like(x)
            for (i, j) in matches:
                p[i, j] = 1.0
            components.append(RadixComponent(matrix=actual_weight * p, weight=actual_weight))
        return components

    elif matching_method == "wfa":
        n = x.shape[0]
        weights, all_matches_list = _jit_decompose_wfa(
            x, n, tol=tol, strategy_int=strategy_int)
        for w, matches in zip(weights, all_matches_list):
            actual_weight = w * unit_weight
            p = np.zeros_like(x)
            for (i, j) in matches:
                p[i, j] = 1.0
            components.append(RadixComponent(matrix=actual_weight * p, weight=actual_weight))
        return components

    # Slow path: Python loop for Hungarian (maximum/minimum) which has no JIT version
    match_func = MATCHING_ALGORITHMS.get(matching_method, sorted_array_matching)
    epsilon = 1e-9

    while np.any(x > epsilon):
        matches = match_func(x)
        if not matches:
            break

        match_values = [x[i, j] for (i, j) in matches]

        if strategy == "max":
            digit_step = max(match_values)
        elif strategy == "median":
            digit_step = float(np.median(match_values))
        else:
            digit_step = min(match_values)

        if digit_step < 1e-12:
            break

        actual_weight = digit_step * unit_weight
        rows_arr, cols_arr = zip(*matches)
        rows_arr = np.array(rows_arr)
        cols_arr = np.array(cols_arr)
        p = np.zeros_like(x)
        p[rows_arr, cols_arr] = 1.0
        x[rows_arr, cols_arr] = np.maximum(0.0, x[rows_arr, cols_arr] - digit_step)
        components.append(RadixComponent(matrix=actual_weight * p, weight=actual_weight))

    return components