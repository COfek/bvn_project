from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Local imports - ensuring paths match your project structure
from .sorted_array_matching import sorted_array_matching, _jit_decompose_sorted
from .wfa import wavefront_matching_vectorized, _jit_decompose_wfa


@dataclass
class RadixComponent:
    matrix: np.ndarray
    weight: float


def maximum_matching_wrapper(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Wrapper for Scipy's Hungarian algorithm.
    Finds the maximum weight matching.
    """
    # Use maximize=True for direct maximization
    row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)

    # Filter out zero-weight matches to ensure we only return valid edges
    matches = []
    for r, c in zip(row_ind, col_ind):
        if matrix[r, c] > 0:
            matches.append((r, c))
    return matches


# Global registry of available matching algorithms
MATCHING_ALGORITHMS: Dict[str, Callable[[np.ndarray], List[Tuple[int, int]]]] = {
    "heavy": sorted_array_matching,
    "wfa": wavefront_matching_vectorized,
    "maximum": maximum_matching_wrapper
}


def decompose_radix(
        matrix: np.ndarray,
        base: int = 3,
        max_workers: int | None = None,
        step_strategy: str = "min",
        matching_method: str = "heavy"
) -> List[RadixComponent]:
    """
    Decomposes an integer matrix into weighted permutations using a Radix approach.

    Args:
        matrix: The K-regular integer matrix.
        base: The radix base (e.g., 2 for bitplane).
        max_workers: Number of parallel processes for plane decomposition.
        step_strategy: "min", "max", or "median" for weight selection.
        matching_method: "heavy", "wfa", or "maximum".
    """
    max_val = np.max(matrix)
    if max_val == 0:
        return []

    # Calculate depth based on actual integer values
    num_planes = int(np.floor(np.log(max_val) / np.log(base))) + 1

    planes: List[Tuple[float, np.ndarray]] = []
    temp_matrix = matrix.copy().astype(np.int64)

    # Extract digit planes
    for d in range(num_planes):
        unit_weight = float(base ** d)
        digit_plane = temp_matrix % base
        if np.any(digit_plane > 0):
            planes.append((unit_weight, digit_plane.astype(np.float64)))
        temp_matrix //= base
        if np.all(temp_matrix == 0):
            break

    all_components: List[RadixComponent] = []

    # Parallelize decomposition of planes
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_plane = {
            executor.submit(
                _decompose_digit_plane,
                plane_matrix,
                weight,
                step_strategy,
                matching_method
            ): weight
            for weight, plane_matrix in planes
        }

        for future in as_completed(future_to_plane):
            try:
                all_components.extend(future.result())
            except Exception as exc:
                # Thread errors are easier to catch and debug than process crashes
                print(f"Radix plane thread worker failed: {exc}")

    return all_components


def _decompose_digit_plane(
        plane: np.ndarray,
        unit_weight: float,
        strategy: str = "min",
        matching_method: str = "heavy"
) -> List[RadixComponent]:
    # Work on a float copy for precision
    x = plane.copy().astype(np.float64)
    components: List[RadixComponent] = []

    # Fast path: Use JIT-compiled decomposition if possible (Releases GIL)
    # Only for "min" strategy which is the standard decomposition
    if strategy == "min":
        # Check matching method
        if matching_method == "heavy":
            # Using Sorted Array Matching (JIT)
            weights, all_matches_list = _jit_decompose_sorted(x, tol=1e-9)
            
            for w, matches in zip(weights, all_matches_list):
                actual_weight = w * unit_weight
                
                # Reconstruct generic permutation matrix (sparse-ish)
                # We could optimize this by storing matching directly, 
                # but RadixComponent expects a matrix.
                p = np.zeros_like(x)
                for (i, j) in matches:
                    p[i, j] = 1.0
                    
                components.append(RadixComponent(matrix=actual_weight * p, weight=actual_weight))
                
            return components
            
        elif matching_method == "wfa":
             # Using Wavefront Matching (JIT)
            n = x.shape[0]
            weights, all_matches_list = _jit_decompose_wfa(x, n, tol=1e-9)
            
            for w, matches in zip(weights, all_matches_list):
                actual_weight = w * unit_weight
                p = np.zeros_like(x)
                for (i, j) in matches:
                    p[i, j] = 1.0
                components.append(RadixComponent(matrix=actual_weight * p, weight=actual_weight))
                
            return components

    # Slow path: Python loop (Holds GIL)
    match_func = MATCHING_ALGORITHMS.get(matching_method, sorted_array_matching)

    # Use a small epsilon to prevent infinite loops from float drift
    epsilon = 1e-9

    while np.any(x > epsilon):
        # Now passing numeric x so Maximum Matching and Heavy Node work correctly
        matches = match_func(x)

        if not matches:
            break

        match_values = [x[i, j] for (i, j) in matches]

        if strategy == "max":
            digit_step = max(match_values)
        elif strategy == "median":
            digit_step = np.median(match_values)
        else:  # "min" strategy (Standard BVN-style)
            digit_step = min(match_values)

        # Safety: if the step is too small, we stop to avoid infinite cycles
        if digit_step < 1e-12:
            break

        actual_weight = digit_step * unit_weight

        # Build the weighted permutation matrix
        p = np.zeros_like(x)
        for (i, j) in matches:
            p[i, j] = 1.0
            # Update working matrix: Clip to zero if we subtract more than exists
            # This is essential for 'max' or 'median' strategies
            x[i, j] = max(0.0, x[i, j] - digit_step)

        components.append(RadixComponent(matrix=actual_weight * p, weight=actual_weight))

    return components