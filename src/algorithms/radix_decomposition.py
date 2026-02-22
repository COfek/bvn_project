from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Local imports - ensuring paths match your project structure
from .sorted_array_matching import sorted_array_matching, _jit_decompose_sorted_static, _jit_decompose_sorted_dynamic
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
        float_precision_bits: Bits of precision for Sinkhorn/Float matrices.
        normalize_input: If True, scale float matrix to int space (Sinkhorn mode).
    """
    max_val = np.max(matrix)
    if max_val == 0:
        return []

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

    # Parallelize decomposition of planes
    if max_workers == 1:
        # Sequential path: completely bypass ThreadPool overhead
        for weight, plane_matrix in planes:
            try:
                comps = _decompose_digit_plane(
                    plane_matrix,
                    weight,
                    step_strategy,
                    matching_method
                )
                all_components.extend(comps)
            except Exception as exc:
                print(f"Radix plane sequential worker failed: {exc}")
    else:
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
                    print(f"Radix plane thread worker failed: {exc}")

    return all_components


def _decompose_digit_plane(
        plane: np.ndarray,
        unit_weight: float,
        strategy: str = "min",
        matching_method: str = "heavy"
) -> List[RadixComponent]:
    # Work on a float copy for precision
    x = plane.astype(np.float64) # Ensure float for clipping
    components: List[RadixComponent] = []
    iterations = 0
    tol = 1e-9 # Tolerance for checking if matrix elements are effectively zero

    # Fast path: Use JIT-compiled decomposition if possible (Releases GIL)
    # Only for "min" strategy which is the standard decomposition
    if strategy == "min":
        # Check matching method
        if matching_method in ["heavy", "heavy_static"]:
            if matching_method == "heavy":
                weights, all_matches_list = _jit_decompose_sorted_dynamic(x, tol=tol)
            else:
                weights, all_matches_list = _jit_decompose_sorted_static(x, tol=tol)
            
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
        
        # Optimize: Convert matches to arrays for vectorized update
        # matches is list of (r, c)
        if len(matches) > 0:
            rows, cols = zip(*matches)
            rows = np.array(rows)
            cols = np.array(cols)
            
            p = np.zeros_like(x)
            p[rows, cols] = 1.0
            
            if strategy == "max":
                # For max strategy, we subtract digit_step but clip at 0
                # We only need to update the matched indices
                current_vals = x[rows, cols]
                # If using 'max', digit_step = max(current_vals). 
                # So we subtract max. Some entries might go negative without clipping.
                x[rows, cols] = np.maximum(0.0, current_vals - digit_step)
            else:
                 # For min strategy, digit_step <= all current_vals, so simple subtraction is safe
                 # usually. But let's use maximum(0) to be safe against float drift.
                 x[rows, cols] = np.maximum(0.0, x[rows, cols] - digit_step)
        
        components.append(RadixComponent(matrix=actual_weight * p, weight=actual_weight))

    return components