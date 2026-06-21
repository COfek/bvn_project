from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from .sorted_array_matching import (
    sorted_array_matching,
    _jit_decompose_sorted_static,
    _jit_decompose_sorted_dynamic,
    _jit_decompose_sorted_dynamic_noaug,
    _jit_decompose_sorted_static_noaug,
)
from .wfa import wavefront_matching_vectorized, _jit_decompose_wfa

FloatMatrix = NDArray[np.float64]


@dataclass
class DecompositionComponent:
    """
    Single component in a decomposition.
    """
    permutation: FloatMatrix | Dict[int, int]
    weight: float


def bvn_decomposition(
    matrix: FloatMatrix,
    matching_algorithm: str = "maximum",
    max_iters: Optional[int] = None,
) -> List[DecompositionComponent]:
    """
    Perform an exact Birkhoff–von Neumann decomposition on a K-regular integer matrix.

    Args:
        matrix: The input matrix (must be Doubly Stochastic / K-regular).
        matching_algorithm: 'maximum' (Hungarian), 'heavy' (Greedy), or 'wfa' (Wavefront).
        max_iters: Optional limit on iterations.

    Because the input is a sum of permutations, this algorithm is guaranteed
    to terminate when the residual matrix is exactly zero.
    """
    # Work on a copy to avoid modifying the original matrix
    work = np.array(matrix, dtype=np.float64, copy=True)
    components: List[DecompositionComponent] = []
    iteration = 0

    n = work.shape[0]

    # In integer space, we continue until the matrix is purely zero
    while np.any(work > 1e-9):
        iteration += 1

        row_ind, col_ind = None, None

        # Select matching algorithm
        if matching_algorithm == "maximum":
            # Only consider rows that still have at least one positive entry.
            # Penalty cost forces the Hungarian to avoid zero-valued columns:
            # zero entries get a large positive cost (= avoid in minimisation
            # of -weight), so every active row gets assigned to a positive cell.
            tol = 1e-9
            active_rows = np.where(work.max(axis=1) > tol)[0]
            if len(active_rows) == 0:
                break
            sub = work[np.ix_(active_rows, np.arange(n))]
            big = float(sub.max() + 1.0) * n   # penalty > any real weight
            cost = np.where(sub > tol, -sub, big)  # minimise → prefer large weights
            sub_row_ind, col_ind_sub = linear_sum_assignment(cost)
            row_ind = active_rows[sub_row_ind]
            col_ind = col_ind_sub

        elif matching_algorithm == "minimum":
            # Minimum-weight perfect matching over positive edges only.
            # Zero entries get a large finite penalty so the problem stays
            # feasible; any zero-weight matches are filtered naturally by the
            # lambda_value <= 1e-12 check below.
            tol = 1e-9
            active_rows = np.where(work.max(axis=1) > tol)[0]
            if len(active_rows) == 0:
                break
            sub = work[np.ix_(active_rows, np.arange(n))]
            big = float(np.max(sub) * 1e6 + 1.0)
            cost = np.where(sub > tol, sub, big)
            sub_row_ind, col_ind_sub = linear_sum_assignment(cost)
            row_ind = active_rows[sub_row_ind]
            col_ind = col_ind_sub

        elif matching_algorithm == "heavy_noaug":
            # GW Dynamic without augmenting paths — weak (partial-matching) decomposition
            weights, all_matches = _jit_decompose_sorted_dynamic_noaug(work, 1e-9)

            for w, matches in zip(weights, all_matches):
                if w <= 1e-12:
                    continue
                permutation = np.zeros_like(work)
                if matches:
                    rows, cols = zip(*matches)
                    permutation[rows, cols] = 1.0
                components.append(DecompositionComponent(permutation=permutation, weight=w))
            return components

        elif matching_algorithm == "heavy_static_noaug":
            # GW Static without augmenting paths — weak (partial-matching) decomposition
            weights, all_matches = _jit_decompose_sorted_static_noaug(work, 1e-9)

            for w, matches in zip(weights, all_matches):
                if w <= 1e-12:
                    continue
                permutation = np.zeros_like(work)
                if matches:
                    rows, cols = zip(*matches)
                    permutation[rows, cols] = 1.0
                components.append(DecompositionComponent(permutation=permutation, weight=w))
            return components

        elif matching_algorithm == "heavy":
            # Optimization: Use JIT-compiled full decomposition for dynamic Heavy
            # This bypasses the Python loop for finding matches and updating the matrix,
            # making it consistent with how WFA and heavy_static are handled.
            weights, all_matches = _jit_decompose_sorted_dynamic(work, 1e-9)

            for w, matches in zip(weights, all_matches):
                if w <= 1e-12:
                    continue

                permutation = np.zeros_like(work)
                if matches:
                    rows, cols = zip(*matches)
                    permutation[rows, cols] = 1.0

                components.append(DecompositionComponent(permutation=permutation, weight=w))

            return components

        elif matching_algorithm == "heavy_static":
            # Optimization: Use JIT-compiled full decomposition for static Heavy
            # This bypasses the Python loop for finding matches and updating the matrix
            weights, all_matches = _jit_decompose_sorted_static(work, 1e-9)
            
            # Convert to DecompositionComponent objects
            for w, matches in zip(weights, all_matches):
                if w <= 1e-12: continue
                
                permutation = np.zeros_like(work)
                if matches:
                    rows, cols = zip(*matches)
                    permutation[rows, cols] = 1.0
                
                components.append(DecompositionComponent(permutation=permutation, weight=w))
                
            # The matrix is fully decomposed by _jit_decompose_sorted_static
            return components

        elif matching_algorithm == "wfa":
            # Optimization: Use JIT-compiled full decomposition for WFA
            # This bypasses the Python loop for finding matches and updating the matrix
            weights, all_matches = _jit_decompose_wfa(work, n, 1e-9)
            
            # Convert to DecompositionComponent objects
            for w, matches in zip(weights, all_matches):
                if w <= 1e-12: continue
                
                permutation = np.zeros_like(work)
                rows, cols = zip(*matches)
                permutation[rows, cols] = 1.0
                
                components.append(DecompositionComponent(permutation=permutation, weight=w))
                
            # The matrix is fully decomposed by _jit_decompose_wfa
            return components
        
        else:
            raise ValueError(f"Unknown matching algorithm: {matching_algorithm}")

        # If we failed to find any matching (should not happen for valid DS matrices with Max Matching,
        # but Greedy/WFA might not find a perfect matching immediately, though they should find *something*)
        if row_ind is None or len(row_ind) == 0:
            # If the matrix is not zero but we found no matches, we are stuck.
            # This can happen if the algorithm is not guaranteed to find a perfect matching
            # and the residual graph has no edges compatible with the heuristic?
            # Actually for K-regular graphs, a perfect matching always exists.
            # But greedy heuristics might not find it. 
            # However, we are just peeling *a* matching, not necessarily a perfect one?
            # Standard BVN peels a permutation (perfect matching).
            # If we peel a partial matching, is it still BVN?
            # Yes, as long as we subtract min weight.
            break

        # Extract the minimum value along the found permutation
        selected_values = work[row_ind, col_ind]
        if len(selected_values) == 0:
             break

        lambda_value = float(np.min(selected_values))

        # If we can't find a matching with weight > 0, the matrix is decomposed
        if lambda_value <= 1e-12:
            break

        # Record the permutation
        permutation = np.zeros_like(work)
        permutation[row_ind, col_ind] = 1.0

        components.append(DecompositionComponent(permutation=permutation, weight=lambda_value))

        # Subtract the weight from the working matrix
        work[row_ind, col_ind] -= lambda_value
        
        # Clean up small residuals
        work[work < 1e-12] = 0.0

        if max_iters is not None and iteration >= max_iters:
            break

    return components
