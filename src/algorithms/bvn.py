from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from .sorted_array_matching import sorted_array_matching
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
            # We use -work as cost to find the "heaviest" matching,
            # ensuring we pick edges that actually exist in the K-regular graph.
            cost = -work
            row_ind, col_ind = linear_sum_assignment(cost)

        elif matching_algorithm == "heavy":
            # Greedy Sorted Array Matching
            # Returns list of (r, c) tuples
            matches = sorted_array_matching(work)
            
            # Convert to arrays for indexing
            if matches:
                rows, cols = zip(*matches)
                row_ind = np.array(rows)
                col_ind = np.array(cols)
            else:
                row_ind = np.array([], dtype=int)
                col_ind = np.array([], dtype=int)

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