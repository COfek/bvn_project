from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from .sorted_array_matching import sorted_array_matching
from .wfa import wavefront_matching_vectorized

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
    max_iters: Optional[int] = None,
) -> List[DecompositionComponent]:
    """
    Perform an exact Birkhoff–von Neumann decomposition on a K-regular integer matrix.

    Because the input is a sum of permutations, this algorithm is guaranteed
    to terminate when the residual matrix is exactly zero.
    """
    # Work on a copy to avoid modifying the original matrix
    work = np.array(matrix, dtype=np.float64, copy=True)
    components: List[DecompositionComponent] = []
    iteration = 0

    # In integer space, we continue until the matrix is purely zero
    while np.any(work > 0):
        iteration += 1

        # We use -work as cost to find the "heaviest" matching,
        # ensuring we pick edges that actually exist in the K-regular graph.
        cost = -work
        row_ind, col_ind = linear_sum_assignment(cost)

        # Extract the minimum value along the found permutation
        selected_values = work[row_ind, col_ind]
        lambda_value = float(np.min(selected_values))

        # If we can't find a matching with weight > 0, the matrix is decomposed
        if lambda_value <= 0:
            break

        # Record the permutation
        permutation = np.zeros_like(work)
        permutation[row_ind, col_ind] = 1.0

        components.append(DecompositionComponent(permutation=permutation, weight=lambda_value))

        # Subtract the weight from the working matrix
        work[row_ind, col_ind] -= lambda_value

        if max_iters is not None and iteration >= max_iters:
            break

    return components