from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

FloatMatrix = NDArray[np.float64]


@dataclass
class BvnComponent:
    """
    Single component in a BVN-style decomposition.

    Attributes:
        permutation: Permutation matrix (0/1 entries).
        weight: Weight lambda associated with this permutation.
    """

    permutation: FloatMatrix
    weight: float


def bvn_upper_bound(n: int) -> int:
    """
    Theoretical upper bound on the number of permutations in BVN decomposition:

        n^2 - 2n + 2

    Args:
        n: Matrix dimension.

    Returns:
        Upper bound value.
    """
    return n * n - 2 * n + 2


def bvn_decomposition(
    matrix: FloatMatrix,
    tol: float = 1e-10,
    max_iters: Optional[int] = None,
) -> List[BvnComponent]:
    """
    Perform a Birkhoff–von Neumann decomposition using the Hungarian algorithm.

    Algorithm:
      - At each iteration, find a permutation (perfect matching) using
        linear_sum_assignment on a cost matrix derived from 'matrix'.
      - Let lambda be the minimum entry along that permutation.
      - Subtract lambda from those entries and record (P, lambda).
      - Stop when the matrix is (approximately) zero.

    Args:
        matrix: Doubly stochastic matrix of shape (n, n).
        tol: Numerical tolerance for stopping.
        max_iters: Optional maximum number of iterations (safety cap).

    Returns:
        List of BvnComponent entries.
    """
    work = np.array(matrix, dtype=float, copy=True)
    components: List[BvnComponent] = []
    iteration = 0

    while True:
        iteration += 1

        if np.all(work < tol):
            break

        cost = -work
        row_ind, col_ind = linear_sum_assignment(cost)

        permutation = np.zeros_like(work)
        permutation[row_ind, col_ind] = 1.0

        selected_values = work[row_ind, col_ind]
        lambda_value = float(np.min(selected_values))

        if lambda_value < tol:
            break

        components.append(BvnComponent(permutation=permutation, weight=lambda_value))
        work[row_ind, col_ind] -= lambda_value

        if max_iters is not None and iteration >= max_iters:
            break

    return components
