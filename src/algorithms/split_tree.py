from __future__ import annotations

from typing import Callable, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from .wfa import wavefront_matching  # use your existing WFA implementation

FloatMatrix = NDArray[np.float64]
PSchedule = Union[float, Callable[[FloatMatrix, int], float]]


@dataclass
class SplitTreeComponent:
    matrix: FloatMatrix
    weight: float


def random_binary_split(x: FloatMatrix, p: float) -> Tuple[FloatMatrix, FloatMatrix]:
    """
    Randomly split non-zero entries of X into two matrices A and B.

    Each positive entry X[i, j] goes entirely to:
        - A[i, j] with probability p
        - B[i, j] with probability (1 - p)

    Zeros remain zero in both A and B.

    This guarantees:
        X = A + B  (exactly)
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")

    a = np.zeros_like(x)
    b = np.zeros_like(x)

    rows, cols = np.nonzero(x)
    if len(rows) == 0:
        return a, b

    rand_vals = np.random.rand(len(rows))

    for idx, (i, j) in enumerate(zip(rows, cols)):
        if rand_vals[idx] < p:
            a[i, j] = x[i, j]
        else:
            b[i, j] = x[i, j]

    return a, b


def split_tree(
    x: FloatMatrix,
    sparsity_target: int,
    max_depth: int,
    p_schedule: PSchedule,
    depth: int = 0,
) -> List[FloatMatrix]:
    """
    Recursively split X into a tree of matrices until each leaf is "sufficiently sparse"
    or the maximum depth is reached.

    Args:
        x:
            The matrix to split (FloatMatrix, e.g. doubly stochastic or general non-negative).
        sparsity_target:
            Stop splitting when nnz(X) <= sparsity_target.
        max_depth:
            Maximum recursion depth allowed.
        p_schedule:
            Either:
              - a float p in (0,1), used at all levels, or
              - a callable p_schedule(X, depth) -> float in (0,1),
                allowing depth- or structure-dependent probabilities.
        depth:
            Current recursion depth (internal use).

    Returns:
        A list of leaf matrices [L_1, L_2, ..., L_k] such that:

            X = sum_i L_i

        (up to floating-point arithmetic).
    """
    # Count nonzero entries
    nnz = int(np.count_nonzero(x))

    # Stopping conditions
    if nnz == 0:
        return []
    if nnz <= sparsity_target or depth >= max_depth:
        return [x]

    # Choose p for this node
    if callable(p_schedule):
        p = float(p_schedule(x, depth))
    else:
        p = float(p_schedule)

    if not (0.0 < p < 1.0):
        raise ValueError(f"p_schedule produced invalid p={p} at depth={depth}")

    # Perform one binary split
    a, b = random_binary_split(x, p)

    leaves: List[FloatMatrix] = []

    if np.count_nonzero(a) > 0:
        leaves.extend(split_tree(a, sparsity_target, max_depth, p_schedule, depth + 1))
    if np.count_nonzero(b) > 0:
        leaves.extend(split_tree(b, sparsity_target, max_depth, p_schedule, depth + 1))

    return leaves


def verify_reconstruction(original: FloatMatrix, leaves: List[FloatMatrix], tol: float = 1e-12) -> bool:
    """
    Verify that the sum of leaf matrices reconstructs the original matrix.

    Returns True if:
        || original - sum(leaves) ||_max <= tol

    This is useful for debugging and validating probability schedules.
    """
    if len(leaves) == 0:
        return np.allclose(original, 0, atol=tol)

    s = np.zeros_like(original)
    for L in leaves:
        s += L

    return np.allclose(original, s, atol=tol)


def split_tree_decomposition(
    x: FloatMatrix,
    sparsity_target: int,
    max_depth: int,
    p_schedule: PSchedule,
    tol: float = 1e-12,
) -> List[SplitTreeComponent]:
    """
    High-level wrapper that:
      1. Runs the recursive split-tree algorithm to produce leaf matrices.
      2. For each leaf, runs a WFA-based decomposition into weighted
         permutation matrices (using your wavefront_matching).
      3. Flattens all leaf decompositions into a single list of components.

    The resulting components satisfy (up to numerical error):

        X ≈ sum_k (component_k.matrix)

    and each component_k has:
        - matrix: lambda_k * P_k  (weighted permutation matrix)
        - weight: lambda_k
    """
    # 1. Split into leaves
    leaves = split_tree(
        x,
        sparsity_target=sparsity_target,
        max_depth=max_depth,
        p_schedule=p_schedule,
    )

    all_components: List[SplitTreeComponent] = []

    # 2. Decompose each leaf into weighted permutations using WFA
    for leaf in leaves:
        leaf_components = decompose_leaf_with_wfa(leaf, tol=tol)
        all_components.extend(leaf_components)

    return all_components


def decompose_leaf_with_wfa(
    leaf: FloatMatrix,
    tol: float = 1e-12,
) -> List[SplitTreeComponent]:
    """
    Decompose a non-negative leaf matrix into weighted permutation matrices
    using your normal WFA (wavefront_matching) as the matching oracle.

    We always keep the *full n×n shape* for X, but when we construct the
    mask for WFA we may have different numbers of active rows vs active
    columns. Since WFA is defined for *square* switch fabrics, we:

      - identify active rows and cols
      - take m = min(#active_rows, #active_cols)
      - restrict to the first m rows and m cols (forming an m×m mask)
      - call wavefront_matching on that square mask
      - map the matches back to the original coordinates

    Repeatedly:
      1. Build boolean mask of entries > tol.
      2. Take a square active Sub-mask and run WFA to get a matching.
      3. Take lambda = min(leaf[i, j]) over matching edges.
      4. Subtract lambda * P from X.
      5. Store matrix = lambda * P, weight = lambda.

    Stops when all remaining entries are <= tol or no matching is possible.
    """
    x = leaf.copy()
    components: List[SplitTreeComponent] = []

    while True:
        mask = x > tol
        if not mask.any():
            break

        # Identify rows/cols that still have positive entries
        active_rows = np.where(mask.sum(axis=1) > 0)[0]
        active_cols = np.where(mask.sum(axis=0) > 0)[0]

        if len(active_rows) == 0 or len(active_cols) == 0:
            break

        # --- Force the reduced mask to be square for WFA ---
        r = len(active_rows)
        c = len(active_cols)
        m = min(r, c)
        if m == 0:
            break

        rows_sq = active_rows[:m]
        cols_sq = active_cols[:m]

        mask_sq = mask[np.ix_(rows_sq, cols_sq)]

        # Run your square WFA on the m×m mask
        matches_sq = wavefront_matching(mask_sq)
        if not matches_sq:
            # No matching found on this square submatrix – stop for this leaf
            break

        # Map matches back to full coordinates
        matches = [(rows_sq[i], cols_sq[j]) for (i, j) in matches_sq]

        # Compute lambda = minimum value over the matched edges
        lam = min(float(x[i, j]) for (i, j) in matches)

        # Build permutation matrix P in FULL space and subtract lam*P from X
        p = np.zeros_like(x)
        for (i, j) in matches:
            p[i, j] = 1.0
            x[i, j] -= lam
            if x[i, j] < tol:
                x[i, j] = 0.0

        comp_matrix = lam * p
        components.append(SplitTreeComponent(matrix=comp_matrix, weight=lam))

    return components
