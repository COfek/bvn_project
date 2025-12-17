from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .wfa import select_matching  # use your existing WFA implementation

float_matrix = NDArray[np.float64]
pschedule = Union[float, Callable[[float_matrix, int], float]]


@dataclass
class split_tree_component:
    matrix: float_matrix
    weight: float


def random_binary_split(x: float_matrix, p: float) -> Tuple[float_matrix, float_matrix]:
    """
    Randomly split non-zero entries of x into two matrices a and b.

    Each positive entry x[i, j] goes entirely to:
        - a[i, j] with probability p
        - b[i, j] with probability (1 - p)

    Zeros remain zero in both a and b.

    This guarantees:
        x = a + b  (exactly)
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


def pivot_split(
    x: float_matrix,
    *,
    min_matching_frac: float = 0.8,
    cv_threshold: float = 0.15,
    tol: float = 1e-12,
) -> Optional[Tuple[float_matrix, float_matrix]]:
    """
    Split matrix x into two matrices based on a value pivot (median),
    while preserving matchability.

    Returns:
        (a, b) such that x = a + b
        or None if splitting is deemed unhelpful or unsafe.
    """
    values = x[x > tol]
    if values.size == 0:
        return None

    mean = float(values.mean())
    std = float(values.std())
    if mean <= tol:
        return None

    cv = std / mean
    if cv <= cv_threshold:
        return None

    pivot = float(np.median(values))

    a = np.zeros_like(x)
    b = np.zeros_like(x)

    high_mask = x >= pivot
    low_mask = (x > tol) & (x < pivot)

    a[high_mask] = x[high_mask]
    b[low_mask] = x[low_mask]

    if not a.any() or not b.any():
        return None

    support = high_mask.astype(np.int8)
    row_deg = support.sum(axis=1)
    col_deg = support.sum(axis=0)

    active_rows = int(np.sum(row_deg > 0))
    active_cols = int(np.sum(col_deg > 0))
    if active_rows == 0 or active_cols == 0:
        return None

    max_possible = min(active_rows, active_cols)
    if max_possible < min_matching_frac * min(x.shape):
        return None

    return a, b


def split_tree(
    x: float_matrix,
    sparsity_target: int,
    max_depth: int,
    p_schedule: pschedule,
    *,
    split_method: str,
    cv_threshold: float,
    min_matching_frac: float,
    tol: float,
    depth: int = 0,
) -> List[float_matrix]:
    """
    Recursively split a nonnegative matrix into a tree of submatrices until each
    leaf is sufficiently sparse, value-homogeneous, or the maximum depth is reached.
    """
    nnz = int(np.count_nonzero(x))

    if nnz == 0:
        return []
    if nnz <= sparsity_target or depth >= max_depth:
        return [x]

    if split_method == "pivot":
        res = pivot_split(
            x,
            cv_threshold=cv_threshold,
            min_matching_frac=min_matching_frac,
            tol=tol,
        )
        if res is None:
            return [x]
        a, b = res

    elif split_method == "random":
        if callable(p_schedule):
            p = float(p_schedule(x, depth))
        else:
            p = float(p_schedule)

        if not (0.0 < p < 1.0):
            raise ValueError(f"p_schedule produced invalid p={p} at depth={depth}")

        a, b = random_binary_split(x, p)

    else:
        raise ValueError(f"unknown split_method: {split_method}")

    leaves: List[float_matrix] = []

    if np.count_nonzero(a) > 0:
        leaves.extend(
            split_tree(
                a,
                sparsity_target,
                max_depth,
                p_schedule,
                split_method=split_method,
                cv_threshold=cv_threshold,
                min_matching_frac=min_matching_frac,
                tol=tol,
                depth=depth + 1,
            )
        )

    if np.count_nonzero(b) > 0:
        leaves.extend(
            split_tree(
                b,
                sparsity_target,
                max_depth,
                p_schedule,
                split_method=split_method,
                cv_threshold=cv_threshold,
                min_matching_frac=min_matching_frac,
                tol=tol,
                depth=depth + 1,
            )
        )

    return leaves


def verify_reconstruction(original: float_matrix, leaves: List[float_matrix], tol: float = 1e-12) -> bool:
    """
    Verify that the sum of leaf matrices reconstructs the original matrix.

    Returns True if:
        || original - sum(leaves) ||_max <= tol
    """
    if len(leaves) == 0:
        return np.allclose(original, 0, atol=tol)

    s = np.zeros_like(original)
    for leaf in leaves:
        s += leaf

    return np.allclose(original, s, atol=tol)


def split_tree_decomposition(
    x: float_matrix,
    sparsity_target: int,
    max_depth: int,
    p_schedule: pschedule,
    split_method: str = "random",
    cv_threshold: float = 0.15,
    min_matching_frac: float = 0.8,
    tol: float = 1e-12,
) -> List[split_tree_component]:
    """
    High-level wrapper:
      1) Split x into leaves using split_tree
      2) Decompose each leaf into weighted (partial) permutation/matching matrices
      3) Flatten into a single component list
    """
    leaves = split_tree(
        x,
        sparsity_target=sparsity_target,
        max_depth=max_depth,
        p_schedule=p_schedule,
        split_method=split_method,
        cv_threshold=cv_threshold,
        min_matching_frac=min_matching_frac,
        tol=tol,
    )

    all_components: List[split_tree_component] = []
    for leaf in leaves:
        leaf_components = decompose_leaf_with_wfa(leaf, tol=tol)
        all_components.extend(leaf_components)

    return all_components


def decompose_leaf_with_wfa(
    leaf: float_matrix,
    tol: float = 1e-12,
) -> List[split_tree_component]:
    """
    Decompose a nonnegative leaf matrix into weighted matching/permutation components
    using your WFA-based matching oracle (select_matching).

    Notes:
      - This produces weighted matchings (not necessarily full permutations) when the
        active support is not a perfect matching.
      - The current implementation crops to an m×m active submask (m=min(r,c)).
        This is consistent with your existing approach but can discard feasible edges.
    """
    x = leaf.copy()
    components: List[split_tree_component] = []

    while True:
        mask = x > tol
        if not mask.any():
            break

        active_rows = np.where(mask.sum(axis=1) > 0)[0]
        active_cols = np.where(mask.sum(axis=0) > 0)[0]

        if len(active_rows) == 0 or len(active_cols) == 0:
            break

        r = len(active_rows)
        c = len(active_cols)
        m = min(r, c)
        if m == 0:
            break

        rows_sq = active_rows[:m]
        cols_sq = active_cols[:m]
        mask_sq = mask[np.ix_(rows_sq, cols_sq)]

        matches_sq, match_type = select_matching(mask_sq)
        if not matches_sq:
            break

        matches = [(rows_sq[i], cols_sq[j]) for (i, j) in matches_sq]

        lam = min(float(x[i, j]) for (i, j) in matches)
        if lam <= tol:
            break

        p = np.zeros_like(x)
        for (i, j) in matches:
            p[i, j] = 1.0
            x[i, j] -= lam
            if x[i, j] <= tol:
                x[i, j] = 0.0

        components.append(split_tree_component(matrix=lam * p, weight=lam))

    return components
