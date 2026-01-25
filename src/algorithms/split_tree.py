from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

float_matrix = NDArray[np.float64]
pschedule = Union[float, Callable[[float_matrix, int], float]]


@dataclass
class split_tree_component:
    matrix: float_matrix
    weight: float


def random_binary_split(x: float_matrix, p: float) -> Tuple[float_matrix, float_matrix]:
    """
    OPTIMIZED: Vectorized split replacing the slow Python for-loop.
    This ensures the 'Split Phase' doesn't become the new bottleneck.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")

    # Use NumPy masks for near-instantaneous splitting across 1M+ elements
    mask_a = (np.random.rand(*x.shape) < p) & (x > 0)

    a = np.where(mask_a, x, 0.0)
    b = x - a
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
    Recursively split a nonnegative matrix into a tree of submatrices.
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
        max_workers: int | None = None,
) -> List[split_tree_component]:
    """
    High-level wrapper for Parallel Split-Tree Decomposition.
    
    This function performs a recursive splitting of the input matrix into sparser sub-matrices (leaves),
    and then decomposes each leaf in PARALLEL using the JIT-optimized WFA engine.
    
    Args:
        x (float_matrix): The input N x N doubly stochastic matrix.
        sparsity_target (int): Target number of non-zeros for leaf nodes (stop splitting).
        max_depth (int): Maximum recursion depth for splitting.
        p_schedule (pschedule): Probability schedule for random splitting.
        split_method (str): Strategy for splitting ('random' or 'pivot').
        tol (float): Numerical tolerance.
        max_workers (int | None): Number of threads for parallel leaf decomposition.
                                  If None, defaults to os.cpu_count().
                                  
    Returns:
        List[split_tree_component]: A list of all permutation matrices and weights found.
    """
    # 1. Recursive splitting (now vectorized)
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

    # 2. Parallel Decomposition of leaves using a ThreadPoolExecutor
    # With max_depth=1, this will launch 2 concurrent workers.
    # The GIL is released inside 'decompose_leaf_with_wfa' -> '_jit_decompose_wfa', 
    # enabling true parallelism.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_leaf = {
            executor.submit(decompose_leaf_with_wfa, leaf, tol=tol): leaf
            for leaf in leaves
        }

        for future in as_completed(future_to_leaf):
            try:
                leaf_components = future.result()
                all_components.extend(leaf_components)
            except Exception as exc:
                print(f"Leaf decomposition generated an exception: {exc}")

    return all_components


def decompose_leaf_with_wfa(
        leaf: float_matrix,
        tol: float = 1e-12,
) -> List[split_tree_component]:
    """
    Decompose leaf matrix using WFA.
    OPTIMIZED: Uses Full Loop JIT to release GIL and enable true parallelism.
    """
    # 1. Prepare data for JIT
    # We must pass a copy because the JIT kernel modifies it in-place
    x = leaf.copy().astype(np.float64) 
    n = x.shape[0]
    
    # 2. Call JIT Kernel (releases GIL)
    # Returns: weights (List[float]), matches (List[List[Tuple]])
    from .wfa import _jit_decompose_wfa
    
    weights, matches_list = _jit_decompose_wfa(x, n, tol)
    
    # 3. Reconstruct Objects (Fast Python loop, GIL re-acquired)
    components: List[split_tree_component] = []
    
    for k in range(len(weights)):
        w = weights[k]
        match_indices = matches_list[k]
        
        # We need to construct the sparse matrix 'p' * 'w'
        # Since 'p' is a permutation matrix, it's just 1.0 at indices.
        # But split_tree_component expects a full matrix `matrix`.
        # Constructing full NxN arrays here might be slow? 
        # Actually the original code did `p = np.zeros_like(x)` inside the loop.
        # So it's the same cost, just deferred.
        
        # Optimization: We can reconstruct the matrix efficiently
        p_matrix = np.zeros((n, n), dtype=np.float64)
        for r, c in match_indices:
            p_matrix[r, c] = w
            
        components.append(split_tree_component(matrix=p_matrix, weight=w))
        
    return components