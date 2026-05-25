"""
Euler Splitting decomposition framework for S-regular integer matrices.

Two split strategies are available:

  "euler"  — Classic Euler-tour 2-colouring (Cole & Hopcroft 1982 / Alon 2003).
             Alternates Red/Blue along each Euler circuit → each sub-matrix has
             the same non-zero *positions* as the original, with values halved.
             Guaranteed doubly stochastic in O(N·S) time.

  "greedy" — Greedy integral split (new heuristic).
             Sorts all (i,j) blocks by weight descending and assigns each block
             *entirely* to Red or Blue, tracking row/col budgets of S/2.  A
             second pass fills residual row/col deficits.  Falls back to Euler
             2-colouring only if the greedy cannot complete a valid assignment
             (which happens for uniform matrices with no concentrated entries).
             When it succeeds, the sub-matrices have fewer non-zero entries,
             making subsequent BvN decomposition cheaper per leaf.

Public API
----------
euler_split_once(matrix, split_method)          → (red, blue)
decompose_euler_framework(matrix, ...)          → (components, max_rt, n_leaves)
euler_decomposition(matrix)                     → List[DecompositionComponent]  # legacy
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .bvn import DecompositionComponent, bvn_decomposition
from .sorted_array_matching import (
    sorted_array_matching,
    _jit_decompose_sorted_dynamic,
    _jit_decompose_sorted_static,
)
from .wfa import _jit_decompose_wfa

FloatMatrix = NDArray[np.float64]
AdjType = List[Dict[int, int]]  # adj[u][v] = remaining edge count


# ---------------------------------------------------------------------------
# Internal: Bipartite multigraph builder
# ---------------------------------------------------------------------------

def _build_adj(matrix: np.ndarray) -> AdjType:
    """
    Dict-based adjacency for the bipartite multigraph.
    Row node i → index i; col node j → index n+j.
    adj[u][v] = number of remaining parallel edges between u and v.
    """
    n = matrix.shape[0]
    adj: AdjType = [dict() for _ in range(2 * n)]
    rows, cols = np.nonzero(matrix)
    for i, j in zip(rows.tolist(), cols.tolist()):
        k = int(round(matrix[i, j]))
        if k <= 0:
            continue
        col_node = n + j
        adj[i][col_node] = adj[i].get(col_node, 0) + k
        adj[col_node][i] = adj[col_node].get(i, 0) + k
    return adj


# ---------------------------------------------------------------------------
# Internal: Hierholzer Euler tour
# ---------------------------------------------------------------------------

def _euler_tour_single(adj: AdjType, start: int) -> List[Tuple[int, int]]:
    """Hierholzer on one component. Mutates adj. Returns (u,v) edge list."""
    stack = [start]
    node_path: List[int] = []
    while stack:
        v = stack[-1]
        if adj[v]:
            u = next(iter(adj[v]))
            adj[v][u] -= 1
            if adj[v][u] == 0:
                del adj[v][u]
            adj[u][v] -= 1
            if adj[u][v] == 0:
                del adj[u][v]
            stack.append(u)
        else:
            node_path.append(stack.pop())
    edges: List[Tuple[int, int]] = []
    for k in range(len(node_path) - 1, 0, -1):
        edges.append((node_path[k], node_path[k - 1]))
    return edges


def _all_euler_tours(adj: AdjType) -> List[List[Tuple[int, int]]]:
    components: List[List[Tuple[int, int]]] = []
    for start in range(len(adj)):
        if not adj[start]:
            continue
        edges = _euler_tour_single(adj, start)
        if edges:
            components.append(edges)
    return components


# ---------------------------------------------------------------------------
# Internal: Euler 2-colouring
# ---------------------------------------------------------------------------

def _colour_edges(
    component_edges: List[List[Tuple[int, int]]],
    n: int,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Alternate Red/Blue along each circuit. Returns (red, blue) integer matrices."""
    red  = np.zeros(shape, dtype=np.int64)
    blue = np.zeros(shape, dtype=np.int64)
    for edges in component_edges:
        colour = 0
        for (u, v) in edges:
            if u < n and v >= n:
                row, col = u, v - n
            elif v < n and u >= n:
                row, col = v, u - n
            else:
                continue
            if colour == 0:
                red[row, col]  += 1
            else:
                blue[row, col] += 1
            colour ^= 1
    return red, blue


# ---------------------------------------------------------------------------
# Internal: extract one permutation (for odd S)
# ---------------------------------------------------------------------------

def _extract_one_perm(int_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract one valid permutation from int_matrix.
    Returns (perm, residual) where residual = int_matrix - perm.
    """
    float_mat = int_matrix.astype(np.float64)
    matches = sorted_array_matching(float_mat)
    n = int_matrix.shape[0]
    perm = np.zeros((n, n), dtype=np.int64)
    for r, c in matches:
        perm[r, c] = 1
    residual = (int_matrix - perm).clip(min=0)
    return perm, residual


# ---------------------------------------------------------------------------
# Split strategy A: Euler 2-colouring
# ---------------------------------------------------------------------------

def _euler_split(int_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    One level of Euler-tour 2-colouring.
    Input must be even-regular (all row/col sums equal even S).
    Returns (red, blue) each (S/2)-regular.
    """
    n = int_matrix.shape[0]
    adj = _build_adj(int_matrix)
    component_edges = _all_euler_tours(adj)
    return _colour_edges(component_edges, n, int_matrix.shape)


# ---------------------------------------------------------------------------
# Split strategy B: Greedy integral split
# ---------------------------------------------------------------------------

def _greedy_split(int_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy integral split: tries to assign each block (i,j) entirely to Red
    or Blue to maximize sparsity of the leaves.

    Algorithm:
      Pass 1 — Sort all non-zero (i,j) entries by weight descending.
               Greedily assign each block to Red if row and column budgets allow.
      Pass 2 — For rows whose budget is still > 0, try to pull from partially-
               assigned entries (those where blue[i,j] > 0 and col still has room).
      Fallback — If row/col sums still violate S/2, fall back to Euler 2-colouring.

    Input must be even-regular.  Returns (red, blue) each (S/2)-regular.
    """
    n = int_matrix.shape[0]
    S = int(int_matrix.sum(axis=1).max())
    half = S // 2

    red = np.zeros((n, n), dtype=np.int64)
    row_budget = np.full(n, half, dtype=np.int64)
    col_budget = np.full(n, half, dtype=np.int64)

    # Pass 1: assign whole blocks, largest first
    rows_idx, cols_idx = np.nonzero(int_matrix)
    if len(rows_idx) > 0:
        values = int_matrix[rows_idx, cols_idx]
        order = np.argsort(-values)  # descending
        for idx in order:
            i, j = int(rows_idx[idx]), int(cols_idx[idx])
            assign = int(min(int_matrix[i, j], row_budget[i], col_budget[j]))
            if assign > 0:
                red[i, j] = assign
                row_budget[i] -= assign
                col_budget[j] -= assign

    # Pass 2: fill residual row deficits from still-unassigned capacity
    for i in range(n):
        if row_budget[i] == 0:
            continue
        # Try to take more from entries in this row where blue has excess
        row_entries = np.argsort(-int_matrix[i])  # columns sorted by entry size
        for j in row_entries:
            if int_matrix[i, j] == 0:
                break
            can_add = int(min(
                row_budget[i],
                int_matrix[i, j] - red[i, j],   # what's still in blue for (i,j)
                col_budget[j],
            ))
            if can_add > 0:
                red[i, j] += can_add
                row_budget[i] -= can_add
                col_budget[j] -= can_add
            if row_budget[i] == 0:
                break

    # Validity check — fall back to Euler if greedy failed
    valid = (
        np.all(red.sum(axis=1) == half) and
        np.all(red.sum(axis=0) == half)
    )
    if not valid:
        return _euler_split(int_matrix)

    blue = int_matrix - red
    return red, blue


# ===========================================================================
# Public: one-level split (both strategies)
# ===========================================================================

def euler_split_once(
    matrix: np.ndarray,
    split_method: str = "euler",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split an S-regular integer matrix into two (S/2)-regular integer
    sub-matrices.  Both are doubly stochastic (equal row and column sums).

    Args:
        matrix:       n×n non-negative integer matrix with equal row/col sums S.
        split_method: "euler" (Euler 2-colouring) or "greedy" (integral split).

    Returns:
        (red, blue) — two integer matrices each with row/col sums ⌊S/2⌋.
        When S is odd, one permutation is extracted first and folded into red
        (red has sums ⌈S/2⌉, blue has sums ⌊S/2⌋).
    """
    n = matrix.shape[0]
    int_matrix = np.round(matrix).astype(np.int64)
    S = int(int_matrix.sum(axis=1).max())

    if S == 0:
        z = np.zeros((n, n), dtype=np.int64)
        return z, z.copy()

    extra_perm = None
    if S % 2 == 1:
        extra_perm, int_matrix = _extract_one_perm(int_matrix)

    if split_method == "greedy":
        red, blue = _greedy_split(int_matrix)
    else:  # default: euler
        red, blue = _euler_split(int_matrix)

    if extra_perm is not None:
        red = red + extra_perm

    return red, blue


# ===========================================================================
# Internal: fast leaf BvN (JIT path mirrors radix _decompose_digit_plane)
# ===========================================================================

def _leaf_bvn(
    leaf_float: np.ndarray,
    matching_method: str,
    tol: float = 1e-9,
) -> Tuple[List[DecompositionComponent], float]:
    """
    Run BvN on a single leaf matrix using the JIT-compiled fast path where
    available.  Mirrors the "min" strategy fast path in radix_decomposition so
    that simulated runtimes are on the same footing for fair comparison.

    Returns (components, wall_clock_seconds).
    """
    t0 = time.perf_counter()

    if matching_method in ("heavy", "heavy_static"):
        work = leaf_float.copy()
        if matching_method == "heavy":
            weights, all_matches = _jit_decompose_sorted_dynamic(work, tol=tol)
        else:
            weights, all_matches = _jit_decompose_sorted_static(work, tol=tol)
        comps = _matches_to_components(weights, all_matches, leaf_float)

    elif matching_method == "wfa":
        work = leaf_float.copy()
        weights, all_matches = _jit_decompose_wfa(work, work.shape[0], tol=tol)
        comps = _matches_to_components(weights, all_matches, leaf_float)

    else:
        # maximum / minimum — fall back to scipy Hungarian via Python BvN
        comps = bvn_decomposition(leaf_float.copy(), matching_algorithm=matching_method)

    return comps, time.perf_counter() - t0


def _matches_to_components(
    weights: List[float],
    all_matches: List[List[Tuple[int, int]]],
    template: np.ndarray,
) -> List[DecompositionComponent]:
    """Convert JIT output (weights + match lists) → DecompositionComponent list."""
    n = template.shape[0]
    comps: List[DecompositionComponent] = []
    for w, matches in zip(weights, all_matches):
        p = np.zeros((n, n), dtype=np.float64)
        for r, c in matches:
            p[r, c] = 1.0
        comps.append(DecompositionComponent(permutation=p, weight=float(w)))
    return comps


# ===========================================================================
# Public: configurable Euler framework
# ===========================================================================

def decompose_euler_framework(
    matrix: FloatMatrix,
    matching_method: str = "heavy",
    depth: int = 1,
    split_method: str = "euler",
    max_workers: Optional[int] = None,
) -> Tuple[List[DecompositionComponent], float, int]:
    """
    Euler splitting framework decomposition.

    Splits the input matrix *depth* times using the chosen split strategy,
    producing up to 2^depth doubly-stochastic leaf sub-matrices.  Each leaf is
    decomposed with BvN (matching_method).  Parallelism is simulated as the
    maximum leaf runtime.

    Args:
        matrix:         n×n non-negative integer matrix with equal row/col sums.
        matching_method: BvN matching engine ("heavy", "heavy_static", "wfa",
                         "maximum", "minimum").
        depth:          Number of splitting rounds (≥ 0).
                        depth=0 → plain BvN on the full matrix.
                        depth=d → up to 2^d leaf matrices, each (S/2^d)-regular.
        split_method:   "euler" or "greedy".
        max_workers:    Reserved (currently ignored; sequential loop with max
                        runtime simulates hardware parallelism).

    Returns:
        (components, max_leaf_runtime_s, num_leaves)
    """
    int_matrix = np.round(matrix).astype(np.int64)

    # Build list of leaf matrices by iteratively splitting
    leaves: List[np.ndarray] = [int_matrix]
    for _ in range(depth):
        next_leaves: List[np.ndarray] = []
        for leaf in leaves:
            S_leaf = int(leaf.sum(axis=1).max())
            if S_leaf <= 1:
                next_leaves.append(leaf)   # cannot split further
            else:
                red, blue = euler_split_once(leaf, split_method=split_method)
                next_leaves.append(red)
                next_leaves.append(blue)
        leaves = next_leaves

    non_empty = [lf for lf in leaves if np.any(lf > 0)]

    all_components: List[DecompositionComponent] = []
    max_leaf_runtime = 0.0

    for leaf in non_empty:
        leaf_float = leaf.astype(np.float64)
        comps, leaf_rt = _leaf_bvn(leaf_float, matching_method)
        max_leaf_runtime = max(max_leaf_runtime, leaf_rt)
        all_components.extend(comps)

    return all_components, max_leaf_runtime, len(non_empty)


# ===========================================================================
# Legacy: full recursive split to S=1
# ===========================================================================

def _recurse_to_perms(matrix: np.ndarray) -> List[np.ndarray]:
    if matrix.size == 0 or np.all(matrix == 0):
        return []
    S = int(matrix.sum(axis=1).max())
    if S == 0:
        return []
    n = matrix.shape[0]
    if S == 1:
        return [(matrix > 0).astype(np.float64)]
    if S % 2 == 1:
        perm, residual = _extract_one_perm(matrix)
        return [perm.astype(np.float64)] + _recurse_to_perms(residual)
    red, blue = _euler_split(matrix)
    return _recurse_to_perms(red) + _recurse_to_perms(blue)


def euler_decomposition(
    matrix: FloatMatrix,
) -> List[DecompositionComponent]:
    """
    Legacy: split all the way to S=1.  Prefer decompose_euler_framework for new code.
    """
    int_matrix = np.round(matrix).astype(np.int64)
    perms = _recurse_to_perms(int_matrix)
    return [DecompositionComponent(permutation=p, weight=1.0) for p in perms]
