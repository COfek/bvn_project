"""
Euler Splitting decomposition for S-regular integer matrices.

Based on Cole & Hopcroft (1982) and Alon (2003):
  1. Model the matrix as a bipartite multigraph (row-nodes U, col-nodes V).
  2. Find Euler circuits across all connected components (every node has even
     degree S when S is even, so each component has an Euler circuit).
  3. 2-colour edges Red/Blue along each circuit → two (S/2)-regular sub-matrices.
  4. Recurse until S=1 (a single permutation matrix).

Complexity: O(N · S · log₂ S)  vs  iterative BvN O(S² · N · √N)

Adjacency is stored as a Counter-style dict {neighbour: remaining_count} so
that each edge traversal costs O(1) rather than O(degree) list search.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .bvn import DecompositionComponent
from .sorted_array_matching import sorted_array_matching   # guarantees perfect matching via augmenting paths

FloatMatrix = NDArray[np.float64]

# Type alias: adj list maps node_id → {neighbour_id: edge_count}
AdjType = List[Dict[int, int]]


# ---------------------------------------------------------------------------
# Bipartite multigraph builder
# ---------------------------------------------------------------------------

def _build_adj(matrix: np.ndarray) -> AdjType:
    """
    Build dict-based adjacency for the bipartite multigraph.

    Node indices:
      Row node i  →  i          (0 .. n-1)
      Col node j  →  n + j      (n .. 2n-1)

    matrix[i, j] = k  ↔  k undirected parallel edges between i and n+j.
    adj[u][v] = number of remaining edges between u and v.
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
# Euler tour (Hierholzer) with O(1) edge removal via dict
# ---------------------------------------------------------------------------

def _euler_tour_single(adj: AdjType, start: int) -> List[Tuple[int, int]]:
    """
    Hierholzer's algorithm on one connected component starting at *start*.
    Mutates adj (edges are consumed).
    Returns list of (u, v) directed edge pairs in Euler-circuit order.
    """
    stack = [start]
    edge_path: List[Tuple[int, int]] = []
    node_path: List[int] = []

    while stack:
        v = stack[-1]
        if adj[v]:
            # Pick any neighbour in O(1)
            u = next(iter(adj[v]))
            # Consume one edge (v, u)
            adj[v][u] -= 1
            if adj[v][u] == 0:
                del adj[v][u]
            adj[u][v] -= 1
            if adj[u][v] == 0:
                del adj[u][v]
            stack.append(u)
        else:
            node = stack.pop()
            node_path.append(node)

    # node_path is reversed; convert consecutive pairs to edges
    # The tour visits node_path[-1] → node_path[-2] → ... → node_path[0]
    edges: List[Tuple[int, int]] = []
    for k in range(len(node_path) - 1, 0, -1):
        edges.append((node_path[k], node_path[k - 1]))
    return edges


def _all_euler_tours(adj: AdjType) -> List[List[Tuple[int, int]]]:
    """
    Find Euler circuits across ALL connected components.
    Returns one edge-list per component, each edge as (u, v).
    """
    components: List[List[Tuple[int, int]]] = []
    for start in range(len(adj)):
        if not adj[start]:
            continue
        edges = _euler_tour_single(adj, start)
        if edges:
            components.append(edges)
    return components


# ---------------------------------------------------------------------------
# 2-colouring of Euler-tour edges
# ---------------------------------------------------------------------------

def _colour_edges(
    component_edges: List[List[Tuple[int, int]]],
    n: int,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alternate Red/Blue along each Euler circuit independently.
    Returns (M_red, M_blue) — each (S/2)-regular integer matrix.
    """
    red  = np.zeros(shape, dtype=np.int64)
    blue = np.zeros(shape, dtype=np.int64)

    for edges in component_edges:
        colour = 0          # reset per component
        for (u, v) in edges:
            if u < n and v >= n:
                row, col = u, v - n
            elif v < n and u >= n:
                row, col = v, u - n
            else:
                continue    # shouldn't happen in a bipartite graph

            if colour == 0:
                red[row, col]  += 1
            else:
                blue[row, col] += 1
            colour ^= 1

    return red, blue


# ---------------------------------------------------------------------------
# One-permutation extractor using augmenting-path matching (used when S odd)
# ---------------------------------------------------------------------------

def _extract_one_permutation(matrix: np.ndarray) -> np.ndarray:
    """
    Extract one valid permutation matrix from *matrix* using the
    augmenting-path greedy matching (Phase 1 + Phase 2 BFS), which
    guarantees a PERFECT matching for any regular bipartite matrix.

    The float64 matrix passed to sorted_array_matching is a view; we
    convert the int64 input to float64 for compatibility.
    """
    n = matrix.shape[0]
    float_mat = matrix.astype(np.float64)
    matches = sorted_array_matching(float_mat)   # list of (row, col) pairs

    perm = np.zeros((n, n), dtype=np.float64)
    for (r, c) in matches:
        perm[r, c] = 1.0
    return perm


# ---------------------------------------------------------------------------
# Recursive Euler splitting
# ---------------------------------------------------------------------------

def _recurse(matrix: np.ndarray) -> List[np.ndarray]:
    """
    Recursively split *matrix* (integer) until S=1.
    Returns list of binary {0,1} permutation matrices.
    """
    if matrix.size == 0 or np.all(matrix == 0):
        return []

    S = int(round(int(matrix.sum(axis=1).max())))
    if S == 0:
        return []

    n = matrix.shape[0]

    if S == 1:
        perm = (matrix > 0).astype(np.float64)
        return [perm]

    # If S is odd, extract one permutation first
    if S % 2 == 1:
        perm = _extract_one_permutation(matrix)
        residual = (matrix - perm.astype(np.int64)).clip(min=0)
        return [perm] + _recurse(residual)

    adj = _build_adj(matrix)
    component_edges = _all_euler_tours(adj)
    red, blue = _colour_edges(component_edges, n, matrix.shape)

    return _recurse(red) + _recurse(blue)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def euler_decomposition(
    matrix: FloatMatrix,
) -> List[DecompositionComponent]:
    """
    Decompose an S-regular integer matrix via recursive Euler splitting.

    Each output component has weight=1.0 and is a binary permutation matrix.
    Total components = S (the common row/column sum).

    Args:
        matrix: n×n non-negative integer matrix with equal row and column sums.

    Returns:
        List of DecompositionComponent(permutation=P, weight=1.0).
    """
    int_matrix = np.round(matrix).astype(np.int64)
    perms = _recurse(int_matrix)
    return [
        DecompositionComponent(permutation=p, weight=1.0)
        for p in perms
    ]
