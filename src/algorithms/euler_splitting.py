"""
Euler Splitting decomposition framework for S-regular integer matrices.

Three split strategies are available:

  "euler"     — Classic Euler-tour 2-colouring (Cole & Hopcroft 1982 / Alon 2003).
                Alternates Red/Blue along each Euler circuit → each sub-matrix has
                the same non-zero *positions* as the original, with values halved.
                Guaranteed doubly stochastic in O(N·S) time.

  "greedy"    — Greedy integral split.
                Sorts all (i,j) blocks by weight descending and assigns each block
                *entirely* to Red or Blue, tracking row/col budgets of S/2.  Falls
                back to Euler 2-colouring when it cannot complete a valid assignment.
                When it succeeds, the sub-matrices have fewer non-zero entries.

  "heuristic" — Same-direction Euler split.
                Builds the bipartite multigraph (M[i,j] parallel edges between row i
                and col j) and runs a modified Hierholzer Euler circuit where:
                  • Traversal row→col is always RED,  col→row is always BLUE.
                  • At each step we prefer edges whose direction is already locked
                    for that (i,j) pair — keeping all copies the same colour.
                  • "Smallest first": among equal-priority choices, prefer smaller
                    M[i,j] to close cheap entries before moving on.
                Goal: each (i,j) pair lands entirely in one sub-matrix → genuinely
                sparser halves.  The split is always doubly stochastic (each half
                has row/col sums = S/2) by the structure of the Euler circuit.

Public API
----------
euler_split_once(matrix, split_method)          → (red, blue)
decompose_euler_framework(matrix, ...)          → (components, split_time, max_rt, n_leaves)
euler_decomposition(matrix)                     → List[DecompositionComponent]  # legacy
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import jit
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
# Split strategy A2: Euler 2-colouring via GROUPED (bulk) edge traversal
# ---------------------------------------------------------------------------
#
# The bipartite multigraph puts M[i, j] *parallel* unit-edges between row
# node i and column node n+j.  `_euler_split` (and the heuristic variants)
# walk every one of those `sum(M)` unit-edges individually -- O(n * S) work,
# which dominates runtime on "dense-weight" matrices (large S, modest nnz).
#
# Key observation: because the graph is BIPARTITE, every step of *any*
# Eulerian circuit alternates row-side <-> col-side, so the colour of a
# crossing is fully determined by which side it departs from:
#
#     row -> col  =  RED
#     col -> row  =  BLUE
#
# That means if we fully drain a freshly-encountered (i, j) group of c
# parallel copies via a back-and-forth detour i <-> j right when we first
# reach it, the colours of those c crossings are simply a strict
# alternation starting with the colour of the departure side -- computable
# in closed form as (ceil(c/2), floor(c/2)) without walking a single one of
# them.  And after the detour we are back at the start vertex if c is even,
# or at the other endpoint if c is odd -- so the outer DFS can resume
# exactly as if the unit-by-unit walk had happened.
#
# This is a valid Hierholzer execution order (Hierholzer's algorithm is
# selection-order agnostic: it always completes a full Eulerian circuit
# regardless of which available edge is chosen at each step, as long as the
# graph is connected and every vertex has even degree -- both properties
# depend only on aggregate degree / connectivity, which grouping leaves
# untouched).  It therefore carries the exact same balance guarantee as the
# textbook alternating colouring (every vertex visit pairs one in-edge with
# one out-edge in adjacent -- hence opposite-coloured -- circuit positions),
# just computed in O(nnz) bulk steps instead of O(sum(M)) unit steps.
# ---------------------------------------------------------------------------

def _euler_tour_grouped(
    adj: AdjType,
    start: int,
    n: int,
    red: np.ndarray,
    blue: np.ndarray,
) -> None:
    """
    Hierholzer traversal that bulk-consumes each freshly-chosen parallel-edge
    group in one O(1) step and accumulates `red`/`blue` directly (no
    intermediate edge list, no separate colouring pass).

    Mutates `adj`, `red`, `blue` in place.
    """
    stack = [start]
    while stack:
        v = stack[-1]
        nbrs = adj[v]
        if not nbrs:
            stack.pop()
            continue

        u = next(iter(nbrs))
        c = nbrs[u]

        # Remove the *entire* parallel-edge group at once (symmetric / undirected)
        del adj[v][u]
        del adj[u][v]

        is_row = v < n
        if is_row:
            i, j = v, u - n
            first, second = red, blue     # row -> col departs RED first
        else:
            i, j = u, v - n
            first, second = blue, red     # col -> row departs BLUE first

        first[i, j]  += (c + 1) // 2      # ceil(c/2)  copies of the departure colour
        second[i, j] += c // 2            # floor(c/2) copies of the other colour

        # c alternating crossings starting at v: end back at v if c is even
        # (equal numbers of round trips), or at u if c is odd (one extra hop).
        if c % 2 == 1:
            stack.append(u)
        # else: stay at v -- adj[v][u] is now gone, loop re-evaluates v's neighbours


def _euler_split_grouped_py(int_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Euler 2-colouring via grouped (bulk) edge traversal — pure-Python
    reference implementation.

    Same contract and guarantees as `_euler_split` (lossless, each half
    (S/2)-regular -- "This holds for ANY Euler circuit, regardless of the
    heuristic order"), but each distinct (i, j) pair is coloured in one
    O(1) closed-form step instead of `M[i, j]` individual unit-edge steps.
    Cost drops from O(sum(M)) = O(n * S) to O(nnz(M)), the number of
    distinct non-zero entries -- a large win whenever entries are "heavy"
    (large S relative to nnz), e.g. the high-max-weight "dense" configs.

    See the module-level comment above `_euler_tour_grouped` for the full
    correctness argument (selection-order-agnostic Hierholzer + bipartite
    departure-side colouring).

    Kept as a readable reference / cross-check for `_euler_split_grouped`,
    which runs the same algorithm compiled with Numba (see
    `_euler_split_grouped_jit` further down) for another large constant-
    factor speedup on top of the O(nnz) algorithmic win.
    """
    n = int_matrix.shape[0]
    M = np.round(int_matrix).astype(np.int64)
    adj = _build_adj(M)
    red  = np.zeros((n, n), dtype=np.int64)
    blue = np.zeros((n, n), dtype=np.int64)
    for start in range(2 * n):
        if adj[start]:
            _euler_tour_grouped(adj, start, n, red, blue)
    return red, blue


def _euler_split_grouped(int_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Euler 2-colouring via grouped (bulk) edge traversal — JIT-compiled.

    Thin wrapper around `_euler_split_grouped_jit` (Numba nopython), which
    ports the dict-of-dicts traversal in `_euler_split_grouped_py` /
    `_euler_tour_grouped` to a flat int64 adjacency-count matrix (mirroring
    `_build_adj_jit`).  Same O(nnz(M)) algorithmic complexity, plus the
    usual Numba constant-factor win from removing Python dict/object
    overhead from the hot loop — this is the version actually dispatched
    for split_method == "euler_grouped".
    """
    n = int_matrix.shape[0]
    M = np.round(int_matrix).astype(np.int64)
    red, blue = _euler_split_grouped_jit(M, n)
    return red, blue


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
# Split strategy C: same-direction Euler heuristic  (JIT fast path)
# ===========================================================================
#
# The four @jit functions below implement the full heuristic Hierholzer
# traversal in Numba nopython mode.  They replace the pure-Python
# _choose_next_neighbor / _hierholzer_heuristic pair whose bottleneck was
# walking ~2 M edges one-by-one through Python dicts and dynamically
# allocated tier lists.
#
# Data-structure mapping  (Python → JIT)
# ----------------------------------------
#   adj[u][v] dict-of-dicts  →  adj_mat[2n,2n] int64  (edge counts)
#   target_dir[(i,j)] str/None  →  dir_arr[n,n] int8   (0=free,1=red,2=blue)
#   original_count[(i,j)]  →  M[i,j] directly (the matrix itself)
#   Python list stack/path  →  pre-allocated int64 arrays
# ===========================================================================


@jit(nopython=True, nogil=True, cache=True)
def _build_adj_jit(M, n):
    """
    Build a (2n x 2n) integer adjacency-count matrix from the demand matrix M.

    Row nodes  0 … n-1  correspond to matrix rows.
    Col nodes  n … 2n-1 correspond to matrix columns.
    adj_mat[i, n+j] = adj_mat[n+j, i] = M[i, j]  (parallel-edge count).
    """
    adj = np.zeros((2 * n, 2 * n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            k = M[i, j]
            if k > 0:
                adj[i, n + j] = k
                adj[n + j, i] = k
    return adj


@jit(nopython=True, nogil=True, cache=True)
def _choose_next_jit(v, n, adj, dir_arr, M):
    """
    JIT neighbour selector for the heuristic Hierholzer traversal.

    Replaces _choose_next_neighbor (Python).  A single O(2n) scan over all
    potential neighbours picks the best candidate without building lists or
    calling list.sort().

    Priority (unchanged from the Python version):
      Tier 1 – dir_arr[i,j] == current direction (direction already locked).
      Tier 2 – dir_arr[i,j] == 0              (first visit — free choice).
      Tier 3 – dir_arr[i,j] conflicts         (unavoidable split).
    Within each tier: smallest original M[i,j] is preferred.

    Direction encoding: 0 = free, 1 = red (row→col), 2 = blue (col→row).
    Returns the chosen neighbour node index, or -1 when v has no edges left.
    """
    # Bipartite structure: row nodes (0..n-1) only connect to col nodes (n..2n-1)
    # and vice versa.  Scanning the other half is always zero — restrict the
    # range to the live half only, halving the inner loop.
    is_row = v < n
    current_dir = 1 if is_row else 2

    best_t1_u = np.int64(-1)
    best_t1_w = np.int64(-1)
    best_t2_u = np.int64(-1)
    best_t2_w = np.int64(-1)
    best_t3_u = np.int64(-1)
    best_t3_w = np.int64(-1)

    if is_row:
        # v is a row node → neighbours are col nodes n..2n-1
        i = v
        for u in range(n, 2 * n):
            if adj[v, u] == 0:
                continue
            j    = u - n
            orig = M[i, j]
            td   = np.int64(dir_arr[i, j])
            if td == current_dir:
                if best_t1_u == -1 or orig < best_t1_w:
                    best_t1_w = orig
                    best_t1_u = np.int64(u)
            elif td == 0:
                if best_t2_u == -1 or orig < best_t2_w:
                    best_t2_w = orig
                    best_t2_u = np.int64(u)
            else:
                if best_t3_u == -1 or orig < best_t3_w:
                    best_t3_w = orig
                    best_t3_u = np.int64(u)
    else:
        # v is a col node → neighbours are row nodes 0..n-1
        j = v - n
        for u in range(n):
            if adj[v, u] == 0:
                continue
            i    = u
            orig = M[i, j]
            td   = np.int64(dir_arr[i, j])
            if td == current_dir:
                if best_t1_u == -1 or orig < best_t1_w:
                    best_t1_w = orig
                    best_t1_u = np.int64(u)
            elif td == 0:
                if best_t2_u == -1 or orig < best_t2_w:
                    best_t2_w = orig
                    best_t2_u = np.int64(u)
            else:
                if best_t3_u == -1 or orig < best_t3_w:
                    best_t3_w = orig
                    best_t3_u = np.int64(u)

    if best_t1_u != -1:
        return best_t1_u
    if best_t2_u != -1:
        return best_t2_u
    return best_t3_u


@jit(nopython=True, nogil=True, cache=True)
def _hierholzer_heuristic_jit(adj, n, dir_arr, M, start, stack_buf, red, blue):
    """
    JIT Hierholzer Euler-circuit with the same-direction heuristic —
    colours each edge DIRECTLY at the moment it is traversed.

    Replaces _hierholzer_heuristic (Python).  Uses caller-supplied
    stack_buf (pre-allocated int64 array) to avoid heap allocation in the
    hot loop.  Mutates adj, dir_arr, red, blue in-place.

    BUGFIX (was: build a `path_buf` via pop-and-reverse, then colour
    adjacent pairs in a separate `_color_path_jit` pass): that
    reconstruction can — under the same-direction heuristic's "forced
    ruin" (Tier-3) choices — splice sub-tours in a way that places two
    same-bipartite-side nodes adjacently in the popped sequence (e.g.
    two column nodes back to back).  `_color_path_jit` would then treat
    that *non-edge* pair as a traversed edge and colour it, indexing
    red/blue with a node id ≥ n — an out-of-bounds write that Numba's
    nopython mode does not bounds-check, corrupting the heap (observed as
    a process-killing access violation on the dense paper config).

    The fix: never reconstruct a circuit at all.  Each iteration of this
    loop already knows EXACTLY which real edge (v, u) it is consuming and
    which physical direction it departs — row→col is unconditionally RED,
    col→row is unconditionally BLUE (see _euler_split_heuristic_jit
    docstring) — so we can accumulate red/blue right here, on genuine
    edges only, with no reconstruction step that could ever fabricate a
    phantom edge.
    """
    stack_top = np.int64(0)

    stack_buf[0] = np.int64(start)
    stack_top = np.int64(1)

    while stack_top > 0:
        v = stack_buf[stack_top - 1]
        u = _choose_next_jit(v, n, adj, dir_arr, M)

        if u != -1:
            # ── Traverse edge v → u ──────────────────────────────────────
            adj[v, u] -= 1
            adj[u, v] -= 1

            is_row = v < n
            if is_row:
                i = v
                j = u - n
                red[i, j] += 1                       # row → col = RED
                if dir_arr[i, j] == 0:
                    dir_arr[i, j] = np.int8(1)        # lock to red
            else:
                i = u
                j = v - n
                blue[i, j] += 1                       # col → row = BLUE
                if dir_arr[i, j] == 0:
                    dir_arr[i, j] = np.int8(2)        # lock to blue

            stack_buf[stack_top] = u
            stack_top += 1
        else:
            # No edges left from v — node is done; nothing to record,
            # colouring already happened on traversal.
            stack_top -= 1


@jit(nopython=True, nogil=True, cache=True)
def _euler_split_heuristic_jit(M, n):
    """
    JIT top-level: full same-direction Euler heuristic split.

    Builds the bipartite multigraph adjacency array, then runs heuristic
    Hierholzer on each connected component — colouring red/blue directly
    during traversal (see _hierholzer_heuristic_jit bugfix note: no
    separate circuit-reconstruction/colouring pass exists anymore, which
    is what makes this immune to the phantom-edge corruption that used to
    crash the process on the dense paper config).

    Stack buffer is allocated once and reused across components.

    Returns (red, blue) int64 matrices each with row/col sums = S/2.
    """
    adj     = _build_adj_jit(M, n)
    dir_arr = np.zeros((n, n), dtype=np.int8)
    red     = np.zeros((n, n), dtype=np.int64)
    blue    = np.zeros((n, n), dtype=np.int64)

    # Pre-allocate the stack buffer sized to the total number of parallel
    # edges (its depth is bounded by the total edge count + 1).
    total_edges = np.int64(0)
    for i in range(n):
        for j in range(n):
            total_edges += M[i, j]
    buf_size = total_edges + 2

    stack_buf = np.empty(buf_size, dtype=np.int64)

    for start in range(2 * n):
        # Check whether this node has any remaining edges.
        # Exploit bipartite structure: row nodes (0..n-1) only connect to col
        # nodes (n..2n-1) and vice versa — only scan the live half.
        has_edges = False
        if start < n:                        # row node → check col side
            for u in range(n, 2 * n):
                if adj[start, u] > 0:
                    has_edges = True
                    break
        else:                                # col node → check row side
            for u in range(n):
                if adj[start, u] > 0:
                    has_edges = True
                    break
        if not has_edges:
            continue

        _hierholzer_heuristic_jit(
            adj, n, dir_arr, M, start, stack_buf, red, blue
        )

    return red, blue


# ===========================================================================
# JIT port of the grouped / bulk-edge Euler split (see _euler_tour_grouped /
# _euler_split_grouped_py above for the readable pure-Python reference and
# the full correctness argument).
#
# Data-structure mapping (Python → JIT) — identical scheme to the heuristic
# JIT port above:
#   adj[u][v] dict-of-dicts (edge counts)  →  adj_mat[2n,2n] int64
#   Python list `stack`                    →  pre-allocated int64 array
#
# Unlike the heuristic variant, no direction-locking state (`dir_arr`) or
# separate colouring pass is needed: colour is determined purely by which
# bipartite side the current node departs from (row→col = RED-first,
# col→row = BLUE-first — see the correctness comment above), so each
# consumed edge-group is coloured and accumulated directly into red/blue
# the moment it is traversed.
# ===========================================================================


@jit(nopython=True, nogil=True, cache=True)
def _choose_any_jit(v, n, adj):
    """
    Return any neighbour u of v that still has a non-empty parallel-edge
    group (adj[v, u] > 0), or -1 if v has no edges left.

    Unlike `_choose_next_jit` (heuristic variant), no priority/ordering is
    needed here: Hierholzer's algorithm is selection-order-agnostic — ANY
    choice that empties an edge group still yields a valid single-traversal
    Euler circuit covering every edge exactly once (see the correctness
    argument above `_euler_tour_grouped`).  We therefore just take the
    first live neighbour found, restricting the scan to the live bipartite
    half (row nodes only connect to col nodes and vice versa).
    """
    if v < n:
        for u in range(n, 2 * n):
            if adj[v, u] > 0:
                return u
    else:
        for u in range(n):
            if adj[v, u] > 0:
                return u
    return -1


@jit(nopython=True, nogil=True, cache=True)
def _hierholzer_grouped_jit(adj, n, red, blue, start, stack_buf):
    """
    JIT Hierholzer traversal that bulk-consumes each freshly-chosen
    parallel-edge group in a single O(1) step, accumulating red/blue
    colour counts directly — no separate path buffer or colouring pass.

    Mutates `adj`, `red`, `blue` in place.  `stack_buf` is a caller-
    supplied pre-allocated int64 buffer.  Each push corresponds to fully
    consuming one distinct (i, j) edge-group with an odd remaining count
    (the even-count case stays at v without pushing), so across the whole
    traversal of this component the number of pushes is bounded by the
    number of distinct edge-groups touched — i.e. by nnz(M).

    Colouring rule (departure side ⇒ colour, see module comment):
        row  → col   departs RED  first  (ceil(c/2) red, floor(c/2) blue)
        col  → row   departs BLUE first  (ceil(c/2) blue, floor(c/2) red)
    """
    stack_top = np.int64(0)
    stack_buf[0] = np.int64(start)
    stack_top = np.int64(1)

    while stack_top > 0:
        v = stack_buf[stack_top - 1]
        u = _choose_any_jit(v, n, adj)

        if u == -1:
            stack_top -= 1
            continue

        c = adj[v, u]
        adj[v, u] = 0
        adj[u, v] = 0

        if v < n:
            i = v
            j = u - n
            red[i, j]  += (c + 1) // 2   # ceil(c/2) — row→col departs RED first
            blue[i, j] += c // 2         # floor(c/2)
        else:
            i = u
            j = v - n
            blue[i, j] += (c + 1) // 2   # ceil(c/2) — col→row departs BLUE first
            red[i, j]  += c // 2         # floor(c/2)

        if c % 2 == 1:
            stack_buf[stack_top] = u
            stack_top += 1
        # else: remain at v — its edge to u is now fully consumed, loop continues


@jit(nopython=True, nogil=True, cache=True)
def _euler_split_grouped_jit(M, n):
    """
    JIT top-level: full grouped/bulk-edge Euler split.

    Builds the bipartite multigraph adjacency-count matrix, runs the
    grouped Hierholzer traversal on each connected component (colouring
    directly as it goes), and returns (red, blue) int64 matrices each with
    row/col sums = S/2.

    Stack buffer is sized to nnz(M) + 2: every push fully empties one
    distinct (i, j) edge-group (adj[v,u] is zeroed on consumption), so at
    most nnz(M) pushes can occur in total across all components.
    """
    adj  = _build_adj_jit(M, n)
    red  = np.zeros((n, n), dtype=np.int64)
    blue = np.zeros((n, n), dtype=np.int64)

    nnz = np.int64(0)
    for i in range(n):
        for j in range(n):
            if M[i, j] > 0:
                nnz += 1
    buf_size = nnz + 2

    stack_buf = np.empty(buf_size, dtype=np.int64)

    for start in range(2 * n):
        has_edges = False
        if start < n:
            for u in range(n, 2 * n):
                if adj[start, u] > 0:
                    has_edges = True
                    break
        else:
            for u in range(n):
                if adj[start, u] > 0:
                    has_edges = True
                    break
        if not has_edges:
            continue

        _hierholzer_grouped_jit(adj, n, red, blue, start, stack_buf)

    return red, blue


def _choose_next_neighbor(
    v: int,
    n: int,
    adj: AdjType,
    target_dir: Dict[Tuple[int, int], Optional[str]],
    original_count: Dict[Tuple[int, int], int],
) -> Optional[int]:
    """
    Heuristic edge selector for the Euler circuit.

    DIRECTION CONVENTION
    --------------------
    The bipartite graph has row nodes (0..n-1) on the left and col nodes
    (n..2n-1) on the right.  Traversal direction determines colour:

        row  →  col   (left  → right)  =  RED
        col  →  row   (right → left)   =  BLUE

    Because the graph is bipartite, the colour is entirely determined by
    which side the CURRENT node v is on — no parity counting needed.

    PRIORITY (from "prefer edges already going the same direction"):
    ----------------------------------------------------------------
    Tier 1 – target_dir[(i,j)] already matches current direction.
             "We've been going this way for (i,j) before — stay consistent."
    Tier 2 – target_dir[(i,j)] is None (first visit to this pair).
             "Free choice; we'll lock it to the current direction now."
    Tier 3 – target_dir[(i,j)] conflicts.
             "Forced ruin — only taken when no other option exists."

    SMALLEST FIRST: within each tier, prefer edges with the smallest
    original M[i,j].  Small entries are easiest to close in one colour.
    """
    is_row = v < n
    current_dir = 'red' if is_row else 'blue'

    tier1: List[Tuple[int, int]] = []   # same direction already locked
    tier2: List[Tuple[int, int]] = []   # direction not yet decided (free)
    tier3: List[Tuple[int, int]] = []   # direction conflict (forced ruin)

    for u in adj[v]:
        # Map graph nodes back to the matrix (i, j) pair
        #   row node v  + col node u=n+j  →  key (v, u-n)
        #   col node v  + row node u      →  key (u, v-n)
        key = (v, u - n) if is_row else (u, v - n)

        td   = target_dir.get(key)           # None / 'red' / 'blue'
        orig = original_count.get(key, 1)    # original M[i,j]

        if td == current_dir:
            tier1.append((orig, u))   # Tier 1: keep all copies same colour
        elif td is None:
            tier2.append((orig, u))   # Tier 2: first traversal — free
        else:
            tier3.append((orig, u))   # Tier 3: unavoidable conflict

    # Sort each tier by original M[i,j] — smallest entries first
    for tier in (tier1, tier2, tier3):
        tier.sort()
        if tier:
            return tier[0][1]   # return the chosen neighbor node

    return None   # no edges left from v


def _hierholzer_heuristic(
    adj: AdjType,
    start: int,
    n: int,
    target_dir: Dict[Tuple[int, int], Optional[str]],
    original_count: Dict[Tuple[int, int], int],
) -> List[Tuple[int, int]]:
    """
    Hierholzer's Euler circuit algorithm with the same-direction heuristic.

    Returns the circuit as a list of directed (u, v) edge tuples.

    KEY STEP — direction locking
    ----------------------------
    Every time we traverse an edge for the first time we lock in the
    direction for that (i,j) pair:

        target_dir[(i,j)] = 'red'   if we went row_i → col_j
        target_dir[(i,j)] = 'blue'  if we went col_j → row_i

    All future copies of that pair will be Tier-1 candidates in
    _choose_next_neighbor, so the heuristic naturally keeps them all
    going the same way.  If the circuit ever forces a conflict the pair
    is "split" between sub-matrices — we accept that and move on.
    """
    stack = [start]
    path: List[int] = []

    while stack:
        v = stack[-1]
        u = _choose_next_neighbor(v, n, adj, target_dir, original_count)

        if u is not None:
            # ── Traverse edge v → u ──────────────────────────────────
            adj[v][u] -= 1
            if adj[v][u] == 0:
                del adj[v][u]
            adj[u][v] -= 1
            if adj[u][v] == 0:
                del adj[u][v]

            # Lock direction on first traversal of this (i,j) pair
            is_row = v < n
            key = (v, u - n) if is_row else (u, v - n)
            if target_dir.get(key) is None:
                target_dir[key] = 'red' if is_row else 'blue'

            stack.append(u)
        else:
            # No edges left from v — this node is done
            path.append(stack.pop())

    path.reverse()
    return [(path[k], path[k + 1]) for k in range(len(path) - 1)]


def _euler_split_heuristic(int_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Euler 2-colouring with the same-direction heuristic.

    GRAPH CONSTRUCTION
    ------------------
    Represent M as a bipartite multigraph:
        Left  nodes  0 … n-1    =  matrix rows
        Right nodes  n … 2n-1   =  matrix columns
        M[i,j] parallel edges between node i and node n+j

    COLOUR CONVENTION
    -----------------
        Row → Col  (left  → right)  =  RED   →  red[i,j]  += 1
        Col → Row  (right → left)   =  BLUE  →  blue[i,j] += 1

    DOUBLY STOCHASTIC GUARANTEE
    ----------------------------
    In a bipartite Euler circuit every row node is entered (BLUE in) and
    exited (RED out) exactly S/2 times each.  Therefore:
        red.sum(axis=1)  ==  S/2  for every row   (row sums balanced)
        red.sum(axis=0)  ==  S/2  for every col   (col sums balanced)
    This holds for ANY Euler circuit, regardless of the heuristic order.

    SPARSITY GOAL
    -------------
    By keeping all copies of (i,j) the same direction, each (i,j) pair
    lands entirely in one sub-matrix (either red[i,j]=M[i,j], blue[i,j]=0
    or vice versa).  This reduces the number of non-zeros in each half
    compared with the plain alternating split which distributes every pair
    across both sub-matrices.

    IMPLEMENTATION
    --------------
    The inner Hierholzer traversal and neighbour-selection heuristic are
    compiled with Numba (nopython=True, nogil=True) via
    _euler_split_heuristic_jit(), which replaces the original pure-Python
    loop that walked ~2 M edges one-by-one through Python dicts and
    dynamically allocated tier lists (the observed 17.5 s bottleneck).
    """
    n = int_matrix.shape[0]
    M = np.round(int_matrix).astype(np.int64)
    red, blue = _euler_split_heuristic_jit(M, n)
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
        split_method: "euler"         — classic alternating Euler 2-colouring
                                         (unit-by-unit edge traversal).
                      "euler_grouped" — same alternating 2-colouring, but
                                         walks each distinct (i,j) parallel-edge
                                         group once in O(1) closed form instead
                                         of M[i,j] individual unit-edge steps
                                         (O(nnz) vs O(sum(M)) traversal cost).
                      "greedy"        — greedy integral split (falls back to euler).
                      "heuristic"     — same-direction Euler split (new strategy).

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
    elif split_method == "heuristic":
        red, blue = _euler_split_heuristic(int_matrix)
    elif split_method == "euler_grouped":
        red, blue = _euler_split_grouped(int_matrix)
    else:  # default: "euler"
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
) -> Tuple[List[DecompositionComponent], float, float, int]:
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
        (components, split_time_s, max_leaf_runtime_s, num_leaves)

        split_time_s is the wall-clock time spent in the splitting phase only
        (building the leaf tree), so callers can separate "cost of splitting"
        from "cost of extracting matchings from the leaves".
    """
    int_matrix = np.round(matrix).astype(np.int64)

    # Build list of leaf matrices by iteratively splitting
    t_split0 = time.perf_counter()
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
    split_time = time.perf_counter() - t_split0

    non_empty = [lf for lf in leaves if np.any(lf > 0)]

    all_components: List[DecompositionComponent] = []
    max_leaf_runtime = 0.0

    for leaf in non_empty:
        leaf_float = leaf.astype(np.float64)
        comps, leaf_rt = _leaf_bvn(leaf_float, matching_method)
        max_leaf_runtime = max(max_leaf_runtime, leaf_rt)
        all_components.extend(comps)

    return all_components, split_time, max_leaf_runtime, len(non_empty)


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
