from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from src.algorithms.split_tree import (
    float_matrix,
    pschedule,
    split_tree_component,
    random_binary_split,
    verify_reconstruction,
    decompose_leaf_with_wfa,
)


@dataclass
class SplitTreeNode:
    node_id: int
    depth: int
    matrix: float_matrix
    left: Optional["SplitTreeNode"] = None
    right: Optional["SplitTreeNode"] = None

    # --- Annotated by analysis (so visualization can read it) ---
    num_permutations: Optional[int] = None
    cycle_length: Optional[float] = None
    recon_error: Optional[float] = None

    @property
    def nnz(self) -> int:
        return int(np.count_nonzero(self.matrix))

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class NodeDecompReport:
    node_id: int
    depth: int
    nnz: int
    num_permutations: int
    cycle_length: float
    recon_error: float


@dataclass
class LevelDecompReport:
    depth: int
    num_nodes: int
    total_nnz: int
    total_permutations: int
    total_cycle_length: float
    max_node_recon_error: float


@dataclass
class SplitTreeLevelAnalysisResult:
    root: Optional[SplitTreeNode]
    leaves: List[SplitTreeNode]

    # Tree reconstruction check: sum(leaves) ≈ original
    tree_reconstruction_ok: bool
    tree_reconstruction_error: float

    # Per-level aggregation of node decompositions
    levels: List[LevelDecompReport]

    # Optional per-node detailed reports
    nodes: List[NodeDecompReport]


def build_split_tree_full(
    x: float_matrix,
    sparsity_target: int,
    max_depth: int,
    p_schedule: pschedule,
) -> Optional[SplitTreeNode]:
    """
    Build the entire split-tree and store it as a binary tree (internal nodes included).

    Uses the same splitting logic as split_tree(), but keeps internal nodes.
    """
    next_id = [0]

    def _rec(mat: float_matrix, depth: int) -> Optional[SplitTreeNode]:
        if np.count_nonzero(mat) == 0:
            return None

        node = SplitTreeNode(node_id=next_id[0], depth=depth, matrix=mat.copy())
        next_id[0] += 1

        # stopping
        if node.nnz <= sparsity_target or depth >= max_depth:
            return node

        # choose p
        p = float(p_schedule(mat, depth)) if callable(p_schedule) else float(p_schedule)
        if not (0.0 < p < 1.0):
            raise ValueError(f"p_schedule produced invalid p={p} at depth={depth}")

        a, b = random_binary_split(mat, p)

        if np.count_nonzero(a) > 0:
            node.left = _rec(a, depth + 1)
        if np.count_nonzero(b) > 0:
            node.right = _rec(b, depth + 1)

        return node

    return _rec(x, 0)


def collect_leaves(root: Optional[SplitTreeNode]) -> List[SplitTreeNode]:
    if root is None:
        return []
    out: List[SplitTreeNode] = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.is_leaf:
            out.append(n)
        else:
            if n.left is not None:
                stack.append(n.left)
            if n.right is not None:
                stack.append(n.right)
    return out


def nodes_by_level(root: Optional[SplitTreeNode]) -> Dict[int, List[SplitTreeNode]]:
    if root is None:
        return {}
    levels: Dict[int, List[SplitTreeNode]] = {}
    q = [root]
    while q:
        n = q.pop(0)
        levels.setdefault(n.depth, []).append(n)
        if n.left is not None:
            q.append(n.left)
        if n.right is not None:
            q.append(n.right)
    return levels


def _reconstruct_from_components(shape: Tuple[int, int], comps: List[split_tree_component]) -> float_matrix:
    s = np.zeros(shape, dtype=np.float64)
    for c in comps:
        s += c.matrix
    return s


def analyze_split_tree_levels(
    x: float_matrix,
    sparsity_target: int,
    max_depth: int,
    p_schedule: pschedule,
    tol: float = 1e-12,
    include_node_details: bool = True,
) -> SplitTreeLevelAnalysisResult:
    """
    1) Build full tree structure (internal nodes included)
    2) For each level: decompose every node.matrix using your decompose_leaf_with_wfa
    3) Save per-level stats: total permutations, total cycle length
    4) Verify:
       - tree reconstruction: sum(leaves) == original
       - per-node decomposition reconstruction: sum(components) == node.matrix
    Also: annotates each node with (num_permutations, cycle_length, recon_error)
    so visualization can label nodes.
    """
    root = build_split_tree_full(
        x=x,
        sparsity_target=sparsity_target,
        max_depth=max_depth,
        p_schedule=p_schedule,
    )

    leaves = collect_leaves(root)

    # --- verify tree reconstruction via leaves ---
    leaf_mats = [leaf.matrix for leaf in leaves]
    tree_ok = verify_reconstruction(x, leaf_mats, tol=tol)

    if len(leaf_mats) == 0:
        tree_err = float(np.max(np.abs(x)))
    else:
        s = np.zeros_like(x)
        for lm in leaf_mats:
            s += lm
        tree_err = float(np.max(np.abs(x - s)))

    # --- per-level decomposition reports ---
    lvl_map = nodes_by_level(root)
    level_reports: List[LevelDecompReport] = []
    node_reports: List[NodeDecompReport] = []

    for depth in sorted(lvl_map.keys()):
        nodes = lvl_map[depth]

        total_nnz = 0
        total_perm = 0
        total_cycle = 0.0
        max_err = 0.0

        for node in nodes:
            total_nnz += node.nnz

            comps = decompose_leaf_with_wfa(node.matrix, tol=tol)

            node_cycle = float(sum(c.weight for c in comps))
            recon = _reconstruct_from_components(node.matrix.shape, comps)
            err = float(np.max(np.abs(node.matrix - recon)))

            # --- annotate the node for visualization ---
            node.num_permutations = len(comps)
            node.cycle_length = node_cycle
            node.recon_error = err

            total_perm += node.num_permutations
            total_cycle += node.cycle_length
            max_err = max(max_err, err)

            if include_node_details:
                node_reports.append(
                    NodeDecompReport(
                        node_id=node.node_id,
                        depth=node.depth,
                        nnz=node.nnz,
                        num_permutations=node.num_permutations,
                        cycle_length=node.cycle_length,
                        recon_error=node.recon_error,
                    )
                )

        level_reports.append(
            LevelDecompReport(
                depth=depth,
                num_nodes=len(nodes),
                total_nnz=int(total_nnz),
                total_permutations=int(total_perm),
                total_cycle_length=float(total_cycle),
                max_node_recon_error=float(max_err),
            )
        )

    return SplitTreeLevelAnalysisResult(
        root=root,
        leaves=leaves,
        tree_reconstruction_ok=bool(tree_ok),
        tree_reconstruction_error=float(tree_err),
        levels=level_reports,
        nodes=node_reports,
    )
