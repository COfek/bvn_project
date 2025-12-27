from __future__ import annotations

import numpy as np

from src.utils.matrix_generator import random_sparse_doubly_stochastic
from src.algorithms.split_tree import decompose_leaf_with_wfa

# analysis modules (same folder)
from split_tree_level_analysis import analyze_split_tree_levels, nodes_by_level
from analysis.visualize.split_tree_visualization import plot_split_tree_as_tree


def print_matrix(mat: np.ndarray, indent: int = 0):
    pad = " " * indent
    for row in mat:
        print(pad + np.array2string(row, precision=6, floatmode="fixed"))


def main():
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------
    # Generate test matrix
    # ------------------------------------------------------------
    M = random_sparse_doubly_stochastic(
        n=4,
        density=1.0,
        iters=50,
        rng=rng,
    )

    print("\n=== ORIGINAL MATRIX ===")
    print_matrix(M, indent=2)
    print("Total sum:", float(M.sum()))

    # ------------------------------------------------------------
    # Split-tree level analysis
    # ------------------------------------------------------------
    res = analyze_split_tree_levels(
        x=M,
        sparsity_target=3,
        max_depth=2,
        p_schedule=0.5,
        tol=1e-12,
        include_node_details=False,  # keep console output readable
    )

    print("\nTree reconstruction OK:", res.tree_reconstruction_ok)
    print("Tree reconstruction error:", res.tree_reconstruction_error)

    # ------------------------------------------------------------
    # Per-level summary
    # ------------------------------------------------------------
    print("\n============================================================")
    print("PER-LEVEL SUMMARY")
    print("============================================================")
    print("depth | nodes | total_nnz | total_perm | total_cycle | max_node_err")
    print("--------------------------------------------------------------------")
    for lvl in res.levels:
        print(
            f"{lvl.depth:5d} | {lvl.num_nodes:5d} | {lvl.total_nnz:9d} | "
            f"{lvl.total_permutations:10d} | {lvl.total_cycle_length:10.6f} | "
            f"{lvl.max_node_recon_error:.2e}"
        )

    # ------------------------------------------------------------
    # Optional: detailed node matrices + WFA decomposition
    # ------------------------------------------------------------
    PRINT_NODE_MATRICES = True
    PRINT_WFA_COMPONENTS = True

    if PRINT_NODE_MATRICES:
        lvl_map = nodes_by_level(res.root)

        for depth in sorted(lvl_map.keys()):
            print(f"\n================ LEVEL {depth} ================")
            for node in lvl_map[depth]:
                print(
                    f"\n--- Node {node.node_id} | depth={node.depth} | "
                    f"nnz={node.nnz} ---"
                )

                print("Node matrix:")
                print_matrix(node.matrix, indent=4)

                if PRINT_WFA_COMPONENTS:
                    comps = decompose_leaf_with_wfa(node.matrix)

                    print("\nWFA decomposition:")
                    recon = np.zeros_like(node.matrix)

                    for k, c in enumerate(comps):
                        print(f"\n  Component {k}: λ = {c.weight:.6f}")
                        print_matrix(c.matrix, indent=6)
                        recon += c.matrix

                    err = float(np.max(np.abs(node.matrix - recon)))
                    print("\nReconstructed node matrix:")
                    print_matrix(recon, indent=4)
                    print(f"Node reconstruction error: {err:.3e}")
                    print(f"Node cycle length: {sum(c.weight for c in comps):.6f}")

    # ------------------------------------------------------------
    # Plot split tree (TRUE TREE LAYOUT)
    # ------------------------------------------------------------
    plot_split_tree_as_tree(res.root)


if __name__ == "__main__":
    main()
