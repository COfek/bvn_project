import numpy as np

from analysis.visualize.frontier_visualization import plot_frontier_evolution
from analysis.split_tree_level_analysis import analyze_split_tree_levels
from src.utils.matrix_generator import random_sparse_doubly_stochastic
from analysis.split_frontier_analysis import analyze_split_frontier
from src.algorithms.split_tree import decompose_leaf_with_wfa
from analysis.visualize.split_tree_visualization import plot_split_tree_as_tree
from analysis.utils.plot_saver import save_current_figures


# ---------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------

def print_matrix(mat: np.ndarray, indent: int = 0):
    pad = " " * indent
    for row in mat:
        print(pad + np.array2string(row, precision=6, floatmode="fixed"))


# ---------------------------------------------------------------------
# Debug script
# ---------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------
    # Generate test matrix
    # ------------------------------------------------------------
    matrix = random_sparse_doubly_stochastic(
        n=4,
        density=1.0,
        iters=50,
        rng=rng,
    )

    print("\n================ ORIGINAL MATRIX ================")
    print_matrix(matrix, indent=2)
    print("Total sum:", float(matrix.sum()))

    tree_res = analyze_split_tree_levels(
        x=matrix,
        sparsity_target=3,
        max_depth=5,
        p_schedule=0.5,
        tol=1e-12,
        include_node_details=False,  # keep console output readable
    ) #Only so we can plot the tree we got
    # ------------------------------------------------------------
    # Frontier-based split analysis
    # ------------------------------------------------------------
    res = analyze_split_frontier(
        x=matrix,
        sparsity_target=3,
        max_depth=6,
        p_schedule=0.5,
        tol=1e-12,
    )

    print("\n================ FRONTIER ANALYSIS ================")
    print("step | matrices | total_nnz | permutations | cycle | max_err")
    print("--------------------------------------------------------------")

    for r in res.steps:
        print(
            f"{r.step:4d} | "
            f"{r.num_matrices:8d} | "
            f"{r.total_nnz:9d} | "
            f"{r.total_permutations:12d} | "
            f"{r.total_cycle_length:7.6f} | "
            f"{r.max_reconstruction_error:.2e}"
        )

    print("\nFinal reconstruction OK:", res.reconstruction_ok)

    # ------------------------------------------------------------
    # Optional: inspect final frontier in detail
    # ------------------------------------------------------------
    PRINT_FRONTIER_MATRICES = True
    PRINT_WFA_COMPONENTS = True

    if PRINT_FRONTIER_MATRICES:
        print("\n================ FINAL FRONTIER ================")

        for idx, mat in enumerate(res.final_frontier):
            print(f"\n--- Frontier matrix {idx} | nnz={np.count_nonzero(mat)} ---")
            print_matrix(mat, indent=4)

            if PRINT_WFA_COMPONENTS:
                comps = decompose_leaf_with_wfa(mat)

                recon = np.zeros_like(mat)
                print("\n  WFA decomposition:")
                for k, c in enumerate(comps):
                    print(f"\n    Component {k}: λ = {c.weight:.6f}")
                    print_matrix(c.matrix, indent=8)
                    recon += c.matrix

                err = np.max(np.abs(mat - recon))
                cycle = sum(c.weight for c in comps)

                print("\n  Reconstructed matrix:")
                print_matrix(recon, indent=6)
                print(f"  Reconstruction error: {err:.2e}")
                print(f"  Cycle length: {cycle:.6f}")

    print("\n================ DONE ================")

    # --- Generate figures ---
    fig_frontier = plot_frontier_evolution(res)
    fig_tree = plot_split_tree_as_tree(tree_res.root)

    # --- Save figures with timestamp ---
    save_current_figures(
        plot_names=[
            "frontier_evolution",
            "split_tree_structure",
        ],
        base_dir="plots",
        dpi=200,
    )


if __name__ == "__main__":
    main()
