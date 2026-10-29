import time
import numpy as np
from src.utils.matrix_generator import random_sparse_doubly_stochastic
from scipy.optimize import linear_sum_assignment

def benchmark_bvn_solver(matrix, name, tol=1e-10):
    print(f"\n--- Testing on {name} Matrix ---")
    work = np.array(matrix, dtype=float, copy=True)
    start_time = time.perf_counter()
    
    iteration = 0
    while True:
        iteration += 1
        mask = work >= tol
        if not mask.any():
            break

        # Run the exact same Scipy Hungarian linear_sum_assignment from bvn.py
        cost = -work
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Filter negative or super tiny edges
        valid_edges = [(r, c) for r, c in zip(row_ind, col_ind) if work[r, c] >= tol]
        if not valid_edges:
            break
            
        selected_values = [work[r, c] for r, c in valid_edges]
        lambda_value = float(np.min(selected_values))
        
        for r, c in valid_edges:
            work[r, c] -= lambda_value

    runtime = time.perf_counter() - start_time
    print(f"Algorithm finished in {runtime:.4f} seconds")
    print(f"Total iterations (Permutations stripped): {iteration}")


if __name__ == "__main__":
    n = 256
    rng = np.random.default_rng(42)
    
    print("Generating noisy floating-point Sinkhorn matrix...")
    original_matrix = random_sparse_doubly_stochastic(
        n=n, density=0.5, iters=200, rng=rng
    )

    benchmark_bvn_solver(original_matrix, "RAW SINKHORN FLOAT")
    
    print("\nQuantizing matrix into scaled integers just like Radix does...")
    max_val = np.max(original_matrix)
    scaling_factor = 2 ** 16
    quantized_matrix = np.round((original_matrix / max_val) * scaling_factor).astype(np.int64)

    benchmark_bvn_solver(quantized_matrix, "QUANTIZED INTEGER")
