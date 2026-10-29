import time
import numpy as np
from src.utils.matrix_generator import random_sparse_doubly_stochastic
from src.algorithms.sorted_array_matching import _jit_decompose_sorted

if __name__ == "__main__":
    n = 256
    rng = np.random.default_rng(42)
    
    print("Generating noisy floating-point matrix...")
    matrix1 = random_sparse_doubly_stochastic(n=n, density=0.5, iters=20, rng=rng)
    matrix2 = matrix1.copy()

    # Pre-warm JIT
    print("Warming up JIT...")
    _jit_decompose_sorted(matrix1.copy())
    
    start_time = time.perf_counter()
    weights, matches = _jit_decompose_sorted(matrix2)
    runtime = time.perf_counter() - start_time
    
    print(f"Algorithm finished in {runtime:.4f} seconds")
    print(f"Total iterations (Permutations stripped): {len(weights)}")
    
    rebuilt_matrix = np.zeros_like(matrix2)
    for w, mlist in zip(weights, matches):
        for r, c in mlist:
            rebuilt_matrix[r, c] += w
            
    # Correctness assert
    diff = np.max(np.abs(matrix1 - rebuilt_matrix))
    print(f"Max reconstruction error: {diff:.2e} (Pass: {diff < 1e-5})")
