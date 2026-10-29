
import numpy as np
from src.utils.matrix_generator import generate_scaled_doubly_stochastic_matrix

def show_integer_examples():
    n = 256
    k = 512
    rng = np.random.default_rng(42)

    print(f"--- Permutation Sum Matrix (Integer) (N={n}, K={k}) ---")
    # Integer matrix (Avg val ~ K/N = 2)
    perm_sum_matrix = generate_scaled_doubly_stochastic_matrix(n, k, rng).astype(np.int64)
    
    # Show 10x10 corner
    sub = perm_sum_matrix[:10, :10]
    print("Top-left 10x10 subgrid:")
    print(sub)
    
    # Statistics
    min_val = perm_sum_matrix.min()
    max_val = perm_sum_matrix.max()
    mean_val = perm_sum_matrix.mean()
    sparsity = np.count_nonzero(perm_sum_matrix) / (n*n)
    
    print(f"\nStats:")
    print(f"  Min Value: {min_val}")
    print(f"  Max Value: {max_val}")
    print(f"  Mean Value: {mean_val:.2f} (Expected: {k}/{n} = {k/n:.2f})")
    print(f"  Sparsity (Non-zeros): {sparsity:.1%}")
    print("-" * 40)

if __name__ == "__main__":
    show_integer_examples()
