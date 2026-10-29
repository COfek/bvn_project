import numpy as np
from src.utils.matrix_generator import generate_weighted_sum_matrix

def debug_matrix():
    n = 256
    weights = [3, 5, 7]
    sub_k = 10
    rng = np.random.default_rng(42)
    
    print(f"Generating matrix with weights={weights}, sub_k={sub_k}")
    matrix = generate_weighted_sum_matrix(n, weights, sub_k, rng)
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Max value: {matrix.max()}")
    print(f"Min value: {matrix.min()}")
    
    unique_vals, counts = np.unique(matrix, return_counts=True)
    print("\nValue Distribution:")
    for v, c in zip(unique_vals, counts):
        if c > 0:
            print(f"  Value {v}: {c} instances")
            
    # Check if values > 15 exist. 
    # Max possible value should be roughly sum(weights * sub_k) if all perms overlap?
    # No, weights are coefficients. 
    # M = 3*Layer1 + 5*Layer2 + 7*Layer3.
    # Each layer is a sum of 'sub_k' permutations.
    # So max value could be very high if many permutations overlap at the same cell.
    
    # But for Base 16 to be efficient, we wanted the VALUES to be decomposable?
    # No, we wanted the matrix to be representable as sum(d_i * P_i) where d_i are digits. 
    
    # The hypothesis was: Sum of weights 3+5+7 = 15.
    # If we have 1 permutation of each, M_ij is in {0, 3, 5, 7, 8, 10, 12, 15}.
    # All these are < 16. So Base 16 should peel them in 1 go?
    # BUT, we used sub_k=10. 
    # So we have 10 permutations of weight 3, 10 of weight 5, 10 of weight 7.
    # Total weight = 10*3 + 10*5 + 10*7 = 150.
    # Average element value = 150 / N = 150/256 < 1. 
    # But since it's sparse, non-zero entries will be higher.
    
    # If M_ij = 30 (e.g. 10 overlaps of weight 3), then 30 in base 16 is 1E (1*16 + 14).
    # This requires 2 planes.
    
    # To test the "single plane" hypothesis, we needed sub_k=1?
    pass

if __name__ == "__main__":
    debug_matrix()
