import numpy as np
from src.utils.matrix_generator import generate_matrix

# Parameters from your command line
N = 256
K = 2048
MAX_WEIGHT = 64
BASE_SEED = 42

def view_sample(sample_index: int):
    """View the matrix for a specific sample index (0 to 999)"""
    rng = np.random.default_rng(BASE_SEED + sample_index)
    
    matrix = generate_matrix(
        n=N, 
        k=K, 
        max_weight=MAX_WEIGHT, 
        rng=rng, 
        float_weights=False
    )
    
    print(f"--- Matrix for sample {sample_index} (Seed: {BASE_SEED + sample_index}) ---")
    print(matrix)
    print(f"Shape: {matrix.shape}, Min: {matrix.min()}, Max: {matrix.max()}, Sum: {matrix.sum()}")
    print(f"Mean (Expected Value): {matrix.mean():.4f}")
    print(f"Variance: {matrix.var():.4f}")
    print(f"Standard Deviation: {matrix.std():.4f}")

if __name__ == "__main__":
    # Example: View the very first matrix (index 0)
    view_sample(0)
    
    # You can also view a specific one like the 500th matrix by calling view_sample(499)
