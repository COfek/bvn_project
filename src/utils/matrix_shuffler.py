import numpy as np
from typing import Tuple, List, Optional
from ..algorithms.bvn import DecompositionComponent

class MatrixShuffler:
    """
    A utility class to shuffle a matrix's rows and columns and map results back to original indices.
    """
    def __init__(self, matrix: np.ndarray, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.n = matrix.shape[0]
        
        # Create permutations
        self.row_perm = self.rng.permutation(self.n)
        self.col_perm = self.rng.permutation(self.n)
        
        # Create inverse permutations for mapping back
        # if row_perm[i] = k, then inv_row_perm[k] = i
        self.inv_row_perm = np.zeros(self.n, dtype=int)
        self.inv_col_perm = np.zeros(self.n, dtype=int)
        self.inv_row_perm[self.row_perm] = np.arange(self.n)
        self.inv_col_perm[self.col_perm] = np.arange(self.n)
        
        # Create the shuffled view
        # M'[i, j] = M[row_perm[i], col_perm[j]]
        self.shuffled_matrix = matrix[self.row_perm][:, self.col_perm]

    def get_shuffled_matrix(self) -> np.ndarray:
        return self.shuffled_matrix

    def map_matching_to_original(self, matching_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Maps a list of (row, col) pairs from the shuffled matrix back to the original matrix indices.
        """
        original_pairs = []
        for r_prime, c_prime in matching_pairs:
            r = self.row_perm[r_prime]
            c = self.col_perm[c_prime]
            original_pairs.append((r, c))
        return original_pairs

    def unshuffle_component(self, component: DecompositionComponent) -> DecompositionComponent:
        """
        Takes a DecompositionComponent computed on the shuffled matrix and returns
        a new component with the permutation mapped back to original indices.
        """
        # component.permutation is a dense array P where P[i] = j means row i matches column j
        # In sparse representation (which we might need if component uses it), it's list of pairs.
        # But DecompositionComponent usually stores a dense geometric permutation or a sparse dict.
        # Let's check DecompositionComponent definition.
        
        # Assuming component.permutation is a dict {row: col} or list of (row, col)
        # Actually in bvn.py: perm: Dict[int, int]
        
        shuffled_perm = component.permutation
        original_perm = {}
        
        for r_prime, c_prime in shuffled_perm.items():
            r = self.row_perm[r_prime]
            c = self.col_perm[c_prime]
            original_perm[r] = c
            
        return DecompositionComponent(weight=component.weight, permutation=original_perm)
