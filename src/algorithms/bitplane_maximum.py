from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
import networkx as nx

from networkx.algorithms.bipartite.matching import maximum_matching
from .bvn import BvnComponent

FloatMatrix = NDArray[np.float64]
BoolMatrix = NDArray[np.bool_]


def _process_single_bitplane_maximum(
        bit_index: int,
        int_matrix: NDArray[np.int64],
        n: int,
        scale: int,
) -> Tuple[int, List[BvnComponent]]:
    """
    Worker for maximum bipartite matching on a specific bit-plane.
    """
    bit_value = 1 << bit_index
    # Extract the bitplane using bitwise logic
    bitplane = ((int_matrix & bit_value) >> bit_index).astype(bool)

    if not np.any(bitplane):
        return bit_index, []

    # Build the bipartite graph for this specific bit-plane
    graph = nx.Graph()
    row_nodes = [f"r{i}" for i in range(n)]
    col_nodes = [f"c{j}" for j in range(n)]
    graph.add_nodes_from(row_nodes, bipartite=0)
    graph.add_nodes_from(col_nodes, bipartite=1)

    # Add edges where the bit is active
    rows, cols = np.where(bitplane)
    for r, c in zip(rows, cols):
        graph.add_edge(f"r{r}", f"c{c}")

    weight = bit_value / scale
    components: List[BvnComponent] = []

    # Iteratively extract maximum matchings until the bit-plane is empty
    while graph.number_of_edges() > 0:
        matching = maximum_matching(graph, top_nodes=row_nodes)

        # Convert the matching dictionary back to a permutation matrix
        perm = np.zeros((n, n), dtype=float)
        edges_to_remove = []

        for r_node, c_node in matching.items():
            if r_node.startswith("r"):  # maximum_matching returns entries for both sides
                r_idx = int(r_node[1:])
                c_idx = int(c_node[1:])
                perm[r_idx, c_idx] = 1.0
                edges_to_remove.append((r_node, c_node))

        if not np.any(perm):
            break

        components.append(BvnComponent(permutation=perm, weight=weight))
        graph.remove_edges_from(edges_to_remove)

    return bit_index, components


def bitplane_decomposition_maximum(
        matrix: FloatMatrix,
        bits: int = 8,
        tol: float = 0.0,
        max_workers: int | None = None,
) -> List[BvnComponent]:
    """
    Bit-plane decomposition using MAXIMUM matching with Dynamic Bit-Depth optimization.
    """
    n = matrix.shape[0]
    scale = 2 ** bits
    # Quantize the float matrix into an integer matrix
    int_matrix = np.round(matrix * scale).astype(np.int64)

    # DYNAMIC BIT-DEPTH OPTIMIZATION
    # Instead of range(bits), we find the Most Significant Bit actually present.
    max_val = np.max(int_matrix)
    if max_val <= 0:
        return []

    # bit_length() gives us the number of bits required to represent the max integer.
    actual_bits = int(max_val).bit_length()

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Only launch threads for the bits that actually contain data.
        futures = {
            executor.submit(
                _process_single_bitplane_maximum,
                bit_idx,
                int_matrix,
                n,
                scale
            ): bit_idx
            for bit_idx in range(actual_bits)
        }

        for future in as_completed(futures):
            idx, comps = future.result()
            results.append((idx, comps))

    # Sort results by bit index to ensure deterministic output order
    results.sort(key=lambda x: x[0])

    flat = []
    for _, comps in results:
        flat.extend(comps)

    return flat