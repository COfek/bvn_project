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
    Worker for maximum bipartite matching.
    """
    bit_value = 1 << bit_index
    bitplane = ((int_matrix & bit_value) >> bit_index).astype(bool)

    if not np.any(bitplane):
        return bit_index, []

    graph = nx.Graph()

    row_nodes = [f"r{i}" for i in range(n)]
    col_nodes = [f"c{j}" for j in range(n)]

    graph.add_nodes_from(row_nodes, bipartite=0)
    graph.add_nodes_from(col_nodes, bipartite=1)

    for i in range(n):
        for j in range(n):
            if bitplane[i, j]:
                graph.add_edge(f"r{i}", f"c{j}")

    weight = bit_value / scale
    components: List[BvnComponent] = []

    while graph.number_of_edges() > 0:
        matching = maximum_matching(graph, top_nodes=row_nodes)
        perm = np.zeros((n, n), dtype=float)

        for r in row_nodes:
            if r in matching:
                c = matching[r]
                if c.startswith("c"):
                    perm[int(r[1:]), int(c[1:])] = 1.0

        if not np.any(perm):
            break

        components.append(BvnComponent(permutation=perm, weight=weight))

        edges_to_remove = [
            (f"r{i}", f"c{j}")
            for i in range(n)
            for j in range(n)
            if perm[i, j] == 1
        ]
        graph.remove_edges_from(edges_to_remove)

    return bit_index, components


def bitplane_decomposition_maximum(
    matrix: FloatMatrix,
    bits: int = 8,
    tol: float = 0.0,
    max_workers: int | None = None,
) -> List[BvnComponent]:
    """
    Bit-plane decomposition using MAXIMUM matching (optimal cardinality).
    """
    n = matrix.shape[0]
    scale = 2 ** bits
    int_matrix = np.round(matrix * scale).astype(np.int64)

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_bitplane_maximum,
                bit_idx,
                int_matrix,
                n,
                scale
            ): bit_idx
            for bit_idx in range(bits)
        }

        for future in as_completed(futures):
            idx, comps = future.result()
            results.append((idx, comps))

    results.sort(key=lambda x: x[0])

    flat = []
    for _, comps in results:
        flat.extend(comps)

    return flat
