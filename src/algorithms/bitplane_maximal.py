from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from .wfa import wavefront_matching
from .bvn import BvnComponent

BoolMatrix = NDArray[np.bool_]
FloatMatrix = NDArray[np.float64]


def _process_single_bitplane_maximal(
    bit_index: int,
    int_matrix: NDArray[np.int64],
    n: int,
    scale: int,
) -> Tuple[int, List[BvnComponent]]:
    """
    Worker for a single WFA bitplane.
    """
    bit_value = 1 << bit_index
    bitplane = ((int_matrix & bit_value) >> bit_index).astype(bool)

    if not np.any(bitplane):
        return bit_index, []

    mask = bitplane.copy()
    weight = bit_value / scale
    components: List[BvnComponent] = []

    while True:
        matches = wavefront_matching(mask)
        if not matches:
            break

        perm = np.zeros((n, n), dtype=float)
        for i, j in matches:
            perm[i, j] = 1.0
            mask[i, j] = False

        components.append(BvnComponent(permutation=perm, weight=weight))

    return bit_index, components


def bitplane_decomposition_maximal(
    matrix: FloatMatrix,
    bits: int = 8,
    max_workers: int | None = None,
) -> List[BvnComponent]:
    """
    Bit-plane decomposition using WAVEFRONT ARBITER (maximal matching).
    """
    n = matrix.shape[0]
    scale = 2 ** bits
    int_matrix = np.round(matrix * scale).astype(np.int64)

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_bitplane_maximal,
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

    flat_components = []
    for _, comps in results:
        flat_components.extend(comps)

    return flat_components
