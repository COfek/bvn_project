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
    # Extract the bitplane using bitwise logic
    bitplane = ((int_matrix & bit_value) >> bit_index).astype(bool)

    if not np.any(bitplane):
        return bit_index, []

    mask = bitplane.copy()
    weight = bit_value / scale
    components: List[BvnComponent] = []

    # Repeatedly extract maximal matchings until the bitplane mask is cleared
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
        tol: float = 0.0,
        max_workers: int | None = None,
) -> List[BvnComponent]:
    """
    Bit-plane decomposition using WAVEFRONT ARBITER (maximal matching)
    with Dynamic Bit-Depth optimization.
    """
    n = matrix.shape[0]
    scale = 2 ** bits
    # Quantize the float matrix into an integer matrix
    int_matrix = np.round(matrix * scale).astype(np.int64)

    # DYNAMIC BIT-DEPTH OPTIMIZATION
    # Find the maximum value to determine the actual number of bits needed.
    max_val = np.max(int_matrix)
    if max_val <= 0:
        return []

    # Calculate the number of bits required to represent max_val.
    actual_bits = int(max_val).bit_length()

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Only submit futures for bit-planes that actually contain data.
        futures = {
            executor.submit(
                _process_single_bitplane_maximal,
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

    # Sort results to maintain bit-plane order
    results.sort(key=lambda x: x[0])

    flat_components = []
    for _, comps in results:
        flat_components.extend(comps)

    return flat_components