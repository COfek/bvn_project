from __future__ import annotations
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from .wfa import select_matching


@dataclass
class RadixComponent:
    matrix: np.ndarray
    weight: float


def decompose_radix(
        matrix: np.ndarray,
        base: int = 8,
        precision_bits: int = 16,  # Now used below
        tol: float = 1e-12,
        max_workers: int | None = None
) -> List[RadixComponent]:
    max_val = np.max(matrix)
    if max_val < tol:
        return []
    # We scale the matrix into integer space based on the precision requested.
    # Higher bits = more digit planes = higher accuracy.
    scaling_factor = 2 ** precision_bits
    scaled_matrix = np.round((matrix / max_val) * scaling_factor).astype(np.int64)

    # Calculate how many planes are needed for this base
    num_planes = int(np.ceil(np.log(scaling_factor) / np.log(base)))

    planes: List[Tuple[float, np.ndarray]] = []
    temp_matrix = scaled_matrix.copy()

    for d in range(num_planes):
        unit_weight = (base ** d) * (max_val / scaling_factor)
        digit_plane = temp_matrix % base

        if np.any(digit_plane > 0):
            planes.append((unit_weight, digit_plane.astype(np.float64)))

        temp_matrix //= base
        if np.all(temp_matrix == 0):
            break

    all_components: List[RadixComponent] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_plane = {
            executor.submit(_decompose_digit_plane, plane_matrix, weight, tol): weight
            for weight, plane_matrix in planes
        }

        for future in as_completed(future_to_plane):
            try:
                all_components.extend(future.result())
            except Exception as exc:
                print(f"Radix plane worker failed: {exc}")

    return all_components


def _decompose_digit_plane(
        plane: np.ndarray,
        unit_weight: float,
        tol: float
) -> List[RadixComponent]:
    x = plane.copy()
    components: List[RadixComponent] = []

    while True:
        mask = x > tol
        if not mask.any():
            break

        matches, _ = select_matching(mask)
        if not matches:
            break

        digit_step = min(x[i, j] for (i, j) in matches)
        actual_weight = digit_step * unit_weight

        p = np.zeros_like(x)
        for (i, j) in matches:
            p[i, j] = 1.0
            x[i, j] -= digit_step

        components.append(RadixComponent(matrix=actual_weight * p, weight=actual_weight))

    return components