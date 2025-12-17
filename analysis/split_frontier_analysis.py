from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Union

import numpy as np
from numpy.typing import NDArray

from src.algorithms.split_tree import (
    float_matrix,
    pschedule,
    random_binary_split,
    verify_reconstruction,
    decompose_leaf_with_wfa,
)


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class FrontierStepReport:
    step: int
    num_matrices: int
    total_nnz: int
    total_permutations: int
    total_cycle_length: float
    max_reconstruction_error: float


@dataclass
class SplitFrontierAnalysisResult:
    steps: List[FrontierStepReport]
    final_frontier: List[float_matrix]
    reconstruction_ok: bool


# ---------------------------------------------------------------------
# Core frontier analysis
# ---------------------------------------------------------------------

def analyze_split_frontier(
    x: float_matrix,
    sparsity_target: int,
    max_depth: int,
    p_schedule: pschedule,
    tol: float = 1e-12,
) -> SplitFrontierAnalysisResult:
    """
    Analyze split-tree behavior using *frontier-based* steps.

    Step 0:
        frontier = [x]

    Each step:
        - Split every matrix in the frontier that is still splittable
        - Keep unsplittable matrices unchanged
        - The resulting set is the new frontier

    At each step:
        - Run WFA decomposition on *every matrix in the frontier*
        - Aggregate number of permutations and cycle length
        - Verify reconstruction correctness

    Returns:
        A SplitFrontierAnalysisResult containing per-step statistics.
    """
    frontier: List[float_matrix] = [x.copy()]
    reports: List[FrontierStepReport] = []

    for step in range(max_depth + 1):
        # --------------------------------------------------
        # Analyze current frontier
        # --------------------------------------------------
        total_nnz = 0
        total_perm = 0
        total_cycle = 0.0
        max_err = 0.0

        for mat in frontier:
            total_nnz += int(np.count_nonzero(mat))

            comps = decompose_leaf_with_wfa(mat, tol=tol)
            total_perm += len(comps)
            total_cycle += float(sum(c.weight for c in comps))

            recon = np.zeros_like(mat)
            for c in comps:
                recon += c.matrix

            err = float(np.max(np.abs(mat - recon)))
            max_err = max(max_err, err)

        reports.append(
            FrontierStepReport(
                step=step,
                num_matrices=len(frontier),
                total_nnz=int(total_nnz),
                total_permutations=int(total_perm),
                total_cycle_length=float(total_cycle),
                max_reconstruction_error=float(max_err),
            )
        )

        # --------------------------------------------------
        # Build next frontier
        # --------------------------------------------------
        next_frontier: List[float_matrix] = []
        any_split = False

        for mat in frontier:
            nnz = int(np.count_nonzero(mat))

            # Stop splitting if sparse enough
            if nnz <= sparsity_target:
                next_frontier.append(mat)
                continue

            # Choose p
            if callable(p_schedule):
                p = float(p_schedule(mat, step))
            else:
                p = float(p_schedule)

            if not (0.0 < p < 1.0):
                raise ValueError(f"Invalid p={p} at step={step}")

            left, right = random_binary_split(mat, p)

            if np.count_nonzero(left) > 0:
                next_frontier.append(left)
            if np.count_nonzero(right) > 0:
                next_frontier.append(right)

            any_split = True

        frontier = next_frontier

        # If nothing was split this round, we're done
        if not any_split:
            break

    # --------------------------------------------------
    # Final reconstruction check
    # --------------------------------------------------
    recon_ok = verify_reconstruction(x, frontier, tol=tol)

    return SplitFrontierAnalysisResult(
        steps=reports,
        final_frontier=frontier,
        reconstruction_ok=bool(recon_ok),
    )
