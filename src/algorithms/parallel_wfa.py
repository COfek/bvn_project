import numpy as np
import threading
from typing import List, Tuple
from numpy.typing import NDArray

BoolMatrix = NDArray[np.bool_]


class PWFAWorker(threading.Thread):
    """
    Worker thread: scans assigned diagonals, records potential matches,
    waits for synchronization, then proceeds to next step.
    """
    def __init__(
        self,
        worker_id: int,
        diagonals: List[int],
        mask: BoolMatrix,
        row_free: NDArray[np.bool_],
        col_free: NDArray[np.bool_],
        barrier: threading.Barrier,
        result_pool: List[List[Tuple[int, int]]],
        step_ref: List[int]
    ):
        super().__init__()
        self.worker_id = worker_id
        self.diagonals = diagonals
        self.mask = mask
        self.n = mask.shape[0]
        self.row_free = row_free
        self.col_free = col_free
        self.barrier = barrier
        self.result_pool = result_pool
        self.step_ref = step_ref  # shared mutable "current step"

    def run(self):
        while True:
            step = self.step_ref[0]
            if step >= len(self.diagonals):
                break

            diag = self.diagonals[step]
            potential = []

            # --- Scan diagonal for candidate matches ---
            for i in range(self.n):
                j = diag - i
                if 0 <= j < self.n:
                    if self.mask[i, j]:
                        potential.append((i, j))

            # Store results for this step
            self.result_pool[self.worker_id] = potential

            # --- Synchronize: wait for all threads to finish scanning ---
            self.barrier.wait()

            # --- Wait again: after main thread commits matches ---
            self.barrier.wait()


def wfa_parallel_threaded_synchronized(mask: BoolMatrix, num_threads: int = 4) -> List[Tuple[int, int]]:
    """
    Fully synchronized parallel WFA:
    - Threads scan diagonals in lock-step
    - Main thread resolves conflicts and commits matches
    - Ensures strict synchronization and correctness
    """
    n = mask.shape[0]
    K = 2 * n - 1

    # Split diagonals among threads (round-robin partition)
    diag_groups = [[] for _ in range(num_threads)]
    for idx in range(K):
        diag_groups[idx % num_threads].append(idx)

    # Shared state
    row_free = np.ones(n, dtype=bool)
    col_free = np.ones(n, dtype=bool)
    matches: List[Tuple[int, int]] = []

    # Synchronization tools
    barrier = threading.Barrier(num_threads + 1)
    result_pool = [[] for _ in range(num_threads)]
    step_refs = [0]  # mutable int for controlling thread steps

    # Create threads
    workers = [
        PWFAWorker(i, diag_groups[i], mask, row_free, col_free,
                   barrier, result_pool, step_refs)
        for i in range(num_threads)
    ]

    # Start workers
    for w in workers:
        w.start()

    # --- Main arbitration loop ---
    max_steps = max(len(g) for g in diag_groups)

    for step in range(max_steps):
        step_refs[0] = step

        # Wait for all workers to finish scanning
        barrier.wait()

        # --- ARBITRATION: main thread resolves conflicts ---
        for worker_res in result_pool:
            for (i, j) in worker_res:
                if row_free[i] and col_free[j]:
                    matches.append((i, j))
                    row_free[i] = False
                    col_free[j] = False
                    break  # only one match per diagonal

        # Signal workers to continue
        barrier.wait()

    # Join workers
    for w in workers:
        w.join()

    return matches
