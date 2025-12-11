import numpy as np
import threading
from typing import List, Tuple
from numpy.typing import NDArray

BoolMatrix = NDArray[np.bool_]


class PWFAWorker(threading.Thread):
    """
    Worker thread: scans assigned diagonals, records potential matches,
    waits for synchronization with the main thread.
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
        step_ref: List[int],
        max_steps: int,
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
        self.step_ref = step_ref
        self.max_steps = max_steps

    def run(self):
        while True:
            step = self.step_ref[0]

            if step >= self.max_steps:
                print(f"[worker {self.worker_id}] exiting at step={step}")
                break

            potential = []

            # Only scan if this worker has a diagonal at this step
            if step < len(self.diagonals):
                diagonal = self.diagonals[step]
                # Scan diagonal
                for i in range(self.n):
                    j = diagonal - i
                    if 0 <= j < self.n:
                        if self.mask[i, j]:
                            potential.append((i, j))

                print(f"[worker {self.worker_id}] step={step} scanned diag={self.diagonals[step]} "
                      f"found {len(potential)} potentials")
            else:
                print(f"[worker {self.worker_id}] step={step} has no diagonal assigned")

            self.result_pool[self.worker_id] = potential

            # First barrier
            print(f"[worker {self.worker_id}] waiting at barrier 1 (step={step})")
            self.barrier.wait()
            print(f"[worker {self.worker_id}] passed barrier 1 (step={step})")

            # Second barrier
            print(f"[worker {self.worker_id}] waiting at barrier 2 (step={step})")
            self.barrier.wait()
            print(f"[worker {self.worker_id}] passed barrier 2 (step={step})")


def wfa_parallel_threaded_synchronized(mask: BoolMatrix, num_threads: int = 4) -> List[Tuple[int, int]]:
    n = mask.shape[0]
    k = 2 * n - 1

    # Partition diagonals
    diag_groups = [[] for _ in range(num_threads)]
    for idx in range(k):
        diag_groups[idx % num_threads].append(idx)

    max_steps = max(len(g) for g in diag_groups)

    row_free = np.ones(n, dtype=bool)
    col_free = np.ones(n, dtype=bool)
    matches: List[Tuple[int, int]] = []

    barrier = threading.Barrier(num_threads + 1)
    result_pool = [[] for _ in range(num_threads)]
    step_ref = [0]

    workers = [
        PWFAWorker(
            worker_id=i,
            diagonals=diag_groups[i],
            mask=mask,
            row_free=row_free,
            col_free=col_free,
            barrier=barrier,
            result_pool=result_pool,
            step_ref=step_ref,
            max_steps=max_steps,
        )
        for i in range(num_threads)
    ]

    for w in workers:
        print(f"[main] starting worker {w.worker_id}")
        w.start()

    # MAIN LOOP
    for step in range(max_steps):
        step_ref[0] = step
        print(f"\n[main] === step {step} === waiting for barrier 1 ===")
        barrier.wait()
        print(f"[main] step {step} passed barrier 1")

        # Arbitration
        for worker_res in result_pool:
            for (i, j) in worker_res:
                if row_free[i] and col_free[j]:
                    matches.append((i, j))
                    row_free[i] = False
                    col_free[j] = False
                    print(f"[main] committed match ({i},{j})")
                    break

        print(f"[main] step {step} waiting barrier 2")
        barrier.wait()
        print(f"[main] step {step} passed barrier 2")

    for w in workers:
        w.join()
        print(f"[main] worker {w.worker_id} joined")

    print("[main] done with WFA")
    return matches
