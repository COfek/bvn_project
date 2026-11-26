"""Why Radix is not monotone in B on the traffic model: the digit planes of the TM dense matrices.

Regenerates the 10 TM dense matrices of tm_workload_experiment.py (same seed 42, same warm-up
matrix discarded first) and, for B = 16 and B = 8, prints per digit plane k (weight B^k): the
number of non-zero entries, the range of the row sums, the largest row or column sum S*, and the
term B^k S* of the Radix cycle bound (Proposition prop:radixbound), together with the number of
configurations Hungarian needs for the plane. Means over the 10 matrices. No timing is taken:
the times quoted in the paper come from tm_results.csv.
"""
from __future__ import annotations
from _paths import REPO
import sys
import numpy as np
sys.path.insert(0, str(REPO))
from src.utils.griner_avin_tm import tm_integer
from src.algorithms.bvn import bvn_decomposition_timed

N, N_F, N_MATRICES = 256, 512, 10
rng = np.random.default_rng(42)
tm_integer(N, N_F, rng, w_small=7)                # the campaign's warm-up matrix, discarded
mats = [tm_integer(N, N_F, rng, w_small=7) for _ in range(N_MATRICES)]
S = np.mean([M.sum(1)[0] for M in mats])
print(f"TM dense: {len(mats)} matrices, mean S = {S:.1f}, "
      f"mean non-zeros {np.mean([(M > 0).sum() for M in mats]):.0f}, "
      f"largest entry {max(M.max() for M in mats):.0f}")
print("unsplit Hungarian N:", np.mean([len(bvn_decomposition_timed(M.copy(), "maximum")[0])
                                       for M in mats]))
for B in (16, 8):
    K = max(int(np.floor(np.log(M.max()) / np.log(B))) + 1 for M in mats)
    bound = np.zeros(len(mats))
    print(f"--- B = {B}: {K} planes")
    for k in range(K):
        nnz, lo, hi, sst, nk = [], [], [], [], []
        for i, M in enumerate(mats):
            P = (M // B ** k) % B
            r, c = P.sum(1), P.sum(0)
            nnz.append((P > 0).sum()); lo.append(r.min()); hi.append(r.max())
            sst.append(max(r.max(), c.max()))
            bound[i] += B ** k * sst[-1]
            nk.append(len(bvn_decomposition_timed(P.copy(), "maximum")[0]) if P.any() else 0)
        print(f"  plane {k} (x{B ** k:>3}): non-zeros {np.mean(nnz):7.0f}  row sums "
              f"{np.mean(lo):6.1f}-{np.mean(hi):6.1f}  S* {np.mean(sst):6.1f}  "
              f"B^k S* {B ** k * np.mean(sst):7.0f}  Hungarian N {np.mean(nk):6.1f}")
    print(f"  bound sum_k B^k S* = {bound.mean():.0f} = {bound.mean() / S:.3f} S")
