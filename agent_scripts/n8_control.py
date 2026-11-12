"""Control for the MoE paragraph "The weak engine is penalised less" (Section 6.5).

The MoE traces have n = 8, the generated campaigns n = 256. To separate the effect of the size
from that of the traffic, the generated workload is repeated at n = 8 and the BvN baseline's
cycle ratio C(D) / S is measured for every engine.

Untimed: only the cycle length and the configuration count are recorded, so the machine's load
does not matter. Matrices as in the campaigns (generate_matrix, max_weight = 64, seed 42 + i,
matrices 5.. since the paper discards the first 5); k = n = 8 and, for comparison, k = 256 (the
campaign's permutation count).

Usage:  .venv/Scripts/python.exe agent_scripts/n8_control.py [n_matrices]
Output: euler_exp/n8_control/results.csv, summary.csv and a printed summary.
"""
from __future__ import annotations
from _paths import REPO  # repository-relative paths
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(REPO))
from src.algorithms.bvn import bvn_decomposition_timed  # noqa: E402
from src.utils.matrix_generator import generate_matrix  # noqa: E402

OUT = os.path.join(str(REPO), "euler_exp", "n8_control")
ENGINES = [("Hungarian", "maximum"), ("GW Dynamic", "heavy"), ("GW Static", "heavy_static"),
           ("WFA", "wfa"), ("WWFA", "wwfa")]
N = 8
KS = (8, 256)
N_MATRICES = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def line_max(M):
    return float(max(M.sum(axis=0).max(), M.sum(axis=1).max()))


rows = []
for k in KS:
    for i in range(5, 5 + N_MATRICES):
        M = generate_matrix(N, k=k, max_weight=64.0, rng=np.random.default_rng(42 + i))
        S = line_max(M)
        for name, key in ENGINES:
            comps, _ = bvn_decomposition_timed(M.copy(), matching_algorithm=key)
            C = float(sum(c.weight for c in comps))
            recon = sum(c.weight * c.permutation for c in comps)
            rows.append(dict(n=N, k=k, matrix=i, engine=name, C=C, N=len(comps), S=S,
                             ratio=C / S, recon_err=float(np.abs(recon - M).max())))
    print(f"k={k} done", flush=True)

os.makedirs(OUT, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "results.csv"), index=False)
summary = (df.groupby(["k", "engine"])
             .agg(ratio_mean=("ratio", "mean"), ratio_min=("ratio", "min"),
                  ratio_max=("ratio", "max"), N_mean=("N", "mean"),
                  recon_err_max=("recon_err", "max"), count=("ratio", "count"))
             .round(4))
summary.to_csv(os.path.join(OUT, "summary.csv"))
pd.set_option("display.width", 160)
print(summary)
