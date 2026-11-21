"""Augment ablation, untimed stage: the GW engines with and without Augment at the BvN baseline.

Chen (meeting of 2026-11-12) asked why the scanning engines repair with Augment, which costs
time. This stage measures what Augment buys: the cycle length C and the number of
configurations N of GW Static and GW Dynamic, with Augment ('heavy_static', 'heavy') and without
('heavy_static_noaug', 'heavy_noaug', the scan alone), on the paper's dense workload.

Matrices: n=256, k=256, max weight 64, matrix i generated from seed 42 + i, exactly as
src/runner.py does; matrices FIRST .. LAST-1 (default 5 .. 999, the 995 the paper keeps).
No timing here: computation time is the timed stage, run separately on a free machine.

Output: run/augment_ablation_2026_11/untimed.csv (one row per matrix and engine) and a summary
printed to stdout.
Usage:  .venv/Scripts/python.exe agent_scripts/augment_ablation.py [FIRST] [LAST]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from _paths import REPO  # repository-relative paths

sys.path.insert(0, str(REPO))
from src.algorithms.bvn import bvn_decomposition_timed  # noqa: E402
from src.utils.matrix_generator import generate_matrix  # noqa: E402

N_DIM, K, MAXW, SEED = 256, 256, 64.0, 42
ENGINES = ["heavy_static", "heavy_static_noaug", "heavy", "heavy_noaug"]
FIRST = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LAST = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
OUT = REPO / "run" / "augment_ablation_2026_11"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for i in range(FIRST, LAST):
        D = generate_matrix(N_DIM, K, MAXW, np.random.default_rng(SEED + i))
        S = float(D.sum(axis=1)[0])
        for eng in ENGINES:
            comps, _ = bvn_decomposition_timed(D.copy(), eng, step_strategy="min")
            C = float(sum(c.weight for c in comps))
            rows.append(dict(matrix_index=i, engine=eng, S=S, N=len(comps), C=C,
                             C_over_S=C / S))
        if (i - FIRST) % 50 == 0:
            print(f"matrix {i} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "untimed.csv", index=False)
    summary = df.groupby("engine")[["N", "C", "C_over_S"]].agg(["mean", "min", "max"])
    print(summary.loc[ENGINES].round(3).to_string())


if __name__ == "__main__":
    main()
