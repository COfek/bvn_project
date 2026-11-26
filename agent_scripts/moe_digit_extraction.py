"""Time of the Radix digit extraction on the measured MoE traces (n = 8).

Explains why Radix saves the scanning engines no time on these traces (Chen K16, K19): the
digit extraction, which T_parallel charges in full, costs about as much as the whole compiled
decomposition of an 8 x 8 matrix. For each of the 12 traces of moe_trace_experiment.py and each
base, times extract_digit_planes (the call decompose_radix times) as in the campaign: 9 runs,
the first 4 discarded, the median kept. Writes moe_digit_extraction.csv next to moe_results.csv
and prints the mean per family and base, with the campaign's unsplit times and Hungarian's time
per extraction for comparison.
Run with BVN_CAMPAIGN=2026_11 BVN_RUN_ROOT=run/rerun_2026_11e, on an otherwise idle machine.
"""
from __future__ import annotations
from _paths import REPO
import os
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, str(REPO))
from src.algorithms.radix_decomposition import extract_digit_planes
from src.utils.moe_trace import load_all
import moe_trace_experiment as mt

RUN = os.environ.get("BVN_RUN_ROOT", "run")
OUT = os.path.join(str(REPO), RUN, "moe_workload")
BASES = [16, 8, 4, 2]
REPEATS, WARMUP = mt.REPEATS, mt.WARMUP

rows = []
for t in load_all(mt.TRACES):
    M = t.matrix.astype(np.float64)
    regime = "MoE mmlu" if "mmlu" in t.dataset else "MoE throughput"
    for B in BASES:
        ts = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            planes = extract_digit_planes(M.copy(), B)
            ts.append((time.perf_counter() - t0) * 1e6)
        rows.append(dict(regime=regime, matrix=t.label, base=B, planes=len(planes),
                         extract_us=float(np.median(ts[WARMUP:]))))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "moe_digit_extraction.csv"), index=False)

print(df.groupby(["regime", "base"])[["planes", "extract_us"]].mean().round(1).to_string())
print("overall digit extraction:", f"{df.extract_us.min():.0f}-{df.extract_us.max():.0f} us")
res = pd.read_csv(os.path.join(OUT, "moe_results.csv"))
u = res[res.config == "BvN"].groupby("engine").runtime_ms.mean() * 1000
print("campaign, unsplit decomposition [us]:", u.round(0).to_dict())
h = res[(res.config == "BvN") & (res.engine == "Hungarian")]
print(f"campaign, Hungarian unsplit time per extraction: {1000 * (h.runtime_ms / h.N).mean():.1f} us")
