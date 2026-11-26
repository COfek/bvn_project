"""MoE traffic at n = 256 (Chen K18): the measured traces, upsampled, through every engine.

Each of the 12 measured 8x8 traces (paper_traces/) gives SAMPLES matrices of 256 x 256 by
src.utils.moe_trace.upsample (each GPU split into 32; every token picks a destination group with
the trace's probability, then one of its 32 GPUs uniformly; Ofek 2026-11-26), seeded per
(trace, sample). Every matrix runs the unsplit decomposition and Radix B = 16, 8, 4, 2 for the
four engines, with the measurement code of the n = 8 campaign (moe_trace_experiment.run_all):
T_parallel of Radix = digit extraction + slowest plane, C/S with S = the largest row or column
sum. No Euler: the matrices are not regular. One warm-up matrix runs first and is discarded
(Numba compilation); each configuration is then timed once (REPEATS = 1, Ofek 2026-11-26: as
the other n = 256 campaigns; the three samples per trace give the spread).
Results are appended matrix by matrix, so an interrupted run keeps what it finished.

Outputs (under BVN_RUN_ROOT, default run/):  moe256/moe256_results.csv, moe256/moe256_summary.csv
Run on an otherwise idle machine:
  BVN_CAMPAIGN=2026_11 BVN_RUN_ROOT=run/rerun_2026_11e python agent_scripts/moe256_campaign.py
Set MOE256_QUICK=1 for a check run (one trace, one sample, not written to the campaign files).
"""
from __future__ import annotations
from _paths import REPO
import os
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, str(REPO))
import moe_trace_experiment as mt
from src.utils.moe_trace import load_all, upsample

FACTOR, SAMPLES, REPEATS = 32, 3, 1           # one timing per run, as the other n = 256 campaigns
QUICK = os.environ.get("MOE256_QUICK") == "1"
OUTDIR = os.path.join(str(REPO), os.environ.get("BVN_RUN_ROOT", "run"), "moe256")
OUT = os.path.join(OUTDIR, "moe256_quick.csv" if QUICK else "moe256_results.csv")
mt.REPEATS, mt.WARMUP = REPEATS, 0          # warm-up is the discarded matrix below
mt.ENGINES = [e for e in mt.ENGINES if e[0] != "WFA"]   # the paper's four engines


def matrix_for(trace_idx, sample, trace):
    rng = np.random.default_rng(1000 * trace_idx + sample)
    return upsample(trace.matrix, FACTOR, rng).astype(np.float64)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    traces = load_all(mt.TRACES)
    jobs = [(i, s) for i in range(len(traces)) for s in range(SAMPLES)]
    if QUICK:
        jobs = jobs[:1]
    done = set()
    if os.path.exists(OUT) and not QUICK:
        prev = pd.read_csv(OUT)
        done = set(zip(prev.trace_idx, prev["sample"]))
    t_all = time.time()
    print("warm-up matrix (discarded) ...", flush=True)
    W = upsample(traces[0].matrix, FACTOR, np.random.default_rng(999_999)).astype(np.float64)
    for _, ek in mt.ENGINES:
        mt.run_all(W, ek)
    for k, (i, s) in enumerate(jobs):
        if (i, s) in done:
            continue
        t = traces[i]
        M = matrix_for(i, s, t)
        S = mt.lower_bound_S(M)
        regime = "MoE mmlu" if "mmlu" in t.dataset else "MoE throughput"
        rows = []
        t0 = time.time()
        for eng_name, ek in mt.ENGINES:
            for rec in mt.run_all(M, ek):
                rec.update(regime=regime, trace=t.label, trace_idx=i, sample=s, engine=eng_name,
                           S=S, n=M.shape[0], density=float((M > 0).mean()),
                           col_skew=float(M.sum(0).max() / M.sum(0).min()))
                rows.append(rec)
        df = pd.DataFrame(rows)
        df["C_over_S"] = df.C / df.S
        df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"[{k + 1}/{len(jobs)}] {regime} {t.label} sample {s}: S={S} "
              f"({time.time() - t0:.0f} s, total {(time.time() - t_all) / 60:.1f} min)", flush=True)
    df = pd.read_csv(OUT)
    print(f"max reconstruction error: {df.recon_err.max():.3g}")
    summ = (df.groupby(["regime", "engine", "config"], sort=False)
              [["S", "N", "C", "C_over_S", "runtime_ms", "units"]].mean().reset_index())
    if not QUICK:
        summ.to_csv(os.path.join(OUTDIR, "moe256_summary.csv"), index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
