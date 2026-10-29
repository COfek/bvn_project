"""Evaluate every engine x framework on the Griner & Avin traffic model TM(t_l, n_f, c_l, n).

Workload: the traffic model of
  C. Griner and C. Avin, "Integrating topology and traffic engineering to maximize
  throughput in reconfigurable networks", Computer Networks 274 (2026) 111803,
using the authors' released generator (see src/utils/griner_avin_tm.py), with the
paper's parameters t_l=0.2, c_l=0.7, sigma=0.01, and n_f as the density knob.
Flow weights are integer-quantised so the matrix stays exactly S-regular (required
by the Euler split); the model's large/small load structure is preserved.

Runs, for each of {Hungarian, WFA, GW Dynamic, GW Static}:
    BvN baseline | Radix B16, B8, B4, B2 | Euler d1..d4 (split-inclusive runtime)
in a sparse (n_f=64) and a dense (n_f=512) regime.

Outputs:
  run/tm_workload/tm_results.csv     per-matrix raw results
  run/tm_workload/tm_summary.csv     mean per (regime, engine, config)
  paper/plots_pdf/tm_workload.pdf    trade-off figure
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, r"C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.algorithms.bvn import bvn_decomposition
from src.algorithms.radix_decomposition import decompose_radix
from src.algorithms.euler_splitting import decompose_euler_framework
from src.utils.griner_avin_tm import tm_integer
from src.utils.matrix_generator import generate_matrix

ROOT = r"C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project"
OUTDIR = os.path.join(ROOT, "run", "tm_workload")
PLOTS = r"C:\Users\ofekc\Desktop\Msc\Thesis\paper\plots_pdf"

N = 256
N_MATRICES = 10          # after 1 warm-up matrix (JIT)
# (label, kind, knob):  kind "tm" -> n_f flows;  kind "uniform" -> thesis generator k
REGIMES = [("TM sparse", "tm", 64),
           ("TM dense", "tm", 512),
           ("Uniform (thesis)", "uniform", 256)]
ENGINES = [("Hungarian", "maximum"), ("WFA", "wfa"),
           ("GW Dynamic", "heavy"), ("GW Static", "heavy_static")]
BASES = [16, 8, 4, 2]
DEPTHS = [1, 2, 3, 4]


def make_matrix(kind, knob, rng):
    if kind == "tm":
        return tm_integer(N, knob, rng, w_small=7)
    return generate_matrix(N, k=knob, max_weight=64, rng=rng)


def _cycle(comps):
    return float(sum(c.weight for c in comps))


def run_all(M, engine_key):
    """Return list of dicts: one per configuration, for a single matrix+engine."""
    out = []

    # --- BvN baseline ---
    t0 = time.perf_counter()
    comps = bvn_decomposition(M.copy(), matching_algorithm=engine_key)
    rt = (time.perf_counter() - t0) * 1000
    out.append(dict(config="BvN", N=len(comps), C=_cycle(comps), runtime_ms=rt,
                    split_ms=0.0, unit_ms=rt, units=1))

    # --- Radix (digit extraction ~0; runtime = slowest plane) ---
    for b in BASES:
        comps, max_plane_s, n_planes = decompose_radix(
            M.copy(), base=b, matching_method=engine_key)
        out.append(dict(config=f"B{b}", N=len(comps), C=_cycle(comps),
                        runtime_ms=float(max_plane_s) * 1000,
                        split_ms=0.0, unit_ms=float(max_plane_s) * 1000, units=n_planes))

    # --- Euler (runtime = split + max leaf, end-to-end) ---
    for d in DEPTHS:
        comps, split_s, max_leaf_s, n_leaves = decompose_euler_framework(
            M.copy(), matching_method=engine_key, depth=d, split_method="heuristic")
        out.append(dict(config=f"Euler d{d}", N=len(comps), C=_cycle(comps),
                        runtime_ms=(split_s + max_leaf_s) * 1000,
                        split_ms=split_s * 1000, unit_ms=max_leaf_s * 1000,
                        units=n_leaves))
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []

    for regime, kind, knob in REGIMES:
        rng = np.random.default_rng(42)
        # warm-up matrix (JIT compile) -- discarded
        Mw = make_matrix(kind, knob, rng)
        for _, ek in ENGINES:
            run_all(Mw, ek)
        print(f"[{regime}] warm-up done", flush=True)

        for m in range(N_MATRICES):
            M = make_matrix(kind, knob, rng)
            S = int(M.sum(axis=1)[0])
            dens = float((M > 0).mean())
            for eng_name, ek in ENGINES:
                for rec in run_all(M, ek):
                    rec.update(regime=regime, knob=knob, matrix=m, engine=eng_name,
                               S=S, density=dens)
                    rows.append(rec)
            print(f"[{regime}] matrix {m+1}/{N_MATRICES}  S={S}  density={dens*100:.0f}%",
                  flush=True)

    df = pd.DataFrame(rows)
    df["C_over_S"] = df["C"] / df["S"]
    df.to_csv(os.path.join(OUTDIR, "tm_results.csv"), index=False)

    summary = (df.groupby(["regime", "engine", "config"], sort=False)
                 .agg(S=("S", "mean"), N=("N", "mean"), C=("C", "mean"),
                      C_over_S=("C_over_S", "mean"), runtime_ms=("runtime_ms", "mean"),
                      split_ms=("split_ms", "mean"), unit_ms=("unit_ms", "mean"),
                      units=("units", "mean"), density=("density", "mean"))
                 .reset_index())
    summary.to_csv(os.path.join(OUTDIR, "tm_summary.csv"), index=False)

    order = ["BvN"] + [f"B{b}" for b in BASES] + [f"Euler d{d}" for d in DEPTHS]
    for regime, _, _ in REGIMES:
        sub = summary[summary.regime == regime]
        print(f"\n================ {regime.upper()}  (n={N}, TM t_l=0.2, c_l=0.7) "
              f"S={sub.S.iloc[0]:.0f} ================")
        print(f"{'engine':11s} {'config':10s} {'runtime(ms)':>12s} {'N':>8s} {'C':>9s} {'C/S':>6s}")
        for eng_name, _ in ENGINES:
            for cfg in order:
                r = sub[(sub.engine == eng_name) & (sub.config == cfg)]
                if r.empty:
                    continue
                r = r.iloc[0]
                print(f"{eng_name:11s} {cfg:10s} {r.runtime_ms:12.1f} {r.N:8.0f} "
                      f"{r.C:9.0f} {r.C_over_S:6.3f}")

    print(f"\nsaved -> {OUTDIR}\\tm_results.csv")
    print(f"saved -> {OUTDIR}\\tm_summary.csv")
    print("Figures/tables: run agent_scripts/tm_paper_figures.py")


if __name__ == "__main__":
    main()
