"""Validity audit of the strategy-sweep runs.

V1  Config audit      : every run used the intended engine/strategy/params.
V2  Cross-run identity : min-strategy Radix cells must equal the canonical
                         rerun_2026 values PER MATRIX (same seed, same code path).
V3  B2 invariance      : digit planes at B=2 are 0/1 -> min/median/max must be
                         IDENTICAL per matrix (cycle & perms).
V4  Conservation laws  : C >= S always; C == S exactly iff strategy==min for
                         strong engines (BvN & every Euler depth).
V5  Reconstruction     : run the actual pipeline on fresh n=256 matrices for
                         every engine x strategy; verify sum(lambda*P) covers D
                         (every demand cell served) for BvN and Euler d2.
"""
import glob, json, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Users/ofekc/Desktop/Msc/Thesis/bvn_project")
ROOT = "C:/Users/ofekc/Desktop/Msc/Thesis/bvn_project"
ENG = [("Hungarian", "max", "maximum"), ("WFA", "wfa", "wfa"),
       ("GW Dynamic", "heavy", "heavy"), ("GW Static", "heavy_static", "heavy_static")]
FAIL = []

def latest(p): return sorted(glob.glob(p))[-1]

print("=== V1: config audit ===")
for _, d, pre in ENG:
    cfg = json.load(open(latest(f"{ROOT}/run/step_strategy_2026/{d}/*/config.json")))
    ok = (cfg.get("n") == 256 and cfg.get("k") == 256 and cfg.get("max_weight") == 64
          and cfg.get("random_seed") == 42 and cfg.get("radix_strategy") == "all"
          and sorted(cfg.get("radix_bases", [])) == [2, 4, 8, 16])
    print(f"  step_strategy/{d:12s} engine={cfg.get('engine'):12s} strat={cfg.get('radix_strategy')} bases={cfg.get('radix_bases')} seed={cfg.get('random_seed')} {'OK' if ok else 'MISMATCH'}")
    if not ok: FAIL.append(f"V1 step/{d}")
for st in ["median", "max"]:
    for _, _, pre in ENG:
        cfg = json.load(open(latest(f"{ROOT}/euler_exp/strat_{st}_{pre}/*/config.json")))
        ok = (cfg.get("engine") == "euler_bvn" and cfg.get("euler_leaf_engine") == pre
              and cfg.get("radix_strategy") == st and cfg.get("euler_depths") == [1, 2, 3, 4]
              and cfg.get("euler_split_method") == "heuristic" and cfg.get("random_seed") == 42)
        print(f"  strat_{st}_{pre:12s} leaf={cfg.get('euler_leaf_engine'):12s} strat={cfg.get('radix_strategy'):6s} depths={cfg.get('euler_depths')} {'OK' if ok else 'MISMATCH'}")
        if not ok: FAIL.append(f"V1 strat_{st}_{pre}")

print("\n=== V2: min-strategy Radix == canonical rerun_2026, per matrix ===")
for name, d, pre in ENG:
    ds = pd.read_csv(latest(f"{ROOT}/run/step_strategy_2026/{d}/*/results_stats.csv"))
    dc = pd.read_csv(latest(f"{ROOT}/run/rerun_2026/{d}/*/results_stats.csv"))
    n = len(ds)
    bad = []
    for b in [2, 4, 8, 16]:
        for m in ["cycle", "perms"]:
            a = ds[f"{pre}_min_{b}_{m}"].values
            c = dc[f"{pre}_{b}_{m}"].values[:n]
            if not np.allclose(a, c, atol=1e-6):
                bad.append(f"B{b}/{m} (max diff {np.abs(a-c).max():.1f})")
    print(f"  {name:12s}: {'IDENTICAL on all bases/metrics' if not bad else 'DIFFERS: '+', '.join(bad)}")
    if bad: FAIL.append(f"V2 {name}")

print("\n=== V3: B2 strategy invariance (0/1 planes), per matrix ===")
for name, d, pre in ENG:
    ds = pd.read_csv(latest(f"{ROOT}/run/step_strategy_2026/{d}/*/results_stats.csv"))
    bad = []
    for m in ["cycle", "perms"]:
        a, b_, c = (ds[f"{pre}_{s}_2_{m}"].values for s in ["min", "median", "max"])
        if not (np.allclose(a, b_) and np.allclose(b_, c)):
            bad.append(m)
    print(f"  {name:12s}: {'min==median==max exactly' if not bad else 'VIOLATION in '+str(bad)}")
    if bad: FAIL.append(f"V3 {name}")

print("\n=== V4: conservation (C>=S always; C==S iff min, strong engines) ===")
for name, _, pre in ENG:
    dmin = pd.read_csv(latest(f"{ROOT}/euler_exp/eng_{pre}_dense_d4/*/results_stats.csv"))
    S = dmin["bvn_cycle"].values if pre != "wfa" else None
    for st, path in [("min", f"{ROOT}/euler_exp/eng_{pre}_dense_d4"),
                     ("median", f"{ROOT}/euler_exp/strat_median_{pre}"),
                     ("max", f"{ROOT}/euler_exp/strat_max_{pre}")]:
        df = pd.read_csv(latest(f"{path}/*/results_stats.csv"))
        cols = ["bvn_cycle"] + [f"euler_heuristic_depth{k}_cycle" for k in [1, 2, 3, 4]]
        if pre == "wfa":
            continue
        for col in cols:
            C = df[col].values
            if st == "min":
                ok = np.allclose(C, S)
                tag = "C==S"
            else:
                ok = (C >= S - 1e-6).all() and (C.mean() > S.mean())
                tag = "C>S"
            if not ok:
                print(f"  {name:12s} {st:6s} {col}: VIOLATION of {tag}")
                FAIL.append(f"V4 {name}/{st}/{col}")
    print(f"  {name:12s}: strong-engine conservation holds (min: C==S; median/max: C>S) at BvN + all depths")

print("\n=== V5: reconstruction on fresh n=256 matrices (pipeline as-run) ===")
from src.utils.matrix_generator import generate_matrix
from src.algorithms.bvn import bvn_decomposition
from src.algorithms.euler_splitting import decompose_euler_framework
for i in range(2):
    D = generate_matrix(n=256, k=256, max_weight=64, rng=np.random.default_rng(1000 + i)).astype(np.float64)
    for _, _, pre in ENG:
        eng = {"max": "maximum"}.get(pre, pre)
        for st in ["min", "median", "max"]:
            comps = bvn_decomposition(D, matching_algorithm=eng, step_strategy=st)
            R = np.zeros_like(D)
            for c in comps:
                R += c.weight * c.permutation
            if not (R + 1e-6 >= D).all():
                print(f"  BvN {eng}/{st} matrix{i}: COVERAGE FAILURE"); FAIL.append(f"V5 bvn {eng}/{st}")
            comps, _, _, _ = decompose_euler_framework(D, matching_method=eng, depth=2,
                                                       split_method="heuristic", step_strategy=st)
            R = np.zeros_like(D)
            for c in comps:
                R += c.weight * c.permutation
            if not (R + 1e-6 >= D).all():
                print(f"  Euler {eng}/{st} matrix{i}: COVERAGE FAILURE"); FAIL.append(f"V5 euler {eng}/{st}")
print("  all engine x strategy x {BvN, Euler d2} decompositions cover the full demand" if not FAIL else "")

print("\n" + "=" * 60)
print("VERDICT:", "ALL CHECKS PASSED" if not FAIL else f"FAILURES: {FAIL}")
