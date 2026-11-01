"""Summarise the three- vs four-tier Euler split comparison (euler_ruined_tier_compare.py).

For each leaf engine and depth, over matrices i >= SKIP (the campaign tables discard the
first 5): N under both methods with the paired difference and win count; the slowest
leaf's extraction time (T_P-compute) and the split time (T_split) under both methods with
the geometric mean of the paired ratio four-tier / three-tier and a bootstrap 95% interval
over matrices; the cycle ratio C/S (1 for the strong engines, whose C = S is checked in
every run; above 1 for WFA, which is weak); and, from the split statistics, the leaf
non-zeros and the pairs ruined.

Usage: python agent_scripts/analyze_ruined_tier.py [tag] [skip=5]
Output: euler_exp/ruined_tier/summary{tag}.csv, splits_summary{tag}.csv, and a printout.
"""
import sys

import numpy as np
import pandas as pd

from _paths import REPO  # repository-relative paths

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
SKIP = int(sys.argv[2]) if len(sys.argv) > 2 else 5
DIR = REPO / "euler_exp" / "ruined_tier"
A, B = "heuristic", "heuristic_ruined"
ENGINES = {"maximum": "Hungarian", "heavy": "GW Dynamic", "heavy_static": "GW Static", "wfa": "WFA"}
STRONG = ("maximum", "heavy", "heavy_static")
rng = np.random.default_rng(0)


def geo_ratio(x, y, reps=4000):
    """Geometric mean of y/x over paired rows, with a bootstrap 95% interval."""
    lr = np.log(np.asarray(y) / np.asarray(x))
    boot = lr[rng.integers(0, len(lr), (reps, len(lr)))].mean(axis=1)
    return np.exp(lr.mean()), np.exp(np.percentile(boot, 2.5)), np.exp(np.percentile(boot, 97.5))


def main():
    res = pd.read_csv(DIR / f"paired_results{TAG}.csv")
    spl = pd.read_csv(DIR / f"paired_splits{TAG}.csv")
    res, spl = res[res.matrix >= SKIP], spl[spl.matrix >= SKIP]
    n_mat = res.matrix.nunique()

    strong = res[res.engine.isin(STRONG)]
    bad_c = strong[(strong.C - strong.S).abs() > 1e-6]
    print(f"{n_mat} matrices (i >= {SKIP}); C = S in {len(strong) - len(bad_c)}/{len(strong)} "
          f"runs of the strong engines")

    rows = []
    for eng, name in ENGINES.items():
        for d in sorted(res.depth.unique()):
            sub = res[(res.engine == eng) & (res.depth == d)]
            a = sub[sub.method == A].set_index("matrix").sort_index()
            b = sub[sub.method == B].set_index("matrix").sort_index()
            a, b = a.loc[a.index.intersection(b.index)], b.loc[a.index.intersection(b.index)]
            if a.empty:
                continue
            dN = b.N - a.N
            lr, llo, lhi = geo_ratio(a.leaf_ms, b.leaf_ms)
            # the same ratio on each order class: the alternation cancels a position
            # effect p only on average (3-tier first gives r*p, 4-tier first r/p)
            first3 = a.order == 0
            lr3 = float(np.exp(np.log(b.leaf_ms[first3] / a.leaf_ms[first3]).mean()))
            lr4 = float(np.exp(np.log(b.leaf_ms[~first3] / a.leaf_ms[~first3]).mean()))
            sr, slo, shi = geo_ratio(a.split_ms, b.split_ms)
            tr, tlo, thi = geo_ratio(a.leaf_ms + a.split_ms, b.leaf_ms + b.split_ms)
            rows.append(dict(
                engine=name, depth=d, matrices=len(a),
                N_3tier=a.N.mean(), N_4tier=b.N.mean(), dN_pct=100 * dN.mean() / a.N.mean(),
                N_fewer=int((dN < 0).sum()), N_same=int((dN == 0).sum()), N_more=int((dN > 0).sum()),
                CS_3tier=(a.C / a.S).mean(), CS_4tier=(b.C / b.S).mean(),
                C_shorter=int((b.C < a.C - 1e-9).sum()), C_longer=int((b.C > a.C + 1e-9).sum()),
                leaf_ms_3tier=a.leaf_ms.mean(), leaf_ms_4tier=b.leaf_ms.mean(),
                leaf_ratio=lr, leaf_ratio_lo=llo, leaf_ratio_hi=lhi,
                leaf_ratio_3tier_first=lr3, leaf_ratio_4tier_first=lr4,
                split_ms_3tier=a.split_ms.mean(), split_ms_4tier=b.split_ms.mean(),
                split_ratio=sr, split_ratio_lo=slo, split_ratio_hi=shi,
                total_ratio=tr, total_ratio_lo=tlo, total_ratio_hi=thi))
    summ = pd.DataFrame(rows)
    summ.to_csv(DIR / f"summary{TAG}.csv", index=False)

    srows = []
    for d in sorted(spl.depth.unique()):
        a = spl[(spl.method == A) & (spl.depth == d)].set_index("matrix").sort_index()
        b = spl[(spl.method == B) & (spl.depth == d)].set_index("matrix").sort_index()
        ex_a, ex_b = a.nnz_total - a.nnz_D, b.nnz_total - b.nnz_D
        srows.append(dict(
            depth=d, nnz_D=a.nnz_D.mean(),
            nnz_3tier=a.nnz_total.mean(), nnz_4tier=b.nnz_total.mean(),
            nnz_pct=100 * (b.nnz_total - a.nnz_total).mean() / a.nnz_total.mean(),
            excess_3tier_pct=100 * ex_a.mean() / a.nnz_D.mean(),
            excess_4tier_pct=100 * ex_b.mean() / a.nnz_D.mean(),
            ruined_level_3tier=a.ruined_at_level.mean(), ruined_level_4tier=b.ruined_at_level.mean(),
            sparser=int((b.nnz_total < a.nnz_total).sum()), matrices=len(a),
            nnz_max_3tier=a.nnz_max.mean(), nnz_max_4tier=b.nnz_max.mean()))
    ss = pd.DataFrame(srows)
    ss.to_csv(DIR / f"splits_summary{TAG}.csv", index=False)

    print("\nLeaf non-zeros (sum over all leaves) and pairs ruined at each level")
    for r in ss.itertuples():
        print(f"  depth {r.depth}: nnz {r.nnz_3tier:8.0f} -> {r.nnz_4tier:8.0f} ({r.nnz_pct:+.2f}%), "
              f"excess over nnz(D) {r.excess_3tier_pct:5.2f}% -> {r.excess_4tier_pct:5.2f}%, "
              f"ruined at this level {r.ruined_level_3tier:6.0f} -> {r.ruined_level_4tier:6.0f}, "
              f"sparser on {r.sparser}/{r.matrices}")

    print("\nN, slowest-leaf time and split time: three tiers -> four tiers "
          "(time ratios: geometric mean of the paired ratio, bootstrap 95% interval)")
    for r in summ.itertuples():
        print(f"  {r.engine:10s} d{r.depth}  N {r.N_3tier:8.1f} -> {r.N_4tier:8.1f} ({r.dN_pct:+.2f}%, "
              f"fewer/same/more {r.N_fewer}/{r.N_same}/{r.N_more})  "
              f"leaf {r.leaf_ms_3tier:7.1f} -> {r.leaf_ms_4tier:7.1f} ms x{r.leaf_ratio:.3f} "
              f"[{r.leaf_ratio_lo:.3f}, {r.leaf_ratio_hi:.3f}]  "
              f"split {r.split_ms_3tier:5.1f} -> {r.split_ms_4tier:5.1f} ms x{r.split_ratio:.3f} "
              f"[{r.split_ratio_lo:.3f}, {r.split_ratio_hi:.3f}]"
              + (f"  C/S {r.CS_3tier:.4f} -> {r.CS_4tier:.4f} (shorter/longer {r.C_shorter}/{r.C_longer})"
                 if r.CS_3tier > 1 + 1e-9 or r.CS_4tier > 1 + 1e-9 else ""))


if __name__ == "__main__":
    main()
