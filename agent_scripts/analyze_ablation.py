"""Summarise the four-tier ablation (ablate_ruined_tier.py).

Split statistics, per variant and depth over matrices i >= 5: leaf non-zeros (all leaves
and the largest leaf) with the change against 3tier, the excess over nnz(D), the pairs in
both halves created at that level, on how many matrices the variant is sparser than 3tier
and than 4tier, and on how many (matrix, depth) its leaves equal those of 3tier and 4tier.

N, if ablation_N.csv exists: first a check that the ablation's N and C for 3tier and 4tier
equal paired_results.csv (heuristic, heuristic_ruined) on the matrices it counted; then,
per engine and depth, N of every other counted variant against both, with the paired
difference and win counts, and C/S where it is not 1.

Usage: python agent_scripts/analyze_ablation.py
Output: euler_exp/ruined_tier/ablation_summary.csv, ablation_N_summary.csv, and a printout.
"""
import pandas as pd

from _paths import REPO  # repository-relative paths

DIR = REPO / "euler_exp" / "ruined_tier"
SKIP = 5
ENG = {"maximum": "Hungarian", "heavy": "GW Dynamic", "heavy_static": "GW Static", "wfa": "WFA"}
METHOD = {"3tier": "heuristic", "4tier": "heuristic_ruined"}


def main():
    spl = pd.read_csv(DIR / "ablation_splits.csv")
    spl = spl[spl.matrix >= SKIP]
    base = spl[spl.variant == "3tier"].set_index(["matrix", "depth"])
    four = spl[spl.variant == "4tier"].set_index(["matrix", "depth"])
    rows = []
    for (v, d), s in spl.groupby(["variant", "depth"], sort=False):
        s = s.set_index(["matrix", "depth"])
        b, f = base.loc[s.index], four.loc[s.index]
        rows.append(dict(
            variant=v, depth=d, matrices=len(s), nnz_D=s.nnz_D.mean(), nnz_total=s.nnz_total.mean(),
            nnz_vs_3tier_pct=100 * (s.nnz_total - b.nnz_total).mean() / b.nnz_total.mean(),
            excess_pct=100 * (s.nnz_total - s.nnz_D).mean() / s.nnz_D.mean(),
            excess_vs_3tier_pct=100 * ((s.nnz_total - s.nnz_D).mean() / (b.nnz_total - b.nnz_D).mean() - 1),
            nnz_max=s.nnz_max.mean(), split_at_level=s.split_at_level.mean(),
            sparser_than_3tier=int((s.nnz_total < b.nnz_total).sum()),
            denser_than_3tier=int((s.nnz_total > b.nnz_total).sum()),
            sparser_than_4tier=int((s.nnz_total < f.nnz_total).sum()),
            denser_than_4tier=int((s.nnz_total > f.nnz_total).sum()),
            same_as_3tier=int(s.same_as_3tier.sum()), same_as_4tier=int(s.same_as_4tier.sum())))
    summ = pd.DataFrame(rows)
    summ.to_csv(DIR / "ablation_summary.csv", index=False)
    print(f"{spl.matrix.nunique()} matrices (i >= {SKIP})\n")
    for r in summ.itertuples():
        print(f"{r.variant:10s} d{r.depth}  leaf nnz {r.nnz_total:8.0f} ({r.nnz_vs_3tier_pct:+.2f}% vs 3tier)  "
              f"excess {r.excess_pct:5.2f}% ({r.excess_vs_3tier_pct:+.1f}% vs 3tier)  largest leaf {r.nnz_max:7.1f}  "
              f"vs 3tier sparser/denser {r.sparser_than_3tier}/{r.denser_than_3tier}  "
              f"vs 4tier {r.sparser_than_4tier}/{r.denser_than_4tier}  "
              f"identical to 3tier {r.same_as_3tier}, to 4tier {r.same_as_4tier}")

    f = DIR / "ablation_N.csv"
    if not f.exists():
        return
    abl = pd.read_csv(f)
    abl = abl[abl.matrix >= SKIP]
    paired = pd.read_csv(DIR / "paired_results.csv")
    paired = paired[paired.matrix >= SKIP].set_index(["matrix", "engine", "depth", "method"])
    for v, m in METHOD.items():
        a = abl[abl.variant == v]
        if a.empty:
            continue
        p = paired.loc[[(r.matrix, r.engine, r.depth, m) for r in a.itertuples()]]
        assert (a.N.to_numpy() == p.N.to_numpy()).all(), (v, "N differs from the paired run")
        assert (abs(a.C.to_numpy() - p.C.to_numpy()) < 1e-6).all(), (v, "C differs from the paired run")
        print(f"\nablation N and C for {v} equal the paired run's {m} on matrices "
              f"{sorted(a.matrix.unique())}, all engines and depths")

    nrows = []
    for v in sorted(set(abl.variant) - set(METHOD)):
        a = abl[abl.variant == v]
        for (eng, d), s in a.groupby(["engine", "depth"]):
            s = s.set_index("matrix").sort_index()
            p3 = paired.xs((eng, d, "heuristic"), level=["engine", "depth", "method"]).loc[s.index]
            p4 = paired.xs((eng, d, "heuristic_ruined"), level=["engine", "depth", "method"]).loc[s.index]
            nrows.append(dict(
                variant=v, engine=ENG[eng], depth=d, matrices=len(s), N=s.N.mean(),
                N_3tier=p3.N.mean(), N_4tier=p4.N.mean(),
                vs_3tier_pct=100 * (s.N - p3.N).mean() / p3.N.mean(),
                vs_4tier_pct=100 * (s.N - p4.N).mean() / p4.N.mean(),
                fewer_than_4tier=int((s.N < p4.N).sum()), same_as_4tier=int((s.N == p4.N).sum()),
                more_than_4tier=int((s.N > p4.N).sum()),
                CS=(s.C / s.S).mean(), CS_3tier=(p3.C / p3.S).mean(), CS_4tier=(p4.C / p4.S).mean()))
    ns = pd.DataFrame(nrows)
    if ns.empty:
        return
    ns["o"] = ns.engine.map(list(ENG.values()).index)
    ns = ns.sort_values(["variant", "o", "depth"]).drop(columns="o")
    ns.to_csv(DIR / "ablation_N_summary.csv", index=False)
    print()
    for r in ns.itertuples():
        print(f"{r.variant:8s} {r.engine:10s} d{r.depth}  N {r.N:8.1f}  (3tier {r.N_3tier:8.1f}, {r.vs_3tier_pct:+.2f}%;"
              f" 4tier {r.N_4tier:8.1f}, {r.vs_4tier_pct:+.2f}%, fewer/same/more {r.fewer_than_4tier}/"
              f"{r.same_as_4tier}/{r.more_than_4tier})"
              + (f"  C/S {r.CS:.4f} (3tier {r.CS_3tier:.4f}, 4tier {r.CS_4tier:.4f})" if r.CS > 1 + 1e-9 else ""))


if __name__ == "__main__":
    main()
