"""Markdown tables for euler_exp/ruined_tier/REPORT.md, printed from the summary CSVs so
that no number in the report is typed by hand. Run the analysis scripts first:
  analyze_ruined_tier.py "" 5        -> summary.csv, splits_summary.csv
  analyze_ruined_tier.py _rep2 5     -> summary_rep2.csv, splits_summary_rep2.csv
  analyze_ruined_tier_campaigns.py   -> campaigns_summary.csv
  analyze_timing_drift.py            -> drift_*.csv
Tables whose CSV is missing are skipped.

Usage: python agent_scripts/ruined_tier_report_tables.py
"""
import pandas as pd

from _paths import REPO  # repository-relative paths

DIR = REPO / "euler_exp" / "ruined_tier"
ORDER = ["Hungarian", "GW Dynamic", "GW Static", "WFA"]
ENG = {"maximum": "Hungarian", "heavy": "GW Dynamic", "heavy_static": "GW Static", "wfa": "WFA"}


def read(name):
    f = DIR / name
    return pd.read_csv(f) if f.exists() else None


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


def ci(r, p):
    return f"{r[p]:.3f} [{r[p + '_lo']:.3f}, {r[p + '_hi']:.3f}]"


def by_engine(df):
    df = df.copy()
    df["o"] = df.engine.map(ORDER.index)
    return df.sort_values(["o", "depth"])


def main():
    ss = read("splits_summary.csv")
    if ss is not None:
        print("### Leaf non-zeros\n")
        table(["depth", "nnz(D)", "leaf nnz, 3 tiers", "4 tiers", "change",
               "excess over nnz(D), 3 tiers", "4 tiers", "pairs split at this level, 3 tiers",
               "4 tiers", "4 tiers sparser on"],
              [[int(r.depth), f"{r.nnz_D:.0f}", f"{r.nnz_3tier:.0f}", f"{r.nnz_4tier:.0f}",
                f"{r.nnz_pct:+.2f}%", f"{r.excess_3tier_pct:.2f}%", f"{r.excess_4tier_pct:.2f}%",
                f"{r.ruined_level_3tier:.0f}", f"{r.ruined_level_4tier:.0f}",
                f"{r.sparser}/{r.matrices}"] for r in ss.itertuples()])

    s1, s2 = read("summary.csv"), read("summary_rep2.csv")
    if s1 is not None:
        s1 = by_engine(s1)
        print("### Configurations N (mean over matrices)\n")
        table(["engine", "depth", "N, 3 tiers", "N, 4 tiers", "change", "fewer / same / more"],
              [[r.engine, r.depth, f"{r.N_3tier:.1f}", f"{r.N_4tier:.1f}", f"{r.dN_pct:+.2f}%",
                f"{r.N_fewer} / {r.N_same} / {r.N_more}"] for r in s1.itertuples()])
        w = s1[s1.engine == "WFA"]
        print("### Cycle length of WFA (C/S, mean over matrices; S = 8174)\n")
        table(["depth", "C/S, 3 tiers", "C/S, 4 tiers", "shorter / longer"],
              [[r.depth, f"{r.CS_3tier:.4f}", f"{r.CS_4tier:.4f}", f"{r.C_shorter} / {r.C_longer}"]
               for r in w.itertuples()])

        runs = [("run 1", s1)] + ([("run 2", by_engine(s2))] if s2 is not None else [])
        for label, s in runs:
            print(f"### Paired times, {label}: four tiers / three tiers, geometric mean of the "
                  f"per-matrix ratio [bootstrap 95% interval]\n")
            table(["engine", "depth", "slowest leaf (ms), 3 tiers", "ratio", "split (ms), 3 tiers",
                   "ratio"],
                  [[r.engine, r.depth, f"{r.leaf_ms_3tier:.1f}", ci(r._asdict(), "leaf_ratio"),
                    f"{r.split_ms_3tier:.1f}", ci(r._asdict(), "split_ratio")]
                   for r in s.itertuples()])

    c = read("campaigns_summary.csv")
    if c is not None:
        c = by_engine(c)
        print("### Campaigns: mean slowest-leaf time (ms) and its ratios\n")
        table(["engine", "depth", "August (mean of 3 sweeps)", "tonight, 3 tiers", "tonight / August",
               "tonight, 4 tiers", "4 tiers / 3 tiers", "split (ms), 3 -> 4 tiers"],
              [[r.engine, r.depth, f"{r.leaf_ms_paper:.1f}", f"{r.leaf_ms_3tier:.1f}",
                f"{r.tonight_vs_paper:.3f}", f"{r.leaf_ms_4tier:.1f}", f"{r.leaf_4_vs_3:.3f}",
                f"{r.split_ms_3tier:.1f} -> {r.split_ms_4tier:.1f}"] for r in c.itertuples()])

    if c is not None and "leaf_ms_rep2" in c:
        print("### What the held memory costs: August campaign means against the paired run 2 "
              "(three tiers, every result freed before the next timed call)\n")
        table(["engine", "depth", "slowest leaf (ms), August", "freed", "ratio", "split (ms), August",
               "freed", "ratio", "split + leaf (ms), August", "freed", "ratio"],
              [[r.engine, r.depth, f"{r.leaf_ms_paper:.1f}", f"{r.leaf_ms_rep2:.1f}",
                f"{r.rep2_vs_paper:.3f}", f"{r.split_ms_paper:.1f}", f"{r.split_ms_rep2:.1f}",
                f"{r.split_rep2_vs_paper:.3f}", f"{r.leaf_ms_paper + r.split_ms_paper:.1f}",
                f"{r.leaf_ms_rep2 + r.split_ms_rep2:.1f}",
                f"{(r.leaf_ms_rep2 + r.split_ms_rep2) / (r.leaf_ms_paper + r.split_ms_paper):.3f}"]
               for r in c.itertuples()])

    ab = read("ablation_summary.csv")
    if ab is not None:
        print("### Ablation: leaf non-zeros per variant (mean over matrices)\n")
        table(["variant", "depth", "leaf nnz", "vs 3 tiers", "excess over nnz(D)", "largest leaf",
               "sparser / denser than 3 tiers", "than 4 tiers", "leaves identical to 3 tiers",
               "to 4 tiers"],
              [[r.variant, r.depth, f"{r.nnz_total:.0f}", f"{r.nnz_vs_3tier_pct:+.2f}%",
                f"{r.excess_pct:.2f}%", f"{r.nnz_max:.1f}",
                f"{r.sparser_than_3tier} / {r.denser_than_3tier}",
                f"{r.sparser_than_4tier} / {r.denser_than_4tier}",
                f"{r.same_as_3tier}/{r.matrices}", f"{r.same_as_4tier}/{r.matrices}"]
               for r in ab.itertuples()])
    an = read("ablation_N_summary.csv")
    if an is not None:
        print("### Ablation: N of the variants that split differently from both rules\n")
        table(["variant", "engine", "depth", "N", "vs 3 tiers", "vs 4 tiers",
               "fewer / same / more than 4 tiers", "C/S (3 tiers, 4 tiers)"],
              [[r.variant, r.engine, r.depth, f"{r.N:.1f}", f"{r.vs_3tier_pct:+.2f}%",
                f"{r.vs_4tier_pct:+.2f}%", f"{r.fewer_than_4tier} / {r.same_as_4tier} / {r.more_than_4tier}",
                (f"{r.CS:.4f} ({r.CS_3tier:.4f}, {r.CS_4tier:.4f})" if r.CS > 1 + 1e-9 else "1")]
               for r in an.itertuples()])

    dp = read("drift_paired.csv")
    if dp is not None:
        dp = dp.assign(engine=dp.engine.map(ENG))
        piv = dp.pivot_table(index=["engine", "depth"], columns=["run", "method"],
                             values=["leaf_drift", "split_drift"])
        print("### Paired runs: drift, median of the last 20 matrices / median of the first 20\n")
        cols = [(v, run, m) for v in ("leaf_drift", "split_drift") for run in sorted(dp.run.unique())
                for m in ("heuristic", "heuristic_ruined") if (v, run, m) in piv.columns]
        names = {"leaf_drift": "leaf", "split_drift": "split",
                 "heuristic": "3 tiers", "heuristic_ruined": "4 tiers"}
        rows = []
        for eng in ORDER:
            for d in (1, 2, 3, 4):
                if (eng, d) in piv.index:
                    rows.append([eng, d] + [f"{piv.loc[(eng, d), col]:.2f}" for col in cols])
        table(["engine", "depth"] + [f"{names[v]} {run.strip('_')} {names[m]}" for v, run, m in cols], rows)

    da = read("drift_august.csv")
    if da is not None:
        e = da[da.algorithm == "euler"]
        piv = e.pivot_table(index=["engine", "column"], columns="sweep", values="drift")
        print("### August Euler campaigns: drift, median of the last 20 matrices / first 20\n")
        rows = []
        for eng in ("maximum", "heavy", "heavy_static", "wfa"):
            for col in ("bvn_runtime", "euler_heuristic_depth2_time", "euler_heuristic_depth4_time",
                        "euler_heuristic_depth4_split"):
                if (eng, col) in piv.index:
                    rows.append([ENG[eng], col] + [f"{piv.loc[(eng, col), s]:.2f}"
                                                   for s in ("sweep1", "sweep2", "sweep3")])
        table(["engine", "column", "sweep 1", "sweep 2", "sweep 3"], rows)
        other = da[da.algorithm != "euler"]
        print(f"Radix and the BvN baselines, all {len(other)} timing columns of the three sweeps: "
              f"drift between {other.drift.min():.2f} and {other.drift.max():.2f}.\n")

    dc = read("drift_campaigns.csv")
    if dc is not None and not dc.empty:
        print("### Tonight's campaigns: drift, median of the last 20 matrices / first 20\n")
        rows = []
        for eng in ("maximum", "heavy", "heavy_static", "wfa"):
            row = [ENG[eng]]
            for m in ("heuristic", "heuristic_ruined"):
                sub = dc[dc.campaign == f"eng_{eng}_dense_d4_{m}"].set_index("column").drift
                row += [f"{sub.get('bvn_runtime', float('nan')):.2f}"] + [
                    f"{sub.get(f'euler_{m}_depth{d}_{q}', float('nan')):.2f}"
                    for d, q in ((2, "time"), (4, "time"), (4, "split"))]
            rows.append(row)
        table(["engine"] + [f"{lab}, {r}" for r in ("3 tiers", "4 tiers")
                            for lab in ("BvN baseline", "leaf d2", "leaf d4", "split d4")], rows)


if __name__ == "__main__":
    main()
