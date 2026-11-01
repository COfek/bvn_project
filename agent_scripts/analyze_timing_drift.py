"""Do the measured times drift over a run? The matrices are independent draws, so a time
that grows with the matrix index measures the state of the process, not the work.

For every timing column (*_time, *_runtime, *_split), the drift is the median over the
last 20 matrices (i = 85..104) divided by the median over the first 20 kept (i = 5..24):
  drift_august.csv     the variance campaign behind the paper's tables
                       (run/variance_2026/sweep{1,2,3}/{baseline,euler,radix}/<engine>/)
  drift_campaigns.csv  tonight's campaigns (euler_exp/ruined_tier/eng_*/)
  drift_paired.csv     the paired comparisons (paired_results{tag}.csv), per engine, depth
                       and method, slowest leaf and split
  drift_paired_blocks.csv  the paired four-tier / three-tier leaf-time ratio (geometric
                       mean) per block of 10 matrices, to show whether the comparison
                       itself moves with the drift
  split_vs_hungarian.csv  in each paired run, the split time of every engine's call over
                       the Hungarian call's on the same matrix, depth and rule (the split
                       is the same work in all of them): geometric mean over matrices
                       5..104 with a bootstrap 95% interval

Usage: python agent_scripts/analyze_timing_drift.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import REPO  # repository-relative paths

FIRST, LAST = (5, 24), (85, 104)
DIR = REPO / "euler_exp" / "ruined_tier"
TAGS = ("", "_rep2")


def drift(s):
    """median over the last 20 matrices / median over the first 20 (index = matrix)."""
    a, b = s.loc[FIRST[0]:FIRST[1]].median(), s.loc[LAST[0]:LAST[1]].median()
    return b / a if a > 0 else np.nan, a * 1000


def campaign_drift(pattern, label_parts):
    rows = []
    for f in sorted(glob.glob(str(REPO / pattern))):
        rel = Path(f).relative_to(REPO).parts
        d = pd.read_csv(f).set_index("matrix_index").sort_index()
        for c in d.columns:
            if c.endswith(("_time", "_runtime", "_split")):
                r, first_ms = drift(d[c])
                rows.append(dict(**{k: rel[i] for k, i in label_parts.items()},
                                 column=c, drift=r, first20_median_ms=first_ms))
    return pd.DataFrame(rows)


def main():
    aug = campaign_drift("run/variance_2026/sweep[123]/*/*/*/results_stats.csv",
                         dict(sweep=2, algorithm=3, engine=4))
    aug.to_csv(DIR / "drift_august.csv", index=False)
    tonight = campaign_drift("euler_exp/ruined_tier/eng_*/*/results_stats.csv", dict(campaign=2))
    tonight.to_csv(DIR / "drift_campaigns.csv", index=False)

    prow, brow = [], []
    for tag in TAGS:
        f = DIR / f"paired_results{tag}.csv"
        if not f.exists():
            continue
        res = pd.read_csv(f)
        for (eng, d, m), s in res.groupby(["engine", "depth", "method"]):
            s = s.set_index("matrix").sort_index()
            lr, lf = drift(s.leaf_ms / 1000)
            sr, sf = drift(s.split_ms / 1000)
            prow.append(dict(run=tag or "_rep1", engine=eng, depth=d, method=m, leaf_drift=lr,
                             leaf_first20_ms=lf, split_drift=sr, split_first20_ms=sf))
        for (eng, d), s in res[res.matrix >= FIRST[0]].groupby(["engine", "depth"]):
            a = s[s.method == "heuristic"].set_index("matrix").leaf_ms.sort_index()
            b = s[s.method == "heuristic_ruined"].set_index("matrix").leaf_ms.sort_index()
            lr = np.log(b / a)
            for lo in range(FIRST[0], LAST[1] + 1, 10):
                blk = lr.loc[lo:lo + 9]
                brow.append(dict(run=tag or "_rep1", engine=eng, depth=d, first=lo, last=lo + 9,
                                 ratio_4_vs_3=float(np.exp(blk.mean())),
                                 leaf_ms_3tier_median=float(a.loc[lo:lo + 9].median())))
    pd.DataFrame(prow).to_csv(DIR / "drift_paired.csv", index=False)
    pd.DataFrame(brow).to_csv(DIR / "drift_paired_blocks.csv", index=False)

    rng = np.random.default_rng(0)
    srow = []
    for tag in TAGS:
        f = DIR / f"paired_results{tag}.csv"
        if not f.exists():
            continue
        res = pd.read_csv(f)
        res = res[res.matrix >= FIRST[0]].set_index(["engine", "depth", "method", "matrix"]).sort_index()
        for d in sorted(res.index.get_level_values("depth").unique()):
            for m in ("heuristic", "heuristic_ruined"):
                ref = res.loc[("maximum", d, m)].split_ms
                for eng in ("heavy", "heavy_static", "wfa"):
                    lr = np.log(res.loc[(eng, d, m)].split_ms.loc[ref.index] / ref)
                    boot = lr.to_numpy()[rng.integers(0, len(lr), (4000, len(lr)))].mean(axis=1)
                    srow.append(dict(run=tag or "_rep1", depth=d, method=m, engine=eng,
                                     ratio=float(np.exp(lr.mean())),
                                     lo=float(np.exp(np.percentile(boot, 2.5))),
                                     hi=float(np.exp(np.percentile(boot, 97.5)))))
    pd.DataFrame(srow).to_csv(DIR / "split_vs_hungarian.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    print("August variance campaign: drift (last 20 / first 20 matrices), per sweep")
    print(aug.pivot_table(index=["algorithm", "engine", "column"], columns="sweep",
                          values="drift").round(2).to_string())
    if not tonight.empty:
        print("\nTonight's campaigns: drift")
        print(tonight.pivot_table(index="column", columns="campaign", values="drift").round(2).to_string())
    if prow:
        p = pd.DataFrame(prow)
        print("\nPaired comparisons: slowest-leaf drift (both methods)")
        print(p.pivot_table(index=["engine", "depth"], columns=["run", "method"],
                            values="leaf_drift").round(2).to_string())
        print("\nPaired comparisons: split drift (both methods)")
        print(p.pivot_table(index=["engine", "depth"], columns=["run", "method"],
                            values="split_drift").round(2).to_string())
        b = pd.DataFrame(brow)
        print("\nPaired four-tier / three-tier leaf-time ratio per block of 10 matrices, depth 4")
        print(b[b.depth == 4].pivot_table(index=["run", "engine"], columns="first",
                                          values="ratio_4_vs_3").round(3).to_string())


if __name__ == "__main__":
    main()
