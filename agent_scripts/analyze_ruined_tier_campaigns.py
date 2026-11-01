"""The three- and four-tier Euler splits as full campaigns (run_euler_ruined_tier_detached.ps1,
step 2), set against the variance campaign behind the paper's Euler table.

For every engine and depth, over matrices i >= 5 as in the paper's tables: the slowest
leaf's time, the split time, the cycle length and N, for
  paper   the mean of run/variance_2026/sweep{1,2,3}/euler/ (three-tier, what
          tab_euler_all_engines.tex reports)
  3-tier  tonight's rerun of the same campaign
  4-tier  tonight's campaign with --euler-split-method heuristic_ruined
N and the cycle length are deterministic, so 3-tier must equal every paper sweep exactly,
matrix by matrix; the script asserts it, and that all the campaigns cover the same matrices.
Both campaigns must also agree, matrix by matrix, with the paired comparison's N and C for
the same method (paired_results.csv), which reaches the decomposition through a different
route (the paired script calls decompose_euler_framework directly, the campaign goes
through main.py). If paired_results_rep2.csv exists, its three-tier means are added
(leaf_ms_rep2, split_ms_rep2): that run frees every result before the next timed call,
while the campaigns keep the baseline's and the previous depth's results alive during each
timed call, so rep2_vs_paper measures what the held memory costs the campaign's times.
The 3-tier/paper time ratio shows how far tonight's machine state is from
the August sweeps, the 4-tier/3-tier ratio is the effect at campaign level (one sweep
each, so inside the between-run spread of a few percent; the paired comparison,
euler_ruined_tier_compare.py, is the precise measurement). leaf_4_vs_3 is the ratio of the
100-matrix means; leaf_geo_4_vs_3 is the geometric mean of the per-matrix ratio, with a
bootstrap 95% interval over matrices (the two campaigns share their matrices but not their
run). leaf_ms_paper_min/max and split_ms_paper_min/max give the range of the three sweeps.

Output: euler_exp/ruined_tier/campaigns_summary.csv and a printout.
"""
import glob
import sys

import numpy as np
import pandas as pd

from _paths import REPO  # repository-relative paths

WARMUP = 5
DEPTHS = (1, 2, 3, 4)
ENGINES = {"maximum": "Hungarian", "heavy": "GW Dynamic", "heavy_static": "GW Static", "wfa": "WFA"}
DIR = REPO / "euler_exp" / "ruined_tier"
rng = np.random.default_rng(0)


def geo_ratio(x, y, reps=4000):
    """Geometric mean of y/x over the matrices, with a bootstrap 95% interval."""
    lr = np.log(np.asarray(y) / np.asarray(x))
    boot = lr[rng.integers(0, len(lr), (reps, len(lr)))].mean(axis=1)
    return np.exp(lr.mean()), np.exp(np.percentile(boot, 2.5)), np.exp(np.percentile(boot, 97.5))


def load(pattern):
    hits = sorted(glob.glob(str(REPO / pattern)))
    if not hits:
        sys.exit(f"no campaign matches {pattern}")
    d = pd.read_csv(hits[-1]).set_index("matrix_index").sort_index()
    return d[d.index >= WARMUP]


def main():
    paired = pd.read_csv(DIR / "paired_results.csv")
    paired = paired[paired.matrix >= WARMUP].set_index(["engine", "depth", "method", "matrix"]).sort_index()
    rows = []
    for eng, name in ENGINES.items():
        paper = [load(f"run/variance_2026/sweep{s}/euler/{eng}/*/results_stats.csv") for s in (1, 2, 3)]
        three = load(f"euler_exp/ruined_tier/eng_{eng}_dense_d4_heuristic/*/results_stats.csv")
        four = load(f"euler_exp/ruined_tier/eng_{eng}_dense_d4_heuristic_ruined/*/results_stats.csv")
        assert all(list(x.index) == list(three.index) for x in paper + [four]), (name, "matrix sets differ")
        for d in DEPTHS:
            p3, p4 = f"euler_heuristic_depth{d}", f"euler_heuristic_ruined_depth{d}"
            for q in ("perms", "cycle"):
                for x in paper:
                    assert (three[f"{p3}_{q}"] - x[f"{p3}_{q}"]).abs().max() < 1e-6, (name, d, q)
            for camp, method, pre in ((three, "heuristic", p3), (four, "heuristic_ruined", p4)):
                pr = paired.loc[(eng, d, method)].reindex(camp.index)
                assert pr.N.notna().all(), (name, d, method, "matrix missing from the paired run")
                assert (pr.N - camp[f"{pre}_perms"]).abs().max() < 1e-6, (name, d, method, "N")
                assert (pr.C - camp[f"{pre}_cycle"]).abs().max() < 1e-6, (name, d, method, "C")
            leaf_paper = sum(x[f"{p3}_time"].mean() for x in paper) / 3 * 1000
            split_paper = sum(x[f"{p3}_split"].mean() for x in paper) / 3 * 1000
            g, glo, ghi = geo_ratio(three[f"{p3}_time"], four[f"{p4}_time"])
            rows.append(dict(
                engine=name, depth=d, matrices=len(three),
                leaf_ms_paper=leaf_paper,
                leaf_ms_paper_min=min(x[f"{p3}_time"].mean() for x in paper) * 1000,
                leaf_ms_paper_max=max(x[f"{p3}_time"].mean() for x in paper) * 1000,
                split_ms_paper_min=min(x[f"{p3}_split"].mean() for x in paper) * 1000,
                split_ms_paper_max=max(x[f"{p3}_split"].mean() for x in paper) * 1000,
                leaf_geo_4_vs_3=g, leaf_geo_4_vs_3_lo=glo, leaf_geo_4_vs_3_hi=ghi,
                leaf_ms_3tier=three[f"{p3}_time"].mean() * 1000,
                leaf_ms_4tier=four[f"{p4}_time"].mean() * 1000,
                split_ms_paper=split_paper, split_ms_3tier=three[f"{p3}_split"].mean() * 1000,
                split_ms_4tier=four[f"{p4}_split"].mean() * 1000,
                cycle_3tier=three[f"{p3}_cycle"].mean(), cycle_4tier=four[f"{p4}_cycle"].mean(),
                N_3tier=three[f"{p3}_perms"].mean(), N_4tier=four[f"{p4}_perms"].mean()))
    out = pd.DataFrame(rows)
    out["tonight_vs_paper"] = out.leaf_ms_3tier / out.leaf_ms_paper
    out["leaf_4_vs_3"] = out.leaf_ms_4tier / out.leaf_ms_3tier
    rep2 = DIR / "paired_results_rep2.csv"
    if rep2.exists():
        r2 = pd.read_csv(rep2)
        r2 = r2[(r2.matrix >= WARMUP) & (r2.method == "heuristic")]
        r2 = r2.groupby(["engine", "depth"])[["leaf_ms", "split_ms"]].mean()
        names = {v: k for k, v in ENGINES.items()}
        out["leaf_ms_rep2"] = [r2.loc[(names[e], d), "leaf_ms"] for e, d in zip(out.engine, out.depth)]
        out["split_ms_rep2"] = [r2.loc[(names[e], d), "split_ms"] for e, d in zip(out.engine, out.depth)]
        out["rep2_vs_paper"] = out.leaf_ms_rep2 / out.leaf_ms_paper
        out["split_rep2_vs_paper"] = out.split_ms_rep2 / out.split_ms_paper
        # the same night's three-tier campaign, memory held, as a second reference
        out["rep2_vs_tonight"] = out.leaf_ms_rep2 / out.leaf_ms_3tier
        out["split_rep2_vs_tonight"] = out.split_ms_rep2 / out.split_ms_3tier
        out["total_rep2_vs_tonight"] = ((out.leaf_ms_rep2 + out.split_ms_rep2)
                                        / (out.leaf_ms_3tier + out.split_ms_3tier))
    out.to_csv(DIR / "campaigns_summary.csv", index=False)
    print(f"{out.matrices.iloc[0]} matrices per campaign; N and cycle of tonight's three-tier "
          f"campaign equal all three paper sweeps on every matrix, and both campaigns equal "
          f"the paired comparison's N and C on every matrix.\n")
    for r in out.itertuples():
        print(f"{r.engine:10s} d{r.depth}  slowest leaf (ms) paper {r.leaf_ms_paper:7.1f}  "
              f"3-tier {r.leaf_ms_3tier:7.1f} (x{r.tonight_vs_paper:.3f})  4-tier {r.leaf_ms_4tier:7.1f} "
              f"(x{r.leaf_4_vs_3:.3f})  split {r.split_ms_3tier:5.1f} -> {r.split_ms_4tier:5.1f}  "
              f"cycle {r.cycle_3tier:8.1f} -> {r.cycle_4tier:8.1f}  N {r.N_3tier:8.1f} -> {r.N_4tier:8.1f}"
              + (f"  | memory freed (rep2, 3-tier): leaf {r.leaf_ms_rep2:7.1f} (x{r.rep2_vs_paper:.3f} paper)"
                 f"  split {r.split_ms_rep2:5.1f} (x{r.split_rep2_vs_paper:.3f} paper)"
                 if "leaf_ms_rep2" in out else ""))


if __name__ == "__main__":
    main()
