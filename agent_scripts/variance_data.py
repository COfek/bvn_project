"""Sweep-averaged access to the variance campaign, for the figure scripts.

run/variance_2026 holds three full repetitions of the campaign, measured on an
idle machine on 2026-08-30 and 2026-08-31. Chapter 6's tables already read it
through make_paper_tables.py; the Chapter 7 figure scripts read the older
run/rerun_2026 and euler_exp campaigns, whose timings are 5 to 14 percent
higher, so the two chapters disagreed. This module gives the figure scripts the
same data the tables use.

Two things it handles that a plain glob does not:

  * sweep3_contaminated must not be included. It is the original sweep 3,
    measured while the machine was in use, kept as evidence.
  * the runner computes the BvN baseline only for engines in a gate list, so
    plain heavy/heavy_static/wfa runs have an empty bvn column. Their baselines
    come from a separate _bvn run under sweep*/baseline/. This module merges
    them back in, so callers see one frame with every column populated.

Timings are the mean over the three sweeps. Cycle length and permutation count
are deterministic given the seed and identical across sweeps; the caller gets
the same value either way.

    from variance_data import radix_frame, euler_frame
    dr = radix_frame("max")        # BvN + Radix columns, Hungarian
    de = euler_frame("maximum")    # Euler depths 1-4, Hungarian leaves
"""
from __future__ import annotations
from _paths import REPO  # repository-relative paths

import glob
import os

import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radix_extraction import add_to_radix_times  # noqa: E402

ROOT = str(REPO)
SWEEPS = (1, 2, 3)   # named, never globbed: sweep3_contaminated must stay out
WARMUP = 5


# Which campaign the scripts read. "variance_2026" (the default) is what the paper
# quotes today; "2026_11" is the November rerun, assembled from the runs below.
CAMPAIGN = os.environ.get("BVN_CAMPAIGN", "variance_2026")


def campaign_dir(sweep: int, kind: str, engine: str) -> str:
    """The run/ folder holding one sweep of one step, relative to the repository.

    For 2026_11:
      Radix Hungarian (and its baseline)  rerun_2026_11c, sweeps 1-3 (bvn.py's rule)
      WWFA, every step                    sweep 1 rerun_2026_11c, sweeps 2-3 wwfa_2026_11
      Euler, the other engines            rerun_2026_11b, sweeps 1-3 (four tiers)
      Radix and baseline, GW and WFA      sweeps 1-2 rerun_2026_11, sweep 3 rerun_2026_11b
    """
    if CAMPAIGN == "variance_2026":
        root = "run/variance_2026"
    elif CAMPAIGN == "2026_11":
        if engine == "wwfa":
            root = "run/rerun_2026_11c" if sweep == 1 else "run/wwfa_2026_11"
        elif kind == "radix" and engine == "max":
            root = "run/rerun_2026_11c"
        elif kind == "euler":
            root = "run/rerun_2026_11b"
        else:
            root = "run/rerun_2026_11" if sweep in (1, 2) else "run/rerun_2026_11b"
    else:
        raise ValueError(f"unknown BVN_CAMPAIGN {CAMPAIGN!r}")
    return f"{root}/sweep{sweep}/{kind}/{engine}"


def campaign_csv(sweep: int, kind: str, engine: str) -> str:
    hits = sorted(glob.glob(os.path.join(
        ROOT, campaign_dir(sweep, kind, engine), "*", "results_stats.csv")))
    if not hits:
        raise FileNotFoundError(f"no results for {campaign_dir(sweep, kind, engine)}")
    return hits[-1]


# the campaigns before 2026-10-04 did not time the digit extraction
CHARGE_EXTRACTION = CAMPAIGN == "variance_2026"


def _one(sweep: int, kind: str, engine: str) -> pd.DataFrame:
    d = pd.read_csv(campaign_csv(sweep, kind, engine))
    if kind == "radix" and CHARGE_EXTRACTION:
        # the campaign timed the slowest plane only; charge the digit extraction
        d = add_to_radix_times(d, "variance")
    return d.iloc[WARMUP:].reset_index(drop=True)


def _mean_over_sweeps(kind: str, engine: str) -> pd.DataFrame:
    """Element-wise mean across the three sweeps, row by row and column by column."""
    frames = [_one(s, kind, engine) for s in SWEEPS]
    n = min(len(f) for f in frames)
    frames = [f.iloc[:n] for f in frames]
    out = frames[0].copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = sum(f[col] for f in frames) / len(frames)
    return out


def radix_frame(engine: str, rows: int | None = None) -> pd.DataFrame:
    """BvN baseline and Radix columns for one engine, averaged over the sweeps.

    engine is the --engine value: max, heavy, heavy_static, wfa.
    rows, if given, keeps only the first `rows` matrices. Pass EULER_WINDOW to
    put Radix and BvN on exactly the matrices the Euler campaign covers, so that
    a comparison across frameworks is not also a comparison across draws: over
    all 995 matrices the baseline is S = 8211, over the first 100 it is 8174.
    """
    d = _mean_over_sweeps("radix", engine)
    if d["bvn_cycle"].isna().all():
        # this engine's baseline lives in the separate _bvn run
        b = _mean_over_sweeps("baseline", engine)
        n = min(len(d), len(b))
        d = d.iloc[:n].copy()
        for col in ("bvn_perms", "bvn_cycle", "bvn_runtime"):
            d[col] = b[col].iloc[:n].values
    return d.iloc[:rows].reset_index(drop=True) if rows else d


def euler_frame(leaf_engine: str) -> pd.DataFrame:
    """Euler depths 1-4 for one leaf engine, averaged over the sweeps.

    leaf_engine is the --euler-leaf-engine value: maximum, heavy,
    heavy_static, wfa. Note maximum, not max.
    """
    return _mean_over_sweeps("euler", leaf_engine)


# the number of matrices the Euler campaign covers, after the warm-up
EULER_WINDOW = len(_mean_over_sweeps("euler", "maximum"))


if __name__ == "__main__":
    engines = [("max", "maximum"), ("wfa", "wfa"),
               ("heavy", "heavy"), ("heavy_static", "heavy_static")]
    if CAMPAIGN != "variance_2026":
        engines.append(("wwfa", "wwfa"))
    for e, p in engines:
        dr, de = radix_frame(e), euler_frame(p)
        print(f"{e:13s} rows {len(dr):4d}  BvN {dr.bvn_runtime.mean()*1000:7.1f} ms "
              f"N {dr.bvn_perms.mean():8.1f}  C {dr.bvn_cycle.mean():9.2f}   "
              f"euler rows {len(de):4d}  S {de.bvn_cycle.mean():9.2f}")
