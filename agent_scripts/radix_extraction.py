"""Digit-extraction time for the Radix campaigns, which did not record it.

Until 2026-10-04 decompose_radix returned the slowest plane's decomposition time
alone; the extraction of the digit planes, which runs once before them, was never
timed. It is 0.9 ms (B=16) to 2.4 ms (B=2) per dense 256x256 matrix, which is up
to 13% of the fastest Radix setups, so it is now charged (radix_decomposition.py
times it), and the campaigns run before that are corrected here: each campaign
matrix is rebuilt from its seed, extract_digit_planes is timed on it (best of
REPEATS runs, so background activity barely enters), and the result is stored
next to the campaign. The table and figure scripts add it to the Radix times.

    python agent_scripts/radix_extraction.py variance      # run/variance_2026, dense
    from radix_extraction import extraction_ms             # {(matrix_index, base): ms}
"""
from __future__ import annotations
from _paths import REPO  # repository-relative paths

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = str(REPO)
sys.path.insert(0, ROOT)
from src.algorithms.radix_decomposition import extract_digit_planes  # noqa: E402
from src.utils.matrix_generator import generate_matrix  # noqa: E402

OUT = os.path.join(ROOT, "run", "radix_extraction")
REPEATS = 5
BASES = (16, 8, 4, 2)

# the variance campaign: -n 256 -k 256 --max-weight 64 -s 1000 --random-seed 42,
# matrix i seeded with 42 + i (runner.py)
CAMPAIGNS = {
    "variance": dict(n=256, k=256, max_weight=64, seed=42, count=1000),
}


def time_extraction(matrix: np.ndarray, base: int) -> float:
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        extract_digit_planes(matrix, base)
        best = min(best, time.perf_counter() - t0)
    return best


def measure(name: str) -> str:
    c = CAMPAIGNS[name]
    rows = []
    for i in range(c["count"]):
        D = generate_matrix(c["n"], c["k"], c["max_weight"], np.random.default_rng(c["seed"] + i))
        for b in BASES:
            rows.append((i, b, time_extraction(D, b) * 1000.0))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.csv")
    pd.DataFrame(rows, columns=["matrix_index", "base", "extraction_ms"]).to_csv(path, index=False)
    return path


def _save(name: str, rows: list[dict]) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def measure_tm() -> str:
    """run/tm_workload (Table 5): per regime, seed 42, one warm-up matrix, then 10."""
    sys.path.insert(0, os.path.join(ROOT, "agent_scripts"))
    import tm_workload_experiment as tw
    stored = pd.read_csv(os.path.join(ROOT, "run", "tm_workload", "tm_results.csv"))
    rows = []
    for regime, kind, knob in tw.REGIMES:
        rng = np.random.default_rng(42)
        tw.make_matrix(kind, knob, rng)                       # the warm-up draw
        for m in range(tw.N_MATRICES):
            M = tw.make_matrix(kind, knob, rng)
            S_stored = stored[(stored.regime == regime) & (stored.matrix == m)].S.iloc[0]
            assert int(M.sum(axis=1)[0]) == int(S_stored), (regime, m, "matrix not reproduced")
            for b in BASES:
                rows.append(dict(regime=regime, matrix=m, base=b,
                                 extraction_ms=time_extraction(M, b) * 1000.0))
    return _save("tm", rows)


def measure_scale() -> str:
    """run/radix_scale (Figure 4): per regime, seed 42, one warm-up matrix, then 4,
    each decomposed at every candidate scale c as c * M0."""
    sys.path.insert(0, os.path.join(ROOT, "agent_scripts"))
    import radix_scale_campaign as rs
    from src.utils.griner_avin_tm import tm_integer
    stored = pd.read_csv(os.path.join(ROOT, "run", "radix_scale", "scale_results.csv"))
    rows = []
    for regime, kind, knob in rs.REGIMES:
        rng = np.random.default_rng(42)
        mk = (lambda r: tm_integer(rs.N, knob, r, w_small=7)) if kind == "tm" \
            else (lambda r: generate_matrix(rs.N, k=knob, max_weight=64, rng=r))
        mk(rng)                                                # the warm-up draw
        for m in range(rs.N_MATRICES):
            M0 = mk(rng)
            mx0 = int(M0.max())
            for B in rs.BASES:
                for c, _D in rs.scale_candidates(B, mx0):
                    M = M0 * c
                    st = stored[(stored.regime == regime) & (stored.matrix == m)
                                & (stored.base == B) & (stored.scale == c)]
                    assert len(st) and int(st.max_entry.iloc[0]) == int(M.max()), \
                        (regime, m, B, c, "matrix not reproduced")
                    rows.append(dict(regime=regime, matrix=m, base=B, scale=c,
                                     extraction_ms=time_extraction(M, B) * 1000.0))
    return _save("radix_scale", rows)


def measure_moe() -> str:
    """run/moe_workload (Table 6): the 12 MoE trace matrices."""
    sys.path.insert(0, os.path.join(ROOT, "agent_scripts"))
    import moe_trace_experiment as mt
    from src.utils.moe_trace import load_all
    stored = pd.read_csv(os.path.join(ROOT, "run", "moe_workload", "moe_results.csv"))
    rows = []
    for t in load_all(mt.TRACES):
        M = t.matrix.astype(np.float64)
        assert t.label in set(stored.matrix), (t.label, "trace not in the stored results")
        regime = "MoE mmlu" if "mmlu" in t.dataset else "MoE throughput"
        for b in BASES:
            rows.append(dict(regime=regime, matrix=t.label, base=b,
                             extraction_ms=time_extraction(M, b) * 1000.0))
    return _save("moe", rows)


def correct_summary(summary: pd.DataFrame, name: str) -> pd.DataFrame:
    """Add the mean extraction time of each (regime, base) to the Radix rows of a
    per-(regime, engine, config) summary whose configs are named 'B<base>'. The
    summary's runtime_ms is a mean over matrices, so adding the mean extraction is
    exact. The extraction also goes into split_ms, Radix's counterpart of Euler's split."""
    ext = pd.read_csv(os.path.join(OUT, f"{name}.csv"))
    mean = ext.groupby(["regime", "base"]).extraction_ms.mean().to_dict()
    out = summary.copy()
    for i, r in out.iterrows():
        cfg = str(r["config"])
        if cfg.startswith("B") and cfg[1:].isdigit():
            add = mean[(r["regime"], int(cfg[1:]))]
            out.at[i, "runtime_ms"] = r["runtime_ms"] + add
            if "split_ms" in out.columns:
                out.at[i, "split_ms"] = r["split_ms"] + add
    return out


def correct_scale_results(res: pd.DataFrame) -> pd.DataFrame:
    """Add each row's extraction time to run/radix_scale's runtime_ms."""
    ext = pd.read_csv(os.path.join(OUT, "radix_scale.csv"))
    key = ["regime", "matrix", "base", "scale"]
    out = res.merge(ext, on=key, how="left", validate="many_to_one")
    assert out.extraction_ms.notna().all(), "a radix_scale row has no extraction time"
    out["runtime_ms"] = out["runtime_ms"] + out["extraction_ms"]
    return out.drop(columns="extraction_ms")


def extraction_ms(name: str) -> dict[tuple[int, int], float]:
    d = pd.read_csv(os.path.join(OUT, f"{name}.csv"))
    return {(int(r.matrix_index), int(r.base)): float(r.extraction_ms) for r in d.itertuples()}


def add_to_radix_times(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Add each matrix's extraction time to every `<engine>_<base>_time` column (seconds)."""
    ext = extraction_ms(name)
    out = frame.copy()
    for col in out.columns:
        parts = col.split("_")
        if col.endswith("_time") and len(parts) >= 3 and parts[-2].isdigit():
            b = int(parts[-2])
            out[col] = out[col] + out["matrix_index"].map(lambda i: ext[(int(i), b)] / 1000.0)
    return out


MEASURE = {"variance": lambda: measure("variance"), "tm": measure_tm,
           "radix_scale": measure_scale, "moe": measure_moe}

if __name__ == "__main__":
    for name in sys.argv[1:] or MEASURE:
        t0 = time.time()
        path = MEASURE[name]()
        d = pd.read_csv(path)
        print(f"{name}: {path}  ({time.time() - t0:.0f} s)")
        grp = ["regime", "base"] if "regime" in d.columns else ["base"]
        print(d.groupby(grp)["extraction_ms"].describe()[["mean", "min", "max"]].round(3))
