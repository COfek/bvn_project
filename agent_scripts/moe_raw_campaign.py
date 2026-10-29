"""Section 5.8 -- decomposition of measured MoE traffic, WITHOUT regularisation.

The traces in paper_traces/ are real expert-parallel all-to-all matrices and are
NOT S-regular: column sums are expert popularities, which are skewed by
construction.  We decompose them as measured.  No padding, no scaling, no
rounding -- the matrices go into the engines exactly as recorded.

The consequence is the point of the section.  A non-regular residual admits no
perfect matching on its positive entries, so the engines peel *partial* matchings
and the decomposition is weak: by Proposition~\\ref{prop:cycle}, C > S strictly.
We report the excess directly as the port utilization

    eta = S / C,

the fraction of scheduled port-time that carries traffic.  S here is the lower
bound max(max row sum, max column sum): no schedule can finish sooner, because
some port must transmit that many units.

Outputs (run/moe_raw/):
    moe_raw_results.csv   one row per (trace, framework, engine)
    moe_raw_summary.txt   human-readable digest
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from src.algorithms.bvn import bvn_decomposition
from src.algorithms.radix_decomposition import decompose_radix
from src.utils.moe_trace import load_all

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACES = os.path.join(ROOT, "paper_traces")
OUT = os.path.join(ROOT, "run", "moe_raw")

ENGINES = ["maximum", "heavy", "heavy_static", "wfa"]
LABEL = {"maximum": "Hungarian", "heavy": "GW Dynamic",
         "heavy_static": "GW Static", "wfa": "WFA"}
BASES = [2, 4, 8, 16]
REPEATS = 5          # timing repeats; the first is discarded as JIT warm-up
WARMUP = 1


def lower_bound_S(M: np.ndarray) -> int:
    """Cycle-length lower bound for a non-regular matrix.

    Every unit in a row must occupy a distinct slot on that input port, and the
    same holds per column, so C >= max over all row and column sums.  For a
    regular matrix this collapses to the usual S.
    """
    return int(max(M.sum(axis=1).max(), M.sum(axis=0).max()))


def measure(fn, *args, **kwargs):
    """Run fn REPEATS times, discard the warm-up, return (result, median ms)."""
    times = []
    result = None
    for r in range(REPEATS):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000.0)
    return result, float(np.median(times[WARMUP:]))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    traces = load_all(TRACES)
    rows = []

    for t in traces:
        M = t.matrix.astype(np.float64)
        S = lower_bound_S(t.matrix)
        n = t.n
        # How far from regular this trace is, reported alongside every result.
        rs, cs = t.matrix.sum(1), t.matrix.sum(0)
        skew = float(cs.max() / cs.min())

        for eng in ENGINES:
            comps, ms = measure(bvn_decomposition, M.copy(), matching_algorithm=eng)
            recon = sum(c.weight * c.permutation for c in comps)
            N = len(comps)
            C = float(sum(c.weight for c in comps))
            full = sum(1 for c in comps if c.permutation.sum() == n)
            rows.append(dict(
                trace=t.label, model=t.model, dataset=t.dataset, layer=t.layer_id,
                framework="BvN", base=0, engine=LABEL[eng], n=n, tokens=int(t.matrix.sum()),
                S=S, N=N, C=C, eta=S / C, C_over_S=C / S, full_perms=full,
                frac_full=full / N, runtime_ms=ms, col_skew=skew,
                recon_err=float(np.abs(recon - M).max()),
            ))

        for base in BASES:
            for eng in ENGINES:
                (comps, plane_ms, planes), ms = measure(
                    decompose_radix, M.copy(), base=base, matching_method=eng)
                recon = sum(c.matrix for c in comps)
                N = len(comps)
                C = float(sum(c.weight for c in comps))
                full = sum(1 for c in comps if (c.matrix > 0).sum() == n)
                rows.append(dict(
                    trace=t.label, model=t.model, dataset=t.dataset, layer=t.layer_id,
                    framework=f"Radix B={base}", base=base, engine=LABEL[eng], n=n,
                    tokens=int(t.matrix.sum()), S=S, N=N, C=C, eta=S / C, C_over_S=C / S,
                    full_perms=full, frac_full=full / N,
                    runtime_ms=plane_ms * 1000.0, col_skew=skew,
                    recon_err=float(np.abs(recon - M).max()),
                ))
        print(f"  done {t.label}")

    df = pd.DataFrame(rows)
    csv = os.path.join(OUT, "moe_raw_results.csv")
    df.to_csv(csv, index=False)

    bad = df[df.recon_err > 1e-6]
    lines = [
        "Measured MoE traffic, decomposed as recorded (no regularisation)",
        "=" * 72, "",
        f"traces: {df.trace.nunique()}   n = {df.n.unique().tolist()}",
        f"exact reconstruction on all {len(df)} runs: {bad.empty}",
        "",
        "Per-trace lower bound S and column skew",
        "-" * 72,
    ]
    # NB: name this column anything but `skew` -- DataFrame.skew is a method, so
    # `r.skew` would resolve to the bound method rather than the value.
    per = df.groupby("trace").agg(tokens=("tokens", "first"), S=("S", "first"),
                                  cskew=("col_skew", "first")).reset_index()
    for _, r in per.iterrows():
        lines.append(f"  {r.trace:42s} tokens={r.tokens:8d}  S={r.S:7d}  col-skew={r.cskew:5.2f}x")

    lines += ["", "BvN: cycle inflation and utilization (mean over traces)", "-" * 72,
              f"  {'engine':12s} {'N':>7s} {'C/S':>7s} {'eta':>7s} {'full perms':>11s} {'T [ms]':>9s}"]
    g = df[df.framework == "BvN"].groupby("engine")
    for eng, sub in g:
        lines.append(f"  {eng:12s} {sub.N.mean():7.1f} {sub.C_over_S.mean():7.3f} "
                     f"{sub.eta.mean():7.3f} {sub.frac_full.mean():10.1%} {sub.runtime_ms.mean():9.2f}")

    lines += ["", "Radix: cycle inflation by base (mean over traces and engines)", "-" * 72,
              f"  {'framework':12s} {'N':>7s} {'C/S':>7s} {'eta':>7s} {'T [ms]':>9s}"]
    for fw, sub in df[df.framework != "BvN"].groupby("framework"):
        lines.append(f"  {fw:12s} {sub.N.mean():7.1f} {sub.C_over_S.mean():7.3f} "
                     f"{sub.eta.mean():7.3f} {sub.runtime_ms.mean():9.2f}")

    txt = os.path.join(OUT, "moe_raw_summary.txt")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {csv}\nwrote {txt}")


if __name__ == "__main__":
    main()
