"""
Post-run comparison summaries.

The runner records raw per-matrix metrics (one row per matrix, one column
group per configuration). This module turns those into the actual
comparisons of interest for the Euler framework experiments:

  1. Matching-extraction time vs. the BvN baseline (per depth, with the
     "vs previous depth" ratio that tests the halving hypothesis), with
     split cost reported separately.
  2. Euler leaves vs. Radix digit planes at comparable parallel-unit
     counts (only when Radix keys are present in the same run).

Aggregation is the mean over matrices (the first WARMUP_DROP samples are
excluded when there are enough, mirroring the plotting convention for
JIT warm-up).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .stats import DecompositionStats

WARMUP_DROP = 5


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _key_mean(stats: List[DecompositionStats], key: str, index: int) -> Optional[float]:
    vals = []
    for s in stats:
        res = s.radix_multi_results.get(key)
        if res is not None and len(res) > index and res[index] is not None:
            vals.append(res[index])
    return _mean(vals)


def write_euler_comparison_summary(
    stats_list: List[DecompositionStats],
    out_path: Path,
) -> Optional[str]:
    """
    Build the Euler-framework comparison summary.

    Returns the formatted text (also written to out_path), or None when the
    stats contain no Euler framework keys.
    """
    stats = (stats_list[WARMUP_DROP:]
             if len(stats_list) > 2 * WARMUP_DROP else stats_list)

    all_keys = set()
    for s in stats:
        all_keys.update(s.radix_multi_results.keys())

    def _depth_of(key: str) -> int:
        try:
            return int(key.rsplit("depth", 1)[1])
        except (ValueError, IndexError):
            return 0  # unknown depth format — sort first, never crash

    euler_keys = sorted(
        (k for k in all_keys if k.startswith("euler_") and "depth" in k),
        key=_depth_of,
    )
    radix_keys = sorted(
        (k for k in all_keys if not k.startswith("euler_")),
        key=lambda k: _key_mean(stats, k, 3) or 0,   # by mean plane count
    )
    if not euler_keys:
        return None

    baseline_ms = _mean([s.runtime_bvn for s in stats])
    baseline_ms = baseline_ms * 1e3 if baseline_ms is not None else None
    baseline_cycle = _mean([s.cycle_length_bvn for s in stats])
    baseline_perms = _mean([float(s.num_permutations_bvn) for s in stats])

    lines: List[str] = []
    add = lines.append

    # ------------------------------------------------------------------
    # Section 1: extraction time vs. baseline (the depth sweep)
    # ------------------------------------------------------------------
    add("=" * 78)
    add("Euler framework comparison summary"
        f"  ({len(stats)} matrices, first {len(stats_list) - len(stats)} dropped as warm-up)")
    add("=" * 78)
    add("")
    add("[1] Matching-extraction time vs. BvN baseline (same engine, same matrices)")
    add("    'max-leaf' = slowest single leaf (simulated parallel extraction);")
    add("    'split' = sequential splitting phase, reported separately.")
    add("")
    add(f"{'config':<28} {'units':>6} {'split ms':>9} {'max-leaf ms':>11} "
        f"{'vs baseline':>11} {'vs lvl':>8} {'cycle':>9} {'perms':>7}")
    add("-" * 96)
    if baseline_ms is not None:
        add(f"{'bvn_baseline (depth 0)':<28} {1:>6} {'-':>9} {baseline_ms:>11.2f} "
            f"{1.0:>11.3f} {'-':>8} "
            f"{baseline_cycle if baseline_cycle is not None else float('nan'):>9.0f} "
            f"{baseline_perms if baseline_perms is not None else float('nan'):>7.0f}")

    prev_ms = baseline_ms
    prev_depth = 0
    for key in euler_keys:
        t_ms = (_key_mean(stats, key, 0) or 0.0) * 1e3
        cyc = _key_mean(stats, key, 1)
        perms = _key_mean(stats, key, 2)
        units = _key_mean(stats, key, 3)
        split = _key_mean(stats, key, 4)
        depth = _depth_of(key)
        vs_base = t_ms / baseline_ms if baseline_ms else float("nan")
        # Per-level ratio: normalised by the number of depth levels between
        # this row and the previous one, so the halving hypothesis reads as
        # ~0.5 even when --euler-depths skips levels (e.g. 1 3).
        d_levels = max(depth - prev_depth, 1)
        vs_lvl = ((t_ms / prev_ms) ** (1.0 / d_levels)
                  if prev_ms and t_ms > 0 else float("nan"))
        add(f"{key:<28} {units or 0:>6.1f} "
            f"{(split or 0.0) * 1e3:>9.2f} {t_ms:>11.2f} "
            f"{vs_base:>11.3f} {vs_lvl:>8.3f} "
            f"{cyc if cyc is not None else float('nan'):>9.0f} "
            f"{perms if perms is not None else float('nan'):>7.0f}")
        prev_ms = t_ms
        prev_depth = depth
    add("")
    add("    Halving hypothesis: 'vs lvl' ~= 0.5 means each split level halves the")
    add("    max-leaf extraction time (geometric per-level ratio when depths skip).")

    # ------------------------------------------------------------------
    # Section 2: Euler leaves vs. Radix planes (matched unit counts)
    # ------------------------------------------------------------------
    if radix_keys:
        add("")
        add("[2] Euler leaves vs. Radix digit planes (same matching engine)")
        add("    Compare rows with similar 'units' - that is the matched-parallelism")
        add("    comparison; 'max-unit' is the slowest leaf/plane.")
        add("")
        add(f"{'config':<28} {'units':>6} {'max-unit ms':>11} {'cycle':>9} "
            f"{'cycle/S':>8} {'perms':>7}")
        add("-" * 74)
        rows = []
        for key in radix_keys + euler_keys:
            t_ms = (_key_mean(stats, key, 0) or 0.0) * 1e3
            cyc = _key_mean(stats, key, 1)
            perms = _key_mean(stats, key, 2)
            units = _key_mean(stats, key, 3) or 0.0
            rows.append((units, key, t_ms, cyc, perms))
        for units, key, t_ms, cyc, perms in sorted(rows):
            rel_c = (cyc / baseline_cycle
                     if cyc is not None and baseline_cycle else float("nan"))
            add(f"{key:<28} {units:>6.1f} {t_ms:>11.2f} "
                f"{cyc if cyc is not None else float('nan'):>9.0f} "
                f"{rel_c:>8.3f} "
                f"{perms if perms is not None else float('nan'):>7.0f}")
        add("")
        add("    cycle/S > 1.0 marks cycle-length inflation (Radix pays this; Euler")
        add("    leaves stay doubly stochastic, so their cycle stays at S).")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return text
