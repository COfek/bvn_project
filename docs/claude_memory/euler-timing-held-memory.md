---
name: euler-timing-held-memory
description: "The Euler campaign runner times each depth while earlier results sit in memory; GW Static and WFA Euler times in the paper are inflated (split up to 42%), Hungarian and GW Dynamic are not"
metadata:
  node_type: memory
  type: project
  originSessionId: 26bcde8f-a14e-4d6d-9736-b16700cca933
  modified: 2026-11-01T07:32:25.946Z
---

Found overnight 2026-11-01 (bvn_project `euler_exp/ruined_tier/REPORT.md`, section 4).

- **Cause.** `src/runner.py` `_compute_for_index` keeps the BvN baseline's
  `bvn_components` and the previous depth's `comps` alive while each Euler depth is timed.
  Every configuration is a dense n x n float64 matrix (512 KB at n = 256) built inside the
  timed region. The WFA campaign peaks at 19 GB with 0.8 GB free, GW Static at 12.9 GB.
- **Effect.** In all three August sweeps (`run/variance_2026`) and again on 2026-11-01:
  - GW Static's and WFA's Euler times grow through a campaign: slowest leaf ×1.2-1.27,
    split at depth 4 ×1.32-1.49, comparing the last 20 matrices with the first 20.
  - Hungarian, GW Dynamic, Radix and every BvN baseline do not drift.
  - The split is identical work for every engine, yet its depth-4 mean is 38.0 / 40.4 /
    54.3 / 59.9 ms in the Hungarian / GW Dynamic / GW Static / WFA campaigns.
- **Size.** Measured with every result freed before each timed call
  (`paired_results_rep2.csv`), the depth-4 split takes 38-42 ms for every engine and the
  drift disappears. GW Static's and WFA's depth-4 split + slowest leaf is 13-14% below the
  August means.
- **Not affected.** N and C (deterministic); Hungarian and GW Dynamic times (within 5%).

**Why:** the paper's `tab:eulertab` times, `fig:euler`, §6.3-6.4 and §7's T_compute use
these campaigns. R12's pending wording ("split 38-60 ms at depth 4 per engine") is this
artefact.

**How to apply:**
- Never quote a per-engine Euler split time, or a GW Static or WFA Euler time, as a property
  of the engine until the campaigns are re-run with the memory freed.
- The decision is open (Ofek): fix the runner, rerun one sweep, then decide.

See [[campaign-data-provenance]] and [[four-tier-euler-split]].
