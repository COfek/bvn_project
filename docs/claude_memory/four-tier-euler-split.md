---
name: four-tier-euler-split
description: "Ofek's four-tier Euler split (heuristic_ruined) and the better b_last variant: small gains, why they work, and the open decision; data in euler_exp/ruined_tier (experimental)"
metadata:
  node_type: memory
  type: project
  originSessionId: 26bcde8f-a14e-4d6d-9736-b16700cca933
  modified: 2026-11-02T10:15:45.061Z
---

Overnight 2026-11-01. Report: bvn_project `euler_exp/ruined_tier/REPORT.md`. Data are
experimental, never the paper's canonical source.

- **The rule.** `--euler-split-method heuristic_ruined` (commit 49e0266) is an option in
  `src/`; the default `heuristic` (three tiers, the paper's) is unchanged.
- **The gain** (n = 256, 100 matrices):
  - It is sparser on every matrix and depth: leaf non-zeros -1.6% at depth 4, which is 8%
    of the excess splitting adds.
  - N -0.95% for Hungarian at depth 4, 0.2-0.4% for the other engines; WFA's cycle -0.2%.
  - No time cost.
- **Why** (ablation `agent_scripts/ablate_ruined_tier.py`, a proof checked by three
  reviewers):
  - Under three tiers a split pair is used up at once, by an exchange between its two
    vertices.
  - The only effective change is that a split pair waits below the free pairs at its
    lock-side vertex.
  - "Split pairs before whole ones at the other vertex" changes nothing. Ranking split pairs
    above the free pairs reproduces three tiers.
- **`b_last`** (split pairs taken last at their lock-side vertex) is better still: 12.7% of
  the excess removed, N -1.43% for Hungarian and -1.14% for GW Dynamic at depth 4. It is
  now `--euler-split-method heuristic_ruined_last` in `src/`; first timed in the overnight
  run of 2026-11-02 (`euler_exp/tier_rules_freed/`).

**Why:** adopting any new rule means re-running every Euler experiment (the dense campaign,
TM, MoE, skew, §7) and telling Chen, who agreed the three-tier text (changelog line 656).

**How to apply:**
- The paper keeps three tiers unless Ofek decides otherwise. The proposed text (CURRENT and
  SUGGESTED) is in the report, section 6.
- Tie the decision to [[euler-timing-held-memory]]: change the rule only if the Euler
  campaigns are re-run for that reason.
- **Decided 2026-11-02 (M19): option A.** The 2026-11-02 run (`euler_exp/tier_rules_freed/`,
  committed abdb073 with REPORT.md) showed N -0.94% (four tiers) / -1.41% (b_last) at best,
  times unchanged within noise. Paper keeps three tiers plus one sentence ("less than 1%",
  paper commit c417e90). Revisit only if the Euler campaigns are re-run anyway.
