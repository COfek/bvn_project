# Tier rules, timed (overnight 2026-11-02)

Experimental, not one of the paper's canonical campaigns. Backs one sentence of §5.1.2
(the fourth-tier remark, Chen M19, option A).

## What ran

- Rules: `heuristic` (three tiers, the paper's), `heuristic_ruined` (four tiers: already
  non-consistent pairs ranked just below the empty tier), `heuristic_ruined_last` (`b_last`).
- Engines: Hungarian (`maximum`), GW Dynamic (`heavy`), GW Static (`heavy_static`), WFA.
- n = 256, k = 256, max weight 64, 105 matrices (first 5 dropped as warm-up), depths 0-4.
- Two sweeps, the second in a rotated rule order. Ran 01:00-05:28, nothing else on the
  machine; memory trace in `memtrace.csv`.
- Per run: `sweep{1,2}/eng_<engine>_<rule>/<stamp>/results_stats.csv` and
  `euler_comparison_summary.txt`.

## Depth 4 (times: mean of the two sweeps)

| Engine | N: 3 tiers / 4 tiers / b_last | max-leaf ms: 3 tiers / b_last | split ms |
|---|---|---|---|
| Hungarian | 851 / 843 / 839 | 41.1 / 40.4 | 37.5 / 37.5 |
| GW Dynamic | 3268 / 3256 / 3231 | 44.5 / 44.0 | 38.0 / 38.4 |
| GW Static | 7533 / 7517 / 7508 | 52.2 / 52.0 | 39.7 / 40.0 |
| WFA | 11436 / 11417 / 11398 | 147 / 147 | 40.8 / 40.6 |

N is identical across the two sweeps. The cycle stays S = 8174 for the three exact engines;
WFA's moves by under 0.3%. Depths 1-3 show smaller N gains.

## Reading

- Four tiers lower N by at most 0.94% (Hungarian, 851 -> 843); b_last by at most 1.41%.
- Every time difference is 1-2%, inside the known 6% between-run spread: no measurable
  change in running time, either way.

## Decision

Ofek chose option A on 2026-11-02: keep three tiers, and state the fourth tier's effect in
one sentence ("lowered N by less than 1% and left the running time unchanged").
