# Four-tier Euler split: overnight results, 1 November 2026

**The question.** Your four-tier rule (`--euler-split-method heuristic_ruined`) holds pairs
that are already split back in a tier of their own. Does it improve on the paper's
three-tier same-direction heuristic, and what does it cost?

Nothing in the paper was changed. Where the numbers come from:
- **The tables** are printed by `agent_scripts/ruined_tier_report_tables.py`, some as
  selections of its rows.
- **The memory figures** come from `summarise_memtrace.py`.
- **Every other figure** comes from a file or script listed in section 8. The exceptions are
  marked where they appear: numbers reported by a reviewer, and the held-memory sizes
  computed in the text.

## In short

1. **It works, a little.** On the paper's dense workload (n = 256, the 100 matrices i =
   5..104 of the Euler tables), the four-tier rule gives sparser leaves on every matrix at
   every depth.
   - Leaf non-zeros fall by 1.6% at depth 4. That removes 8% of the non-zeros splitting
     adds.
   - N falls by 0.95% for Hungarian at depth 4 and by 0.2-0.4% for the other engines.
   - WFA's cycle shortens by 0.2% at depth 4.
   - No change in time is established. The split takes about the same time. For the slowest
     leaf, the paired runs and the campaigns disagree by up to 2.5%.
2. **Why it works.** An ablation of seven variants on all 400 (matrix, depth) cases settles
   which part of the rule matters. A proof sketch explains the result. Three reviewers
   checked it against the code; none found a counterexample to the identities. One showed
   that a one-line description of the rule must be stated more exactly (below).
   - Under the three-tier rule, a pair the traversal splits is used up at once: the
     traversal goes back and forth between the pair's two vertices.
   - The four-tier rule's one effective change: at the vertex of its lock, a split pair
     leaves tier 1 and is taken after the pairs not yet locked, but before the pairs locked
     the other way. So its copies are kept for later.
   - Its other half, taking split pairs before whole ones at the other vertex, never
     changes a pick.
   - Ranking split pairs above the free pairs gives back exactly the three-tier rule.
3. **A better variant came out of the ablation.** `b_last` takes a split pair *last* at the
   vertex of its lock, after the pairs locked the other way too.
   - It removes 12.7% of the excess at depth 4 (four tiers: 7.8%) and is sparser than the
     four-tier rule on all 100 matrices at every depth.
   - At depth 4, N falls by 1.43% for Hungarian and 1.14% for GW Dynamic against three
     tiers.
   - It exists only in the ablation script and has not been timed.
4. **The paper's Euler timings have a problem, found on the way.** The campaign runner
   keeps earlier results in memory while it times each Euler depth: the BvN baseline's,
   and the previous depth's. Every configuration is stored as a dense 256 x 256 matrix, so
   the WFA campaigns reach 18-19 GiB.
   - The likely consequence: GW Static's and WFA's Euler times grow over a campaign. The
     split, which is identical work for every engine, costs 38-60 ms at depth 4 depending on
     the engine: the figure pending for R12.
   - Measured with every result freed, the split takes 38-42 ms in every engine's calls.
     It falls by 27% (GW Static) and 30% (WFA) at depth 4.
   - The slowest leaf moves less, and partly within run-to-run variation.
   - This is a well-supported hypothesis, not a proven cause. The campaign runner itself
     has not yet been rerun with the memory freed.
   - Section 7's maps should not change: GW Static's and WFA's Euler setups win nowhere.
5. **Recommendation.**
   - **Now:** word R12 so that it quotes no per-engine split time.
   - **Next:** quantify the timing problem with one rerun of the four Euler campaigns
     after a small runner fix (a ready task is in the app).
   - **The split rule:** keep three tiers in this paper unless you want to rerun every Euler
     experiment for a gain of 1-1.5% in N (section 5).

## What we ran

All runs were on the paper's machine: 24 logical cores, 31.2 GiB of RAM, Windows 11,
Python 3.13.0, NumPy 2.3.5, SciPy 1.17.0 and Numba 0.63.1. All used the paper's Euler
matrices, `generate_matrix(256, k=256, max_weight=64, rng=default_rng(42 + i))` for
i = 0..104, at depths 1-4 with the four leaf engines. As in the paper's tables, statistics
are over i = 5..104.

| Part | What | Command | Time |
|---|---|---|---|
| 1 | both rules back to back on every matrix, engine and depth; the order alternates with the matrix | `agent_scripts/euler_ruined_tier_compare.py 105` at commit `ff5a7e2` | 01:42-03:16 (94 min) |
| 2 | both rules as full campaigns, with the parameters of the paper's campaign (8 campaigns) | `main.py ... --engine euler_bvn --euler-split-method {heuristic, heuristic_ruined}` | 03:16-04:58 |
| 3 | part 1 again, with each result freed before the next timed call ("paired run 2") | `agent_scripts/euler_ruined_tier_compare.py 105 _rep2` | 04:58-06:08 (69 min) |
| 4 | the ablation: seven variants of the split, nothing timed | `agent_scripts/ablate_ruined_tier.py 105 16 1` | 06:08-06:11 |

**The code.**
- The rule is in commit `49e0266`, as an option; the default path is unchanged.
- Tests: `tests/test_euler_splitting.py::TestHeuristicRuinedTier`. The commit message records
  1031 passing tests; a reviewer re-ran the 69 four-tier tests.
- Before the run, a read-only review (4 agents) of the kernel, the benchmark and the
  statistics found one flaw: part 1's script kept the previous result alive while the next
  call was timed. Part 3 fixes it. The review found nothing else.

**What the night checked.** Every check passed:
- Tonight's three-tier campaigns reproduce N and the cycle length of all three August sweeps
  on every matrix.
- Both campaigns equal the paired runs' N and C on every matrix.
- The ablation kernel's 3tier and 4tier settings reproduce the production splits leaf by
  leaf on all 105 matrices. Their N and C equal the paired run's on the five matrices
  counted under both rules.

## 1. Four tiers against three: the exact results

N, C and the leaf non-zeros depend on no timing. They are identical in parts 1 and 3.

### Leaf non-zeros

| depth | nnz(D) | leaf nnz, 3 tiers | 4 tiers | change | excess over nnz(D), 3 tiers | 4 tiers | pairs newly in both halves at this level, 3 tiers | 4 tiers | 4 tiers sparser on |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 41075 | 41800 | 41740 | -0.14% | 1.77% | 1.62% | 725 | 665 | 100/100 |
| 2 | 41075 | 43247 | 43067 | -0.42% | 5.29% | 4.85% | 1447 | 1327 | 100/100 |
| 3 | 41075 | 46148 | 45736 | -0.89% | 12.35% | 11.35% | 2901 | 2669 | 100/100 |
| 4 | 41075 | 51922 | 51072 | -1.64% | 26.41% | 24.34% | 5774 | 5336 | 100/100 |

The excess is what splitting adds: the leaves' non-zeros above those of D, which is
41,075 per matrix on average.
- The four-tier rule removes about 8% of it at every depth (7.8-8.3%).
- Odd-S levels carry a caveat. There, one matching is taken out first and added to red, so a
  few pairs land in both halves without being split by the traversal. The last two columns
  count pairs in both halves, whatever the cause.

### Configurations N (mean over matrices)

| engine | depth | N, 3 tiers | N, 4 tiers | change | fewer / same / more |
|---|---|---|---|---|---|
| Hungarian | 1 | 507.5 | 505.8 | -0.33% | 61 / 9 / 30 |
| Hungarian | 2 | 562.0 | 560.2 | -0.33% | 65 / 6 / 29 |
| Hungarian | 3 | 671.1 | 667.0 | -0.61% | 80 / 8 / 12 |
| Hungarian | 4 | 850.6 | 842.6 | -0.95% | 94 / 2 / 4 |
| GW Dynamic | 1 | 1408.8 | 1408.3 | -0.04% | 46 / 1 / 53 |
| GW Dynamic | 2 | 1839.2 | 1837.5 | -0.09% | 49 / 1 / 50 |
| GW Dynamic | 3 | 2444.1 | 2445.4 | +0.05% | 52 / 0 / 48 |
| GW Dynamic | 4 | 3268.3 | 3256.2 | -0.37% | 64 / 1 / 35 |
| GW Static | 1 | 7531.4 | 7527.9 | -0.05% | 54 / 1 / 45 |
| GW Static | 2 | 7429.5 | 7424.9 | -0.06% | 55 / 0 / 45 |
| GW Static | 3 | 7449.7 | 7441.5 | -0.11% | 58 / 0 / 42 |
| GW Static | 4 | 7533.3 | 7516.5 | -0.22% | 79 / 1 / 20 |
| WFA | 1 | 10603.6 | 10603.3 | -0.002% | 49 / 3 / 48 |
| WFA | 2 | 10815.9 | 10810.7 | -0.05% | 53 / 1 / 46 |
| WFA | 3 | 11136.0 | 11123.0 | -0.12% | 62 / 1 / 37 |
| WFA | 4 | 11436.4 | 11416.6 | -0.17% | 73 / 0 / 27 |

**Reading N.**
- The gain grows with depth and is clearest for Hungarian: fewer configurations at depth 4
  on 94 of 100 matrices.
- At depths 1-3 the other engines change by at most 0.12%, mostly downward. Only WFA at
  depth 3 is consistent across matrices (62 fewer, 37 more). GW Dynamic changes sign at
  depth 3.

**What it would mean for `tab:eulertab`.** Under four tiers every N cell of its Euler rows
and every WFA cycle cell changes, for example:
- Hungarian depth 4: 851 → 843;
- GW Dynamic depth 4: 3268 → 3256;
- WFA's depth-4 cycle: 13,122 → 13,092.

Numbers quoted from the table in the text change too:
- §6.3's 2444 (GW Dynamic, depth 3) → 2445;
- WFA's 1.61 at depth 4 → 1.60;
- GW Static's 7450 → about 7442;
- §6.4's 507 → 506.

### Cycle length

**The strong engines.** For Hungarian, GW Dynamic and GW Static, C = S in all 2400 paired
runs per part, under both rules. S differs by matrix, from 7,584 to 8,907; its mean is
8,174.

**WFA** (a weak engine), as C/S averaged over matrices:

| depth | C/S, 3 tiers | C/S, 4 tiers | shorter / longer |
|---|---|---|---|
| 1 | 1.4877 | 1.4872 | 48 / 52 |
| 2 | 1.5287 | 1.5263 | 57 / 43 |
| 3 | 1.5785 | 1.5774 | 53 / 47 |
| 4 | 1.6053 | 1.6016 | 62 / 38 |

### The paper's worked examples

The three worked examples split identically under both rules:
- `fig:eulermaps`: 16 and 17 non-zeros;
- `fig:eulergraph`: 8 of 10 pairs whole;
- `fig:splitting` (d): 15 and 14 non-zeros.

Their figures and captions would not change (`agent_scripts/check_ruined_tier_paper_examples.py`).

## 2. Why it works: the ablation

At its own lock-side vertex, the four-tier rule takes a split pair out of tier 1 and puts
it in a tier of split pairs. At the vertex where the pair was split, against its lock, it
also takes split pairs before whole pairs locked the other way. The variants differ in
three choices:
- whether split pairs leave tier 1 at their lock-side vertex;
- whether the other vertex gets a split tier;
- where the split tier sits.

| variant | split pair leaves tier 1 at its lock side | split tier at the other vertex | split tier read | leaves, all 100 matrices x 4 depths |
|---|---|---|---|---|
| `3tier` | no | no | (none) | (reference) |
| `4tier` | yes | yes | after the free pairs, before the opposite pairs | sparser than 3tier on 400/400 |
| `a_only` | no | yes | after the free pairs | **identical to 3tier** on 400/400 |
| `b_only` | yes | no | after the free pairs, before the opposite pairs | **identical to 4tier** on 400/400 |
| `4tier_swap` | yes | yes | *before* the free pairs | **identical to 3tier** on 400/400 |
| `a_swap` | no | yes | before the free pairs | **identical to 3tier** on 400/400 |
| `b_last` | yes | no | *after* the opposite pairs (last) | sparser than 4tier on 400/400 |

The kernel is one copy of the production kernel with five switches. Its 3tier and 4tier
settings reproduce the production splits leaf by leaf on all 105 matrices. The other five
settings have no production counterpart.

### The reason

1. **One live lock per row.** A row gains a red lock only from tier 2. Tier 2 is reached
   only when tier 1 is empty, that is, when no red-locked pair of the row has copies left.
   So under three tiers each row has at most one red-locked pair with copies at any
   moment. The same holds for each column and blue.
2. **No locks after a forced pick.** A pair is split only at a forced pick, at a vertex
   with no free pair left. No pair at that vertex can be locked afterwards, so the split
   pair stays that vertex's first forced choice while it has copies.
3. **The exchange.** At its lock-side vertex, the split pair is the only tier-1 candidate
   (point 1). So the traversal alternates between the pair's two vertices until its copies
   are used up. Under three tiers no split pair ever has copies outside such an exchange.
4. **Consequences.**
   - The split tier at the other vertex never changes a pick: it only ever holds the pair
     being exchanged, which three tiers take anyway.
   - Ranking split pairs above the free pairs reproduces the exchange.
   - Only ranking them below the free pairs, at their lock-side vertex, keeps copies for
     later.

**The steps the sketch needs** are written in the docstring of
`agent_scripts/ablate_ruined_tier.py`:
- locks are permanent and the validity tests monotone, so lazy deletion is harmless;
- no pop or restart can interrupt an exchange;
- why `b_only` equals `4tier`.

**What the reviewers did.** Three read-only reviewers checked the argument against the code.
One built an exact transliteration of the kernel, numba's tie order included. On small
random matrices (n ≤ 40), they reported:
- zero violations of the four points over 5.2 million steps and 385,000 splits;
- the identities above holding on 2,814 further matrices.

Their transcripts are not kept in the repository. The paper-scale evidence is the 400/400
identities in `ablation_splits.csv`, which `ablate_ruined_tier.py` reproduces.

**The one correction.** One reviewer showed that "below the free pairs" does not fix the
rule: other rankings that also put a split pair below the free pairs split differently.
`b_last` is one of them. The exact statement is the four-tier rule's: after the free pairs,
before the pairs locked the other way, most recently split first.

### `b_last`

The exchange suggests where a split pair's copies are worth most. At the vertex where it
was split, every later pick is forced. Taking one of its copies there avoids splitting a
whole pair. Taking one at its lock-side vertex leads the traversal to that vertex, where the
next pick is forced again. `b_last` takes split pairs at their lock-side vertex only when
nothing else is left.

**Leaf non-zeros, `b_last` against three tiers:**

| depth | leaf nnz | vs 3 tiers | excess | excess vs 3 tiers | sparser than 4tier |
|---|---|---|---|---|---|
| 1 | 41710 | -0.22% | 1.55% | -12.5% | 100/100 |
| 2 | 42976 | -0.63% | 4.63% | -12.5% | 100/100 |
| 3 | 45502 | -1.40% | 10.78% | -12.7% | 100/100 |
| 4 | 50549 | -2.64% | 23.07% | -12.7% | 100/100 |

**N at depth 4, `b_last` against both rules:**

| engine | N | vs 3 tiers | vs 4 tiers | fewer / same / more than 4 tiers |
|---|---|---|---|---|
| Hungarian | 838.5 | -1.43% | -0.48% | 82 / 4 / 14 |
| GW Dynamic | 3231.1 | -1.14% | -0.77% | 74 / 1 / 25 |
| GW Static | 7507.9 | -0.34% | -0.11% | 60 / 2 / 38 |
| WFA | 11398.4 | -0.33% | -0.16% | 74 / 0 / 26 |

The full table by depth is in `ablation_N_summary.csv`.

**Caveats.**
- At depths 1-2, `b_last` and four tiers are within ±0.3% of each other on N.
- WFA's cycle is the one exception. At depth 4 it sits between the two rules: C/S is
  1.6029, against 1.6053 (three tiers) and 1.6016 (four).
- At depths 1-3, WFA's cycle under `b_last` is slightly longer than under either rule, by
  0.02-0.21%.
- `b_last` is not in `src/` and has not been timed.

## 3. Time

**How the time is charged.**
- The slowest leaf is the largest of the leaves' times, measured one leaf after another.
- The split is measured as a wall-clock time; its levels run on threads.
- The Euler compute time is their sum.

**The split.** The four-tier split is within 3% of the three-tier one in paired run 2
(ratios 0.960-1.027), on average 0.5% slower. 1 lies inside 10 of the 16 intervals. The
four engines' ratios at one depth time the same work and still differ by up to 7%, so most
of that variation is run noise. In the campaigns the split times agree within 1.2 ms.

**The slowest leaf.** Four tiers over three tiers, as the geometric mean of the per-matrix
ratio with a bootstrap 95% interval:

| engine | depth | paired run 1 | paired run 2 | campaigns (per matrix) |
|---|---|---|---|---|
| Hungarian | 1 | 0.992 [0.943, 1.043] | 0.989 [0.981, 0.997] | 1.000 [0.997, 1.004] |
| Hungarian | 2 | 0.995 [0.985, 1.006] | 0.996 [0.990, 1.002] | 1.001 [0.997, 1.004] |
| Hungarian | 3 | 0.993 [0.985, 1.002] | 0.994 [0.986, 1.003] | 0.997 [0.992, 1.002] |
| Hungarian | 4 | 0.987 [0.980, 0.994] | 0.975 [0.966, 0.984] | 1.000 [0.994, 1.007] |
| GW Dynamic | 1 | 0.995 [0.989, 1.001] | 0.999 [0.988, 1.010] | 0.988 [0.984, 0.993] |
| GW Dynamic | 2 | 0.999 [0.966, 1.035] | 1.016 [1.002, 1.030] | 1.002 [0.996, 1.009] |
| GW Dynamic | 3 | 0.989 [0.951, 1.029] | 1.008 [0.993, 1.024] | 1.008 [0.997, 1.019] |
| GW Dynamic | 4 | 0.928 [0.901, 0.957] | 1.002 [0.980, 1.027] | 1.007 [0.983, 1.031] |
| GW Static | 1 | 0.998 [0.975, 1.020] | 0.999 [0.991, 1.008] | 0.992 [0.988, 0.997] |
| GW Static | 2 | 0.991 [0.958, 1.026] | 1.007 [0.992, 1.023] | 0.981 [0.972, 0.990] |
| GW Static | 3 | 1.008 [0.970, 1.048] | 0.990 [0.965, 1.015] | 0.981 [0.962, 0.999] |
| GW Static | 4 | 0.939 [0.903, 0.977] | 1.003 [0.969, 1.036] | 0.985 [0.957, 1.016] |
| WFA | 1 | 1.004 [0.984, 1.025] | 1.000 [0.994, 1.005] | 0.996 [0.995, 0.998] |
| WFA | 2 | 0.997 [0.975, 1.018] | 1.004 [0.994, 1.014] | 0.988 [0.985, 0.990] |
| WFA | 3 | 0.998 [0.983, 1.012] | 0.990 [0.974, 1.007] | 0.992 [0.977, 1.007] |
| WFA | 4 | 0.994 [0.981, 1.007] | 0.992 [0.967, 1.017] | 0.983 [0.967, 0.999] |

**What the columns mean.**
- **The paired runs** run both rules back to back, in alternating order.
- **The campaigns** are one campaign per rule, on the same matrices. Their interval covers
  only the variation between matrices, not between the two runs. On identical work, the
  BvN baselines of the two campaigns differ by 0.5-1.0%.

**Reading it.** The three measurements do not agree well enough to establish a change.
- **Hungarian at depth 4** is 1.3% faster in part 1 and 2.5% faster in part 3, but 0.0% in
  the campaigns.
  - In part 3 the gain appears only when the four-tier call runs second (0.948, against
    1.002 when it runs first).
  - Both its N (-0.95%) and its largest leaf's non-zeros (-1.6%) fall. Which of them would
    drive the time is not measured.
- **The campaigns** put GW Static and WFA 1-2% faster under four tiers at several depths,
  while paired run 2 gives about 1.00.
- **Taken together,** any effect of the rule on time is within about ±2.5%. None is
  confirmed.

**Part 1's 6-7% speed-up for the GW engines at depth 4 (0.928, 0.939) did not repeat** once
every result was freed (1.002, 1.003).
- In part 1 the GW times tripled over the run, and the ratio moved with them: from 1.06 in
  matrices 5-14 to 0.85 in matrices 95-104 (`drift_paired_blocks.csv`).
- The three-tier calls drifted more than the four-tier ones (×3.25 against ×2.63). Why is
  not known.
- So part 1's GW timings are unusable.

## 4. The problem with the paper's Euler timings

### What happens

In `src/runner.py`, `_compute_for_index`:
- The BvN baseline's result, `bvn_components`, stays referenced through the whole Euler
  depth loop.
- Each depth's `comps` is still alive while the next depth is timed.
- Every configuration is a dense n x n float64 matrix (512 KiB), built inside the timed
  region: `_matches_to_components` for the leaves, and the same in BvN and Radix.

Counting N × 512 KiB from the mean N values in `tab:eulertab` gives the earlier results held
during each timed call:

| engine | held at depth 1 | held at depths 2-4 |
|---|---|---|
| GW Static | 3.7 GiB | 7.4 GiB |
| WFA | 5.1 GiB | 10.3-10.6 GiB |

The memory trace (`memtrace_overnight.csv`) was sampled every 2 minutes from 03:22, so each
peak below is the highest sample, a lower bound on the true peak:

| tonight's campaigns | samples | highest private memory | lowest free RAM |
|---|---|---|---|
| Hungarian, 3 and 4 tiers | 1, 2 | 2.6, 2.0 GiB | 15.7, 16.2 GiB |
| GW Dynamic | 4, 5 | 4.6, 3.9 GiB | 13.7, 14.2 GiB |
| GW Static | 6, 7 | 12.0, 12.5 GiB | 6.6, 5.7 GiB |
| WFA | 12, 11 | 18.1, 18.7 GiB | 1.7, 0.8 GiB |
| paired run 2 (results freed) | 35 | 7.3 GiB | 12.2 GiB |

### The evidence

1. **Drift over a campaign.** The matrices are independent draws, so a time that grows with
   the matrix index measures the process, not the work. Drift is the median of matrices
   85-104 over the median of matrices 5-24.

   August campaigns, lowest and highest of the three sweeps:

   | engine | time | depth 1 | depth 2 | depth 3 | depth 4 |
   |---|---|---|---|---|---|
   | Hungarian | slowest leaf | 1.02-1.02 | 1.02-1.02 | 1.02-1.02 | 1.02-1.03 |
   | Hungarian | split | 0.97-0.98 | 1.00-1.00 | 0.98-0.99 | 0.99-1.00 |
   | GW Dynamic | slowest leaf | 1.03-1.05 | 1.05-1.06 | 1.06-1.06 | 1.02-1.04 |
   | GW Dynamic | split | 1.00-1.05 | 0.99-1.03 | 0.99-1.03 | 1.03-1.05 |
   | GW Static | slowest leaf | 1.03-1.04 | 1.23-1.26 | 1.06-1.09 | 1.24-1.27 |
   | GW Static | split | 1.01-1.03 | 1.14-1.18 | 1.10-1.12 | 1.38-1.42 |
   | WFA | slowest leaf | 1.06-1.07 | 1.19-1.22 | 1.08-1.17 | 0.93-1.16 |
   | WFA | split | 1.08-1.13 | 1.20-1.25 | 1.11-1.30 | 1.32-1.49 |

   Tonight's campaigns repeat the pattern under both rules (`drift_campaigns.csv`).

   **Reading it.**
   - **The split** is the clearest signal: it drifts up to ×1.32-1.49 at depth 4 for GW
     Static and WFA, and not for Hungarian or GW Dynamic.
   - **The slowest leaf** drifts ×1.19-1.27 at depths 2 and 4, but only ×1.03-1.17 at
     depths 1 and 3, although depths 2-4 hold about the same memory.
   - **The BvN baselines inside these same campaigns** run before the Euler loop and do not
     drift (×0.99-1.02).
   - **Radix and the standalone BvN baseline campaigns** drift ×0.87-1.13 across their 60
     timed columns. The low end is GW Static's baseline campaigns, which get *faster* over
     a run (×0.87-0.94).
2. **The same split, different times.** The split depends only on the matrix. Yet its mean
   at depth 4 is 38.0, 40.4, 54.3 and 59.9 ms in the Hungarian, GW Dynamic, GW Static and
   WFA campaigns. Against the Hungarian call on the same matrix (`split_vs_hungarian.csv`,
   three tiers, depth 4):

   | run | GW Dynamic | GW Static | WFA |
   |---|---|---|---|
   | part 1, previous result held | ×1.24 | ×1.60 | ×1.59 |
   | part 3, freed | ×1.04 | ×1.03 | ×1.08 |

   All of these intervals exclude 1. Freeing removes most of the spread, not all.
3. **The predictions.** Before part 3 ran, I wrote down three predictions for it
   (`prediction_before_rep2.txt`, saved at 04:34:49; part 3 started at 04:58). All three
   held:
   - The depth-4 split takes 38.7-42.1 ms in every engine's calls, three tiers (part 1:
     41.1-66.5 ms). It was predicted within about 10%; it came in at 9%.
   - GW Dynamic's depth-4 drift falls from ×3.25 to ×1.01 under three tiers, and from ×2.63
     to ×1.05 under four.
   - The paired four-tier/three-tier ratios move to about 1.

   But the freed run is not free of drift either: under three tiers GW Static's slowest leaf
   *falls* over it, ×0.85-0.88 at depths 3-4.

**What the evidence does not settle.**
- The held memory is about the same for every matrix, so it predicts a shift, not growth
  with the matrix index. Something accumulates, and what is not identified. The state of
  the process heap is one candidate, named in the predictions file.
- Part 3 changed more than the held memory: it calls `gc.collect()` before every timed call,
  runs four engines in one process, and runs no baseline.
- The campaign runner itself has not been rerun with the memory freed.

### How large it is

Three-tier means: the campaigns, with memory held (August, with the range of its three
sweeps, and tonight), against paired run 2, freed.

| engine | depth | slowest leaf (ms): August [range] | tonight | freed | freed / tonight | split (ms): August [range] | tonight | freed | freed / tonight | split + leaf: freed / August | freed / tonight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hungarian | 4 | 42.5 [42.4, 42.7] | 42.1 | 41.4 | 0.983 | 38.0 [37.7, 38.3] | 38.7 | 38.7 | 1.000 | 0.996 | 0.991 |
| GW Dynamic | 4 | 48.9 [48.3, 49.5] | 49.8 | 49.9 | 1.002 | 40.4 [40.2, 40.5] | 41.1 | 40.4 | 0.984 | 1.011 | 0.994 |
| GW Static | 2 | 242.7 [240.4, 246.0] | 242.3 | 222.1 | 0.916 | 28.8 [28.3, 29.0] | 28.9 | 26.5 | 0.918 | 0.916 | 0.917 |
| GW Static | 3 | 115.1 [113.4, 117.4] | 114.0 | 117.9 | 1.034 | 38.3 [38.0, 38.8] | 39.1 | 33.5 | 0.859 | 0.988 | 0.989 |
| GW Static | 4 | 66.8 [66.1, 67.7] | 66.8 | 65.1 | 0.975 | 54.3 [53.1, 54.9] | 54.3 | 39.8 | 0.734 | 0.866 | 0.867 |
| WFA | 2 | 505.6 [496.5, 514.3] | 494.9 | 465.2 | 0.940 | 29.8 [29.4, 30.2] | 30.0 | 26.9 | 0.896 | 0.919 | 0.937 |
| WFA | 3 | 282.4 [276.0, 288.3] | 275.4 | 269.4 | 0.978 | 42.3 [41.7, 43.1] | 43.2 | 34.0 | 0.788 | 0.934 | 0.953 |
| WFA | 4 | 172.2 [163.9, 180.3] | 162.7 | 157.6 | 0.969 | 59.9 [58.6, 60.9] | 60.1 | 42.1 | 0.701 | 0.860 | 0.896 |

All 16 engine-depth rows are printed by the table script.

**Reading it.**
- **The split is the robust part.** Freed, it is 27% (GW Static) and 30% (WFA) lower at
  depth 4, against both references, far outside the sweeps' own range.
- **The slowest leaf moves less.** Freed, it is 3-8% lower than tonight's campaigns at most
  depths for GW Static and WFA, and 3% higher at GW Static depth 3. At WFA depth 4 the
  August sweeps alone span 164-180 ms.
- **Split plus slowest leaf at depth 4** is 13% lower for GW Static, and 10-14% lower for WFA
  depending on the reference.
- **Hungarian and GW Dynamic** hold little memory and agree within 5%. That is about what
  the other differences between paired run 2 and a campaign are worth.

### What it touches in the paper

**What may change, once the campaigns are rerun with the memory freed:**
- `tab:eulertab`'s max-leaf cells for GW Static and WFA (§6.3 and §6.4 quote no GW Static or
  WFA Euler time);
- `fig:euler`, which plots split plus slowest leaf for every engine;
- the Euler rows of `tab:tm_comparison`, since `tm_workload_experiment.py` keeps the previous
  result during each timed call too;
- §6.1's three-run timing spread, computed over these campaigns;
- R12's pending wording: "38-60 ms at depth 4" per engine is mostly this artefact.

**Section 7** takes GW Static's and WFA's Euler times as inputs, but those setups win
nowhere. Hungarian Euler holds the map at depths 4, 3 and 2, and freed GW Static at depth 4
is still slower than Hungarian (105.0 against 80.1 ms). So the maps and shares should not
change. Rerunning the DCT script would confirm it.

**What the held memory does not touch:**
- N and C;
- Hungarian's and GW Dynamic's times;
- Radix and the baselines;
- §6.4's split cost, "16 ms at depth 1 to 38 ms at depth 4". It matches the Hungarian
  campaign; freed, the split takes 16.8 and 38.7 ms.

A rerun may still move any of these by a few percent, the size of run-to-run variation.

### Two further points

**The likely fix** is small. Drop `bvn_components` once its statistics are taken, and free
`comps` (with `gc.collect()`) before each depth's timed call. Then rerun the four Euler
campaigns.

**How much of the leaf time is building the dense matrices** is a separate question.
- Part 3 took 69 minutes for 105 matrices and both rules with the matrices built: about
  20 s per matrix and rule.
- The ablation counted N for 100 matrices (one variant) on every engine and depth, building
  the dense matrices only for Hungarian. The whole run took 2.3 minutes on 16 processes,
  including a 105-matrix check in one process (`ablation.log`).
- This only hints that the share may be large. It needs a proper measurement.

## 5. Recommendation

1. **R12, now.** Whatever is decided below, the paper should not present the split time as
   depending on the engine: the work is the same, and the spread comes from the
   measurement. The Hungarian campaign's figures, 16 ms at depth 1 and 38 ms at depth 4,
   are unaffected and are what §6.4 already quotes.
2. **The timing problem, next.** Fix the runner as above, behind an option so that the
   paper's numbers stay reproducible. Then rerun one sweep of the four Euler campaigns on
   an idle machine: about 50 minutes, going by tonight's campaigns.
   - Compare it with the August sweeps.
   - Then decide whether to rerun all three sweeps and regenerate `tab:eulertab`,
     `fig:euler` and the TM table.
   - The suggested task in the app sets this up.
3. **The split rule.** Keep three tiers in this paper unless you want the best rule
   available enough to pay for it. The gains are 1-1.5% in N at depth 4 at best, and no
   conclusion of the paper depends on them. A change means:
   - rerunning the dense Euler campaigns, and the TM and MoE experiments (the skew
     experiment is not in the paper);
   - rewriting §5.1.2's tier paragraph, Algorithm 1 and the cost sentence;
   - telling Chen, who reviews the paper. The paragraph follows your own review of the
     Chapter 5 drafts (14-16 August, changelog line 656), which asked for "the three-tier
     mechanism with pseudocode".
   - `b_last` would also need to be added to `src/`, tested, timed and drafted.

   If the Euler campaigns are rerun for item 2 anyway, the campaign rerun is shared, but the
   rest is not. Either way, the four-tier result and the ablation could go into the
   conclusion's future work in one or two sentences, if you want them recorded.

## 6. If a rule is adopted: what would change

### The paper text, for the four-tier rule

This is not applied, and is for your approval. It was drafted twice and reviewed by two
critics, one checking consistency with the paper and one reading as the advisor would.
This version takes draft B's structure with both critics' fixes. The reason it gives for
the place of split pairs comes from the ablation. That answers the advisor critic's
objection that the drafts' reason could be turned around.

**§5.1.2, lines 709-715.**

CURRENT:
```latex
\emph{same-direction heuristic}. A pair becomes \emph{locked} to a direction the first time it is
traversed, and the candidates at each vertex are ranked in three tiers: (1) pairs already locked to
the current direction, which stay whole; (2) pairs not yet locked, which are locked now and stay
whole; and (3) pairs locked to the opposite direction, which are split. Tier 1 is preferred, then
tier 2, and tier 3 only when nothing else is available. Within tier 2 the smallest $D_{ij}$ is
taken first, since a pair with few copies is easiest to finish in one colour; within tiers 1 and 3
the most recently locked pair is taken first. Because colouring
```
SUGGESTED:
```latex
\emph{same-direction heuristic}. A pair becomes \emph{locked} to a direction the first time it is
traversed. Once traversed against that direction, it is no longer locked and becomes \emph{split}:
it then has copies in both halves. The candidates at each vertex are ranked in four tiers: pairs
(1) locked to the current direction, (2) not yet locked, (3) already split, and (4) locked to the
opposite direction. A tier is used only when every tier above it is empty, and only a tier-4 pick
splits a pair. We rank split pairs below the pairs not yet locked: ranked above them, a split pair
would be the only candidate above the free pairs at the vertex it is locked to and the first at the
other, so the traversal would go back and forth between its two vertices until its copies were used
up. Within tier 2 the smallest $D_{ij}$ is taken first, since a pair with few copies is easiest to
finish in one colour; within tiers 1 and 4 the most recently locked pair is taken first, and within
tier 3 the most recently split one. Because colouring
```

**Algorithm 1, lines 731-737.**

CURRENT:
```latex
    \State choose an unused edge $(u,x)$, of pair $(i,j)$, by tier:
    \Statex \hskip\algorithmicindent\hskip\algorithmicindent
           (1) $\textit{lock}[i,j]=\textit{dir}$, (2) $\textit{lock}[i,j]=\textbf{none}$, (3) otherwise;
    \Statex \hskip\algorithmicindent\hskip\algorithmicindent
           tier 2: smallest $D_{ij}$ first; tiers 1, 3: latest locked first
    \State \textbf{if} $\textit{lock}[i,j]=\textbf{none}$ \textbf{then}
           $\textit{lock}[i,j] \leftarrow \textit{dir}$
```
SUGGESTED:
```latex
    \State choose an unused edge $(u,x)$, of pair $(i,j)$, by tier:
    \Statex \hskip\algorithmicindent\hskip\algorithmicindent
           (1) $\textit{lock}[i,j]=\textit{dir}$, latest locked first;
    \Statex \hskip\algorithmicindent\hskip\algorithmicindent
           (2) $\textit{lock}[i,j]=\textbf{none}$, smallest $D_{ij}$ first;
    \Statex \hskip\algorithmicindent\hskip\algorithmicindent
           (3) $\textit{lock}[i,j]=\textbf{split}$, latest split first;
    \Statex \hskip\algorithmicindent\hskip\algorithmicindent
           (4) otherwise, latest locked first
    \State \textbf{if} $\textit{lock}[i,j]=\textbf{none}$ \textbf{then}
           $\textit{lock}[i,j] \leftarrow \textit{dir}$
    \State \textbf{else if} $\textit{lock}[i,j]\neq\textit{dir}$ \textbf{then}
           $\textit{lock}[i,j] \leftarrow \textbf{split}$
```

**The cost sentence, lines 746-747.**

CURRENT:
```latex
Each pair enters a vertex's tier list at most once per side, and stale entries are discarded when
popped, so tier selection costs $O(1)$ amortised rather than a scan of the vertex's neighbours.
```
SUGGESTED:
```latex
A pair enters each tier list at most once: the tier-2 lists of both its vertices at the start,
the tier-1 list of the vertex it leaves and the tier-4 list of the other when it is locked, and
the tier-3 lists of both when it is split. An entry whose pair has no copies left, or has left
that tier, is discarded when popped, so tier selection costs $O(1)$ amortised rather than a scan
of the vertex's neighbours.
```

**Line 717.** It is loose already under three tiers, since line 712 says tier-3 pairs "are
split". So this edit is worth making even if the rule is kept.

CURRENT: `regularity: the heuristic changes which half a pair lands in, never the degree balance`

SUGGESTED: `regularity: the heuristic changes which half each copy of a pair lands in, never the degree balance`

None of this has been compiled. The usual `/check-paper` follows approval.

**For `b_last`** the tiers would read:
1. locked to the current direction and not split;
2. not yet locked;
3. locked to the opposite direction, whole or split at this vertex, latest locked first;
4. split at the other vertex, with their lock the current direction, latest split first.

That version needs its own drafting and review round.

### Everything else that would change

**The paper's numbers.**
- `tab:eulertab`: every N cell of its Euler rows and every WFA cycle cell (section 1).
- `tab:evr`, and the Euler rows of `tab:tm_comparison` and `tab:moe`.
- `fig:euler`, `fig:winner`, `fig:3d` and `fig:sens`.
- The numbers quoted from these in §6.3-6.6 and §7, and §6.1's three-run spread.
- Every Euler number in the paper was measured with the three-tier rule.

**The code.** These pass `split_method="heuristic"` explicitly.
- Behind the paper's data:
  - `agent_scripts/run_variance_campaign.ps1`, `run_sweep3_again.ps1` (sweep 3) and
    `run_euler_d4_detached.ps1` (`euler_exp/eng_*_dense_d4`);
  - `tm_workload_experiment.py`, `moe_trace_experiment.py` and `moe_workload_experiment.py`.
- Also: `check_support_and_N.py`, `run_euler_static_detached.ps1`,
  `run_euler_strategies_detached.ps1`, `verify_strategy_runs.py` and `skew_campaign.py`.
- The three figure scripts, whose output would not change: `fig_euler_heatmap_paper.py`,
  `fig_euler_bipartite_paper.py` and `fig_splitting_paper.py`.

**The changelog.**
- R9's "+5% by depth 2" comes from matrices 0 and 1 and would still round to +5%. The
  100-matrix mean is 5.29% under three tiers and 4.85% under four.
- R12 rests on the campaigns' split times (section 4).
- O5's heuristic ablation also ran three tiers.

## 7. Noticed on the way, not caused by this work

- **The split's cost bound.** Lines 766-767 give the split's cost as O(nS). Each split
  first sorts every vertex's tier-2 list, which takes O(nnz log n). The kernel's own comment
  says O(nS + nnz log n). On this workload nS dominates: about 2.1 million against 0.3
  million.
- **One concept, two names.** The `fig:eulergraph` caption (line 754) calls a pair in both
  halves "divided", while the text (line 712) and the figure's own legend say "split".

## 8. Files

All files are in `euler_exp/ruined_tier/`. The scripts are in `agent_scripts/`. Parts 2 and
4 ran in the same `.venv` as parts 1 and 3, whose environment files record the versions.

| file | what | made by |
|---|---|---|
| `paired_results.csv`, `paired_splits.csv`, `environment.json`, `paired.log` | part 1 | `euler_ruined_tier_compare.py 105` at commit `ff5a7e2` (later commits free each result, so they reproduce part 3) |
| `paired_results_rep2.csv`, `paired_splits_rep2.csv`, `environment_rep2.json`, `paired_rep2.log` | part 3 (paired run 2) | `euler_ruined_tier_compare.py 105 _rep2` |
| `eng_{engine}_dense_d4_{heuristic,heuristic_ruined}/`, `campaign_*.log`, `progress.log` | part 2 | `run_euler_ruined_tier_detached.ps1` |
| `ablation_splits.csv`, `ablation_N.csv`, `ablation.log` | part 4 | `ablate_ruined_tier.py 105 16 1`, via `run_ablation_after_ruined_tier.ps1` |
| `summary.csv`, `splits_summary.csv`, `summary_rep2.csv`, `splits_summary_rep2.csv` | paired summaries | `analyze_ruined_tier.py "" 5`, `analyze_ruined_tier.py _rep2 5` |
| `campaigns_summary.csv` | campaigns against August and against the freed run | `analyze_ruined_tier_campaigns.py` |
| `ablation_summary.csv`, `ablation_N_summary.csv` | ablation summaries | `analyze_ablation.py` |
| `drift_august.csv`, `drift_campaigns.csv`, `drift_paired.csv`, `drift_paired_blocks.csv`, `split_vs_hungarian.csv` | drift, and the split per engine | `analyze_timing_drift.py` |
| `memtrace_overnight.csv`, `memtrace_summary.csv` | memory, MiB, sampled every 120 s | the loop in `sample_memory.ps1`; `summarise_memtrace.py` |
| `prediction_before_rep2.txt` | the predictions for part 3; copied here at 06:16 from the file saved at 04:34:49 | |
| this report's tables | | `ruined_tier_report_tables.py` |

This folder is experimental, not one of the paper's canonical campaigns. The paper's
Euler numbers still come from `run/variance_2026/sweep{1,2,3}/euler/`.
