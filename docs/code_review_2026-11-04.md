# Code review, 2026-11-02 to 11-04

Six reviewers (Euler code, engines, runner, tests, paper scripts, repository hygiene) produced
96 findings. Each Euler, engine and test finding was checked by two independent skeptics trying to
refute it, and each runner finding by one. The paper-script and hygiene findings, and about half the
test findings, were not checked by an agent (the usage limit stopped it); the ones in A below I
checked against the code myself. Full data: the workflow output in the session folder.

Nothing below has been fixed yet.

## A. Findings that change what the paper's times mean (your decision)

1. **Every timed region includes building the output as dense matrices** (confirmed 2/2).
   `bvn_decomposition`, `decompose_radix` and the Euler leaves (`_leaf_bvn`) stop the clock only
   after each permutation has been turned into a dense n x n float64 matrix (512 KB at n=256).
   That cost grows with N, not with the matching work. Measured on one paper-config matrix:

   | Setup | Matching kernel | Building matrices |
   |---|---|---|
   | GW Static, Euler depth 1 leaf (N=3588) | 180 ms | 373 ms |
   | WFA, Euler depth 1 leaf (N=5090) | 433 ms | 541 ms |
   | GW Dynamic, Euler depth 1 leaf (N=730) | 502 ms | 71 ms |
   | GW Static, BvN baseline | 0.51 s | 1.15-1.51 s total |

   So for GW Static and WFA about half the reported time is memory allocation. It narrows GW
   Static's advantage and inflates every engine with many configurations. It is also the likely
   cause of the R12 held-memory effect (one GW Static leaf materialises about 1.8 GB).
   *Options:* (a) keep it, and say in the paper that $\Tc$ includes producing the configurations as
   matrices; (b) time the kernel only (return the matchings, build matrices outside the timer) and
   rerun every campaign: Radix about 11.5 h, Euler about 2.5 h, plus the workload campaigns.

2. **The Euler split is timed as a real multi-threaded run** (confirmed 2/2). Every level after the
   first runs in a thread pool, one thread per sub-matrix, and the kernels release the GIL. The
   leaves, by contrast, are timed one after another and the maximum taken. The code comments
   (`runner.py:191`, `stats.py:20`, `analysis.py:94`) and the paper (line ~1394, "timed alone ...
   not a measured parallel run") call the split sequential. Depth 4: threaded 47 ms, sequential
   sum 82 ms, sum of per-level maxima 38 ms (reviewer's scratch run).
   *Options:* (a) keep the threads and correct the wording; (b) time each split alone and charge the
   sum of per-level maxima, the same model as the leaves (needs the Euler rerun).

3. **Held memory is still present in two places** (confirmed). Inside `decompose_radix`, every
   earlier plane's dense components stay alive while later planes are timed (350-620 MB); the runner
   keeps the last Radix result alive while it times the Euler depths (only matters when one run
   does both). The Radix campaigns in the paper were not rerun with memory freed.

4. **Euler and Radix now come from different campaigns.** Euler is from 2026-11-02 with memory
   freed; Radix and the separate baselines are from August with memory held. Table `tab:evr` puts
   them side by side, and the Euler table's BvN baselines (from the Euler runs) now differ from the
   Radix table's for GW Static (1037 against 1071 ms) and WFA.

5. **The scaled-Radix figure** (`fig_paper_tradeoffs.py`, Figure `scaled_tradeoff`): for each base
   it picks the (base, scale) group with the lowest mean C, and those groups hold different matrices;
   its C=S line sits at matrix 0's S, and its BvN point comes from another 10-matrix campaign.
   (Not checked by an agent; I have not verified it yet.)

6. **Smaller measurement points:** the digit-extraction timing has no warm-up, so the first ~100
   matrices (the window `tab:evr` uses) read about 25% high; the between-sweep spread statistic
   pools campaigns of different dates and counts some cells twice.

*Recommendation:* 1(b) together with 2(b) and 3, and one overnight-plus rerun of every campaign, so
the paper's times measure the algorithms under one method. If that is too much before the
deadline, 1(a) and 2(a): state precisely what is timed, and rerun nothing.

## B. Bugs with no effect on the paper's numbers (fix now)

- Radix plane count uses floating logarithms: float matrices at bases 2, 4, 16 lose their largest
  entries; integer matrices whose maximum is an exact power of the base (1000 at base 10, 243 at
  base 3) lose the top digit, sometimes the whole matrix. Paper bases on integers are exact.
- Errors are swallowed: a failing Radix plane returns an incomplete decomposition; a failing matrix
  is dropped and the run still exits 0, even when no CSV is written; Ctrl-C exits 0 and discards
  every finished matrix.
- Silent fallbacks: an unknown split method runs the plain Euler split; an unknown matching method
  or step strategy falls back silently; Euler gets no regularity or integrality check, so a float or
  non-regular input is rounded or split unevenly without an error (`--float-weights` with
  `euler_bvn` decomposes a different matrix).
- Radix `median` step strategy with Hungarian gives fractional steps (not used by the paper).
- `--max-workers` says "Radix planes" but Radix ignores it and it sizes the Euler split's threads;
  `--step-strategy` also changes the BvN baseline and Euler leaves; `max_iters` is ignored by every
  JIT engine; `*_noaug` ignore the step strategy.
- `python main.py --help` crashes on a Windows console (non-ASCII help text).
- `ExperimentConfig` defaults differ from the CLI, and its default engine `bvn` is invalid.
- Summary file: warm-up not excluded for 10 or fewer samples; 'vs lvl' chained across split methods.
- Run folders: one-second names with `exist_ok=True`, so two runs in the same second share a folder.
- `--plot-from-csv` titles come from CLI defaults, not the run; plot helpers misalign x when values
  are missing; cycle 'normalisation' divides by k instead of S in one plot.
- Generators: `tm_integer` loops forever at n=1; `moe_trace.coarsen` does not check regularity.
- The campaign launcher writes DONE even when a run fails, and records HEAD without a dirty flag.
- `add_to_radix_times` would double-charge extraction if applied to a campaign run after
  2026-10-04 (no guard).
- Dead code, unused imports, stale docstrings (Euler, engines, runner, figure scripts), and a
  misleading 'BUGFIX' comment in `euler_splitting.py`.

## C. Tests

- **CI has failed on every push since July:** `*.ini` in `.gitignore` keeps `pytest.ini` out of
  git, so CI cannot import `src`. CI also runs the visual tests, which write PNGs into the tree.
- `test_metrics_cycle_and_permutations` asserts C >= k, a wrong bound; the WFA bound C >= S cannot
  fail; `test_bvn_cycle_invariant` never checks reconstruction.
- `test_free_between_runs` checks only that results are unchanged, never that memory is freed.
- Missing tests: the paper's own Euler configuration (`heuristic` at depths 2-4) at framework level;
  the sparsity claim (heuristic sparser than plain Euler); minimum-weight Hungarian, `*_noaug`,
  median/max strategies, float Radix; a regression test pinning `generate_matrix` to the campaign
  seeds; the CLI and config; the CSV round trip and summary; `griner_avin_tm` and `moe_trace`; the
  table and data scripts (`make_paper_tables`, `variance_data`, `radix_extraction`).
- A paper-figure script imports a private helper from the test suite.

## D. Repository

- A stray worktree is committed as a gitlink: `.claude/worktrees/vibrant-zhukovsky-ed86fe`.
- `run/variance_freed_2026` cannot be added with plain `git add` (`*.csv` ignored); not committed.
- `plot_scaled_tradeoff.py` silently overwrites the paper's `scaled_tradeoff.pdf` with a different
  figure.
- Old launchers no longer reproduce their campaigns now that freeing is the default.
- README: wrong engine name, missing flags and split methods, Python 3.11 instead of 3.13, wrong
  directory tree, CI claim. AGENT_GUIDE.md and the HANDOFF files contradict the current repository.
- `requirements.txt`: unused networkx, nothing pinned for the timed environment, pytest mixed in.
- `agent_scripts/`: about 12 live scripts among about 70 stale or one-off ones, inconsistent
  prefixes. Root clutter: `benchmark_heuristic.py` (compares sequential Radix with parallel Euler),
  `view_matrix.py` (stale constants), untracked logs.
- CSV column names are inconsistent and carry no units. `docs/claude_memory` has drifted from the
  memory it copies.
