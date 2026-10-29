# Agent Guide — Writing and Critiquing the Paper

You are helping Ofek Cohen with an MSc thesis project (Ben-Gurion University,
advisor Prof. Chen Avin) and the conference paper that comes out of it. This file
tells you what the work is, where everything lives, how the author wants to be
worked with, and what to read before you offer an opinion.

Read this file first. Read `HANDOFF.md` second if you are going to touch code.

---

## 0. Thirty-second orientation

The project decomposes an **S-regular integer demand matrix** into weighted
permutation matrices, `D = Σᵢ λᵢ Pᵢ`, for scheduling a reconfigurable optical
circuit switch. Each `Pᵢ` is one switch configuration; `λᵢ` is how long it is held.

The paper's central claim is that this is a **three-way trade-off**, not a
one-dimensional optimisation:

| metric | meaning | physical cost |
|---|---|---|
| `N` | number of permutations | one reconfiguration each, `T_config` |
| `C(𝒟) = Σ λᵢ` | cycle length | transmission slots, `T_unit` each; optimum is `C = S` |
| `T_compute` | time to compute the schedule | paid once, up front |

priced together by the **demand completion time**

```
DCT = T_compute + N·T_config + C·T_unit
```

Everything in the paper exists to characterise that trade-off. If a proposed
change does not help the reader understand the trade-off, it does not belong.

**Three frameworks** are compared: classical BvN, **Radix(B,c)** (digit planes in
base `B` with a scale `c`), and **Euler splitting** (recursive Euler-tour
2-colouring into `2^d` regular leaves that can be decomposed in parallel).

**Four matching engines**: Hungarian (exact LSAP, `scipy.optimize.linear_sum_assignment`),
GW Dynamic (`heavy`), GW Static (`heavy_static`), WFA (`wfa`).

**Three step strategies**: `λᵢ` = min / median / max of the matched entries.

---

## 1. The single most important thing

**There are many stale paper files. Only one is live.**

| file | status |
|---|---|
| `../paper/main_paper.tex` | **THE PAPER.** This is the only .tex to edit. |
| `../paper/test.tex` → `test.pdf` | The 42-page thesis. **Read-only source material** — content gets lifted from here into the paper, but never edit it. |
| `../meeting_doc.pdf` | The chapter layout the author agreed with Chen Avin. The paper's structure must follow it. |
| `../paper/file.tex` | **Corrupt. Ignore.** |
| `../paper/conference_draft.tex`, `full_summary.tex`, `main_paper_working.tex`, `euler_section_draft.tex` | Superseded. Ignore. |
| `../paper/HANDOFF.md` | **Stale** — it still calls `file.tex` the main paper. Do not trust its file table. |
| `HANDOFF.md` (this repo) | Mostly accurate for *code and data*; its paper row is stale. |

If you are ever unsure which file is live, it is `main_paper.tex`. Check the
modification date before believing anything else.

---

## 2. Paper structure and what is written

`../paper/main_paper.tex` — two-column `\documentclass[sigconf,nonacm]{acmart}`,
target ~12 pages, venue-neutral (no venue chosen yet).

| § | title | status |
|---|---|---|
| 1 | Introduction | **DO NOT WRITE.** Reserved to be written last, with Chen Avin. |
| 2 | Background and Related Work | not started |
| 3 | Problem Definition and Model (3.1–3.6) | **complete and approved** |
| 4 | Solution Framework (4.1–4.4) | **complete and approved** |
| 5 | Experimental Evaluation (5.1–5.7) | **complete**; 5.8 (non-regular MoE traffic) open |
| 6 | DCT Analysis | not started; the break-even formula was deferred here from §3.4 |
| 7 | Conclusions and Future Work | not started |
| — | Abstract | write last, after the Introduction |
| — | Bibliography | ~19 `\bibitem`s, inline, maintained **as we write** — not deferred to the end |

Custom macros in the preamble — use them, do not spell the names out:
`\Tc` (`T_compute`), `\Tcfg` (`T_config`), `\Tu` (`T_unit`), `\Dc` (`𝒟`),
`\TODO{...}` (renders red).

Compile from `../paper/`:

```bash
pdflatex -interaction=nonstopmode main_paper.tex
```

Run it **twice** — cross-references and float placement do not settle on one pass.

### Section drafts

`../paper/drafts/chapter3.md` and `chapter4.md` are the working drafts. They record
not only what was written but **what was rejected and why**. Read them before
re-litigating a wording choice — several phrasings in the paper are the third or
fourth attempt and the earlier ones were explicitly turned down by the author.

---

## 3. Files to read when critiquing

Match what you are being asked about to the column on the right. Do not read the
whole repo; it contains years of superseded runs.

### Critiquing the paper's argument or writing

1. `../paper/main_paper.tex` — the text itself
2. `../paper/drafts/chapter3.md`, `chapter4.md` — decisions and rejected alternatives
3. `../meeting_doc.pdf` — the agreed chapter layout and the story beats
4. `../paper/test.pdf` — the thesis, for anything the paper compresses
5. `../article_summaries.md` — Harvest (SIGCOMM'26) and Birkhoff-MoE (HotOptics'26),
   the two closest related works, both from Purdue/StygianNet

### Critiquing an algorithm or a claimed property

| topic | file |
|---|---|
| classical BvN, engines, step strategies | `src/algorithms/bvn.py` |
| Radix / digit planes | `src/algorithms/radix_decomposition.py` |
| Euler splitting, three-tier heuristic | `src/algorithms/euler_splitting.py` |
| WFA | `src/algorithms/wfa.py` |
| scanning-engine kernels | `src/algorithms/sorted_array_matching.py` |
| experiment driver | `src/runner.py`, `src/cli.py`, `src/config.py` |
| generated matrices | `src/utils/matrix_generator.py` |
| Griner–Avin traffic model | `src/utils/griner_avin_tm.py` |
| MoE trace loading | `src/utils/moe_trace.py` |
| tests | `tests/` (`tests/test_euler_splitting.py` is the most load-bearing) |

Note: `euler_splitting.py` is Numba-JIT'd. The `__pycache__` `.nbc`/`.nbi` files
are compiled artefacts — never read or edit them.

### Critiquing a number, table, or figure

Every number in the paper traces back to a CSV. Find it before you dispute it.

| result | data |
|---|---|
| **canonical dense BvN + Radix** | `run/rerun_2026/{max,heavy,heavy_static,wfa}/*/results_stats.csv` (1000 samples, seed 42) |
| Radix with per-base scaling | `run/radix_scale/scale_results.csv` |
| step strategies (Radix) | `run/step_strategy_2026/` |
| **Euler d1–d4** | `euler_exp/eng_{maximum,heavy,heavy_static,wfa}_dense_d4/*/results_stats.csv` |
| Euler median/max strategies | `euler_exp/strat_{median,max}_{engine}/` |
| Griner–Avin TM workload | `run/tm_workload/tm_summary.csv` |
| time-to-first-permutation | `run/ttfp/ttfp_results.csv` |
| MoE traces (raw) | `paper_traces/*.csv` |

**Superseded — do not cite:** `final_runs/20260522_*` (the Hungarian run there had
16 impossible `C < S` matrices from a bug fixed in `rerun_2026`), and the many
bare-timestamp directories under `run/` and `runs/`.

Figure and table generators live in `agent_scripts/`. The ones that produce
current paper artefacts:

| output | script |
|---|---|
| `scaled_tradeoff.pdf`, `euler_tradeoff_paper.pdf` | `agent_scripts/fig_paper_tradeoffs.py` |
| `euler_bipartite_paper.pdf` | `agent_scripts/fig_euler_bipartite_paper.py` |
| `euler_split_heatmaps.pdf` | `agent_scripts/fig_euler_heatmap_paper.py` |
| DCT winner maps / surfaces | `agent_scripts/dct_parameter_space.py`, `winner_map_tcalc_sensitivity.py` |
| scale table | `agent_scripts/make_scale_table.py` |
| TM workload figures | `agent_scripts/tm_paper_figures.py` |

`agent_scripts/` also holds many one-off diagnostics (`diagnose_*`, `prove_*`,
`verify_*`, `task*_*`). They were written to answer a single question and are not
maintained. Read them for evidence of what was already checked; do not assume they
still run.

---

## 4. Running things

Python is **not** on `PATH` under the plain name. Use the venv interpreter:

```bash
./.venv/Scripts/python.exe -m src.cli --help
```

Main flags: `--size --k --engine --samples --output --max-weight --radix-bases
--euler-depths --euler-colors --euler-split-method --euler-leaf-engine
--step-strategy --random-seed --max-workers --plot-from-csv`.

**Conventions that must not be broken — changing them invalidates every existing
comparison:**

- dense workload: `n=256, k=256, max_weight=64, seed 42+index`; sparse: `k=16`
- the first **5 matrices of every campaign are discarded** as Numba JIT warm-up.
  Analysis code does this with `.iloc[5:]` — if you write new analysis, do the same.
- runtimes are reported in **milliseconds**; the CSVs store seconds in some columns
  and ms in others. Check the column name before you multiply.

Long campaigns are launched detached via the `agent_scripts/run_*_detached.ps1`
scripts, which drop a `*_DONE.marker` file in `euler_exp/` when finished.

---

## 5. How the author wants to be worked with

These are not preferences to be inferred — they were stated explicitly and
enforced repeatedly. Violating them wastes the author's time.

**Write section by section, never the whole thing at once.** Produce one
subsection, show it, wait for a reaction. A dump of five sections will be rejected
wholesale. The one exception is when the author explicitly says "generate the
whole chapter".

**Never write the Introduction or the Abstract.** They are reserved for the author
and Chen Avin, last.

**Every figure in the paper must be a vector PDF.** This is stated as a standard.
Generate `.pdf` (a matching `.png` alongside is fine for preview). Never
`\includegraphics` a raster.

**Do not over-claim.** The author has caught and corrected several statements that
sounded good and were false. Real examples:

- "no decomposition minimises more than one metric" — false, domination exists
- "the Hungarian algorithm computes a maximum-weight perfect matching" — it solves
  the min-cost linear sum assignment problem
- "the geometric sweep is what makes WFA cheap enough for hardware" — an
  unsupported causal claim
- calling the Euler halving property "our Proposition 2" — it is classical
  (Gabow 1976; Cole–Hopcroft 1982)
- inventing the term "exact assignment" and using it as if it were standard

If you cannot point to a citation or a CSV, do not assert it. Say it is unverified.

**Do not introduce new terminology.** If a quantity was defined in §3, use that
name. "Sum of the hold times" was rejected because `C(𝒟)` was already defined.

**Register: formal, but not jargon-laden.** Drafts have been rejected in both
directions — once as "unclear and jargon-laden", once as "too informal, the
sentences seem immature". Aim for the register of a good systems paper: plain
words, precise claims, no hype, no chatty asides.

**Do not forward-reference.** Do not write "as we will see in §5". If the reader
needs something from a later section, either move it earlier or do not lean on it.

**Numbers need interpretation.** "The decomposition space is large" and "there are
48,840 decompositions" were both rejected as saying nothing. State the number *and*
what it implies.

**When the author says a paragraph is wrong, start over rather than patching.**
§3.5 was rewritten from scratch on request; incremental edits to a paragraph the
author dislikes get rejected again.

---

## 6. Known open items

- **§5.8 non-regular traffic** — the MoE traces are in `paper_traces/` and load via
  `src/utils/moe_trace.py`. They are 8×8 (everything else in the paper is n=256) and
  **not regular**: 16–42% slack padding is required to regularise them. Any §5.8
  text must be explicit that this is a different scale, not a like-for-like
  comparison. An open design decision remains on where the padding is placed
  (currently diagonal-heavy, which biases engines toward the identity permutation).
- **§2, §6, §7** unwritten.
- **Float placement** in §5 is congested — 2 figures + 3 tables across ~2 pages.
  LaTeX emits floats in *declaration order*, so a table declared after a figure
  cannot appear before it. The float parameters in the preamble were tuned to stop
  figure-only pages; do not revert them casually.

---

## 7. Before you give a critique

Ask yourself, in order:

1. Have I read the current `main_paper.tex`, or am I reacting to a stale draft?
2. Does my objection already appear in `drafts/chapter3.md` / `chapter4.md` as a
   rejected alternative?
3. Am I disputing a number? Then which CSV disagrees with it?
4. Does my suggestion serve the three-way trade-off claim, or is it scope creep on
   a paper that is already at its page budget?
5. Am I asserting something I cannot cite or measure?

A critique that survives all five is worth the author's time. One that does not
will be corrected, and the correction costs more than the critique was worth.
