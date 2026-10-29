A modular Python framework for decomposing K-regular / doubly-stochastic traffic matrices into weighted permutation matrices for crossbar switch scheduling. The suite benchmarks several bipartite matching engines and two decomposition strategies (classical BvN and Radix / digit-plane) against a three-way tradeoff between Permutation Count, Cycle Length, and Algorithmic Runtime.

## Start here (new Claude session)

This repository is the code half of Ofek Cohen's MSc thesis (Ben-Gurion University, advisor
Prof. Chen Avin), "Scheduling Reconfigurable Circuit Switches is a Three-Way Trade-off". The
paper itself lives in a separate repository, `chenavin/Ofek-Cohen-BvN-Thesis` (linked to
Overleaf), together with the strict rules for working on it (`.claude/CLAUDE.md` there).

Read, in this order:

1. `AGENT_GUIDE.md`: what the work is, which files are live and which are stale, how the
   author wants to be worked with.
2. `docs/claude_memory/`: the knowledge accumulated across sessions. `MEMORY.md` is the
   index; `campaign-data-provenance.md` says which campaign backs which number in the paper,
   `paper-writing-principles.md` and `paper-notation-conventions.md` how the paper is written.
3. `HANDOFF_laptop.md`: the full machine setup (folders, venv, LaTeX, hooks, open tasks) as
   of 2026-10-29.
4. `HANDOFF.md`: code and data details (its paper-file table is stale).

Layout:

- `src/algorithms/`: `bvn.py`, `radix_decomposition.py`, `euler_splitting.py`, `wfa.py`,
  `sorted_array_matching.py` (GW Dynamic / GW Static).
- `agent_scripts/`: every campaign runner and every script that draws a paper figure or table
  (for example `fig_splitting_paper.py` for Figure 3, `make_paper_tables.py` for the tables).
  No path is absolute: scripts take their folders from `agent_scripts/_paths.py` (`REPO`, and
  `PAPER`/`PLOTS` = a `paper/` folder next to this repository, or `BVN_PAPER_DIR` if set), and
  the PowerShell launchers from `$PSScriptRoot`. Keep it that way.
- `run/<campaign>/`, `euler_exp/`: the campaign results the paper quotes (named folders only;
  the old timestamped sweeps `run/2026MMDD_*`, `summary/` and `step_exp/` are not in git).
- `tests/`: `pytest -q -k "not visual"`.

Environment: Python 3.13, `python -m venv .venv`, `.venv\Scripts\pip install -r
requirements.txt`. The paper's timings were measured on the author's desktop (24 logical
cores, Windows 11, NumPy 2.3.5, SciPy 1.17.0, Numba 0.63.1); do not mix in timings from
another machine.

Never invent a number: every figure quoted in the paper must trace to a CSV in `run/` or
`euler_exp/` and the script that produced it.
