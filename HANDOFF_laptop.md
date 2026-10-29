# Handoff: thesis setup on the desktop (Windows), for the laptop session

Written 2026-10-29 by the Claude session on the desktop, for Ofek's Claude session on the
laptop, which cannot see this machine. Everything below was read from the machine on that
date; nothing is from memory. Paths are the desktop's.

**Read section 7 (risks) first.** Two things exist only on this desktop and nowhere online.

---

## 1. Folder layout

Root: `C:\Users\ofekc\Desktop\Msc\Thesis\`

| Path | What it is | Git? |
|---|---|---|
| `CLAUDE.md` | 5-line pointer: "the rules live in `paper_overleaf/.claude/CLAUDE.md`" (imports it) | no |
| `bvn_project/` | the code: framework, campaigns, figure/table scripts, results (~976 MB) | **yes** |
| `paper/` | **the paper's working copy** (~232 MB incl. old drafts, logs, archives) | **yes** |
| `paper_overleaf/` | git *worktree* of the `paper` repo, branch `overleaf-main`; mirror of what Overleaf compiles, plus `.claude/` (rules, change log, hooks, skills, agents) | worktree of `paper` |
| `articles/` | PDFs of related work (Griner-Avin Computer Networks 2026, Birkhoff-MoE HotOptics 26, Harvest SIGCOMM 26) | no |
| `article_summaries.md` | notes on Harvest and Birkhoff-MoE | no |
| `final_paper/` | July snapshot: `test.tex/pdf` (42-page thesis draft, read-only source material), `meeting_doc.pdf`, `figure_board.pdf`, `plots_pdf/` | no |
| `meeting_doc.pdf`, `figure_board.pdf` | chapter layout agreed with Chen (July) | no |
| `Thesis/paper/plots/` | empty leftover folder | no |
| `test/test.py` | scratch | no |

### Git repositories, remotes, branches

**`bvn_project`** (code)
- remote `origin` = `https://github.com/COfek/bvn_project.git`
- current branch `euler_splitting`, **5 commits ahead of `origin/euler_splitting`** (not pushed;
  they are included in the `handoff/laptop-setup` branch this file is on):
  `2757433` overnight launcher to remeasure the digit extraction; `2c068f1` charge the digit
  extraction in Radix's computation time; `2de512a` ablation launcher same-direction vs
  alternating; `5bbc738` sensitivity map at three budgets; `db8a406` every setup on the Euler
  campaign's 100 matrices.
- local branches: `euler_splitting`*, `master`, `Radix`, `sequential_exp`, `stable_test`,
  `claude/vibrant-zhukovsky-ed86fe` (a Claude worktree at `.claude/worktrees/vibrant-zhukovsky-ed86fe`)
- remote branches: `master` (HEAD), `Radix`, `comprehensive-exp`, `euler_k_way_split`,
  `euler_splitting`, `sequential_exp`, `split_tree`, `stable_test`

**`paper`** (the paper)
- **no `origin` remote.** Only remote: `overleaf` = `https://github.com/chenavin/Ofek-Cohen-BvN-Thesis.git`
  (Chen's repo, linked to the Overleaf project through Overleaf's GitHub sync)
- branch `master` (checked out in `paper/`): the full edit history, **166 commits, local only**.
  Last: `b49049e` 2026-10-29 "pseudocode fixes".
- branch `overleaf-main` (checked out in `paper_overleaf/`), tracks `overleaf/main`: what
  Overleaf compiles (`main_paper.tex`, 5 `tab_*`/`*_tab_*` tables, `plots_pdf/` 8 PDFs) plus
  `.claude/`. In sync with `overleaf/main` at `34987ac`.
- remote also has `overleaf-2026-10-29-0633`: a side branch Overleaf created on 10-29 holding
  one stray space; already merged into `main` (ours-strategy). Harmless; can be deleted with
  Chen's OK.

### How a paper change flows

1. Edit `paper/main_paper_with_comments.tex` (the working file).
2. `cp main_paper_with_comments.tex main_paper.tex` (kept byte-identical).
3. Run the checks (section 5), commit in `paper/` (`master`).
4. Log the change in `paper_overleaf/.claude/changelog.md`, commit in `paper_overleaf/`.
5. `bash paper/push_overleaf.sh "message"` copies the paper, tables and figures into
   `paper_overleaf/`, commits on `overleaf-main` and pushes to `overleaf/main`.
6. In Overleaf: Menu > GitHub > Pull GitHub changes.

If anyone edited in Overleaf without pushing to GitHub, Overleaf's pull fails and it parks its
edits on a side branch (`overleaf-YYYY-MM-DD-HHMM`). Fetch that branch, merge it into
`overleaf-main` (bring real edits into the working file first), push, then pull again.

**On 2026-10-29: `main_paper_with_comments.tex` and `main_paper.tex` are byte-identical**
(checked with `cmp`).

---

## 2. Claude hooks (`C:\Users\ofekc\.claude\settings.json`, user scope)

The file holds only `enableWorkflows`, `skipWorkflowUsageWarning` and `hooks`. No secrets.

```json
"hooks": {
  "PreToolUse": [
    { "matcher": "Edit|Write|MultiEdit",
      "hooks": [ { "type": "command",
        "command": "\"C:/Users/ofekc/Desktop/Msc/Thesis/bvn_project/.venv/Scripts/python.exe\" \"C:/Users/ofekc/Desktop/Msc/Thesis/paper_overleaf/.claude/hooks/protect_comments.py\"" } ] }
  ],
  "PostToolUse": [
    { "matcher": "Bash|PowerShell",
      "hooks": [ { "type": "command",
        "command": "\"C:/Users/ofekc/Desktop/Msc/Thesis/bvn_project/.venv/Scripts/python.exe\" \"C:/Users/ofekc/Desktop/Msc/Thesis/paper_overleaf/.claude/hooks/bash_marker_guard.py\"" } ] }
  ]
}
```

- `protect_comments.py` refuses any Edit/Write that removes a `\chen{...}`, `\TODO{...}`,
  `\ready` or `%` comment from a `.tex` under the Thesis folder. Not overridable from a
  session: Ofek removes markers himself.
- `bash_marker_guard.py` does the same after shell commands (it silently reverts a `sed` that
  deletes a marker; note it also treats anything starting with `\chen` as protected).
- Both live in `paper_overleaf/.claude/hooks/` (with `markers.py`). On the laptop, adjust the
  two absolute paths (venv python, hook scripts).

---

## 3. Python environment

- venv: `C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project\.venv` (git-ignored)
- created with: `C:\Users\ofekc\AppData\Local\Programs\Python\Python313\python.exe -m venv ...\bvn_project\.venv`
- Python **3.13.0**
- `requirements.txt`: `numpy>=2.0, scipy>=1.17, matplotlib>=3.10, networkx>=3.6, rich>=14.0, pytest>=9.0, numba>=0.63, pandas>=2.0`
- installed (the versions the paper quotes in section 6.1): numpy 2.3.5, scipy 1.17.0, numba 0.63.1,
  matplotlib 3.10.8, pandas 3.0.0, networkx 3.6.1, pytest 9.0.2
- recreate on the laptop: `python -m venv .venv`, then `.venv\Scripts\pip install -r requirements.txt`
  (pin the versions above if timings must match the paper).
- Measurements in the paper were taken on this desktop: 24 logical cores, AMD64, Windows 11.
  **Do not rerun timing campaigns on the laptop and mix the numbers.**

---

## 4. LaTeX

- **MiKTeX 25.12**, `pdflatex` = MiKTeX-pdfTeX 4.23, at
  `C:\Users\ofekc\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`
- class `acmart` v2.19 (2026/06/27), from MiKTeX's package tree
- the bibliography is inline (`thebibliography` at the end of the .tex); no BibTeX run needed
- `pdftotext` 4.00 comes from Git for Windows (`/mingw64/bin`), used by the checker
- Overleaf compiles the same `main_paper.tex` with its own TeX Live; both build cleanly.

---

## 5. Checks before every commit

`paper_overleaf/.claude/skills/check-paper/scripts/check_paper.py <file.tex>`, run with the
venv python. It:

- copies the .tex and every `\input`/`\includegraphics` file into a fresh temp folder
  (`tempfile.mkdtemp(prefix="check_paper_")`; a stale `.aux` once caused wrong numbers),
- runs `pdflatex -interaction=nonstopmode -output-directory=<tmp> <file>` **3 times**,
- fails on any `!` error or undefined reference, checks for a phantom "Temporary page!" with
  `pdftotext`,
- then checks protected markers against `git show HEAD:<file>`, citations (cited vs defined),
  references, forward references (WARN), em dashes, and file sync.

Expected result today: everything PASS, one WARN (forward refs, open item O2), one INFO
(5 bib entries defined but uncited, waiting for Chapter 2). Since 2026-10-24 the paper
compiles with **zero** errors.

---

## 6. Uncommitted work and open tasks

### `bvn_project` (branch `euler_splitting`), not committed
- **staged, not committed (79 paths):** `AGENT_GUIDE.md`, `.claude/agents/*.md` (7 subagent
  definitions), `src/utils/griner_avin_tm.py`, `src/utils/moe_trace.py`, changes to
  `src/algorithms/bvn.py`, `tests/test_euler_splitting.py`, 5 `agent_scripts/*.py`, and
  campaign summaries under `euler_exp/` (35) and `run/` (33).
- modified, unstaged: `run/radix_extraction/{moe,radix_scale,tm,variance}.csv` (overnight rerun
  of the digit extraction, 2026-10-05).
- untracked: `run/ablation_heuristic_2026/` (heuristic ablation results, not yet analysed,
  open item O5), `run/radix_extraction/RERUN_DONE.marker`.
- **`agent_scripts/` is git-ignored** (`.gitignore` line 36); only 26 of 57 scripts are
  force-tracked. The 31 untracked ones include figure and table generators the paper depends
  on: `fig_splitting_paper.py` (Figure 3), `fig_euler_heatmap_paper.py`,
  `fig_euler_bipartite_paper.py`, `plot_scaled_tradeoff.py`, `make_scale_table.py`,
  `radix_scale_campaign.py`, `tm_workload_experiment.py`, `moe_raw_campaign.py`,
  `moe_raw_figures.py`, `skew_campaign.py`, `skew_figures.py`, and `run_extraction_rerun.ps1`.
  **These exist only on this desktop.**

### `paper` / `paper_overleaf`
- `paper/`: clean.
- `paper_overleaf/`: one untracked file, `README.md` (a README for Chen's repo, drafted
  2026-10-12, **waiting for Ofek's OK**; deliberately kept out of every push since).

### Open items (`paper_overleaf/.claude/changelog.md`, table at the top)
- O2 forward references (15) - keep or remove one by one
- O4 section 7.3 quotes T_c/16, T_c/128, T_c=0 that Figure 8 no longer shows
- O5 heuristic ablation ran, not analysed
- O6 make `paper_overleaf` the single working copy?
- O8 six bib entries uncited (wait for Chapter 2)
- O9 Chen's point "After perfect matching put" - ask Chen
- O11 Chapter 8 must discuss scaling rational demand to integers
- O12 Rule 6 check of the two Griner-Avin quotes (verified 2026-10-12: section 8 and section 9 of
  their paper; can be closed)
- O13 Chen's `preview` class option (probably meant `review`) - ask Chen
- O14 Chronos does not mention bit-planes or 1.2x - find the real source before Chapter 2
- O15 Chen dislikes `\paragraph{}` headings; 5.1.1 and 5.1.2 done, rest of Chapters 3-7 pending
- O16 Figure 3: one column (current) or full width - Ofek and Chen decide
- E10 merge Propositions 3.1 and 3.2 - deferred by Chen

### Other unfinished threads from the 2026-10 sessions
- **Citation audit (2026-10-12)**, reported, not yet applied:
  - section 3.4 should credit Griner and Avin for the *name* DCT (their Definition 1, "matrix
    completion time", is our per-matrix DCT minus T_compute);
  - section 4.1: Kuhn/Munkres give O(n^4); the O(n^3) needs Edmonds-Karp / Tomizawa / Jonker-Volgenant (crouse2016);
  - section 5.2 table cites `delgado2003` (wrapped WFA) for the plain anti-diagonal WFA; the plain one
    is Tamir and Chi, IEEE TPDS 4(1):13-27, 1993;
  - missing citations: "elephant/mice of measured traffic" (section 6.1), Mixtral, DeepSeekMoE,
    MMLU, SPEED-Bench (section 6.1), "MoE routing *deliberately* uneven" (section 6.6, also wrong), the
    hardware range T_config 1 us-1 ms (section 7);
  - metadata: delgado2003 add 50(1):135-141; chronos add pp. 89-97; dufosse2016 theorem
    number still unknown (HAL blocks automated access).
- **Pseudocode check (2026-10-29)**, still open: engine-table cost column omits Augment;
  GW Static is O(nnz(D)), not O(m); Algorithm 1 restarts from any vertex while the code
  resumes Hierholzer-style (minor, probably leave).
- **Ready-to-review markers** now sit above section 3, section 5 (opening), 5.1.1, 5.1.2 and 5.2.

### Memory (Claude auto-memory on the desktop)
`C:\Users\ofekc\.claude\projects\C--Users-ofekc-Desktop-Msc-Thesis-bvn-project\memory\`:
agent-guide, campaign-data-provenance, paper-latex-location, paper-notation-conventions,
paper-working-files, paper-writing-principles, related-work-articles (+ MEMORY.md index).
Not synced anywhere. The rules that matter are in `paper_overleaf/.claude/` (in Chen's repo).

---

## 7. Risks: things that exist only on this desktop

1. **The paper's history (`paper` repo, branch `master`, 166 commits) has no remote.** Only the
   latest `main_paper.tex` reaches GitHub, through Chen's Overleaf repo. If this disk dies the
   history, the old drafts and `paper/archive/` are gone. Suggested: add a private remote
   (for example `COfek/bvn-paper`) and push `master`. Not done; Ofek to decide.
2. **31 git-ignored scripts in `bvn_project/agent_scripts/`** (section 6), including generators of
   paper figures. Suggested: un-ignore `agent_scripts/` or force-add the paper ones. Not done.
3. The staged-but-uncommitted `bvn_project` work (79 paths) and the 5 local commits. The 5
   commits ride along on this branch; the staged work does not.

To work on the laptop: clone `COfek/bvn_project` (this branch) for the code, and
`chenavin/Ofek-Cohen-BvN-Thesis` for the paper and `.claude/` rules. Read
`.claude/CLAUDE.md` (six strict rules) and `.claude/changelog.md` before editing the paper.
Note that the two `.tex` files to edit on the desktop do not exist in Chen's repo: there,
`main_paper.tex` is the only copy.

---

## 8. Verbatim: `paper/push_overleaf.sh`

```bash
#!/usr/bin/env bash
# Publish the paper to the Overleaf project through its GitHub mirror.
#
#   bash push_overleaf.sh "commit message"      # copy, commit, push
#   bash push_overleaf.sh --dry-run             # show what would change
#   bash push_overleaf.sh --integrated "msg"    # Overleaf edits merged by hand; go ahead
#
# The Overleaf repo (chenavin/Ofek-Cohen-BvN-Thesis) holds only what Overleaf
# compiles, so this repo's history is not pushed there. Instead ../paper_overleaf
# is a worktree of the `overleaf-main` branch, which tracks overleaf/main; this
# script copies into it every file Overleaf already has, plus anything
# main_paper.tex newly \input's or \includegraphics's, then commits and pushes.
#
# If Overleaf has commits we have not seen (someone edited there), the script
# stops before copying anything, so their edits are never overwritten. Bring
# them into main_paper_with_comments.tex first, then run it with --integrated.
set -euo pipefail
cd "$(dirname "$0")"
SRC=$PWD
WT=$SRC/../paper_overleaf
DRY=0; MSG=""; OK=0
[[ "${1:-}" == "--integrated" ]] && { OK=1; shift; }
[[ "${1:-}" == "--dry-run" ]] && DRY=1 || MSG="${1:-}"
[[ $DRY == 1 || -n "$MSG" ]] || { echo "usage: $0 \"message\" | --dry-run"; exit 2; }

git remote get-url overleaf >/dev/null 2>&1 || \
  git remote add overleaf https://github.com/chenavin/Ofek-Cohen-BvN-Thesis.git
git fetch -q overleaf
if [[ ! -d $WT ]]; then
  git show-ref -q refs/heads/overleaf-main || git branch -q --track overleaf-main overleaf/main
  git worktree add -q "$WT" overleaf-main
fi

cd "$WT"
if [[ -n "$(git rev-list HEAD..overleaf/main)" && $OK == 1 ]]; then
  git merge -q --ff-only overleaf/main
elif [[ -n "$(git rev-list HEAD..overleaf/main)" ]]; then
  echo "Overleaf has changes not seen here:"
  git log --format='  %h %an %ad %s' --date=short HEAD..overleaf/main
  git diff --stat HEAD overleaf/main
  echo "Merge them into main_paper_with_comments.tex, then rerun with --integrated. Nothing pushed."
  exit 1
fi

# every file Overleaf already tracks, plus whatever main_paper.tex references
{
  git ls-files
  grep -oE '\\(input|include)\{[^}]+\}' "$SRC/main_paper.tex" \
    | sed -E 's/.*\{([^}]+)\}/\1/; /\.tex$/!s/$/.tex/'
  grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$SRC/main_paper.tex" \
    | sed -E 's/.*\{([^}]+)\}/\1/'
} | sort -u | while read -r f; do
  [[ -f "$SRC/$f" ]] || continue
  if [[ $DRY == 1 ]]; then
    cmp -s "$SRC/$f" "$f" 2>/dev/null || echo "  would update $f"
  else
    mkdir -p "$(dirname "$f")"; cp "$SRC/$f" "$f"
  fi
done

[[ $DRY == 1 ]] && exit 0
git add -A
if git diff --cached --quiet; then echo "Overleaf is already up to date."; exit 0; fi
git diff --cached --stat
git commit -q -m "$MSG"
git push -q overleaf HEAD:main
echo "Pushed $(git rev-parse --short HEAD). In Overleaf: Menu > GitHub > Pull GitHub changes."
```

---

## 9. Verbatim: `paper/AGENT_TEAM_CONTEXT.md`

**Stale in places** (last changed 2026-08-29). Since then: Radix uses `K`, `k` (planes `k = 0..K-1`), not `L`, `ll`; component matrices replaced "parts"; `T_P-compute` names Eq. (13); Proposition 3.2 (`S*`) and 5.2 (Radix cycle bound) exist; strong/weak engines are defined in 5.2; Figure 3 is vertical; most `\paragraph{}` headings in 5.1 are gone. Trust `paper_overleaf/.claude/rules/notation.md` and the change log over this file.

````markdown
# Agent team context — read this before doing anything

Paper: *Scheduling Reconfigurable Circuit Switches is a Three-Way Trade-off:
Permutations, Cycle Length, and the Cost of Computing the Schedule*
Author: Ofek Cohen (MSc, Ben-Gurion University). Advisor: Prof. Chen Avin.
Format: `acmart[sigconf,nonacm]`, two-column, bibliography inline (no `.bib`).

**Working file:** `C:\Users\ofekc\Desktop\Msc\Thesis\paper\main_paper_with_comments.tex`
`main_paper.tex` is kept byte-identical by copying after each edit. Never edit
`overleaf_updated.tex` (a stale snapshot) or any `*_draft*.tex`.

---

## Hard constraints — violating any of these is a failed handoff

1. **Never write the Abstract or the Introduction.** Reserved for the author and advisor.
2. **Never invent a number.** Every numeric claim traces to a named CSV, or to a table or
   figure already in the paper. If you cannot name the source, mark it `[UNVERIFIED]` and stop.
3. **Do not preview conclusions.** A section introduces a method by what it *does*, in one
   sentence. The trade-off verdict belongs to the chapter that measures it.
4. **Do not forward-reference.** If a term is needed early, define it there.
5. **Every equation is followed by a "where X is …" clause** naming every symbol in it.
6. **First person, telling the research story:** "we consider", "we tested", "we claim".
7. **No em dashes.** Use a colon, parentheses, or a sentence split.
8. **Do not introduce a new term** when one is already defined.
9. **Figures are vector PDF**, regenerated by a script in `bvn_project/agent_scripts/`.
   Never TikZ, never raster, never hand-edited.
10. **Anything requiring Chapter 2 waits.** Background and Related Work is unwritten.

## Notation contract

- `\mathcal{C}_n` — partial permutation matrices, at most one 1 per row and column. This is
  what a **configuration** is. `\mathcal{P}_n \subset \mathcal{C}_n` — **full** permutations,
  exactly one. Never write "a configuration is a permutation matrix".
- Family index is a **subscript**: `P_i`, `P_t`, `D_k`, `\Delta_\ell`. Entry access uses
  **brackets**: `P[i,j]`, `P_i[j,k]`. `D_{ij}` and `R_{jk}` keep plain subscripts.
- Parts run `k = 1..K`; configurations run `i`; hence `P_{k,i}`.
- `\eqref` is redefined in the preamble to render `Eq. (n)`. Never write `Equation~\eqref`.
- Operators: `\msplit(D,K)`, `\radixsplit(D,B)`, `\eulersplit(D,d)`. The name order is
  *radix-split*, not *split-radix*.
- `\PD{splitting}{engine}` renders `PD(radix-split(D,2), WFA)`. Use it **only where a
  specific pair is the subject**; most of Chapter 6 holds one factor and moves the other.
- Method names go through macros: `\BvN \Hung \GWD \GWS \WFA \Scan \ScanEngine \Augment
  \EulerSplit`. Never inside a `\section{}` heading or a bibliography title.
- `\chen{...}` is an advisor comment (blue). `\ready` is a green `[ready to review]`, placed
  below the heading as `\noindent\ready\par\smallskip`, never inside `\section{}`.

## Ground truth — do not contradict

- Demand matrix `D`, `n x n`, non-negative integers, **S-regular**: every row and column sums
  to `S`. Equivalent to scaled doubly stochastic.
- Decomposition `\Dc = {(P_i, lambda_i)}`; cycle length `C(\Dc) = sum lambda_i`; permutation
  count `N`; port utilization `eta = S / C(\Dc)`.
- **Proposition 3.1:** `C(\Dc) >= S`, with equality iff every `P_i` is a *full* permutation.
- `DCT = T_compute + N * T_config + C(\Dc) * T_unit`. This arbitrates between methods; no
  single axis does.
- **Engines** (code identifier -> paper name): `maximum` -> Hungarian (exact, LSAP, `O(n^3)`);
  `heavy` -> GW Dynamic; `heavy_static` -> GW Static; `wfa` -> WFA. The first three are
  *strong* (full permutations on regular input). WFA may return a partial matching, hence `C > S`.
- **Splittings:** `radix-split(D,B)` with `B` in {16,8,4,2}, giving `K = L` digit planes;
  `euler-split(D,d)` with depth `d` in 1..4, giving `K = 2^d` parts. Both reduce to plain BvN
  at their trivial parameter, which is the baseline.
- A **setup** is the pair `PD(splitting, engine)`.
- **Workloads:** generated (`n=256`, `k=256` dense / `k=16` sparse, `omega=64` per-permutation
  weight bound, largest entry ~300, `S ~ 8200`
  dense); Griner–Avin traffic model; measured MoE traces (`n=8`, **not** regular, reported
  separately in 6.6).
- **Step-size strategies** (min/median/max) exist in code and data but are **PARKED**:
  sections 5.4 and 6.5 are commented out. Do not reintroduce them without instruction.
- **Sinkhorn** is a float-to-integer scaling option in the code only. It is **not** in the
  paper and explains nothing about permutation counts. If you see this claim, reject it.

## Where the data lives

```
bvn_project/
  src/algorithms/   bvn.py radix_decomposition.py euler_splitting.py wfa.py
  src/utils/        matrix_generator.py moe_trace.py
  agent_scripts/    campaign runners and every figure-generating script
  run/  euler_exp/  campaign outputs (CSV), by campaign and timestamp
  AGENT_GUIDE.md    live vs stale files, data provenance
```

Campaign directories are named by workload and setup, not by a fixed pattern. Confirm a path
exists before quoting from it.

## Paper status

| § | | |
|---|---|---|
| 1 | Introduction | **RESERVED — do not write** |
| 2 | Background and Related Work | **UNWRITTEN** — blocks 6 uncited references |
| — | Overview | written |
| 3 | Problem Definition and Model | written |
| 4 | The Design Space | written |
| 5 | Parallel Matrix Decomposition Framework | written, marked ready to review |
| 6 | Experimental Evaluation | written, under review now |
| 7 | DCT Analysis | written |
| 8 | Conclusions and Future Work | **UNWRITTEN** |
| — | Abstract | **RESERVED — do not write** |

## Known open gaps a reviewer will find

- No measurement environment stated anywhere: no CPU, core count, or library versions, though
  `T_compute` is a headline metric.
- No sample size and no variance reported; every figure is a point estimate.
- `S` is stated for the dense regime only; sparse is roughly `S = 512`.
- The parallel charge is a computed maximum over independently timed pieces, not a measured
  parallel run. Defensible, but the paper does not say so.

## Session state as of 2026-08-29

Chapters 3-6 have had a full pass this session. What changed, so nobody re-proposes it:

- A configuration is a **partial** permutation (`C_n`); full permutations are `P_n` subset `C_n`.
- Family index is a subscript, entries use brackets: `P_i`, `P[i,j]`, `P_i[j,k]`.
- `\eqref` renders "Eq. (n)"; method names go through macros; `\PD{split}{engine}` names a setup.
- Section 5 runs Phase 1, Phase 2, the two splittings, then why the pair decomposes D. Its
  subsections are Radix, Euler, Matching engines. The trade-off verdict was stripped out.
- Step-size strategies are **parked**: 5.4 and 6.5 commented out, the framework presents two
  choices. 3.3 keeps the step size itself.
- `W_max` was **removed from the model**. Equation (2) asserts regularity alone. The
  permutation-count floor in 4.2 is stated in terms of `w`, the largest entry present
  (~300), giving ~27 rather than the old 129.
- Chapter 6's five subsections each open on a question. Chapter 6 uses **setup**, not
  "configuration", for a (splitting, engine) pair.
- The three Chapter 6 tables are generated by `agent_scripts/make_paper_tables.py`. Never edit
  them by hand.

**Known open items in Chapter 7**: it still says "configuration" where Chapter 6 says "setup";
the figures mix `S = 8211` (Radix/BvN) with `S = 8174` (Euler), which is admitted in a code
comment in `dct_parameter_space.py` but nowhere in the paper; and 7.2's claim that
`C(D)*T_unit` is identical across contenders depends on exactly that.

## Build check

```bash
cd "/c/Users/ofekc/Desktop/Msc/Thesis/paper" && pdflatex -interaction=nonstopmode main_paper_with_comments.tex >_b.log 2>&1 && pdflatex -interaction=nonstopmode main_paper_with_comments.tex >_b.log 2>&1; grep -c '^!' _b.log; grep -c undefined _b.log
```

Both counts must be `0` before any work is offered to the author.

````
