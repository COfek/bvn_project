---
name: paper-working-files
description: "Which paper file to edit, how it syncs to Overleaf via GitHub, and what is parked rather than deleted"
metadata: 
  node_type: memory
  type: project
  originSessionId: 26bcde8f-a14e-4d6d-9736-b16700cca933
  modified: 2026-08-29T09:28:49.864Z
---

As of 2026-08-29, in `C:\Users\ofekc\Desktop\Msc\Thesis\paper\`:

- **`main_paper_with_comments.tex` is the working file** — it is Chen Avin's copy, carrying
  the `\chen{}` macro. `main_paper.tex` is kept byte-identical by copying after each edit.
  `overleaf_updated.tex` is a dated snapshot of Overleaf and is *behind*, not ahead.
- The folder is now **its own git repo** (initialised this session; `.gitignore` covers
  LaTeX build artifacts, root-level compiled PDFs, and the raster `plots/` folder). Commit
  each accepted change separately.
- **Overleaf is synced through GitHub (since 2026-09-24).** Chen linked the Overleaf
  project to the private repo `chenavin/Ofek-Cohen-BvN-Thesis` (COfek has write access).
  The local repo's history is NOT pushed there: remote `overleaf` plus a worktree
  `Thesis\paper_overleaf` on branch `overleaf-main`. Publish with
  `bash paper/push_overleaf.sh "msg"` (or `--dry-run`); it copies the files Overleaf tracks
  plus anything `main_paper.tex` references, commits, pushes. If Overleaf has new commits
  (Chen edited there) it stops; merge those into `main_paper_with_comments.tex`, then rerun
  with `--integrated`. After a push, Ofek clicks Menu > GitHub > Pull GitHub changes in
  Overleaf. The old `overleaf` git tag is obsolete.
- **The paper needs 14 files to build**: `main_paper.tex`, five `\input` tables
  (`tab_results_summary`, `tab_euler_all_engines`, `tab_euler_vs_radix`, `tm_tab_comparison`,
  `moe_tab_summary`), and eight PDFs in `plots_pdf/`. The bibliography is inline — no `.bib`.
  Everything else in the folder is superseded. It is **not** self-contained; an earlier
  claim that it was came from a grep that silently failed.
- **Parked, not deleted:** step-size strategies. §5.4 and §6.5 are commented out with
  banners, because the framework now presents two choices (splitting, engine) after Chen
  commented out the third. The data and `strategy_full_table.tex` still exist. Restoring is
  one uncomment plus putting the sentence back in "The pipeline is a product".
- **Still unwritten:** Abstract, Chapter 1, Chapter 2 (Background and Related Work),
  Chapter 8. Six bibliography entries are defined but uncited (`hopcroftkarp1973`,
  `chang2000`, `keslassy2003`, `avinschmid2025`, `harvest`, `wu2026`) — all waiting on
  Chapter 2. Ofek's instruction: anything needing Chapter 2 waits.

**Why:** work was applied to the wrong file for three whole comments before the working
file was identified, and the two copies had diverged in both directions.

**How to apply:** edit `main_paper_with_comments.tex`, copy over `main_paper.tex`, compile
twice, commit. See [[paper-writing-principles]] and [[paper-notation-conventions]].

**Claude setup lives in the paper repo (since 2026-09-24).** `paper_overleaf/.claude/CLAUDE.md`
holds Ofek's five strict rules (precision/no contradictions, be critical of both of us,
report every change, never touch Chen's or Ofek's comments or `\ready`/`\TODO`, readability
with "where" clauses and "we tested"). `Thesis/CLAUDE.md` only imports it. The seven paper
agents are copied to `paper_overleaf/.claude/agents/` (originals still in bvn_project).
Skills load only from the session's start dir up to its git root, so for the repo's skills
to load, sessions should start in `paper_overleaf/` (or `Thesis/`).
