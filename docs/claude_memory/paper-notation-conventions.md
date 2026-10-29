---
name: paper-notation-conventions
description: "Notation and macro rules settled with Chen Avin for the thesis paper — configurations vs permutations, index placement, operator and method-name macros"
metadata: 
  node_type: memory
  type: project
  originSessionId: 26bcde8f-a14e-4d6d-9736-b16700cca933
  modified: 2026-08-29T09:28:17.556Z
---

Notation decided in Chen Avin's review of the paper (2026-08-29 session) and applied
throughout `main_paper_with_comments.tex`:

- **Configurations vs permutations.** `\mathcal{C}_n` is the set of *partial permutation
  matrices* (at most one 1 per row and column) — this is what a switch configuration is.
  `\mathcal{P}_n \subset \mathcal{C}_n` is the *full* permutations (exactly one). A
  decomposition's `P_i \in \mathcal{C}_n`; **strong** means every `P_i \in \mathcal{P}_n`.
  Never write "a configuration is a permutation matrix" — that identity was deleted.
- **Index placement.** The index of a matrix within a family is a **subscript**: `P_i`,
  `P_t`, `D_k`, `\Delta_\ell`. Entry access uses **brackets**: `P[i,j]`, `P_i[j,k]`.
  Rationale: entries of a permutation are named only twice in the paper, so the subscript
  slot is better spent on the family index, and `P_i` pairs visually with `\lambda_i`.
  `D_{ij}` and `R_{jk}` keep plain subscripts — neither carries a family index.
- **Parts and configurations.** Parts run `k = 1..K`, configurations `i`, so `P_{k,i}`.
- **Equation references.** `\eqref` is redefined in the preamble to render `Eq.\nobreak\ (n)`.
  Use `\nobreak\ `, not `~`: a tie after the abbreviating period inherits sentence spacing.
- **Operators.** `\msplit(D,K)`, `\radixsplit(D,B)`, `\eulersplit(D,d)` — the name order is
  *radix-split*, not *split-radix*. `B` is the base, `d` is the recursion depth giving `2^d`
  parts; Radix gives `K = L` parts. `\split` is unusable: amsmath owns it, and a hyphen in
  math mode renders as a minus, hence `\text{-}`.
- **Setups.** `\PD{splitting}{engine}` renders `PD(radix-split(D,2), WFA)`. Use it **only
  where a specific pair is the subject** — most of Chapter 6 holds one factor and moves the
  other, and naming both each time buries the one that changed.
- **Method names go through macros**: `\BvN \Hung \GWD \GWS \WFA \Scan \ScanEngine \Augment
  \EulerSplit`, all with `\xspace`. Never inside a `\section{}` heading (hyperref bookmark)
  and never inside a quoted bibliography title.
- **Markers.** `\chen{...}` is a blue supervisor comment; `\ready` is a green
  `[ready to review]`, placed *below* the heading as
  `\noindent\ready\par\smallskip`, not inside `\section{}`.

See [[paper-writing-principles]] and [[paper-working-files]].
