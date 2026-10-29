---
name: paper-writing-principles
description: "How Ofek wants the thesis paper written and revised — no conclusion previews, no forward references, one comment at a time, verify before claiming"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 26bcde8f-a14e-4d6d-9736-b16700cca933
  modified: 2026-08-29T09:28:34.794Z
---

Principles established across the paper-revision sessions, reinforced in Chen Avin's
review (2026-08-29):

- **Do not preview conclusions.** A section introduces methods by *what they do*, in one
  sentence each, and leaves the trade-off verdict to the chapter that measures it. This is
  why "Radix does not preserve regularity / Euler does" was cut from §5, and why Chapter 6
  no longer opens by stating its own result.
- **Do not forward-reference.** If a term is needed early, define it there or drop the
  pointer. When a subsection is removed, its inbound pointers go with it.
- **Order follows the argument.** §5's subsections were reordered to Radix, Euler, engines
  so they match Phase 1 then Phase 2 as the introduction presents them.
- **Every equation gets a "where var is X" clause.** Every symbol used must be defined.
- **First person, telling the research story**: "we consider", "we tested", "we claim".
- **Never write the Introduction or the Abstract** — reserved for Ofek and Chen, written last.
- **All figures must be vector PDF**, regenerated from a script in `agent_scripts/`.
- **Do not over-claim**; cite or measure everything. Do not introduce a new term when one
  is already defined.
- **No em dashes** in prose. Beware: converting a bracketing dash pair to commas creates
  comma splices — the signature is a sentence with four or more commas.

**Why:** the paper is read by a supervisor looking for exactly these faults, and several
rounds were lost to re-proposing wording already rejected.

**How to apply — the working loop that Ofek wants:**
1. He brings **one** comment at a time. Do not batch or run ahead to other sections.
2. Show CURRENT vs SUGGESTED before editing; he often wants the *compiled* result, not raw
   LaTeX — build a preview copy in the scratchpad, `pdftoppm` the page, and send the image.
3. Apply, compile twice, check `errors: 0` and `undefined: 0` and the page count, then
   commit with a message saying *why*.
4. Report what was changed **and what was deliberately left**, so he can overrule.

Watch for consequential ripples and raise them rather than silently fixing: a definitional
change breaks downstream claims, a removed subsection orphans its references, and a
reordered block can strand "as we saw above" phrasing.

See [[paper-notation-conventions]], [[paper-working-files]], [[agent-guide]].
