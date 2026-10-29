---
name: related-work-articles
description: Two Purdue related-work papers reviewed for the thesis and where the summary lives
metadata: 
  node_type: memory
  type: reference
  originSessionId: 26bcde8f-a14e-4d6d-9736-b16700cca933
  modified: 2026-07-23T17:13:19.559Z
---

Lecturer-assigned related work (both from the Purdue StygianNet group, Vamsi Addanki):

1. **Harvest** (SIGCOMM '26, Rahman/Joseph/Kodkani/Arzani/Addanki) — reconfiguration-delay-aware
   photonic switching schedules for collectives. Derives the same DCT / α–β cost model we use
   (`α + δ·ℓ + β·m/θ` + per-reconfig `α_r`); BvN-every-step is their beaten baseline; Fig 3 is a
   design-spectrum precedent for our winner map. URL: https://stygianet.cs.purdue.edu/papers/harvest-sigcomm26.pdf
2. **Birkhoff-MoE / "Don't Forget the Compute!"** (HotOptics '26, Amponsah/Addanki) — decomposing
   MoE all-to-all matrices; real Mixtral/DeepSeek routing matrices as a non-synthetic, non-doubly-
   stochastic workload (needs Sinkhorn); their greedy max-weight decomposition ≈ our Hungarian/GW
   engines; Fig 2 shows BvN (many small matchings) vs max-weight (few dense) — the "matrices from
   the article" the lecturer wants brought in. URL: https://stygianet.cs.purdue.edu/papers/birkhoff-moe-hotoptics26.pdf

Full thesis-focused reading notes: `Desktop\Msc\Thesis\article_summaries.md`.
See also [[paper-latex-location]].
