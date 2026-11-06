# Module 27 Case Studies â€” Data Quality, Observability and Reliability Engineering

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Broken upstream schema incident | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | silent label corruption investigation | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR001 | Gate retraining on corpus provenance and long-tail quality | Proposed home | hybrid | Which documents should be admitted or quarantined before the next training run? | `distribution-shift` | `data-leakage`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR033 | Rebalance caches without overwhelming the authoritative store | Supporting curriculum; shared brief | hybrid | How should cache fills and ring migration preserve freshness while limiting database amplification? | `materialized-view` | `resource-admission`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR001 | [REF-IR001](../../../../INTERVIEW_CASES/Agentic_AI_System/AI_engg_challenges/synthetic_corpus_contamination.md) | A synthetic-text classifier cannot establish authorship; source incident names and costs are unverified |
| PLAN-2026-IR033 | [REF-IR033](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Cache_Resilience/cold_cache_and_ring_migration.md) | Local single-flight is not fleet-wide coordination; expected key movement is not the same as request miss share |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
