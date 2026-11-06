# Module 19 Case Studies â€” Transformers, LLMs and Foundation-Model Internals

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Choose a model for code assistance | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | diagnose long-context quality loss | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR007 | Balance interactive inference latency and admitted throughput | Supporting curriculum; shared brief | hybrid | Which scheduling and KV-memory policy meets latency and isolation requirements at the admitted load? | `resource-admission` | `cost-latency-tradeoff`, `authorization-scope` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR009 | Allocate a domain-training budget with restart headroom | Supporting curriculum; shared brief | hybrid | Should a domain model be trained from scratch or continued from an existing checkpoint within an all-in budget? | `cost-latency-tradeoff` | `data-leakage`, `atomic-release-bundle` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR007 | [REF-IR007](../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_Inference/continuous_batching_serving.md) | GPU geometry, concurrency limits and speedups require workload measurements; speculative decoding can be slower |
| PLAN-2026-IR009 | [REF-IR009](../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_TrainingSCALING/legal_pretraining_scaling.md) | The reference compute and GPU figures are assumptions; an internal legal checkpoint is not validated legal advice |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
