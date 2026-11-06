# Module 29 Case Studies â€” API Platform Engineering, Cloud-Native Systems and FinOps

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Internal AI platform design | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | cloud-cost anomaly response | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR002 | Select model routes within hard customer constraints | Supporting curriculum; shared brief | llm-agentic | Which eligible provider and fallback path satisfy residency, capability, deadline and spend requirements? | `eligibility-before-ranking` | `cost-latency-tradeoff` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR022 | Bound global API admission with token leases | Supporting curriculum; shared brief | hybrid | Which admission policy remains within its global budget during contention or coordinator failure? | `resource-admission` | `correctness-contract`, `bounded-work` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR002 | [REF-IR002](../../../../INTERVIEW_CASES/Agentic_AI_System/AI_frontier_sys/constrained_provider_routing.md) | Revalidate provider capabilities and economics when authoring; source tables are not current service guarantees |
| PLAN-2026-IR022 | [REF-IR022](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Rate_Limiting/distributed_token_buckets.md) | Periodic usage reporting alone is not coordinated allocation; token buckets and exact rolling windows have different contracts |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
