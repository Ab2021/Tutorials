# Module 29 Cheat Sheets â€” API Platform Engineering, Cloud-Native Systems and FinOps

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | API design review | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | Kubernetes AI workload patterns | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | FinOps dashboard metrics | Concise patterns, decision rules, assumptions, and common traps. | planned |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR002 | Eligibility before scoring; capability adapters; health-aware routing; bounded retry budgets | Constraint violations, successful-task cost, deadline misses and fallback correctness on fault replays. Boundary: Revalidate provider capabilities and economics when authoring; source tables are not current service guarantees. | [REF-IR002](../../../../INTERVIEW_CASES/Agentic_AI_System/AI_frontier_sys/constrained_provider_routing.md) | planned; not written |
| PLAN-2026-IR022 | Token-bucket invariants; atomic refill and debit; leased budgets; expiry; failure policy | Aggregate-admission histories, refill boundaries, coordinator outages, lease races and rejected-request latency. Boundary: Periodic usage reporting alone is not coordinated allocation; token buckets and exact rolling windows have different contracts. | [REF-IR022](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Rate_Limiting/distributed_token_buckets.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
