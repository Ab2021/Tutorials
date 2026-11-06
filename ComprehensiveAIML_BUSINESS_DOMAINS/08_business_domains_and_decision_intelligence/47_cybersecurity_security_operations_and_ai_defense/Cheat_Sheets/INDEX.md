# Module 47 Cheat Sheets â€” Cybersecurity, Security Operations and AI Defense

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | threat-model template | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | detection rule guide | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | AI red-team checklist | Concise patterns, decision rules, assumptions, and common traps. | planned |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR028 | Hash collision amplification; catastrophic regex backtracking; pre-parse limits; bounded queues and downstream budgets | Adversarial near-match inputs, collision workload, CPU per request, timeout cascades and legitimate-input acceptance. Boundary: The hash and regex scenarios are separate; legacy runtime behavior and example input limits require fresh validation. | [REF-IR028](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Algorithmic_DoS/hash_and_regex_resource_exhaustion.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
