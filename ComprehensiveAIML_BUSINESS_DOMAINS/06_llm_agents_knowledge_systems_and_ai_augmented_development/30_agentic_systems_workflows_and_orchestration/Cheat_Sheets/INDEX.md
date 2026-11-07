# Module 30 Cheat Sheets â€” Agentic Systems, Workflows and Orchestration

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. The catalog below includes a draft and planned sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Cheat sheet catalog

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | agent-pattern catalog | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | workflow selection | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | handoff protocol | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 4 | [Agent sessions, state ownership, isolation and run lifecycle](04_agent_sessions_state_isolation_and_lifecycle.md) | Control-plane ownership, lifecycle recovery, isolation, external effects and sizing; based on the Agents Stack Infra Module 03 directory crosswalk. | draft; not reviewed |
| 5 | [Durable workflows and idempotent execution](05_durable_workflows_and_idempotent_execution.md) | Checkpoints, retries, timeouts, business-side effect safety, queue sizing, reconciliation and diagnostic scenarios; source title mapped from Agents Stack Infra Module 04, but source contents unavailable. | draft; not reviewed |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR005 | Immutable release manifests; compatibility gates; side-effect-free shadowing; staged canaries; incident containment | Bundle-compatibility tests, rollback drills, guardrail breaches and unresolved side effects after recovery. Boundary: A prompt-only rollback cannot restore all dependencies; example dwell times and canary percentages are not defaults. | [REF-IR005](../../../../INTERVIEW_CASES/Agentic_AI_System/INDEX/Agent_Release_And_Incident_Operations.md) | planned; not written |
| PLAN-2026-IR006 | LangGraph checkpoints and interrupts; authenticated resume; approval binding and expiry; idempotent submission | Crash-window replay, duplicate webhooks, changed purchase payloads, checkpoint migration and reconciliation tests. Boundary: External APIs may not support exactly-once effects; approval waiting times and run counts are exercise assumptions. | [REF-IR006](../../../../INTERVIEW_CASES/Agentic_AI_System/Langraph/durable_human_approval_graph.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
