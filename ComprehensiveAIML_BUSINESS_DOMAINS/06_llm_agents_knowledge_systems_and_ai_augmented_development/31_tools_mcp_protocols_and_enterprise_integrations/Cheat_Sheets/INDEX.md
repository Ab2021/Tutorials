# Module 31 Cheat Sheets â€” Tools, MCP, Protocols and Enterprise Integrations

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | tool schema guide | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | MCP server checklist | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | side-effect risk matrix | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 4 | [MCP-01: Model Context Protocol](01_model_context_protocol.md) | Versioned lifecycle, protocol/executor boundaries, security, uncertain writes, sizing and diagnostics. | draft; unreviewed; publication checks pending |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR006 | LangGraph checkpoints and interrupts; authenticated resume; approval binding and expiry; idempotent submission | Crash-window replay, duplicate webhooks, changed purchase payloads, checkpoint migration and reconciliation tests. Boundary: External APIs may not support exactly-once effects; approval waiting times and run counts are exercise assumptions. | [REF-IR006](../../../../INTERVIEW_CASES/Agentic_AI_System/Langraph/durable_human_approval_graph.md) | planned; not written |
| PLAN-2026-IR014 | Schema versioning; semantic checks; dead-letter replay; idempotency; quality-constrained tiered RAG routing | Gold field accuracy and reconciliation for extraction; full-path cost and grounded quality by tier for routing. Boundary: Contains two reference problems: author separate narratives; valid JSON and cheap token prices do not establish successful outcomes. | [REF-IR014](../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Structured_Extraction_And_Cost_Routing.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
