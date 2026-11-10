# Module 31 â€” Tools, MCP, Protocols and Enterprise Integrations

> **Part:** [Part 06 â€” LLM Agents, Knowledge Systems and AI-Augmented Development](../INDEX.md)  
> **Prerequisites:** 04,20,30  
> **Purpose:** Expose capabilities safely through tool schemas, protocols, authentication, authorization, and side-effect controls.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | tool schemas and ergonomics | planned |
| 2 | MCP and protocol design | planned |
| 3 | identity and OAuth | planned |
| 4 | transaction safety | planned |
| 5 | integration testing | planned |
## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Authored course-topic drafts

- [Tool Schema and Side-Effect Boundaries](Cheat_Sheets/02_Tool_Schema_and_Side_Effect_Boundaries.md) — schema, semantic validation, authorization boundaries and retry recovery; visible source inventory only.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR006 | LangGraph checkpoints and interrupts; authenticated resume; approval binding and expiry; idempotent submission | How can a purchase proceed only after approval of its exact version despite retries and restarts? | Crash-window replay, duplicate webhooks, changed purchase payloads, checkpoint migration and reconciliation tests |
| PLAN-2026-IR014 | Schema versioning; semantic checks; dead-letter replay; idempotency; quality-constrained tiered RAG routing | How should extraction contracts and model routes be accepted without confusing valid syntax with business success? | Gold field accuracy and reconciliation for extraction; full-path cost and grounded quality by tier for routing |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Tool schemas, MCP, capability versus execution, tool risk, MCP server and Sourcegraph MCP](../../00_SOURCE_COVERAGE_AGENTS.md#agent-tools-mcp) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [API, Agent SDK, MCP, Claude Code, anti-patterns and decision-flow references](../../00_SOURCE_COVERAGE_AGENTS.md#claude-reference) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Tool descriptions/boundaries, structured errors, distribution/tool choice, MCP integration, built-ins and practice](../../00_SOURCE_COVERAGE_AGENTS.md#claude-mcp) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [production anti-patterns and MCP integration](../../00_SOURCE_COVERAGE_AGENTS.md#scenario-15) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [LangChain foundations, ecosystem, LCEL, prompts/models/parsers](../../00_SOURCE_COVERAGE_AGENTS.md#lang-foundations) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
