# Module 30 â€” Agentic Systems, Workflows and Orchestration

> **Part:** [Part 06 â€” LLM Agents, Knowledge Systems and AI-Augmented Development](../INDEX.md)  
> **Prerequisites:** 20,25,28  
> **Purpose:** Design bounded agent workflows with state, planning, delegation, durable execution, and human oversight.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | [agent boundaries](./Cheat_Sheets/01_agent_boundaries_and_request_ownership.md) | draft; not reviewed |
| 2 | workflow vs agent | planned |
| 3 | planning and reflection | planned |
| 4 | multi-agent orchestration | planned |
| 5 | durable state and recovery | partial draft coverage in cheat sheets 04–05; not reviewed |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

### Authored case draft

- [Enterprise agent approval gate](./Case_Studies/01_enterprise_agent_approval_gate.md) — hypothetical ticket-writing workflow covering approval binding, retries, production controls and interview decisions. Draft; not reviewed or deployed. The broader topics above remain planned.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR005 | Immutable release manifests; compatibility gates; side-effect-free shadowing; staged canaries; incident containment | Which model, prompt, tools, index and policy versions can be promoted and recovered together? | Bundle-compatibility tests, rollback drills, guardrail breaches and unresolved side effects after recovery |
| PLAN-2026-IR006 | LangGraph checkpoints and interrupts; authenticated resume; approval binding and expiry; idempotent submission | How can a purchase proceed only after approval of its exact version despite retries and restarts? | Crash-window replay, duplicate webhooks, changed purchase payloads, checkpoint migration and reconciliation tests |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Agent source references, hidden folders, PDFs and incomplete exam/harness evidence](../../00_SOURCE_COVERAGE_AGENTS.md#agent-source-artifacts) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Overloaded agent term, ten-layer stack, request path and boundary distinctions](../../00_SOURCE_COVERAGE_AGENTS.md#agent-stack-model) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Control planes, session ownership, state management, isolation and run lifecycle](../../00_SOURCE_COVERAGE_AGENTS.md#agent-control-sessions) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Runtime responsibilities, durable execution, LangGraph, workflow patterns and runtime choice](../../00_SOURCE_COVERAGE_AGENTS.md#agent-runtimes) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Reflection, tool use, planning, multi-agent, orchestration, human review, composition and emerging patterns](../../00_SOURCE_COVERAGE_AGENTS.md#agent-patterns) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [OpenClaw gateway, sessions, lane-aware queues, WebSockets, runtime loop, file memory, concurrency and security](../../00_SOURCE_COVERAGE_AGENTS.md#agent-openclaw) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Agentic architecture: loop lifecycle, orchestration, subagent context, enforcement/handoff, SDK hooks, decomposition, sessions and practice](../../00_SOURCE_COVERAGE_AGENTS.md#claude-architecture) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Pharma landscape, cross-domain architecture, issues and implemented RAG/agent case research](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#pharma-patterns) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [LangGraph state, nodes/edges, agents, tools and multi-agent systems](../../00_SOURCE_COVERAGE_AGENTS.md#lang-graphs) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
