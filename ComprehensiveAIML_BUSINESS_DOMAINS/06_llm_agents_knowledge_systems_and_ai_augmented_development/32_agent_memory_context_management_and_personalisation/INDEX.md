# Module 32 â€” Agent Memory, Context Management and Personalisation

> **Part:** [Part 06 â€” LLM Agents, Knowledge Systems and AI-Augmented Development](../INDEX.md)  
> **Prerequisites:** 20,28,30  
> **Purpose:** Manage working context, durable memory, user preferences, provenance, retention, and retrieval quality explicitly.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | context assembly | planned |
| 2 | episodic and semantic memory | planned |
| 3 | memory write policies | planned |
| 4 | personalisation | planned |
| 5 | privacy and deletion | [Memory lifecycle and retention draft](Cheat_Sheets/01_Memory_Lifecycle_and_Retention.md); broader topic planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Authored course-topic drafts

- [Memory Lifecycle and Retention](Cheat_Sheets/01_Memory_Lifecycle_and_Retention.md) — lifecycle, retrieval, correction and deletion controls; visible source inventory only.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Context, memory, transcript, assembly, RAG, Self-RAG, memory lifecycle/types and Sourcegraph code context](../../00_SOURCE_COVERAGE_AGENTS.md#agent-context-memory) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Hermes self-improvement, autonomous skills, persistent memory, user modeling, platform deployment, tools and OpenClaw comparison](../../00_SOURCE_COVERAGE_AGENTS.md#agent-hermes) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Context preservation, escalation, error propagation, large-codebase exploration, human-review confidence, provenance/uncertainty and practice](../../00_SOURCE_COVERAGE_AGENTS.md#claude-reliability) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [LangGraph memory/persistence and session isolation](../../00_SOURCE_COVERAGE_AGENTS.md#lang-memory) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.