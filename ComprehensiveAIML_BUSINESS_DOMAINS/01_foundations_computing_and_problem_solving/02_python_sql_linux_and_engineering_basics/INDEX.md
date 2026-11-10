# Module 02 â€” Python, SQL, Linux and Engineering Basics

> **Part:** [Part 01 â€” Foundations, Computing and Problem Solving](../INDEX.md)  
> **Prerequisites:** 01  
> **Purpose:** The working toolkit for data, ML, and AI systems: reproducible code, queries, APIs, containers, and collaboration.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | Python data workflows | [authored](Cheat_Sheets/01_Python_Async_and_Concurrency.md) |
| 2 | SQL joins and windows | [authored](Cheat_Sheets/05_SQL_Analytics_Patterns.md) |
| 3 | Linux and networking | [authored](Cheat_Sheets/99_25_Common_Issues.md) |
| 4 | Git testing and packaging | planned |
| 5 | HTTP APIs and JSON | planned |
## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Authored source submodules

Visible source title `01_python_async_and_concurrency.md` is mapped from [`Train/Templates/directory.txt`](../../../Templates/directory.txt). This is a new unreviewed draft; original source body unavailable.

- [Python Async and Concurrency](Cheat_Sheets/01_Python_Async_and_Concurrency.md) — async I/O, capacity, cancellation, side-effect ownership and validation.
- [REST APIs and WebSockets](Cheat_Sheets/02_REST_APIs_and_WebSockets.md) — request/stream contracts, retry/effect semantics, backpressure and failure validation.
- [LLM Fundamentals](Cheat_Sheets/03_LLM_Fundamentals.md) — token prediction, inference contracts, decoding boundaries and validation.
- [Docker and Containers](Cheat_Sheets/04_Docker_and_Containers.md) — build/runtime identity, secrets, health, state and capacity controls.

## Canonical case studies

- [Build a Data Quality CLI](Case_Studies/01_Build_Data_Quality_CLI.md) — illustrative batch-gating decision with idempotency and replay checks; draft.
- [Design an Analytics API](Case_Studies/02_Design_Analytics_API.md) — illustrative query-publication decision with a contract and safe deferral path; draft.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Python async, concurrency, REST, WebSockets, LLM basics and Docker](../../00_SOURCE_COVERAGE_AGENTS.md#agent-prerequisites) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.