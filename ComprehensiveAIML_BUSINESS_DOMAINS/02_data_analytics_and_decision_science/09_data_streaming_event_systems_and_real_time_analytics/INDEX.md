# Module 09 â€” Data Streaming, Event Systems and Real-Time Analytics

> **Part:** [Part 02 â€” Data, Analytics and Decision Science](../INDEX.md)  
> **Prerequisites:** 04,06  
> **Purpose:** Use event-driven patterns for temporal data, operational alerts, feature freshness, and real-time products.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | event schemas and contracts | planned |
| 2 | Kafka and stream processing | planned |
| 3 | time windows and state | [authored](Cheat_Sheets/03_Flink_Event_Time_Session_Windows_and_Stateful_Processing.md) |
| 4 | CDC and event sourcing | planned |
| 5 | exactly-once trade-offs | planned |
## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR035 | Score buckets; Fenwick count summaries; deterministic ties; deduplicated events; cross-shard moves; closing watermarks | How should score events support fast display ranks and a reproducible final prize result? | Reference sorted snapshots, duplicate/late events, negative scores, bucket crossings and exact final-rank replay |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Streaming framework, HLD/LLD and fixes/tradeoffs source documents](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#stream-framework) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Flink CEP/broadcast fraud, IoT session windows and financial reconciliation exactly-once](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#flink) | Partial, unreviewed CEP fraud case and session-window sheet; IoT and reconciliation cases remain planned. |
| [Multi-tenant platform, Debezium CDC and tiered/infinite-retention design](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#kafka) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Event-sourced CQRS orders, ecommerce clickstream analytics and SIEM/log aggregation](../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#stream-design) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
