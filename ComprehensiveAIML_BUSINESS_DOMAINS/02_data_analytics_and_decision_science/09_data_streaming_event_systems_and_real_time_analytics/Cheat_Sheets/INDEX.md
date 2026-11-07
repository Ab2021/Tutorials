# Data Streaming, Event Systems and Real-Time Analytics — Cheat Sheets

**Status:** Unreviewed drafts and planned topics, not a completed collection. Follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and linked cheat-sheet master. The original source case files are unavailable; their titles do not establish their contents or an observed incident.

## 1. Overview

For engineers designing and operating durable event pipelines, this collection connects partition ownership and replay to source-database handoff, downstream materialization, and event-time session state. Readiness means explaining the applicable ordering, state, time, correction and failure contracts, and testing them against actual product versions and workloads.

## 2. Topic ladder

| Level | Reference | Outcome and prerequisite | Status |
|---|---|---|---|
| Practitioner | [Kafka consumer partitions and lag](01_Kafka_Consumer_Partitions_and_Lag.md) | Explain assignment, partition offsets, lag and skew; requires basic logs and queues | Draft; unreviewed |
| Advanced | [CDC ordering, snapshot handoff and materialization](02_CDC_Ordering_Snapshot_and_Materialization.md) | Explain no-gap handoff and replay-safe per-key state; requires the Kafka sheet and database transactions | Draft; unreviewed |
| Advanced | [Flink event-time session windows and stateful processing](03_Flink_Event_Time_Session_Windows_and_Stateful_Processing.md) | Trace watermark-driven sessions, late merges, state recovery and corrections; requires Kafka, CDC, timestamp arithmetic and keyed state | Draft; unreviewed |
| Planned | Event schema checklist | Contract evolution and compatibility | Not written |
| Planned | Streaming semantics | Delivery, ordering and effect boundaries | Not written |
| Planned | Broader windowing patterns | Window families and wider event-time patterns beyond the focused Flink session sheet | Not written |
| Planned | Common Issues | Distinct diagnostic cases across this collection | Not written |

## 3. Quick topic lookup

| Question | Reference | Condition and warning |
|---|---|---|
| Why is one consumer partition falling behind? | [Kafka failure modes](01_Kafka_Consumer_Partitions_and_Lag.md#8-failure-modes--edge-cases) | Compare per-partition rates; aggregate capacity can conceal skew. |
| Can a late snapshot overwrite a newer row? | [CDC failure modes](02_CDC_Ordering_Snapshot_and_Materialization.md#8-failure-modes--edge-cases) | Only compare versions when the source protocol defines the comparator. |
| How long will replay take? | [CDC worked example](02_CDC_Ordering_Snapshot_and_Materialization.md#5-worked-numerical-example) | Backlog divided by surplus assumes stable rates and positive service surplus. |
| How can a late event merge published sessions? | [Flink worked trace](03_Flink_Event_Time_Session_Windows_and_Stateful_Processing.md#5-worked-numerical-example) | Trace uses an explicitly generic strict-overlap contract; deployed equality and cleanup behavior require version-specific checks. |
| Why did a session stop firing or leave an obsolete row? | [Flink failure modes](03_Flink_Event_Time_Session_Windows_and_Stateful_Processing.md#8-failure-modes--edge-cases) | Compare per-input watermarks and correction effects; a checkpoint alone does not fix either. |

## 4. Reading paths

- **Beginner:** Learn database transactions and message queues first, then read the Kafka sheet's definitions and worked example; foundational schema and broader window sheets remain planned.
- **Practitioner:** Read the complete Kafka sheet, the CDC handoff and troubleshooting sections, then the Flink session trace and failure modes; apply contracts to the deployed connector and sink.
- **Advanced:** Read all three sheets in full, trace crash boundaries and a late session merge, and design reconciliation and recovery tests. No cross-key atomicity or automatic sink retraction follows from checkpointing alone.

## 5. Prerequisite skills

Understand relational transaction commit and snapshot isolation, broker partitions and offsets, rate-by-time arithmetic, idempotency, and the deployed system's log-retention behavior. Database and connector guarantees must be checked in the version-specific documentation before implementation.

## 6. What this collection does not cover

The current drafts do not provide complete event-sourcing, general windowing-pattern coverage, Flink connector-configuration, or cross-database transaction recipes. The focused session sheet is not a complete windowing survey. Planned titles are not completed lessons. A diagnostic example is not an observed production case.

## 7. Master symbol glossary

| Symbol | Meaning and units | Scope / warning | Reference |
|---|---|---|---|
| W | Source snapshot boundary, source-specific coordinate | Valid only with the source's comparison and inclusion rules | [CDC definitions](02_CDC_Ordering_Snapshot_and_Materialization.md#3-foundational-definitions) |
| Offset | Record position within one Kafka partition | Not comparable to a database log coordinate | [Kafka sheet](01_Kafka_Consumer_Partitions_and_Lag.md) |
| Backlog | Outstanding events at a stated time | Count alone is not event age or proof of loss | [CDC example](02_CDC_Ordering_Snapshot_and_Materialization.md#5-worked-numerical-example) |
| `t`, `w` | Event timestamp and event-time watermark, integer milliseconds | Watermark is progress, not proof of completeness; `w` differs from CDC snapshot boundary `W` | [Flink definitions](03_Flink_Event_Time_Session_Windows_and_Stateful_Processing.md#3-foundational-definitions) |
| `g`, `L` | Session inactivity gap and allowed lateness, milliseconds | Equality, trigger and cleanup behavior are contract-specific | [Flink definitions](03_Flink_Event_Time_Session_Windows_and_Stateful_Processing.md#3-foundational-definitions) |

## 8. Quick Reference Card directory

| Card | Core rule | Use and critical condition |
|---|---|---|
| [Kafka](01_Kafka_Consumer_Partitions_and_Lag.md#12-quick-reference-card) | Track partition-level progress and durable effects. | Offset commits must match the processing contract. |
| [CDC](02_CDC_Ordering_Snapshot_and_Materialization.md#12-quick-reference-card) | Cover the handoff without gaps and prevent stale replay. | Ordering, identity and boundary semantics must be documented. |
| [Flink sessions](03_Flink_Event_Time_Session_Windows_and_Stateful_Processing.md#12-quick-reference-card) | Late bridges can merge sessions and invalidate earlier output identities. | Strict overlap is the generic trace's rule; verify deployed equality, cleanup and sink corrections. |

## 9. Common Issues locator

A standalone Common Issues artifact is **planned, not written**; no collection-wide issue count is claimed. For current diagnostics use the [Kafka failure modes](01_Kafka_Consumer_Partitions_and_Lag.md#8-failure-modes--edge-cases), [CDC failure modes](02_CDC_Ordering_Snapshot_and_Materialization.md#8-failure-modes--edge-cases), and [Flink session failures](03_Flink_Event_Time_Session_Windows_and_Stateful_Processing.md#8-failure-modes--edge-cases). Each entry needs a discriminating diagnostic, mitigation and verification before promotion into a consolidated issue catalog.

## 10. Sources, versions and maintenance

All three files are illustrative, unreviewed drafts. The source inventory's Debezium CDC and IoT session-window cases were unavailable. Recheck the actual source database, [Debezium connector](https://debezium.io/documentation/), [Kafka](https://kafka.apache.org/documentation/), [Flink 1.20 window documentation](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/dev/datastream/operators/windows/), sink version and retention settings whenever implementation or assumptions change; use documentation matching the deployed release. The [CDC](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#kafka) and [Flink](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#flink) source coverage briefs are topic maps, not empirical evidence. The session trace uses an explicit generic contract and has not been run on Flink.

**Additional planned brief:** [PLAN-2026-IR035](../../../00_REFERENCE_INFORMED_CASE_PLAN.md) covers score buckets, Fenwick summaries, deterministic ties, deduplicated and late events, shard moves and closing watermarks. Compare reference sorted snapshots and exact final-rank replay; a sum of stale counters is not an exact settlement snapshot. The [interview example](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Leaderboards/score_bucket_rank_consistency.md) is a reference only, not a finished sheet or verified deployment.

## 11. Collection acceptance checklist

- [ ] All planned source topics and a distinct Common Issues artifact are authored and reviewed.
- [ ] Each sheet retains all 12 sections, checked calculations or state traces, diagnostic failures, answered interview questions and its final card.
- [ ] Every source-specific assertion is supported or explicitly marked open; product versions and review triggers are recorded.
- [ ] Links, learning paths, glossary, issue locations and status are reconciled against the completed files.
