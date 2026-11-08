---
collection: "system-design-distributed-systems-cloud"
file_id: "01"
topic: "System design interview flow"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-07"
reviewed: null
source_status: "course-topic draft; visible source directory consulted, original bodies unavailable"
---

# System Design Interview Flow

> **Reference boundary:** The visible system-design source titles in [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) were consulted as inventory only. This is a new course-topic draft; it does not recover or claim content from those unavailable source files.

## 1. Decision-first flow

1. State user action, success/failure, scale horizon, latency/availability/cost/privacy constraints and excluded scope.
2. Identify read/write paths, data ownership, source of truth, idempotency key and consistency contract.
3. Size offered load, payload, retention, peak/burst, storage, bandwidth and cache/queue headroom with units.
4. Draw components only after contracts: ingress/admission, compute, durable state, asynchronous path, cache/search, observability and operations.
5. Name failure/recovery: timeout, retry, duplicate, stale read, partial write, overload, dependency outage, migration and rollback.
6. Define metrics, load/failure test and a bounded tradeoff decision.

Architecture is a hypothesis. A diagram without invariants and verification does not establish reliability.

## 2. Core invariants

| Concern | Required question |
|---|---|
| Identity | Who is authorized and what tenant/data scope applies? |
| Write | What exact effect occurs once, and how is duplicate delivery handled? |
| Read | What freshness/version is required and what happens when stale? |
| Capacity | What is admitted, queued, shed or degraded at saturation? |
| Recovery | What durable record supports replay/reconciliation? |
| Change | How is schema/model/config rolled out and rolled back? |

## 3. Sizing example

At 200 requests/s, 4 KiB mean request and 24-hour raw retention, inbound payload is `200*4096*86400=70,778,880,000` bytes, about 65.92 GiB before replication, indexes, metadata, compression, retries or output. Treat this as a lower-bound input; choose retention/partitioning from measured distribution and recovery contract.

## 4. Failure diagnostics

| Symptom | First check | Bounded response |
|---|---|---|
| Duplicate charge/action | Idempotency/effect-log trace | Deduplicate/reconcile at durable effect owner |
| Queue grows indefinitely | Offered/accepted/service rate and deadline | Bound admission; shed/degrade deliberately |
| Cache serves deleted/stale data | Version/invalidation ordering trace | Define freshness/fencing contract |
| Retry amplifies outage | Retry budget and dependency error rate | Backoff/jitter/circuit and cap attempts |
| “Exactly once” claim fails | Producer, broker, sink and transaction boundaries | State actual guarantee; implement reconciliation |

## 5. Validation

Use deterministic request/effect histories for duplicates, timeout-after-commit, dependency failure, overload, stale cache, schema migration and rollback. Compare against a simple reference state. Measure p50/p95/p99 latency, availability, freshness, accepted/rejected work, recovery time and data loss/replay behavior by scenario.

## 6. Interview checks

1. **What comes before components?** Success, constraints and correctness contract.
2. **Why is a queue not a solution alone?** It shifts load; it needs bounds, delay and recovery policy.
3. **What makes a retry safe?** Idempotent effect or reconciliation.
4. **What does raw retention sizing omit?** Replication/index/metadata/compression/overhead and bursts.
5. **What proves reliability?** Failure replay plus observed metrics, not a diagram.

## 7. Quick reference

| Deliverable | Must include |
|---|---|
| Design | Contracts, ownership, units, failure policy and alternatives |
| Capacity | Offered/accepted load, assumptions and headroom |
| Validation | Baseline, failure traces, metrics and rollback |

**Limitations:** No source body, workload or deployed system was available; arithmetic is illustrative.
