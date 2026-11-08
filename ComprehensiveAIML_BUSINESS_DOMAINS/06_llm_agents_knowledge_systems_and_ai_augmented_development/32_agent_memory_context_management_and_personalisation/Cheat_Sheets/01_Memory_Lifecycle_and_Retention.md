---
collection: "agent-memory-context-management-and-personalisation"
file_id: "01"
topic: "Memory lifecycle and retention"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# Memory Lifecycle and Retention

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) was consulted as inventory only. This draft does not claim a deployed memory or privacy system.

## 1. Separate records

Transcript is an interaction log; working context is selected current input; memory is a retained representation with purpose, provenance, scope and lifecycle. Never promote every transcript assertion to durable memory.

## 2. Write policy

A write needs purpose, subject/tenant scope, source/provenance, confidence, sensitivity, retention, update/conflict rule, deletion route and owner. Prefer explicit user-confirmed preferences and task facts over inferred sensitive traits.

## 3. Retrieval policy

Retrieve only records authorized for requester and purpose; include provenance/time/confidence; bound context; reconcile conflicts; expose uncertainty or abstain. A semantically similar record may be stale, unauthorized or contextually wrong.

## 4. Lifecycle

Capture → validate/classify → store with metadata → retrieve with policy → update/supersede → expire/delete → audit/reconcile. Deletion must cover derived indexes/caches within a stated propagation contract; deleting one display row is insufficient.

## 5. Failure modes

| Failure | Signal | Containment |
|---|---|---|
| stale preference | user correction conflicts | recency/source rules and confirmation |
| cross-user leakage | memory scope mismatch | tenant/subject authorization before retrieval |
| sensitive inference stored | derived trait lacks purpose | minimize, block or require explicit consent |
| deletion incomplete | record appears after request | index/cache propagation and audit replay |
| context overload | irrelevant memories dominate | budget/relevance/provenance filters |

## 6. Quick Reference Card

For each record retain purpose, subject/tenant, source, time, confidence, sensitivity, retention, update and deletion behavior; enforce authorized retrieval; surface uncertainty; test correction, conflict, expiry and deletion propagation.

**Limitations:** No user data, consent mechanism or retention law was evaluated.
