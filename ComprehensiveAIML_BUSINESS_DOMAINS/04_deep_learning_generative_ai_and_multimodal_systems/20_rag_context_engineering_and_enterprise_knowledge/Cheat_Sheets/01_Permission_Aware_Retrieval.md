---
collection: "rag-context-engineering-and-enterprise-knowledge"
file_id: "01"
topic: "Permission-aware retrieval"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# Permission-Aware Retrieval

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) was consulted as inventory only. This draft does not claim a secure deployed retrieval system.

## 1. Retrieval eligibility precedes ranking

Resolve requester identity, tenant, role/attributes, effective policy and document/version state before candidate access. Ranking an unauthorized document and filtering it only after content enters model context is a boundary failure.

## 2. Retrieval contract

Each chunk needs stable identity, parent/version, tenant, access policy/version, effective/recorded time, deletion state, embedding/index version and provenance. The answer path preserves selected evidence and policy decision for review.

## 3. Evaluation

Measure eligible-span recall: required authorized spans retrieved divided by required authorized spans. Also test ACL violations, wrong-version citations, deletion visibility, abstention quality and latency. High generic recall is not evidence of authorization correctness.

## 4. Failure modes

| Failure | Signal | Containment |
|---|---|---|
| late permission filter | unauthorized candidate reaches context | authorize before retrieval/context assembly |
| stale policy/index | access differs from source policy | version/refresh/revocation test |
| deleted content returns | tombstone ignored in index | deletion propagation and replay checks |
| unsupported answer | claim lacks evidence | cite eligible span or abstain/escalate |
| tenant mixing | cross-tenant result | hard tenant scope and adversarial tests |

## 5. Quick Reference Card

Bind identity, tenant, policy/version and time to retrieval; retain document/version/deletion provenance; evaluate eligible recall and ACL violations; cite or abstain; test revocation, migration and cross-tenant boundaries.

**Limitations:** No policy source, authorization service or retrieval index was inspected.
