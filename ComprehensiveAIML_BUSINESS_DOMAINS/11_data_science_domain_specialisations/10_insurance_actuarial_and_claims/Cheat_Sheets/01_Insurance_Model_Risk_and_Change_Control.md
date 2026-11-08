---
collection: "insurance-actuarial-and-claims"
file_id: "01"
topic: "Insurance model risk and change control"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# Insurance Model Risk and Change Control

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) is source inventory only. This draft is not actuarial, pricing, reserving, legal, regulatory or insurance advice.

## 1. Model inventory and intended use

Record business use, owner, decision authority, population, data boundary, version, assumptions, limits, validation evidence, approval and review/retirement date. A technically valid prediction can be unsuitable for a regulated decision or its declared population.

## 2. Change classification

Classify data, feature, model, threshold, policy, deployment and vendor changes by potential impact. Material changes require defined impact assessment, independent review, evidence and approval before use. Emergency containment still requires retrospective evidence and review.

## 3. Monitoring

Monitor data completeness/freshness, input shifts, score/action distributions, outcome delay, performance where labels mature, fairness/segment checks where lawful and appropriate, overrides, complaints and operational errors. A stable aggregate metric does not prove stable performance across products or cohorts.

## 4. Control record

| Control | Evidence |
|---|---|
| Reproducibility | versioned code, data definition and artifact |
| Validation | method, sample boundary, results and limitations |
| Explainability | intended audience, method limits and reason records |
| Approval | authority, conditions and expiry |
| Change/incident | trigger, decision, rollback and reconciliation |

## 5. Failure modes

| Failure | Signal | Containment |
|---|---|---|
| Data definition changes | unexpected distribution/coverage shift | pause affected use; reconcile source/version |
| Undocumented override | action differs from policy record | log authority/reason and review pattern |
| Delayed outcomes ignored | stale performance claim | use maturity windows and provisional labels |
| Model reused out of scope | new product/population | assess and approve intended use change |
| Incomplete evidence | decision cannot be reproduced | block promotion or classify as exception |

## 6. Quick Reference Card

For every model: name use and owner; retain assumptions/data/version/evaluation; classify changes; approve with expiry; monitor data, actions and mature outcomes; retain overrides/incidents; reconcile and retire deliberately.

**Limitations:** No insurer, model, population or regulation was evaluated.
