---
collection: "data-modeling-databases-lakehouse-and-governance"
file_id: "01"
topic: "Data contract checklist"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# Data Contract Checklist

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) was consulted as an inventory. This new course-topic draft does not reconstruct unavailable source bodies.

## 1. Purpose

A data contract is an explicit producer–consumer agreement for a named dataset or event: identity, schema, semantics, quality, delivery, ownership and change process. It does not turn unknown source semantics into truth.

## 2. Contract surface

| Field | Specify | Avoid |
|---|---|---|
| Identity | dataset/event name, owner, consumers | an ambiguous business label |
| Schema | field type, nullability, keys, units, timezone | implicit coercion |
| Semantics | event versus processing time, grain, update/delete meaning | assuming one row equals one business fact |
| Quality | freshness, completeness, uniqueness, validity thresholds | a dashboard-only promise |
| Delivery | cadence, partition/order scope, retention, replay | global-order assumptions |
| Change | compatibility rule, notice, rollback/contact | silent field reuse |

## 3. Grain before metrics

State exactly what one record represents. Example: `order_line_snapshot` has one row per `(order_id, line_id, version)`; summing amount without selecting the latest version double-counts revisions. A primary key protects identity only when the key’s business meaning and mutation model are declared.

## 4. Quality equations

For a batch expected to contain `E` records and containing `V` records passing all validity rules,

`completeness = received / E`, and `validity = V / received`.

If `E=10,000`, `received=9,800`, and `V=9,604`, completeness is `0.98` and validity is `0.98`. These ratios need an agreed denominator; an unknown expected count makes completeness unmeasurable, not 100%.

## 5. Change protocol

1. Classify change as compatible, conditional or breaking for each declared consumer.
2. Version schema/contract and publish examples plus effective time.
3. Run producer and consumer validation in a bounded shadow/canary path.
4. Monitor contract metrics and acknowledgements.
5. Roll back or dual-publish only under a documented expiry and reconciliation rule.

## 6. Validation

Automate schema, key uniqueness, null/range/domain, referential, freshness and volume checks. Keep a small reference/oracle query for critical aggregates. A passing schema check does not prove semantic correctness, lineage or authorization.

## 7. Failure modes

| Failure | Signal | Containment |
|---|---|---|
| Grain changed | aggregate jump without business explanation | block affected metric; reconcile at declared grain |
| Late/duplicate delivery | key/version collision or freshness breach | idempotent merge using declared ordering |
| Nullable required field | null-rate threshold breached | quarantine/alert; avoid coercing silently |
| Schema-compatible semantic change | field still parses but metric shifts | version semantics and require consumer approval |
| Ownership gap | alert has no accountable responder | name owner and escalation target |

## 8. Interview checks

1. **Why is schema insufficient?** It cannot state grain, business semantics or quality objective.
2. **How is completeness computed?** Received divided by an agreed expected count.
3. **What makes a change breaking?** A declared consumer cannot safely interpret or process it.
4. **How handle duplicates?** Use declared identity and ordering/idempotency rules.
5. **What proves quality?** Contract-specific automated checks plus reconciliation, not a single pass.

## 9. Quick Reference Card

Before relying on data, record: owner; grain; key and mutation rule; fields/units/time; quality SLOs and denominators; delivery/replay contract; consumers; change/rollback process; access and retention constraints. Verify against real source and deployed platform behavior.

**Limitations:** Illustrative calculations; no production dataset or source lesson was measured.
