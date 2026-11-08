---
collection: "data-quality-observability-and-reliability-engineering"
file_id: "01"
topic: "Data observability and incident triage"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# Data Observability and Incident Triage

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) is source inventory only. This draft does not claim an observed incident or production SLO.

## 1. Observability contract

For each data product, name owner, consumers, grain/key, freshness point, expected volume, schema/semantic checks, lineage, SLO, alert routing and safe degraded behavior. A dashboard alone is not observability if it cannot identify contract breach and owner.

## 2. Freshness

Freshness lag is `now − latest valid event/partition time` under declared event/processing-time semantics. If data is due every 15 minutes and the maximum allowed lag is 30 minutes, alert after two missed intervals only when late arrival and clock definitions are explicit.

## 3. Incident triage

1. Bound impact: consumers, decisions and time range.
2. Validate source arrival, schema, volume and pipeline execution.
3. Compare lineage and last known-good version.
4. Contain: pause, quarantine, backfill or use declared degraded path.
5. Reconcile outputs/actions, restore, then record root cause and preventive control.

## 4. Failure modes

| Failure | Signal | Containment |
|---|---|---|
| silent null expansion | required-field/null threshold | quarantine and trace producer |
| late data treated current | freshness breach | watermark/reprocess/reconcile |
| semantic change parses | metric shift with schema pass | contract version and consumer review |
| alert fatigue | many unactionable pages | tune to decision/owner, preserve detection |
| bad backfill | historical/recent data conflict | versioned replay and reconciliation |

## 5. Quick Reference Card

Declare grain/key/time semantics, owner/consumers, lineage, schema/volume/freshness checks, SLO, alert receiver, containment/degraded mode, replay/reconciliation and post-incident evidence.

**Limitations:** Interval example is illustrative; no operational dataset or SLO was measured.
