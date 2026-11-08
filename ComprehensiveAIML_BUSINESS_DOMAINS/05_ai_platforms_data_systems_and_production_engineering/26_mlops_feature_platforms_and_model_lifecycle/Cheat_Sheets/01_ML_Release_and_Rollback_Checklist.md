---
collection: "mlops-feature-platforms-and-model-lifecycle"
file_id: "01"
topic: "ML release and rollback checklist"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# ML Release and Rollback Checklist

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) is an inventory. This course-topic draft does not reconstruct unavailable source lessons or measured operational outcomes.

## 1. Release is a versioned bundle

A deployable ML release names model artifact, code, feature definitions, data/schema contract, prompt/template where applicable, retrieval index, policy/guardrail configuration, runtime image, evaluation evidence and owner. A model version alone cannot recreate or safely undo a dependency-changing release.

## 2. Promotion gates

| Gate | Evidence | Reject when |
|---|---|---|
| Reproducibility | immutable manifest and artifact hash | dependencies/data snapshot unknown |
| Compatibility | schema/API/model-input tests | producer/consumer contract breaks |
| Quality | frozen evaluation and slice thresholds | baseline regresses beyond declared bound |
| Safety | abuse, privacy and policy tests | known critical control fails |
| Operations | latency/cost/capacity and alert runbook | SLO or on-call ownership absent |

## 3. Rollout path

Promote only through a declared path: offline evaluation → side-effect-free shadow → bounded canary → staged expansion → full release. Shadow output must not trigger external effects. Canary assignment, success metric, minimum sample/time, stop condition and responsible operator are release metadata.

## 4. Rollback invariant

Rollback restores a coherent previously approved bundle, or contains impact using an explicit degraded mode. It is not merely changing a model alias when feature schema, prompt, index, tool behavior or policy changed. Test rollback before relying on it.

## 5. Worked availability budget

For a 30-day period, 99.9% availability permits `30 × 24 × 60 × (1−0.999) = 43.2` minutes unavailable. This is an error-budget arithmetic example; whether maintenance, partial outage or dependency loss counts depends on the stated SLO.

## 6. Release record

Record manifest ID, approver, source/train/eval time boundaries, metrics and slices, dependency versions, rollout cohort, monitoring queries, stop/rollback triggers, customer communication and post-release review owner. Retain release evidence according to policy and privacy constraints.

## 7. Failure modes

| Failure | Signal | Containment |
|---|---|---|
| Model-only rollback | behavior remains changed | restore full compatible bundle |
| Canary not representative | production slice regressions | cohort rules and slice monitoring |
| Shadow has side effect | duplicate writes/actions | enforce dry-run boundary |
| Nonreproducible artifact | cannot rebuild/inspect release | immutable manifest and registry |
| Alert without owner | breach persists | named responder/runbook/escalation |

## 8. Interview checks

1. **Why bundle releases?** Runtime behavior depends on more than weights.
2. **What makes shadow safe?** It cannot cause the real side effect.
3. **Why can a canary mislead?** The cohort may not represent exposed traffic or harms.
4. **What proves rollback works?** A exercised rollback drill restoring compatible dependencies.
5. **What is an error budget?** Allowed unreliability under an explicitly defined SLO.

## 9. Quick Reference Card

Before promotion: freeze manifest; verify contracts, quality/slices and controls; define rollout cohort, stop condition, monitoring and owner; prove coherent rollback/degraded mode; preserve evidence. Re-approve if any bundle component changes.

**Limitations:** Availability calculation is illustrative; no release system, customer impact or source result is claimed.
