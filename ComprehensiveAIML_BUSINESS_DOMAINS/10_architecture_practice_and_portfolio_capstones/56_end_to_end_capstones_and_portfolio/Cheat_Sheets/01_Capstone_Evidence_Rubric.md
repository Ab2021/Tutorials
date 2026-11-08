---
collection: "end-to-end-capstones-and-portfolio"
file_id: "01"
topic: "Capstone evidence rubric"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# Capstone Evidence Rubric

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) was consulted as inventory only. This draft does not claim a source-system build, deployment or measured portfolio result.

## 1. Deliverable contract

A credible capstone states a user/problem, decision, constraints, evidence boundary and non-goals. It distinguishes proposed design, simulated result and observed deployment result. Do not present a diagram, benchmark borrowed from another system or synthetic trace as operational proof.

## 2. Minimum evidence set

| Area | Required evidence |
|---|---|
| Problem | user, decision, baseline, success/failure criteria |
| Data | source/rights, grain, quality, split and limitations |
| Architecture | components, interfaces, state, dependency and failure paths |
| Evaluation | method, metrics, slices, oracle/trace and limitations |
| Operations | owner, monitoring, rollback, cost/capacity assumptions |
| Trust | risk, controls, review and residual-risk statement |

## 3. Rubric

Score each area 0–2: 0 absent/unfalsifiable; 1 stated but unverified; 2 evidence-backed and limitations explicit. Total `12` is a completeness aid, not a quality certificate; a zero in safety, evaluation or data evidence blocks a high-stakes release regardless of total.

## 4. Review procedure

1. Read the decision and evidence boundary first.
2. Trace one input through architecture, evaluation and operational recovery.
3. Challenge availability, ownership, data timing and claimed metrics.
4. Test a stated failure/rollback path or label it untested.
5. Record decisions, conditions and next evidence required.

## 5. Failure modes

| Failure | Check | Response |
|---|---|---|
| demo treated as production | missing operational evidence | label simulation and scope honestly |
| metric without decision | no threshold/cost/user | define action and trade-off |
| diagram-only architecture | interfaces/failure modes absent | add contracts and recovery paths |
| copied benchmark | provenance mismatch | cite and exclude from own result claim |
| unowned system | no responder/maintenance path | name owner and operating plan |

## 6. Quick Reference Card

Before presenting: state problem, decision, evidence boundary and non-goals; provide data/architecture/evaluation/operations/trust evidence; label simulations; include limitations and one falsifiable failure path.

**Limitations:** Rubric scores are illustrative and are not a hiring, safety or deployment certification.
