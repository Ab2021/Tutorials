---
collection: "experimentation-causal-inference-and-decision-science"
file_id: "01"
topic: "Experiment design and guardrails"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# Experiment Design and Guardrails

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) was consulted as inventory only; this draft does not reconstruct unavailable source content.

## 1. Decision first

State intervention, eligible unit, randomization unit, assignment time, primary outcome, guardrails, duration, stopping/analysis plan and owner. A metric movement without a decision rule is not an experiment conclusion.

## 2. Estimand and assignment

For an intent-to-treat comparison, estimate mean outcome under assignment to treatment versus control for declared eligible units. Randomize at the unit where interference is controlled: user, account, session, cluster or geography. Do not treat correlated or cross-exposed units as independent without a stated design.

## 3. Worked estimate

If treatment conversion is `120/2,000 = 0.06` and control conversion is `100/2,000 = 0.05`, absolute lift is `0.01` (one percentage point) and relative lift is `0.01/0.05 = 0.20` (20%). This does not establish significance, causal validity beyond assignment, customer value or safety.

## 4. Guardrails

Predeclare metrics that block or condition rollout: errors, latency, complaints, refunds, safety events, fairness/segment outcomes, revenue/cost and operational load. Check sample ratio mismatch, logging quality, exposure/compliance, novelty and contamination before interpreting lift.

## 5. Failure modes

| Failure | Signal | Containment |
|---|---|---|
| Assignment/log mismatch | sample ratio or exposure anomaly | pause/repair instrumentation |
| Peeking/optional stopping | repeated unplanned decisions | predeclare method or use sequential design |
| Interference | control affected by treatment | redesign unit/cluster or qualify claim |
| Metric gaming | primary lift with guardrail harm | block rollout and investigate |
| Nonrepresentative rollout | cohort differs from intended population | define eligibility and segment checks |

## 6. Quick Reference Card

Declare intervention, eligible/randomized unit, estimand, metrics, guardrails, duration, analysis/stopping plan, exposure/log validation, decision authority and rollout/rollback condition.

**Limitations:** Arithmetic is illustrative; no test, population or business decision was measured.
