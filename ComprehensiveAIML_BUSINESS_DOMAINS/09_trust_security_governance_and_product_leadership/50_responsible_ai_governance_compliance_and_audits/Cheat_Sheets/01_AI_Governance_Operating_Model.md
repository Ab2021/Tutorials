---
collection: "responsible-ai-governance-compliance-and-audits"
file_id: "01"
topic: "AI governance operating model"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# AI Governance Operating Model

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) was consulted as inventory only. This is not legal advice, compliance certification, or reconstruction of unavailable source content.

## 1. Governance is an operating control

Assign accountable owner, decision authority, risk tier, required evidence, approval gate, monitoring, incident route and change trigger for each AI system. A policy without an owner, enforceable gate and evidence location is aspiration, not an operating control.

## 2. Risk tiering

Classify intended use, affected people, autonomy, decision consequence, data sensitivity, external dependencies, security exposure and reversibility. Tiering determines proportional controls; it does not establish legal status. Re-assess after material changes to model, data, use, jurisdiction, integrations or user population.

## 3. Evidence record

| Record | Minimum content |
|---|---|
| System inventory | owner, purpose, users, version, dependencies |
| Impact assessment | harms, affected groups, controls, residual risks |
| Evaluation | task, data boundary, metrics, slices, limitations |
| Decision log | approver, rationale, conditions, expiry |
| Change log | component, reason, assessment and rollback |
| Incident record | detection, impact, response, notification and lessons |

## 4. Control lifecycle

Intake → risk assessment → control design → evidence review → bounded release → monitoring → change/incident reassessment → retirement. Do not permit a generic approval to cover new uses or material dependency changes.

## 5. Failure modes

| Failure | Signal | Containment |
|---|---|---|
| No accountable owner | review/incident stalls | assign decision owner and escalation |
| Stale assessment | system differs from record | change gate and periodic review |
| Evidence inaccessible | audit cannot reproduce claim | versioned, access-controlled repository |
| Tier underestimated | high-consequence use bypasses controls | independent challenge/escalation |
| Metric-only assurance | harms appear outside benchmark | operational, security and user monitoring |

## 6. Interview checks

1. **Why inventory systems?** You cannot govern unknown deployed uses.
2. **What triggers reassessment?** A material change in use, data, model, dependency or context.
3. **What is residual risk?** Risk remaining after declared controls.
4. **Why expiry on approval?** Conditions and evidence become stale.
5. **What proves governance?** Traceable decisions and operating evidence, not a policy document alone.

## 7. Quick Reference Card

Name owner and use; classify risk; preserve impact/evaluation/decision evidence; gate release; monitor controls; record changes/incidents; set review/expiry; retire data, access and dependencies deliberately.

**Limitations:** No regulation, jurisdiction, organization or system was evaluated.
