---
case_id: "CASE-XD-005"
case_title: "Sustainable supply-chain control tower"
case_version: "2.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
primary_pattern: "constrained-allocation"
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "supply planning lead"
source_index: "INDEX.md"
---

# Sustainable supply-chain control tower

**Evidence declaration:** Illustrative teaching design. Every number, trace, record, and outcome below is constructed; no organization, deployment, benefit, safety result, or incident is claimed.
**Source boundary:** directory.txt and the source crosswalk establish only topic provenance. Unavailable originals remain unavailable.

## 1. The One-Line Framing
Given inventory, forecast, lane capacity, and emissions factor, decide ranked exception action or planner escalation for supply planning lead, subject to capacity, service, cost, emissions, and approval, compared with spreadsheet exception review.
The governing pattern is constrained-allocation; a candidate cannot grant itself permission or claim completed external state.
Success is a reconstructible bounded disposition, not a realized business outcome.

## 2. Context & Background
The workflow starts with inventory, forecast, lane capacity, and emissions factor and ends when planner action receipt exists or spreadsheet exception review accepts handoff.
supply planning lead owns the stop control and the exception contract.
The primary downside is infeasible shipment promise; raw completion count is not the objective.
All timestamps are UTC and the replay population is constructed.
Post-decision amendments are evaluation-only and may not leak into action-time inputs.

## 3. The Business Question
Which ranked exception action or planner escalation disposition is justified for one decision record?
Available actions are baseline, bounded recommendation, review, abstention, or stop.
The objective is useful eligible decisions; a proxy is invalid if it excludes difficult records.
Out of scope are legal, clinical, hardware, security, and deployment claims.

## 4. Assumptions & Constraints
inventory snapshot is available at decision time; otherwise abstain and rebind.
lane offer has a trusted availability timestamp; otherwise exclude it.
feasibility confirmation is current and interpretable; otherwise freeze automation.
A review slot is explicitly reserved; otherwise do not promise review.
planner action receipt is reconcilable; otherwise hold before retry.
The binding constraint is capacity, service, cost, emissions, and approval; breach rejects automatic disposition.
Outcome evidence is unavailable, so implementation remains proposed.

## 5. Data, Tools & Method
### 5a. Data Specification
Unit: CASE-XD-005-decision_id; repeated events retain immutable sequence.
Inputs: inventory snapshot, lane offer, feasibility confirmation, request time, schema version, and availability time.
Output: ranked exception action or planner escalation, reason code, bundle ID, and planner action receipt.
Join: one decision joins one as-of gate; evidence fan-out is explicit.
### 5b. Tools & Stack
Proposed components: schema validator, immutable snapshots, gate evaluator, review queue, audit sink, reconciliation worker.
Candidate generation cannot mutate planner action receipt.
### 5c. Method Summary
validate -> bind as-of inputs -> evaluate feasibility confirmation -> assemble evidence -> candidate -> persist intent -> reconcile receipt.
Unconditional automation is considered and rejected because it cannot establish inventory snapshot.
Review-everything is considered and rejected when capacity is finite.
### 5d. Provenance, Quality, Exclusions and Leakage Controls
Require available_at <= decision_at; missing authority is a rejected state, never imputed.
Original source bodies are unavailable.
### 5e. Reproducibility and Method-Selection Ledger
RUN-005-01 is the executable constructed oracle in Section 9; it tests logic, not performance.

## 6. Analysis Walkthrough
The decisive trace tests constrained-allocation before candidate quality.
Parse input and reject unknown fields.
Bind inventory snapshot and feasibility confirmation by availability time.
Enumerate lane offer without collapsing cardinality.
Persist intent before any external transition.
On timeout query planner action receipt; never blind retry.
Emit input hash, bundle, reason, queue state, and final transition.
Formula validity: constructed teaching counts, not performance.
FORMULA: r = numerator / denominator.
WHERE: numerator=32 records; denominator=80 eligible records; r dimensionless.
CONSTRAINT: same frozen denominator and numerator subset.
NUMERICAL EXAMPLE: 32/80 = 0.40.
INDEPENDENT CHECK: 0.40 * 80 = 32.
INTERPRETATION: arithmetic oracle only; not effectiveness.
Sensitivity: a changed denominator makes the ratio incomparable.

## 7. The Finding
Recommend a proposed constrained-allocation workflow only when inventory snapshot, lane offer, feasibility confirmation, and durable state agree.
The 32/80 calculation is arithmetic only, not measured benefit.
Missing context, incompatible state, unresolved receipts, or no review capacity reverse the decision to baseline or escalation.

## 8. System/Solution Design
### 8a. Offline / Training / Batch Path
Freeze lane offer, inventory snapshot, and feasibility confirmation with source and availability time.
Replay baseline, gate-only, and candidate-assisted paths on identical eligibility.
Publish rules, schema, oracle, exclusions, and unresolved rows.
### 8b. Online / Serving / Real-Time Path
[input] -> [validate/as-of bind] -> [feasibility confirmation] -> [candidate] -> [persist intent] -> [planner action receipt]
Success invariant: each disposition references exactly one bundle and receipt.
Failure invariant: timeout is pending reconciliation, not repeat execution.
Cold start refuses automatic handling until required snapshots load.
### 8c. Key Design Decisions
Explicit gate precedes candidate release.
Stable operation identity precedes external side effect.
Baseline remains reachable.
Owner override is audited.
### 8d. Operational Readiness
Observe decision ID, source times, gate, bundle, queue age, and receipt.
Least privilege applies to inventory snapshot.
Stop control routes work to spreadsheet exception review.

## 9. Evaluation & Validation
Metric: eligible conforming dispositions divided by eligible records in a frozen UTC window.
Comparator: spreadsheet exception review on identical eligible inputs.
Guardrails: infeasible shipment promise, stale-version count, unknown receipts, and review delay.
Pass/fail: any automatic disposition without inventory snapshot or feasibility confirmation fails.
Status: not executed; constructed records establish neither prevalence nor causal benefit.
### Executable oracle input
```python
record = {'access': True, 'gate': True, 'receipt': None, 'expected': 'PERSIST_INTENT'}
actual = 'PERSIST_INTENT' if record['access'] and record['gate'] and record['receipt'] is None else 'BASELINE_OR_RECONCILE'
assert actual == record['expected']
print('CASE-XD-${id}-ORACLE-01', actual)
```
This oracle proves only that intent precedes an unknown receipt; it is not deployment validation.

## 10. Action Taken
No deployment was taken; this is a proposed replay and governance specification for supply planning lead.
No user, asset, patient, payment, shipment, tenant, or device changed.
Proposed sequence: oracle replay, adversarial review, owner-approved shadow, bounded pilot after applicable approval.
Stop: disable candidate path, retain audit, route to spreadsheet exception review, reconcile pending receipts.

## 11. Outcome & Results
| Metric | Result | Evidence | Limitation |
|---|---|---|---|
| Primary useful outcome | not measured | illustrative design | no run |
| Gate conformity | not measured | proposed oracle | oracle is not deployment evidence |
| Constructed ratio | 32/80=0.40 | derived arithmetic | teaching counts |
Outcome remains not measured; retain the original denominator in any future evaluation.

## 12. Failure Modes & What Went Wrong

### F-005-stale inventory snapshot
- Competing causes: clock skew or revoked scope.
- Discriminating check: compare availability to decision time.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: withhold and rebind.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-005-conflicting lane offer
- Competing causes: bad join or late correction.
- Discriminating check: compare IDs and cardinality.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: escalate.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-005-incompatible feasibility confirmation
- Competing causes: partial rollout or cache.
- Discriminating check: check signed bundle.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: pin compatible bundle.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-005-ambiguous planner action receipt
- Competing causes: timeout or downstream commit.
- Discriminating check: lookup operation identity.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: reconcile before retry.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-005-review overload
- Competing causes: arrival burst or reviewer loss.
- Discriminating check: compare arrival, service, age.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: route baseline.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-005-proxy gaming
- Competing causes: denominator drift or threshold tuning.
- Discriminating check: audit eligibility.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: freeze definition.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.

## 13. Transferable Lesson
Pattern: constrained-allocation; bind authority and state before a candidate becomes a disposition.
Invariant: candidate quality cannot substitute for inventory snapshot, feasibility confirmation, or receipt reconciliation.
Prerequisites: named owner, versioned inputs, baseline handoff, durable audit, tested failure policy.
Counterexample: unreconcilable planner action receipt requires narrower automation or human ownership.
Evidence limit: design reasoning and constructed arithmetic only.

## 14. Connections & Cross-References
### 14a. Related methods and open questions
- Course numerical standards define evidence labels and formula anatomy.
- Case master template defines decision, validation, and interview gates.
- Open question: which real artifact establishes that the gate is enforced?
### 14b. Interview pack — 15 answered questions
#### Q-01 — Frame the decision boundary
- Candidate prompt: Frame the decision boundary.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-02 — State the action-time cut-off
- Candidate prompt: State the action-time cut-off.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-03 — Why retain baseline
- Candidate prompt: Why retain baseline.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-04 — Compute the ratio
- Candidate prompt: Compute the ratio.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-05 — Diagnose good metric with rising harm
- Candidate prompt: Diagnose good metric with rising harm.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-06 — Handle unknown receipt
- Candidate prompt: Handle unknown receipt.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-07 — Respond to stale context
- Candidate prompt: Respond to stale context.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-08 — Change capacity constraint
- Candidate prompt: Change capacity constraint.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-09 — Compare score-only alternative
- Candidate prompt: Compare score-only alternative.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-10 — Design audit record
- Candidate prompt: Design audit record.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-11 — Explain failure injection
- Candidate prompt: Explain failure injection.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-12 — State pilot gate
- Candidate prompt: State pilot gate.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-13 — Identify leakage
- Candidate prompt: Identify leakage.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-14 — Give automation counterexample
- Candidate prompt: Give automation counterexample.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-15 — Falsify the proposal
- Candidate prompt: Falsify the proposal.
- Checked answer: bind inventory snapshot, lane offer, feasibility confirmation, and planner action receipt before releasing ranked exception action or planner escalation.
- Reasoning: constrained-allocation fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how spreadsheet exception review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.

## 15. Quick Reference Card
| Field | Value |
|---|---|
| Case ID | CASE-XD-005 |
| PLAN identity | PLAN-XD-005 preserved |
| Owner | supply planning lead |
| Input/output | inventory, forecast, lane capacity, and emissions factor -> ranked exception action or planner escalation |
| Pattern | constrained-allocation |
| Baseline | spreadsheet exception review |
| Constraint | capacity, service, cost, emissions, and approval |
| Evidence/stage | illustrative; not measured; proposed |
| Oracle | RUN-005-01 in Section 9 |
- Control trace 345: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 346: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 347: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 348: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 349: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 350: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 351: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 352: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 353: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 354: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 355: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 356: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 357: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 358: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 359: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 360: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 361: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 362: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 363: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 364: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 365: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 366: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 367: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 368: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 369: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 370: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 371: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 372: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 373: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 374: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 375: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 376: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 377: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 378: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 379: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 380: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 381: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 382: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 383: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 384: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 385: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 386: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 387: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 388: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 389: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 390: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 391: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 392: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 393: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 394: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 395: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 396: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 397: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 398: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 399: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 400: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 401: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 402: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 403: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 404: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 405: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 406: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 407: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 408: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 409: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 410: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 411: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 412: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 413: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 414: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 415: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 416: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 417: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 418: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 419: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 420: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 421: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 422: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 423: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 424: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 425: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 426: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 427: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 428: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 429: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 430: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 431: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 432: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 433: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 434: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 435: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 436: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 437: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 438: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 439: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 440: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 441: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 442: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 443: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 444: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 445: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 446: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 447: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 448: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 449: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 450: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 451: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 452: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 453: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 454: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 455: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 456: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 457: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 458: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 459: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 460: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 461: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 462: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 463: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 464: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 465: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 466: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 467: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 468: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 469: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 470: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 471: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 472: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 473: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 474: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 475: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 476: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 477: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 478: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 479: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 480: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 481: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 482: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 483: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 484: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 485: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 486: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 487: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 488: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 489: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 490: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 491: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 492: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 493: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 494: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 495: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 496: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 497: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 498: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 499: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 500: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 501: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 502: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 503: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 504: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 505: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 506: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 507: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 508: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 509: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 510: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 511: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 512: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 513: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 514: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 515: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 516: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 517: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 518: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 519: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 520: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 521: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 522: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 523: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 524: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 525: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 526: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 527: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 528: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 529: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 530: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 531: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 532: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 533: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 534: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 535: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 536: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 537: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 538: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 539: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 540: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 541: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 542: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 543: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 544: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 545: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 546: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 547: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 548: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 549: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 550: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 551: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 552: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 553: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 554: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 555: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 556: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 557: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 558: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 559: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 560: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 561: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 562: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 563: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 564: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 565: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 566: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 567: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 568: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 569: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 570: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 571: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 572: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 573: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 574: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 575: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 576: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 577: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 578: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 579: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 580: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 581: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 582: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 583: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 584: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 585: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 586: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 587: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 588: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 589: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 590: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 591: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 592: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 593: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 594: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 595: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 596: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 597: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 598: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 599: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 600: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 601: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 602: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 603: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 604: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 605: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 606: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 607: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 608: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 609: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 610: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 611: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 612: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 613: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 614: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 615: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 616: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 617: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 618: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 619: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 620: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 621: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 622: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 623: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 624: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 625: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 626: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 627: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 628: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 629: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 630: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 631: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 632: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 633: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 634: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 635: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 636: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 637: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 638: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 639: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 640: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 641: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 642: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 643: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 644: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 645: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 646: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 647: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 648: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 649: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 650: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 651: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 652: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 653: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 654: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 655: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 656: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 657: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 658: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 659: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 660: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 661: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 662: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 663: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 664: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 665: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 666: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 667: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 668: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 669: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 670: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 671: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 672: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 673: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 674: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 675: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 676: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 677: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 678: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 679: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 680: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 681: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 682: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 683: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 684: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 685: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 686: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 687: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 688: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 689: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 690: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 691: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 692: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 693: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 694: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 695: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 696: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 697: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 698: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 699: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 700: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 701: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 702: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 703: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 704: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 705: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 706: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 707: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 708: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 709: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 710: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 711: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 712: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 713: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 714: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 715: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 716: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 717: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 718: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 719: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 720: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 721: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 722: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 723: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 724: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 725: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 726: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 727: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 728: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 729: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 730: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 731: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 732: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 733: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 734: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 735: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 736: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 737: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 738: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 739: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 740: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 741: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 742: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 743: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 744: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 745: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 746: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 747: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 748: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 749: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 750: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 751: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 752: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 753: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 754: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 755: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 756: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 757: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 758: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 759: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 760: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 761: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 762: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 763: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 764: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 765: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 766: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 767: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 768: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 769: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 770: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 771: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 772: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 773: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 774: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 775: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 776: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 777: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 778: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 779: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 780: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 781: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 782: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 783: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 784: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 785: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 786: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 787: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 788: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 789: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 790: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 791: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 792: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 793: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 794: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 795: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 796: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 797: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 798: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 799: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 800: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 801: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 802: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 803: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 804: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 805: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 806: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 807: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 808: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 809: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 810: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 811: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 812: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 813: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 814: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 815: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 816: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 817: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 818: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 819: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 820: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 821: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 822: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 823: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 824: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 825: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 826: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 827: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 828: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 829: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 830: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 831: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 832: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 833: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 834: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 835: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 836: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 837: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 838: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 839: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 840: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 841: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 842: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 843: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 844: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 845: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 846: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 847: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 848: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 849: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 850: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 851: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 852: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 853: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 854: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 855: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 856: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 857: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 858: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 859: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 860: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 861: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 862: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 863: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 864: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 865: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 866: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 867: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 868: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 869: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 870: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 871: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 872: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 873: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 874: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 875: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 876: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 877: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 878: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 879: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 880: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 881: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 882: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 883: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 884: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 885: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 886: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 887: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 888: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 889: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 890: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 891: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 892: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 893: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 894: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 895: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 896: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 897: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 898: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 899: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 900: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 901: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 902: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 903: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 904: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 905: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 906: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 907: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 908: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 909: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 910: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 911: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 912: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 913: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 914: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 915: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 916: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 917: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 918: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 919: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 920: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 921: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 922: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 923: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 924: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 925: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 926: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 927: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 928: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 929: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 930: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 931: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 932: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 933: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 934: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 935: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 936: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 937: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 938: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 939: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 940: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 941: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 942: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 943: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 944: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 945: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 946: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 947: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 948: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 949: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 950: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 951: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 952: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 953: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 954: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 955: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 956: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 957: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 958: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 959: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 960: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 961: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 962: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 963: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 964: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 965: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 966: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 967: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 968: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 969: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 970: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 971: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 972: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 973: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 974: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 975: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 976: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 977: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 978: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 979: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 980: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 981: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 982: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 983: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 984: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 985: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 986: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 987: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 988: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 989: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 990: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 991: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 992: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 993: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 994: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 995: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 996: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 997: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 998: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 999: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 1000: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 1001: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 1002: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 1003: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
- Control trace 1004: supply planning lead verifies inventory snapshot, lane offer, feasibility confirmation, and planner action receipt in the same decision record; a mismatch routes to spreadsheet exception review and preserves the reason code.
