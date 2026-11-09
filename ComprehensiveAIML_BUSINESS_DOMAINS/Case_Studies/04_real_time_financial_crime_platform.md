---
case_id: "CASE-XD-004"
case_title: "Real-time financial crime platform"
case_version: "2.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
primary_pattern: "delayed-feedback"
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "financial-crime operations lead"
source_index: "INDEX.md"
---

# Real-time financial crime platform

**Evidence declaration:** Illustrative teaching design. Every number, trace, record, and outcome below is constructed; no organization, deployment, benefit, safety result, or incident is claimed.
**Source boundary:** directory.txt and the source crosswalk establish only topic provenance. Unavailable originals remain unavailable.

## 1. The One-Line Framing
Given payment event and event-time account context, decide allow, step-up, block, or investigation queue for financial-crime operations lead, subject to review capacity, latency, and delayed adjudication, compared with static rules and manual triage.
The governing pattern is delayed-feedback; a candidate cannot grant itself permission or claim completed external state.
Success is a reconstructible bounded disposition, not a realized business outcome.

## 2. Context & Background
The workflow starts with payment event and event-time account context and ends when hold/release receipt exists or static rules and manual triage accepts handoff.
financial-crime operations lead owns the stop control and the exception contract.
The primary downside is unnecessary customer friction; raw completion count is not the objective.
All timestamps are UTC and the replay population is constructed.
Post-decision amendments are evaluation-only and may not leak into action-time inputs.

## 3. The Business Question
Which allow, step-up, block, or investigation queue disposition is justified for one decision record?
Available actions are baseline, bounded recommendation, review, abstention, or stop.
The objective is useful eligible decisions; a proxy is invalid if it excludes difficult records.
Out of scope are legal, clinical, hardware, security, and deployment claims.

## 4. Assumptions & Constraints
account snapshot is available at decision time; otherwise abstain and rebind.
payment authorization has a trusted availability timestamp; otherwise exclude it.
case disposition is current and interpretable; otherwise freeze automation.
A review slot is explicitly reserved; otherwise do not promise review.
hold/release receipt is reconcilable; otherwise hold before retry.
The binding constraint is review capacity, latency, and delayed adjudication; breach rejects automatic disposition.
Outcome evidence is unavailable, so implementation remains proposed.

## 5. Data, Tools & Method
### 5a. Data Specification
Unit: CASE-XD-004-decision_id; repeated events retain immutable sequence.
Inputs: account snapshot, payment authorization, case disposition, request time, schema version, and availability time.
Output: allow, step-up, block, or investigation queue, reason code, bundle ID, and hold/release receipt.
Join: one decision joins one as-of gate; evidence fan-out is explicit.
### 5b. Tools & Stack
Proposed components: schema validator, immutable snapshots, gate evaluator, review queue, audit sink, reconciliation worker.
Candidate generation cannot mutate hold/release receipt.
### 5c. Method Summary
validate -> bind as-of inputs -> evaluate case disposition -> assemble evidence -> candidate -> persist intent -> reconcile receipt.
Unconditional automation is considered and rejected because it cannot establish account snapshot.
Review-everything is considered and rejected when capacity is finite.
### 5d. Provenance, Quality, Exclusions and Leakage Controls
Require available_at <= decision_at; missing authority is a rejected state, never imputed.
Original source bodies are unavailable.
### 5e. Reproducibility and Method-Selection Ledger
RUN-004-01 is the executable constructed oracle in Section 9; it tests logic, not performance.

## 6. Analysis Walkthrough
The decisive trace tests delayed-feedback before candidate quality.
Parse input and reject unknown fields.
Bind account snapshot and case disposition by availability time.
Enumerate payment authorization without collapsing cardinality.
Persist intent before any external transition.
On timeout query hold/release receipt; never blind retry.
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
Recommend a proposed delayed-feedback workflow only when account snapshot, payment authorization, case disposition, and durable state agree.
The 32/80 calculation is arithmetic only, not measured benefit.
Missing context, incompatible state, unresolved receipts, or no review capacity reverse the decision to baseline or escalation.

## 8. System/Solution Design
### 8a. Offline / Training / Batch Path
Freeze payment authorization, account snapshot, and case disposition with source and availability time.
Replay baseline, gate-only, and candidate-assisted paths on identical eligibility.
Publish rules, schema, oracle, exclusions, and unresolved rows.
### 8b. Online / Serving / Real-Time Path
[input] -> [validate/as-of bind] -> [case disposition] -> [candidate] -> [persist intent] -> [hold/release receipt]
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
Least privilege applies to account snapshot.
Stop control routes work to static rules and manual triage.

## 9. Evaluation & Validation
Metric: eligible conforming dispositions divided by eligible records in a frozen UTC window.
Comparator: static rules and manual triage on identical eligible inputs.
Guardrails: unnecessary customer friction, stale-version count, unknown receipts, and review delay.
Pass/fail: any automatic disposition without account snapshot or case disposition fails.
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
No deployment was taken; this is a proposed replay and governance specification for financial-crime operations lead.
No user, asset, patient, payment, shipment, tenant, or device changed.
Proposed sequence: oracle replay, adversarial review, owner-approved shadow, bounded pilot after applicable approval.
Stop: disable candidate path, retain audit, route to static rules and manual triage, reconcile pending receipts.

## 11. Outcome & Results
| Metric | Result | Evidence | Limitation |
|---|---|---|---|
| Primary useful outcome | not measured | illustrative design | no run |
| Gate conformity | not measured | proposed oracle | oracle is not deployment evidence |
| Constructed ratio | 32/80=0.40 | derived arithmetic | teaching counts |
Outcome remains not measured; retain the original denominator in any future evaluation.

## 12. Failure Modes & What Went Wrong

### F-004-stale account snapshot
- Competing causes: clock skew or revoked scope.
- Discriminating check: compare availability to decision time.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: withhold and rebind.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-004-conflicting payment authorization
- Competing causes: bad join or late correction.
- Discriminating check: compare IDs and cardinality.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: escalate.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-004-incompatible case disposition
- Competing causes: partial rollout or cache.
- Discriminating check: check signed bundle.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: pin compatible bundle.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-004-ambiguous hold/release receipt
- Competing causes: timeout or downstream commit.
- Discriminating check: lookup operation identity.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: reconcile before retry.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-004-review overload
- Competing causes: arrival burst or reviewer loss.
- Discriminating check: compare arrival, service, age.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: route baseline.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-004-proxy gaming
- Competing causes: denominator drift or threshold tuning.
- Discriminating check: audit eligibility.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: freeze definition.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.

## 13. Transferable Lesson
Pattern: delayed-feedback; bind authority and state before a candidate becomes a disposition.
Invariant: candidate quality cannot substitute for account snapshot, case disposition, or receipt reconciliation.
Prerequisites: named owner, versioned inputs, baseline handoff, durable audit, tested failure policy.
Counterexample: unreconcilable hold/release receipt requires narrower automation or human ownership.
Evidence limit: design reasoning and constructed arithmetic only.

## 14. Connections & Cross-References
### 14a. Related methods and open questions
- Course numerical standards define evidence labels and formula anatomy.
- Case master template defines decision, validation, and interview gates.
- Open question: which real artifact establishes that the gate is enforced?
### 14b. Interview pack — 15 answered questions
#### Q-01 — Frame the decision boundary
- Candidate prompt: Frame the decision boundary.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-02 — State the action-time cut-off
- Candidate prompt: State the action-time cut-off.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-03 — Why retain baseline
- Candidate prompt: Why retain baseline.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-04 — Compute the ratio
- Candidate prompt: Compute the ratio.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-05 — Diagnose good metric with rising harm
- Candidate prompt: Diagnose good metric with rising harm.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-06 — Handle unknown receipt
- Candidate prompt: Handle unknown receipt.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-07 — Respond to stale context
- Candidate prompt: Respond to stale context.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-08 — Change capacity constraint
- Candidate prompt: Change capacity constraint.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-09 — Compare score-only alternative
- Candidate prompt: Compare score-only alternative.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-10 — Design audit record
- Candidate prompt: Design audit record.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-11 — Explain failure injection
- Candidate prompt: Explain failure injection.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-12 — State pilot gate
- Candidate prompt: State pilot gate.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-13 — Identify leakage
- Candidate prompt: Identify leakage.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-14 — Give automation counterexample
- Candidate prompt: Give automation counterexample.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-15 — Falsify the proposal
- Candidate prompt: Falsify the proposal.
- Checked answer: bind account snapshot, payment authorization, case disposition, and hold/release receipt before releasing allow, step-up, block, or investigation queue.
- Reasoning: delayed-feedback fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how static rules and manual triage handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.

## 15. Quick Reference Card
| Field | Value |
|---|---|
| Case ID | CASE-XD-004 |
| PLAN identity | PLAN-XD-004 preserved |
| Owner | financial-crime operations lead |
| Input/output | payment event and event-time account context -> allow, step-up, block, or investigation queue |
| Pattern | delayed-feedback |
| Baseline | static rules and manual triage |
| Constraint | review capacity, latency, and delayed adjudication |
| Evidence/stage | illustrative; not measured; proposed |
| Oracle | RUN-004-01 in Section 9 |
- Control trace 345: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 346: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 347: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 348: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 349: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 350: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 351: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 352: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 353: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 354: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 355: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 356: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 357: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 358: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 359: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 360: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 361: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 362: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 363: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 364: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 365: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 366: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 367: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 368: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 369: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 370: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 371: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 372: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 373: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 374: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 375: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 376: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 377: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 378: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 379: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 380: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 381: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 382: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 383: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 384: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 385: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 386: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 387: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 388: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 389: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 390: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 391: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 392: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 393: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 394: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 395: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 396: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 397: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 398: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 399: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 400: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 401: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 402: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 403: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 404: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 405: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 406: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 407: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 408: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 409: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 410: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 411: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 412: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 413: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 414: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 415: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 416: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 417: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 418: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 419: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 420: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 421: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 422: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 423: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 424: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 425: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 426: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 427: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 428: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 429: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 430: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 431: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 432: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 433: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 434: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 435: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 436: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 437: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 438: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 439: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 440: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 441: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 442: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 443: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 444: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 445: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 446: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 447: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 448: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 449: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 450: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 451: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 452: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 453: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 454: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 455: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 456: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 457: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 458: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 459: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 460: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 461: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 462: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 463: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 464: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 465: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 466: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 467: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 468: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 469: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 470: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 471: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 472: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 473: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 474: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 475: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 476: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 477: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 478: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 479: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 480: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 481: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 482: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 483: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 484: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 485: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 486: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 487: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 488: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 489: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 490: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 491: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 492: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 493: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 494: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 495: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 496: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 497: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 498: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 499: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 500: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 501: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 502: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 503: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 504: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 505: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 506: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 507: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 508: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 509: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 510: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 511: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 512: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 513: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 514: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 515: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 516: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 517: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 518: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 519: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 520: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 521: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 522: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 523: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 524: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 525: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 526: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 527: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 528: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 529: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 530: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 531: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 532: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 533: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 534: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 535: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 536: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 537: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 538: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 539: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 540: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 541: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 542: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 543: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 544: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 545: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 546: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 547: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 548: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 549: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 550: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 551: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 552: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 553: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 554: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 555: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 556: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 557: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 558: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 559: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 560: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 561: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 562: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 563: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 564: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 565: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 566: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 567: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 568: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 569: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 570: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 571: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 572: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 573: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 574: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 575: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 576: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 577: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 578: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 579: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 580: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 581: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 582: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 583: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 584: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 585: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 586: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 587: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 588: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 589: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 590: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 591: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 592: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 593: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 594: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 595: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 596: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 597: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 598: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 599: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 600: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 601: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 602: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 603: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 604: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 605: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 606: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 607: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 608: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 609: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 610: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 611: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 612: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 613: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 614: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 615: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 616: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 617: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 618: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 619: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 620: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 621: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 622: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 623: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 624: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 625: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 626: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 627: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 628: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 629: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 630: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 631: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 632: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 633: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 634: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 635: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 636: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 637: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 638: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 639: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 640: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 641: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 642: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 643: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 644: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 645: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 646: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 647: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 648: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 649: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 650: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 651: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 652: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 653: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 654: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 655: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 656: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 657: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 658: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 659: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 660: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 661: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 662: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 663: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 664: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 665: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 666: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 667: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 668: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 669: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 670: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 671: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 672: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 673: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 674: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 675: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 676: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 677: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 678: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 679: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 680: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 681: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 682: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 683: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 684: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 685: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 686: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 687: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 688: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 689: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 690: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 691: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 692: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 693: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 694: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 695: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 696: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 697: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 698: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 699: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 700: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 701: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 702: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 703: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 704: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 705: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 706: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 707: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 708: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 709: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 710: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 711: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 712: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 713: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 714: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 715: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 716: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 717: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 718: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 719: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 720: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 721: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 722: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 723: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 724: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 725: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 726: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 727: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 728: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 729: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 730: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 731: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 732: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 733: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 734: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 735: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 736: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 737: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 738: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 739: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 740: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 741: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 742: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 743: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 744: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 745: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 746: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 747: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 748: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 749: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 750: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 751: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 752: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 753: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 754: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 755: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 756: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 757: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 758: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 759: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 760: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 761: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 762: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 763: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 764: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 765: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 766: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 767: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 768: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 769: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 770: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 771: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 772: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 773: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 774: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 775: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 776: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 777: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 778: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 779: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 780: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 781: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 782: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 783: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 784: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 785: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 786: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 787: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 788: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 789: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 790: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 791: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 792: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 793: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 794: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 795: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 796: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 797: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 798: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 799: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 800: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 801: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 802: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 803: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 804: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 805: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 806: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 807: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 808: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 809: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 810: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 811: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 812: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 813: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 814: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 815: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 816: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 817: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 818: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 819: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 820: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 821: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 822: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 823: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 824: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 825: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 826: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 827: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 828: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 829: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 830: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 831: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 832: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 833: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 834: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 835: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 836: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 837: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 838: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 839: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 840: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 841: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 842: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 843: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 844: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 845: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 846: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 847: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 848: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 849: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 850: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 851: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 852: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 853: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 854: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 855: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 856: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 857: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 858: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 859: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 860: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 861: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 862: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 863: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 864: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 865: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 866: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 867: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 868: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 869: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 870: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 871: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 872: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 873: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 874: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 875: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 876: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 877: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 878: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 879: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 880: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 881: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 882: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 883: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 884: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 885: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 886: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 887: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 888: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 889: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 890: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 891: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 892: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 893: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 894: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 895: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 896: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 897: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 898: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 899: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 900: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 901: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 902: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 903: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 904: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 905: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 906: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 907: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 908: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 909: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 910: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 911: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 912: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 913: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 914: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 915: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 916: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 917: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 918: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 919: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 920: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 921: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 922: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 923: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 924: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 925: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 926: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 927: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 928: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 929: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 930: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 931: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 932: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 933: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 934: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 935: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 936: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 937: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 938: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 939: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 940: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 941: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 942: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 943: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 944: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 945: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 946: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 947: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 948: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 949: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 950: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 951: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 952: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 953: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 954: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 955: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 956: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 957: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 958: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 959: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 960: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 961: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 962: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 963: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 964: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 965: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 966: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 967: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 968: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 969: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 970: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 971: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 972: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 973: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 974: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 975: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 976: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 977: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 978: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 979: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 980: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 981: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 982: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 983: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 984: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 985: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 986: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 987: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 988: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 989: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 990: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 991: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 992: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 993: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 994: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 995: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 996: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 997: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 998: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 999: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 1000: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 1001: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 1002: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 1003: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
- Control trace 1004: financial-crime operations lead verifies account snapshot, payment authorization, case disposition, and hold/release receipt in the same decision record; a mismatch routes to static rules and manual triage and preserves the reason code.
