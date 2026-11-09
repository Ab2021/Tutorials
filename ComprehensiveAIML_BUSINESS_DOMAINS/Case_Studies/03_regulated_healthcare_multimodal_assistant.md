---
case_id: "CASE-XD-003"
case_title: "Regulated healthcare multimodal assistant"
case_version: "2.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
primary_pattern: "human-in-the-loop"
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "clinical governance lead"
source_index: "INDEX.md"
---

# Regulated healthcare multimodal assistant

**Evidence declaration:** Illustrative teaching design. Every number, trace, record, and outcome below is constructed; no organization, deployment, benefit, safety result, or incident is claimed.
**Source boundary:** directory.txt and the source crosswalk establish only topic provenance. Unavailable originals remain unavailable.

## 1. The One-Line Framing
Given clinician request and authorized encounter context, decide cited draft, abstention, or escalation packet for clinical governance lead, subject to clinician review; no diagnosis or order, compared with manual chart review.
The governing pattern is human-in-the-loop; a candidate cannot grant itself permission or claim completed external state.
Success is a reconstructible bounded disposition, not a realized business outcome.

## 2. Context & Background
The workflow starts with clinician request and authorized encounter context and ends when draft audit record exists or manual chart review accepts handoff.
clinical governance lead owns the stop control and the exception contract.
The primary downside is unsupported clinical statement; raw completion count is not the objective.
All timestamps are UTC and the replay population is constructed.
Post-decision amendments are evaluation-only and may not leak into action-time inputs.

## 3. The Business Question
Which cited draft, abstention, or escalation packet disposition is justified for one decision record?
Available actions are baseline, bounded recommendation, review, abstention, or stop.
The objective is useful eligible decisions; a proxy is invalid if it excludes difficult records.
Out of scope are legal, clinical, hardware, security, and deployment claims.

## 4. Assumptions & Constraints
encounter access grant is available at decision time; otherwise abstain and rebind.
note/image reference has a trusted availability timestamp; otherwise exclude it.
clinician sign-off is current and interpretable; otherwise freeze automation.
A review slot is explicitly reserved; otherwise do not promise review.
draft audit record is reconcilable; otherwise hold before retry.
The binding constraint is clinician review; no diagnosis or order; breach rejects automatic disposition.
Outcome evidence is unavailable, so implementation remains proposed.

## 5. Data, Tools & Method
### 5a. Data Specification
Unit: CASE-XD-003-decision_id; repeated events retain immutable sequence.
Inputs: encounter access grant, note/image reference, clinician sign-off, request time, schema version, and availability time.
Output: cited draft, abstention, or escalation packet, reason code, bundle ID, and draft audit record.
Join: one decision joins one as-of gate; evidence fan-out is explicit.
### 5b. Tools & Stack
Proposed components: schema validator, immutable snapshots, gate evaluator, review queue, audit sink, reconciliation worker.
Candidate generation cannot mutate draft audit record.
### 5c. Method Summary
validate -> bind as-of inputs -> evaluate clinician sign-off -> assemble evidence -> candidate -> persist intent -> reconcile receipt.
Unconditional automation is considered and rejected because it cannot establish encounter access grant.
Review-everything is considered and rejected when capacity is finite.
### 5d. Provenance, Quality, Exclusions and Leakage Controls
Require available_at <= decision_at; missing authority is a rejected state, never imputed.
Original source bodies are unavailable.
### 5e. Reproducibility and Method-Selection Ledger
RUN-003-01 is the executable constructed oracle in Section 9; it tests logic, not performance.

## 6. Analysis Walkthrough
The decisive trace tests human-in-the-loop before candidate quality.
Parse input and reject unknown fields.
Bind encounter access grant and clinician sign-off by availability time.
Enumerate note/image reference without collapsing cardinality.
Persist intent before any external transition.
On timeout query draft audit record; never blind retry.
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
Recommend a proposed human-in-the-loop workflow only when encounter access grant, note/image reference, clinician sign-off, and durable state agree.
The 32/80 calculation is arithmetic only, not measured benefit.
Missing context, incompatible state, unresolved receipts, or no review capacity reverse the decision to baseline or escalation.

## 8. System/Solution Design
### 8a. Offline / Training / Batch Path
Freeze note/image reference, encounter access grant, and clinician sign-off with source and availability time.
Replay baseline, gate-only, and candidate-assisted paths on identical eligibility.
Publish rules, schema, oracle, exclusions, and unresolved rows.
### 8b. Online / Serving / Real-Time Path
[input] -> [validate/as-of bind] -> [clinician sign-off] -> [candidate] -> [persist intent] -> [draft audit record]
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
Least privilege applies to encounter access grant.
Stop control routes work to manual chart review.

## 9. Evaluation & Validation
Metric: eligible conforming dispositions divided by eligible records in a frozen UTC window.
Comparator: manual chart review on identical eligible inputs.
Guardrails: unsupported clinical statement, stale-version count, unknown receipts, and review delay.
Pass/fail: any automatic disposition without encounter access grant or clinician sign-off fails.
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
No deployment was taken; this is a proposed replay and governance specification for clinical governance lead.
No user, asset, patient, payment, shipment, tenant, or device changed.
Proposed sequence: oracle replay, adversarial review, owner-approved shadow, bounded pilot after applicable approval.
Stop: disable candidate path, retain audit, route to manual chart review, reconcile pending receipts.

## 11. Outcome & Results
| Metric | Result | Evidence | Limitation |
|---|---|---|---|
| Primary useful outcome | not measured | illustrative design | no run |
| Gate conformity | not measured | proposed oracle | oracle is not deployment evidence |
| Constructed ratio | 32/80=0.40 | derived arithmetic | teaching counts |
Outcome remains not measured; retain the original denominator in any future evaluation.

## 12. Failure Modes & What Went Wrong

### F-003-stale encounter access grant
- Competing causes: clock skew or revoked scope.
- Discriminating check: compare availability to decision time.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: withhold and rebind.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-003-conflicting note/image reference
- Competing causes: bad join or late correction.
- Discriminating check: compare IDs and cardinality.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: escalate.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-003-incompatible clinician sign-off
- Competing causes: partial rollout or cache.
- Discriminating check: check signed bundle.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: pin compatible bundle.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-003-ambiguous draft audit record
- Competing causes: timeout or downstream commit.
- Discriminating check: lookup operation identity.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: reconcile before retry.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-003-review overload
- Competing causes: arrival burst or reviewer loss.
- Discriminating check: compare arrival, service, age.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: route baseline.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.
### F-003-proxy gaming
- Competing causes: denominator drift or threshold tuning.
- Discriminating check: audit eligibility.
- Evidence: hypothetical teaching failure, not observed incident.
- Containment: freeze definition.
- Prevention: versioned contract, negative-control replay, named alert owner.
- Verification boundary: reproduce constructed trigger; real behavior unmeasured.

## 13. Transferable Lesson
Pattern: human-in-the-loop; bind authority and state before a candidate becomes a disposition.
Invariant: candidate quality cannot substitute for encounter access grant, clinician sign-off, or receipt reconciliation.
Prerequisites: named owner, versioned inputs, baseline handoff, durable audit, tested failure policy.
Counterexample: unreconcilable draft audit record requires narrower automation or human ownership.
Evidence limit: design reasoning and constructed arithmetic only.

## 14. Connections & Cross-References
### 14a. Related methods and open questions
- Course numerical standards define evidence labels and formula anatomy.
- Case master template defines decision, validation, and interview gates.
- Open question: which real artifact establishes that the gate is enforced?
### 14b. Interview pack — 15 answered questions
#### Q-01 — Frame the decision boundary
- Candidate prompt: Frame the decision boundary.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-02 — State the action-time cut-off
- Candidate prompt: State the action-time cut-off.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-03 — Why retain baseline
- Candidate prompt: Why retain baseline.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-04 — Compute the ratio
- Candidate prompt: Compute the ratio.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-05 — Diagnose good metric with rising harm
- Candidate prompt: Diagnose good metric with rising harm.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-06 — Handle unknown receipt
- Candidate prompt: Handle unknown receipt.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-07 — Respond to stale context
- Candidate prompt: Respond to stale context.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-08 — Change capacity constraint
- Candidate prompt: Change capacity constraint.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-09 — Compare score-only alternative
- Candidate prompt: Compare score-only alternative.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-10 — Design audit record
- Candidate prompt: Design audit record.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-11 — Explain failure injection
- Candidate prompt: Explain failure injection.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-12 — State pilot gate
- Candidate prompt: State pilot gate.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-13 — Identify leakage
- Candidate prompt: Identify leakage.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-14 — Give automation counterexample
- Candidate prompt: Give automation counterexample.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.
#### Q-15 — Falsify the proposal
- Candidate prompt: Falsify the proposal.
- Checked answer: bind encounter access grant, note/image reference, clinician sign-off, and draft audit record before releasing cited draft, abstention, or escalation packet.
- Reasoning: human-in-the-loop fails closed when a decision-time prerequisite is absent.
- Follow-up: explain how manual chart review handles the missing prerequisite.
- Acceptable alternative: stricter owner-controlled response with the same evidence boundary.
- Red flag: treating candidate score or retry as permission.
- Scoring: 0 unsafe; 1 names control; 2 links data to failure; 3 defends trade-off.
- Answer support: Sections 4–6, 8–9, and failure register.

## 15. Quick Reference Card
| Field | Value |
|---|---|
| Case ID | CASE-XD-003 |
| PLAN identity | PLAN-XD-003 preserved |
| Owner | clinical governance lead |
| Input/output | clinician request and authorized encounter context -> cited draft, abstention, or escalation packet |
| Pattern | human-in-the-loop |
| Baseline | manual chart review |
| Constraint | clinician review; no diagnosis or order |
| Evidence/stage | illustrative; not measured; proposed |
| Oracle | RUN-003-01 in Section 9 |
- Control trace 345: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 346: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 347: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 348: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 349: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 350: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 351: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 352: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 353: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 354: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 355: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 356: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 357: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 358: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 359: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 360: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 361: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 362: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 363: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 364: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 365: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 366: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 367: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 368: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 369: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 370: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 371: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 372: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 373: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 374: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 375: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 376: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 377: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 378: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 379: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 380: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 381: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 382: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 383: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 384: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 385: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 386: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 387: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 388: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 389: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 390: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 391: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 392: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 393: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 394: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 395: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 396: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 397: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 398: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 399: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 400: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 401: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 402: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 403: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 404: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 405: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 406: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 407: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 408: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 409: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 410: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 411: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 412: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 413: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 414: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 415: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 416: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 417: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 418: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 419: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 420: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 421: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 422: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 423: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 424: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 425: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 426: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 427: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 428: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 429: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 430: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 431: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 432: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 433: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 434: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 435: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 436: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 437: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 438: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 439: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 440: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 441: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 442: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 443: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 444: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 445: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 446: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 447: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 448: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 449: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 450: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 451: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 452: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 453: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 454: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 455: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 456: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 457: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 458: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 459: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 460: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 461: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 462: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 463: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 464: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 465: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 466: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 467: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 468: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 469: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 470: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 471: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 472: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 473: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 474: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 475: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 476: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 477: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 478: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 479: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 480: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 481: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 482: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 483: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 484: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 485: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 486: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 487: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 488: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 489: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 490: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 491: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 492: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 493: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 494: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 495: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 496: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 497: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 498: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 499: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 500: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 501: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 502: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 503: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 504: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 505: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 506: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 507: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 508: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 509: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 510: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 511: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 512: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 513: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 514: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 515: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 516: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 517: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 518: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 519: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 520: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 521: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 522: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 523: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 524: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 525: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 526: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 527: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 528: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 529: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 530: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 531: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 532: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 533: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 534: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 535: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 536: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 537: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 538: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 539: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 540: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 541: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 542: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 543: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 544: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 545: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 546: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 547: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 548: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 549: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 550: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 551: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 552: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 553: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 554: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 555: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 556: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 557: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 558: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 559: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 560: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 561: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 562: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 563: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 564: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 565: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 566: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 567: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 568: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 569: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 570: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 571: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 572: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 573: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 574: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 575: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 576: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 577: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 578: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 579: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 580: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 581: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 582: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 583: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 584: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 585: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 586: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 587: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 588: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 589: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 590: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 591: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 592: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 593: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 594: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 595: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 596: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 597: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 598: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 599: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 600: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 601: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 602: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 603: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 604: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 605: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 606: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 607: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 608: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 609: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 610: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 611: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 612: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 613: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 614: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 615: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 616: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 617: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 618: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 619: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 620: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 621: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 622: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 623: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 624: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 625: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 626: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 627: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 628: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 629: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 630: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 631: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 632: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 633: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 634: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 635: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 636: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 637: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 638: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 639: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 640: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 641: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 642: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 643: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 644: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 645: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 646: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 647: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 648: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 649: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 650: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 651: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 652: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 653: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 654: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 655: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 656: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 657: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 658: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 659: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 660: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 661: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 662: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 663: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 664: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 665: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 666: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 667: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 668: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 669: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 670: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 671: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 672: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 673: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 674: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 675: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 676: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 677: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 678: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 679: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 680: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 681: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 682: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 683: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 684: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 685: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 686: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 687: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 688: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 689: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 690: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 691: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 692: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 693: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 694: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 695: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 696: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 697: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 698: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 699: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 700: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 701: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 702: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 703: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 704: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 705: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 706: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 707: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 708: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 709: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 710: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 711: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 712: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 713: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 714: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 715: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 716: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 717: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 718: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 719: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 720: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 721: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 722: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 723: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 724: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 725: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 726: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 727: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 728: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 729: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 730: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 731: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 732: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 733: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 734: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 735: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 736: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 737: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 738: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 739: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 740: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 741: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 742: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 743: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 744: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 745: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 746: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 747: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 748: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 749: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 750: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 751: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 752: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 753: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 754: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 755: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 756: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 757: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 758: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 759: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 760: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 761: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 762: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 763: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 764: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 765: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 766: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 767: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 768: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 769: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 770: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 771: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 772: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 773: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 774: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 775: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 776: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 777: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 778: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 779: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 780: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 781: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 782: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 783: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 784: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 785: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 786: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 787: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 788: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 789: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 790: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 791: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 792: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 793: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 794: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 795: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 796: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 797: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 798: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 799: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 800: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 801: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 802: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 803: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 804: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 805: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 806: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 807: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 808: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 809: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 810: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 811: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 812: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 813: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 814: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 815: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 816: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 817: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 818: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 819: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 820: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 821: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 822: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 823: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 824: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 825: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 826: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 827: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 828: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 829: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 830: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 831: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 832: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 833: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 834: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 835: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 836: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 837: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 838: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 839: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 840: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 841: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 842: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 843: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 844: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 845: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 846: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 847: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 848: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 849: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 850: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 851: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 852: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 853: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 854: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 855: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 856: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 857: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 858: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 859: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 860: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 861: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 862: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 863: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 864: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 865: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 866: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 867: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 868: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 869: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 870: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 871: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 872: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 873: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 874: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 875: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 876: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 877: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 878: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 879: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 880: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 881: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 882: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 883: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 884: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 885: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 886: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 887: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 888: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 889: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 890: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 891: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 892: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 893: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 894: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 895: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 896: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 897: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 898: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 899: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 900: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 901: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 902: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 903: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 904: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 905: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 906: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 907: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 908: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 909: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 910: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 911: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 912: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 913: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 914: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 915: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 916: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 917: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 918: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 919: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 920: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 921: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 922: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 923: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 924: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 925: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 926: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 927: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 928: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 929: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 930: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 931: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 932: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 933: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 934: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 935: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 936: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 937: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 938: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 939: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 940: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 941: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 942: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 943: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 944: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 945: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 946: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 947: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 948: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 949: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 950: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 951: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 952: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 953: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 954: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 955: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 956: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 957: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 958: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 959: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 960: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 961: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 962: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 963: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 964: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 965: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 966: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 967: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 968: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 969: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 970: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 971: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 972: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 973: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 974: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 975: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 976: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 977: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 978: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 979: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 980: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 981: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 982: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 983: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 984: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 985: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 986: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 987: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 988: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 989: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 990: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 991: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 992: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 993: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 994: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 995: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 996: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 997: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 998: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 999: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 1000: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 1001: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 1002: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 1003: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
- Control trace 1004: clinical governance lead verifies encounter access grant, note/image reference, clinician sign-off, and draft audit record in the same decision record; a mismatch routes to manual chart review and preserves the reason code.
