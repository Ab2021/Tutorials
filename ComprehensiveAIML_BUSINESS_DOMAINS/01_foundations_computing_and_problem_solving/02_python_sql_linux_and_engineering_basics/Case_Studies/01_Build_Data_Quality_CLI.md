---
case_id: "CASE-02-001"
case_title: "Build a Data Quality CLI"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Foundations and computing"
primary_pattern: "data-quality-gate"
secondary_patterns: ["correctness-contract"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "data-platform lead"
source_index: "02_python_sql_linux_and_engineering_basics/Case_Studies/INDEX.md"
review_trigger: "Any change to workload, decision authority, or acceptance metric"
---

# Build a Data Quality CLI

**Evidence declaration:** This is an ILLUSTRATIVE teaching case. Every workload, result, and architecture below is constructed to make the decision auditable; none is a company deployment or measured outcome. **Implementation stage:** proposed. **Editorial status:** draft; source verification and publication review remain OPEN.

## 1. The One-Line Framing
Given one row and its decision-time evidence, decide **whether to block a bad batch** for the data-platform lead to produce a valid batch decision, subject to an explicit budget and human authority, compared with **ad-hoc notebook checks**. The case teaches a decision contract, not a claim that a system has been operated.


## 2. Context & Background
Workflow begins with an intake record, then validates inputs, computes a candidate result, records the reason, and routes exceptions to a named person.
The owner can approve, defer, or reject the proposed action; software may recommend but does not replace that authority.
The baseline persists because it is understandable and reversible, despite inconsistent evidence capture.
Scale is illustrative: a steady workload is used only to expose units, queueing, and cost boundaries.
**Takeaway:** Context establishes which decision is consequential before a tool is selected.


## 3. The Business Question
| Field | Contract |
|---|---|
| Question | Should the team make **whether to block a bad batch** now? |
| Owner | data-platform lead |
| Trigger | A candidate arrives before its decision deadline. |
| Actions | approve, defer for evidence, reject, or retain ad-hoc notebook checks. |
| Objective | Maximize correct, reviewable decisions rather than raw automation. |
| Hard constraint | No irreversible action without the owner-approved rule. |
| Scope | One illustrative rows in one decision window. |
The proxy metric is not the business objective: a fast answer can be wrong, and a high score cannot grant authority.


## 4. Assumptions & Constraints
- **A-01:** Inputs are available before the decision cutoff. Verification: inspect the timestamped record. If violated: defer to ad-hoc notebook checks.
- **A-02:** A stable identifier links evidence to the decision. Verification: inspect the timestamped record. If violated: defer to ad-hoc notebook checks.
- **A-03:** The workload rate is a planning scenario, not an observed benchmark. Verification: inspect the timestamped record. If violated: defer to ad-hoc notebook checks.
- **A-04:** A reviewer can inspect a rejected or deferred result. Verification: inspect the timestamped record. If violated: defer to ad-hoc notebook checks.
- **A-05:** The fallback baseline remains available during an outage. Verification: inspect the timestamped record. If violated: defer to ad-hoc notebook checks.
**Takeaway:** Each assumption has a test, owner, and safe fallback.


## 5. Data, Tools & Method
### 5a. Data Specification
Unit of analysis: one row with `request_id`, event time, input version, decision-time fields, proposed action, final owner action, and reason code. Labels are optional downstream adjudications and must not leak into the decision-time record.
### 5b. Tools & Stack
Proposed components are a versioned input store, deterministic validator, policy evaluator, append-only decision ledger, queue, and reviewer console. Versions and configuration hashes travel with every result.
### 5c. Method Summary
Input -> schema and freshness checks -> bounded calculation -> candidate valid batch decision -> policy gate -> owner action -> immutable audit record. The baseline is ad-hoc notebook checks; alternatives are a stricter rule and defer-all. None is claimed tested.
### 5d. Data provenance, quality, exclusions and leakage controls
Reject missing identifiers, future timestamps, duplicate decision IDs, and fields whose availability is after the cutoff. Keep source version and extraction time; do not fit thresholds on the evaluation window.
### 5e. Reproducibility and method-selection ledger
| Artifact | Purpose | Accessibility | Limitation |
|---|---|---|---|
| ILL-01 | worked arithmetic | constructed | not production evidence |
| CFG-01 | policy configuration | proposed | requires change control |
| ORC-01 | small reference oracle | proposed | validates only stated contract |


## 6. Analysis Walkthrough
**ILLUSTRATIVE and VERIFIED arithmetic.** Assume arrival rate λ=120 batches per day, sustainable service μ=100 batches per day, and a 6-minute overload (t=360 s).
```text
FORMULA [C-01]: B = max(0, λ - μ) × t
WHERE: B [requests], λ and μ [requests/s], t [s].
CONSTRAINT: initially empty fluid queue; constant rates; no retries or loss.
NUMERICAL EXAMPLE: B=(120-100)×360=7,200 requests.
INTERPRETATION: the buffer absorbs a finite burst; it does not make overload safe.
```
If recovery service is μr=140 requests/s while arrivals are λr=80 requests/s, drain time is T=B/(μr−λr)=7,200/60=120 s. Independent check: 60 requests/s × 120 s = 7,200 requests. At μr=λr, T is undefined because no positive backlog drains. This is a workload model, not a latency guarantee.
- Validate the input schema and cutoff time.
- Compute the candidate result with a pinned configuration.
- Apply the policy gate and record its reason.
- Compare the result to a small reference oracle.
- Route any mismatch or missing evidence to the owner.
**Sensitivity:** doubling overload duration doubles B to 14,400 requests; reducing recovery surplus from 60 to 20 requests/s raises T to 360 s. The decision consequence is to cap intake or add measured capacity, not to claim resilience.


## 7. The Finding
The supported conclusion is conditional: use a deterministic, auditable gate for **whether to block a bad batch** only after the stated input and capacity checks pass. The magnitude above is illustrative, not measured. If evidence or authority is absent, defer rather than infer approval.


## 8. System/Solution Design
```text
[input] -> [validate/cutoff] -> [bounded evaluator] -> [policy gate] -> [owner action]
                  | failure                         |                 |
                  +-> [defer + reason ledger] <-----+-----------------+
```
State ownership: the ledger owns idempotency keys and final action history; the evaluator owns no durable authority. Retry reads the prior key before creating a new action. Capacity protection occurs before expensive parsing.


## 9. Evaluation & Validation
- Correctness: compare 100 hand-constructed records with an independent simple oracle.
- Safety: inject missing, future, duplicate, and malformed fields; expected result is defer with a reason.
- Load: replay the illustrative λ/μ scenario and observe queue state, not a production SLA.
- Recovery: stop a worker, restart it, and verify one final action per idempotency key.
- Review: sample decisions with the owner and record disagreements without relabeling them as accuracy.
Acceptance is a proposed gate: zero duplicate final actions in the test trace and every deferred item has a reason. It is not evidence of deployment.


## 10. Action Taken
No production action was taken. The proposed next action is a time-bounded offline replay with synthetic records, owner sign-off on policy semantics, and a rollback that disables the evaluator while retaining the ledger.


## 11. Outcome & Results
**Result status: not measured.** The only observed result in this manuscript is the checked illustrative arithmetic in Section 6. A future evaluation must report numerator, denominator, population, time window, comparator, uncertainty, and excluded records before claiming improvement.


## 12. Failure Modes & What Went Wrong
- **Duplicate action:** symptom=Duplicate action; competing cause=retry after timeout; diagnostic=check idempotency key and ledger cardinality; immediate recovery=return stored action; then repair caller retry contract; verification=replay the same key and inspect the reason ledger.
- **Stale input:** symptom=Stale input; competing cause=late event or clock skew; diagnostic=compare source time with cutoff; immediate recovery=defer; then fix timestamp ownership; verification=replay the same key and inspect the reason ledger.
- **Queue growth:** symptom=Queue growth; competing cause=arrival rate exceeds service; diagnostic=measure λ, μ, and backlog slope; immediate recovery=cap intake or add capacity after measurement; verification=replay the same key and inspect the reason ledger.
- **Wrong policy version:** symptom=Wrong policy version; competing cause=uncontrolled configuration change; diagnostic=compare configuration hash; immediate recovery=rollback and require version approval; verification=replay the same key and inspect the reason ledger.
- **False confidence:** symptom=False confidence; competing cause=score mistaken for authority; diagnostic=audit decision and owner fields; immediate recovery=require explicit human approval; verification=replay the same key and inspect the reason ledger.
**Takeaway:** A safe system makes uncertainty and recovery visible rather than silently completing work.


## 13. Transferable Lesson
Transfer the contract, not the numbers: define the unit, cutoff, authority, baseline, invariant, bounded-work rule, and independent oracle before optimizing. A different domain needs its own evidence and cannot inherit this illustrative workload.


## 14. Connections & Cross-References
### Q1. What is the decision unit?
**Answer:** The decision contract requires a timestamped rows and a named owner. Reasoning: a reproducible gate is useful only if its inputs were available at the cutoff and its effect is reversible. Follow-up: inspect the ledger and compare with the independent oracle. Acceptable alternative: defer-all under uncertainty. Red flag: claiming the illustrative calculation proves production performance. Scoring: full credit names the invariant, boundary, and verification.
### Q2. Why is the baseline retained?
**Answer:** The decision contract requires a timestamped rows and a named owner. Reasoning: a reproducible gate is useful only if its inputs were available at the cutoff and its effect is reversible. Follow-up: inspect the ledger and compare with the independent oracle. Acceptable alternative: defer-all under uncertainty. Red flag: claiming the illustrative calculation proves production performance. Scoring: full credit names the invariant, boundary, and verification.
### Q3. Which field is most dangerous if stale?
**Answer:** The decision contract requires a timestamped rows and a named owner. Reasoning: a reproducible gate is useful only if its inputs were available at the cutoff and its effect is reversible. Follow-up: inspect the ledger and compare with the independent oracle. Acceptable alternative: defer-all under uncertainty. Red flag: claiming the illustrative calculation proves production performance. Scoring: full credit names the invariant, boundary, and verification.
### Q4. What makes the queue formula invalid?
**Answer:** The decision contract requires a timestamped rows and a named owner. Reasoning: a reproducible gate is useful only if its inputs were available at the cutoff and its effect is reversible. Follow-up: inspect the ledger and compare with the independent oracle. Acceptable alternative: defer-all under uncertainty. Red flag: claiming the illustrative calculation proves production performance. Scoring: full credit names the invariant, boundary, and verification.
### Q5. How do retries avoid duplicate effects?
**Answer:** The decision contract requires a timestamped rows and a named owner. Reasoning: a reproducible gate is useful only if its inputs were available at the cutoff and its effect is reversible. Follow-up: inspect the ledger and compare with the independent oracle. Acceptable alternative: defer-all under uncertainty. Red flag: claiming the illustrative calculation proves production performance. Scoring: full credit names the invariant, boundary, and verification.
### Q6. Which result is actually measured?
**Answer:** The decision contract requires a timestamped rows and a named owner. Reasoning: a reproducible gate is useful only if its inputs were available at the cutoff and its effect is reversible. Follow-up: inspect the ledger and compare with the independent oracle. Acceptable alternative: defer-all under uncertainty. Red flag: claiming the illustrative calculation proves production performance. Scoring: full credit names the invariant, boundary, and verification.
### Q7. When should the owner defer?
**Answer:** The decision contract requires a timestamped rows and a named owner. Reasoning: a reproducible gate is useful only if its inputs were available at the cutoff and its effect is reversible. Follow-up: inspect the ledger and compare with the independent oracle. Acceptable alternative: defer-all under uncertainty. Red flag: claiming the illustrative calculation proves production performance. Scoring: full credit names the invariant, boundary, and verification.
### Q8. How would you test leakage?
**Answer:** The decision contract requires a timestamped rows and a named owner. Reasoning: a reproducible gate is useful only if its inputs were available at the cutoff and its effect is reversible. Follow-up: inspect the ledger and compare with the independent oracle. Acceptable alternative: defer-all under uncertainty. Red flag: claiming the illustrative calculation proves production performance. Scoring: full credit names the invariant, boundary, and verification.
### Q9. What changes if service equals arrivals?
**Answer:** The decision contract requires a timestamped rows and a named owner. Reasoning: a reproducible gate is useful only if its inputs were available at the cutoff and its effect is reversible. Follow-up: inspect the ledger and compare with the independent oracle. Acceptable alternative: defer-all under uncertainty. Red flag: claiming the illustrative calculation proves production performance. Scoring: full credit names the invariant, boundary, and verification.
### Q10. Why is a score not authority?
**Answer:** The decision contract requires a timestamped rows and a named owner. Reasoning: a reproducible gate is useful only if its inputs were available at the cutoff and its effect is reversible. Follow-up: inspect the ledger and compare with the independent oracle. Acceptable alternative: defer-all under uncertainty. Red flag: claiming the illustrative calculation proves production performance. Scoring: full credit names the invariant, boundary, and verification.
### Q11. What does the oracle check?
**Answer:** The decision contract requires a timestamped rows and a named owner. Reasoning: a reproducible gate is useful only if its inputs were available at the cutoff and its effect is reversible. Follow-up: inspect the ledger and compare with the independent oracle. Acceptable alternative: defer-all under uncertainty. Red flag: claiming the illustrative calculation proves production performance. Scoring: full credit names the invariant, boundary, and verification.
### Q12. Which changed constraint reverses the design?
**Answer:** The decision contract requires a timestamped rows and a named owner. Reasoning: a reproducible gate is useful only if its inputs were available at the cutoff and its effect is reversible. Follow-up: inspect the ledger and compare with the independent oracle. Acceptable alternative: defer-all under uncertainty. Red flag: claiming the illustrative calculation proves production performance. Scoring: full credit names the invariant, boundary, and verification.

## 15. Quick Reference Card
| Field | Decision card |
|---|---|
| Unit | one timestamped rows |
| Baseline | ad-hoc notebook checks |
| Rule | validate → compute → gate → owner action |
| Invariant | one final action per idempotency key |
| Formula | B=max(0, λ−μ)t; only constant-rate fluid planning |
| Boundary | if recovery surplus ≤0, backlog cannot drain |
| Evidence | illustrative arithmetic; not a deployment measurement |
| Escalate | missing evidence, policy mismatch, or authority ambiguity |

