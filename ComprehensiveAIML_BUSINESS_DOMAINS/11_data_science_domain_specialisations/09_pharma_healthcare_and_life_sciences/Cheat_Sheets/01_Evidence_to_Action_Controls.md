---
collection: "pharma-healthcare-life-sciences-cross-domain"
file_id: "01"
topic: "Evidence-to-Action Controls Across Pharma, Healthcare and Life Sciences"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Primary workflow evidence, source access, or authority policy changes"
notation_style: "plain-text"
prerequisites: []
pattern_tags: [point-in-time-correctness, evidence-ledger, human-authorization]
source_status: "OPEN"
owner_role: "Curriculum maintainer"
source_index: "INDEX.md"
---

# Evidence-to-Action Controls Across Pharma, Healthcare and Life Sciences
> **Purpose: construct a bounded, auditable recommendation path across commercial, clinical, and operations workflows.**
> **Prerequisites: ratios, timestamp ordering, experimental reasoning, and the difference between recommendation and authorized action.**
> **Key outcome: given a decision record, calculate a rate or queue consequence, reject leakage, and choose abstain, review, or bounded action.**
> **This file does NOT cover diagnosis, legal interpretation, or product-specific regulated validation.**
> **Level: 1 Foundations with production boundaries.**
> **Version / reviewed: 1.0 / not reviewed.**
> **Evidence: DERIVED and ILLUSTRATIVE calculations; OPEN primary workflow evidence.**
> **Retrieval cue: if an output will influence a workflow, start at Section 4.**

## 2. Concept Map / Mental Model
[source event] -> [as-of eligibility check] -> [bounded transformation] -> [recommendation]
[recommendation] -> [evidence packet + confidence] -> [authorized review] -> [recorded decision]
[missing provenance] -> [abstain / collect evidence]
The same graph applies to a commercial contact, RWE cohort record, safety intake, batch exception, or document field.
A prediction is intermediate; authority and traceability follow prediction.
Takeaway: preserve event clock and authority boundary before optimizing a model.

## 3. Foundational Definitions
- **decision event:** one action opportunity, keyed and timestamped. Nonexample: an unkeyed dashboard aggregate cannot justify an individual action.
- **as-of timestamp t_d:** latest time at which inputs may be used. Nonexample: an unkeyed dashboard aggregate cannot justify an individual action.
- **eligibility:** documented condition allowing an option to be considered. Nonexample: an unkeyed dashboard aggregate cannot justify an individual action.
- **recommendation:** ranked or classified advice with no authority to commit. Nonexample: an unkeyed dashboard aggregate cannot justify an individual action.
- **authorized action:** state transition performed by an approved owner. Nonexample: an unkeyed dashboard aggregate cannot justify an individual action.
- **adjudicated outcome:** reviewed label whose maturity and provenance are known. Nonexample: an unkeyed dashboard aggregate cannot justify an individual action.
- **denominator:** eligible units counted in a rate. Nonexample: an unkeyed dashboard aggregate cannot justify an individual action.
- **evidence packet:** source identifiers, versions, transformations, and rationale. Nonexample: an unkeyed dashboard aggregate cannot justify an individual action.
- **abstention:** explicit non-action because a precondition fails. Nonexample: an unkeyed dashboard aggregate cannot justify an individual action.
- **point-in-time join:** link using records available at or before t_d. Nonexample: an unkeyed dashboard aggregate cannot justify an individual action.
Takeaway: definitions prevent unit, timing, and authority from changing silently.

## 4. Core Formulas with Full Derivation
### 4.1 EAC-01
Validity before use: n > 0 and eligibility is frozen before counting
Derivation: rate is count divided by declared eligible count; conservation gives arrivals minus completions; comparison subtracts like-defined rates.
FORMULA [EAC-01]: actionable rate r = a / n
WHERE: a actionable units [count]; n eligible units [count]; r dimensionless
CONSTRAINT: n > 0 and eligibility is frozen before counting
NUMERICAL EXAMPLE: ILLUSTRATIVE: a=15; n=30; r=15/30=0.50.
INTERPRETATION: Only half of eligible records are actionable.
Boundary/counterexample: If n excludes difficult records after outcome, r is not comparable.
Independent check: substitute stated counts; units cancel for a rate and reviews/hour times hour yields reviews.
### 4.2 EAC-02
Validity before use: constant rates, empty initial queue, no retries or rework
Derivation: rate is count divided by declared eligible count; conservation gives arrivals minus completions; comparison subtracts like-defined rates.
FORMULA [EAC-02]: queue change B = max(0, lambda - mu) * t
WHERE: B [reviews]; lambda arrivals [reviews/hour]; mu completions [reviews/hour]; t [hour]
CONSTRAINT: constant rates, empty initial queue, no retries or rework
NUMERICAL EXAMPLE: ILLUSTRATIVE: lambda=18, mu=12, t=4; B=24 reviews.
INTERPRETATION: The threshold cannot be sustained by 12 reviews/hour.
Boundary/counterexample: Rework lowers effective service and understates backlog.
Independent check: substitute stated counts; units cancel for a rate and reviews/hour times hour yields reviews.
### 4.3 EAC-03
Validity before use: comparable eligibility; arithmetic does not establish causality
Derivation: rate is count divided by declared eligible count; conservation gives arrivals minus completions; comparison subtracts like-defined rates.
FORMULA [EAC-03]: absolute response difference d = r_T - r_C
WHERE: r_T and r_C response fractions; d percentage points
CONSTRAINT: comparable eligibility; arithmetic does not establish causality
NUMERICAL EXAMPLE: ILLUSTRATIVE: 20/50=0.40; 15/50=0.30; d=0.10=10 percentage points.
INTERPRETATION: This is descriptive input to designed evaluation.
Boundary/counterexample: Selection into treatment invalidates an intervention interpretation.
Independent check: substitute stated counts; units cancel for a rate and reviews/hour times hour yields reviews.
Takeaway: calculations require a frozen population, timestamp, and workflow state.

## 5. Worked Numerical Example
Evidence: ILLUSTRATIVE teaching workload, not a clinical, commercial, or manufacturing measurement.
At 09:00, 100 candidate records arrive. Eligibility rejects 40 for missing authority evidence and routes 30 to mandatory human review.
Of the 30, 15 have complete evidence and are eligible for bounded recommendation; actionable rate is 15/30=0.50.
If reviewers complete 12 records/hour while 18 arrive for four hours, backlog is 24 reviews by EAC-02.
Sensitivity: 13.5 arrivals/hour gives (13.5-12)*4=6 reviews.
Decision: reduce low-value alerts or add qualified capacity; plausible scores do not make a workflow operable.
Arithmetic oracle: 15+15=30; 18-12=6; 6*4=24.
Takeaway: action rate and queue estimate jointly reveal operability.

## 6. Visual / Geometric Intuition
Imagine a timeline: every input lands at or before t_d.
Imagine a funnel: candidates become eligible, evidence-complete, reviewable, then authorized.
Recommendation-to-action is a gate owned by a person or validated workflow.
A missing source is a broken edge, not a low-confidence fact to smooth over.
Takeaway: use a time-ordered state machine with named owners.

## 7. Comparison / Selection Table
| Option | Benefit | Use when | Avoid when |
|---|---|---|---|
| manual review | context and authority | small ambiguous queue | queue unstable |
| rules only | transparent eligibility | policy complete | boundary cases dominate |
| scored recommendation + review | prioritizes attention | labels and capacity exist | as-of features are unaudited |
| bounded automation | reduces approved repetition | recovery explicit | exception handling absent |
Takeaway: choose the least autonomous option meeting decision and evidence constraints.

## 8. Failure Modes & Edge Cases
### ISSUE-001 — VIOLATED_ASSUMPTION
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: association called action.
Competing causes: promotion timing also changed; stale snapshot; denominator drift.
Discriminating diagnostic: compare predeclared concurrent control.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: hold recommendation.
Correction: require intervention design.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-002 — NUMERICAL_EDGE
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: future update joined backward.
Competing causes: late status update; stale snapshot; denominator drift.
Discriminating diagnostic: audit availability timestamps.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: exclude future field.
Correction: store as-of snapshots.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-003 — SCALE_FAILURE
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: retrieval called support.
Competing causes: claim support conflated; stale snapshot; denominator drift.
Discriminating diagnostic: sample citations against source spans.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: abstain unsupported claim.
Correction: require evidence mapping.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-004 — CONCEPT_CONFUSION
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: rate improves after denominator shrinks.
Competing causes: case mix changed; stale snapshot; denominator drift.
Discriminating diagnostic: report numerator denominator exclusions.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: freeze denominator.
Correction: monitor eligibility composition.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-005 — VIOLATED_ASSUMPTION
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: alert called confirmed signal.
Competing causes: adjudication delayed; stale snapshot; denominator drift.
Discriminating diagnostic: separate alert and outcome.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: route to review.
Correction: track label maturity.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-006 — NUMERICAL_EDGE
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: queue grows under stable score.
Competing causes: arrivals exceed capacity; stale snapshot; denominator drift.
Discriminating diagnostic: measure arrivals and completions.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: throttle low-value alerts.
Correction: reserve capacity.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-007 — SCALE_FAILURE
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: importance called effect.
Competing causes: correlated features; stale snapshot; denominator drift.
Discriminating diagnostic: require causal design.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: remove causal wording.
Correction: document association.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-008 — CONCEPT_CONFUSION
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: impossible patient sequence.
Competing causes: identity linkage crossed people; stale snapshot; denominator drift.
Discriminating diagnostic: inspect cardinality and order.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: quarantine links.
Correction: enforce linkage rules.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-009 — VIOLATED_ASSUMPTION
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: yield exceeds one.
Competing causes: reject/rework overlap; stale snapshot; denominator drift.
Discriminating diagnostic: reconcile states.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: stop release calculation.
Correction: define state machine.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-010 — NUMERICAL_EDGE
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: document latency breaches window.
Competing causes: complex pages slow path; stale snapshot; denominator drift.
Discriminating diagnostic: profile document class.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: route complex pages.
Correction: plan percentile capacity.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-011 — SCALE_FAILURE
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: approval inferred from text.
Competing causes: source lacks signed state; stale snapshot; denominator drift.
Discriminating diagnostic: query authoritative state.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: require explicit approval.
Correction: preserve provenance.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-012 — CONCEPT_CONFUSION
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: forecast treated commitment.
Competing causes: interval omitted; stale snapshot; denominator drift.
Discriminating diagnostic: show interval and rule.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: add reserve policy.
Correction: backtest coverage.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-013 — VIOLATED_ASSUMPTION
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: precision rises while misses rise.
Competing causes: threshold moved; stale snapshot; denominator drift.
Discriminating diagnostic: plot precision recall queue.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: restore capacity-aware threshold.
Correction: fix objective.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-014 — NUMERICAL_EDGE
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: site failure hidden by pool.
Competing causes: volume weighting; stale snapshot; denominator drift.
Discriminating diagnostic: slice by site.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: hold rollout.
Correction: set site guardrail.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-015 — SCALE_FAILURE
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: output directly changes record.
Competing causes: authority boundary omitted; stale snapshot; denominator drift.
Discriminating diagnostic: trace actor transition.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: disable write path.
Correction: require authorization.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-016 — CONCEPT_CONFUSION
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: missing coded absent.
Competing causes: not-recorded conflated; stale snapshot; denominator drift.
Discriminating diagnostic: inspect source semantics.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: add unknown state.
Correction: document coding.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-017 — VIOLATED_ASSUMPTION
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: calibration judged from few positives.
Competing causes: uncertainty ignored; stale snapshot; denominator drift.
Discriminating diagnostic: report bin counts.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: widen margin.
Correction: minimum evidence count.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-018 — NUMERICAL_EDGE
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: retries duplicate recommendation.
Competing causes: no idempotency key; stale snapshot; denominator drift.
Discriminating diagnostic: replay same request.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: deduplicate boundary.
Correction: persist key.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-019 — SCALE_FAILURE
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: title treated as source body.
Competing causes: body unavailable; stale snapshot; denominator drift.
Discriminating diagnostic: record OPEN gap.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: narrow claim.
Correction: obtain primary text.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-020 — CONCEPT_CONFUSION
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: validation set called evidence.
Competing causes: tuning mixed evaluation; stale snapshot; denominator drift.
Discriminating diagnostic: inspect ledger.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: reserve test set.
Correction: separate selection.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-021 — VIOLATED_ASSUMPTION
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: excursion duration double-counted.
Competing causes: windows overlap; stale snapshot; denominator drift.
Discriminating diagnostic: merge intervals.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: quarantine calculation.
Correction: define interval union.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-022 — NUMERICAL_EDGE
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: freshness silently lags.
Competing causes: extract missed schedule; stale snapshot; denominator drift.
Discriminating diagnostic: compare event and availability clocks.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: fallback rule.
Correction: monitor lag.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-023 — SCALE_FAILURE
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: subgroup excluded by proxy.
Competing causes: coverage differs; stale snapshot; denominator drift.
Discriminating diagnostic: tabulate inclusion.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: do not automate slice.
Correction: coverage gate.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-024 — CONCEPT_CONFUSION
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: design described deployed.
Competing causes: labels mixed; stale snapshot; denominator drift.
Discriminating diagnostic: inspect claim ledger.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: relabel illustrative.
Correction: separate plan.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
### ISSUE-025 — VIOLATED_ASSUMPTION
Trigger/environment: cross-domain evidence-to-action workflow.
Symptom: audit cannot reconstruct action.
Competing causes: versions absent; stale snapshot; denominator drift.
Discriminating diagnostic: attempt trace reconstruction.
Expected signal: trace exposes the first violated precondition.
Immediate mitigation: block autonomy.
Correction: immutable ledger.
Prevention: assign owner, retain versions, make precondition executable.
Mitigation risk: holding work can delay a legitimate workflow; escalate to the authorized owner.
Verification: replay a bounded record and confirm the correction.
Provenance: HYPOTHETICAL teaching diagnostic; no incident is claimed.
Takeaway: diagnose the first violated precondition, not merely the final score or queue.

## 9. Complexity & Cost Analysis
For N decision records and K evidence checks, validation is O(NK) time and O(NK) retained outcomes.
At 24 queued reviews and 12 completions/hour with arrivals paused, drain time is 2 hours.
With 9 arrivals/hour, spare capacity is 3/hour and drain time is 8 hours.
Retaining 2 KiB evidence per 100 records uses 200 KiB payload-only; indexes, encryption, replication, retention excluded.
These are DERIVED workload arithmetic, not benchmarks.
Takeaway: capacity and evidence-retention costs belong in the decision contract.

## 10. Key Theorems / Results Reference
Conservation: queue change equals arrivals minus completions when no item is lost or duplicated.
Temporal validity: a prediction-time feature is available no later than decision time.
Causal warning: observed rate difference is not an intervention effect without identification.
Auditability invariant: every committed transition references input, version, authority, and state.
Takeaway: invariants survive method changes.

## 11. Interview / Exam Questions
### Q01. What is the unit of analysis?
Answer: Declare one keyed decision event and decision timestamp; never merge patient, HCP, batch, and document grain.
Reasoning: name population, timestamp, owner, comparator before metric or model.
Follow-up: what evidence reverses your decision?
Acceptable alternative: simpler rule meeting same guardrails.
Red flag: score or retrieved text is treated as authority.
Scoring: full credit requires mechanism and boundary.
Answer support: Sections 3–10; illustrative calculations checked above.
### Q02. Why is an accurate model insufficient?
Answer: Accuracy omits authority, error cost, denominator, capacity, and input availability.
Reasoning: name population, timestamp, owner, comparator before metric or model.
Follow-up: what evidence reverses your decision?
Acceptable alternative: simpler rule meeting same guardrails.
Red flag: score or retrieved text is treated as authority.
Scoring: full credit requires mechanism and boundary.
Answer support: Sections 3–10; illustrative calculations checked above.
### Q03. How does an alert differ from a signal?
Answer: An alert is output; a signal is adjudicated conclusion.
Reasoning: name population, timestamp, owner, comparator before metric or model.
Follow-up: what evidence reverses your decision?
Acceptable alternative: simpler rule meeting same guardrails.
Red flag: score or retrieved text is treated as authority.
Scoring: full credit requires mechanism and boundary.
Answer support: Sections 3–10; illustrative calculations checked above.
### Q04. When should the system abstain?
Answer: Abstain when provenance, freshness, eligibility, or authority fails.
Reasoning: name population, timestamp, owner, comparator before metric or model.
Follow-up: what evidence reverses your decision?
Acceptable alternative: simpler rule meeting same guardrails.
Red flag: score or retrieved text is treated as authority.
Scoring: full credit requires mechanism and boundary.
Answer support: Sections 3–10; illustrative calculations checked above.
### Q05. Why record a baseline?
Answer: A manual/rule baseline establishes added value and workload.
Reasoning: name population, timestamp, owner, comparator before metric or model.
Follow-up: what evidence reverses your decision?
Acceptable alternative: simpler rule meeting same guardrails.
Red flag: score or retrieved text is treated as authority.
Scoring: full credit requires mechanism and boundary.
Answer support: Sections 3–10; illustrative calculations checked above.
### Q06. What makes a recommendation causal?
Answer: Use justified identification and an estimand; attribution alone is associative.
Reasoning: name population, timestamp, owner, comparator before metric or model.
Follow-up: what evidence reverses your decision?
Acceptable alternative: simpler rule meeting same guardrails.
Red flag: score or retrieved text is treated as authority.
Scoring: full credit requires mechanism and boundary.
Answer support: Sections 3–10; illustrative calculations checked above.
### Q07. How do you size review capacity?
Answer: Use arrivals minus completions over a window and prove recovery service exceeds arrivals.
Reasoning: name population, timestamp, owner, comparator before metric or model.
Follow-up: what evidence reverses your decision?
Acceptable alternative: simpler rule meeting same guardrails.
Red flag: score or retrieved text is treated as authority.
Scoring: full credit requires mechanism and boundary.
Answer support: Sections 3–10; illustrative calculations checked above.
### Q08. What is point-in-time correctness?
Answer: Every feature is available by t_d; later update is leakage.
Reasoning: name population, timestamp, owner, comparator before metric or model.
Follow-up: what evidence reverses your decision?
Acceptable alternative: simpler rule meeting same guardrails.
Red flag: score or retrieved text is treated as authority.
Scoring: full credit requires mechanism and boundary.
Answer support: Sections 3–10; illustrative calculations checked above.
### Q09. How should a source gap be written?
Answer: State OPEN, name missing artifact, narrow conclusion, and recheck trigger.
Reasoning: name population, timestamp, owner, comparator before metric or model.
Follow-up: what evidence reverses your decision?
Acceptable alternative: simpler rule meeting same guardrails.
Red flag: score or retrieved text is treated as authority.
Scoring: full credit requires mechanism and boundary.
Answer support: Sections 3–10; illustrative calculations checked above.
### Q10. What is the key governance invariant?
Answer: No external transition without authorized actor and recorded evidence.
Reasoning: name population, timestamp, owner, comparator before metric or model.
Follow-up: what evidence reverses your decision?
Acceptable alternative: simpler rule meeting same guardrails.
Red flag: score or retrieved text is treated as authority.
Scoring: full credit requires mechanism and boundary.
Answer support: Sections 3–10; illustrative calculations checked above.
### Q11. How validate a document assistant?
Answer: Separate retrieval, extraction, support, and reviewer decision.
Reasoning: name population, timestamp, owner, comparator before metric or model.
Follow-up: what evidence reverses your decision?
Acceptable alternative: simpler rule meeting same guardrails.
Red flag: score or retrieved text is treated as authority.
Scoring: full credit requires mechanism and boundary.
Answer support: Sections 3–10; illustrative calculations checked above.
### Q12. What changes under a capacity cut?
Answer: Compare missed-value cost, queue, and slice impact; do not optimize precision alone.
Reasoning: name population, timestamp, owner, comparator before metric or model.
Follow-up: what evidence reverses your decision?
Acceptable alternative: simpler rule meeting same guardrails.
Red flag: score or retrieved text is treated as authority.
Scoring: full credit requires mechanism and boundary.
Answer support: Sections 3–10; illustrative calculations checked above.
Takeaway: defend workflow boundaries, not fashionable algorithms.

## 12. Quick Reference Card
| Field | Card |
|---|---|
| Decision | recommend, review, abstain, or bounded authorized transition |
| Unit | keyed decision event with t_d |
| Critical invariant | no future input; no unauthorized state transition |
| Core checks | eligibility, provenance, denominator, queue capacity, authority |
| Formula | r=a/n; B=max(0,lambda-mu)t |
| Use when | evidence-backed workflow has named owner |
| Avoid when | source, timing, or authority is unknown |
| Evidence | DERIVED + ILLUSTRATIVE; source-specific evidence OPEN |
