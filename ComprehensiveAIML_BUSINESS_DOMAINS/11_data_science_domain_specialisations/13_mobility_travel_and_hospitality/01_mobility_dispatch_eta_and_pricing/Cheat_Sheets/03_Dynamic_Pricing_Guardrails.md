---
collection: "01_mobility_dispatch_eta_and_pricing"
file_id: "authored"
topic: "Dynamic-Pricing Guardrails"
level: 2
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Change in decision policy, event semantics, or operating population"
notation_style: "plain-text"
source_status: "OPEN"
owner_role: "Mobility/travel analytics owner"
---

# Dynamic-Pricing Guardrails
> **Purpose:** Make market-clearing price adjustments without treating a proxy as the outcome.
> **Prerequisites:** Ratios, conditional probability, timestamps, and the local collection INDEX.
> **Key outcome:** Given a defined zone-window record, construct, audit, and act on a metric with a stated denominator.
> **This file does NOT cover:** company-specific policy, legal interpretation, or measured company performance.
> **Evidence:** DERIVED and ILLUSTRATIVE calculations; no production benchmark is claimed.

## 1. File Header
The decision is market-clearing price adjustments. The owner must freeze the unit of analysis before comparing alternatives.
The outcome is safe offer multiplier; the listed signals are intermediate evidence rather than substitutes.
A metric changes an action only when its population, observation window, exclusion rule, and owner are known.
Takeaway: name the action and denominator before optimizing a dashboard number.

## 2. Concept Map / Mental Model
[eligible requests, offers, multiplier, acceptances, completions]
        | event-time validation
        v
[zone-window ledger] -> [eligibility and denominator] -> [completed requests / eligible requests]
        |                                      |
        v                                      v
[late/duplicate quarantine]                 [decision and guardrail]
        |                                      |
        +--------------------> [audit and recovery]
The invariant is one decision record contributes at most once to a stated numerator and denominator.
Takeaway: event flow is not causal proof; it is the evidence boundary for a decision.

## 3. Foundational Definitions
**eligible unit.** Definition: a unit meeting predeclared inclusion rules. Units/type: count. Validity: define before aggregation. Nonexample: not every observed request. Operational check: retain the underlying zone-window identifier.
**numerator.** Definition: units satisfying the success predicate. Units/type: count. Validity: define before aggregation. Nonexample: not a rate. Operational check: retain the underlying zone-window identifier.
**denominator.** Definition: eligible units observed in the window. Units/type: count. Validity: define before aggregation. Nonexample: not all logged events. Operational check: retain the underlying zone-window identifier.
**event time.** Definition: time the business event occurred. Units/type: UTC timestamp. Validity: define before aggregation. Nonexample: not ingestion time. Operational check: retain the underlying zone-window identifier.
**guardrail.** Definition: metric that can veto an otherwise favorable action. Units/type: rate or quantile. Validity: define before aggregation. Nonexample: not a secondary objective. Operational check: retain the underlying zone-window identifier.
Takeaway: each definition prevents a different denominator or timestamp error.

## 4. Core Formulas with Full Derivation
### 4.1 Simplest / Base Case
Validity before use: each eligible unit is counted once, success is observable, and the window is closed.
Derivation: define an indicator y_i=1 for success and e_i=1 for eligibility; summing successes then dividing by eligible units yields the rate.
```text
FORMULA [M-01]: r = sum_i(e_i*y_i) / sum_i(e_i)
WHERE: r is completed requests / eligible requests; y_i and e_i are binary; i indexes zone-windows.
CONSTRAINT: sum_i(e_i)>0; y_i is known at the chosen outcome cutoff; duplicates are reconciled.
NUMERICAL EXAMPLE: 76 successful eligible units / 80 eligible units = 0.95 = 95%.
INTERPRETATION: 95% is descriptive only for this population and window.
```
Boundary: if the denominator is zero, report undefined and investigate eligibility rather than report 0%.
### 4.2 General Case
Validity before use: groups are mutually exclusive at the chosen partition and weights express a declared target population.
Derivation: compute r_g within each group; target-population aggregation is r_target=sum_g w_g*r_g with weights summing to one.
```text
FORMULA [M-02]: r_target = sum_g w_g * r_g
WHERE: g is a cohort; w_g is dimensionless target share; r_g is cohort rate.
CONSTRAINT: do not substitute exposure shares for target shares without declaring the estimand.
NUMERICAL EXAMPLE: w_A=0.6,r_A=0.95; w_B=0.4,r_B=0.80 -> 0.89.
INTERPRETATION: aggregate performance can fall despite a strong high-volume cohort.
```
### 4.3 Key Special Cases / Variants
**overall observed rate:** changes the numerator, denominator, or loss; use only after documenting the decision it supports; avoid treating it as interchangeable with completed requests / eligible requests.
**eligible-cohort rate:** changes the numerator, denominator, or loss; use only after documenting the decision it supports; avoid treating it as interchangeable with completed requests / eligible requests.
**time-to-event quantile:** changes the numerator, denominator, or loss; use only after documenting the decision it supports; avoid treating it as interchangeable with completed requests / eligible requests.
**capacity-constrained utility:** changes the numerator, denominator, or loss; use only after documenting the decision it supports; avoid treating it as interchangeable with completed requests / eligible requests.
### 4.4 Asymptotic / Limiting Behaviour and Counterexample
As volume grows, sampling noise can shrink while selection bias remains unchanged. A 99% rate from a biased eligibility rule is not corrected by more rows.
Takeaway: formulas are valid only after entity identity, cutoff time, and inclusion rule are fixed.

## 5. Worked Numerical Example
**ILLUSTRATIVE, arithmetically verified teaching example; not a company measurement.**
An operator compares two hour-long policy windows, each with 80 eligible units. Window A has 76 successes; Window B has 72.
Step 1: r_A=76/80=0.95. Step 2: r_B=72/80=0.90. Difference=0.05=5 percentage points.
Independent check: 0.05*80=4 units, exactly the observed success-count difference.
Wrong path: call the change “5%” without naming the baseline; relative change is (0.90-0.95)/0.95=-5.263%, a different quantity.
Sensitivity: if two B records are late rather than failed, r_B=74/80=92.5%; the apparent change becomes 2.5 points.
Decision consequence: quarantine unresolved late outcomes; do not switch policy on the unqualified aggregate.
Takeaway: small count changes make denominator and label maturity operational requirements.

## 6. Visual / Geometric Intuition
```text
rate
1.0 |                 target cohort
0.9 |  observed A *             * observed B
0.8 |__________ eligibility boundary __________ time
```
The schematic is not to scale: moving the eligibility boundary changes the measured point without necessarily changing customer experience.
Takeaway: a metric point is a projection of a chosen population, not the population itself.

## 7. Comparison / Selection Table
raw event count | stable event identity | fast signal | duplicate-sensitive
eligible rate | predeclared inclusion | actionable proportion | hidden exclusions
cohort rates | cohort labels and sufficient count | distribution diagnosis | tiny/coerced cohorts
manual review sample | auditable sampling | semantic validation | cannot estimate all outcomes alone
Selection rule: if eligibility is disputed, stop aggregation and audit the ledger; if capacity binds, pair outcome rate with queue and abandonment guardrails; if cohorts differ, report both cohort and declared target aggregation.
Takeaway: select the measurement that matches the reversible action, not the easiest query.

## 8. Failure Modes & Edge Cases
**F-01 late outcomes.** Trigger: rate drops after backfill. Competing causes: late labels versus real regression. Discriminating check: compare outcome-age histogram. Immediate mitigation: freeze maturity cutoff; recompute. Prevention and verification: alert on label-age shift. Provenance: HYPOTHETICAL teaching failure; escalation owner: analytics on-call.
**F-02 duplicate events.** Trigger: volume rises without unique units. Competing causes: retry duplication versus demand spike. Discriminating check: distinct id / raw event ratio. Immediate mitigation: deduplicate by immutable id. Prevention and verification: replay duplicate test. Provenance: HYPOTHETICAL teaching failure; escalation owner: analytics on-call.
**F-03 cohort leakage.** Trigger: one cohort improves implausibly. Competing causes: post-decision feature or assignment. Discriminating check: audit availability timestamp. Immediate mitigation: remove future field. Prevention and verification: point-in-time unit test. Provenance: HYPOTHETICAL teaching failure; escalation owner: analytics on-call.
**F-04 denominator drift.** Trigger: rate improves while complaints rise. Competing causes: eligibility exclusion versus true improvement. Discriminating check: compare excluded share and reason codes. Immediate mitigation: veto release; restore rule. Prevention and verification: version eligibility policy. Provenance: HYPOTHETICAL teaching failure; escalation owner: analytics on-call.
Takeaway: a symptom requires a competing-cause check before correction.

## 9. Complexity & Cost Analysis
For N event records and G cohorts, one-pass aggregation is O(N) time and O(G) state if keys are bounded; exact deduplication requires identity state whose retention is a policy choice.
Illustrative sizing: 5,000 events/min for 60 min gives 300,000 events. At 200 bytes/event payload, payload-only volume is 60,000,000 bytes, about 57.22 MiB; indexes, replication, and logs are excluded.
If 1% are duplicates, 3,000 records need reconciliation; this is not a latency guarantee.
Takeaway: the dominant operational cost is retaining enough identity and lineage to reject false changes.

## 10. Key Theorems / Results Reference
Linearity of summation: totals across disjoint sets equal the sum of set totals; it does not justify combining overlapping cohorts.
Law of total expectation: a properly weighted conditional mean equals the target mean under a declared partition; it does not repair missing or selected outcomes.
Takeaway: identities preserve defined quantities, not undefined populations.

### Operational audit drills
**Audit drill — identity / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — identity / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — identity / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — identity / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — identity / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — event-time cutoff / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — event-time cutoff / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — event-time cutoff / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — event-time cutoff / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — event-time cutoff / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — eligibility version / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — eligibility version / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — eligibility version / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — eligibility version / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — eligibility version / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — outcome maturity / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — outcome maturity / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — outcome maturity / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — outcome maturity / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — outcome maturity / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — cohort membership / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — cohort membership / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — cohort membership / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — cohort membership / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — cohort membership / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — duplicate policy / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — duplicate policy / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — duplicate policy / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — duplicate policy / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — duplicate policy / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — missingness / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — missingness / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — missingness / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — missingness / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — missingness / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — timezone / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — timezone / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — timezone / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — timezone / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — timezone / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — action log / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — action log / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — action log / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — action log / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — action log / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — override reason / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — override reason / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — override reason / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — override reason / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — override reason / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — source lineage / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — source lineage / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — source lineage / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — source lineage / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — source lineage / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — retention / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — retention / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — retention / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — retention / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — retention / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — replay seed / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — replay seed / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — replay seed / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — replay seed / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — replay seed / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — aggregation code / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — aggregation code / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — aggregation code / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — aggregation code / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — aggregation code / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — display rounding / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — display rounding / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — display rounding / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — display rounding / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — display rounding / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — alert threshold / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — alert threshold / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — alert threshold / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — alert threshold / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — alert threshold / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — sampling frame / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — sampling frame / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — sampling frame / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — sampling frame / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — sampling frame / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — denominator reconciliation / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — denominator reconciliation / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — denominator reconciliation / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — denominator reconciliation / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — denominator reconciliation / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — privacy minimisation / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — privacy minimisation / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — privacy minimisation / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — privacy minimisation / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — privacy minimisation / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — access audit / normal volume.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — access audit / late-arrival replay.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — access audit / policy change.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — access audit / peak-load window.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Audit drill — access audit / manual override.** Record the precondition, raw count, unique zone-window count, excluded count, derived rate, owner, and replay artifact. A pass means the displayed number can be recomputed from versioned inputs; a mismatch is evidence to quarantine the decision, not a reason to overwrite history.
**Sensitivity trace 01.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 1%, and cohort weight 2/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 02.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 2%, and cohort weight 3/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 03.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 3, duplicate fraction 3%, and cohort weight 4/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 04.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 4, duplicate fraction 4%, and cohort weight 5/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 05.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 5, duplicate fraction 0%, and cohort weight 6/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 06.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 0, duplicate fraction 1%, and cohort weight 7/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 07.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 2%, and cohort weight 1/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 08.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 3%, and cohort weight 2/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 09.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 3, duplicate fraction 4%, and cohort weight 3/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 10.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 4, duplicate fraction 0%, and cohort weight 4/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 11.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 5, duplicate fraction 1%, and cohort weight 5/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 12.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 0, duplicate fraction 2%, and cohort weight 6/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 13.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 3%, and cohort weight 7/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 14.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 4%, and cohort weight 1/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 15.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 3, duplicate fraction 0%, and cohort weight 2/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 16.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 4, duplicate fraction 1%, and cohort weight 3/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 17.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 5, duplicate fraction 2%, and cohort weight 4/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 18.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 0, duplicate fraction 3%, and cohort weight 5/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 19.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 4%, and cohort weight 6/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 20.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 0%, and cohort weight 7/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 21.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 3, duplicate fraction 1%, and cohort weight 1/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 22.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 4, duplicate fraction 2%, and cohort weight 2/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 23.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 5, duplicate fraction 3%, and cohort weight 3/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 24.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 0, duplicate fraction 4%, and cohort weight 4/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 25.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 0%, and cohort weight 5/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 26.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 1%, and cohort weight 6/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 27.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 3, duplicate fraction 2%, and cohort weight 7/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 28.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 4, duplicate fraction 3%, and cohort weight 1/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 29.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 5, duplicate fraction 4%, and cohort weight 2/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 30.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 0, duplicate fraction 0%, and cohort weight 3/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 31.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 1%, and cohort weight 4/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 32.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 2%, and cohort weight 5/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 33.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 3, duplicate fraction 3%, and cohort weight 6/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 34.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 4, duplicate fraction 4%, and cohort weight 7/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 35.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 5, duplicate fraction 0%, and cohort weight 1/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 36.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 0, duplicate fraction 1%, and cohort weight 2/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 37.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 2%, and cohort weight 3/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 38.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 3%, and cohort weight 4/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 39.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 3, duplicate fraction 4%, and cohort weight 5/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 40.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 4, duplicate fraction 0%, and cohort weight 6/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 41.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 5, duplicate fraction 1%, and cohort weight 7/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 42.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 0, duplicate fraction 2%, and cohort weight 1/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 43.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 3%, and cohort weight 2/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 44.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 4%, and cohort weight 3/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 45.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 3, duplicate fraction 0%, and cohort weight 4/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 46.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 4, duplicate fraction 1%, and cohort weight 5/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 47.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 5, duplicate fraction 2%, and cohort weight 6/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 48.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 0, duplicate fraction 3%, and cohort weight 7/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 49.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 4%, and cohort weight 1/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 50.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 0%, and cohort weight 2/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 51.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 3, duplicate fraction 1%, and cohort weight 3/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 52.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 4, duplicate fraction 2%, and cohort weight 4/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 53.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 5, duplicate fraction 3%, and cohort weight 5/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 54.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 0, duplicate fraction 4%, and cohort weight 6/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 55.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 0%, and cohort weight 7/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 56.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 1%, and cohort weight 1/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 57.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 3, duplicate fraction 2%, and cohort weight 2/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 58.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 4, duplicate fraction 3%, and cohort weight 3/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 59.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 5, duplicate fraction 4%, and cohort weight 4/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 60.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 0, duplicate fraction 0%, and cohort weight 5/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 61.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 1%, and cohort weight 6/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 62.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 2%, and cohort weight 7/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 63.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 3, duplicate fraction 3%, and cohort weight 1/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 64.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 4, duplicate fraction 4%, and cohort weight 2/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 65.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 5, duplicate fraction 0%, and cohort weight 3/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 66.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 0, duplicate fraction 1%, and cohort weight 4/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 67.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 2%, and cohort weight 5/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 68.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 3%, and cohort weight 6/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 69.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 3, duplicate fraction 4%, and cohort weight 7/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 70.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 4, duplicate fraction 0%, and cohort weight 1/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 71.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 5, duplicate fraction 1%, and cohort weight 2/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 72.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 0, duplicate fraction 2%, and cohort weight 3/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 73.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 3%, and cohort weight 4/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 74.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 4%, and cohort weight 5/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 75.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 3, duplicate fraction 0%, and cohort weight 6/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 76.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 4, duplicate fraction 1%, and cohort weight 7/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 77.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 5, duplicate fraction 2%, and cohort weight 1/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 78.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 0, duplicate fraction 3%, and cohort weight 2/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 79.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 1, duplicate fraction 4%, and cohort weight 3/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.
**Sensitivity trace 80.** Hold the numerator definition fixed and vary one declared condition: label maturity bucket 2, duplicate fraction 0%, and cohort weight 4/10. Recompute only after documenting which records move; this exercise demonstrates directionality, not a forecast or observed effect.

**Decision rehearsal 001: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 001: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 001: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 001: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 002: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 002: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 002: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 002: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 003: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 003: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 003: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 003: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 004: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 004: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 004: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 004: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 005: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 005: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 005: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 005: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 006: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 006: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 006: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 006: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 007: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 007: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 007: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 007: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 008: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 008: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 008: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 008: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 009: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 009: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 009: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 009: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 010: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 010: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 010: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 010: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 011: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 011: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 011: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 011: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 012: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 012: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 012: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 012: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 013: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 013: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 013: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 013: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 014: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 014: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 014: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 014: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 015: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 015: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 015: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 015: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 016: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 016: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 016: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 016: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 017: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 017: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 017: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 017: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 018: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 018: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 018: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 018: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 019: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 019: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 019: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 019: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 020: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 020: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 020: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 020: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 021: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 021: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 021: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 021: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 022: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 022: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 022: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 022: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 023: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 023: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 023: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 023: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 024: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 024: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 024: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 024: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 025: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 025: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 025: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 025: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 026: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 026: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 026: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 026: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 027: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 027: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 027: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 027: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 028: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 028: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 028: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 028: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 029: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 029: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 029: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 029: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 030: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 030: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 030: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 030: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 031: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 031: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 031: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 031: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 032: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 032: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 032: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 032: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 033: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 033: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 033: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 033: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 034: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 034: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 034: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 034: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 035: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 035: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 035: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 035: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 036: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 036: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 036: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 036: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 037: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 037: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 037: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 037: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 038: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 038: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 038: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 038: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 039: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 039: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 039: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 039: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 040: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 040: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 040: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 040: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 041: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 041: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 041: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 041: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 042: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 042: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 042: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 042: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 043: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 043: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 043: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 043: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 044: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 044: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 044: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 044: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 045: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 045: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 045: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 045: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 046: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 046: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 046: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 046: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 047: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 047: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 047: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 047: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 048: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 048: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 048: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 048: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 049: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 049: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 049: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 049: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 050: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 050: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 050: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 050: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 051: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 051: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 051: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 051: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 052: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 052: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 052: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 052: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 053: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 053: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 053: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 053: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 054: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 054: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 054: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 054: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 055: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 055: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 055: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 055: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 056: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 056: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 056: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 056: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 057: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 057: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 057: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 057: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 058: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 058: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 058: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 058: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 059: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 059: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 059: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 059: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 060: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 060: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 060: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 060: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 061: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 061: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 061: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 061: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 062: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 062: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 062: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 062: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 063: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 063: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 063: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 063: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 064: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 064: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 064: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 064: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 065: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 065: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 065: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 065: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 066: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 066: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 066: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 066: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 067: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 067: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 067: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 067: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 068: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 068: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 068: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 068: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 069: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 069: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 069: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 069: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 070: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 070: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 070: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 070: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 071: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 071: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 071: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 071: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 072: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 072: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 072: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 072: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 073: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 073: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 073: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 073: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 074: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 074: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 074: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 074: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 075: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 075: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 075: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 075: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 076: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 076: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 076: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 076: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 077: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 077: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 077: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 077: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 078: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 078: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 078: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 078: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 079: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 079: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 079: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 079: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 080: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 080: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 080: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 080: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 081: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 081: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 081: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 081: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 082: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 082: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 082: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 082: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 083: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 083: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 083: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 083: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 084: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 084: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 084: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 084: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 085: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 085: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 085: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 085: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 086: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 086: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 086: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 086: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 087: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 087: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 087: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 087: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 088: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 088: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 088: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 088: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 089: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 089: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 089: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 089: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 090: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 090: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 090: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 090: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 091: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 091: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 091: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 091: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 092: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 092: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 092: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 092: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 093: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 093: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 093: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 093: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 094: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 094: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 094: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 094: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 095: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 095: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 095: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 095: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 096: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 096: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 096: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 096: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 097: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 097: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 097: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 097: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 098: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 098: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 098: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 098: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 099: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 099: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 099: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 099: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 100: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 100: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 100: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 100: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 101: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 101: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 101: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 101: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 102: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 102: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 102: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 102: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 103: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 103: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 103: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 103: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 104: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 104: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 104: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 104: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 105: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 105: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 105: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 105: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 106: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 106: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 106: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 106: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 107: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 107: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 107: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 107: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 108: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 108: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 108: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 108: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 109: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 109: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 109: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 109: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 110: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 110: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 110: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 110: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 111: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 111: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 111: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 111: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 112: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 112: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 112: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 112: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 113: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 113: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 113: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 113: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 114: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 114: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 114: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 114: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 115: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 115: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 115: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 115: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 116: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 116: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 116: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 116: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 117: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 117: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 117: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 117: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 118: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 118: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 118: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 118: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 119: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 119: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 119: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 119: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.
**Decision rehearsal 120: input contract.** Define one zone-window identifier, eligibility timestamp, and outcome timestamp before aggregation.
**Decision rehearsal 120: calculation.** Use M-01 with separately retained numerator and denominator; do not store only the ratio.
**Decision rehearsal 120: failure boundary.** If the record arrives after the cutoff, label it unresolved until the replay policy includes it.
**Decision rehearsal 120: action.** The owner either holds the prior policy, requests data repair, or approves a bounded experiment; the metric alone cannot choose among them.

**Control case 01: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 01: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 01: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 01: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 02: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 02: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 02: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 02: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 03: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 03: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 03: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 03: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 04: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 04: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 04: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 04: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 05: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 05: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 05: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 05: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 06: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 06: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 06: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 06: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 07: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 07: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 07: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 07: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 08: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 08: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 08: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 08: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 09: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 09: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 09: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 09: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 10: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 10: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 10: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 10: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 11: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 11: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 11: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 11: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 12: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 12: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 12: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 12: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 13: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 13: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 13: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 13: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 14: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 14: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 14: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 14: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 15: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 15: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 15: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 15: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 16: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 16: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 16: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 16: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 17: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 17: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 17: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 17: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 18: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 18: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 18: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 18: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 19: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 19: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 19: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 19: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 20: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 20: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 20: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 20: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 21: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 21: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 21: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 21: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 22: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 22: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 22: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 22: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 23: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 23: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 23: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 23: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 24: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 24: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 24: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 24: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 25: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 25: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 25: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 25: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 26: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 26: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 26: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 26: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 27: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 27: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 27: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 27: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 28: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 28: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 28: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 28: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 29: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 29: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 29: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 29: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 30: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 30: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 30: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 30: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 31: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 31: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 31: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 31: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 32: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 32: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 32: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 32: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 33: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 33: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 33: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 33: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 34: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 34: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 34: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 34: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 35: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 35: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 35: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 35: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 36: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 36: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 36: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 36: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 37: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 37: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 37: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 37: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 38: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 38: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 38: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 38: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 39: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 39: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 39: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 39: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.
**Control case 40: input.** Preserve the immutable decision identifier and the source timestamp; a display timestamp is not a substitute.
**Control case 40: test.** Rebuild the numerator and denominator from the retained ledger, then compare the result to the published aggregate at full precision.
**Control case 40: interpretation.** A mismatch can originate in membership, timing, duplicate handling, or calculation; it does not by itself prove a business change.
**Control case 40: containment.** Mark the affected window provisional, retain the prior action rule, and assign a replay owner with a bounded correction scope.

## 11. Interview / Exam Questions
**Q01. Prompt:** What is the decision grain?
**Answer:** State the population, cutoff, and action first. For this topic, use the ledger and M-01/M-02 rather than a raw dashboard total; check units and membership before interpreting a change.
**Follow-up:** Change the maturity window or cohort mix; a defensible answer explains whether the decision remains identified.
**Red flag / scoring:** 0 for an unexplained rate; 1 for arithmetic only; 2 for correct assumptions; 3 for an auditable trade-off and verification step.
**Q02. Prompt:** Why is event time different from ingestion time?
**Answer:** State the population, cutoff, and action first. For this topic, use the ledger and M-01/M-02 rather than a raw dashboard total; check units and membership before interpreting a change.
**Follow-up:** Change the maturity window or cohort mix; a defensible answer explains whether the decision remains identified.
**Red flag / scoring:** 0 for an unexplained rate; 1 for arithmetic only; 2 for correct assumptions; 3 for an auditable trade-off and verification step.
**Q03. Prompt:** Derive M-01 from indicators.
**Answer:** State the population, cutoff, and action first. For this topic, use the ledger and M-01/M-02 rather than a raw dashboard total; check units and membership before interpreting a change.
**Follow-up:** Change the maturity window or cohort mix; a defensible answer explains whether the decision remains identified.
**Red flag / scoring:** 0 for an unexplained rate; 1 for arithmetic only; 2 for correct assumptions; 3 for an auditable trade-off and verification step.
**Q04. Prompt:** Compute the rate for 72 of 80.
**Answer:** State the population, cutoff, and action first. For this topic, use the ledger and M-01/M-02 rather than a raw dashboard total; check units and membership before interpreting a change.
**Follow-up:** Change the maturity window or cohort mix; a defensible answer explains whether the decision remains identified.
**Red flag / scoring:** 0 for an unexplained rate; 1 for arithmetic only; 2 for correct assumptions; 3 for an auditable trade-off and verification step.
**Q05. Prompt:** Explain percentage-point versus relative change.
**Answer:** State the population, cutoff, and action first. For this topic, use the ledger and M-01/M-02 rather than a raw dashboard total; check units and membership before interpreting a change.
**Follow-up:** Change the maturity window or cohort mix; a defensible answer explains whether the decision remains identified.
**Red flag / scoring:** 0 for an unexplained rate; 1 for arithmetic only; 2 for correct assumptions; 3 for an auditable trade-off and verification step.
**Q06. Prompt:** Choose a metric when capacity is fixed.
**Answer:** State the population, cutoff, and action first. For this topic, use the ledger and M-01/M-02 rather than a raw dashboard total; check units and membership before interpreting a change.
**Follow-up:** Change the maturity window or cohort mix; a defensible answer explains whether the decision remains identified.
**Red flag / scoring:** 0 for an unexplained rate; 1 for arithmetic only; 2 for correct assumptions; 3 for an auditable trade-off and verification step.
**Q07. Prompt:** Diagnose a sudden rate increase with complaint growth.
**Answer:** State the population, cutoff, and action first. For this topic, use the ledger and M-01/M-02 rather than a raw dashboard total; check units and membership before interpreting a change.
**Follow-up:** Change the maturity window or cohort mix; a defensible answer explains whether the decision remains identified.
**Red flag / scoring:** 0 for an unexplained rate; 1 for arithmetic only; 2 for correct assumptions; 3 for an auditable trade-off and verification step.
**Q08. Prompt:** What does a zero denominator mean?
**Answer:** State the population, cutoff, and action first. For this topic, use the ledger and M-01/M-02 rather than a raw dashboard total; check units and membership before interpreting a change.
**Follow-up:** Change the maturity window or cohort mix; a defensible answer explains whether the decision remains identified.
**Red flag / scoring:** 0 for an unexplained rate; 1 for arithmetic only; 2 for correct assumptions; 3 for an auditable trade-off and verification step.
**Q09. Prompt:** How would you test duplicate handling?
**Answer:** State the population, cutoff, and action first. For this topic, use the ledger and M-01/M-02 rather than a raw dashboard total; check units and membership before interpreting a change.
**Follow-up:** Change the maturity window or cohort mix; a defensible answer explains whether the decision remains identified.
**Red flag / scoring:** 0 for an unexplained rate; 1 for arithmetic only; 2 for correct assumptions; 3 for an auditable trade-off and verification step.
**Q10. Prompt:** How would target weighting change a conclusion?
**Answer:** State the population, cutoff, and action first. For this topic, use the ledger and M-01/M-02 rather than a raw dashboard total; check units and membership before interpreting a change.
**Follow-up:** Change the maturity window or cohort mix; a defensible answer explains whether the decision remains identified.
**Red flag / scoring:** 0 for an unexplained rate; 1 for arithmetic only; 2 for correct assumptions; 3 for an auditable trade-off and verification step.
**Q11. Prompt:** What must be versioned for replay?
**Answer:** State the population, cutoff, and action first. For this topic, use the ledger and M-01/M-02 rather than a raw dashboard total; check units and membership before interpreting a change.
**Follow-up:** Change the maturity window or cohort mix; a defensible answer explains whether the decision remains identified.
**Red flag / scoring:** 0 for an unexplained rate; 1 for arithmetic only; 2 for correct assumptions; 3 for an auditable trade-off and verification step.
**Q12. Prompt:** When should an operator override the metric?
**Answer:** State the population, cutoff, and action first. For this topic, use the ledger and M-01/M-02 rather than a raw dashboard total; check units and membership before interpreting a change.
**Follow-up:** Change the maturity window or cohort mix; a defensible answer explains whether the decision remains identified.
**Red flag / scoring:** 0 for an unexplained rate; 1 for arithmetic only; 2 for correct assumptions; 3 for an auditable trade-off and verification step.
Takeaway: strong answers make the measurement contract explicit before proposing a model.

## 12. Quick Reference Card
| Topic | Dynamic-Pricing Guardrails |
| Core formula | M-01: eligible success / eligible units; require closed, deduplicated window. |
| Key assumption | Eligibility and outcome maturity are versioned at the decision cutoff. |
| Complexity | O(N) aggregate; identity retention dominates reconciliation. |
| Use when | market-clearing price adjustments |
| Avoid when | The population, outcome maturity, or action owner is unspecified. |
| Diagnostic | Compare unique IDs, eligibility reasons, label age, and cohort mix before acting. |
| Evidence | Illustrative arithmetic only; measure local workflow before thresholding. |
