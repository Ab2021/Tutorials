---
collection: "02_marketplace_seller_trust_and_economics"
file_id: "01"
topic: "Marketplace health and liquidity"
level: 2
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "A material change in event semantics, decision objective, or source access"
notation_style: "plain-text"
prerequisites: []
pattern_tags: ["commerce-decision", "measurement"]
source_status: "OPEN"
owner_role: "commerce data practitioner"
source_index: "INDEX.md"
---
# Marketplace health and liquidity

> **Purpose:** Select, measure, and diagnose decisions spanning marketplace flywheel, seller quality, catalogue governance, liquidity.
> **Prerequisites:** Basic probability, rates, joins, and time-window reasoning.
> **Key outcome:** Given a decision log and stated constraints, compute a bounded indicator, reject an invalid comparison, and choose a verification step.
> **This file does NOT cover:** Company-specific systems, legal advice, or measured production claims.
> **Level:** 2 — Core Methods
> **Version / reviewed:** 1.0 / not reviewed
> **Evidence:** DERIVED and ILLUSTRATIVE calculations; OPEN source-specific evidence.
> **Retrieval cue:** For a surprising metric, start at Section 8.

## 1. File Header
This is a reusable method sheet, not evidence that any marketplace deployed the described control. The unit of decision is one eligible commerce interaction or state transition, with a declared observation window.
**Takeaway:** A metric without an eligible unit, time boundary, and owner cannot safely choose an action.

## 2. Concept Map / Mental Model
```text
Observed event or state -> validate schema and eligibility -> compute denominator -> estimate signal -> choose bounded action
                                     |                                      |
                                     v                                      v
                              reject/repair input                    monitor guardrail and rollback
```
The map separates recording correctness from decision usefulness. A clean event can still be a poor proxy, while a useful proxy can be invalid if it is available after the decision.
**Takeaway:** First establish what existed at decision time; only then score or optimize it.

## 3. Foundational Definitions
### D-01 — eligible unit
**Meaning:** In marketplace health and liquidity, eligible unit is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-02 — exposure
**Meaning:** In marketplace health and liquidity, exposure is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-03 — conversion
**Meaning:** In marketplace health and liquidity, conversion is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-04 — availability
**Meaning:** In marketplace health and liquidity, availability is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-05 — contribution
**Meaning:** In marketplace health and liquidity, contribution is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-06 — freshness
**Meaning:** In marketplace health and liquidity, freshness is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-07 — false positive
**Meaning:** In marketplace health and liquidity, false positive is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-08 — false negative
**Meaning:** In marketplace health and liquidity, false negative is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-09 — guardrail
**Meaning:** In marketplace health and liquidity, guardrail is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-10 — snapshot
**Meaning:** In marketplace health and liquidity, snapshot is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-11 — late event
**Meaning:** In marketplace health and liquidity, late event is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-12 — counterfactual
**Meaning:** In marketplace health and liquidity, counterfactual is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-13 — review capacity
**Meaning:** In marketplace health and liquidity, review capacity is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-14 — tombstone
**Meaning:** In marketplace health and liquidity, tombstone is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-15 — idempotency
**Meaning:** In marketplace health and liquidity, idempotency is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-16 — selection bias
**Meaning:** In marketplace health and liquidity, selection bias is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-17 — calibration
**Meaning:** In marketplace health and liquidity, calibration is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-18 — latency budget
**Meaning:** In marketplace health and liquidity, latency budget is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-19 — reconciliation
**Meaning:** In marketplace health and liquidity, reconciliation is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
### D-20 — rollback
**Meaning:** In marketplace health and liquidity, rollback is defined operationally before it is counted.
**Type and units:** A scalar count, rate, currency amount, timestamp, or boolean state as declared by the owning contract.
**Boundary:** Do not substitute a similarly named dashboard field; join keys, eligibility, and window must match.
**Nonexample:** A post-decision outcome is not a valid prediction-time input merely because it shares the same entity key.
**Takeaway:** Definitions create the denominator and time boundary that make later arithmetic interpretable.

## 4. Core Formulas with Full Derivation
### 4.1 Base rate and decision value
Validity before use: the numerator and denominator refer to the same eligible population and closed window; missing records are reported rather than silently zero-filled.
Derivation: counting favorable eligible units C within eligible opportunities N gives a conditional empirical rate C/N.
```text
FORMULA [COM-01]: r = C / N
WHERE: r [dimensionless rate]; C [eligible favorable units]; N [eligible opportunities].
CONSTRAINT: N > 0; same entity population, eligibility rule, and observation window.
NUMERICAL EXAMPLE: C=84 completed actions; N=1,200 eligible actions; r=84/1,200=0.07=7%.
INTERPRETATION: 7% describes this constructed window, not a causal effect or a future guarantee.
```
Boundary/counterexample: if 200 unavailable offers were excluded from N after ranking, r cannot compare ranking policies without documenting that exclusion.
### 4.2 General expected net value
Validity before use: action outcomes and costs are mutually aligned to one unit and currency period.
Derivation: expected benefit is probability times benefit; subtract action and error costs.
```text
FORMULA [COM-02]: EV(a) = p(a) * B(a) - K(a) - (1-p(a)) * L(a)
WHERE: p(a) [probability]; B,K,L [currency per eligible action]; a [candidate action].
CONSTRAINT: probabilities are calibrated for the target population; costs exclude no material component.
NUMERICAL EXAMPLE: p=0.30, B=12 currency units, K=1, L=2 gives EV=3.6-1-1.4=1.2 units/action.
INTERPRETATION: act only if the comparable fallback has lower expected value and guardrails pass.
```
Boundary/counterexample: a score rank is not a probability; using it as p can reverse the selection.
### 4.3 Variant: freshness lag
```text
FORMULA [COM-03]: lag = t_observed - t_effective
WHERE: lag [seconds]; t_observed and t_effective [UTC timestamps].
CONSTRAINT: clocks, event semantics, and time zones are reconciled; negative lag is investigated.
NUMERICAL EXAMPLE: 10:05:30 minus 10:04:42 = 48 s.
INTERPRETATION: a 48 s lag may be acceptable only against an explicit decision SLA.
```
### 4.4 Limit and counterexample
Averages hide concentration. Two partitions with rates 1% and 9% cannot be summarized for policy selection without their denominators and exposure mix.
**Takeaway:** Rate, expected value, and lag become decision tools only after population alignment and a stated counterfactual.

## 5. Worked Numerical Example
**Evidence: ILLUSTRATIVE and VERIFIED hand arithmetic; not a company measurement.** A commerce operator considers escalating 120 eligible records. A reviewer can process 40/day. The estimated calibrated probability of a recoverable issue is 0.30; benefit is 12 units, review cost is 1 unit, and an unnecessary review carries 2 units of opportunity cost.
1. Capacity: 120 records / 40 records/day = 3 days; this exceeds a same-day decision requirement unless priority is used.
2. Per-record EV: 0.30*12 - 1 - 0.70*2 = 1.20 units.
3. For the 40 highest eligible records, illustrative projected net value = 40*1.20 = 48 units, conditional on calibration.
4. Sensitivity: if p falls to 0.20, EV = 2.40 - 1 - 1.60 = -0.20 units; escalation loses value before considering delay.
5. Independent check: 0.30+0.70=1 and 3.60-2.40=1.20; units are currency/action throughout.
The decision is not “review every flagged record”; it is “review only a calibrated, capacity-feasible subset while measuring outcomes.”
**Takeaway:** Sensitivity around the threshold is more valuable than reporting a single score or aggregate rate.

## 6. Visual / Geometric Intuition
Imagine a rectangle whose width is eligible exposure and whose height is outcome rate. The rectangle area is count. Changing a denominator changes width; changing selection changes which rectangle is measured.
A timeline adds a second constraint: features must lie left of the decision cut-off, while labels may mature to the right. A state machine adds a third: a terminal state must not be overwritten by late duplicate events.
**Takeaway:** Draw exposure, time, and state boundaries before trusting a chart or funnel.

## 7. Comparison / Selection Table
| Option | Use when | Benefit | Disqualifier | Fallback |
|---|---|---|---|---|
| Rule with audit | policy is explicit and volume is low | inspectable | threshold has no stable meaning | sampled manual review |
| Calibrated score | labels and point-in-time inputs exist | ranks finite capacity | drift or leakage is untested | conservative queue |
| Experiment | action can be randomized safely | causal comparison | interference dominates | phased observational study |
| Reconciliation control | state disagreement is costly | catches silent loss | no authoritative source | quarantine and investigate |
| Cache/index overlay | freshness matters | fast reads | version fence absent | serve older immutable snapshot |
Selection is conditional: choose the simplest mechanism that meets the owner’s objective and retains a measurable rollback.
**Takeaway:** A more complex method is not an upgrade if its data contract or counterfactual is weaker.

## 8. Failure Modes & Edge Cases
### F-01 — rate rises
**Category:** VIOLATED_ASSUMPTION. **Competing causes:** denominator shrinkage or genuine improvement.
**Discriminating diagnostic / expected signal:** compare eligible counts and exclusion reasons.
**Correction:** freeze definition; recompute cohorts. **Mitigation risk:** a freeze can delay needed response.
**Verification:** backfill audit and matched-window replay. **Provenance:** HYPOTHETICAL teaching diagnostic.
### F-02 — score looks excellent offline
**Category:** VIOLATED_ASSUMPTION. **Competing causes:** leakage or easy historical split.
**Discriminating diagnostic / expected signal:** trace feature availability to cutoff.
**Correction:** remove future fields; use time split. **Mitigation risk:** performance can fall sharply.
**Verification:** holdout with feature-time assertions. **Provenance:** HYPOTHETICAL teaching diagnostic.
### F-03 — freshness lag spikes
**Category:** VIOLATED_ASSUMPTION. **Competing causes:** upstream delay or consumer backlog.
**Discriminating diagnostic / expected signal:** compare producer timestamp, ingest timestamp, and queue depth.
**Correction:** shed noncritical work; repair consumer. **Mitigation risk:** shedding can reduce coverage.
**Verification:** replay idempotently and reconcile. **Provenance:** HYPOTHETICAL teaching diagnostic.
### F-04 — conversion changes
**Category:** VIOLATED_ASSUMPTION. **Competing causes:** traffic mix shift or treatment effect.
**Discriminating diagnostic / expected signal:** stratify exposure and inspect assignment balance.
**Correction:** pause rollout; analyze randomized cohorts. **Mitigation risk:** pause loses learning.
**Verification:** guardrail and sample-ratio checks. **Provenance:** HYPOTHETICAL teaching diagnostic.
### F-05 — duplicate action
**Category:** VIOLATED_ASSUMPTION. **Competing causes:** retry race or missing idempotency key.
**Discriminating diagnostic / expected signal:** count repeated keys across retry attempts.
**Correction:** deduplicate at write boundary. **Mitigation risk:** dedup can hide legitimate updates.
**Verification:** immutable event audit. **Provenance:** HYPOTHETICAL teaching diagnostic.
### F-06 — inventory/state is negative
**Category:** VIOLATED_ASSUMPTION. **Competing causes:** out-of-order event or double decrement.
**Discriminating diagnostic / expected signal:** compare sequence/version to authority.
**Correction:** quarantine affected item; resync. **Mitigation risk:** resync may overwrite concurrent update.
**Verification:** version-fenced replay. **Provenance:** HYPOTHETICAL teaching diagnostic.
### F-07 — review queue overflows
**Category:** VIOLATED_ASSUMPTION. **Competing causes:** threshold drift or capacity reduction.
**Discriminating diagnostic / expected signal:** measure arrivals, completions, and age.
**Correction:** raise threshold temporarily with owner approval. **Mitigation risk:** missed true cases rise.
**Verification:** daily capacity and outcome audit. **Provenance:** HYPOTHETICAL teaching diagnostic.
### F-08 — search/recommendation is stale
**Category:** VIOLATED_ASSUMPTION. **Competing causes:** snapshot publication lag or cache key mismatch.
**Discriminating diagnostic / expected signal:** compare version headers end-to-end.
**Correction:** serve prior complete snapshot. **Mitigation risk:** older content persists.
**Verification:** atomic manifest and tombstone test. **Provenance:** HYPOTHETICAL teaching diagnostic.
**Takeaway:** A symptom is not a root cause; use a signal that separates at least two plausible mechanisms.

## 9. Complexity & Cost Analysis
For N eligible records, one pass that validates required fields is O(N) time and O(1) additional state excluding error records. Sorting M scored eligible records for a finite queue costs O(M log M); maintaining a size-K heap costs O(M log K) time and O(K) space.
Illustrative sizing: 1,200 events/minute at 600 bytes/event gives 720,000 bytes/minute payload. A 15-minute replay buffer holds 10,800,000 bytes payload-only, about 10.30 MiB; metadata, replication, and compression are deliberately excluded.
A capacity calculation is a workload model, not a latency benchmark. Recheck with observed payload percentiles, retries, and partition skew before provisioning.
**Takeaway:** State the operation mix and excluded overhead; asymptotic cost does not certify a service-level objective.

## 10. Key Theorems / Results Reference
1. Linearity of expectation permits summing stated per-action expected values without independence; it does not make probability estimates calibrated.
2. Bayes’ rule conditions a rate on an observation; base rates remain material for review precision.
3. Conservation of state requires every accepted transition to be represented once in the authoritative history; exactly-once delivery is not assumed.
4. Randomized assignment supports causal comparison only for the assigned population and intact assignment/measurement process.
**Takeaway:** Name the assumption that licenses a result; a theorem label is not a production guarantee.

## 11. Interview / Exam Questions
### Q01. Why is a click rate not enough?
**Answer:** Its denominator may omit unavailable or ineligible exposure, and clicks can conflict with completed, profitable, or trusted outcomes.
**Reasoning:** Tie the answer to population, time, state, or capacity rather than a tool name.
**Follow-up:** State the one measurement that could overturn this answer.
### Q02. Compute the example expected value.
**Answer:** 0.30*12-1-0.70*2=1.20 currency units per action; it is conditional on calibrated probability and stated costs.
**Reasoning:** Tie the answer to population, time, state, or capacity rather than a tool name.
**Follow-up:** State the one measurement that could overturn this answer.
### Q03. What invalidates that calculation?
**Answer:** A score used as an uncalibrated probability, costs from another population, or a review action unavailable at decision time.
**Reasoning:** Tie the answer to population, time, state, or capacity rather than a tool name.
**Follow-up:** State the one measurement that could overturn this answer.
### Q04. How would you diagnose a sudden rate jump?
**Answer:** First compare numerator, denominator, eligibility exclusions, traffic mix, and event-version changes before attributing it to a product change.
**Reasoning:** Tie the answer to population, time, state, or capacity rather than a tool name.
**Follow-up:** State the one measurement that could overturn this answer.
### Q05. Why use a time split?
**Answer:** It prevents later information and future population structure from contaminating a prediction-time evaluation.
**Reasoning:** Tie the answer to population, time, state, or capacity rather than a tool name.
**Follow-up:** State the one measurement that could overturn this answer.
### Q06. When does a simple rule win?
**Answer:** When policy is explicit, labels are unreliable, volume is manageable, and auditability outweighs marginal modeling value.
**Reasoning:** Tie the answer to population, time, state, or capacity rather than a tool name.
**Follow-up:** State the one measurement that could overturn this answer.
### Q07. How do retries cause business harm?
**Answer:** They may duplicate charges, decrements, messages, or decisions unless an idempotency boundary owns the effect.
**Reasoning:** Tie the answer to population, time, state, or capacity rather than a tool name.
**Follow-up:** State the one measurement that could overturn this answer.
### Q08. What does freshness not prove?
**Answer:** A recent timestamp does not prove correctness, completeness, authorization, or suitability for a downstream decision.
**Reasoning:** Tie the answer to population, time, state, or capacity rather than a tool name.
**Follow-up:** State the one measurement that could overturn this answer.
### Q09. What changes under half the review capacity?
**Answer:** Recompute queue feasibility and the threshold; do not retain the old score cutoff by habit.
**Reasoning:** Tie the answer to population, time, state, or capacity rather than a tool name.
**Follow-up:** State the one measurement that could overturn this answer.
### Q10. What is an acceptable alternative to randomization?
**Answer:** A carefully bounded observational design can inform hypotheses, but causal conclusions remain weaker and need explicit confounding checks.
**Reasoning:** Tie the answer to population, time, state, or capacity rather than a tool name.
**Follow-up:** State the one measurement that could overturn this answer.
### Q11. How do you verify a rollback?
**Answer:** Exercise the rollback on a versioned snapshot, then reconcile state and guardrail metrics against the authoritative history.
**Reasoning:** Tie the answer to population, time, state, or capacity rather than a tool name.
**Follow-up:** State the one measurement that could overturn this answer.
### Q12. What should be logged?
**Answer:** Decision input version, eligibility, action, timestamp, owner, outcome maturity, and a correlation/idempotency key.
**Reasoning:** Tie the answer to population, time, state, or capacity rather than a tool name.
**Follow-up:** State the one measurement that could overturn this answer.
**Takeaway:** Strong answers expose assumptions, specify a discriminating check, and name the action owner.

### Applied check 001 — eligible unit during normalization
**Question:** Does the eligible unit record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 002 — exposure during eligibility
**Question:** Does the exposure record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 003 — conversion during scoring
**Question:** Does the conversion record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 004 — availability during action
**Question:** Does the availability record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 005 — contribution during reconciliation
**Question:** Does the contribution record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 006 — freshness during intake
**Question:** Does the freshness record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 007 — false positive during normalization
**Question:** Does the false positive record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 008 — false negative during eligibility
**Question:** Does the false negative record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 009 — guardrail during scoring
**Question:** Does the guardrail record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 010 — snapshot during action
**Question:** Does the snapshot record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 011 — late event during reconciliation
**Question:** Does the late event record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 012 — counterfactual during intake
**Question:** Does the counterfactual record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 013 — review capacity during normalization
**Question:** Does the review capacity record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 014 — tombstone during eligibility
**Question:** Does the tombstone record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 015 — idempotency during scoring
**Question:** Does the idempotency record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 016 — selection bias during action
**Question:** Does the selection bias record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 017 — calibration during reconciliation
**Question:** Does the calibration record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 018 — latency budget during intake
**Question:** Does the latency budget record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 019 — reconciliation during normalization
**Question:** Does the reconciliation record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 020 — rollback during eligibility
**Question:** Does the rollback record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 021 — eligible unit during scoring
**Question:** Does the eligible unit record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 022 — exposure during action
**Question:** Does the exposure record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 023 — conversion during reconciliation
**Question:** Does the conversion record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 024 — availability during intake
**Question:** Does the availability record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 025 — contribution during normalization
**Question:** Does the contribution record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 026 — freshness during eligibility
**Question:** Does the freshness record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 027 — false positive during scoring
**Question:** Does the false positive record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 028 — false negative during action
**Question:** Does the false negative record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 029 — guardrail during reconciliation
**Question:** Does the guardrail record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 030 — snapshot during intake
**Question:** Does the snapshot record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 031 — late event during normalization
**Question:** Does the late event record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 032 — counterfactual during eligibility
**Question:** Does the counterfactual record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 033 — review capacity during scoring
**Question:** Does the review capacity record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 034 — tombstone during action
**Question:** Does the tombstone record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 035 — idempotency during reconciliation
**Question:** Does the idempotency record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 036 — selection bias during intake
**Question:** Does the selection bias record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 037 — calibration during normalization
**Question:** Does the calibration record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 038 — latency budget during eligibility
**Question:** Does the latency budget record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 039 — reconciliation during scoring
**Question:** Does the reconciliation record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 040 — rollback during action
**Question:** Does the rollback record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 041 — eligible unit during reconciliation
**Question:** Does the eligible unit record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 042 — exposure during intake
**Question:** Does the exposure record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 043 — conversion during normalization
**Question:** Does the conversion record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 044 — availability during eligibility
**Question:** Does the availability record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 045 — contribution during scoring
**Question:** Does the contribution record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 046 — freshness during action
**Question:** Does the freshness record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 047 — false positive during reconciliation
**Question:** Does the false positive record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 048 — false negative during intake
**Question:** Does the false negative record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 049 — guardrail during normalization
**Question:** Does the guardrail record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 050 — snapshot during eligibility
**Question:** Does the snapshot record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 051 — late event during scoring
**Question:** Does the late event record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 052 — counterfactual during action
**Question:** Does the counterfactual record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 053 — review capacity during reconciliation
**Question:** Does the review capacity record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 054 — tombstone during intake
**Question:** Does the tombstone record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 055 — idempotency during normalization
**Question:** Does the idempotency record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 056 — selection bias during eligibility
**Question:** Does the selection bias record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 057 — calibration during scoring
**Question:** Does the calibration record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 058 — latency budget during action
**Question:** Does the latency budget record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 059 — reconciliation during reconciliation
**Question:** Does the reconciliation record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 060 — rollback during intake
**Question:** Does the rollback record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 061 — eligible unit during normalization
**Question:** Does the eligible unit record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 062 — exposure during eligibility
**Question:** Does the exposure record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 063 — conversion during scoring
**Question:** Does the conversion record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 064 — availability during action
**Question:** Does the availability record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 065 — contribution during reconciliation
**Question:** Does the contribution record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 066 — freshness during intake
**Question:** Does the freshness record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 067 — false positive during normalization
**Question:** Does the false positive record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 068 — false negative during eligibility
**Question:** Does the false negative record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 069 — guardrail during scoring
**Question:** Does the guardrail record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 070 — snapshot during action
**Question:** Does the snapshot record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 071 — late event during reconciliation
**Question:** Does the late event record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 072 — counterfactual during intake
**Question:** Does the counterfactual record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 073 — review capacity during normalization
**Question:** Does the review capacity record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 074 — tombstone during eligibility
**Question:** Does the tombstone record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 075 — idempotency during scoring
**Question:** Does the idempotency record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 076 — selection bias during action
**Question:** Does the selection bias record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 077 — calibration during reconciliation
**Question:** Does the calibration record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 078 — latency budget during intake
**Question:** Does the latency budget record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 079 — reconciliation during normalization
**Question:** Does the reconciliation record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 080 — rollback during eligibility
**Question:** Does the rollback record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 081 — eligible unit during scoring
**Question:** Does the eligible unit record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 082 — exposure during action
**Question:** Does the exposure record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 083 — conversion during reconciliation
**Question:** Does the conversion record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 084 — availability during intake
**Question:** Does the availability record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 085 — contribution during normalization
**Question:** Does the contribution record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 086 — freshness during eligibility
**Question:** Does the freshness record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 087 — false positive during scoring
**Question:** Does the false positive record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 088 — false negative during action
**Question:** Does the false negative record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 089 — guardrail during reconciliation
**Question:** Does the guardrail record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 090 — snapshot during intake
**Question:** Does the snapshot record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 091 — late event during normalization
**Question:** Does the late event record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 092 — counterfactual during eligibility
**Question:** Does the counterfactual record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 093 — review capacity during scoring
**Question:** Does the review capacity record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 094 — tombstone during action
**Question:** Does the tombstone record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 095 — idempotency during reconciliation
**Question:** Does the idempotency record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 096 — selection bias during intake
**Question:** Does the selection bias record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 097 — calibration during normalization
**Question:** Does the calibration record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 098 — latency budget during eligibility
**Question:** Does the latency budget record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 099 — reconciliation during scoring
**Question:** Does the reconciliation record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 100 — rollback during action
**Question:** Does the rollback record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 101 — eligible unit during reconciliation
**Question:** Does the eligible unit record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 102 — exposure during intake
**Question:** Does the exposure record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 103 — conversion during normalization
**Question:** Does the conversion record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 104 — availability during eligibility
**Question:** Does the availability record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 105 — contribution during scoring
**Question:** Does the contribution record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 106 — freshness during action
**Question:** Does the freshness record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 107 — false positive during reconciliation
**Question:** Does the false positive record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 108 — false negative during intake
**Question:** Does the false negative record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 109 — guardrail during normalization
**Question:** Does the guardrail record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 110 — snapshot during eligibility
**Question:** Does the snapshot record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 111 — late event during scoring
**Question:** Does the late event record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 112 — counterfactual during action
**Question:** Does the counterfactual record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 113 — review capacity during reconciliation
**Question:** Does the review capacity record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 114 — tombstone during intake
**Question:** Does the tombstone record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 115 — idempotency during normalization
**Question:** Does the idempotency record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 116 — selection bias during eligibility
**Question:** Does the selection bias record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 117 — calibration during scoring
**Question:** Does the calibration record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 118 — latency budget during action
**Question:** Does the latency budget record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 119 — reconciliation during reconciliation
**Question:** Does the reconciliation record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 120 — rollback during intake
**Question:** Does the rollback record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 121 — eligible unit during normalization
**Question:** Does the eligible unit record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 122 — exposure during eligibility
**Question:** Does the exposure record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 123 — conversion during scoring
**Question:** Does the conversion record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 124 — availability during action
**Question:** Does the availability record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 125 — contribution during reconciliation
**Question:** Does the contribution record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 126 — freshness during intake
**Question:** Does the freshness record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 127 — false positive during normalization
**Question:** Does the false positive record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 128 — false negative during eligibility
**Question:** Does the false negative record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 129 — guardrail during scoring
**Question:** Does the guardrail record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 130 — snapshot during action
**Question:** Does the snapshot record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 131 — late event during reconciliation
**Question:** Does the late event record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 132 — counterfactual during intake
**Question:** Does the counterfactual record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 133 — review capacity during normalization
**Question:** Does the review capacity record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 134 — tombstone during eligibility
**Question:** Does the tombstone record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 135 — idempotency during scoring
**Question:** Does the idempotency record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 136 — selection bias during action
**Question:** Does the selection bias record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 137 — calibration during reconciliation
**Question:** Does the calibration record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 138 — latency budget during intake
**Question:** Does the latency budget record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 139 — reconciliation during normalization
**Question:** Does the reconciliation record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 140 — rollback during eligibility
**Question:** Does the rollback record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 141 — eligible unit during scoring
**Question:** Does the eligible unit record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 142 — exposure during action
**Question:** Does the exposure record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 143 — conversion during reconciliation
**Question:** Does the conversion record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 144 — availability during intake
**Question:** Does the availability record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 145 — contribution during normalization
**Question:** Does the contribution record retain its unit, owner, version, and decision-time timestamp during normalization?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 146 — freshness during eligibility
**Question:** Does the freshness record retain its unit, owner, version, and decision-time timestamp during eligibility?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 147 — false positive during scoring
**Question:** Does the false positive record retain its unit, owner, version, and decision-time timestamp during scoring?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 148 — false negative during action
**Question:** Does the false negative record retain its unit, owner, version, and decision-time timestamp during action?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 149 — guardrail during reconciliation
**Question:** Does the guardrail record retain its unit, owner, version, and decision-time timestamp during reconciliation?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
### Applied check 150 — snapshot during intake
**Question:** Does the snapshot record retain its unit, owner, version, and decision-time timestamp during intake?
**Method:** Sample the immutable record, compare its declared contract to the derived output, and record any transformation or exclusion.
**Expected result:** The output is either traceable to a valid input or explicitly quarantined; an untraceable value cannot support an automated action.
**Decision consequence:** If the check fails, use the documented fallback and open a bounded correction with verification rather than silently imputing certainty.
## 12. Quick Reference Card
| Field | Card |
|---|---|
| Decision | Marketplace health and liquidity: choose a bounded action from aligned commerce evidence |
| Input | Eligible unit, decision-time snapshot, numerator, denominator, owner, and time window |
| Core rule | Compute COM-01 before comparing; use COM-02 only with calibrated probability and aligned costs |
| Guardrail | Availability, state reconciliation, capacity, latency, and adverse-slice checks |
| Avoid when | Eligibility, time semantics, or authoritative state are unknown |
| Verify | Replay an immutable sample, recompute arithmetic, and inspect failed paths |
| Evidence | ILLUSTRATIVE/DERIVED; no company outcome asserted |
