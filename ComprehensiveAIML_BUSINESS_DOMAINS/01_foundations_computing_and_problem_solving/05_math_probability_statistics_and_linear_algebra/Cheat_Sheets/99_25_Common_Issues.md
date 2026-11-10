---
collection: "05_math_probability_statistics_and_linear_algebra"
file_id: "99"
topic: "25 Common Issues"
level: 3
status: "draft"
version: "1.0"
created: "2026-11-10"
reviewed: null
review_trigger: "Any change to a collection decision contract"
notation_style: "plain-text"
prerequisites: []
pattern_tags: ["troubleshooting"]
source_status: "OPEN"
owner_role: "curriculum maintainer"
source_index: "INDEX.md"
---

# 25 Common Issues — 05_math_probability_statistics_and_linear_algebra

> **Purpose:** Diagnose recurring failures without converting illustrative examples into deployment claims.
> **Evidence:** All issues are HYPOTHETICAL teaching scenarios unless a linked manuscript provides separate evidence.

## 1. File Header
Scope: input contracts, arithmetic, scale, and decision authority. Reader task: isolate a symptom, discriminate competing causes, apply a bounded correction, and verify recovery. Issue count: 25. Provenance policy: HYPOTHETICAL is not observed incident evidence.

## 2. Concept Map / Mental Model
```text
symptom -> classify category -> inspect contract and signals -> contain safely -> correct cause -> replay verification
```
Takeaway: diagnose evidence and mechanism before choosing a fix.

## 3. Foundational Definitions
A symptom is observable behavior; a cause is a mechanism; a diagnostic distinguishes plausible causes; mitigation contains harm; correction removes the cause; verification is a repeatable check. Severity is the consequence within the declared scope, not an incident claim.
Takeaway: names prevent a workaround from being reported as a fix.

## 4. Core Formulas with Full Derivation
Validity: an initially empty fluid queue with constant rates and no loss/retries. Derivation: conservation gives backlog rate λ−μ; integrate positive imbalance over time.
```text
FORMULA [ISS-QUEUE]: B=max(0,λ−μ)t
WHERE: B [items], λ and μ [items/s], t [s].
CONSTRAINT: planning model only.
NUMERICAL EXAMPLE: (120−100)×360=7,200 items.
INTERPRETATION: positive backlog requires a later service surplus to drain.
```
Takeaway: use units and assumptions to rule out false diagnoses.

## 5. Worked Numerical Example
With B=7,200 items, recovery service 140 items/s, and arrivals 80 items/s, T=B/(140−80)=120 s. Check: 60×120=7,200. Boundary: if recovery service equals arrivals, no positive backlog drains. This arithmetic is VERIFIED for illustrative inputs only.
Takeaway: a capacity calculation diagnoses an imbalance, not a latency guarantee.

## 6. Visual / Geometric Intuition
```text
valid input -> bounded operation -> recorded decision
     | failure          |
     +-> defer + reason + owner review
```
Takeaway: a safe fallback is part of the diagnostic design.

## 7. Comparison / Selection Table
| Condition | First action | Disqualifier | Verification |
|---|---|---|---|
| malformed or late input | defer | do not repair silently | ledger reason and replay |
| rate imbalance | cap or add measured capacity | do not promise drain at zero surplus | backlog slope changes |
| duplicate effect | return prior result | do not retry blindly | one final effect per key |
Takeaway: select a response from observed evidence, not a familiar slogan.

## 8. Failure Modes & Edge Cases
| ID | Category | Symptom | Root cause | Diagnostic / expected signal | Fix | Provenance | Related file |
|---|---|---|---|---|---|---|---|
| ISSUE-001 | VIOLATED_ASSUMPTION | Input arrives after its decision cutoff | clock skew, delayed extraction, or a wrong timezone | compare source event time, ingestion time, and cutoff in UTC | defer the item; correct timestamp ownership and replay | HYPOTHETICAL | collection contracts |
### ISSUE-001 — Input arrives after its decision cutoff
**Trigger/environment:** A bounded teaching workflow exposes `input arrives after its decision cutoff`.
**Symptom:** Input arrives after its decision cutoff.
**Competing causes:** clock skew, delayed extraction, or a wrong timezone.
**Discriminating diagnostic:** compare source event time, ingestion time, and cutoff in UTC.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** defer the item; correct timestamp ownership and replay.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-002 | NUMERICAL_EDGE | A rate denominator is zero | empty window or failed counter | inspect numerator and denominator separately | emit undefined rather than zero; define empty-window policy | HYPOTHETICAL | collection contracts |
### ISSUE-002 — A rate denominator is zero
**Trigger/environment:** A bounded teaching workflow exposes `a rate denominator is zero`.
**Symptom:** A rate denominator is zero.
**Competing causes:** empty window or failed counter.
**Discriminating diagnostic:** inspect numerator and denominator separately.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** emit undefined rather than zero; define empty-window policy.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-003 | SCALE_FAILURE | Queue grows without bound | arrival rate exceeds sustainable service | measure λ, μ, and backlog slope | cap intake or add measured service; verify recovery surplus | HYPOTHETICAL | collection contracts |
### ISSUE-003 — Queue grows without bound
**Trigger/environment:** A bounded teaching workflow exposes `queue grows without bound`.
**Symptom:** Queue grows without bound.
**Competing causes:** arrival rate exceeds sustainable service.
**Discriminating diagnostic:** measure λ, μ, and backlog slope.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** cap intake or add measured service; verify recovery surplus.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-004 | CONCEPT_CONFUSION | A score is treated as authority | threshold confused with an approval policy | audit actor, policy version, and final action | require named owner approval and retain reason code | HYPOTHETICAL | collection contracts |
### ISSUE-004 — A score is treated as authority
**Trigger/environment:** A bounded teaching workflow exposes `a score is treated as authority`.
**Symptom:** A score is treated as authority.
**Competing causes:** threshold confused with an approval policy.
**Discriminating diagnostic:** audit actor, policy version, and final action.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** require named owner approval and retain reason code.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-005 | VIOLATED_ASSUMPTION | A retry creates a duplicate effect | missing or unstable idempotency key | replay same request key and count final effects | return prior result; persist deduplication before effect | HYPOTHETICAL | collection contracts |
### ISSUE-005 — A retry creates a duplicate effect
**Trigger/environment:** A bounded teaching workflow exposes `a retry creates a duplicate effect`.
**Symptom:** A retry creates a duplicate effect.
**Competing causes:** missing or unstable idempotency key.
**Discriminating diagnostic:** replay same request key and count final effects.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** return prior result; persist deduplication before effect.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-006 | NUMERICAL_EDGE | Bytes are reported as items | unit conversion omitted | dimensional check every multiplication | label units and distinguish MB from MiB | HYPOTHETICAL | collection contracts |
### ISSUE-006 — Bytes are reported as items
**Trigger/environment:** A bounded teaching workflow exposes `bytes are reported as items`.
**Symptom:** Bytes are reported as items.
**Competing causes:** unit conversion omitted.
**Discriminating diagnostic:** dimensional check every multiplication.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** label units and distinguish MB from MiB.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-007 | SCALE_FAILURE | A hot key dominates latency | skewed workload or single partition | compare p50 and p99 by key | shard carefully or bound per-key work | HYPOTHETICAL | collection contracts |
### ISSUE-007 — A hot key dominates latency
**Trigger/environment:** A bounded teaching workflow exposes `a hot key dominates latency`.
**Symptom:** A hot key dominates latency.
**Competing causes:** skewed workload or single partition.
**Discriminating diagnostic:** compare p50 and p99 by key.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** shard carefully or bound per-key work.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-008 | CONCEPT_CONFUSION | A planned test is called a result | proposal confused with observation | inspect artifact, window, and comparator | label not-measured and write validation protocol | HYPOTHETICAL | collection contracts |
### ISSUE-008 — A planned test is called a result
**Trigger/environment:** A bounded teaching workflow exposes `a planned test is called a result`.
**Symptom:** A planned test is called a result.
**Competing causes:** proposal confused with observation.
**Discriminating diagnostic:** inspect artifact, window, and comparator.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** label not-measured and write validation protocol.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-009 | VIOLATED_ASSUMPTION | A feature leaks future information | join ignores prediction-time availability | trace each field to earliest timestamp | remove field or use point-in-time join | HYPOTHETICAL | collection contracts |
### ISSUE-009 — A feature leaks future information
**Trigger/environment:** A bounded teaching workflow exposes `a feature leaks future information`.
**Symptom:** A feature leaks future information.
**Competing causes:** join ignores prediction-time availability.
**Discriminating diagnostic:** trace each field to earliest timestamp.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** remove field or use point-in-time join.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-010 | NUMERICAL_EDGE | Rounding reverses a boundary decision | threshold too close to displayed precision | compute with unrounded values | retain precision and specify tie policy | HYPOTHETICAL | collection contracts |
### ISSUE-010 — Rounding reverses a boundary decision
**Trigger/environment:** A bounded teaching workflow exposes `rounding reverses a boundary decision`.
**Symptom:** Rounding reverses a boundary decision.
**Competing causes:** threshold too close to displayed precision.
**Discriminating diagnostic:** compute with unrounded values.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** retain precision and specify tie policy.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-011 | SCALE_FAILURE | Logging exhausts storage during an incident | unbounded payload or retry logging | measure bytes/event and retention | sample safely and set quotas | HYPOTHETICAL | collection contracts |
### ISSUE-011 — Logging exhausts storage during an incident
**Trigger/environment:** A bounded teaching workflow exposes `logging exhausts storage during an incident`.
**Symptom:** Logging exhausts storage during an incident.
**Competing causes:** unbounded payload or retry logging.
**Discriminating diagnostic:** measure bytes/event and retention.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** sample safely and set quotas.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-012 | CONCEPT_CONFUSION | A baseline is absent | new method compared only with itself | request no-change or manual comparator | restore baseline and measure both paths | HYPOTHETICAL | collection contracts |
### ISSUE-012 — A baseline is absent
**Trigger/environment:** A bounded teaching workflow exposes `a baseline is absent`.
**Symptom:** A baseline is absent.
**Competing causes:** new method compared only with itself.
**Discriminating diagnostic:** request no-change or manual comparator.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** restore baseline and measure both paths.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-013 | VIOLATED_ASSUMPTION | Configuration changes without version trace | mutable default or manual edit | compare run record and configuration hash | pin config, approve change, and rerun oracle | HYPOTHETICAL | collection contracts |
### ISSUE-013 — Configuration changes without version trace
**Trigger/environment:** A bounded teaching workflow exposes `configuration changes without version trace`.
**Symptom:** Configuration changes without version trace.
**Competing causes:** mutable default or manual edit.
**Discriminating diagnostic:** compare run record and configuration hash.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** pin config, approve change, and rerun oracle.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-014 | NUMERICAL_EDGE | Negative duration appears | end time precedes start time | validate ordering and timezone | reject or quarantine malformed record | HYPOTHETICAL | collection contracts |
### ISSUE-014 — Negative duration appears
**Trigger/environment:** A bounded teaching workflow exposes `negative duration appears`.
**Symptom:** Negative duration appears.
**Competing causes:** end time precedes start time.
**Discriminating diagnostic:** validate ordering and timezone.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** reject or quarantine malformed record.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-015 | SCALE_FAILURE | Fan-out amplifies one request | unbounded downstream calls | count calls per input and dependency saturation | set budgets and batch or cache | HYPOTHETICAL | collection contracts |
### ISSUE-015 — Fan-out amplifies one request
**Trigger/environment:** A bounded teaching workflow exposes `fan-out amplifies one request`.
**Symptom:** Fan-out amplifies one request.
**Competing causes:** unbounded downstream calls.
**Discriminating diagnostic:** count calls per input and dependency saturation.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** set budgets and batch or cache.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-016 | CONCEPT_CONFUSION | Correlation is used as causation | selection or intervention confounding | draw causal path and comparator | narrow claim or design appropriate experiment | HYPOTHETICAL | collection contracts |
### ISSUE-016 — Correlation is used as causation
**Trigger/environment:** A bounded teaching workflow exposes `correlation is used as causation`.
**Symptom:** Correlation is used as causation.
**Competing causes:** selection or intervention confounding.
**Discriminating diagnostic:** draw causal path and comparator.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** narrow claim or design appropriate experiment.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-017 | VIOLATED_ASSUMPTION | Schema evolution silently changes meaning | field reused with different units | compare schema version and semantic contract | version field and migrate explicitly | HYPOTHETICAL | collection contracts |
### ISSUE-017 — Schema evolution silently changes meaning
**Trigger/environment:** A bounded teaching workflow exposes `schema evolution silently changes meaning`.
**Symptom:** Schema evolution silently changes meaning.
**Competing causes:** field reused with different units.
**Discriminating diagnostic:** compare schema version and semantic contract.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** version field and migrate explicitly.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-018 | NUMERICAL_EDGE | Integer overflow corrupts totals | narrow accumulator or unchecked conversion | test max values and use wide reference arithmetic | widen type and reject unsafe conversion | HYPOTHETICAL | collection contracts |
### ISSUE-018 — Integer overflow corrupts totals
**Trigger/environment:** A bounded teaching workflow exposes `integer overflow corrupts totals`.
**Symptom:** Integer overflow corrupts totals.
**Competing causes:** narrow accumulator or unchecked conversion.
**Discriminating diagnostic:** test max values and use wide reference arithmetic.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** widen type and reject unsafe conversion.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-019 | SCALE_FAILURE | Cold start stampedes an origin | simultaneous cache misses | observe origin concurrency after expiry | single-flight, jitter, and admission cap | HYPOTHETICAL | collection contracts |
### ISSUE-019 — Cold start stampedes an origin
**Trigger/environment:** A bounded teaching workflow exposes `cold start stampedes an origin`.
**Symptom:** Cold start stampedes an origin.
**Competing causes:** simultaneous cache misses.
**Discriminating diagnostic:** observe origin concurrency after expiry.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** single-flight, jitter, and admission cap.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-020 | CONCEPT_CONFUSION | Exact and approximate answers are mixed | approximation contract omitted | inspect error bound and consumer expectation | label approximation and supply exact fallback | HYPOTHETICAL | collection contracts |
### ISSUE-020 — Exact and approximate answers are mixed
**Trigger/environment:** A bounded teaching workflow exposes `exact and approximate answers are mixed`.
**Symptom:** Exact and approximate answers are mixed.
**Competing causes:** approximation contract omitted.
**Discriminating diagnostic:** inspect error bound and consumer expectation.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** label approximation and supply exact fallback.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-021 | VIOLATED_ASSUMPTION | Authorization is checked after data load | unsafe ordering or broad query | trace requested and returned scope | authorize before access and test forbidden descendant | HYPOTHETICAL | collection contracts |
### ISSUE-021 — Authorization is checked after data load
**Trigger/environment:** A bounded teaching workflow exposes `authorization is checked after data load`.
**Symptom:** Authorization is checked after data load.
**Competing causes:** unsafe ordering or broad query.
**Discriminating diagnostic:** trace requested and returned scope.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** authorize before access and test forbidden descendant.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-022 | NUMERICAL_EDGE | NaN enters a comparison | missing arithmetic guard | test finite predicate before ordering | quarantine invalid value and report source | HYPOTHETICAL | collection contracts |
### ISSUE-022 — NaN enters a comparison
**Trigger/environment:** A bounded teaching workflow exposes `nan enters a comparison`.
**Symptom:** NaN enters a comparison.
**Competing causes:** missing arithmetic guard.
**Discriminating diagnostic:** test finite predicate before ordering.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** quarantine invalid value and report source.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-023 | SCALE_FAILURE | Retry storm hides the original outage | synchronized exponential retries | plot attempt rate versus success rate | budget retries, jitter, and trip circuit | HYPOTHETICAL | collection contracts |
### ISSUE-023 — Retry storm hides the original outage
**Trigger/environment:** A bounded teaching workflow exposes `retry storm hides the original outage`.
**Symptom:** Retry storm hides the original outage.
**Competing causes:** synchronized exponential retries.
**Discriminating diagnostic:** plot attempt rate versus success rate.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** budget retries, jitter, and trip circuit.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-024 | CONCEPT_CONFUSION | A formula is used outside validity domain | independence or stationarity assumed | state assumptions and construct boundary case | switch method or explicitly defer conclusion | HYPOTHETICAL | collection contracts |
### ISSUE-024 — A formula is used outside validity domain
**Trigger/environment:** A bounded teaching workflow exposes `a formula is used outside validity domain`.
**Symptom:** A formula is used outside validity domain.
**Competing causes:** independence or stationarity assumed.
**Discriminating diagnostic:** state assumptions and construct boundary case.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** switch method or explicitly defer conclusion.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

| ISSUE-025 | VIOLATED_ASSUMPTION | Human override is not recorded | side channel or missing actor field | compare final action to recommendation ledger | record override reason and reconcile audit trail | HYPOTHETICAL | collection contracts |
### ISSUE-025 — Human override is not recorded
**Trigger/environment:** A bounded teaching workflow exposes `human override is not recorded`.
**Symptom:** Human override is not recorded.
**Competing causes:** side channel or missing actor field.
**Discriminating diagnostic:** compare final action to recommendation ledger.
**Root-cause evidence:** Record the raw input, version, timestamp, and decision ledger; absence of this record leaves the cause OPEN.
**Immediate mitigation:** record override reason and reconcile audit trail.
**Correction and prevention:** change the contract, add a precondition check, assign an owner, and keep the old safe path until replay passes.
**Verification:** replay the triggering record plus a nearby boundary; inspect output, reason code, side effect count, and owner action.
**Mitigation risk / escalation:** a strict guard can defer legitimate work; escalate sustained deferrals to the accountable owner.

Takeaway: each issue supplies symptom, competing causes, a discriminating check, correction, and verification.

## 9. Complexity & Cost Analysis
A per-item validation is O(1) only if lookups, payload size, and retry count are bounded. Logging cost is items × bytes/item × retention; measure actual overhead before imposing a production budget. Takeaway: diagnostic work must itself be bounded.

## 10. Key Theorems / Results Reference
Conservation of work supports the queue calculation under its stated conditions. Idempotency is an application invariant, not a property granted by transport retries. Takeaway: name the condition that makes a result applicable.

## 11. Interview / Exam Questions
### Q1
**Question:** How would you diagnose issue 001 without assuming its cause?
**Answer:** Start with the observable symptom, collect timestamped input and configuration evidence, run the listed discriminating check, contain risk with the safe fallback, then replay the original and boundary records. A correct answer distinguishes mitigation from correction and does not call a hypothetical scenario observed.

### Q2
**Question:** How would you diagnose issue 002 without assuming its cause?
**Answer:** Start with the observable symptom, collect timestamped input and configuration evidence, run the listed discriminating check, contain risk with the safe fallback, then replay the original and boundary records. A correct answer distinguishes mitigation from correction and does not call a hypothetical scenario observed.

### Q3
**Question:** How would you diagnose issue 003 without assuming its cause?
**Answer:** Start with the observable symptom, collect timestamped input and configuration evidence, run the listed discriminating check, contain risk with the safe fallback, then replay the original and boundary records. A correct answer distinguishes mitigation from correction and does not call a hypothetical scenario observed.

### Q4
**Question:** How would you diagnose issue 004 without assuming its cause?
**Answer:** Start with the observable symptom, collect timestamped input and configuration evidence, run the listed discriminating check, contain risk with the safe fallback, then replay the original and boundary records. A correct answer distinguishes mitigation from correction and does not call a hypothetical scenario observed.

### Q5
**Question:** How would you diagnose issue 005 without assuming its cause?
**Answer:** Start with the observable symptom, collect timestamped input and configuration evidence, run the listed discriminating check, contain risk with the safe fallback, then replay the original and boundary records. A correct answer distinguishes mitigation from correction and does not call a hypothetical scenario observed.

### Q6
**Question:** How would you diagnose issue 006 without assuming its cause?
**Answer:** Start with the observable symptom, collect timestamped input and configuration evidence, run the listed discriminating check, contain risk with the safe fallback, then replay the original and boundary records. A correct answer distinguishes mitigation from correction and does not call a hypothetical scenario observed.

### Q7
**Question:** How would you diagnose issue 007 without assuming its cause?
**Answer:** Start with the observable symptom, collect timestamped input and configuration evidence, run the listed discriminating check, contain risk with the safe fallback, then replay the original and boundary records. A correct answer distinguishes mitigation from correction and does not call a hypothetical scenario observed.

### Q8
**Question:** How would you diagnose issue 008 without assuming its cause?
**Answer:** Start with the observable symptom, collect timestamped input and configuration evidence, run the listed discriminating check, contain risk with the safe fallback, then replay the original and boundary records. A correct answer distinguishes mitigation from correction and does not call a hypothetical scenario observed.

### Q9
**Question:** How would you diagnose issue 009 without assuming its cause?
**Answer:** Start with the observable symptom, collect timestamped input and configuration evidence, run the listed discriminating check, contain risk with the safe fallback, then replay the original and boundary records. A correct answer distinguishes mitigation from correction and does not call a hypothetical scenario observed.

### Q10
**Question:** How would you diagnose issue 010 without assuming its cause?
**Answer:** Start with the observable symptom, collect timestamped input and configuration evidence, run the listed discriminating check, contain risk with the safe fallback, then replay the original and boundary records. A correct answer distinguishes mitigation from correction and does not call a hypothetical scenario observed.

## 12. Quick Reference Card
| Step | Rule |
|---|---|
| 1 | State symptom and category. |
| 2 | List competing causes. |
| 3 | Run a discriminating diagnostic. |
| 4 | Contain with a bounded safe fallback. |
| 5 | Correct the contract and replay verification. |
| 6 | Record evidence limits and owner. |
