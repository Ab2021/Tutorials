---
collection: "cross-domain-cheat-sheets"
file_id: "07"
topic: "25 Common Issues"
level: 3
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Any control, workload, dependency, or source change."
notation_style: "plain-text"
prerequisites: ["01–06"]
source_status: "OPEN"
source_index: "INDEX.md"
---

# 25 Common Issues
> **Purpose:** Diagnose cross-domain AI delivery failures with competing causes, bounded fixes, and verification.
> **Evidence:** Every entry is a HYPOTHETICAL teaching scenario; no incident is asserted.
> **Issue status:** 25 indexed HYPOTHETICAL entries are present, but their required distinct investigations have not been reviewed to completion; do not treat the title as a completion claim.


## 1. File Header

**Reader task:** Given an observable delivery symptom, preserve the evidence boundary, identify competing mechanisms, select a discriminating diagnostic, and make a bounded response.

**Scope:** The entries cover sizing, release compatibility, agent authority, edge timing, evaluation, and investment accounting in Sheets 01–06. They are teaching scenarios only; none reports a real incident, test execution, or measured outcome.

**Environment boundary:** An issue record is about one versioned request, cohort, service, device, or cost ledger in a stated time window. It is not permission to infer root cause from a dashboard aggregate.

**Count policy:** There are 25 stable ISSUE IDs. An ID counts as complete only after its trigger, symptom, competing causes, discriminating diagnostic, root-cause evidence, immediate mitigation, correction, prevention, verification, mitigation risk, limitation, and related sheet are specifically reviewed. This draft has not passed that review.

**Evidence and escalation policy:** Preserve raw evidence before mutation. If an issue involves irreversible external action, authorization, safety state, or suspected exposure of secrets, stop the affected path and escalate to the accountable owner rather than repeatedly retrying.

Takeaway: an issue title names a hypothesis to investigate, not a completed diagnosis.
## 2. Concept Map / Mental Model
```text
[symptom] -> [category] -> [competing causes] -> [discriminating signal] -> [bounded mitigation] -> [verification]
```
A symptom is not a root cause. The collection covers sizing, release, agent, edge, evaluation, and economic boundaries taught by Sheets 01–06.
Takeaway: a diagnostic is useful only when its result changes the next action.

## 3. Foundational Definitions
- **symptom:** observable behavior, not cause.
- **root-cause evidence:** artifact favoring a cause over alternatives.
- **mitigation:** bounded immediate risk reduction.
- **correction:** change removing established cause.
- **prevention:** control reducing recurrence.
- **verification:** specified test plus expected signal.
- **severity:** scoped impact and urgency.
- **provenance:** OBSERVED, REPRODUCED, or HYPOTHETICAL; independent of category.
Takeaway: provenance limits the claim a diagnostic can support.

## 4. Core Formulas with Full Derivation
Validity: use one event eligibility definition and one window.
Derivation: backlog is integrated arrivals minus completions for a defined state.
```text
FORMULA [I-01]: B=max(0,lambda-mu)*t
WHERE: B [items]; lambda [items/s]; mu [items/s]; t [s].
CONSTRAINT: constant rates, known initial state, retry amplification excluded.
NUMERICAL EXAMPLE: lambda=30, mu=20, t=300; B=3000 items.
INTERPRETATION: investigate flow before raising a threshold.
```
Boundary: higher retry arrivals make this estimate too small.
Takeaway: quantify the diagnostic boundary before changing policy.

## 5. Worked Numerical Example
ILLUSTRATIVE: 30 alerts/s arrive, 20/s complete, for 300 s: backlog=(30-20)*300=3000. Recovery at 50/s with arrival 20/s drains in 3000/(50-20)=100 s. Independent check: 10*300=3000 and 3000/30=100.
Oracle inputs: lambda=30; mu=20; t=300; recovery_mu=50; recovery_lambda=20. Oracle outputs: B=3000; drain_s=100.
Takeaway: recovery needs positive spare capacity.

## 6. Visual / Geometric Intuition
```text
cause A --+
cause B --+--> symptom --> discriminating test --> action
cause C --+
```
The test should eliminate at least one plausible branch before a permanent correction is chosen.
Takeaway: do not choose a fix before choosing its discriminating signal.

## 7. Comparison / Selection Table
| Situation | First action | Avoid | Escalate when |
|---|---|---|---|
| irreversible action | stop/preserve audit | blind retry | authority or safety unclear |
| queue growth | measure flow and age | threshold-only change | deadline breached |
| metric shift | version denominator/cohort | unrelated window compare | semantic contract changed |
| deadline miss | capture tail/thermal/load | mean-only tuning | safe state entered |
| cost surprise | reconcile ledger | headline price compare | spend ceiling crossed |
Takeaway: preserve evidence and reversibility first.

## 8. Failure Modes & Edge Cases
| ID | Category | Symptom | Root cause | Diagnostic / expected signal | Fix | Provenance | Related file |
|---|---|---|---|---|---|---|---|
| ISSUE-001 | VIOLATED_ASSUMPTION | feature unavailable at decision time | leakage, clock mismatch, cache future state | compare availability and decision timestamps | remove field; point-in-time replay | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-002 | NUMERICAL_EDGE | average load hides queue overflow | mean/peak confusion, retries, wrong interval | plot per-second arrival/service/depth | cap retries; size burst buffer | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-003 | SCALE_FAILURE | retry duplicates external effect | no idempotency, short TTL, collision | search identical idempotency keys | persist key/outcome before retry | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-004 | CONCEPT_CONFUSION | offline score called business impact | proxy mismatch, selection bias, no comparator | inspect denominator/exposure/guardrail | narrow claim; controlled evaluation | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-005 | VIOLATED_ASSUMPTION | rollback leaves bad state | one-way migration, cache mismatch, side effect | run old binary against migrated state | expand/contract migration | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-006 | NUMERICAL_EDGE | threshold flips at boundary | float, conversion, inequality | log raw value/unit/operator | integer units or tolerance | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-007 | SCALE_FAILURE | review backlog exceeds SLA | arrival growth, staffing, duplicate alerts | measure inflow/outflow/age | prioritize and recalibrate | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-008 | CONCEPT_CONFUSION | tool capability becomes permission | broad role, stale identity, policy conflation | compare signed approval to call | default deny exact binding | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-009 | VIOLATED_ASSUMPTION | approval reused after request mutation | missing hash, context mutation, expiry ignored | recompute canonical request hash | bind hash and expiry | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-010 | NUMERICAL_EDGE | cost omits failed work | retries, cache misses, currency basis | reconcile raw billing/labor/events | cost per useful outcome | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-011 | SCALE_FAILURE | context window collapses | unbounded chunks, duplicates, no budget | log tokens by component | allocate and enforce budget | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-012 | CONCEPT_CONFUSION | citation does not support claim | near match, stale index, entailment gap | inspect quoted span/index version | require claim-to-span evidence | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-013 | VIOLATED_ASSUMPTION | sensor clocks assumed equal | drift, driver buffering, zone conversion | compare hardware/receive/sync time | synchronize and reject stale input | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-014 | NUMERICAL_EDGE | mean latency hides deadline miss | GC, contention, thermal throttle | measure tail under load and temperature | budget worst case/safe state | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-015 | SCALE_FAILURE | thermal throttle after warm-up | cooling, power estimate, enclosure | record temperature/frequency/latency | derate and thermal-soak test | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-016 | CONCEPT_CONFUSION | offline accuracy called safety | shift, actuation error, no hazard analysis | map error slice to hazard envelope | HIL validation/constrained actuation | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-017 | VIOLATED_ASSUMPTION | metric denominator silently changes | schema, filtering, population change | compare versioned numerator/denominator | semantic contract/version series | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-018 | NUMERICAL_EDGE | sample ratio mismatch ignored | assignment bug, exclusions, bots | audit assignment counts/logs | repair assignment and rerun | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-019 | SCALE_FAILURE | hot key creates stale decisions | skew, partition key, fanout join | measure key percentiles and lag | isolate/shard with order contract | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-020 | CONCEPT_CONFUSION | before-after called causal | seasonality, concurrent change, regression | pretrend and exposure audit | experiment/quasi-experiment | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-021 | VIOLATED_ASSUMPTION | secrets believed redacted | new field, exception path, telemetry | scan normal and forced-error logs | allow-list logs and rotate | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-022 | NUMERICAL_EDGE | error budget includes ineligible traffic | health checks, window error, retry counting | reconcile event classes | eligible-event contract | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-023 | SCALE_FAILURE | rate limit causes retry storm | synchronized backoff, no jitter, quota | retry timing and 429 histogram | jitter/circuit breaker | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-024 | CONCEPT_CONFUSION | checklist called safety evidence | untested control, vague owner, process confusion | trace control to test artifact | control-to-evidence ledger | HYPOTHETICAL | [01–06](INDEX.md) |
| ISSUE-025 | VIOLATED_ASSUMPTION | backup assumed restorable | permissions, corruption, version drift | timed isolated restore | automated restore drill | HYPOTHETICAL | [01–06](INDEX.md) |
### ISSUE-001 — feature unavailable at decision time
- **Category / provenance:** VIOLATED_ASSUMPTION / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when feature unavailable at decision time.
- **Symptom:** feature unavailable at decision time.
- **Competing causes:** leakage, clock mismatch, cache future state.
- **Discriminating diagnostic / expected signal:** compare availability and decision timestamps. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** remove field; point-in-time replay.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-002 — average load hides queue overflow
- **Category / provenance:** NUMERICAL_EDGE / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when average load hides queue overflow.
- **Symptom:** average load hides queue overflow.
- **Competing causes:** mean/peak confusion, retries, wrong interval.
- **Discriminating diagnostic / expected signal:** plot per-second arrival/service/depth. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** cap retries; size burst buffer.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-003 — retry duplicates external effect
- **Category / provenance:** SCALE_FAILURE / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when retry duplicates external effect.
- **Symptom:** retry duplicates external effect.
- **Competing causes:** no idempotency, short TTL, collision.
- **Discriminating diagnostic / expected signal:** search identical idempotency keys. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** persist key/outcome before retry.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-004 — offline score called business impact
- **Category / provenance:** CONCEPT_CONFUSION / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when offline score called business impact.
- **Symptom:** offline score called business impact.
- **Competing causes:** proxy mismatch, selection bias, no comparator.
- **Discriminating diagnostic / expected signal:** inspect denominator/exposure/guardrail. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** narrow claim; controlled evaluation.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-005 — rollback leaves bad state
- **Category / provenance:** VIOLATED_ASSUMPTION / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when rollback leaves bad state.
- **Symptom:** rollback leaves bad state.
- **Competing causes:** one-way migration, cache mismatch, side effect.
- **Discriminating diagnostic / expected signal:** run old binary against migrated state. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** expand/contract migration.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-006 — threshold flips at boundary
- **Category / provenance:** NUMERICAL_EDGE / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when threshold flips at boundary.
- **Symptom:** threshold flips at boundary.
- **Competing causes:** float, conversion, inequality.
- **Discriminating diagnostic / expected signal:** log raw value/unit/operator. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** integer units or tolerance.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-007 — review backlog exceeds SLA
- **Category / provenance:** SCALE_FAILURE / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when review backlog exceeds SLA.
- **Symptom:** review backlog exceeds SLA.
- **Competing causes:** arrival growth, staffing, duplicate alerts.
- **Discriminating diagnostic / expected signal:** measure inflow/outflow/age. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** prioritize and recalibrate.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-008 — tool capability becomes permission
- **Category / provenance:** CONCEPT_CONFUSION / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when tool capability becomes permission.
- **Symptom:** tool capability becomes permission.
- **Competing causes:** broad role, stale identity, policy conflation.
- **Discriminating diagnostic / expected signal:** compare signed approval to call. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** default deny exact binding.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-009 — approval reused after request mutation
- **Category / provenance:** VIOLATED_ASSUMPTION / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when approval reused after request mutation.
- **Symptom:** approval reused after request mutation.
- **Competing causes:** missing hash, context mutation, expiry ignored.
- **Discriminating diagnostic / expected signal:** recompute canonical request hash. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** bind hash and expiry.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-010 — cost omits failed work
- **Category / provenance:** NUMERICAL_EDGE / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when cost omits failed work.
- **Symptom:** cost omits failed work.
- **Competing causes:** retries, cache misses, currency basis.
- **Discriminating diagnostic / expected signal:** reconcile raw billing/labor/events. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** cost per useful outcome.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-011 — context window collapses
- **Category / provenance:** SCALE_FAILURE / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when context window collapses.
- **Symptom:** context window collapses.
- **Competing causes:** unbounded chunks, duplicates, no budget.
- **Discriminating diagnostic / expected signal:** log tokens by component. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** allocate and enforce budget.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-012 — citation does not support claim
- **Category / provenance:** CONCEPT_CONFUSION / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when citation does not support claim.
- **Symptom:** citation does not support claim.
- **Competing causes:** near match, stale index, entailment gap.
- **Discriminating diagnostic / expected signal:** inspect quoted span/index version. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** require claim-to-span evidence.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-013 — sensor clocks assumed equal
- **Category / provenance:** VIOLATED_ASSUMPTION / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when sensor clocks assumed equal.
- **Symptom:** sensor clocks assumed equal.
- **Competing causes:** drift, driver buffering, zone conversion.
- **Discriminating diagnostic / expected signal:** compare hardware/receive/sync time. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** synchronize and reject stale input.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-014 — mean latency hides deadline miss
- **Category / provenance:** NUMERICAL_EDGE / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when mean latency hides deadline miss.
- **Symptom:** mean latency hides deadline miss.
- **Competing causes:** GC, contention, thermal throttle.
- **Discriminating diagnostic / expected signal:** measure tail under load and temperature. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** budget worst case/safe state.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-015 — thermal throttle after warm-up
- **Category / provenance:** SCALE_FAILURE / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when thermal throttle after warm-up.
- **Symptom:** thermal throttle after warm-up.
- **Competing causes:** cooling, power estimate, enclosure.
- **Discriminating diagnostic / expected signal:** record temperature/frequency/latency. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** derate and thermal-soak test.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-016 — offline accuracy called safety
- **Category / provenance:** CONCEPT_CONFUSION / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when offline accuracy called safety.
- **Symptom:** offline accuracy called safety.
- **Competing causes:** shift, actuation error, no hazard analysis.
- **Discriminating diagnostic / expected signal:** map error slice to hazard envelope. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** HIL validation/constrained actuation.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-017 — metric denominator silently changes
- **Category / provenance:** VIOLATED_ASSUMPTION / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when metric denominator silently changes.
- **Symptom:** metric denominator silently changes.
- **Competing causes:** schema, filtering, population change.
- **Discriminating diagnostic / expected signal:** compare versioned numerator/denominator. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** semantic contract/version series.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-018 — sample ratio mismatch ignored
- **Category / provenance:** NUMERICAL_EDGE / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when sample ratio mismatch ignored.
- **Symptom:** sample ratio mismatch ignored.
- **Competing causes:** assignment bug, exclusions, bots.
- **Discriminating diagnostic / expected signal:** audit assignment counts/logs. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** repair assignment and rerun.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-019 — hot key creates stale decisions
- **Category / provenance:** SCALE_FAILURE / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when hot key creates stale decisions.
- **Symptom:** hot key creates stale decisions.
- **Competing causes:** skew, partition key, fanout join.
- **Discriminating diagnostic / expected signal:** measure key percentiles and lag. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** isolate/shard with order contract.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-020 — before-after called causal
- **Category / provenance:** CONCEPT_CONFUSION / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when before-after called causal.
- **Symptom:** before-after called causal.
- **Competing causes:** seasonality, concurrent change, regression.
- **Discriminating diagnostic / expected signal:** pretrend and exposure audit. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** experiment/quasi-experiment.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-021 — secrets believed redacted
- **Category / provenance:** VIOLATED_ASSUMPTION / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when secrets believed redacted.
- **Symptom:** secrets believed redacted.
- **Competing causes:** new field, exception path, telemetry.
- **Discriminating diagnostic / expected signal:** scan normal and forced-error logs. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** allow-list logs and rotate.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-022 — error budget includes ineligible traffic
- **Category / provenance:** NUMERICAL_EDGE / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when error budget includes ineligible traffic.
- **Symptom:** error budget includes ineligible traffic.
- **Competing causes:** health checks, window error, retry counting.
- **Discriminating diagnostic / expected signal:** reconcile event classes. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** eligible-event contract.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-023 — rate limit causes retry storm
- **Category / provenance:** SCALE_FAILURE / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when rate limit causes retry storm.
- **Symptom:** rate limit causes retry storm.
- **Competing causes:** synchronized backoff, no jitter, quota.
- **Discriminating diagnostic / expected signal:** retry timing and 429 histogram. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** jitter/circuit breaker.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-024 — checklist called safety evidence
- **Category / provenance:** CONCEPT_CONFUSION / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when checklist called safety evidence.
- **Symptom:** checklist called safety evidence.
- **Competing causes:** untested control, vague owner, process confusion.
- **Discriminating diagnostic / expected signal:** trace control to test artifact. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** control-to-evidence ledger.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
### ISSUE-025 — backup assumed restorable
- **Category / provenance:** VIOLATED_ASSUMPTION / HYPOTHETICAL teaching scenario; it is not an observed incident.
- **Trigger/environment:** investigate when backup assumed restorable.
- **Symptom:** backup assumed restorable.
- **Competing causes:** permissions, corruption, version drift.
- **Discriminating diagnostic / expected signal:** timed isolated restore. A mismatch refutes the assumed explanation rather than being averaged away.
- **Root-cause evidence:** retain raw timestamps, versions, request identity, and test output; without it, cause remains OPEN.
- **Immediate mitigation:** automated restore drill.
- **Correction:** make the mitigation durable only after the signal supports its causal branch.
- **Prevention:** encode an acceptance test, owner, expiry, and alertable invariant.
- **Verification:** rerun the stated diagnostic with recorded input and assert the expected safe or consistent result.
- **Mitigation risks / escalation:** a restrictive control can defer value, bias a cohort, or move work elsewhere; escalate when safety, authorization, or irreversible state is implicated.
- **Related sheet:** use Sheets 01–06 for the system, release, authorization, edge, evaluation, or economic boundary.
Takeaway: every issue ends in a check that can falsify premature certainty.

## 9. Complexity & Cost Analysis
| Diagnostic | Cost driver | Scale boundary |
|---|---|---|
| point-in-time replay | retained data/compute | history and entities |
| trace query | telemetry volume | high-cardinality labels |
| restore drill | isolated capacity/staff | snapshot/dependency version |
| load test | synthetic workload | burst/thermal profile |
| human review | minutes per case | arrival/queue age |
Takeaway: diagnostics consume capacity and need an operating budget.

## 10. Key Theorems / Results Reference
- Conservation: defined state changes by inflow minus outflow.
- Little’s law: L=lambda W needs stable long-run conditions and one boundary.
- Fail-safe state is distinct from successful recovery.
- A checklist does not demonstrate an untested control.
Takeaway: results guide conditions, not incident claims.

## 11. Interview / Exam Questions
### Why competing causes?
- **Checked answer:** A symptom has multiple mechanisms; a discriminating test prevents generic harmful fixes.
- **Reasoning:** identify mechanism and evidence boundary.
- **Follow-up:** name the next artifact to inspect.
- **Acceptable alternatives:** a concrete test with expected signal.
- **Red flags:** apply a fix without separating causes.
- **Scoring:** 2 mechanism plus check; 1 plausible action; 0 unsupported certainty.
### What makes mitigation bounded?
- **Checked answer:** Scope, owner, expiry, side effects, and verification condition.
- **Reasoning:** identify mechanism and evidence boundary.
- **Follow-up:** name the next artifact to inspect.
- **Acceptable alternatives:** a concrete test with expected signal.
- **Red flags:** apply a fix without separating causes.
- **Scoring:** 2 mechanism plus check; 1 plausible action; 0 unsupported certainty.
### Compute backlog at 30/s in,20/s out,300 s.
- **Checked answer:** 3000 items under constant-rate assumptions.
- **Reasoning:** identify mechanism and evidence boundary.
- **Follow-up:** name the next artifact to inspect.
- **Acceptable alternatives:** a concrete test with expected signal.
- **Red flags:** apply a fix without separating causes.
- **Scoring:** 2 mechanism plus check; 1 plausible action; 0 unsupported certainty.
### Why restore drill over backup existence?
- **Checked answer:** It tests permissions, compatibility, integrity, and recovery time.
- **Reasoning:** identify mechanism and evidence boundary.
- **Follow-up:** name the next artifact to inspect.
- **Acceptable alternatives:** a concrete test with expected signal.
- **Red flags:** apply a fix without separating causes.
- **Scoring:** 2 mechanism plus check; 1 plausible action; 0 unsupported certainty.
### What binds approval?
- **Checked answer:** Subject, action, resource, context, expiry, approver.
- **Reasoning:** identify mechanism and evidence boundary.
- **Follow-up:** name the next artifact to inspect.
- **Acceptable alternatives:** a concrete test with expected signal.
- **Red flags:** apply a fix without separating causes.
- **Scoring:** 2 mechanism plus check; 1 plausible action; 0 unsupported certainty.
### Why cost dashboard wrong?
- **Checked answer:** It can omit retries, labor, failed work, or denominator.
- **Reasoning:** identify mechanism and evidence boundary.
- **Follow-up:** name the next artifact to inspect.
- **Acceptable alternatives:** a concrete test with expected signal.
- **Red flags:** apply a fix without separating causes.
- **Scoring:** 2 mechanism plus check; 1 plausible action; 0 unsupported certainty.
### Burst versus sustained overload?
- **Checked answer:** Only burst drains with spare capacity.
- **Reasoning:** identify mechanism and evidence boundary.
- **Follow-up:** name the next artifact to inspect.
- **Acceptable alternatives:** a concrete test with expected signal.
- **Red flags:** apply a fix without separating causes.
- **Scoring:** 2 mechanism plus check; 1 plausible action; 0 unsupported certainty.
### Before/after impact flaw?
- **Checked answer:** Concurrent changes and shifts can explain difference.
- **Reasoning:** identify mechanism and evidence boundary.
- **Follow-up:** name the next artifact to inspect.
- **Acceptable alternatives:** a concrete test with expected signal.
- **Red flags:** apply a fix without separating causes.
- **Scoring:** 2 mechanism plus check; 1 plausible action; 0 unsupported certainty.
### Why edge mean insufficient?
- **Checked answer:** Safety needs tail execution and interference conditions.
- **Reasoning:** identify mechanism and evidence boundary.
- **Follow-up:** name the next artifact to inspect.
- **Acceptable alternatives:** a concrete test with expected signal.
- **Red flags:** apply a fix without separating causes.
- **Scoring:** 2 mechanism plus check; 1 plausible action; 0 unsupported certainty.
### What does HYPOTHETICAL mean?
- **Checked answer:** Teaching scenario, not recorded incident.
- **Reasoning:** identify mechanism and evidence boundary.
- **Follow-up:** name the next artifact to inspect.
- **Acceptable alternatives:** a concrete test with expected signal.
- **Red flags:** apply a fix without separating causes.
- **Scoring:** 2 mechanism plus check; 1 plausible action; 0 unsupported certainty.
Takeaway: strong diagnosis names evidence, not just technology.

## 12. Quick Reference Card
| Cue | First check | Avoid | Verification |
|---|---|---|---|
| queue grows | arrival/service/age | threshold-only fix | drain with surplus |
| metric shifts | denominator/version/cohort | before-after causal claim | overlap query |
| action call | approval binding | capability=permission | negative call |
| deadline miss | tail/thermal/interference | mean-only claim | stress run |
| cost surprise | included-work ledger | unit price alone | recompute |
| recovery claim | restore invariant | backup exists claim | isolated restore |
