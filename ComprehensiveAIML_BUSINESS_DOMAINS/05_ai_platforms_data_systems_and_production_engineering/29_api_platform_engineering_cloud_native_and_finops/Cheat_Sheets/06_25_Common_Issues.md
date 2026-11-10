---
collection: module-29-api-platform-engineering
file_id: "06"
topic: API platform common issues
level: 3
status: draft
created: 2026-11-09
reviewed: null
source_status: OPEN
---

# 25 Common Issues — API Platforms, Cloud-Native Systems and FinOps

> **Purpose:** Diagnose contract, isolation, capacity, and cost failures from discriminating evidence.
> **Evidence:** HYPOTHETICAL teaching traces; no company or provider incident is claimed.

## 1. File Header

This final artifact contains 25 distinct issue records. Each preserves trigger, competing causes, signal, mitigation, correction, prevention, verification, and a scope limit.

## 2. Concept Map / Mental Model

```text
symptom -> state owner -> bounded reproduction -> invariant -> correction -> recovery verification
```

**Takeaway:** a symptom selects a test; it is not root-cause proof.

## 3. Foundational Definitions

| Term | Meaning | Boundary |
|---|---|---|
| invariant | property preserved by all allowed transitions | metric alone is not a guarantee |
| discriminator | observation separating causes | must be replayable |
| containment | reversible harm reduction | does not prove correction |

**Takeaway:** collect evidence before containment changes the trace.

## 4. Core Formulas with Full Derivation

Validity: all counts must share one global scope and interval.

```text
FORMULA [I-01]: admitted + active_leases <= global_budget
WHERE: all terms are integer credits in one scope.
CONSTRAINT: grants, expiry, and revoke are serialized or fenced.
NUMERICAL EXAMPLE: 72 + 28 = 100 credits, so zero additional credits are available from budget 100.
INTERPRETATION: a replica cannot spend credits it does not own.
```

**Takeaway:** scope and owner precede arithmetic.

## 5. Worked Numerical Example

ILLUSTRATIVE: three independent replicas each admit 40 credits against intended global budget 100. Aggregate admission is 120, an excess of 20 credits. An atomic debit admits at most 100; a lease scheme must demonstrate the same invariant under its explicit failure policy. **Takeaway:** sum replicas to test a global claim.

## 6. Visual / Geometric Intuition

```text
client -> authenticated gateway -> admission owner -> workload -> idempotent sink
```

**Takeaway:** every cache, log, queue, and replay path crosses ownership boundaries.

## 7. Comparison / Selection Table

| Need | Mechanism | Disqualifier |
|---|---|---|
| retryable mutation | durable scoped operation record | identity unknown |
| hard budget | atomic debit or fenced lease | periodic report only |
| tenant retrieval | server-side scoped query | client filter only |

**Takeaway:** select by enforceable invariant, not product label.

## 8. Failure Modes & Edge Cases

| ID | Category | Symptom | Provenance |
|---|---|---|---|
| API-ID-01 | VIOLATED_ASSUMPTION | Retry duplicates a payment | HYPOTHETICAL |
| API-ID-02 | VIOLATED_ASSUMPTION | Retry leaks another tenant response | HYPOTHETICAL |
| API-VER-03 | CONCEPT_CONFUSION | Old client breaks after deployment | HYPOTHETICAL |
| API-AUTH-04 | VIOLATED_ASSUMPTION | Valid token reads foreign object | HYPOTHETICAL |
| API-LIMIT-05 | SCALE_FAILURE | Burst exceeds global quota | HYPOTHETICAL |
| API-TIME-06 | NUMERICAL_EDGE | Quota refills early | HYPOTHETICAL |
| API-RETRY-07 | SCALE_FAILURE | Outage becomes retry storm | HYPOTHETICAL |
| API-CACHE-08 | CONCEPT_CONFUSION | Cache survives permission revoke | HYPOTHETICAL |
| K8S-READY-09 | VIOLATED_ASSUMPTION | Traffic reaches warming model | HYPOTHETICAL |
| K8S-CPU-10 | SCALE_FAILURE | Autoscaler misses tail breach | HYPOTHETICAL |
| K8S-DRAIN-11 | VIOLATED_ASSUMPTION | Rollout drops requests | HYPOTHETICAL |
| K8S-IMAGE-12 | SCALE_FAILURE | Rollout stalls all nodes | HYPOTHETICAL |
| TENANT-DATA-13 | VIOLATED_ASSUMPTION | Retrieval exposes foreign embedding | HYPOTHETICAL |
| TENANT-QUOTA-14 | CONCEPT_CONFUSION | Noisy tenant monopolizes accelerator | HYPOTHETICAL |
| TENANT-LOG-15 | VIOLATED_ASSUMPTION | Logs disclose prompt | HYPOTHETICAL |
| FINOPS-UNIT-16 | CONCEPT_CONFUSION | Cost per request falls while value falls | HYPOTHETICAL |
| FINOPS-ALLOC-17 | NUMERICAL_EDGE | Chargeback differs from invoice | HYPOTHETICAL |
| FINOPS-TAG-18 | SCALE_FAILURE | Resources evade budget ownership | HYPOTHETICAL |
| FINOPS-BUDGET-19 | VIOLATED_ASSUMPTION | Alert follows committed spend | HYPOTHETICAL |
| LEASE-20 | SCALE_FAILURE | Partition over-admits leases | HYPOTHETICAL |
| LEASE-21 | NUMERICAL_EDGE | Renewal charges twice | HYPOTHETICAL |
| LEASE-22 | VIOLATED_ASSUMPTION | Late revoke deletes new lease | HYPOTHETICAL |
| OPS-SLO-23 | CONCEPT_CONFUSION | Mean passes but users timeout | HYPOTHETICAL |
| OPS-DLQ-24 | SCALE_FAILURE | Replay duplicates completion | HYPOTHETICAL |
| OPS-SECRET-25 | VIOLATED_ASSUMPTION | Debug endpoint exposes secret | HYPOTHETICAL |

### API-ID-01 — Retry duplicates a payment

**Category / provenance:** `VIOLATED_ASSUMPTION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: idempotency record is written after the side effect. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** replay one key after timeout; two effects proves it. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** atomically reserve key and response before effect. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### API-ID-02 — Retry leaks another tenant response

**Category / provenance:** `VIOLATED_ASSUMPTION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: key omits tenant scope. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** reuse key across tenants; shared response proves it. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** key by tenant, operation, and client key. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### API-VER-03 — Old client breaks after deployment

**Category / provenance:** `CONCEPT_CONFUSION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: field meaning changed silently. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** replay captured old contract. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** publish compatibility policy and dual migration. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### API-AUTH-04 — Valid token reads foreign object

**Category / provenance:** `VIOLATED_ASSUMPTION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: object ownership not checked. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** request known foreign object; 200 proves it. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** authorize each lookup at object boundary. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### API-LIMIT-05 — Burst exceeds global quota

**Category / provenance:** `SCALE_FAILURE` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: replicas hold independent buckets. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** sum accepts during synchronized burst. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** use atomic global debit or fenced leases. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### API-TIME-06 — Quota refills early

**Category / provenance:** `NUMERICAL_EDGE` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: wall-clock jump or rounding. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** test backward clock step. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** use monotonic elapsed time and clamp negatives. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### API-RETRY-07 — Outage becomes retry storm

**Category / provenance:** `SCALE_FAILURE` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: immediate unbounded retries. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** attempt histogram clusters at zero delay. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** bound attempts, jitter, propagate deadline. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### API-CACHE-08 — Cache survives permission revoke

**Category / provenance:** `CONCEPT_CONFUSION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: key omits authorization version. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** revoke then repeat read. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** scope cache or bypass revocation-sensitive reads. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### K8S-READY-09 — Traffic reaches warming model

**Category / provenance:** `VIOLATED_ASSUMPTION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: readiness only checks process. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** call during artifact warmup. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** gate readiness on required state. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### K8S-CPU-10 — Autoscaler misses tail breach

**Category / provenance:** `SCALE_FAILURE` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: CPU ignores queue saturation. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** queue age rises while CPU flat. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** scale on admitted queue and saturation. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### K8S-DRAIN-11 — Rollout drops requests

**Category / provenance:** `VIOLATED_ASSUMPTION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: grace below request bound. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** drain a long request. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** remove endpoint then wait bounded inflight. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### K8S-IMAGE-12 — Rollout stalls all nodes

**Category / provenance:** `SCALE_FAILURE` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: registry pull contention. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** observe concurrent pulls. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** stage rollout and measure registry capacity. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### TENANT-DATA-13 — Retrieval exposes foreign embedding

**Category / provenance:** `VIOLATED_ASSUMPTION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: filter absent on fallback. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** negative cross-tenant query. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** server-side filter every route. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### TENANT-QUOTA-14 — Noisy tenant monopolizes accelerator

**Category / provenance:** `CONCEPT_CONFUSION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: quota counts requests not work. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** equal requests with unequal tokens. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** meter concurrency or token budget. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### TENANT-LOG-15 — Logs disclose prompt

**Category / provenance:** `VIOLATED_ASSUMPTION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: logging before redaction. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** search controlled sensitive marker. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** redact before export and scope access. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### FINOPS-UNIT-16 — Cost per request falls while value falls

**Category / provenance:** `CONCEPT_CONFUSION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: denominator includes failed attempts. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** reconcile attempts and qualified completions. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** report cost per successful qualified outcome. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### FINOPS-ALLOC-17 — Chargeback differs from invoice

**Category / provenance:** `NUMERICAL_EDGE` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: shared or egress costs omitted. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** sum ledger against same-window invoice. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** publish residual and allocation policy. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### FINOPS-TAG-18 — Resources evade budget ownership

**Category / provenance:** `SCALE_FAILURE` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: tagging optional. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** create unowned resource in policy test. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** deny or quarantine unowned creation. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### FINOPS-BUDGET-19 — Alert follows committed spend

**Category / provenance:** `VIOLATED_ASSUMPTION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: billing export delayed. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** compare usage and alert times. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** admit with estimates then reconcile. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### LEASE-20 — Partition over-admits leases

**Category / provenance:** `SCALE_FAILURE` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: issuers lack fencing. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** partition then sum active leases. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** fence issuer epoch; declare fail policy. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### LEASE-21 — Renewal charges twice

**Category / provenance:** `NUMERICAL_EDGE` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: renewal not idempotent. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** repeat renewal after timeout. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** compare-and-set lease generation. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### LEASE-22 — Late revoke deletes new lease

**Category / provenance:** `VIOLATED_ASSUMPTION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: message lacks generation. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** deliver old revoke after new grant. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** reject stale transition versions. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### OPS-SLO-23 — Mean passes but users timeout

**Category / provenance:** `CONCEPT_CONFUSION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: mean replaces deadline rate. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** inspect histogram and misses. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** gate on scoped tail/deadline objective. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### OPS-DLQ-24 — Replay duplicates completion

**Category / provenance:** `SCALE_FAILURE` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: sink lacks operation identity. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** replay already-completed message. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** reconcile before idempotent replay. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.

### OPS-SECRET-25 — Debug endpoint exposes secret

**Category / provenance:** `VIOLATED_ASSUMPTION` / HYPOTHETICAL teaching scenario, not an observed incident.

**Trigger / environment:** Exercise the smallest controlled path capable of the symptom. Record request identity, tenant scope, version, event time, processing time, and state owner before mitigation.

**Competing causes:** Primary candidate: diagnostic path bypasses auth. A competing explanation is a stale observation, client replay, or aggregation artifact that produces a similar symptom without the same failed transition.

**Diagnostic / expected signal:** probe with least privilege. Preserve raw identifiers and transition versions so another operator can audit the conclusion.

**Immediate mitigation:** apply auth, redaction, break-glass control. Make containment reversible; do not silently discard evidence or broaden access.

**Mitigation risks:** A containment action may deny healthy work, retain stale state, or conceal the first failure reason. Escalate when authorization, billing, or customer data changes.

**Correction:** Encode the invariant at its authoritative state owner. Add a regression for this trigger and an adversarial retry, delay, or partial-failure ordering.

**Prevention:** Require contract review, emit a scoped audit event, version policy with stored state, and alert on the invariant rather than only the symptom.

**Verification:** Re-run the trigger after correction. Expected result: no symptom and a trace/ledger demonstrating the invariant. Repeat with a retry and one partial failure.

**Limit:** This check validates only its stated scope; it does not prove provider behavior, system-wide capacity, or future compatibility.

**Related sheets:** API contracts, workload placement, multi-tenancy, FinOps, and global admission.

**Takeaway:** repair the transition that makes the unsafe outcome possible.
## 9. Complexity & Cost Analysis

Log lookup is O(k) matching records; full reconciliation is O(n) events in its stated window. For 120 illustrative admissions and 64-byte audit records, raw evidence is 7,680 bytes before indexes and replication. Sampling can miss ordering failures. **Takeaway:** size diagnostic retention from fault-replay needs.

## 10. Key Theorems / Results Reference

- Idempotency requires operation identity, scope, retained outcome, and side-effect boundary.
- A local limiter cannot prove a global budget without coordination or bounded allocation.
- Authorization applies to every path that can expose protected state.
- Reconciliation requires matching population, window, and units.

**Takeaway:** these are conditional design results, not vendor claims.

## 11. Interview / Exam Questions

**Q1. How do you defend a platform diagnosis?**  
**Answer:** Name the invariant, decision unit, owner, competing cause, bounded test, correction risk, and verification evidence.  
**Follow-up / red flag:** Change one failure condition. A product name without transition and recovery semantics is insufficient.

**Q2. How do you defend a platform diagnosis?**  
**Answer:** Name the invariant, decision unit, owner, competing cause, bounded test, correction risk, and verification evidence.  
**Follow-up / red flag:** Change one failure condition. A product name without transition and recovery semantics is insufficient.

**Q3. How do you defend a platform diagnosis?**  
**Answer:** Name the invariant, decision unit, owner, competing cause, bounded test, correction risk, and verification evidence.  
**Follow-up / red flag:** Change one failure condition. A product name without transition and recovery semantics is insufficient.

**Q4. How do you defend a platform diagnosis?**  
**Answer:** Name the invariant, decision unit, owner, competing cause, bounded test, correction risk, and verification evidence.  
**Follow-up / red flag:** Change one failure condition. A product name without transition and recovery semantics is insufficient.

**Q5. How do you defend a platform diagnosis?**  
**Answer:** Name the invariant, decision unit, owner, competing cause, bounded test, correction risk, and verification evidence.  
**Follow-up / red flag:** Change one failure condition. A product name without transition and recovery semantics is insufficient.

**Q6. How do you defend a platform diagnosis?**  
**Answer:** Name the invariant, decision unit, owner, competing cause, bounded test, correction risk, and verification evidence.  
**Follow-up / red flag:** Change one failure condition. A product name without transition and recovery semantics is insufficient.

**Q7. How do you defend a platform diagnosis?**  
**Answer:** Name the invariant, decision unit, owner, competing cause, bounded test, correction risk, and verification evidence.  
**Follow-up / red flag:** Change one failure condition. A product name without transition and recovery semantics is insufficient.

**Q8. How do you defend a platform diagnosis?**  
**Answer:** Name the invariant, decision unit, owner, competing cause, bounded test, correction risk, and verification evidence.  
**Follow-up / red flag:** Change one failure condition. A product name without transition and recovery semantics is insufficient.

**Q9. How do you defend a platform diagnosis?**  
**Answer:** Name the invariant, decision unit, owner, competing cause, bounded test, correction risk, and verification evidence.  
**Follow-up / red flag:** Change one failure condition. A product name without transition and recovery semantics is insufficient.

**Q10. How do you defend a platform diagnosis?**  
**Answer:** Name the invariant, decision unit, owner, competing cause, bounded test, correction risk, and verification evidence.  
**Follow-up / red flag:** Change one failure condition. A product name without transition and recovery semantics is insufficient.

**Q11. How do you defend a platform diagnosis?**  
**Answer:** Name the invariant, decision unit, owner, competing cause, bounded test, correction risk, and verification evidence.  
**Follow-up / red flag:** Change one failure condition. A product name without transition and recovery semantics is insufficient.

**Q12. How do you defend a platform diagnosis?**  
**Answer:** Name the invariant, decision unit, owner, competing cause, bounded test, correction risk, and verification evidence.  
**Follow-up / red flag:** Change one failure condition. A product name without transition and recovery semantics is insufficient.

## Appendix A — Per-Issue Reproduction Cards

These cards turn the issue entries into bounded state-oracle exercises. Values are illustrative; record actual environment values before applying a correction.

### API-ID-01 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use operation key, tenant, body digest; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** one durable debit.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** same response and one ledger entry.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### API-ID-02 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use two tenants and same client key; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** tenant-scoped response.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** no cross-tenant body or status.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### API-VER-03 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use old schema fixture; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** semantic compatibility.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** old assertion still passes.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### API-AUTH-04 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use foreign object identifier; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** object ownership check.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** 403 or indistinguishable not-found.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### API-LIMIT-05 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use simultaneous replicas; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** global credit total.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** accepted credits never exceed budget.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### API-TIME-06 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use backward clock movement; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** monotonic elapsed refill.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** balance never jumps upward.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### API-RETRY-07 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use five failing callers; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** attempt budget and jitter.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** offered load falls after error.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### API-CACHE-08 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use permission revoke event; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** auth-versioned cache key.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** post-revoke read is denied.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### K8S-READY-09 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use cold artifact load; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** ready endpoint state.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** no request reaches unready pod.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### K8S-CPU-10 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use queueing load step; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** queue-age scaler input.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** scale occurs before deadline misses.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### K8S-DRAIN-11 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use long running request; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** endpoint removal then grace.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** completion or specified cancellation.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### K8S-IMAGE-12 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use fleet image rollout; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** registry concurrency.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** scheduling remains within bound.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### TENANT-DATA-13 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use foreign retrieval probe; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** mandatory tenant predicate.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** zero foreign document IDs.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### TENANT-QUOTA-14 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use unequal token requests; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** work-unit quota.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** each tenant receives declared share.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### TENANT-LOG-15 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use sensitive marker prompt; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** pre-export redaction.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** marker absent from all sinks.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### FINOPS-UNIT-16 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use failed and qualified outcomes; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** qualified-success denominator.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** metric changes when failures rise.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### FINOPS-ALLOC-17 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use same-window invoice; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** residual ledger.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** allocation plus residual equals invoice.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### FINOPS-TAG-18 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use untagged create request; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** policy enforcement.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** creation denied or quarantined.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### FINOPS-BUDGET-19 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use delayed export; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** admission estimate.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** estimated committed spend is bounded.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### LEASE-20 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use coordinator partition; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** issuer epoch fence.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** active leases stay within policy.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### LEASE-21 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use duplicate renewal; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** lease generation CAS.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** single debit transition.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### LEASE-22 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use late revoke message; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** generation comparison.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** newer lease remains active.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### OPS-SLO-23 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use tail-heavy trace; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** deadline-miss numerator.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** gate fails despite acceptable mean.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### OPS-DLQ-24 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use completed message replay; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** sink operation identity.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** no second durable effect.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

### OPS-SECRET-25 reproduction card

1. **Owner:** The on-call platform engineer owns the drill; the security or finance owner approves any boundary-changing containment.
2. **Input grain:** Use least-privilege probe; retain a correlation ID, tenant scope, and configuration version.
3. **Invariant under test:** diagnostic auth path.
4. **Setup:** Begin from known state, record the baseline state snapshot, and keep the test isolated from production side effects.
5. **Action:** Inject the single trigger described in the matching issue; do not combine faults until the single-fault oracle is understood.
6. **Expected observation:** secret never reaches caller.
7. **Failed-path evidence:** Capture request, transition, and audit records in timestamp order; absence of telemetry is an inconclusive result, not a pass.
8. **Recovery:** Restore the recorded baseline or expire the bounded test state, then rerun one clean-path request.
9. **Decision rule:** Promote the correction only if the expected observation holds and no new authorization, accounting, or availability guardrail fails.
10. **Limit:** This exercise does not establish production load behavior; repeat with measured workload before claiming capacity.

**Appendix takeaway:** a reproducible issue has an owner, input grain, invariant, oracle, recovery step, and limit.

## 12. Quick Reference Card

| Situation | First check | Correction | Verification |
|---|---|---|---|
| retryable mutation | scoped operation record | atomic reservation | replay same key |
| global budget | aggregate accepts and leases | fencing/atomic owner | partition drill |
| tenant access | server-side scope | authorization boundary | negative tenant test |
| rollout | readiness and inflight bound | drain controller | long-request test |
| cost anomaly | matched meter/invoice window | reconciliation owner | residual check |
