---
case_id: "CASE-02-001"
case_title: "Build a Data Quality CLI"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Data engineering quality controls"
primary_pattern: "data-quality-gate"
secondary_patterns: ["contract-first", "idempotent-operation"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "data platform lead"
source_index: "02_python_sql_linux_and_engineering_basics/Case_Studies/INDEX.md"
review_trigger: "Schema, source SLA, threshold, runtime, or release-policy change"
---

# Build a Data Quality CLI

**Evidence record:** This is an ILLUSTRATIVE teaching design. Dataset sizes, failures, run times, and costs are constructed calculations, not a production CLI or observed incident. **Implementation stage:** proposed. **Editorial status:** draft. Source verification and publication review remain OPEN.

## 1. The One-Line Framing
Given one versioned batch and a declared data contract, decide whether the data platform lead should block, warn, quarantine, or publish the batch for downstream analytics, subject to a bounded run window, explicit owner approval, and a recoverable baseline of manual inspection. The output is a signed quality decision with evidence, never an automatic statement that data is correct.

## 2. Context & Background
A daily ingestion pipeline receives customer-event records from an upstream producer and writes a partition for analysts.
The baseline is an analyst opening a sample file, running ad hoc checks, and deciding informally whether to proceed.
The proposed CLI evaluates a versioned contract before publication and emits machine-readable findings plus a human-readable report.
The decision owner is the data platform lead; source owners explain anomalies, while downstream users decide whether a warned dataset is fit for their use.
A data-quality check measures an observable contract property, not business truth. A complete schema can still describe the wrong population.
The case keeps the output action reversible: quarantined files remain available to authorized investigation; publication can be rolled back by pointer, not deleted.
Illustrative scale is 10 million rows per day; this value exposes cost and capacity boundaries only.
The decision timing is before a partition becomes the default downstream input, so checks must not use labels or outcomes available only later.
The CLI has no authority to repair source data silently. It may propose a bounded transformation only if that transformation is separately versioned and approved.
The system is not a security, privacy, or compliance certification tool; those matters require appropriate review.

**Takeaway:** quality gating is a controlled release decision around a contract, not a generic “clean data” score.

## 3. The Business Question
| Field | Contract |
|---|---|
| Question | Should batch B be published, published with warning, quarantined, or blocked under contract C? |
| Owner | Data platform lead records the release recommendation; source owner supplies remediation evidence. |
| Trigger | Batch arrival and completion of its declared ingestion window. |
| Actions | Publish, publish-with-warning, quarantine, block, or retain prior published partition. |
| Objective | Prevent known contract violations from silently becoming default downstream inputs while preserving an auditable recovery path. |
| Hard constraints | Stable batch ID, contract version, no silent repair, bounded runtime, and explicit action policy. |
| Scope | One partitioned batch, its declared schema and quality rules, and one release decision. |
| Out of scope | Semantic correctness beyond the declared contract, privacy determination, and permanent source remediation. |

A low violation rate is not automatically acceptable: one invalid primary key can corrupt a join, while a small expected null rate may be harmless. The policy maps each check to a severity and owner rather than averaging unlike failures.

## 4. Assumptions & Constraints
| ID | Assumption / constraint | Check | If wrong | Fallback | Owner |
|---|---|---|---|---|
| A-01 | Batch identifier is immutable | compare manifest and path | results attach to wrong data | quarantine | ingestion |
| A-02 | Contract version is pinned before execution | hash contract | threshold drift | block release | platform |
| A-03 | Row count is available without scanning full payload | read manifest and independent count | cost estimate wrong | use bounded scan | platform |
| A-04 | Reference partition is comparable | check calendar and producer version | false drift alert | warn and request review | source owner |
| A-05 | Event timestamps use known timezone | validate metadata | freshness misread | quarantine time rule | producer |
| A-06 | Quarantine storage has capacity | measure bytes and retention | evidence lost | stop intake | operations |
| C-01 | CLI cannot overwrite source batch | write permissions test | irreversible mutation | read-only execution | platform |
| C-02 | Critical check failure blocks default publication | policy test | unsafe default | retain prior pointer | release owner |
| C-03 | Run has timeout budget | timer and cancel test | backlog growth | degrade to declared fast checks | operations |

**Takeaway:** each assumption is testable and each constraint has a release consequence.

## 5. Data, Tools & Method
### 5a. Data Specification
| Element | Definition |
|---|---|
| Unit | one batch-partition and contract pair, keyed by `batch_id`, `partition_date`, and `contract_hash` |
| Row grain | one source event with stable `event_id`, source timestamp, ingestion timestamp, and producer version |
| Inputs | manifest, schema, sampled values, aggregate counts, null counts, distinct estimates, referential samples, and prior comparable partition |
| Outputs | finding ID, rule ID, severity, observed value, threshold, sample evidence, action, and run status |
| Population | all rows in the declared partition; samples are explicitly labeled and never silently generalized |
| Exclusions | unreadable objects, missing manifests, unsupported encodings, and records outside declared partition, each with a reason |
| Time | event time and processing time are separate fields; freshness references processing completion and declared source SLA |
| Label | release action is a policy outcome, not a supervised-learning label |

### 5b. Tools & Stack
- The proposed CLI reads an immutable manifest and contract from versioned storage.
- A planner selects metadata-only, sample, or full-scan checks according to rule cost and severity.
- A runner executes checks with per-rule time and memory budgets.
- A result writer appends findings keyed by batch and contract hash; reruns create attempts but do not overwrite prior evidence.
- A policy engine maps findings to publish, warn, quarantine, or block.
- A report renderer produces JSON for automation and a concise terminal summary for operators.
- A release adapter moves only a dataset pointer after the owner-approved action; source objects remain read-only.

### 5c. Method Summary
1. Validate manifest, identity, schema, and contract before expensive reads.
2. Plan checks by severity, estimated cost, and required sample/full-scan evidence.
3. Execute deterministic rules with bounded resources and record their inputs.
4. Aggregate findings without averaging severities; apply the explicit release policy.
5. Require a human owner for overrides and retain the override rationale.
6. Publish only through an idempotent release pointer update; retain the prior pointer for rollback.

### 5d. Data provenance, quality, exclusions and leakage controls
- Tie every finding to batch ID, object path, content/manifest hash, contract hash, runner revision, and check timestamp.
- Record whether a check inspected metadata, a deterministic sample, or every row; a sample failure and a full-scan failure have different scopes.
- Do not compute freshness from a later corrected timestamp when deciding the original release.
- Do not compare to a prior partition produced by a different schema version without declaring the mapping.
- Retain failed parsing counts rather than dropping unreadable rows before completeness calculations.
- Do not use downstream aggregate outcomes to decide whether the upstream batch was acceptable at release time.
- Treat a manually amended contract as a new version requiring a new decision; do not rerun silently under altered thresholds.

### 5e. Reproducibility and method-selection ledger
| ID | Artifact | Purpose | Access | Limitation |
|---|---|---|---|---|
| CON-01 | quality contract | rule and severity definition | proposed | not an approved organizational policy |
| MAN-01 | batch manifest | identity and expected shape | illustrative | no real source |
| RUN-01 | CLI run record | replay inputs and output | proposed | no executable run attached |
| SMP-01 | deterministic sample seed | sample reproducibility | proposed | sampling error remains |
| POL-01 | release policy | action selection | proposed | requires owner approval |
| ORC-01 | independent count oracle | arithmetic check | proposed | checks only declared property |

**Takeaway:** a useful CLI makes a quality decision reproducible even when the underlying data remains imperfect.

## 6. Analysis Walkthrough
All numbers in this walkthrough are ILLUSTRATIVE and arithmetic is VERIFIED from stated inputs.

### Completeness check
```text
FORMULA [DQ-01]: completeness = non-null required values / rows examined
WHERE: both values are counts [rows]; a required value is defined by contract C.
CONSTRAINT: count parsing failures separately; a sample result estimates only the sample property.
NUMERICAL EXAMPLE: 9,950 non-null `event_id` values / 10,000 examined rows = 0.995 = 99.5%.
INTERPRETATION: compare 99.5% to the rule threshold, not to an undefined notion of clean data.
```

Suppose the critical `event_id` rule requires 100.0% non-null on a full scan. Even 50 missing IDs in 10,000 rows fail this rule because joins and idempotency depend on identity. The release decision is block or quarantine according to the approved policy, despite a high percentage.

### Volume drift check
```text
FORMULA [DQ-02]: drift = (N_current - N_reference) / N_reference
WHERE: N values are comparable row counts [rows] for the same declared partition type.
CONSTRAINT: calendar effects, backfills, schema changes, and source outages can make a reference incomparable.
NUMERICAL EXAMPLE: (9,400,000 - 10,000,000) / 10,000,000 = -0.06 = -6.0%.
INTERPRETATION: a 6.0% drop is a signal to investigate against a stated threshold, not proof of lost data.
```

If the warning threshold is ±5% and the source owner declares a planned holiday effect, the CLI records a warning and owner rationale; it does not silently pass. If no explanation exists, quarantine may be safer than publication.

### Runtime and queue sizing
A full scan at 50,000 rows/s over 10,000,000 rows takes 10,000,000/50,000=200 s=3.333 minutes, excluding object listing and startup. If batches arrive every 10 minutes, service μ=0.3 batches/min and arrivals λ=0.1 batches/min, utilization is 0.333 under this crude steady model. If a degraded scan takes 15 minutes, μ=1/15=0.0667 batches/min < λ; backlog grows by (0.1−0.0667)×60≈2 batches/hour. This is a planning calculation, not a throughput benchmark.

### Cost and sensitivity
If an illustrative full scan consumes 3.333 compute-minutes at $0.20/compute-minute, compute cost is $0.6666 per batch. If 20 minutes/day of operator review costs $0.75/minute, review cost is $15/day and dominates 10 daily scan costs of about $6.67. Doubling full-scan speed halves scan time but does not reduce review cost; reducing false alerts may matter more.

### State trace
- `RECEIVED`: manifest recorded; no release action.
- `VALIDATED`: identity and contract hash match.
- `RUNNING`: checks execute with budgets.
- `WARN`: noncritical finding; owner must acknowledge before pointer update.
- `QUARANTINED`: evidence retained and no default pointer update.
- `BLOCKED`: critical constraint fails; prior published pointer remains.
- `PUBLISHED`: policy and owner approval recorded.
- `REVIEW_REQUIRED`: contract, source, or runner revision changed after a prior result.

**Sensitivity:** changing the full-scan rate from 50,000 to 25,000 rows/s changes run time from 200 to 400 s. Changing the null threshold from 100.0% to 99.0% changes the example action only if the owner has shown that missing IDs cannot break downstream identity; absent that proof, do not relax the rule.

## 7. The Finding
The illustrative analysis supports a contract-driven CLI with block, warn, quarantine, and publish actions—not an automated declaration of correctness. A 99.5% completeness result can be a critical failure when the missing field is an identity key, while a 6.0% volume change is an investigation signal whose release action depends on context. The proposed system is eligible only for a bounded offline rehearsal after contract ownership, rollback, and resource budgets are approved. No production quality improvement is measured.

## 8. System/Solution Design
```text
[immutable batch + manifest] -> [identity/schema validator] -> [check planner] -> [bounded runners]
                                                                      |
[contract store] -------------------------------> [finding ledger] -> [policy engine] -> [owner]
                                                                                         |
                                              [publish pointer] <- publish / warn / quarantine / block
                                                                      |
                                                   [prior pointer + evidence retention]
```

- The manifest validator owns batch identity and prevents a result from attaching to a path that changed after planning.
- The planner owns resource selection but may not downgrade a critical full-scan rule to a sample without a new contract version.
- The finding ledger owns append-only evidence and idempotency of attempted runs.
- The policy engine owns deterministic severity aggregation; it never suppresses a failed critical check.
- The owner console owns acknowledgements and overrides, including rationale and expiry.
- The release adapter owns atomic default-pointer transitions and retains the last known published pointer.
- The rollback path changes the pointer, not the historical evidence; a later fix produces a new batch decision.

Invariant: for any `(batch_id, contract_hash)`, there is at most one active release action. A retry may add a run attempt but must either reproduce the same action from the same evidence or move the record to `REVIEW_REQUIRED` when an input version changed.

## 9. Evaluation & Validation
### T01 — identity mismatch
**Setup:** change manifest hash after planning.
**Oracle:** validator.
**Expected result:** quarantine; no pointer update.
**Boundary:** hash check cannot prove content meaning.

### T02 — critical null
**Setup:** inject one missing event_id.
**Oracle:** full-scan rule.
**Expected result:** block.
**Boundary:** rule scope is identity only.

### T03 — noncritical null
**Setup:** inject optional field gaps.
**Oracle:** policy engine.
**Expected result:** warn with count.
**Boundary:** owner still decides fitness.

### T04 — schema addition
**Setup:** add undocumented field.
**Oracle:** schema comparator.
**Expected result:** review required.
**Boundary:** addition may be harmless but unapproved.

### T05 — schema removal
**Setup:** remove required field.
**Oracle:** schema comparator.
**Expected result:** block.
**Boundary:** fallback may retain prior partition.

### T06 — volume drift
**Setup:** use 9.4M versus 10M.
**Oracle:** drift oracle.
**Expected result:** -6.0% warning/quarantine per policy.
**Boundary:** reference comparability required.

### T07 — duplicate IDs
**Setup:** repeat event_id.
**Oracle:** distinct check.
**Expected result:** critical finding.
**Boundary:** hash collision policy matters.

### T08 — late timestamp
**Setup:** inject future event time.
**Oracle:** time validator.
**Expected result:** quarantine.
**Boundary:** clock source must be known.

### T09 — parse failure
**Setup:** use invalid encoding.
**Oracle:** parser count.
**Expected result:** failure count retained.
**Boundary:** cannot call skipped data complete.

### T10 — timeout
**Setup:** cancel full scan at budget.
**Oracle:** runner state.
**Expected result:** incomplete run; no publish.
**Boundary:** fast checks do not replace missing critical proof.

### T11 — retry
**Setup:** repeat run token.
**Oracle:** finding ledger.
**Expected result:** one active action.
**Boundary:** external compute cost may still repeat.

### T12 — rollback
**Setup:** repoint after publish.
**Oracle:** release adapter.
**Expected result:** prior pointer restored.
**Boundary:** downstream caching may delay visibility.

### T13 — contract change
**Setup:** raise/lower threshold.
**Oracle:** state machine.
**Expected result:** review required.
**Boundary:** new policy needs owner.

### T14 — runner change
**Setup:** update rule implementation.
**Oracle:** revision comparator.
**Expected result:** review required.
**Boundary:** code review remains separate.

### T15 — sample reproducibility
**Setup:** re-run same seed.
**Oracle:** sample selector.
**Expected result:** same rows.
**Boundary:** source mutation invalidates sample.

### T16 — capacity
**Setup:** slow scan to 15 minutes.
**Oracle:** queue model.
**Expected result:** backlog growth identified.
**Boundary:** real variance unmodeled.

Acceptance for an offline rehearsal requires correct state transitions for every test above, retained evidence for every finding, an owner-approved contract, and a demonstrated prior-pointer rollback. It is not production validation.

## 10. Action Taken
No CLI was deployed and no batch was published by this manuscript. The proposed next actions are to obtain an approved contract, create synthetic batches that exercise each severity, run the offline state trace, independently recompute the examples, and have the data platform lead decide whether a limited rehearsal is justified. Any real source access, retention, or policy question remains with appropriate owners.

## 11. Outcome & Results
**Outcome status: not measured.** Verified values are only illustrative arithmetic: 9,950/10,000=99.5%; (9.4M−10M)/10M=−6.0%; 10M/50,000=200 s; and 3.333×$0.20≈$0.67. No claim is made about actual data quality, performance, cost, or adoption.

## 12. Failure Modes & What Went Wrong
### F-01 — manifest points to wrong object
**Symptom:** manifest points to wrong object.
**Competing causes:** for “manifest points to wrong object”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `manifest points to wrong object` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `manifest points to wrong object`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `manifest points to wrong object`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `manifest points to wrong object` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `manifest points to wrong object` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-02 — contract changed after run
**Symptom:** contract changed after run.
**Competing causes:** for “contract changed after run”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `contract changed after run` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `contract changed after run`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `contract changed after run`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `contract changed after run` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `contract changed after run` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-03 — critical nulls averaged with warnings
**Symptom:** critical nulls averaged with warnings.
**Competing causes:** for “critical nulls averaged with warnings”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `critical nulls averaged with warnings` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `critical nulls averaged with warnings`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `critical nulls averaged with warnings`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `critical nulls averaged with warnings` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `critical nulls averaged with warnings` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-04 — reference partition is incomparable
**Symptom:** reference partition is incomparable.
**Competing causes:** for “reference partition is incomparable”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `reference partition is incomparable` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `reference partition is incomparable`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `reference partition is incomparable`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `reference partition is incomparable` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `reference partition is incomparable` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-05 — sample hides rare key failure
**Symptom:** sample hides rare key failure.
**Competing causes:** for “sample hides rare key failure”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `sample hides rare key failure` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `sample hides rare key failure`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `sample hides rare key failure`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `sample hides rare key failure` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `sample hides rare key failure` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-06 — parser silently drops rows
**Symptom:** parser silently drops rows.
**Competing causes:** for “parser silently drops rows”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `parser silently drops rows` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `parser silently drops rows`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `parser silently drops rows`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `parser silently drops rows` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `parser silently drops rows` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-07 — duplicate IDs cross partitions
**Symptom:** duplicate IDs cross partitions.
**Competing causes:** for “duplicate IDs cross partitions”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `duplicate IDs cross partitions` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `duplicate IDs cross partitions`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `duplicate IDs cross partitions`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `duplicate IDs cross partitions` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `duplicate IDs cross partitions` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-08 — timezone causes false freshness
**Symptom:** timezone causes false freshness.
**Competing causes:** for “timezone causes false freshness”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `timezone causes false freshness` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `timezone causes false freshness`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `timezone causes false freshness`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `timezone causes false freshness` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `timezone causes false freshness` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-09 — timeout is treated as pass
**Symptom:** timeout is treated as pass.
**Competing causes:** for “timeout is treated as pass”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `timeout is treated as pass` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `timeout is treated as pass`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `timeout is treated as pass`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `timeout is treated as pass` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `timeout is treated as pass` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-10 — override lacks owner
**Symptom:** override lacks owner.
**Competing causes:** for “override lacks owner”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `override lacks owner` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `override lacks owner`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `override lacks owner`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `override lacks owner` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `override lacks owner` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-11 — retry changes evidence
**Symptom:** retry changes evidence.
**Competing causes:** for “retry changes evidence”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `retry changes evidence` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `retry changes evidence`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `retry changes evidence`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `retry changes evidence` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `retry changes evidence` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-12 — rollbacks delete run records
**Symptom:** rollbacks delete run records.
**Competing causes:** for “rollbacks delete run records”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `rollbacks delete run records` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `rollbacks delete run records`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `rollbacks delete run records`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `rollbacks delete run records` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `rollbacks delete run records` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-13 — warnings are never reviewed
**Symptom:** warnings are never reviewed.
**Competing causes:** for “warnings are never reviewed”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `warnings are never reviewed` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `warnings are never reviewed`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `warnings are never reviewed`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `warnings are never reviewed` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `warnings are never reviewed` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-14 — storage fills with quarantines
**Symptom:** storage fills with quarantines.
**Competing causes:** for “storage fills with quarantines”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `storage fills with quarantines` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `storage fills with quarantines`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `storage fills with quarantines`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `storage fills with quarantines` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `storage fills with quarantines` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-15 — check scans unbounded payload
**Symptom:** check scans unbounded payload.
**Competing causes:** for “check scans unbounded payload”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `check scans unbounded payload` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `check scans unbounded payload`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `check scans unbounded payload`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `check scans unbounded payload` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `check scans unbounded payload` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-16 — source backfill looks like drift
**Symptom:** source backfill looks like drift.
**Competing causes:** for “source backfill looks like drift”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `source backfill looks like drift` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `source backfill looks like drift`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `source backfill looks like drift`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `source backfill looks like drift` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `source backfill looks like drift` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-17 — threshold fitted after failure
**Symptom:** threshold fitted after failure.
**Competing causes:** for “threshold fitted after failure”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `threshold fitted after failure` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `threshold fitted after failure`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `threshold fitted after failure`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `threshold fitted after failure` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `threshold fitted after failure` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-18 — schema type changes silently
**Symptom:** schema type changes silently.
**Competing causes:** for “schema type changes silently”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `schema type changes silently` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `schema type changes silently`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `schema type changes silently`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `schema type changes silently` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `schema type changes silently` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-19 — quality score substitutes for contract
**Symptom:** quality score substitutes for contract.
**Competing causes:** for “quality score substitutes for contract”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `quality score substitutes for contract` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `quality score substitutes for contract`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `quality score substitutes for contract`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `quality score substitutes for contract` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `quality score substitutes for contract` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

### F-20 — downstream cache ignores rollback
**Symptom:** downstream cache ignores rollback.
**Competing causes:** for “downstream cache ignores rollback”, distinguish source behavior, a contract ambiguity, a runner defect, and an operator transition before assigning remediation.
**Diagnostic:** inspect the batch ID, contract hash, runner revision, timestamps, raw `downstream cache ignores rollback` evidence, and active pointer to separate the plausible mechanisms.
**Immediate mitigation:** for `downstream cache ignores rollback`, retain the prior pointer or quarantine the candidate and preserve raw evidence; do not silently repair the batch.
**Correction:** modify the responsible contract, source process, or runner mechanism behind `downstream cache ignores rollback`, then issue a new versioned run rather than mutating the record.
**Verification:** replay the `downstream cache ignores rollback` synthetic batch plus its nearest boundary, then compare state transition, finding count, and release-pointer action.
**Risk:** containing `downstream cache ignores rollback` may delay valid data; schedule the relevant owner review and measure that added queue rather than relaxing the rule blindly.

**Takeaway:** failure analysis is part of the release protocol, not an appendix after data is consumed.

## 13. Transferable Lesson
- Define the release unit and contract before choosing a quality library or dashboard.
- Preserve denominator, sample scope, parse failures, and exclusions beside every percentage.
- Map severity to a reversible release action rather than averaging heterogeneous rules.
- Treat a contract or code revision as a new evidence context.
- Size both check runtime and the human exception queue; a fast rule set can still overwhelm review.
- Use a prior-pointer rollback and append-only evidence to make recovery auditable.
- Distinguish a validated invariant from a broader claim that data is correct for every use.

### Detailed contract-gate ledger
#### Gate 01 — batch path
**Question:** Is `batch path` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `batch path` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `batch path` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the batch path finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `batch path` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `batch path` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `batch path` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 02 — manifest hash
**Question:** Is `manifest hash` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `manifest hash` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `manifest hash` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the manifest hash finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `manifest hash` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `manifest hash` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `manifest hash` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 03 — partition date
**Question:** Is `partition date` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `partition date` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `partition date` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the partition date finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `partition date` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `partition date` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `partition date` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 04 — producer version
**Question:** Is `producer version` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `producer version` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `producer version` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the producer version finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `producer version` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `producer version` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `producer version` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 05 — contract hash
**Question:** Is `contract hash` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `contract hash` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `contract hash` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the contract hash finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `contract hash` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `contract hash` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `contract hash` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 06 — schema fields
**Question:** Is `schema fields` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `schema fields` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `schema fields` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the schema fields finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `schema fields` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `schema fields` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `schema fields` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 07 — field types
**Question:** Is `field types` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `field types` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `field types` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the field types finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `field types` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `field types` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `field types` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 08 — nullability
**Question:** Is `nullability` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `nullability` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `nullability` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the nullability finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `nullability` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `nullability` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `nullability` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 09 — primary key
**Question:** Is `primary key` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `primary key` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `primary key` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the primary key finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `primary key` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `primary key` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `primary key` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 10 — duplicate rate
**Question:** Is `duplicate rate` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `duplicate rate` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `duplicate rate` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the duplicate rate finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `duplicate rate` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `duplicate rate` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `duplicate rate` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 11 — row count
**Question:** Is `row count` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `row count` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `row count` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the row count finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `row count` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `row count` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `row count` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 12 — byte count
**Question:** Is `byte count` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `byte count` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `byte count` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the byte count finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `byte count` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `byte count` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `byte count` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 13 — encoding
**Question:** Is `encoding` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `encoding` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `encoding` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the encoding finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `encoding` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `encoding` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `encoding` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 14 — delimiter
**Question:** Is `delimiter` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `delimiter` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `delimiter` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the delimiter finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `delimiter` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `delimiter` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `delimiter` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 15 — timezone
**Question:** Is `timezone` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `timezone` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `timezone` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the timezone finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `timezone` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `timezone` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `timezone` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 16 — event time
**Question:** Is `event time` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `event time` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `event time` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the event time finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `event time` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `event time` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `event time` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 17 — ingestion time
**Question:** Is `ingestion time` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `ingestion time` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `ingestion time` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the ingestion time finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `ingestion time` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `ingestion time` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `ingestion time` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 18 — freshness SLA
**Question:** Is `freshness SLA` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `freshness SLA` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `freshness SLA` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the freshness SLA finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `freshness SLA` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `freshness SLA` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `freshness SLA` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 19 — late-arrival rule
**Question:** Is `late-arrival rule` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `late-arrival rule` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `late-arrival rule` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the late-arrival rule finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `late-arrival rule` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `late-arrival rule` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `late-arrival rule` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 20 — backfill flag
**Question:** Is `backfill flag` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `backfill flag` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `backfill flag` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the backfill flag finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `backfill flag` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `backfill flag` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `backfill flag` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 21 — sample seed
**Question:** Is `sample seed` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `sample seed` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `sample seed` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the sample seed finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `sample seed` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `sample seed` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `sample seed` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 22 — sample fraction
**Question:** Is `sample fraction` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `sample fraction` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `sample fraction` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the sample fraction finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `sample fraction` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `sample fraction` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `sample fraction` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 23 — full-scan requirement
**Question:** Is `full-scan requirement` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `full-scan requirement` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `full-scan requirement` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the full-scan requirement finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `full-scan requirement` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `full-scan requirement` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `full-scan requirement` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 24 — reference partition
**Question:** Is `reference partition` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `reference partition` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `reference partition` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the reference partition finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `reference partition` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `reference partition` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `reference partition` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 25 — calendar effect
**Question:** Is `calendar effect` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `calendar effect` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `calendar effect` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the calendar effect finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `calendar effect` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `calendar effect` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `calendar effect` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 26 — volume threshold
**Question:** Is `volume threshold` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `volume threshold` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `volume threshold` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the volume threshold finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `volume threshold` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `volume threshold` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `volume threshold` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 27 — null threshold
**Question:** Is `null threshold` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `null threshold` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `null threshold` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the null threshold finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `null threshold` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `null threshold` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `null threshold` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 28 — range bounds
**Question:** Is `range bounds` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `range bounds` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `range bounds` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the range bounds finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `range bounds` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `range bounds` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `range bounds` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 29 — enum values
**Question:** Is `enum values` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `enum values` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `enum values` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the enum values finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `enum values` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `enum values` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `enum values` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 30 — regex format
**Question:** Is `regex format` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `regex format` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `regex format` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the regex format finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `regex format` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `regex format` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `regex format` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 31 — referential key
**Question:** Is `referential key` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `referential key` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `referential key` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the referential key finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `referential key` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `referential key` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `referential key` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 32 — orphan rate
**Question:** Is `orphan rate` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `orphan rate` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `orphan rate` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the orphan rate finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `orphan rate` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `orphan rate` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `orphan rate` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 33 — uniqueness
**Question:** Is `uniqueness` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `uniqueness` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `uniqueness` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the uniqueness finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `uniqueness` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `uniqueness` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `uniqueness` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 34 — monotonicity
**Question:** Is `monotonicity` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `monotonicity` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `monotonicity` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the monotonicity finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `monotonicity` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `monotonicity` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `monotonicity` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 35 — distribution drift
**Question:** Is `distribution drift` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `distribution drift` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `distribution drift` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the distribution drift finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `distribution drift` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `distribution drift` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `distribution drift` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 36 — outlier rule
**Question:** Is `outlier rule` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `outlier rule` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `outlier rule` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the outlier rule finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `outlier rule` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `outlier rule` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `outlier rule` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 37 — parse failures
**Question:** Is `parse failures` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `parse failures` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `parse failures` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the parse failures finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `parse failures` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `parse failures` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `parse failures` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 38 — quarantine location
**Question:** Is `quarantine location` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `quarantine location` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `quarantine location` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the quarantine location finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `quarantine location` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `quarantine location` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `quarantine location` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 39 — retention period
**Question:** Is `retention period` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `retention period` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `retention period` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the retention period finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `retention period` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `retention period` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `retention period` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 40 — runner revision
**Question:** Is `runner revision` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `runner revision` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `runner revision` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the runner revision finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `runner revision` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `runner revision` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `runner revision` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 41 — dependency lock
**Question:** Is `dependency lock` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `dependency lock` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `dependency lock` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the dependency lock finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `dependency lock` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `dependency lock` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `dependency lock` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 42 — memory budget
**Question:** Is `memory budget` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `memory budget` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `memory budget` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the memory budget finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `memory budget` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `memory budget` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `memory budget` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 43 — timeout
**Question:** Is `timeout` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `timeout` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `timeout` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the timeout finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `timeout` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `timeout` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `timeout` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 44 — concurrency cap
**Question:** Is `concurrency cap` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `concurrency cap` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `concurrency cap` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the concurrency cap finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `concurrency cap` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `concurrency cap` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `concurrency cap` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 45 — retry key
**Question:** Is `retry key` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `retry key` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `retry key` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the retry key finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `retry key` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `retry key` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `retry key` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 46 — finding ID
**Question:** Is `finding ID` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `finding ID` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `finding ID` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the finding ID finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `finding ID` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `finding ID` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `finding ID` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 47 — severity map
**Question:** Is `severity map` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `severity map` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `severity map` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the severity map finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `severity map` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `severity map` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `severity map` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 48 — override owner
**Question:** Is `override owner` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `override owner` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `override owner` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the override owner finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `override owner` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `override owner` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `override owner` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 49 — release pointer
**Question:** Is `release pointer` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `release pointer` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `release pointer` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the release pointer finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `release pointer` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `release pointer` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `release pointer` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 50 — rollback pointer
**Question:** Is `rollback pointer` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `rollback pointer` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `rollback pointer` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the rollback pointer finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `rollback pointer` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `rollback pointer` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `rollback pointer` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 51 — notification target
**Question:** Is `notification target` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `notification target` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `notification target` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the notification target finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `notification target` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `notification target` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `notification target` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 52 — audit log
**Question:** Is `audit log` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `audit log` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `audit log` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the audit log finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `audit log` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `audit log` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `audit log` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 53 — metric export
**Question:** Is `metric export` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `metric export` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `metric export` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the metric export finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `metric export` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `metric export` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `metric export` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 54 — cost unit
**Question:** Is `cost unit` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `cost unit` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `cost unit` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the cost unit finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `cost unit` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `cost unit` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `cost unit` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 55 — review minutes
**Question:** Is `review minutes` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `review minutes` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `review minutes` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the review minutes finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `review minutes` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `review minutes` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `review minutes` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 56 — queue cap
**Question:** Is `queue cap` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `queue cap` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `queue cap` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the queue cap finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `queue cap` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `queue cap` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `queue cap` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 57 — kill switch
**Question:** Is `kill switch` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `kill switch` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `kill switch` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the kill switch finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `kill switch` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `kill switch` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `kill switch` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 58 — change trigger
**Question:** Is `change trigger` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `change trigger` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `change trigger` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the change trigger finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `change trigger` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `change trigger` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `change trigger` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 59 — expiry date
**Question:** Is `expiry date` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `expiry date` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `expiry date` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the expiry date finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `expiry date` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `expiry date` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `expiry date` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 60 — source owner
**Question:** Is `source owner` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `source owner` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `source owner` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the source owner finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `source owner` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `source owner` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `source owner` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 61 — downstream owner
**Question:** Is `downstream owner` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `downstream owner` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `downstream owner` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the downstream owner finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `downstream owner` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `downstream owner` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `downstream owner` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 62 — exception rationale
**Question:** Is `exception rationale` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `exception rationale` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `exception rationale` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the exception rationale finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `exception rationale` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `exception rationale` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `exception rationale` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 63 — replay artifact
**Question:** Is `replay artifact` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `replay artifact` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `replay artifact` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the replay artifact finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `replay artifact` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `replay artifact` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `replay artifact` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 64 — approval state
**Question:** Is `approval state` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `approval state` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `approval state` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the approval state finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `approval state` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `approval state` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `approval state` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 65 — terminal state
**Question:** Is `terminal state` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `terminal state` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `terminal state` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the terminal state finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `terminal state` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `terminal state` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `terminal state` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 66 — evidence access
**Question:** Is `evidence access` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `evidence access` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `evidence access` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the evidence access finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `evidence access` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `evidence access` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `evidence access` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 67 — redaction rule
**Question:** Is `redaction rule` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `redaction rule` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `redaction rule` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the redaction rule finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `redaction rule` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `redaction rule` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `redaction rule` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 68 — storage capacity
**Question:** Is `storage capacity` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `storage capacity` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `storage capacity` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the storage capacity finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `storage capacity` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `storage capacity` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `storage capacity` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 69 — incident route
**Question:** Is `incident route` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `incident route` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `incident route` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the incident route finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `incident route` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `incident route` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `incident route` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 70 — postmortem link
**Question:** Is `postmortem link` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `postmortem link` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `postmortem link` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the postmortem link finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `postmortem link` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `postmortem link` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `postmortem link` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 71 — deprecation plan
**Question:** Is `deprecation plan` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `deprecation plan` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `deprecation plan` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the deprecation plan finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `deprecation plan` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `deprecation plan` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `deprecation plan` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 72 — retest trigger
**Question:** Is `retest trigger` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `retest trigger` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `retest trigger` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the retest trigger finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `retest trigger` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `retest trigger` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `retest trigger` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 73 — release calendar
**Question:** Is `release calendar` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `release calendar` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `release calendar` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the release calendar finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `release calendar` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `release calendar` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `release calendar` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

#### Gate 74 — availability boundary
**Question:** Is `availability boundary` defined, versioned, and available before the release decision?
**Evidence:** retain the observed `availability boundary` value, source location, collection time, and contract/run hashes in the ledger; an unavailable value must have its own reason code.
**Failure signal:** `availability boundary` is absent, changes after planning, violates its declared type or unit, or conflicts with the batch manifest.
**Decision effect:** classify the availability boundary finding using its contract severity; preserve its observed count and select REVIEW_REQUIRED, WARN, QUARANTINED, or BLOCKED without inferring a pass.
**Recovery:** obtain corrected `availability boundary` evidence from the responsible owner or create a new versioned contract/run; the prior attempt remains searchable for comparison.
**Verification:** replay the batch/contract pair with the corrected `availability boundary` artifact, confirm the expected finding changes, and inspect both the release pointer and append-only history.
**Boundary:** passing `availability boundary` establishes only that declared property for this batch and contract; it cannot prove downstream semantic fitness, source completeness, or policy approval.

## 14. Connections & Cross-References
This case relies on Python/CLI process control, SQL-style aggregation semantics, Linux-style file and process boundaries, and REST-style contracts; those techniques do not eliminate ownership or evidence requirements.

### Q1. What is the release unit?
**Answer:** A batch-partition plus contract hash; it binds evidence to one release decision. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

### Q2. Why is 99.5% completeness sometimes a block?
**Answer:** Identity keys can be critical even when rare; 50 missing IDs can break joins. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

### Q3. How do you distinguish warning from quarantine?
**Answer:** A warning retains a controlled publish path; quarantine removes the default pointer pending evidence. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

### Q4. What does an immutable manifest prevent?
**Answer:** It prevents a result from being attached to mutated or different input. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

### Q5. Why keep parse failures in the denominator?
**Answer:** Dropped unreadable rows make completeness look better while hiding scope. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

### Q6. How is volume drift computed?
**Answer:** (current-reference)/reference with comparable partition definitions and units. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

### Q7. What makes a reference partition incomparable?
**Answer:** Calendar, backfill, source version, schema, or population differences can create false drift. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

### Q8. How do retries remain safe?
**Answer:** A stable run token and append-only ledger yield one active action even if execution repeats. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

### Q9. What is the rollback invariant?
**Answer:** Rollback restores the prior pointer while retaining the failed candidate and findings. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

### Q10. When does a contract change invalidate a run?
**Answer:** Any threshold, schema, source, or runner change requires REVIEW_REQUIRED and new evidence. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

### Q11. Why is a sample not a full-scan proof?
**Answer:** Sampling can miss rare critical violations; the contract must say which rules require full scan. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

### Q12. How is check capacity sized?
**Answer:** Rows divided by service rate gives run time; compare it with arrival interval and reviewer workload. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

### Q13. What is the owner allowed to override?
**Answer:** Only acknowledge or authorize a documented policy action; an override needs rationale and expiry. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

### Q14. Which changed constraint reverses the design?
**Answer:** If runtime exceeds arrival interval, backlog grows; cap input or redesign checks rather than silently skipping critical rules. Follow-up: identify the artifact and state transition you would inspect. Acceptable alternative: defer publication when evidence is incomplete. Red flag: treating a dashboard percentage as authorization. Scoring: full credit names unit, condition, recovery, and verification.

## 15. Quick Reference Card
| Field | Rule |
|---|---|
| Unit | batch-partition + contract hash |
| Baseline | manual inspection and prior published pointer |
| Critical invariant | at most one active release action per batch/contract pair |
| Completeness | non-null required values / rows examined; preserve parse failures |
| Drift | (current−reference)/reference only for comparable partitions |
| Runtime | rows/service rate; overload if batch arrival exceeds sustainable completion |
| Action | block/warn/quarantine/publish by explicit policy, not averaged score |
| Rollback | restore prior pointer; preserve evidence |
| Evidence | illustrative derived arithmetic only |
| Escalate | identity, contract, timeout, capacity, or authority ambiguity |
