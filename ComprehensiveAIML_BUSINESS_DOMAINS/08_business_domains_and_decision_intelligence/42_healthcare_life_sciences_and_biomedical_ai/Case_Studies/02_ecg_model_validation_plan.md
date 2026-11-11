---
case_id: "CASE-M42-002"
case_title: "ECG Signal-Triage Validation: Quality Gates, Adjudication, and Abstention"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Biomedical signal validation"
primary_pattern: "validation-plan-with-adjudication"
secondary_patterns: ["manual-baseline-comparator", "permission-bounded-workflow"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Biomedical validation lead"
source_index: "INDEX.md"
review_trigger: "Any change in intended output, acquisition device, reference standard, population, or review authority."
---

# ECG Signal-Triage Validation: Quality Gates, Adjudication, and Abstention

> **Evidence declaration:** This is an illustrative validation-design exercise. The records, rates, timings, thresholds, workflow, and arithmetic below are constructed teaching inputs, not a clinical deployment, device benchmark, diagnostic claim, or medical advice. No original scenario body, local data, clinical policy, reference labels, implementation, or outcome measurement was available for review.

## 1. The One-Line Framing

Given one acquired ECG segment and its acquisition metadata, decide whether to route the segment to a quality reject, an indeterminate human-review queue, or a validation-only model score for a biomedical validation lead, subject to no autonomous diagnosis or care action, compared with manual quality review and interpretation.

- **Decision frequency:** one decision per submitted segment; this is a routing/validation task, not treatment selection.
- **Primary success condition:** the offline protocol can measure sensitivity, specificity, abstention burden, and reviewer workload on a predeclared eligible population.
- **Primary pattern:** validation-plan-with-adjudication; a score is not a diagnosis until a separate intended-use and clinical evaluation exists.
- **Why a simpler framing fails:** treating every waveform as a label-ready example hides acquisition failure, ambiguous reference labels, and the manual work created by abstention.
- **Boundary:** no result may write to a record, change an order, or communicate a clinical conclusion.

## 2. Context & Background

The illustrative workflow begins when an acquisition system exports a timestamped waveform and ends when the validation dataset records an adjudicated reference label or an owned exclusion reason.

| Context field | Teaching-case statement |
|---|---|
| Domain | ECG waveform quality and offline validation planning; not clinical interpretation. |
| Decision owner | Biomedical validation lead who can approve an offline study protocol, not a care action. |
| Users | Data curator, blinded reviewer, adjudicator, model analyst, and quality owner. |
| Baseline | Manual two-stage review of eligible waveform segments with a documented exclusion log. |
| Unit | One segment `s`, its acquisition metadata, and one reference-label packet. |
| Time boundary | Features must be available at acquisition cutoff `t0`; reference labels are attached later only for evaluation. |
| Population boundary | Illustrative adult-like and pediatric-like cohorts are not claimed representative; actual inclusion requires protocol approval. |
| Operational boundary | Offline immutable snapshot; no live alerting, chart integration, or device control. |
| Scale | Illustrative 2,400 candidate segments in a validation ledger; not observed volume. |

A segment can be technically readable but still unsuitable for this study because consent/access, lead mapping, sampling rate, or reference provenance is unresolved. The exclusion record is therefore data, not a deletion convenience.

## 3. The Business Question

| Decision contract | Definition |
|---|---|
| Question | Should the owner authorize only a pre-specified offline validation run for a quality-gated ECG triage score, or retain manual-only review until the reference and acquisition contracts are resolved? |
| Decision owner | Biomedical validation lead. |
| Trigger and deadline | A proposed model exists conceptually; protocol approval is needed before data extraction or tuning. |
| Actions | Approve protocol with gates; defer for missing evidence; retain manual baseline; reject unsafe scope expansion. |
| Objective | Learn whether a bounded score can be evaluated without changing care or hiding unusable signals. |
| Hard constraints | No autonomous diagnosis, no care action, no chart write, blinded reference review, immutable split before tuning. |
| Scope | Illustrative offline segments from declared acquisition sources and a fixed snapshot. |
| Out of scope | Diagnosis, treatment, regulatory determination, device certification, and claims of performance. |

The proxy metric is agreement with an adjudicated reference label under a fixed task definition. It can disagree with clinical utility because a high offline score may arise on an unrepresentative or low-artifact population.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Verification | Status | Impact if wrong | Fallback | Owner |
|---|---|---|---|---|---|---|---|
| A-01 | Segment lead order and units are recoverable. | data | schema audit | unverified | features may be meaningless | exclude source | data owner |
| A-02 | Reference label precedes no model feature. | leakage | timestamp audit | unverified | invalid apparent performance | rebuild snapshot | analyst |
| A-03 | Two reviewers can apply the rubric. | label | calibration set | unverified | label noise unknown | pause evaluation | quality lead |
| A-04 | Eligible population is declared before split. | population | signed protocol | unverified | selection bias | re-sample | validation lead |
| C-01 | Model output is validation-only. | safety | interface review | required | prohibited action risk | disconnect output | system owner |
| C-02 | Unusable signals retain reason codes. | integrity | ledger reconciliation | required | silent denominator loss | stop run | curator |
| C-03 | Test partition is not used for threshold selection. | evaluation | access log | required | optimistic result | invalidate run | analyst |

The assumptions most likely to reverse the recommendation are unavailable reference provenance, non-comparable acquisition sources, and insufficient adjudicator agreement. Any one requires deferral rather than threshold relaxation.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Contract |
|---|---|
| Unit of analysis | One fixed-duration segment keyed by `segment_id`; repeated segments from one person are grouped by `subject_id` where available. |
| Candidate window | Illustrative snapshot of 2,400 submitted segments; all timestamps are UTC. |
| Inputs | Voltage samples [mV], sample rate [Hz], lead map, device/source identifier, acquisition timestamp, quality flags. |
| Output | Validation score `p`, quality disposition, and later-only reference label `y`; neither is a clinical conclusion. |
| Label | Adjudicated task label with rubric version and disagreement state; unresolved labels remain indeterminate. |
| Exclusions | Missing source authority, unknown lead mapping, malformed samples, duplicate identifiers, or unavailable reference packet. |
| Join | Exact `segment_id` plus immutable source version; one-to-one reference packet after adjudication. |
| Split | Grouped subject/source-aware train, validation, and locked test partitions; no segment family crosses partitions. |

Availability rule: all candidate features have `available_at <= t0`, where `t0` is the recorded acquisition cutoff. Reference labels, later observations, reviewer notes, and downstream care events are forbidden features.

### 5b. Tools & Stack

| Component | Proposed role | Limitation |
|---|---|---|
| Versioned object store | Immutable waveform and metadata snapshot. | Does not prove source authorization. |
| Schema validator | Checks sampling rate, units, lead names, length, and duplicate keys. | Cannot establish physiologic validity. |
| Signal preprocessing library | Applies declared filtering and resampling. | Parameters must be fit/frozen without test access. |
| Review application | Presents blinded segments and rubric choices. | Review click is not ground truth. |
| Experiment ledger | Stores split, configuration, hashes, scores, exclusions, and adjudication. | A ledger cannot repair a bad reference standard. |
| Offline scorer | Produces score and abstention reason. | No live clinical interface is in scope. |

### 5c. Method Summary

Input waveform and metadata pass schema and quality gates, then a fixed preprocessing transform produces model features. A candidate model produces `p = P(y=1 | x)` only under its training distribution; a threshold converts it to a validation routing proposal, while failed gates produce an abstention reason.

| Candidate | Evidence considered | Strength | Limitation | Status | Revisit if |
|---|---|---|---|---|---|
| Manual double review | Proposed baseline protocol | Explicit human judgment | Cost and disagreement | baseline | workload infeasible |
| Rule-only quality gate | Proposed schema/artifact rules | Transparent exclusions | cannot score task | retained gate | artifacts dominate |
| Calibrated classifier | Illustrative candidate | Auditable score/threshold | needs reference labels | proposed | protocol passes |
| End-to-end opaque model | Not tested | may learn features | hard to audit and leakage-prone | not chosen | clear added value |

### 5d. Data provenance, quality, exclusions and leakage controls

Each extraction receives `snapshot_id`, source owner, permission state, hash, extraction time, schema version, and an exclusion disposition. Transformation code records input hash, output hash, parameters, and whether the transform was fit on training data only.

- Reject duplicated `segment_id` records unless a lineage rule identifies the canonical acquisition.
- Do not impute an unknown lead order; route it to `LEAD_MAP_UNKNOWN`.
- Do not infer labels from later notes, billing codes, reviewer comments, or outcomes.
- Preserve low-quality segments in the candidate denominator with an explicit exclusion reason.
- Lock group membership before tuning so related segments cannot create train/test memorization.

### 5e. Reproducibility and method-selection ledger

| Artifact ID | Location | Version/date | Purpose | Accessibility | Limitation |
|---|---|---|---|---|---|
| SRC-01 | Visible curriculum plan | 2026-11-10 | topic commitment | local | original body unavailable |
| CFG-01 | illustrative protocol configuration | 1.0 | split/gate/threshold contract | local | not executed |
| CALC-01 | Section 6 arithmetic | hand-checked | workload and metric teaching | local | illustrative only |
| RUN-01 | no execution artifact | null | planned run identifier | unavailable | no measured output |

Reproduction is limited to the stated arithmetic and protocol logic. No waveform, model, source body, or local policy artifact was available, so no performance result is reproducible.

## 6. Analysis Walkthrough

The decisive supported conclusion is not a model score: a validation run is only meaningful when every candidate segment has a disposition and the held-out test set remains locked. The following arithmetic is illustrative and independently checkable.

```text
FORMULA [ECG-01]: eligible = candidates - excluded
WHERE: candidates [segments], excluded [segments], eligible [segments].
CONSTRAINT: exclusions are mutually exclusive after a precedence rule.
NUMERICAL EXAMPLE: candidates=2400; malformed=120; authority_missing=80; duplicate=40; eligible=2400-(120+80+40)=2160 segments.
INTERPRETATION: reporting on 2160 segments without the 240 excluded would conceal an 10.0% intake loss.

FORMULA [ECG-02]: sensitivity = TP/(TP+FN); specificity = TN/(TN+FP)
WHERE: TP, FN, TN, FP are adjudicated held-out segment counts.
CONSTRAINT: reference label and threshold are frozen before test scoring.
NUMERICAL EXAMPLE: TP=144, FN=36, TN=1710, FP=90; sensitivity=144/180=0.80; specificity=1710/1800=0.95.
INTERPRETATION: these constructed counts do not establish clinical performance.

FORMULA [ECG-03]: review_minutes = (indeterminate + quality_reject) * minutes_per_segment
WHERE: counts are segments and minutes_per_segment is staffed review minutes/segment.
CONSTRAINT: no parallel-review discount; calendar time is not staffed time.
NUMERICAL EXAMPLE: indeterminate=216; quality_reject=144; minutes_per_segment=3; review_minutes=360*3=1080 staffed minutes=18 staffed hours.
INTERPRETATION: a low abstention rate can still create material owned work.
```

| Step | Input | Operation | Intermediate result | Check |
|---|---|---|---|---|
| 1 | 2,400 candidate segments | apply exclusion precedence | 2,160 eligible | 120+80+40+2160=2400 |
| 2 | eligible set | group-aware partition | 1,296 train; 432 validation; 432 test | sums to 2,160 |
| 3 | 432 validation | select threshold only here | proposed threshold 0.70 | test untouched |
| 4 | 432 locked test | compare scores to adjudication | constructed confusion counts | 144+36+1710+90 requires scaled example population; do not combine with 432 partition |
| 5 | all dispositions | workload calculation | 18 staffed hours | 360*3/60=18 |

The confusion-matrix example is intentionally a separate constructed held-out population of 1,980 adjudicated segments; it must not be added to the 432-partition illustration. Keeping denominators separate prevents a common validation reporting error.

| Sensitivity test | Original | Alternative | Consequence |
|---|---|---|---|
| Artifact exclusion | 10% | 18% | eligible count falls to 1,968; re-evaluate representativeness before model selection. |
| Review time | 3 min/segment | 5 min/segment | 360 reviews consume 30 staffed hours, not 18. |
| False negative count | 36 | 72 | sensitivity becomes 144/(144+72)=0.667; threshold decision must be reconsidered. |
| Prevalence | 180/1,980 | 20/1,980 | predictive values change even if sensitivity/specificity do not. |

No candidate model was actually tested; there is no rejected empirical approach to narrate. Rule-only quality gating remains a considered alternative and cannot answer the validation task label by itself.

## 7. The Finding

Authorize neither deployment nor a performance claim. The illustrative protocol supports only a conditional decision to prepare an offline validation run after source authority, acquisition schema, reference rubric, and blinded adjudication are available. The manual comparator and disposition ledger remain mandatory because 10.0% illustrative intake exclusion and 18 illustrative staffed review hours change the operating meaning of a score. Outcome is not measured.

## 8. System/Solution Design

```text
[Acquisition export] -> [immutable snapshot + permission check] -> [schema/quality gate]
        -> reject ledger / owned manual route
        -> [frozen preprocessing] -> [offline scorer] -> [score + reason]
        -> [blinded review/adjudication] -> [locked evaluation ledger] -> [validation lead decision]
```

The control path is separate from data flow. The data owner can approve snapshot access; the quality owner owns rubric calibration; the analyst cannot unlock the test set; the validation lead can stop the study. No component has authority to diagnose, write a record, or initiate care.

| State | Entry condition | Allowed transition | Owner | Recovery |
|---|---|---|---|
| RECEIVED | metadata accepted | QUALITY_CHECK | curator | quarantine malformed input |
| REJECTED | gate failed | MANUAL_REVIEW only | curator | correct source, do not silently retry |
| ELIGIBLE | gates pass | PARTITIONED | analyst | restore immutable snapshot |
| SCORED | frozen model executes | BLINDED_REVIEW | analyst | score invalidated if config drift |
| INDETERMINATE | score/rubric lacks resolution | ADJUDICATION | quality lead | retain reason and queue age |
| ADJUDICATED | reviewers resolve | LOCKED_EVALUATION | adjudicator | version correction with audit trail |
| STOPPED | safety/integrity trigger | INVESTIGATION | validation lead | preserve evidence, revoke run |

Capacity protection: the queue must expose count, oldest age, owner, and reason code. A score is never permitted to bypass an unresolved queue.

## 9. Evaluation & Validation

| Metric | Numerator | Denominator | Window/population | Comparator | Guardrail | Status |
|---|---|---|---|---|---|---|
| Sensitivity | adjudicated TP | TP+FN | locked eligible test subset | manual adjudication | stratify acquisition source | proposed |
| Specificity | adjudicated TN | TN+FP | locked eligible test subset | manual adjudication | stratify quality state | proposed |
| Abstention rate | reject+indeterminate | all candidates | full snapshot | manual worklist | no silent drops | proposed |
| Reviewer agreement | resolved initial agreements | dual-reviewed items | calibration sample | chance-adjusted analysis if appropriate | disagreement reasons | proposed |
| Queue burden | staffed review minutes | assigned reviewers | declared shift | manual baseline | aging and ownership | illustrative |

Validation sequence: preregister eligibility; freeze split; calibrate rubric on a separate set; lock preprocessing; select threshold using validation only; score test once; reconcile every segment; report confidence intervals and stratum counts only when actual data supports them. A single aggregate metric cannot establish safety or portability.

## 10. Action Taken

No operational intervention was taken. The teaching-case action is a proposed protocol: retain manual review, require a source/access inventory, approve a rubric-calibration exercise before model selection, and block any connected interface. This document does not authorize extraction, model training, or clinical use.

## 11. Outcome & Results

No outcome is measured. The only computed outputs are the illustrative arithmetic checks in Section 6: 2,160 eligible segments from 2,400 candidates under the stated exclusions, 0.80 constructed sensitivity and 0.95 constructed specificity on a separate hypothetical confusion matrix, and 18 staffed review hours under the stated queue assumptions. These are not efficacy, safety, device, or workflow results.

## 12. Failure Modes & What Went Wrong

| ID | Symptom | Competing causes | Diagnostic check | Recovery | Verification |
|---|---|---|---|---|---|
| F-01 | apparently excellent test score | related segment leakage; label leakage | compare subject/source overlap and feature availability | rebuild split and feature manifest | zero forbidden overlap |
| F-02 | missing segments from report | silent parser failure; unlogged exclusion | candidate-to-disposition reconciliation | quarantine and add reason codes | dispositions sum to candidates |
| F-03 | reviewer disagreement spike | ambiguous rubric; poor signal; training drift | blinded disagreement matrix and examples | recalibrate rubric, adjudicate | agreement and reason trend reviewed |
| F-04 | queue grows | artifact rate; insufficient staffing | arrival/service rates and oldest age | pause intake or add owned capacity | backlog drains under observed staffing |
| F-05 | threshold changes after test | analyst pressure; unclear governance | ledger and access audit | invalidate run, relock test | immutable threshold record |
| F-06 | false assurance from aggregate score | subgroup shift; denominator confusion | stratum and exclusion table | defer conclusion | declared strata complete |
| F-07 | unintended clinical interpretation | UI wording; unauthorized workflow | interface/permission review | remove output and retrain users | no external action path |
| F-08 | source mapping changes | device update; lead-name mismatch | schema-version diff | create new cohort and protocol amendment | old/new sources separated |

### Failure-injection and recovery catalog

The following are proposed protocol tests, not executed tests. Each uses a synthetic or authorized non-clinical fixture and must preserve the original evidence record. A pass means the state transition and ledger evidence are observed in a future run; it does not establish clinical validity.

#### FI-001: segment identifier — missing-field event

**Injected condition:** two records share an identifier but different waveform hashes. Remove the required field and require a terminal `REJECTED` disposition rather than a default value.
**Discriminating observation:** hash and compare the raw sample bytes before partitioning. For this segment identifier — missing-field event, the check occurs before release from the current state.
**Required recovery:** quarantine both records until source lineage selects a canonical record. This recovery is required specifically for the segment identifier — missing-field event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-001`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-002: segment identifier — out-of-range event

**Injected condition:** two records share an identifier but different waveform hashes. Insert a value outside the schema range and verify that the rejection reason preserves the original field name and unit.
**Discriminating observation:** hash and compare the raw sample bytes before partitioning. For this segment identifier — out-of-range event, the check occurs before release from the current state.
**Required recovery:** quarantine both records until source lineage selects a canonical record. This recovery is required specifically for the segment identifier — out-of-range event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-002`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-003: segment identifier — duplicate-delivery event

**Injected condition:** two records share an identifier but different waveform hashes. Deliver the same source packet twice and confirm that idempotency prevents a second candidate count.
**Discriminating observation:** hash and compare the raw sample bytes before partitioning. For this segment identifier — duplicate-delivery event, the check occurs before release from the current state.
**Required recovery:** quarantine both records until source lineage selects a canonical record. This recovery is required specifically for the segment identifier — duplicate-delivery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-003`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-004: segment identifier — late-arrival event

**Injected condition:** two records share an identifier but different waveform hashes. Deliver a valid-looking update after snapshot lock and verify that it is stored as a new version, not merged into the run.
**Discriminating observation:** hash and compare the raw sample bytes before partitioning. For this segment identifier — late-arrival event, the check occurs before release from the current state.
**Required recovery:** quarantine both records until source lineage selects a canonical record. This recovery is required specifically for the segment identifier — late-arrival event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-004`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-005: segment identifier — authorization-loss event

**Injected condition:** two records share an identifier but different waveform hashes. Revoke access after snapshot registration and verify that downstream steps stop without deleting the audit trail.
**Discriminating observation:** hash and compare the raw sample bytes before partitioning. For this segment identifier — authorization-loss event, the check occurs before release from the current state.
**Required recovery:** quarantine both records until source lineage selects a canonical record. This recovery is required specifically for the segment identifier — authorization-loss event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-005`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-006: segment identifier — format-drift event

**Injected condition:** two records share an identifier but different waveform hashes. Change the source representation while retaining the old filename and require schema version mismatch detection.
**Discriminating observation:** hash and compare the raw sample bytes before partitioning. For this segment identifier — format-drift event, the check occurs before release from the current state.
**Required recovery:** quarantine both records until source lineage selects a canonical record. This recovery is required specifically for the segment identifier — format-drift event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-006`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-007: segment identifier — partial-write event

**Injected condition:** two records share an identifier but different waveform hashes. Interrupt persistence between waveform and metadata writes and check that no half-record becomes eligible.
**Discriminating observation:** hash and compare the raw sample bytes before partitioning. For this segment identifier — partial-write event, the check occurs before release from the current state.
**Required recovery:** quarantine both records until source lineage selects a canonical record. This recovery is required specifically for the segment identifier — partial-write event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-007`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-008: segment identifier — clock-skew event

**Injected condition:** two records share an identifier but different waveform hashes. Shift one source clock by five minutes and require a time-boundary warning before feature assembly.
**Discriminating observation:** hash and compare the raw sample bytes before partitioning. For this segment identifier — clock-skew event, the check occurs before release from the current state.
**Required recovery:** quarantine both records until source lineage selects a canonical record. This recovery is required specifically for the segment identifier — clock-skew event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-008`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-009: segment identifier — manual-override event

**Injected condition:** two records share an identifier but different waveform hashes. Request an undocumented reviewer override and require rationale, actor, timestamp, and a new versioned disposition.
**Discriminating observation:** hash and compare the raw sample bytes before partitioning. For this segment identifier — manual-override event, the check occurs before release from the current state.
**Required recovery:** quarantine both records until source lineage selects a canonical record. This recovery is required specifically for the segment identifier — manual-override event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-009`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-010: segment identifier — recovery event

**Injected condition:** two records share an identifier but different waveform hashes. Restore the service after a controlled stop and require reconciliation of every in-flight record before new intake.
**Discriminating observation:** hash and compare the raw sample bytes before partitioning. For this segment identifier — recovery event, the check occurs before release from the current state.
**Required recovery:** quarantine both records until source lineage selects a canonical record. This recovery is required specifically for the segment identifier — recovery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-010`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-011: subject grouping key — missing-field event

**Injected condition:** one subject has a missing group key on a repeated acquisition. Remove the required field and require a terminal `REJECTED` disposition rather than a default value.
**Discriminating observation:** compare waveform-family and encounter linkage for cross-partition near matches. For this subject grouping key — missing-field event, the check occurs before release from the current state.
**Required recovery:** assign an unresolved group and exclude it from the locked comparison. This recovery is required specifically for the subject grouping key — missing-field event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-011`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-012: subject grouping key — out-of-range event

**Injected condition:** one subject has a missing group key on a repeated acquisition. Insert a value outside the schema range and verify that the rejection reason preserves the original field name and unit.
**Discriminating observation:** compare waveform-family and encounter linkage for cross-partition near matches. For this subject grouping key — out-of-range event, the check occurs before release from the current state.
**Required recovery:** assign an unresolved group and exclude it from the locked comparison. This recovery is required specifically for the subject grouping key — out-of-range event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-012`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-013: subject grouping key — duplicate-delivery event

**Injected condition:** one subject has a missing group key on a repeated acquisition. Deliver the same source packet twice and confirm that idempotency prevents a second candidate count.
**Discriminating observation:** compare waveform-family and encounter linkage for cross-partition near matches. For this subject grouping key — duplicate-delivery event, the check occurs before release from the current state.
**Required recovery:** assign an unresolved group and exclude it from the locked comparison. This recovery is required specifically for the subject grouping key — duplicate-delivery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-013`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-014: subject grouping key — late-arrival event

**Injected condition:** one subject has a missing group key on a repeated acquisition. Deliver a valid-looking update after snapshot lock and verify that it is stored as a new version, not merged into the run.
**Discriminating observation:** compare waveform-family and encounter linkage for cross-partition near matches. For this subject grouping key — late-arrival event, the check occurs before release from the current state.
**Required recovery:** assign an unresolved group and exclude it from the locked comparison. This recovery is required specifically for the subject grouping key — late-arrival event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-014`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-015: subject grouping key — authorization-loss event

**Injected condition:** one subject has a missing group key on a repeated acquisition. Revoke access after snapshot registration and verify that downstream steps stop without deleting the audit trail.
**Discriminating observation:** compare waveform-family and encounter linkage for cross-partition near matches. For this subject grouping key — authorization-loss event, the check occurs before release from the current state.
**Required recovery:** assign an unresolved group and exclude it from the locked comparison. This recovery is required specifically for the subject grouping key — authorization-loss event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-015`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-016: subject grouping key — format-drift event

**Injected condition:** one subject has a missing group key on a repeated acquisition. Change the source representation while retaining the old filename and require schema version mismatch detection.
**Discriminating observation:** compare waveform-family and encounter linkage for cross-partition near matches. For this subject grouping key — format-drift event, the check occurs before release from the current state.
**Required recovery:** assign an unresolved group and exclude it from the locked comparison. This recovery is required specifically for the subject grouping key — format-drift event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-016`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-017: subject grouping key — partial-write event

**Injected condition:** one subject has a missing group key on a repeated acquisition. Interrupt persistence between waveform and metadata writes and check that no half-record becomes eligible.
**Discriminating observation:** compare waveform-family and encounter linkage for cross-partition near matches. For this subject grouping key — partial-write event, the check occurs before release from the current state.
**Required recovery:** assign an unresolved group and exclude it from the locked comparison. This recovery is required specifically for the subject grouping key — partial-write event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-017`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-018: subject grouping key — clock-skew event

**Injected condition:** one subject has a missing group key on a repeated acquisition. Shift one source clock by five minutes and require a time-boundary warning before feature assembly.
**Discriminating observation:** compare waveform-family and encounter linkage for cross-partition near matches. For this subject grouping key — clock-skew event, the check occurs before release from the current state.
**Required recovery:** assign an unresolved group and exclude it from the locked comparison. This recovery is required specifically for the subject grouping key — clock-skew event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-018`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-019: subject grouping key — manual-override event

**Injected condition:** one subject has a missing group key on a repeated acquisition. Request an undocumented reviewer override and require rationale, actor, timestamp, and a new versioned disposition.
**Discriminating observation:** compare waveform-family and encounter linkage for cross-partition near matches. For this subject grouping key — manual-override event, the check occurs before release from the current state.
**Required recovery:** assign an unresolved group and exclude it from the locked comparison. This recovery is required specifically for the subject grouping key — manual-override event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-019`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-020: subject grouping key — recovery event

**Injected condition:** one subject has a missing group key on a repeated acquisition. Restore the service after a controlled stop and require reconciliation of every in-flight record before new intake.
**Discriminating observation:** compare waveform-family and encounter linkage for cross-partition near matches. For this subject grouping key — recovery event, the check occurs before release from the current state.
**Required recovery:** assign an unresolved group and exclude it from the locked comparison. This recovery is required specifically for the subject grouping key — recovery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-020`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-021: acquisition timestamp — missing-field event

**Injected condition:** the timestamp lacks timezone information. Remove the required field and require a terminal `REJECTED` disposition rather than a default value.
**Discriminating observation:** reject naive timestamps and inspect source offset metadata. For this acquisition timestamp — missing-field event, the check occurs before release from the current state.
**Required recovery:** normalize from an evidenced offset or retain an untimeable exclusion. This recovery is required specifically for the acquisition timestamp — missing-field event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-021`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-022: acquisition timestamp — out-of-range event

**Injected condition:** the timestamp lacks timezone information. Insert a value outside the schema range and verify that the rejection reason preserves the original field name and unit.
**Discriminating observation:** reject naive timestamps and inspect source offset metadata. For this acquisition timestamp — out-of-range event, the check occurs before release from the current state.
**Required recovery:** normalize from an evidenced offset or retain an untimeable exclusion. This recovery is required specifically for the acquisition timestamp — out-of-range event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-022`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-023: acquisition timestamp — duplicate-delivery event

**Injected condition:** the timestamp lacks timezone information. Deliver the same source packet twice and confirm that idempotency prevents a second candidate count.
**Discriminating observation:** reject naive timestamps and inspect source offset metadata. For this acquisition timestamp — duplicate-delivery event, the check occurs before release from the current state.
**Required recovery:** normalize from an evidenced offset or retain an untimeable exclusion. This recovery is required specifically for the acquisition timestamp — duplicate-delivery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-023`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-024: acquisition timestamp — late-arrival event

**Injected condition:** the timestamp lacks timezone information. Deliver a valid-looking update after snapshot lock and verify that it is stored as a new version, not merged into the run.
**Discriminating observation:** reject naive timestamps and inspect source offset metadata. For this acquisition timestamp — late-arrival event, the check occurs before release from the current state.
**Required recovery:** normalize from an evidenced offset or retain an untimeable exclusion. This recovery is required specifically for the acquisition timestamp — late-arrival event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-024`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-025: acquisition timestamp — authorization-loss event

**Injected condition:** the timestamp lacks timezone information. Revoke access after snapshot registration and verify that downstream steps stop without deleting the audit trail.
**Discriminating observation:** reject naive timestamps and inspect source offset metadata. For this acquisition timestamp — authorization-loss event, the check occurs before release from the current state.
**Required recovery:** normalize from an evidenced offset or retain an untimeable exclusion. This recovery is required specifically for the acquisition timestamp — authorization-loss event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-025`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-026: acquisition timestamp — format-drift event

**Injected condition:** the timestamp lacks timezone information. Change the source representation while retaining the old filename and require schema version mismatch detection.
**Discriminating observation:** reject naive timestamps and inspect source offset metadata. For this acquisition timestamp — format-drift event, the check occurs before release from the current state.
**Required recovery:** normalize from an evidenced offset or retain an untimeable exclusion. This recovery is required specifically for the acquisition timestamp — format-drift event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-026`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-027: acquisition timestamp — partial-write event

**Injected condition:** the timestamp lacks timezone information. Interrupt persistence between waveform and metadata writes and check that no half-record becomes eligible.
**Discriminating observation:** reject naive timestamps and inspect source offset metadata. For this acquisition timestamp — partial-write event, the check occurs before release from the current state.
**Required recovery:** normalize from an evidenced offset or retain an untimeable exclusion. This recovery is required specifically for the acquisition timestamp — partial-write event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-027`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-028: acquisition timestamp — clock-skew event

**Injected condition:** the timestamp lacks timezone information. Shift one source clock by five minutes and require a time-boundary warning before feature assembly.
**Discriminating observation:** reject naive timestamps and inspect source offset metadata. For this acquisition timestamp — clock-skew event, the check occurs before release from the current state.
**Required recovery:** normalize from an evidenced offset or retain an untimeable exclusion. This recovery is required specifically for the acquisition timestamp — clock-skew event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-028`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-029: acquisition timestamp — manual-override event

**Injected condition:** the timestamp lacks timezone information. Request an undocumented reviewer override and require rationale, actor, timestamp, and a new versioned disposition.
**Discriminating observation:** reject naive timestamps and inspect source offset metadata. For this acquisition timestamp — manual-override event, the check occurs before release from the current state.
**Required recovery:** normalize from an evidenced offset or retain an untimeable exclusion. This recovery is required specifically for the acquisition timestamp — manual-override event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-029`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-030: acquisition timestamp — recovery event

**Injected condition:** the timestamp lacks timezone information. Restore the service after a controlled stop and require reconciliation of every in-flight record before new intake.
**Discriminating observation:** reject naive timestamps and inspect source offset metadata. For this acquisition timestamp — recovery event, the check occurs before release from the current state.
**Required recovery:** normalize from an evidenced offset or retain an untimeable exclusion. This recovery is required specifically for the acquisition timestamp — recovery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-030`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-031: sampling rate — missing-field event

**Injected condition:** a source changes from 500 Hz to 250 Hz without a version marker. Remove the required field and require a terminal `REJECTED` disposition rather than a default value.
**Discriminating observation:** derive sample interval from the header and compare against the declared schema. For this sampling rate — missing-field event, the check occurs before release from the current state.
**Required recovery:** separate the source version; validate any resampling only in a new protocol branch. This recovery is required specifically for the sampling rate — missing-field event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-031`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-032: sampling rate — out-of-range event

**Injected condition:** a source changes from 500 Hz to 250 Hz without a version marker. Insert a value outside the schema range and verify that the rejection reason preserves the original field name and unit.
**Discriminating observation:** derive sample interval from the header and compare against the declared schema. For this sampling rate — out-of-range event, the check occurs before release from the current state.
**Required recovery:** separate the source version; validate any resampling only in a new protocol branch. This recovery is required specifically for the sampling rate — out-of-range event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-032`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-033: sampling rate — duplicate-delivery event

**Injected condition:** a source changes from 500 Hz to 250 Hz without a version marker. Deliver the same source packet twice and confirm that idempotency prevents a second candidate count.
**Discriminating observation:** derive sample interval from the header and compare against the declared schema. For this sampling rate — duplicate-delivery event, the check occurs before release from the current state.
**Required recovery:** separate the source version; validate any resampling only in a new protocol branch. This recovery is required specifically for the sampling rate — duplicate-delivery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-033`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-034: sampling rate — late-arrival event

**Injected condition:** a source changes from 500 Hz to 250 Hz without a version marker. Deliver a valid-looking update after snapshot lock and verify that it is stored as a new version, not merged into the run.
**Discriminating observation:** derive sample interval from the header and compare against the declared schema. For this sampling rate — late-arrival event, the check occurs before release from the current state.
**Required recovery:** separate the source version; validate any resampling only in a new protocol branch. This recovery is required specifically for the sampling rate — late-arrival event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-034`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-035: sampling rate — authorization-loss event

**Injected condition:** a source changes from 500 Hz to 250 Hz without a version marker. Revoke access after snapshot registration and verify that downstream steps stop without deleting the audit trail.
**Discriminating observation:** derive sample interval from the header and compare against the declared schema. For this sampling rate — authorization-loss event, the check occurs before release from the current state.
**Required recovery:** separate the source version; validate any resampling only in a new protocol branch. This recovery is required specifically for the sampling rate — authorization-loss event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-035`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-036: sampling rate — format-drift event

**Injected condition:** a source changes from 500 Hz to 250 Hz without a version marker. Change the source representation while retaining the old filename and require schema version mismatch detection.
**Discriminating observation:** derive sample interval from the header and compare against the declared schema. For this sampling rate — format-drift event, the check occurs before release from the current state.
**Required recovery:** separate the source version; validate any resampling only in a new protocol branch. This recovery is required specifically for the sampling rate — format-drift event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-036`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-037: sampling rate — partial-write event

**Injected condition:** a source changes from 500 Hz to 250 Hz without a version marker. Interrupt persistence between waveform and metadata writes and check that no half-record becomes eligible.
**Discriminating observation:** derive sample interval from the header and compare against the declared schema. For this sampling rate — partial-write event, the check occurs before release from the current state.
**Required recovery:** separate the source version; validate any resampling only in a new protocol branch. This recovery is required specifically for the sampling rate — partial-write event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-037`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-038: sampling rate — clock-skew event

**Injected condition:** a source changes from 500 Hz to 250 Hz without a version marker. Shift one source clock by five minutes and require a time-boundary warning before feature assembly.
**Discriminating observation:** derive sample interval from the header and compare against the declared schema. For this sampling rate — clock-skew event, the check occurs before release from the current state.
**Required recovery:** separate the source version; validate any resampling only in a new protocol branch. This recovery is required specifically for the sampling rate — clock-skew event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-038`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-039: sampling rate — manual-override event

**Injected condition:** a source changes from 500 Hz to 250 Hz without a version marker. Request an undocumented reviewer override and require rationale, actor, timestamp, and a new versioned disposition.
**Discriminating observation:** derive sample interval from the header and compare against the declared schema. For this sampling rate — manual-override event, the check occurs before release from the current state.
**Required recovery:** separate the source version; validate any resampling only in a new protocol branch. This recovery is required specifically for the sampling rate — manual-override event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-039`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-040: sampling rate — recovery event

**Injected condition:** a source changes from 500 Hz to 250 Hz without a version marker. Restore the service after a controlled stop and require reconciliation of every in-flight record before new intake.
**Discriminating observation:** derive sample interval from the header and compare against the declared schema. For this sampling rate — recovery event, the check occurs before release from the current state.
**Required recovery:** separate the source version; validate any resampling only in a new protocol branch. This recovery is required specifically for the sampling rate — recovery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-040`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-041: voltage unit — missing-field event

**Injected condition:** millivolts are labeled as microvolts. Remove the required field and require a terminal `REJECTED` disposition rather than a default value.
**Discriminating observation:** check plausible amplitude ranges against the declared unit field and raw header. For this voltage unit — missing-field event, the check occurs before release from the current state.
**Required recovery:** stop scoring, correct mapping with source owner, and rerun from immutable input. This recovery is required specifically for the voltage unit — missing-field event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-041`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-042: voltage unit — out-of-range event

**Injected condition:** millivolts are labeled as microvolts. Insert a value outside the schema range and verify that the rejection reason preserves the original field name and unit.
**Discriminating observation:** check plausible amplitude ranges against the declared unit field and raw header. For this voltage unit — out-of-range event, the check occurs before release from the current state.
**Required recovery:** stop scoring, correct mapping with source owner, and rerun from immutable input. This recovery is required specifically for the voltage unit — out-of-range event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-042`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-043: voltage unit — duplicate-delivery event

**Injected condition:** millivolts are labeled as microvolts. Deliver the same source packet twice and confirm that idempotency prevents a second candidate count.
**Discriminating observation:** check plausible amplitude ranges against the declared unit field and raw header. For this voltage unit — duplicate-delivery event, the check occurs before release from the current state.
**Required recovery:** stop scoring, correct mapping with source owner, and rerun from immutable input. This recovery is required specifically for the voltage unit — duplicate-delivery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-043`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-044: voltage unit — late-arrival event

**Injected condition:** millivolts are labeled as microvolts. Deliver a valid-looking update after snapshot lock and verify that it is stored as a new version, not merged into the run.
**Discriminating observation:** check plausible amplitude ranges against the declared unit field and raw header. For this voltage unit — late-arrival event, the check occurs before release from the current state.
**Required recovery:** stop scoring, correct mapping with source owner, and rerun from immutable input. This recovery is required specifically for the voltage unit — late-arrival event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-044`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-045: voltage unit — authorization-loss event

**Injected condition:** millivolts are labeled as microvolts. Revoke access after snapshot registration and verify that downstream steps stop without deleting the audit trail.
**Discriminating observation:** check plausible amplitude ranges against the declared unit field and raw header. For this voltage unit — authorization-loss event, the check occurs before release from the current state.
**Required recovery:** stop scoring, correct mapping with source owner, and rerun from immutable input. This recovery is required specifically for the voltage unit — authorization-loss event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-045`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-046: voltage unit — format-drift event

**Injected condition:** millivolts are labeled as microvolts. Change the source representation while retaining the old filename and require schema version mismatch detection.
**Discriminating observation:** check plausible amplitude ranges against the declared unit field and raw header. For this voltage unit — format-drift event, the check occurs before release from the current state.
**Required recovery:** stop scoring, correct mapping with source owner, and rerun from immutable input. This recovery is required specifically for the voltage unit — format-drift event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-046`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-047: voltage unit — partial-write event

**Injected condition:** millivolts are labeled as microvolts. Interrupt persistence between waveform and metadata writes and check that no half-record becomes eligible.
**Discriminating observation:** check plausible amplitude ranges against the declared unit field and raw header. For this voltage unit — partial-write event, the check occurs before release from the current state.
**Required recovery:** stop scoring, correct mapping with source owner, and rerun from immutable input. This recovery is required specifically for the voltage unit — partial-write event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-047`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-048: voltage unit — clock-skew event

**Injected condition:** millivolts are labeled as microvolts. Shift one source clock by five minutes and require a time-boundary warning before feature assembly.
**Discriminating observation:** check plausible amplitude ranges against the declared unit field and raw header. For this voltage unit — clock-skew event, the check occurs before release from the current state.
**Required recovery:** stop scoring, correct mapping with source owner, and rerun from immutable input. This recovery is required specifically for the voltage unit — clock-skew event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-048`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-049: voltage unit — manual-override event

**Injected condition:** millivolts are labeled as microvolts. Request an undocumented reviewer override and require rationale, actor, timestamp, and a new versioned disposition.
**Discriminating observation:** check plausible amplitude ranges against the declared unit field and raw header. For this voltage unit — manual-override event, the check occurs before release from the current state.
**Required recovery:** stop scoring, correct mapping with source owner, and rerun from immutable input. This recovery is required specifically for the voltage unit — manual-override event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-049`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-050: voltage unit — recovery event

**Injected condition:** millivolts are labeled as microvolts. Restore the service after a controlled stop and require reconciliation of every in-flight record before new intake.
**Discriminating observation:** check plausible amplitude ranges against the declared unit field and raw header. For this voltage unit — recovery event, the check occurs before release from the current state.
**Required recovery:** stop scoring, correct mapping with source owner, and rerun from immutable input. This recovery is required specifically for the voltage unit — recovery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-050`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-051: lead mapping — missing-field event

**Injected condition:** two lead labels are swapped in an export. Remove the required field and require a terminal `REJECTED` disposition rather than a default value.
**Discriminating observation:** compare ordered lead manifest against acquisition-system contract. For this lead mapping — missing-field event, the check occurs before release from the current state.
**Required recovery:** route the batch to LEAD_MAP_UNKNOWN and require a corrected manifest. This recovery is required specifically for the lead mapping — missing-field event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-051`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-052: lead mapping — out-of-range event

**Injected condition:** two lead labels are swapped in an export. Insert a value outside the schema range and verify that the rejection reason preserves the original field name and unit.
**Discriminating observation:** compare ordered lead manifest against acquisition-system contract. For this lead mapping — out-of-range event, the check occurs before release from the current state.
**Required recovery:** route the batch to LEAD_MAP_UNKNOWN and require a corrected manifest. This recovery is required specifically for the lead mapping — out-of-range event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-052`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-053: lead mapping — duplicate-delivery event

**Injected condition:** two lead labels are swapped in an export. Deliver the same source packet twice and confirm that idempotency prevents a second candidate count.
**Discriminating observation:** compare ordered lead manifest against acquisition-system contract. For this lead mapping — duplicate-delivery event, the check occurs before release from the current state.
**Required recovery:** route the batch to LEAD_MAP_UNKNOWN and require a corrected manifest. This recovery is required specifically for the lead mapping — duplicate-delivery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-053`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-054: lead mapping — late-arrival event

**Injected condition:** two lead labels are swapped in an export. Deliver a valid-looking update after snapshot lock and verify that it is stored as a new version, not merged into the run.
**Discriminating observation:** compare ordered lead manifest against acquisition-system contract. For this lead mapping — late-arrival event, the check occurs before release from the current state.
**Required recovery:** route the batch to LEAD_MAP_UNKNOWN and require a corrected manifest. This recovery is required specifically for the lead mapping — late-arrival event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-054`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-055: lead mapping — authorization-loss event

**Injected condition:** two lead labels are swapped in an export. Revoke access after snapshot registration and verify that downstream steps stop without deleting the audit trail.
**Discriminating observation:** compare ordered lead manifest against acquisition-system contract. For this lead mapping — authorization-loss event, the check occurs before release from the current state.
**Required recovery:** route the batch to LEAD_MAP_UNKNOWN and require a corrected manifest. This recovery is required specifically for the lead mapping — authorization-loss event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-055`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-056: lead mapping — format-drift event

**Injected condition:** two lead labels are swapped in an export. Change the source representation while retaining the old filename and require schema version mismatch detection.
**Discriminating observation:** compare ordered lead manifest against acquisition-system contract. For this lead mapping — format-drift event, the check occurs before release from the current state.
**Required recovery:** route the batch to LEAD_MAP_UNKNOWN and require a corrected manifest. This recovery is required specifically for the lead mapping — format-drift event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-056`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-057: lead mapping — partial-write event

**Injected condition:** two lead labels are swapped in an export. Interrupt persistence between waveform and metadata writes and check that no half-record becomes eligible.
**Discriminating observation:** compare ordered lead manifest against acquisition-system contract. For this lead mapping — partial-write event, the check occurs before release from the current state.
**Required recovery:** route the batch to LEAD_MAP_UNKNOWN and require a corrected manifest. This recovery is required specifically for the lead mapping — partial-write event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-057`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-058: lead mapping — clock-skew event

**Injected condition:** two lead labels are swapped in an export. Shift one source clock by five minutes and require a time-boundary warning before feature assembly.
**Discriminating observation:** compare ordered lead manifest against acquisition-system contract. For this lead mapping — clock-skew event, the check occurs before release from the current state.
**Required recovery:** route the batch to LEAD_MAP_UNKNOWN and require a corrected manifest. This recovery is required specifically for the lead mapping — clock-skew event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-058`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-059: lead mapping — manual-override event

**Injected condition:** two lead labels are swapped in an export. Request an undocumented reviewer override and require rationale, actor, timestamp, and a new versioned disposition.
**Discriminating observation:** compare ordered lead manifest against acquisition-system contract. For this lead mapping — manual-override event, the check occurs before release from the current state.
**Required recovery:** route the batch to LEAD_MAP_UNKNOWN and require a corrected manifest. This recovery is required specifically for the lead mapping — manual-override event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-059`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-060: lead mapping — recovery event

**Injected condition:** two lead labels are swapped in an export. Restore the service after a controlled stop and require reconciliation of every in-flight record before new intake.
**Discriminating observation:** compare ordered lead manifest against acquisition-system contract. For this lead mapping — recovery event, the check occurs before release from the current state.
**Required recovery:** route the batch to LEAD_MAP_UNKNOWN and require a corrected manifest. This recovery is required specifically for the lead mapping — recovery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-060`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-061: segment duration — missing-field event

**Injected condition:** a 10-second contract contains truncated 7-second arrays. Remove the required field and require a terminal `REJECTED` disposition rather than a default value.
**Discriminating observation:** calculate expected sample count as duration times sampling rate. For this segment duration — missing-field event, the check occurs before release from the current state.
**Required recovery:** reject rather than pad unless the amended protocol specifies a tested method. This recovery is required specifically for the segment duration — missing-field event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-061`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-062: segment duration — out-of-range event

**Injected condition:** a 10-second contract contains truncated 7-second arrays. Insert a value outside the schema range and verify that the rejection reason preserves the original field name and unit.
**Discriminating observation:** calculate expected sample count as duration times sampling rate. For this segment duration — out-of-range event, the check occurs before release from the current state.
**Required recovery:** reject rather than pad unless the amended protocol specifies a tested method. This recovery is required specifically for the segment duration — out-of-range event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-062`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-063: segment duration — duplicate-delivery event

**Injected condition:** a 10-second contract contains truncated 7-second arrays. Deliver the same source packet twice and confirm that idempotency prevents a second candidate count.
**Discriminating observation:** calculate expected sample count as duration times sampling rate. For this segment duration — duplicate-delivery event, the check occurs before release from the current state.
**Required recovery:** reject rather than pad unless the amended protocol specifies a tested method. This recovery is required specifically for the segment duration — duplicate-delivery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-063`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-064: segment duration — late-arrival event

**Injected condition:** a 10-second contract contains truncated 7-second arrays. Deliver a valid-looking update after snapshot lock and verify that it is stored as a new version, not merged into the run.
**Discriminating observation:** calculate expected sample count as duration times sampling rate. For this segment duration — late-arrival event, the check occurs before release from the current state.
**Required recovery:** reject rather than pad unless the amended protocol specifies a tested method. This recovery is required specifically for the segment duration — late-arrival event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-064`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-065: segment duration — authorization-loss event

**Injected condition:** a 10-second contract contains truncated 7-second arrays. Revoke access after snapshot registration and verify that downstream steps stop without deleting the audit trail.
**Discriminating observation:** calculate expected sample count as duration times sampling rate. For this segment duration — authorization-loss event, the check occurs before release from the current state.
**Required recovery:** reject rather than pad unless the amended protocol specifies a tested method. This recovery is required specifically for the segment duration — authorization-loss event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-065`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-066: segment duration — format-drift event

**Injected condition:** a 10-second contract contains truncated 7-second arrays. Change the source representation while retaining the old filename and require schema version mismatch detection.
**Discriminating observation:** calculate expected sample count as duration times sampling rate. For this segment duration — format-drift event, the check occurs before release from the current state.
**Required recovery:** reject rather than pad unless the amended protocol specifies a tested method. This recovery is required specifically for the segment duration — format-drift event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-066`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-067: segment duration — partial-write event

**Injected condition:** a 10-second contract contains truncated 7-second arrays. Interrupt persistence between waveform and metadata writes and check that no half-record becomes eligible.
**Discriminating observation:** calculate expected sample count as duration times sampling rate. For this segment duration — partial-write event, the check occurs before release from the current state.
**Required recovery:** reject rather than pad unless the amended protocol specifies a tested method. This recovery is required specifically for the segment duration — partial-write event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-067`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-068: segment duration — clock-skew event

**Injected condition:** a 10-second contract contains truncated 7-second arrays. Shift one source clock by five minutes and require a time-boundary warning before feature assembly.
**Discriminating observation:** calculate expected sample count as duration times sampling rate. For this segment duration — clock-skew event, the check occurs before release from the current state.
**Required recovery:** reject rather than pad unless the amended protocol specifies a tested method. This recovery is required specifically for the segment duration — clock-skew event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-068`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-069: segment duration — manual-override event

**Injected condition:** a 10-second contract contains truncated 7-second arrays. Request an undocumented reviewer override and require rationale, actor, timestamp, and a new versioned disposition.
**Discriminating observation:** calculate expected sample count as duration times sampling rate. For this segment duration — manual-override event, the check occurs before release from the current state.
**Required recovery:** reject rather than pad unless the amended protocol specifies a tested method. This recovery is required specifically for the segment duration — manual-override event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-069`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-070: segment duration — recovery event

**Injected condition:** a 10-second contract contains truncated 7-second arrays. Restore the service after a controlled stop and require reconciliation of every in-flight record before new intake.
**Discriminating observation:** calculate expected sample count as duration times sampling rate. For this segment duration — recovery event, the check occurs before release from the current state.
**Required recovery:** reject rather than pad unless the amended protocol specifies a tested method. This recovery is required specifically for the segment duration — recovery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-070`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-071: filter configuration — missing-field event

**Injected condition:** the high-pass cutoff changes after validation tuning. Remove the required field and require a terminal `REJECTED` disposition rather than a default value.
**Discriminating observation:** fingerprint parameter file and compare it with the approved configuration hash. For this filter configuration — missing-field event, the check occurs before release from the current state.
**Required recovery:** invalidate affected scores and repeat selection before touching the test partition. This recovery is required specifically for the filter configuration — missing-field event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-071`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-072: filter configuration — out-of-range event

**Injected condition:** the high-pass cutoff changes after validation tuning. Insert a value outside the schema range and verify that the rejection reason preserves the original field name and unit.
**Discriminating observation:** fingerprint parameter file and compare it with the approved configuration hash. For this filter configuration — out-of-range event, the check occurs before release from the current state.
**Required recovery:** invalidate affected scores and repeat selection before touching the test partition. This recovery is required specifically for the filter configuration — out-of-range event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-072`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-073: filter configuration — duplicate-delivery event

**Injected condition:** the high-pass cutoff changes after validation tuning. Deliver the same source packet twice and confirm that idempotency prevents a second candidate count.
**Discriminating observation:** fingerprint parameter file and compare it with the approved configuration hash. For this filter configuration — duplicate-delivery event, the check occurs before release from the current state.
**Required recovery:** invalidate affected scores and repeat selection before touching the test partition. This recovery is required specifically for the filter configuration — duplicate-delivery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-073`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-074: filter configuration — late-arrival event

**Injected condition:** the high-pass cutoff changes after validation tuning. Deliver a valid-looking update after snapshot lock and verify that it is stored as a new version, not merged into the run.
**Discriminating observation:** fingerprint parameter file and compare it with the approved configuration hash. For this filter configuration — late-arrival event, the check occurs before release from the current state.
**Required recovery:** invalidate affected scores and repeat selection before touching the test partition. This recovery is required specifically for the filter configuration — late-arrival event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-074`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-075: filter configuration — authorization-loss event

**Injected condition:** the high-pass cutoff changes after validation tuning. Revoke access after snapshot registration and verify that downstream steps stop without deleting the audit trail.
**Discriminating observation:** fingerprint parameter file and compare it with the approved configuration hash. For this filter configuration — authorization-loss event, the check occurs before release from the current state.
**Required recovery:** invalidate affected scores and repeat selection before touching the test partition. This recovery is required specifically for the filter configuration — authorization-loss event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-075`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-076: filter configuration — format-drift event

**Injected condition:** the high-pass cutoff changes after validation tuning. Change the source representation while retaining the old filename and require schema version mismatch detection.
**Discriminating observation:** fingerprint parameter file and compare it with the approved configuration hash. For this filter configuration — format-drift event, the check occurs before release from the current state.
**Required recovery:** invalidate affected scores and repeat selection before touching the test partition. This recovery is required specifically for the filter configuration — format-drift event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-076`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-077: filter configuration — partial-write event

**Injected condition:** the high-pass cutoff changes after validation tuning. Interrupt persistence between waveform and metadata writes and check that no half-record becomes eligible.
**Discriminating observation:** fingerprint parameter file and compare it with the approved configuration hash. For this filter configuration — partial-write event, the check occurs before release from the current state.
**Required recovery:** invalidate affected scores and repeat selection before touching the test partition. This recovery is required specifically for the filter configuration — partial-write event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-077`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-078: filter configuration — clock-skew event

**Injected condition:** the high-pass cutoff changes after validation tuning. Shift one source clock by five minutes and require a time-boundary warning before feature assembly.
**Discriminating observation:** fingerprint parameter file and compare it with the approved configuration hash. For this filter configuration — clock-skew event, the check occurs before release from the current state.
**Required recovery:** invalidate affected scores and repeat selection before touching the test partition. This recovery is required specifically for the filter configuration — clock-skew event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-078`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-079: filter configuration — manual-override event

**Injected condition:** the high-pass cutoff changes after validation tuning. Request an undocumented reviewer override and require rationale, actor, timestamp, and a new versioned disposition.
**Discriminating observation:** fingerprint parameter file and compare it with the approved configuration hash. For this filter configuration — manual-override event, the check occurs before release from the current state.
**Required recovery:** invalidate affected scores and repeat selection before touching the test partition. This recovery is required specifically for the filter configuration — manual-override event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-079`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-080: filter configuration — recovery event

**Injected condition:** the high-pass cutoff changes after validation tuning. Restore the service after a controlled stop and require reconciliation of every in-flight record before new intake.
**Discriminating observation:** fingerprint parameter file and compare it with the approved configuration hash. For this filter configuration — recovery event, the check occurs before release from the current state.
**Required recovery:** invalidate affected scores and repeat selection before touching the test partition. This recovery is required specifically for the filter configuration — recovery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-080`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-081: quality flag — missing-field event

**Injected condition:** a device quality flag arrives after the acquisition cutoff. Remove the required field and require a terminal `REJECTED` disposition rather than a default value.
**Discriminating observation:** compare flag creation time with the declared cutoff `t0`. For this quality flag — missing-field event, the check occurs before release from the current state.
**Required recovery:** exclude the late flag from features and record the availability violation. This recovery is required specifically for the quality flag — missing-field event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-081`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-082: quality flag — out-of-range event

**Injected condition:** a device quality flag arrives after the acquisition cutoff. Insert a value outside the schema range and verify that the rejection reason preserves the original field name and unit.
**Discriminating observation:** compare flag creation time with the declared cutoff `t0`. For this quality flag — out-of-range event, the check occurs before release from the current state.
**Required recovery:** exclude the late flag from features and record the availability violation. This recovery is required specifically for the quality flag — out-of-range event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-082`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-083: quality flag — duplicate-delivery event

**Injected condition:** a device quality flag arrives after the acquisition cutoff. Deliver the same source packet twice and confirm that idempotency prevents a second candidate count.
**Discriminating observation:** compare flag creation time with the declared cutoff `t0`. For this quality flag — duplicate-delivery event, the check occurs before release from the current state.
**Required recovery:** exclude the late flag from features and record the availability violation. This recovery is required specifically for the quality flag — duplicate-delivery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-083`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-084: quality flag — late-arrival event

**Injected condition:** a device quality flag arrives after the acquisition cutoff. Deliver a valid-looking update after snapshot lock and verify that it is stored as a new version, not merged into the run.
**Discriminating observation:** compare flag creation time with the declared cutoff `t0`. For this quality flag — late-arrival event, the check occurs before release from the current state.
**Required recovery:** exclude the late flag from features and record the availability violation. This recovery is required specifically for the quality flag — late-arrival event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-084`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-085: quality flag — authorization-loss event

**Injected condition:** a device quality flag arrives after the acquisition cutoff. Revoke access after snapshot registration and verify that downstream steps stop without deleting the audit trail.
**Discriminating observation:** compare flag creation time with the declared cutoff `t0`. For this quality flag — authorization-loss event, the check occurs before release from the current state.
**Required recovery:** exclude the late flag from features and record the availability violation. This recovery is required specifically for the quality flag — authorization-loss event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-085`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-086: quality flag — format-drift event

**Injected condition:** a device quality flag arrives after the acquisition cutoff. Change the source representation while retaining the old filename and require schema version mismatch detection.
**Discriminating observation:** compare flag creation time with the declared cutoff `t0`. For this quality flag — format-drift event, the check occurs before release from the current state.
**Required recovery:** exclude the late flag from features and record the availability violation. This recovery is required specifically for the quality flag — format-drift event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-086`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-087: quality flag — partial-write event

**Injected condition:** a device quality flag arrives after the acquisition cutoff. Interrupt persistence between waveform and metadata writes and check that no half-record becomes eligible.
**Discriminating observation:** compare flag creation time with the declared cutoff `t0`. For this quality flag — partial-write event, the check occurs before release from the current state.
**Required recovery:** exclude the late flag from features and record the availability violation. This recovery is required specifically for the quality flag — partial-write event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-087`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-088: quality flag — clock-skew event

**Injected condition:** a device quality flag arrives after the acquisition cutoff. Shift one source clock by five minutes and require a time-boundary warning before feature assembly.
**Discriminating observation:** compare flag creation time with the declared cutoff `t0`. For this quality flag — clock-skew event, the check occurs before release from the current state.
**Required recovery:** exclude the late flag from features and record the availability violation. This recovery is required specifically for the quality flag — clock-skew event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-088`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-089: quality flag — manual-override event

**Injected condition:** a device quality flag arrives after the acquisition cutoff. Request an undocumented reviewer override and require rationale, actor, timestamp, and a new versioned disposition.
**Discriminating observation:** compare flag creation time with the declared cutoff `t0`. For this quality flag — manual-override event, the check occurs before release from the current state.
**Required recovery:** exclude the late flag from features and record the availability violation. This recovery is required specifically for the quality flag — manual-override event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-089`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-090: quality flag — recovery event

**Injected condition:** a device quality flag arrives after the acquisition cutoff. Restore the service after a controlled stop and require reconciliation of every in-flight record before new intake.
**Discriminating observation:** compare flag creation time with the declared cutoff `t0`. For this quality flag — recovery event, the check occurs before release from the current state.
**Required recovery:** exclude the late flag from features and record the availability violation. This recovery is required specifically for the quality flag — recovery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-090`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-091: reference packet — missing-field event

**Injected condition:** a reviewer note contains the model score. Remove the required field and require a terminal `REJECTED` disposition rather than a default value.
**Discriminating observation:** inspect review payload fields and reviewer interface event log. For this reference packet — missing-field event, the check occurs before release from the current state.
**Required recovery:** rebuild blinded packets and treat prior labels as contaminated. This recovery is required specifically for the reference packet — missing-field event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-091`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-092: reference packet — out-of-range event

**Injected condition:** a reviewer note contains the model score. Insert a value outside the schema range and verify that the rejection reason preserves the original field name and unit.
**Discriminating observation:** inspect review payload fields and reviewer interface event log. For this reference packet — out-of-range event, the check occurs before release from the current state.
**Required recovery:** rebuild blinded packets and treat prior labels as contaminated. This recovery is required specifically for the reference packet — out-of-range event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-092`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-093: reference packet — duplicate-delivery event

**Injected condition:** a reviewer note contains the model score. Deliver the same source packet twice and confirm that idempotency prevents a second candidate count.
**Discriminating observation:** inspect review payload fields and reviewer interface event log. For this reference packet — duplicate-delivery event, the check occurs before release from the current state.
**Required recovery:** rebuild blinded packets and treat prior labels as contaminated. This recovery is required specifically for the reference packet — duplicate-delivery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-093`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-094: reference packet — late-arrival event

**Injected condition:** a reviewer note contains the model score. Deliver a valid-looking update after snapshot lock and verify that it is stored as a new version, not merged into the run.
**Discriminating observation:** inspect review payload fields and reviewer interface event log. For this reference packet — late-arrival event, the check occurs before release from the current state.
**Required recovery:** rebuild blinded packets and treat prior labels as contaminated. This recovery is required specifically for the reference packet — late-arrival event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-094`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-095: reference packet — authorization-loss event

**Injected condition:** a reviewer note contains the model score. Revoke access after snapshot registration and verify that downstream steps stop without deleting the audit trail.
**Discriminating observation:** inspect review payload fields and reviewer interface event log. For this reference packet — authorization-loss event, the check occurs before release from the current state.
**Required recovery:** rebuild blinded packets and treat prior labels as contaminated. This recovery is required specifically for the reference packet — authorization-loss event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-095`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-096: reference packet — format-drift event

**Injected condition:** a reviewer note contains the model score. Change the source representation while retaining the old filename and require schema version mismatch detection.
**Discriminating observation:** inspect review payload fields and reviewer interface event log. For this reference packet — format-drift event, the check occurs before release from the current state.
**Required recovery:** rebuild blinded packets and treat prior labels as contaminated. This recovery is required specifically for the reference packet — format-drift event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-096`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-097: reference packet — partial-write event

**Injected condition:** a reviewer note contains the model score. Interrupt persistence between waveform and metadata writes and check that no half-record becomes eligible.
**Discriminating observation:** inspect review payload fields and reviewer interface event log. For this reference packet — partial-write event, the check occurs before release from the current state.
**Required recovery:** rebuild blinded packets and treat prior labels as contaminated. This recovery is required specifically for the reference packet — partial-write event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-097`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-098: reference packet — clock-skew event

**Injected condition:** a reviewer note contains the model score. Shift one source clock by five minutes and require a time-boundary warning before feature assembly.
**Discriminating observation:** inspect review payload fields and reviewer interface event log. For this reference packet — clock-skew event, the check occurs before release from the current state.
**Required recovery:** rebuild blinded packets and treat prior labels as contaminated. This recovery is required specifically for the reference packet — clock-skew event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-098`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-099: reference packet — manual-override event

**Injected condition:** a reviewer note contains the model score. Request an undocumented reviewer override and require rationale, actor, timestamp, and a new versioned disposition.
**Discriminating observation:** inspect review payload fields and reviewer interface event log. For this reference packet — manual-override event, the check occurs before release from the current state.
**Required recovery:** rebuild blinded packets and treat prior labels as contaminated. This recovery is required specifically for the reference packet — manual-override event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-099`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-100: reference packet — recovery event

**Injected condition:** a reviewer note contains the model score. Restore the service after a controlled stop and require reconciliation of every in-flight record before new intake.
**Discriminating observation:** inspect review payload fields and reviewer interface event log. For this reference packet — recovery event, the check occurs before release from the current state.
**Required recovery:** rebuild blinded packets and treat prior labels as contaminated. This recovery is required specifically for the reference packet — recovery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-100`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-101: reviewer identity — missing-field event

**Injected condition:** the same reviewer performs both initial reviews by default. Remove the required field and require a terminal `REJECTED` disposition rather than a default value.
**Discriminating observation:** audit reviewer IDs per segment and assignment rules. For this reviewer identity — missing-field event, the check occurs before release from the current state.
**Required recovery:** reassign a calibration sample to distinct qualified reviewers. This recovery is required specifically for the reviewer identity — missing-field event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-101`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-102: reviewer identity — out-of-range event

**Injected condition:** the same reviewer performs both initial reviews by default. Insert a value outside the schema range and verify that the rejection reason preserves the original field name and unit.
**Discriminating observation:** audit reviewer IDs per segment and assignment rules. For this reviewer identity — out-of-range event, the check occurs before release from the current state.
**Required recovery:** reassign a calibration sample to distinct qualified reviewers. This recovery is required specifically for the reviewer identity — out-of-range event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-102`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-103: reviewer identity — duplicate-delivery event

**Injected condition:** the same reviewer performs both initial reviews by default. Deliver the same source packet twice and confirm that idempotency prevents a second candidate count.
**Discriminating observation:** audit reviewer IDs per segment and assignment rules. For this reviewer identity — duplicate-delivery event, the check occurs before release from the current state.
**Required recovery:** reassign a calibration sample to distinct qualified reviewers. This recovery is required specifically for the reviewer identity — duplicate-delivery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-103`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-104: reviewer identity — late-arrival event

**Injected condition:** the same reviewer performs both initial reviews by default. Deliver a valid-looking update after snapshot lock and verify that it is stored as a new version, not merged into the run.
**Discriminating observation:** audit reviewer IDs per segment and assignment rules. For this reviewer identity — late-arrival event, the check occurs before release from the current state.
**Required recovery:** reassign a calibration sample to distinct qualified reviewers. This recovery is required specifically for the reviewer identity — late-arrival event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-104`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-105: reviewer identity — authorization-loss event

**Injected condition:** the same reviewer performs both initial reviews by default. Revoke access after snapshot registration and verify that downstream steps stop without deleting the audit trail.
**Discriminating observation:** audit reviewer IDs per segment and assignment rules. For this reviewer identity — authorization-loss event, the check occurs before release from the current state.
**Required recovery:** reassign a calibration sample to distinct qualified reviewers. This recovery is required specifically for the reviewer identity — authorization-loss event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-105`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-106: reviewer identity — format-drift event

**Injected condition:** the same reviewer performs both initial reviews by default. Change the source representation while retaining the old filename and require schema version mismatch detection.
**Discriminating observation:** audit reviewer IDs per segment and assignment rules. For this reviewer identity — format-drift event, the check occurs before release from the current state.
**Required recovery:** reassign a calibration sample to distinct qualified reviewers. This recovery is required specifically for the reviewer identity — format-drift event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-106`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-107: reviewer identity — partial-write event

**Injected condition:** the same reviewer performs both initial reviews by default. Interrupt persistence between waveform and metadata writes and check that no half-record becomes eligible.
**Discriminating observation:** audit reviewer IDs per segment and assignment rules. For this reviewer identity — partial-write event, the check occurs before release from the current state.
**Required recovery:** reassign a calibration sample to distinct qualified reviewers. This recovery is required specifically for the reviewer identity — partial-write event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-107`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-108: reviewer identity — clock-skew event

**Injected condition:** the same reviewer performs both initial reviews by default. Shift one source clock by five minutes and require a time-boundary warning before feature assembly.
**Discriminating observation:** audit reviewer IDs per segment and assignment rules. For this reviewer identity — clock-skew event, the check occurs before release from the current state.
**Required recovery:** reassign a calibration sample to distinct qualified reviewers. This recovery is required specifically for the reviewer identity — clock-skew event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-108`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-109: reviewer identity — manual-override event

**Injected condition:** the same reviewer performs both initial reviews by default. Request an undocumented reviewer override and require rationale, actor, timestamp, and a new versioned disposition.
**Discriminating observation:** audit reviewer IDs per segment and assignment rules. For this reviewer identity — manual-override event, the check occurs before release from the current state.
**Required recovery:** reassign a calibration sample to distinct qualified reviewers. This recovery is required specifically for the reviewer identity — manual-override event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-109`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-110: reviewer identity — recovery event

**Injected condition:** the same reviewer performs both initial reviews by default. Restore the service after a controlled stop and require reconciliation of every in-flight record before new intake.
**Discriminating observation:** audit reviewer IDs per segment and assignment rules. For this reviewer identity — recovery event, the check occurs before release from the current state.
**Required recovery:** reassign a calibration sample to distinct qualified reviewers. This recovery is required specifically for the reviewer identity — recovery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-110`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-111: threshold record — missing-field event

**Injected condition:** a threshold value is changed in a presentation deck only. Remove the required field and require a terminal `REJECTED` disposition rather than a default value.
**Discriminating observation:** compare report value, score-run manifest, and signed threshold ledger. For this threshold record — missing-field event, the check occurs before release from the current state.
**Required recovery:** withdraw the report until one immutable threshold record is reconciled. This recovery is required specifically for the threshold record — missing-field event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-111`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-112: threshold record — out-of-range event

**Injected condition:** a threshold value is changed in a presentation deck only. Insert a value outside the schema range and verify that the rejection reason preserves the original field name and unit.
**Discriminating observation:** compare report value, score-run manifest, and signed threshold ledger. For this threshold record — out-of-range event, the check occurs before release from the current state.
**Required recovery:** withdraw the report until one immutable threshold record is reconciled. This recovery is required specifically for the threshold record — out-of-range event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-112`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-113: threshold record — duplicate-delivery event

**Injected condition:** a threshold value is changed in a presentation deck only. Deliver the same source packet twice and confirm that idempotency prevents a second candidate count.
**Discriminating observation:** compare report value, score-run manifest, and signed threshold ledger. For this threshold record — duplicate-delivery event, the check occurs before release from the current state.
**Required recovery:** withdraw the report until one immutable threshold record is reconciled. This recovery is required specifically for the threshold record — duplicate-delivery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-113`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-114: threshold record — late-arrival event

**Injected condition:** a threshold value is changed in a presentation deck only. Deliver a valid-looking update after snapshot lock and verify that it is stored as a new version, not merged into the run.
**Discriminating observation:** compare report value, score-run manifest, and signed threshold ledger. For this threshold record — late-arrival event, the check occurs before release from the current state.
**Required recovery:** withdraw the report until one immutable threshold record is reconciled. This recovery is required specifically for the threshold record — late-arrival event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-114`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-115: threshold record — authorization-loss event

**Injected condition:** a threshold value is changed in a presentation deck only. Revoke access after snapshot registration and verify that downstream steps stop without deleting the audit trail.
**Discriminating observation:** compare report value, score-run manifest, and signed threshold ledger. For this threshold record — authorization-loss event, the check occurs before release from the current state.
**Required recovery:** withdraw the report until one immutable threshold record is reconciled. This recovery is required specifically for the threshold record — authorization-loss event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-115`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-116: threshold record — format-drift event

**Injected condition:** a threshold value is changed in a presentation deck only. Change the source representation while retaining the old filename and require schema version mismatch detection.
**Discriminating observation:** compare report value, score-run manifest, and signed threshold ledger. For this threshold record — format-drift event, the check occurs before release from the current state.
**Required recovery:** withdraw the report until one immutable threshold record is reconciled. This recovery is required specifically for the threshold record — format-drift event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-116`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-117: threshold record — partial-write event

**Injected condition:** a threshold value is changed in a presentation deck only. Interrupt persistence between waveform and metadata writes and check that no half-record becomes eligible.
**Discriminating observation:** compare report value, score-run manifest, and signed threshold ledger. For this threshold record — partial-write event, the check occurs before release from the current state.
**Required recovery:** withdraw the report until one immutable threshold record is reconciled. This recovery is required specifically for the threshold record — partial-write event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-117`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-118: threshold record — clock-skew event

**Injected condition:** a threshold value is changed in a presentation deck only. Shift one source clock by five minutes and require a time-boundary warning before feature assembly.
**Discriminating observation:** compare report value, score-run manifest, and signed threshold ledger. For this threshold record — clock-skew event, the check occurs before release from the current state.
**Required recovery:** withdraw the report until one immutable threshold record is reconciled. This recovery is required specifically for the threshold record — clock-skew event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-118`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-119: threshold record — manual-override event

**Injected condition:** a threshold value is changed in a presentation deck only. Request an undocumented reviewer override and require rationale, actor, timestamp, and a new versioned disposition.
**Discriminating observation:** compare report value, score-run manifest, and signed threshold ledger. For this threshold record — manual-override event, the check occurs before release from the current state.
**Required recovery:** withdraw the report until one immutable threshold record is reconciled. This recovery is required specifically for the threshold record — manual-override event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-119`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-120: threshold record — recovery event

**Injected condition:** a threshold value is changed in a presentation deck only. Restore the service after a controlled stop and require reconciliation of every in-flight record before new intake.
**Discriminating observation:** compare report value, score-run manifest, and signed threshold ledger. For this threshold record — recovery event, the check occurs before release from the current state.
**Required recovery:** withdraw the report until one immutable threshold record is reconciled. This recovery is required specifically for the threshold record — recovery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-120`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-121: queue state — missing-field event

**Injected condition:** an indeterminate item has no assigned owner. Remove the required field and require a terminal `REJECTED` disposition rather than a default value.
**Discriminating observation:** query items lacking owner, disposition deadline, or reason code. For this queue state — missing-field event, the check occurs before release from the current state.
**Required recovery:** assign an accountable queue owner and report age before resuming intake. This recovery is required specifically for the queue state — missing-field event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-121`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-122: queue state — out-of-range event

**Injected condition:** an indeterminate item has no assigned owner. Insert a value outside the schema range and verify that the rejection reason preserves the original field name and unit.
**Discriminating observation:** query items lacking owner, disposition deadline, or reason code. For this queue state — out-of-range event, the check occurs before release from the current state.
**Required recovery:** assign an accountable queue owner and report age before resuming intake. This recovery is required specifically for the queue state — out-of-range event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-122`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-123: queue state — duplicate-delivery event

**Injected condition:** an indeterminate item has no assigned owner. Deliver the same source packet twice and confirm that idempotency prevents a second candidate count.
**Discriminating observation:** query items lacking owner, disposition deadline, or reason code. For this queue state — duplicate-delivery event, the check occurs before release from the current state.
**Required recovery:** assign an accountable queue owner and report age before resuming intake. This recovery is required specifically for the queue state — duplicate-delivery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-123`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-124: queue state — late-arrival event

**Injected condition:** an indeterminate item has no assigned owner. Deliver a valid-looking update after snapshot lock and verify that it is stored as a new version, not merged into the run.
**Discriminating observation:** query items lacking owner, disposition deadline, or reason code. For this queue state — late-arrival event, the check occurs before release from the current state.
**Required recovery:** assign an accountable queue owner and report age before resuming intake. This recovery is required specifically for the queue state — late-arrival event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-124`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-125: queue state — authorization-loss event

**Injected condition:** an indeterminate item has no assigned owner. Revoke access after snapshot registration and verify that downstream steps stop without deleting the audit trail.
**Discriminating observation:** query items lacking owner, disposition deadline, or reason code. For this queue state — authorization-loss event, the check occurs before release from the current state.
**Required recovery:** assign an accountable queue owner and report age before resuming intake. This recovery is required specifically for the queue state — authorization-loss event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-125`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-126: queue state — format-drift event

**Injected condition:** an indeterminate item has no assigned owner. Change the source representation while retaining the old filename and require schema version mismatch detection.
**Discriminating observation:** query items lacking owner, disposition deadline, or reason code. For this queue state — format-drift event, the check occurs before release from the current state.
**Required recovery:** assign an accountable queue owner and report age before resuming intake. This recovery is required specifically for the queue state — format-drift event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-126`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-127: queue state — partial-write event

**Injected condition:** an indeterminate item has no assigned owner. Interrupt persistence between waveform and metadata writes and check that no half-record becomes eligible.
**Discriminating observation:** query items lacking owner, disposition deadline, or reason code. For this queue state — partial-write event, the check occurs before release from the current state.
**Required recovery:** assign an accountable queue owner and report age before resuming intake. This recovery is required specifically for the queue state — partial-write event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-127`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-128: queue state — clock-skew event

**Injected condition:** an indeterminate item has no assigned owner. Shift one source clock by five minutes and require a time-boundary warning before feature assembly.
**Discriminating observation:** query items lacking owner, disposition deadline, or reason code. For this queue state — clock-skew event, the check occurs before release from the current state.
**Required recovery:** assign an accountable queue owner and report age before resuming intake. This recovery is required specifically for the queue state — clock-skew event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-128`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-129: queue state — manual-override event

**Injected condition:** an indeterminate item has no assigned owner. Request an undocumented reviewer override and require rationale, actor, timestamp, and a new versioned disposition.
**Discriminating observation:** query items lacking owner, disposition deadline, or reason code. For this queue state — manual-override event, the check occurs before release from the current state.
**Required recovery:** assign an accountable queue owner and report age before resuming intake. This recovery is required specifically for the queue state — manual-override event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-129`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

#### FI-130: queue state — recovery event

**Injected condition:** an indeterminate item has no assigned owner. Restore the service after a controlled stop and require reconciliation of every in-flight record before new intake.
**Discriminating observation:** query items lacking owner, disposition deadline, or reason code. For this queue state — recovery event, the check occurs before release from the current state.
**Required recovery:** assign an accountable queue owner and report age before resuming intake. This recovery is required specifically for the queue state — recovery event; the event must remain visible in the validation ledger.
**Pass criterion:** The ledger records `FI-130`, the pre-state, terminal state, responsible role, and an auditable reason; no score from the affected record enters aggregate evaluation.

## 13. Transferable Lesson

A biomedical validation plan is an owned state machine and evidence contract before it is a model comparison. The transferable mechanism is to preserve candidate denominators, time-valid inputs, independent reference adjudication, locked evaluation, and a human-owned recovery route. The counterexample is a polished score generated from later labels or silently excluded poor signals: it may be numerically impressive yet cannot answer the intended decision.

## 14. Connections & Cross-References

Related local theory includes Clinical Decision Support Evidence and Handoff, Biomedical Signal Pipeline and Validation, and Human Factors Alert Design and Recovery. These links support method vocabulary only; they are not evidence of this illustrative intervention.

### Interview pack


### Interview 1: Why is this not a diagnostic case?

**Answer:** The output is an offline validation score or routing disposition, with no authorized care action. A diagnostic claim needs an intended-use definition, evidence, governance, and validation not present here.

**Reasoning:** The output is an offline validation score or routing disposition, with no authorized care action. A diagnostic claim needs an intended-use definition, evidence, governance, and validation not present here. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** Ask whether the UI can change care; if yes, stop and redefine scope.

**Acceptable alternatives:** For “Why is this not a diagnostic case?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “Why is this not a diagnostic case?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “Why is this not a diagnostic case?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “Why is this not a diagnostic case?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 2: Why retain excluded segments in the denominator?

**Answer:** They quantify intake loss and can reveal source-specific failure. Deleting them makes the eligible sample look healthier than the operational stream.

**Reasoning:** They quantify intake loss and can reveal source-specific failure. Deleting them makes the eligible sample look healthier than the operational stream. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** Reconcile candidate count to one terminal disposition.

**Acceptable alternatives:** For “Why retain excluded segments in the denominator?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “Why retain excluded segments in the denominator?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “Why retain excluded segments in the denominator?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “Why retain excluded segments in the denominator?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 3: Compute sensitivity for TP=144 and FN=36.

**Answer:** Sensitivity is 144/(144+36)=0.80. It is conditional on the stated reference population and threshold.

**Reasoning:** Sensitivity is 144/(144+36)=0.80. It is conditional on the stated reference population and threshold. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** A claim that this is population performance is a red flag.

**Acceptable alternatives:** For “Compute sensitivity for TP=144 and FN=36.”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “Compute sensitivity for TP=144 and FN=36.”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “Compute sensitivity for TP=144 and FN=36.”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “Compute sensitivity for TP=144 and FN=36.” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 4: What changes if FN doubles to 72 with TP fixed?

**Answer:** Sensitivity becomes 144/216=0.667. The planned threshold decision is materially less attractive and requires review.

**Reasoning:** Sensitivity becomes 144/216=0.667. The planned threshold decision is materially less attractive and requires review. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** Do not hold the denominator at 180 after FN changes.

**Acceptable alternatives:** For “What changes if FN doubles to 72 with TP fixed?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “What changes if FN doubles to 72 with TP fixed?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “What changes if FN doubles to 72 with TP fixed?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “What changes if FN doubles to 72 with TP fixed?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 5: Why must split grouping use subject identity when available?

**Answer:** Related segments may share morphology and acquisition artifacts. Splitting them across partitions can cause memorization and optimistic evaluation.

**Reasoning:** Related segments may share morphology and acquisition artifacts. Splitting them across partitions can cause memorization and optimistic evaluation. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** If identity is unavailable, record the residual leakage risk.

**Acceptable alternatives:** For “Why must split grouping use subject identity when available?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “Why must split grouping use subject identity when available?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “Why must split grouping use subject identity when available?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “Why must split grouping use subject identity when available?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 6: What is the wrong way to use an adjudicated label?

**Answer:** Using it as a feature or choosing preprocessing after inspecting locked-test labels leaks future evaluation information.

**Reasoning:** Using it as a feature or choosing preprocessing after inspecting locked-test labels leaks future evaluation information. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** Keep label attachment and feature manifests separately auditable.

**Acceptable alternatives:** For “What is the wrong way to use an adjudicated label?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “What is the wrong way to use an adjudicated label?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “What is the wrong way to use an adjudicated label?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “What is the wrong way to use an adjudicated label?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 7: How many staffed hours do 360 reviews require at three minutes each?

**Answer:** 360 times 3 equals 1,080 staffed minutes, or 18 staffed hours. Calendar elapsed time depends on staffing concurrency.

**Reasoning:** 360 times 3 equals 1,080 staffed minutes, or 18 staffed hours. Calendar elapsed time depends on staffing concurrency. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** Do not call 18 hours a turnaround guarantee.

**Acceptable alternatives:** For “How many staffed hours do 360 reviews require at three minutes each?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “How many staffed hours do 360 reviews require at three minutes each?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “How many staffed hours do 360 reviews require at three minutes each?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “How many staffed hours do 360 reviews require at three minutes each?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 8: What should happen when lead mapping is unknown?

**Answer:** Route to a named exclusion state and investigate mapping. Guessing order can create invalid features and unverifiable labels.

**Reasoning:** Route to a named exclusion state and investigate mapping. Guessing order can create invalid features and unverifiable labels. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** Verification is a corrected source contract and versioned remapping.

**Acceptable alternatives:** For “What should happen when lead mapping is unknown?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “What should happen when lead mapping is unknown?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “What should happen when lead mapping is unknown?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “What should happen when lead mapping is unknown?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 9: Why is reviewer acceptance not ground truth?

**Answer:** A reviewer may accept an output for workflow reasons, and agreement can be biased or incomplete. The protocol needs blinded rubric-based adjudication.

**Reasoning:** A reviewer may accept an output for workflow reasons, and agreement can be biased or incomplete. The protocol needs blinded rubric-based adjudication. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** Measure and describe disagreement rather than converting it to a positive label.

**Acceptable alternatives:** For “Why is reviewer acceptance not ground truth?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “Why is reviewer acceptance not ground truth?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “Why is reviewer acceptance not ground truth?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “Why is reviewer acceptance not ground truth?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 10: When does a rule-only quality gate win?

**Answer:** It wins as a transparent intake guard when the task label or reference standard is unavailable. It does not establish model discrimination.

**Reasoning:** It wins as a transparent intake guard when the task label or reference standard is unavailable. It does not establish model discrimination. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** Use it to constrain, not replace, the evaluation question.

**Acceptable alternatives:** For “When does a rule-only quality gate win?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “When does a rule-only quality gate win?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “When does a rule-only quality gate win?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “When does a rule-only quality gate win?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 11: What ambiguity must the validation lead resolve before testing?

**Answer:** The target label, eligible population, reference provenance, and error consequence must be specified. Without them, sensitivity has no stable meaning.

**Reasoning:** The target label, eligible population, reference provenance, and error consequence must be specified. Without them, sensitivity has no stable meaning. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** Escalate rather than tune a threshold around ambiguity.

**Acceptable alternatives:** For “What ambiguity must the validation lead resolve before testing?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “What ambiguity must the validation lead resolve before testing?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “What ambiguity must the validation lead resolve before testing?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “What ambiguity must the validation lead resolve before testing?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 12: What is an acceptable alternative to the proposed classifier?

**Answer:** Retain manual double review with a structured rubric and measure its workload and disagreement first. That can be the safer next action.

**Reasoning:** Retain manual double review with a structured rubric and measure its workload and disagreement first. That can be the safer next action. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** Calling manual work a failure without a comparator is a red flag.

**Acceptable alternatives:** For “What is an acceptable alternative to the proposed classifier?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “What is an acceptable alternative to the proposed classifier?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “What is an acceptable alternative to the proposed classifier?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “What is an acceptable alternative to the proposed classifier?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 13: How should a changed sampling rate be handled?

**Answer:** Treat it as a source/version change; test compatibility under a protocol amendment or exclude it. Do not silently resample without documented validation.

**Reasoning:** Treat it as a source/version change; test compatibility under a protocol amendment or exclude it. Do not silently resample without documented validation. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** Compare schema, filter, and performance strata only after data is available.

**Acceptable alternatives:** For “How should a changed sampling rate be handled?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “How should a changed sampling rate be handled?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “How should a changed sampling rate be handled?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “How should a changed sampling rate be handled?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 14: What evidence would permit a stronger conclusion?

**Answer:** Authorized source records, a predeclared task/reference, blinded adjudicated outcomes, locked evaluation artifacts, and review of limitations. Even then, scope stays bounded.

**Reasoning:** Authorized source records, a predeclared task/reference, blinded adjudicated outcomes, locked evaluation artifacts, and review of limitations. Even then, scope stays bounded. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** A slide deck or aggregate score alone is insufficient.

**Acceptable alternatives:** For “What evidence would permit a stronger conclusion?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “What evidence would permit a stronger conclusion?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “What evidence would permit a stronger conclusion?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “What evidence would permit a stronger conclusion?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

### Interview 15: What makes a human fallback real?

**Answer:** A named owner, queue state, staffing/capacity visibility, reason codes, and a terminal disposition. A generic statement that a human is in the loop is not enough.

**Reasoning:** A named owner, queue state, staffing/capacity visibility, reason codes, and a terminal disposition. A generic statement that a human is in the loop is not enough. The conclusion follows only within the stated illustrative protocol and does not establish deployment performance.

**Follow-up:** Test that every abstention enters an owned queue.

**Acceptable alternatives:** For “What makes a human fallback real?”, an answer may defer the study, keep manual review, or choose a narrower protocol only if it preserves the stated decision boundary.

**Red flags:** For “What makes a human fallback real?”, flag any answer that converts incomplete evidence into a diagnosis, hides a denominator, or omits the named verification.

**Scoring:** For “What makes a human fallback real?”, full credit states the mechanism, its constraint, and the operational consequence; partial credit supplies an unqualified slogan or arithmetic result.

**Answer support:** “What makes a human fallback real?” is supported by the relevant contract and worked state in Sections 4–12; its numeric values, if any, are illustrative rather than observed.

## 15. Quick Reference Card

| Field | Card |
|---|---|
| Decision | Approve only a bounded offline validation protocol, not diagnosis or deployment. |
| Owner | Biomedical validation lead; quality, data, and system owners retain separate authority. |
| Input | Time-bounded waveform, acquisition metadata, permission state, and later blinded reference packet. |
| Output | Quality disposition, indeterminate route, or offline validation score with reason. |
| Baseline | Manual double review and an explicit disposition ledger. |
| Critical invariant | Every candidate segment has one terminal disposition; locked test data does not tune the method. |
| Worked check | 2,400-120-80-40=2,160 eligible; 360×3 min=18 staffed hours. |
| Boundary | Illustrative arithmetic only; no clinical, legal, device, workflow, or deployment conclusion. |
| Evidence/stage/review | illustrative / proposed / draft; source and publication review remain OPEN. |
