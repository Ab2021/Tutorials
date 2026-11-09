---
case_id: CASE-01-01-004
case_title: AML alert prioritization at fixed investigator capacity
case_version: "1.0"
created: 2026-11-09
last_reviewed: null
status: proposed
content_status: draft
case_type: predictive-ml
domain: Anti-money-laundering transaction monitoring
primary_pattern: evaluation-mismatch
secondary_patterns: [label-delay, human-in-the-loop]
evidence_level: not-measured
source_basis: illustrative
as_of_date: 2026-11-09
owner_role: Financial-crime operations owner
source_index: INDEX.md
review_trigger: Applicable policy, typology, alert definition, capacity, data lineage, or escalation-process change
---
# Case study: AML alert prioritization at fixed investigator capacity

## 1. The One-Line Framing
Given an eligible monitoring alert with only data available at alert time, prioritize investigator review for the financial-crime operations owner to improve documented investigative value at a fixed review capacity, compared with the existing scenario/rule queue, subject to approved policy, audit, customer-impact, and data-use constraints.

This is an **ILLUSTRATIVE** teaching design, not an account of a financial institution. It asserts no suspicious activity, legal conclusion, filing decision, customer action, deployment, or measured outcome. An alert is a hypothesis; neither a score nor an investigator referral establishes criminal conduct.

**Takeaway:** prioritize review evidence, never automate a legal conclusion from a monitoring score.

## 2. Context & Background
A monitoring system produces candidate alerts from transactions and account relationships. Investigators review a limited queue, assemble permitted evidence, document disposition, and follow the organization’s approved escalation and reporting process. The operations leader owns workload and quality; compliance/legal own applicable procedures; data/platform owners own lineage; model-risk owners own method challenge.

The baseline is the versioned scenario/rule queue. It is retained as comparator, control, and fallback. The proposed scope is triage of already eligible alerts, not transaction blocking, customer de-risking, sanctions screening, or filing automation. Those have distinct actions, evidence standards, and governance.

A queue can seem effective simply because investigators work the alerts it sends. Dispositions also mature late and may reflect investigation effort, policy changes, or unavailable evidence. Therefore an offline label may identify prior-workflow patterns rather than incremental investigator value.

**Takeaway:** start from the investigator’s constrained decision and preserve the boundary between alert, review, and conclusion.

## 3. The Business Question
| Contract | Illustrative definition |
|---|---|
| Question | Which eligible alerts should enter the next 100 investigator-review slots? |
| Owner | Financial-crime operations leader, under approved compliance oversight. |
| Actions | Existing rules priority; proposed ranked priority; random audit slots; defer to next queue segment. |
| Objective | Documented high-value investigative outcomes per review hour, with quality and timeliness guardrails. |
| Constraints | No automatic customer action; 100 slots; approved data and retention; audit trail; applicable requirements determined by owner. |
| Population | Alerts generated in one proposed monitoring workflow with fixed eligibility. |
| Excluded | Determination of suspiciousness, guilt, filing decision, or universal regulatory applicability. |

The primary proposed metric is mature documented-review yield: qualifying investigator outcomes divided by assigned alerts, alongside quality review, rework, timeliness, and random-audit coverage. It is a workflow metric, not a prevalence estimate.

**Takeaway:** define an auditable review outcome rather than pretending that an alert has a direct crime label.

## 4. Assumptions & Constraints
| ID | Assumption/constraint | Check | Fallback if false |
|---|---|---|---|
| A-01 | Alert ID and event-time snapshot are immutable. | replay/version audit | use baseline queue |
| A-02 | Alert disposition taxonomy has maturity states. | quality review + delay curves | retain unresolved state |
| A-03 | Investigators have 100 usable slots in stated horizon. | staffing/time study | reduce K or expose backlog |
| A-04 | Candidate relationship features are available at alert cutoff. | timestamp lineage audit | exclude feature |
| A-05 | Random audit slots are permitted. | compliance approval | limit causal claims |
| C-01 | Policy prohibits score-only customer action. | UI/policy acceptance test | disable action integration |
| C-02 | Reason codes and overrides are retained. | sampled review | revert to rules-only ordering |

All claims above are proposed controls. Capacity and label maturity are most likely to reverse selection.

**Takeaway:** unverified governance or label assumptions are not harmless implementation details.

## 5. Data, Tools & Method
### 5a. Data Specification
The unit is one generated alert keyed by alert ID, monitored entity token, scenario/version, alert generation time, and snapshot version. Candidate inputs include alert scenario, transaction aggregate values, historical alert count, relationship/beneficiary diversity, source freshness, and missingness indicators. Candidate does not mean permitted, reliable, or predictive.

Record event time, ingestion time, availability time, source owner, join cardinality, and permitted use for each feature. Later investigation notes, final disposition, externally obtained information, and relationship links discovered after alert generation cannot be backfilled into the original score. Graph-derived features require temporal edge snapshots; a later connection cannot make an earlier alert look prescient.

Outcomes are documented as unresolved, closed after review, quality-confirmed escalation, or other approved taxonomy values. Their exact meaning must be defined locally; this case does not supply a regulatory definition. Preserve review start/end, reviewer, quality sample, and revision. Never mark unreviewed alerts as negative.

### 5b. Tools & Stack
The proposed design uses a versioned alert snapshot, approved feature builder, ranker, capacity policy, durable queue record, investigator workbench, quality-review workflow, and monitoring/replay store. No technology, vendor, source data, or model version was inspected.

### 5c. Method Summary
Alert snapshot -> permission/schema/availability validation -> point-in-time aggregation -> calibrated ranking where justified -> top-K plus random-audit allocation -> durable queue order/reasons -> human review and quality feedback.

| Candidate | Strength | Limitation | Status |
|---|---|---|---|
| Scenario order | transparent and mandated logic | may not allocate scarce review optimally | baseline |
| Random audit sample | observes some lower-priority outcomes | uses capacity | proposed control |
| Ranker plus top-K | can prioritize a declared outcome | selected labels and drift | proposed |
| Graph score | may capture relationship structure | temporal leakage and proxy risk | considered only |

The ranker is not allowed to replace required scenario controls. Rank scores must not be interpreted as probability of criminal conduct unless an approved, exactly defined, calibrated target supports that interpretation.

### 5d. Provenance, quality, exclusions and leakage controls
Store raw-alert snapshot ID, scenario configuration, feature query revision, availability rule, entity-resolution version, transformation hash, and decision time. Detect duplicate alert IDs, entity merges, late transactions, counter resets, stale aggregates, and relationship snapshots built after cutoff. Fit preprocessing and calibration within each temporal development fold; final forward period is untouched until selection is frozen.

### 5e. Reproducibility ledger
| ID | Source/artifact | Purpose | Boundary |
|---|---|---|---|
| SRC-01 | [FATF Recommendations](https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html) | public reference context | not a jurisdiction-specific determination |
| SRC-02 | Proposed alert data contract | cutoff and lineage | illustrative, unavailable |
| RUN-01 | Proposed replay configuration | compare policies | not executed |
| POL-01 | Proposed capacity/quality policy | top-K and audit allocation | not approved |

**Takeaway:** preserve alert-time snapshots and investigator-quality evidence; neither can be recreated from a current warehouse table.

## 6. Analysis Walkthrough
No replay was executed. This is **ILLUSTRATIVE** arithmetic for a capacity-constrained review policy. Assume 10,000 eligible alerts, 100 mature quality-confirmed outcomes under a specified taxonomy, 200 review slots, and 60 confirmed outcomes in the top 200.

```text
FORMULA [M-01]: precision@K = TP_K / K
WHERE: TP_K [alerts] = mature confirmed outcomes among reviewed top-K; K [alerts] = review slots.
CONSTRAINT: common alert population, stable outcome taxonomy, K>0; selected labels remain a limitation.
NUMERICAL EXAMPLE: TP_200=60; K=200; precision@200=60/200=30%.
INTERPRETATION: 30 constructed confirmed outcomes per 100 reviewed alerts; this is not a crime prevalence estimate.
```

```text
FORMULA [M-02]: recall@K = TP_K / P
WHERE: P [alerts] = all mature confirmed outcomes in the defined eligible cohort.
CONSTRAINT: P excludes unresolved alerts only if the evaluation states that restriction; P>0.
NUMERICAL EXAMPLE: 60/100=60%.
INTERPRETATION: 40 constructed mature outcomes lie outside the 200 slots.
```

Suppose each review consumes 75 minutes including documentation and quality handoff. Workload is 200*75=15,000 minutes=250 investigator-hours. At eight productive hours per investigator-day, this represents 31.25 investigator-days, not a staffing guarantee because work time varies.

```text
FORMULA [M-03]: H = K * m / 60
WHERE: H [hours] = review workload; K [alerts]; m [minutes/alert].
CONSTRAINT: constant average work; excludes wait, rework, and quality sampling.
NUMERICAL EXAMPLE: 200*75/60=250 hours.
INTERPRETATION: capacity K must be reconciled to actual staffed hours, not selected because it flatters precision.
```

Independent check: 200*75=15,000; 15,000/60=250. Sensitivity: if mean review is 100 minutes, workload is 333.33 hours; if capacity is 150 slots, the original 30% precision scenario yields 45 expected positive outcomes only if ranking holds, which must be verified. This explicitly fails if alerts are stale or the outcome taxonomy changes.

**Takeaway:** capacity, outcome maturity, and investigation quality all belong in the arithmetic.

## 7. The Finding
No evidence supports changing the alert queue. A proposed top-K prioritization may be evaluated as a human-review aid only after the organization defines permitted data, mature dispositions, quality standards, capacity, and a rules comparator. The 30% precision@200, 60% recall@200, and 250-hour figures are hand-derived illustrative scenarios, not performance or staffing measurements. The correct immediate action is an evidence and governance gate.

**Takeaway:** an alert-ranking proposal is not a conclusion about financial crime or regulatory effectiveness.

## 8. System/Solution Design
```text
[Versioned alert snapshot]
  -> [permission + schema + cutoff validation] --fail--> [baseline ordering + reason]
  -> [point-in-time features] -> [rank/policy: mandatory rules, top-K, audit slots]
  -> [durable queue record: alert, snapshot, policy, reason, override]
  -> [investigator workbench] -> [quality review / disposition]
  -> [maturity ledger + replay/monitoring]
```

The queue policy has a deterministic tie-breaker and records whether rank, mandatory routing, random audit, or manual override placed the alert. A durable record prevents a retry or replay from creating duplicate active work. Entity-resolution changes create a new, versioned relationship snapshot; they do not silently rewrite a decision record.

When source freshness, policy, or scoring is unavailable, queue using approved baseline ordering and record fallback. Do not drop an alert or make a customer action. Reconcile alert creation, queue acceptance, workbench assignment, closure, quality sample, and disposition daily. Recovery includes pausing model ordering, preserving baseline service, replaying from snapshot, and documenting any re-ordered work.

**Takeaway:** queue correctness and audit lineage are system requirements, not post-hoc reporting.

## 9. Evaluation & Validation
| Metric | Definition | Comparator | Guardrail | Status |
|---|---|---|---|---|
| precision@K | confirmed outcomes / reviewed K | scenario order at same K | quality error | proposed |
| review yield/hour | confirmed outcomes / actual review hours | baseline | backlog age | proposed |
| outcome maturity | resolved outcomes / eligible assigned alerts | cohort age | unresolved rate | proposed |
| queue correctness | one active record per alert version | baseline | duplicate/missing work | proposed |
| slice monitoring | action/outcome/quality by approved slice | baseline | privacy + uncertainty | proposed |

Use time-forward alert snapshots and keep entity-linked alerts within a declared evaluation boundary. Compare policies at same capacity and required scenario constraints. Because investigator attention changes outcomes, retain a permitted random-audit allocation and quality-review sample. Shadow mode can validate data lineage and latency; it cannot demonstrate investigation value or policy effect.

Stress tests change review minutes, K, maturity threshold, entity-resolution version, late-data rate, and quality definition. If selection loses advantage when unresolved outcomes are kept separate or when capacity uses real hours, do not claim benefit.

**Takeaway:** compare complete queues under equal constraints and disclose what review selection leaves unobserved.

## 10. Action Taken
No live or pilot action occurred. Proposed next steps are to document outcome taxonomy and data permissions, measure baseline queue/capacity, freeze alert snapshots, validate replay and idempotency, obtain compliance/model-risk approval, and then decide whether a controlled evaluation is allowed.

**Takeaway:** do not upgrade a planned test into a deployment narrative.

## 11. Outcome & Results
Not measured. There is no asserted institution, suspicious-activity result, review yield, staffing observation, or policy outcome. The numerical values are illustrative calculations whose assumptions are stated in Section 6.

**Takeaway:** evidence maturity remains visible in the result section.

## 12. Failure Modes & What Went Wrong
### F-01 — Future relationship leakage
**Symptom:** graph features sharply improve historical alert yield. **Competing causes:** genuine timely relationship signal; later entity merge; retrospective edge construction. **Diagnostic:** replay edges by availability time and entity-resolution version; expected signal is post-alert edge availability. **Fix:** temporal snapshots and feature exclusion. **Risk:** lower apparent yield. **Verification:** sampled feature lineage precedes alert generation.

### F-02 — Disposition treated as ground truth
**Symptom:** closed or unreviewed alerts are labeled negative. **Competing causes:** true absence of evidence; unresolved file; investigator capacity; taxonomy change. **Diagnostic:** plot disposition/maturity by priority and cohort. **Fix:** retain unresolved state and random audit where permitted. **Risk:** slower feedback. **Verification:** evaluation reports observation/maturity separately. 

### F-03 — Capacity is fictional
**Symptom:** top-K queue ages while policy says slots are available. **Competing causes:** staffing loss; longer cases; duplicates; surge; quality rework. **Diagnostic:** reconcile arrivals, starts, completions, minutes, and active keys. **Fix:** set K from measured usable hours and expose overflow. **Risk:** lower recall. **Verification:** queue-age and reconciliation thresholds pass on replay/operation.

### F-04 — Score changes mandatory routing
**Symptom:** required scenario alerts are demoted or disappear. **Competing causes:** precedence bug; policy version mismatch; schema failure. **Diagnostic:** invariant test: every mandatory alert has an active route and reason. **Fix:** mandatory-rule gate before ranker and policy-version lock. **Risk:** fewer discretionary slots. **Verification:** adversarial test records preserve all mandatory routes.

### F-05 — Entity merge causes duplicate or unfair concentration
**Symptom:** one entity has correlated duplicate alerts after identifier change. **Competing causes:** actual linked behavior; key collision; stale merge; join fan-out. **Diagnostic:** compare join cardinality, merge versions, and raw alert counts. **Fix:** versioned entity mapping, cardinality checks, and queue dedupe. **Risk:** overly aggressive merge hides valid associations. **Verification:** sampled reconciliations preserve raw-source counts and approved linkage.

**Takeaway:** diagnose data lineage, observation process, queue operations, and policy precedence separately.

## 13. Transferable Lesson
Any review-constrained risk queue needs point-in-time evidence, an action owner, capacity-aware policy, selected-label disclosure, quality feedback, and a safe fallback. The pattern transfers to safety reports, moderation, inspections, and claims triage. It does not transfer a conclusion about legal status or permit the same outcome taxonomy in another domain.

**Takeaway:** ranking can allocate attention; it cannot replace the evidence standard of the downstream decision.

## 14. Connections & Cross-References
See [Fraud evaluation, calibration and capacity](../../../02_data_science_method_playbooks/02_fraud_modelling_playbook/Cheat_Sheets/01_Fraud_Evaluation_Calibration_and_Capacity.md) for reusable capacity metrics. The visible original source entry is in `directory.txt` lines 481–486; its contents were unavailable. This case does not infer its implementation.

### 14a. Interview pack
#### Q01 — Is an alert proof of laundering?
**Answer:** no; it is a review hypothesis under a defined monitoring process. **Reasoning:** alert generation, investigator evidence, and conclusion are distinct. **Follow-up:** preserve disposition taxonomy. **Alternative:** call it a prioritization signal. **Red flag:** score as legal finding. **Score:** 0 unsafe; 1 caveat; 2 distinction; 3 governance boundary. **Support:** 1–3.
#### Q02 — Calculate precision@200 and recall@200.
**Answer:** 60/200=30%; 60/100=60%. **Reasoning:** review versus outcome denominators. **Follow-up:** 140 referrals lack mature confirmed outcome and 40 positives are outside K. **Alternative:** counts first. **Red flag:** divide by 10,000. **Score:** 0 wrong; 1 one metric; 2 both; 3 maturity caveat. **Support:** M-01/M-02.
#### Q03 — Calculate workload.
**Answer:** 200*75/60=250 hours. **Reasoning:** 15,000 minutes converted to hours. **Follow-up:** 100 minutes gives 333.33 hours. **Alternative:** use distribution after observing data. **Red flag:** call it 250 people. **Score:** 0 wrong; 1 arithmetic; 2 units; 3 sensitivity. **Support:** M-03.
#### Q04 — Why can a random audit be valuable?
**Answer:** it provides some outcome observation outside selected high-priority alerts. **Reasoning:** historical queue selection biases labels. **Follow-up:** it consumes capacity. **Alternative:** approved independent quality sample. **Red flag:** unreviewed equals negative. **Score:** 0 unsafe; 1 mentions bias; 2 mechanism; 3 tradeoff. **Support:** 5c, 9.
#### Q05 — What leaks in graph features?
**Answer:** edges or entity merges discovered after alert time. **Reasoning:** retrospective graph contains future information. **Follow-up:** use temporal snapshots. **Alternative:** exclude unversioned relationships. **Red flag:** current graph is historical truth. **Score:** 0 wrong; 1 leakage; 2 timing; 3 audit. **Support:** F-01.
#### Q06 — Why retain mandatory rules?
**Answer:** ranked discretionary allocation must not remove required routing. **Reasoning:** controls have precedence. **Follow-up:** invariant-test it. **Alternative:** separate queues. **Red flag:** model replaces all scenarios. **Score:** 0 unsafe; 1 baseline; 2 precedence; 3 verification. **Support:** 5c, F-04.
#### Q07 — What does shadow mode establish?
**Answer:** technical replay/latency compatibility, not policy effectiveness. **Reasoning:** it does not change review assignment. **Follow-up:** controlled evaluation. **Alternative:** no causal claim. **Red flag:** shadow yield proves benefit. **Score:** 0 wrong; 1 caveat; 2 causal mechanism; 3 design. **Support:** 9.
#### Q08 — Queue backlog rises. First diagnosis?
**Answer:** reconcile arrivals, starts, completions, actual minutes, duplicates, and surge. **Reasoning:** age has competing causes. **Follow-up:** expose overflow before increasing K. **Alternative:** queue simulation. **Red flag:** tune score. **Score:** 0 wrong; 1 backlog; 2 signals; 3 remedy. **Support:** F-03.
#### Q09 — What is wrong with a current entity table join?
**Answer:** it can use later merges/relationships unavailable at alert time. **Reasoning:** cutoff violation. **Follow-up:** use versioned mapping. **Alternative:** raw IDs only. **Red flag:** latest is cleanest. **Score:** 0 wrong; 1 notes leak; 2 explains; 3 test. **Support:** 5d.
#### Q10 — What changed constraint halts rollout?
**Answer:** loss of data permission, capacity, or required-quality control. **Reasoning:** policy contract changed. **Follow-up:** baseline fallback. **Alternative:** restrict scope. **Red flag:** continue with missing controls. **Score:** 0 unsafe; 1 pause; 2 fallback; 3 revalidation. **Support:** 4, 10.
#### Q11 — Why report unresolved alerts?
**Answer:** their outcome is unknown and excluding them can inflate claimed yield. **Reasoning:** maturity is cohort-dependent. **Follow-up:** plot delay curves. **Alternative:** conservative sensitivity. **Red flag:** close equals negative. **Score:** 0 unsafe; 1 unknown; 2 selection bias; 3 measure. **Support:** 5a, 9.
#### Q12 — What is the safe fallback?
**Answer:** approved baseline ordering with reason logging and reconciliation, not silent alert loss. **Reasoning:** continuity and audit remain necessary during failure. **Follow-up:** replay snapshot after recovery. **Alternative:** controlled queue pause. **Red flag:** automatic customer action. **Score:** 0 unsafe; 1 baseline; 2 lineage; 3 recovery. **Support:** 8.

## 15. Quick Reference Card
| Field | Value |
|---|---|
| Decision | Prioritize existing eligible alerts for human investigation at fixed capacity. |
| Baseline | Versioned scenario/rule queue and mandatory controls. |
| Formula | precision@K=TP_K/K; recall@K=TP_K/P; workload hours=K*minutes/60. |
| Critical condition | Alert features and graph edges must exist at alert cutoff. |
| Avoid | Treating alert/disposition as proof or unreviewed as negative. |
| Fallback | Approved baseline ordering with durable reason and reconciliation. |
| Evidence/stage | Illustrative; not measured; proposed draft. |
| Review trigger | Policy, typology, lineage, capacity, or process change. |
