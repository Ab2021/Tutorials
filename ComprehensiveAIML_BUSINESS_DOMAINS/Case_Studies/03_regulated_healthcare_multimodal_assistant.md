---
case_id: "CASE-XD-003"
case_title: "Regulated healthcare multimodal assistant"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "Healthcare"
primary_pattern: "human-in-the-loop"
secondary_patterns: ["authorization-scope", "evaluation-mismatch"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "clinical governance lead"
source_index: "INDEX.md"
review_trigger: "Authority, workload, source-access, or validation-evidence change."
---

# Regulated healthcare multimodal assistant

**Evidence declaration:** This is an illustrative design exercise. Every quantity, workflow, record, and outcome is constructed teaching material, not a measurement of a named organization.
**Source inventory:** [directory.txt](../../Templates/directory.txt), [source crosswalk](../00_SOURCE_TOPIC_CROSSWALK.md), and [edge/industry coverage brief](../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md). Visible names establish topics, not proprietary implementation facts.

## 1. The One-Line Framing
> Given **clinician request, authorized patient context, document/image references, and encounter time**, decide **cited draft, abstention, or escalation packet for a licensed clinician** for the **clinical governance lead** to support clinician-supported output without autonomous clinical action, subject to **site-approved access, clinician review, and no diagnosis or order execution**, compared with **manual chart review and templated search**.
- Task category: llm-agentic; primary pattern: `human-in-the-loop`; secondary patterns: `authorization-scope; evaluation-mismatch`.
- Decision frequency is per eligible event; an ineligible or ambiguous input is an explicit abstention.
- The obvious simple framing fails because a prediction or generated answer does not itself confer permission, feasibility, safety, or causal value.
- Success condition: each release is reconstructible from decision-time inputs and a versioned authority boundary.
- The case proposes validation; it does not describe a deployment or realized benefit.

## 2. Context & Background
- Workflow: a request enters with clinician request, authorized patient context, document/image references, and encounter time; the decision is consumed as cited draft, abstention, or escalation packet for a licensed clinician.
- Stakeholders: clinical governance lead, frontline users, data/engineering operators, security/privacy reviewers, and escalation authority.
- Baseline: manual chart review and templated search; it remains the fallback when eligibility, evidence, or dependencies are uncertain.
- Illustrative scale: 1,200 candidate decisions/hour, 3× peak, 30-day evaluation window.
- Timing: only facts available before the decision timestamp may influence the decision.
- Geographic, contractual, and regulatory applicability are unknown; confirm authority before use.
- Known facts are limited to the curriculum brief; no organization-specific feature, volume, or result is implied.
- No prior deployment or failed experiment is asserted.
- Human impact: an error can create rework, safety, privacy, financial, or trust harm.
- The owner needs a decision record, not merely a score.

## 3. The Business Question
- Question: Under the stated workload, which cited draft, abstention, or escalation packet for a licensed clinician may be released, deferred, or rejected?
- Decision owner: clinical governance lead.
- Trigger/deadline: every eligible input; binding limit: site-approved access, clinician review, and no diagnosis or order execution.
- Actions: retain baseline, release bounded recommendation, require review, defer, or reject.
- Value: less avoidable manual work and clearer prioritization; downside: harmful or unauthorized release.
- Objective: useful policy-conforming completed decisions, not raw confidence.
- Hard constraints: decision-time correctness, auditability, bounded latency, safe fallback.
- Scope: one hypothetical workflow; legal, clinical, safety, and commercial authority outside it are excluded.
- Proxy conflict: more automated completion can worsen the real objective by bypassing review.

## 4. Assumptions & Constraints
- A-01 | Input identity and authority are current | data/security | unverified illustrative assumption | impact: wrong recipient or scope | fallback: abstain and re-authenticate | recheck: owner before pilot.
- A-02 | Event timestamps are comparable | data | unverified illustrative assumption | impact: future information contaminates decision | fallback: use conservative cut-off | recheck: owner before pilot.
- A-03 | Downstream action accepts operation key | system | unverified illustrative assumption | impact: retry duplicates an effect | fallback: manual reconciliation | recheck: owner before pilot.
- A-04 | Reviewer can inspect evidence in time | workflow | unverified illustrative assumption | impact: review rubber-stamps | fallback: defer to baseline | recheck: owner before pilot.
- A-05 | Capacity holds during peak | infrastructure | unverified illustrative assumption | impact: tail latency and queues grow | fallback: shed optional work | recheck: owner before pilot.
- A-06 | Outcome labels mature | evaluation | unverified illustrative assumption | impact: offline ranking misleads | fallback: retain action logs | recheck: owner before pilot.
- A-07 | Owner can stop workflow | governance | unverified illustrative assumption | impact: unsafe release persists | fallback: kill switch | recheck: owner before pilot.
- C-01 | Binding limit | site-approved access, clinician review, and no diagnosis or order execution | teaching assumption only | breach rejects automatic release.
- C-02 | Evidence constraint | no proprietary measurements or legal determinations | outcome remains not measured.

## 5. Data, Tools & Method
### 5a. Data Specification
- Unit: one candidate decision keyed by `decision_id`; repeated events retain immutable sequence.
- Volume/window: 864,000 constructed records over 30 days; peak assumption 3,600/hour.
- Inputs: role-scoped notes, document references, image metadata, policies, reviewer adjudications, and audit events; each has source timestamp, schema version, and availability timestamp.
- Output: cited draft, abstention, or escalation packet for a licensed clinician; state is `released`, `review`, `abstain`, `rejected`, or `reconciled`.
- Population: schema and identity-valid inputs; excluded records remain reject/audit events.
- Join: one decision joins one as-of authority snapshot and zero-or-more evidence items; fan-out is explicit.
- Split: frozen time replay; entities do not cross partitions when history would leak.
### 5b. Tools & Stack
- Proposed: schema validator, identity/policy decision point, versioned evidence store, deterministic rules, optional model service, review queue, durable operation ledger, audit sink.
- A model/retriever proposes evidence; it cannot mutate the action ledger directly.
- Configuration bundles schema, policy, prompt/model, threshold, and tool-contract versions.
- No vendor API, pricing, hardware, or deployment behavior is asserted.
### 5c. Method Summary
- Pipeline: validate -> authorize -> collect decision-time evidence -> retrieval, multimodal evidence display, uncertainty routing, and clinician sign-off -> deterministic guardrails -> review/release/abstain -> receipt reconciliation.
- Baseline: manual/fixed rule; explainable but throughput-limited.
- Alternative A: full automatic release; considered, rejected for irreversible or low-evidence cases.
- Alternative B: review every input; considered, but can violate capacity and deadline.
- Proposal: automate only reversible recommendation/evidence preparation, with authority gates before effects.
- Thresholds come from a frozen replay protocol, not desired outcomes.
### 5d. Provenance, Quality, Exclusions and Leakage Controls
- SRC-01: local catalog brief; supports scope only.
- SRC-02: visible source-inventory filenames; support topic provenance only.
- SRC-03: official method references below; support general constraints only.
- Freeze snapshots by `as_of_time`; reject fields available after decision time.
- Record missingness and join cardinality; never impute authority/approval.
- Keep outcome, later adjudication, and post-decision notes outside decision-time view.
- Original source bodies are unavailable; this is an evidence gap.
### 5e. Reproducibility and Method-Selection Ledger
- RUN-01 proposed inputs: `decisions.csv`, `policy_snapshot.json`, `events.ndjson`, `expected_receipts.csv`.
- Use fixed UTC timestamps, deterministic rule order, immutable artifact hashes; record stochastic seed if needed.
- Oracle: compare releases with authorized expected actions and effects by operation key.
- Limit: no real data, deployment, safety validation, or business result is available.
- Ledger retains each rejection and reason code so baseline/abstention choices are reconstructible.

- `SRC-04` [NIST AI Risk Management Framework 1.0](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf), January 2023, accessed 2026-11-09: supports risk-management framing only; it does not validate this illustrative workflow.
- `SRC-05` [Google SRE Workbook: Implementing SLOs](https://sre.google/workbook/implementing-slos/), accessed 2026-11-09: supports user-centered reliability metric method only; it does not establish a target or performance result.
- `SRC-06` [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), accessed 2026-11-09: supports training-time preprocessing/leakage caution only; no model performance is inferred.
- Recheck trigger: source version, domain requirement, or external-interface semantics change before use.

## 6. Analysis Walkthrough
The decisive tradeoff is not score maximization: release requires authority, evidence, and capacity gates.
- Step 1: Normalize input; reject invalid schema or impossible timestamp. Emit `CASE-XD-003-S1` with input hash, version bundle, reason, and state transition.
- Step 2: Resolve exact identity and policy/operating snapshot. Emit `CASE-XD-003-S2` with input hash, version bundle, reason, and state transition.
- Step 3: Build evidence packet with source IDs and freshness. Emit `CASE-XD-003-S3` with input hash, version bundle, reason, and state transition.
- Step 4: Generate or score candidate without granting authority. Emit `CASE-XD-003-S4` with input hash, version bundle, reason, and state transition.
- Step 5: Apply deterministic eligibility, confidence, and reversibility gates. Emit `CASE-XD-003-S5` with input hash, version bundle, reason, and state transition.
- Step 6: Reserve review capacity or abstain before promising review. Emit `CASE-XD-003-S6` with input hash, version bundle, reason, and state transition.
- Step 7: Write durable decision record before external side effect. Emit `CASE-XD-003-S7` with input hash, version bundle, reason, and state transition.
- Step 8: Reconcile returned receipt against same operation key. Emit `CASE-XD-003-S8` with input hash, version bundle, reason, and state transition.

**Checked illustrative throughput calculation**
```text
Validity: constant average rates, empty queue, payload-only estimate; not a latency guarantee.
Derivation: conservation gives backlog change = arrival rate - service rate.
FORMULA [Q-01]: B = max(0, lambda - mu) * t
WHERE: B [decisions], lambda/mu [decisions/s], t [s].
CONSTRAINT: constant rates; retries and abandonment excluded.
NUMERICAL EXAMPLE: lambda=1.20/s, mu=1.00/s, t=600 s; B=(1.20-1.00)*600=120 decisions.
INTERPRETATION: buffer absorbs ten minutes, not sustained overload.
FORMULA [Q-02]: T = B / (mu_r - lambda_r)
WHERE: T [s]; recovery rates [decisions/s].
CONSTRAINT: mu_r > lambda_r.
NUMERICAL EXAMPLE: B=120, mu_r=1.40/s, lambda_r=1.00/s; T=120/0.40=300 s.
INTERPRETATION: five minutes of recovery surplus is required.
```
- Independent check: 0.20/s × 600 s = 120; 120 / 0.40/s = 300 s.
- Sensitivity: recovery arrivals at 1.40/s leave zero surplus; backlog never drains in this model.
- No approach was actually tested; alternatives are considered, not project history.
- Stress 01: arrival rate doubles; preserve deterministic gate, then baseline/abstain and log reason rather than relaxing constraints.
- Stress 02: review capacity halves; preserve deterministic gate, then baseline/abstain and log reason rather than relaxing constraints.
- Stress 03: policy snapshot stale; preserve deterministic gate, then baseline/abstain and log reason rather than relaxing constraints.
- Stress 04: dependency timeout; preserve deterministic gate, then baseline/abstain and log reason rather than relaxing constraints.
- Stress 05: label delayed 30 days; preserve deterministic gate, then baseline/abstain and log reason rather than relaxing constraints.
- Stress 06: payload doubles; preserve deterministic gate, then baseline/abstain and log reason rather than relaxing constraints.
- Stress 07: unknown schema field; preserve deterministic gate, then baseline/abstain and log reason rather than relaxing constraints.
- Stress 08: irreversible action; preserve deterministic gate, then baseline/abstain and log reason rather than relaxing constraints.
- Stress 09: clock skew five minutes; preserve deterministic gate, then baseline/abstain and log reason rather than relaxing constraints.
- Stress 10: tenant revoked; preserve deterministic gate, then baseline/abstain and log reason rather than relaxing constraints.
- Stress 11: confidence rises without evidence; preserve deterministic gate, then baseline/abstain and log reason rather than relaxing constraints.
- Stress 12: replay duplicates; preserve deterministic gate, then baseline/abstain and log reason rather than relaxing constraints.

- **Slice-review load.** Valid only as workload accounting, not clinical efficacy.
- FORMULA [H-01]: H = n * r / 60.
- WHERE `H` review hours, `n` candidate responses [count], `r` minutes/response.
- ILLUSTRATIVE: 240 responses * 3 min / 60 = 12 reviewer-hours.
- Check: 240*3=720 minutes; 720/60=12 hours.
- Sensitivity: a 5-minute review requirement raises load to 20 hours and may require abstention rather than rushed review.

## 7. The Finding
The supported recommendation is a proposed bounded clinical escalation boundary workflow releasing only decision-time-authorized, auditable outputs and otherwise using review, baseline, or abstention. The arithmetic yields a 120-decision backlog and a five-minute drain under stated illustrative rates; it is not a capacity measurement. No effectiveness, safety, financial, clinical, service, cost, or emissions outcome is measured. Missing authority, stale evidence, unavailable review, or an incompatible side-effect contract reverses the recommendation toward no automatic release.

## 8. System/Solution Design
### 8a. Offline / Training / Batch Path
- Ingest versioned input/outcome snapshots into restricted evaluation store.
- Validate schema, availability times, joins, and policy completeness.
- Build replay cohorts without post-decision information.
- Evaluate baseline/rules/optional model on same frozen workload.
- Publish signed bundle of policy, schema, configuration, evaluation report.
- Reject promotion without owner, rollback target, compatibility declaration.
### 8b. Online / Serving / Real-Time Path
```text
[Input + event time + identity] -> [Validate / authorize] -> [Evidence packet]
                                      |                         |
                                      v                         v
                              [Reject + audit]          [Candidate method]
                                                                  |
                                                                  v
[Metrics + trace + receipt] <- [Decision gate + durable ledger] <- [Review reservation]
                                      |
                                      v
                         [Release / abstain / approved operation]
```
Successful path: validate, freeze authority/evidence versions, decide, persist intent, execute once, reconcile receipt.
Failed path: timeout/ambiguous receipt never triggers blind retry; query ledger/downstream, reconcile or queue.
Cold start: baseline or abstain until required snapshots load.
Peak: preserve validation/authority; shed optional generation/ranking first.
### 8c. Key Design Decisions
- DEC-01 | Policy before retrieval/execution | alternative implicit default | evidence design assumption | tradeoff less automation for auditability | revisit contract/workload change.
- DEC-02 | Operation ID before side effect | alternative implicit default | evidence design assumption | tradeoff less automation for auditability | revisit contract/workload change.
- DEC-03 | Bounded review reservation | alternative implicit default | evidence design assumption | tradeoff less automation for auditability | revisit contract/workload change.
- DEC-04 | Version bundle rollback | alternative implicit default | evidence design assumption | tradeoff less automation for auditability | revisit contract/workload change.
- DEC-05 | Abstention valid output | alternative implicit default | evidence design assumption | tradeoff less automation for auditability | revisit contract/workload change.
- DEC-06 | Append-only audit | alternative implicit default | evidence design assumption | tradeoff less automation for auditability | revisit contract/workload change.
### 8d. Architecture Artifacts and Operational Readiness
- Interface: validate schema, identity scope, deadline, idempotency key.
- State: ledger owns transitions; evidence stores immutable snapshots.
- Reliability: useful-decision availability = compliant responses / eligible requests in stated window; proposed target only.
- Resilience: cap retries by deadline/state; circuit-break optional dependency.
- Recovery: restore compatible bundle; replay only absent/safely reconciled receipts.
- Observability: trace/decision/policy/version IDs, reason, queue age, receipt state.
- Deployment: shadow, adversarial replay, bounded canary, compatibility-checked rollback.
- Security: least privilege, scoped credentials, audit access, redaction.
- Economics: measure useful decisions/review minutes; do not infer savings from throughput.
- Human workflow: owner approves, overrides, stops, reconciles; automation cannot self-expand authority.

## 9. Evaluation & Validation
- Primary metrics: supported-statement rate, unsafe-suggestion rate, review minutes, and slice coverage; define numerator, denominator, population, UTC window, comparator, owner.
- Baseline: manual/fixed-rule process under same eligibility rules.
- Pass/fail: zero unauthorized releases in controlled replay; any mismatch fails.
- Guardrails: deadline, duplicate effects, abstention/review rate, stale-version rate, domain safety/privacy.
- Tests: deterministic replay, adversarial authorization, dependency-failure injection, expert review as applicable.
- Sample: constructed records only; no representativeness or domain validity claimed.
- Uncertainty: sensitivity arithmetic, not confidence interval.
- Threat: replay misses operator behavior, interference, bursts.
- Artifact: RUN-01 proposed inputs/expected receipts; status not executed.
- Decision: inconclusive/not executed; supports protocol, not adoption.

## 10. Action Taken
- Recommendation: do not deploy from this document; use as design/replay specification.
- Approval: none recorded; stage proposed.
- Handoff: domain owner, security reviewer, platform operator jointly own pilot.
- Effective scope: none; no user, region, record, model, or device changed.
- Release: proposed shadow replay -> adversarial review -> limited pilot after authority approval.
- Stop: owner disables release, routes baseline, pins compatible bundle, reconciles pending operations.
- Excluded: autonomous irreversible action and realized-benefit claims.

## 11. Outcome & Results
| Metric | Baseline | Treatment | Absolute change | Relative change | Uncertainty | Evidence | Limitation |
|---|---|---|---|---|---|---|---|
| Primary useful outcome | unavailable | not measured | not measured | not defined | unavailable | illustrative design | no execution |
| Unauthorized release | unavailable | not measured | not measured | not defined | unavailable | proposed pass/fail | no replay |
| Queue drain | n/a | 300 s illustrative | not measured | n/a | model assumptions | derived arithmetic | constant rate |
| Review load | unavailable | not measured | not measured | not defined | unavailable | proposed metric | no study |
- Outcome not measured. Owner: clinical governance lead. Proposed window: 30 days after governed pilot. Trigger: compare predeclared metric/guardrails under same eligibility denominator.

## 12. Failure Modes & What Went Wrong
- F-01 | Trigger/symptom: Stale authority.
  - Competing causes: stale cache or clock, dependency, configuration, or operator error.
  - Diagnostic / expected signal: compare snapshot version/time to decision time.
  - Evidence: HYPOTHETICAL teaching failure, not observed incident.
  - Containment: withhold release and re-authorize.
  - Prevention: versioned contract, owned alert, negative-control replay.
  - Verification/limit: reproduce bounded trigger; production behavior unmeasured.
- F-02 | Trigger/symptom: Duplicate effect.
  - Competing causes: ambiguous delivery, dependency, configuration, or operator error.
  - Diagnostic / expected signal: lookup operation key and receipt.
  - Evidence: HYPOTHETICAL teaching failure, not observed incident.
  - Containment: reconcile before retry.
  - Prevention: versioned contract, owned alert, negative-control replay.
  - Verification/limit: reproduce bounded trigger; production behavior unmeasured.
- F-03 | Trigger/symptom: Leakage.
  - Competing causes: join error, dependency, configuration, or operator error.
  - Diagnostic / expected signal: availability-time audit.
  - Evidence: HYPOTHETICAL teaching failure, not observed incident.
  - Containment: remove field and rerun.
  - Prevention: versioned contract, owned alert, negative-control replay.
  - Verification/limit: reproduce bounded trigger; production behavior unmeasured.
- F-04 | Trigger/symptom: Queue overload.
  - Competing causes: capacity/dependency loss, dependency, configuration, or operator error.
  - Diagnostic / expected signal: arrival/service and queue-age dashboard.
  - Evidence: HYPOTHETICAL teaching failure, not observed incident.
  - Containment: shed optional work.
  - Prevention: versioned contract, owned alert, negative-control replay.
  - Verification/limit: reproduce bounded trigger; production behavior unmeasured.
- F-05 | Trigger/symptom: Rubber-stamp review.
  - Competing causes: weak evidence packet, dependency, configuration, or operator error.
  - Diagnostic / expected signal: blind audit/disagreement sample.
  - Evidence: HYPOTHETICAL teaching failure, not observed incident.
  - Containment: improve packet.
  - Prevention: versioned contract, owned alert, negative-control replay.
  - Verification/limit: reproduce bounded trigger; production behavior unmeasured.
- F-06 | Trigger/symptom: Schema drift.
  - Competing causes: producer change, dependency, configuration, or operator error.
  - Diagnostic / expected signal: contract test and unknown-field count.
  - Evidence: HYPOTHETICAL teaching failure, not observed incident.
  - Containment: quarantine version.
  - Prevention: versioned contract, owned alert, negative-control replay.
  - Verification/limit: reproduce bounded trigger; production behavior unmeasured.
- F-07 | Trigger/symptom: Wrong proxy.
  - Competing causes: metric mismatch, dependency, configuration, or operator error.
  - Diagnostic / expected signal: compare outcome/guardrail cohorts.
  - Evidence: HYPOTHETICAL teaching failure, not observed incident.
  - Containment: stop optimization.
  - Prevention: versioned contract, owned alert, negative-control replay.
  - Verification/limit: reproduce bounded trigger; production behavior unmeasured.
- F-08 | Trigger/symptom: Incompatible rollback.
  - Competing causes: partial rollback, dependency, configuration, or operator error.
  - Diagnostic / expected signal: bundle compatibility/restore replay.
  - Evidence: HYPOTHETICAL teaching failure, not observed incident.
  - Containment: restore bundle.
  - Prevention: versioned contract, owned alert, negative-control replay.
  - Verification/limit: reproduce bounded trigger; production behavior unmeasured.

## 13. Transferable Lesson
- Canonical pattern: `human-in-the-loop`.
- Principle: bind decision to authority, evidence, and state available at decision time; prediction is not permission.
- Preconditions: explicit owner, versioned inputs, safe fallback, recoverable action boundary.
- Mechanism: clinical escalation boundary narrows automation to checkable/reconcilable transitions.
- Evidence: derived queue arithmetic and design reasoning only.
- Avoid: unknown authority, unsafe/irreversible action, harmful fallback.
- Portability test: can reviewer reconstruct input, authority, bundle, transition, receipt?
- Adaptation: change gates/owner by domain; do not copy thresholds or workload values.

## 14. Connections & Cross-References
### 14a. Related cases, theory, and cheat sheets
- Uses [course numerical standards](../00_AUTHORING_AND_NUMERICAL_STANDARDS.md): queue conservation/evidence labels.
- Uses [case master template](../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md): decision/evaluation/recovery contract.
- Contrasts with planned CASE-XD-006: system-wide recovery rather than one decision boundary.
- Method homes: [agents](../06_llm_agents_knowledge_systems_and_ai_augmented_development/INDEX.md) and [platforms](../05_ai_platforms_data_systems_and_production_engineering/INDEX.md); not deployment evidence.
### 14b. What this case leaves open
- Which authority/safety rules apply? Resolve with responsible governance function.
- What review capacity/behavior occurs at peak? Resolve by instrumented shadow operation.
- Does proposed metric correlate with real outcome? Resolve by predeclared controlled evaluation.
- Which downstream contract gives durable receipt? Resolve with integration/recovery test.
### 14c. Intense interview pack — questions with checked answers
#### Q-01 / framing / defend a bounded decision
Candidate prompt: Frame decision and unsafe shortcut.
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: Use explicit gates, versioned evidence, durable state, and baseline/abstention when a prerequisite fails.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-02 / framing / defend a bounded decision
Candidate prompt: Unit of analysis and cut-off?
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: Use explicit gates, versioned evidence, durable state, and baseline/abstention when a prerequisite fails.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-03 / framing / defend a bounded decision
Candidate prompt: Why baseline necessary?
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: Use explicit gates, versioned evidence, durable state, and baseline/abstention when a prerequisite fails.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-04 / calculation / defend a bounded decision
Candidate prompt: Compute backlog at 1.2/s versus 1.0/s for 600 s.
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: B=(1.2-1.0)*600=120 decisions; units check is decisions/s times s.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-05 / calculation / defend a bounded decision
Candidate prompt: Compute drain for 120 decisions at 0.4/s surplus.
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: T=120/0.4=300 s; require positive recovery surplus.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-06 / calculation / defend a bounded decision
Candidate prompt: What if recovery arrivals equal service?
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: No drain: denominator is zero; reduce arrivals or increase service.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-07 / calculation / defend a bounded decision
Candidate prompt: Prevent post-decision label leak?
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: Use explicit gates, versioned evidence, durable state, and baseline/abstention when a prerequisite fails.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-08 / design / defend a bounded decision
Candidate prompt: Why is confidence insufficient for release?
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: Use explicit gates, versioned evidence, durable state, and baseline/abstention when a prerequisite fails.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-09 / design / defend a bounded decision
Candidate prompt: State machine for external timeout?
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: Use explicit gates, versioned evidence, durable state, and baseline/abstention when a prerequisite fails.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-10 / design / defend a bounded decision
Candidate prompt: Which denominator changes after ineligibility exclusion?
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: Use explicit gates, versioned evidence, durable state, and baseline/abstention when a prerequisite fails.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-11 / design / defend a bounded decision
Candidate prompt: Diagnose rising queue age.
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: Check arrival/service rates, dependency latency, and queue partition skew before changing thresholds.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-12 / diagnosis / defend a bounded decision
Candidate prompt: Reviewer available but evidence missing?
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: Use explicit gates, versioned evidence, durable state, and baseline/abstention when a prerequisite fails.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-13 / diagnosis / defend a bounded decision
Candidate prompt: How does stale policy alter architecture?
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: Use explicit gates, versioned evidence, durable state, and baseline/abstention when a prerequisite fails.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-14 / diagnosis / defend a bounded decision
Candidate prompt: What result falsifies proposal?
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: Use explicit gates, versioned evidence, durable state, and baseline/abstention when a prerequisite fails.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.
#### Q-15 / diagnosis / defend a bounded decision
Candidate prompt: What changes at 10× traffic?
Clarify: decision time, authority, reversibility, denominator, and whether given inputs are illustrative.
Strong answer: Preserve gates, estimate 12,000/hour peak, and load-test; do not linearly promise tail latency.
Check/conclusion: record reason code and verify against replay oracle or stated formula.
Acceptable alternative: stronger control is valid if it preserves authority, auditability, and bounded recovery.
Follow-up: change one workload, authority, dependency, or evidence assumption and explain fallback.
Red flag: treating score, retrieval, or retry as permission to release.
Scoring: 0 incorrect; 1 partial; 2 correct assumptions; 3 defended tradeoff.
Answer support: Sections 4, 6, 8, 9, and F-01 through F-08.


## 15. Quick Reference Card
| Field | Value |
|---|---|
| Domain | Healthcare |
| Problem type | llm-agentic: bounded decision |
| Method | retrieval, multimodal evidence display, uncertainty routing, and clinician sign-off; baseline manual chart review and templated search |
| Primary result | Not measured; illustrative queue arithmetic only |
| Key lesson | clinical escalation boundary; authority and recovery boundary |
| Tags | human-in-the-loop; authorization-scope; evaluation-mismatch |
| Supplemental field | Value |
|---|---|
| Case ID/version/as-of | CASE-XD-003 / 1.0 / 2026-11-09 |
| Evidence level | illustrative; constructed inputs, derived arithmetic |
| Implementation stage | proposed |
| Baseline/primary metric | manual chart review and templated search; supported-statement rate, unsafe-suggestion rate, review minutes, and slice coverage |
| Key limitation | no deployment, proprietary evidence, or measured outcome |
| Review trigger/owner | authority/workload/evidence change; clinical governance lead |
