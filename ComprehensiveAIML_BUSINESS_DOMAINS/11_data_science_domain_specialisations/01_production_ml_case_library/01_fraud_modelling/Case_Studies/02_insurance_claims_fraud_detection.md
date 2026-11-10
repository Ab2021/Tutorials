---
case_id: CASE-01-01-002
case_title: Claims investigation triage under delayed adjudication
case_version: "1.0"
created: 2026-11-09
last_reviewed: null
status: proposed
content_status: draft
case_type: predictive-ml
domain: Property and casualty insurance claims
primary_pattern: evaluation-mismatch
secondary_patterns: [label-delay, human-in-the-loop]
evidence_level: not-measured
source_basis: illustrative
as_of_date: 2026-11-09
owner_role: Claims special-investigations-unit product owner
source_index: INDEX.md
review_trigger: Material change to claim intake, referral policy, adjudication definition, investigation capacity, or regulated-data approval
---
# Case study: Claims investigation triage under delayed adjudication

## 1. The One-Line Framing
Given one newly reported eligible claim and only information available at intake, rank it for a special-investigations-unit (SIU) referral for the claims operations owner, to improve confirmed recoverable-loss value per investigator hour subject to claimant-service, due-process, privacy, and capacity constraints, compared with the existing rules-and-adjuster-referral queue.

This is an **ILLUSTRATIVE** design exercise. It describes no insurer, real claim population, trained model, deployment, investigation, or measured saving. Numbers below are constructed teaching inputs and are not an estimate of fraud prevalence or outcome.

The output is a prioritized referral recommendation, not an accusation, denial, coverage decision, or automated adverse action. The obvious framing—classify a claim as fraud/not fraud—misses that truth is delayed, some referred claims are investigated more intensely, and an investigator can review only a fixed number of files.

**Takeaway:** select a bounded, reviewable queue decision; do not convert an uncertain risk score into a fraud verdict.

## 2. Context & Background
A claimant reports a loss; an intake service creates a claim record; an adjuster gathers documents and estimates coverage; eligible referrals enter an SIU queue; investigators either close the referral, request more evidence, or refer an established matter through the organization’s approved process. The claim then continues through its ordinary handling path.

The decision owner is the claims operations leader. Adjusters need explanations and a timely response. SIU supervisors own capacity allocation and evidence quality. Legal, compliance, privacy, and model-risk roles approve permitted data, notices, and escalation rules. The model team may supply a score but does not decide coverage or investigation outcomes.

The baseline is a versioned set of intake rules plus voluntary adjuster referral. Its actual referral rate, lead time, and recovery results are unknown here. It persists because it is understandable, can encode known operational signals, and offers a documented fallback. A model can only be considered against the same eligibility population and review capacity.

The illustrative scope is first-notice-of-loss claims in one line of business after eligibility screening. Later investigation notes, settlement amounts, litigation outcome, and recovery information are unavailable at the initial decision unless they existed before the cutoff. Health, criminal, and regulated information may be subject to additional requirements; this case does not determine their applicability.

A high score should trigger a reviewed referral, not a claim denial. A low score must not prevent an adjuster from using an approved referral path for documented concerns. Referral can delay handling and can create claimant burden, so timeliness and disparity guardrails are decision metrics, not optional reporting.

**Takeaway:** the workflow establishes both the data cutoff and the human owner of every consequential action.

## 3. The Business Question
| Decision-contract field | Illustrative contract |
|---|---|
| Question | Which eligible new claims should enter a daily SIU review queue when only 40 investigator starts are available? |
| Decision owner | Claims operations leader, with SIU supervisor approval of queue policy. |
| Trigger and deadline | Claim-intake completion; rank within 15 minutes so ordinary handling is not blocked. |
| Available actions | Keep rules-only queue; refer top-ranked claims; request standard additional intake information; defer due to missing critical intake fields. |
| Value and downside | More substantiated recoverable loss per review hour versus avoidable delay, burden, and investigator opportunity cost. |
| Objective | Mature net recoverable value per 1,000 eligible claims and per investigator hour. |
| Hard constraints | No automated denial; 40 daily starts; approved data only; documented reason codes; applicable notice and review controls. |
| Scope | Eligible first-notice claims in one proposed pilot cohort. |
| Out of scope | Coverage determination, pricing, criminal guilt, provider-network investigation, and claims already referred. |

Define mature net recoverable value as confirmed recovery or avoided payment attributable under the approved outcome policy minus incremental investigation cost and pre-agreed claimant-service cost. Its numerator is currency over a cohort; its denominator is all eligible claims assigned to a policy, not merely investigated claims. It is not identical to the model’s probability score.

**Takeaway:** a queue objective must name its denominator and harms; ranking precision alone cannot choose the policy.

## 4. Assumptions & Constraints
| ID | Assumption or constraint | Type | Verification plan | Status | Impact if wrong | Fallback / owner |
|---|---|---|---|---|---|---|
| A-01 | Intake timestamp and claim ID are stable. | data | replay duplicate and ordering checks | unverified | point-in-time features fail | retain rules queue / platform owner |
| A-02 | Referral outcome has an auditable maturity state. | label | plot resolution delay by cohort | unverified | labels are selected or censored | report maturity strata / analytics owner |
| A-03 | SIU can start 40 files/day. | operations | schedule and time study | illustrative | top-K policy is infeasible | lower K or backlog policy / SIU supervisor |
| A-04 | Permitted features are documented. | privacy | field-level approval register | unverified | use is disallowed | remove feature / privacy owner |
| A-05 | Adjuster referral remains available. | workflow | acceptance test | proposed | model suppresses human concern | manual escalation / claims leader |
| C-01 | Ranking does not autonomously change coverage. | governance | policy and UI test | required | adverse action risk | disable ranked referral / product owner |
| C-02 | Review reasons are recorded. | auditability | sampled file review | proposed | outcome cannot be interpreted | rules-only routing / SIU supervisor |

The two assumptions most likely to reverse the recommendation are label quality (A-02) and actual capacity (A-03). Neither can be repaired by a better offline AUC.

**Takeaway:** list a testable escape route for each assumption that could invalidate the queue.

## 5. Data, Tools & Method
### 5a. Data Specification
The unit is one claim at initial intake, keyed by a tokenized claim identifier and an intake-version identifier. A reopened claim is a new decision only if the policy defines it so; it must retain linkage to the prior record.

Candidate inputs are intake event time; loss type; reported amount and currency; policy tenure band; location supplied for the loss; claim channel; document count; declared incident date; prior claim count as of intake; repair-provider identifier where already supplied; and missingness indicators. They are candidates, not approved or known-available features.

Each field records event time, source-ingestion time, availability time, source version, and permitted-use status. The decision cutoff is the earliest time at which the queue is generated. A later adjuster note, final payment, SIU narrative, or adjudication event is prohibited from an intake feature unless a historical version was already available by that cutoff.

The outcome ledger retains referral status, investigation start/end, disposition code, resolution time, established recovery value if applicable, outcome confidence, and appeal/reversal status. “No confirmed issue” is not synonymous with legitimate claim: a referral might not be investigated, might remain unresolved, or might lack evidence.

Population exclusions must be explicit: incomplete records, claims already in SIU, outside-line-of-business claims, and records without a lawful processing basis. Report exclusion count and reason by cohort. Split by time—not random rows—to prevent future process changes from leaking backwards; keep an untouched forward cohort after policy selection.

### 5b. Tools & Stack
The proposed stack is a versioned intake snapshot, approved feature builder, calibrated tabular ranker, policy service, durable decision log, SIU work queue, and monitoring store. Tool names and versions are intentionally unspecified because no implementation was examined. A durable log stores claim key, cutoff, feature-contract version, score, policy version, action, reason codes, and fallback reason; it never proves an outcome.

### 5c. Method Summary
Input -> schema and availability validation -> point-in-time feature construction -> calibrated risk ranking -> top-K policy plus rules guardrails -> referral recommendation and audit record.

| Candidate | Evidence considered | Strength | Limitation | Status |
|---|---|---|---|---|
| Existing rules + adjuster referral | baseline to measure | transparent and immediate | may miss combinations and cannot prioritize consistently | comparator, not measured |
| Random review sample | identifies some low-score outcomes | reduces selective-label bias | consumes scarce capacity | proposed audit arm |
| Calibrated tabular ranker + top-K | can prioritize review capacity | ranking is tunable and auditable | needs mature labels and calibration | proposed |
| Unsupervised anomaly score | may reveal novel patterns | no confirmed-label requirement | anomaly is not fraud and may be unstable | considered, not tested |

The policy score estimates a defined historical outcome conditional on the data-generation process; it does not estimate the causal benefit of referral. Threshold selection therefore uses review capacity, mature outcomes, claimant-service guardrails, and a controlled evaluation plan rather than a default 0.5 cutoff.

### 5d. Provenance, quality, exclusions and leakage controls
For each feature, maintain owner, extraction query revision, raw snapshot identifier, availability rule, missingness, transformations, join cardinality, and retention basis. Joins use claim key plus a version or event cutoff; a current dimension table cannot overwrite what was known at intake. Fit imputers, encoders, and calibration only on the training period inside each temporal evaluation fold.

Audit duplicate claims, provider-key collisions, late corrections, timezone conversion, unexpected negative durations, and missingness changes. Separate missing because a field was never collected from missing because an upstream feed failed. Exclude a field with unverified availability time rather than silently backfilling it.

### 5e. Reproducibility and method-selection ledger
| Artifact ID | Location | Version/date | Purpose | Accessibility | Limitation |
|---|---|---|---|---|---|
| SRC-01 | [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | accessed 2026-11-09 | method reference for training-only preprocessing | public | not evidence of this case performance |
| SRC-02 | Internal approved data dictionary | to obtain | feature availability and permission | restricted | unavailable in this illustrative case |
| RUN-01 | Proposed temporal replay configuration | not executed | reproduce ranking evaluation | unavailable | no run, data, or result exists |
| POL-01 | Proposed referral-policy specification | draft | top-K and guardrail logic | illustrative | not approved or deployed |

A real run requires snapshot hashes, code revision, package versions, random seed, feature-contract version, policy configuration, score artifact hash, and a retained decision cohort. A seed does not make changing claims data reproducible.

**Takeaway:** reproduce the decision-time dataset and policy, not merely the model binary.

## 6. Analysis Walkthrough
The decisive current finding is negative: no evidence supports replacing the baseline because no data or run has been supplied. The walkthrough below is a checked **ILLUSTRATIVE** capacity calculation showing what a future evaluation must report.

### Step 1 — Fix the cohort and capacity
Assume 10,000 eligible claims in an evaluation cohort; 100 later reach the defined confirmed outcome after maturity; the policy refers the top 200. Of those referrals, 60 later reach the outcome. These are constructed counts, not observed claims.

Validity before use: every claim belongs to the same closed cohort; the outcome definition and maturity rule are fixed; “confirmed” is consistently recorded.

```text
FORMULA [C-01]: precision@K = TP_K / K
WHERE: TP_K = mature positive outcomes among K referrals [claims]; K = referral capacity [claims].
CONSTRAINT: referral set and outcome maturity are fixed; precision is undefined if K=0.
NUMERICAL EXAMPLE: TP_200=60 claims; K=200 claims; precision@200=60/200=0.30=30%.
INTERPRETATION: 30 of each 100 referred claims mature positive under this constructed cohort; it does not measure causal savings.
```

```text
FORMULA [C-02]: recall@K = TP_K / P
WHERE: P = all mature positive outcomes in the eligible cohort [claims].
CONSTRAINT: P uses the same cohort and maturity definition; unknown outcomes cannot be assumed negative.
NUMERICAL EXAMPLE: TP_200=60; P=100; recall@200=60/100=0.60=60%.
INTERPRETATION: 40 constructed positive outcomes remain outside the 200-referral queue.
```

Independent check: 200 referrals contain 60 TP and 140 FP; 10,000 eligible claims contain 100 positives, hence 40 FN and 9,760 TN. Both denominators reconcile: 60+140=200 and 60+40=100.

### Step 2 — Connect capacity to workload
Assume 40 starts/day and five workdays: the weekly capacity is 200 starts, matching K. This does not mean every file closes in a week; starts and completions are different flows.

```text
FORMULA [C-03]: W = s * d
WHERE: W = weekly starts [claims/week]; s = staffed starts [claims/day]; d = operating days [days/week].
CONSTRAINT: constant capacity and no staff absence, rework, or carryover; use queue simulation when this is false.
NUMERICAL EXAMPLE: s=40 claims/day; d=5 days/week; W=40*5=200 claims/week.
INTERPRETATION: choose K no greater than verified usable starts, not merely analyst headcount.
```

### Step 3 — Estimate an illustrative value ledger
Suppose each mature positive referral has illustrative attributable recoverable value of $2,000, each investigation costs $250, and an avoidable referral service cost is $25. If all 200 referrals are investigated, the mechanical estimate is 60*$2,000 - 200*$250 - 200*$25 = $65,000. Units are currency per constructed cohort. It is not a causal result because the counterfactual recovery under the baseline and referral effect are unknown.

```text
FORMULA [C-04]: V = TP_K*v - K*(c_i + c_s)
WHERE: V = illustrative net value [currency/cohort]; v = value/positive referral [currency/claim]; c_i = investigation cost [currency/claim]; c_s = service cost [currency/referred claim].
CONSTRAINT: values are attributable and comparable across actions; this is generally unverified before a controlled study.
NUMERICAL EXAMPLE: 60*2000 - 200*(250+25) = 120000 - 55000 = $65000.
INTERPRETATION: the result is a transparent scenario, not evidence that deployment saves $65,000.
```

Sensitivity: if value per positive is $1,000, V=$5,000; if only 40 positives are found at $2,000, V=$25,000. If only 20 positives are found, V=-$15,000. Therefore a capacity-saturated queue is sensitive to outcome yield and value attribution; a policy must not be justified from rank metrics alone.

What we tried first: **not tested**. No baseline or candidate model was run. The rules baseline is a required comparator, not fabricated evidence.

**Takeaway:** verify counts, capacity, and counterfactual value separately; an attractive illustrative arithmetic result does not establish a rollout case.

## 7. The Finding
Do not replace the existing rules-and-adjuster referral process on the evidence available. A proposed calibrated top-K ranker may be evaluated only as a referral aid with fixed capacity, point-in-time features, mature outcome definitions, and claimant-service guardrails. The constructed example gives 30% precision@200 and 60% recall@200, but neither figure is observed, causal, or portable to an insurer. The recommendation is to collect an approved temporal replay and controlled policy-evaluation evidence before deciding whether to use a ranker.

**Takeaway:** the defensible answer is an evaluation gate, not an invented deployment outcome.

## 8. System/Solution Design
```text
[Intake claim + availability timestamps]
          | schema/permission validation --fails--> [existing routing + reason]
          v
[Point-in-time feature builder] --> [feature freshness/contract checks]
          |                                       | fails
          v                                       v
[Calibrated score artifact] --> [policy: rules guardrail + top-K] --> [durable decision log]
          |                                                   |         |
          v                                                   v         v
[reason-code explanation] ------------------------------> [SIU queue] [monitoring/replay]
                                                                  |
                                                                  v
                                                    [human investigation and documented disposition]
```

The online path accepts an immutable intake snapshot and returns a recommendation before the handling deadline. The policy service must be deterministic for the same claim snapshot and policy version. A durable claim-decision key prevents retry from creating multiple queue entries. A retry returns the stored decision if the key and snapshot hash match; a changed intake version creates an auditable superseding decision under explicit policy.

A queue service reserves capacity by date and priority. It must never silently discard an overflow recommendation; it records a non-referral reason such as capacity exhausted, ineligible channel, stale feature, or baseline-rule override. SIU staff may override a recommendation with a documented reason. Adjuster referrals retain their approved priority path.

Failure recovery: if scoring, feature freshness, or policy retrieval is unavailable, use the approved existing routing policy and log the fallback reason. Do not assume one universal fail-open or fail-closed rule: it depends on claim process, approved controls, and claimant effect. Reconcile decision-log rows to queue entries daily; duplicate keys and missing queue acknowledgements are investigated before replay.

Capacity sizing has no benchmark claim. Measure arrival rate, service-time distribution, rework, carryover, and available staffed hours on the actual queue. A mean throughput calculation is not a p95 completion guarantee.

**Takeaway:** state ownership, idempotency, fallback, and reconciliation; a score service alone is not an operable referral system.

## 9. Evaluation & Validation
| Metric | Numerator / denominator | Population / window | Comparator | Guardrail | Status |
|---|---|---|---|---|---|
| precision@K | mature positives in K / K | eligible claims, fixed cutoff | rules queue at same K | referral delay | proposed |
| recall@K | mature positives in K / all mature positives | same cohort, maturity window | baseline coverage | unknown-label rate | proposed |
| net value | attributable recovery minus costs / eligible claims | assigned cohort, mature window | existing policy | claimant service and appeals | not measured |
| investigation start time | referrals started by deadline / referrals | same period | baseline | staffing distribution | proposed |
| disparity review | action and outcome rates by approved slice | sufficient-count cohort | policy comparator | uncertainty and privacy | proposed |

Use a chronological development window, a later validation window for policy selection, and a final forward holdout. Backtest both candidate and baseline policies on the same eligibility and capacity. However, historical referrals are selected by the old policy; this makes outcomes selectively observed. Include randomized low-score audit referrals where permitted, retain an unresolved state, and quantify the maturity curve rather than treating silence as a negative outcome.

For a prospective evaluation, randomize eligible units to the existing policy versus the proposed policy only after governance approval. Analyze intent-to-treat by assignment, retain all fallback traffic, pre-register the primary metric and stopping rules, and monitor service and safety guardrails. Shadow scoring tests technical compatibility but cannot reveal outcomes under changed referrals.

Sensitivity tests: change maturity window, exclude uncertain outcomes, use conservative value attribution, change K to actual staffing, replay feature delays, and stratify by intake channel. A conclusion that reverses under a reasonable maturity or capacity definition is inconclusive.

**Takeaway:** validate the entire decision policy on mature, comparable cohorts rather than validating the score in isolation.

## 10. Action Taken
No production action was taken. The only action in this illustrative case is a proposed evidence plan: inventory permitted intake data; define outcome-maturity states; measure rules baseline and capacity; build a point-in-time replay; conduct privacy/model-risk review; then decide whether a bounded prospective evaluation is justified.

A real action log should record approver, policy version, eligibility, assignment, fallback, claimant-service measure, and stop decision. It must distinguish “not approved,” “not technically ready,” and “policy evaluated but not beneficial.”

**Takeaway:** a proposal is not a pilot, and a pilot is not a production result.

## 11. Outcome & Results
Outcome is **not measured**. There is no deployed ranker, eligible population count, investigator workload observation, rules baseline, calibration result, or recovery estimate in this case. The $65,000 arithmetic is an illustrative conditional calculation only.

A future result statement must name the policy, cohort, maturity window, comparator, numerator, denominator, uncertainty, guardrails, and whether results were observed, estimated, or simulated. It must not report referral count as proven fraud prevented.

**Takeaway:** preserve the evidence boundary beside the conclusion, not only in metadata.

## 12. Failure Modes & What Went Wrong
### F-01 — Future-information leakage
**Symptom:** offline precision rises sharply after adding adjuster-note or payment fields.

**Competing causes:** real useful intake signal; field was recorded after referral; current-value join overwrote historical value.

**Discriminating check:** compare each field’s availability timestamp with queue cutoff and replay from an immutable snapshot. Expected signal: a leaked field is unavailable for many historical decisions.

**Bounded fix:** remove or lag the field; rebuild features from cutoff snapshots; refit preprocessing within temporal folds. **Risk:** removal may reduce apparent rank quality. **Verification:** sampled decision records show every retained feature existed before the decision.

### F-02 — Selective and immature labels
**Symptom:** referred claims appear far more positive than nonreferred claims, while recent cohorts have few outcomes.

**Competing causes:** good targeting; SIU investigation intensity; unresolved claims; different follow-up windows.

**Discriminating check:** plot disposition/maturity by referral status and cohort age; audit a permitted random sample outside the queue. Expected signal: outcome observability differs by action or cohort.

**Bounded fix:** retain unresolved labels, wait for maturity, use audit sampling, and report observation propensity. **Risk:** slower learning and use of scarce review capacity. **Verification:** maturity curves and audit outcomes are included in the evaluation ledger.

### F-03 — Capacity overflow and stale priority
**Symptom:** recommended referrals wait days, while new high-priority claims enter behind an old backlog.

**Competing causes:** capacity decrease; surge in arrivals; duplicate queue messages; long investigation cycle.

**Discriminating check:** reconcile accepted recommendations, queue positions, starts, completions, and duplicate keys by day. Expected signal: backlog growth equals arrivals minus starts after accounting for cancellation.

**Bounded fix:** set K from measured starts, expire/re-rank under approved policy, deduplicate, and expose overflow. **Risk:** lower recall and delayed valuable referrals. **Verification:** replay shows no silent drops and queue age stays within the approved operational limit.

### F-04 — Proxy harm or opaque adverse handling
**Symptom:** one approved review slice receives materially more referrals or delay without a documented explanation.

**Competing causes:** channel mix; missing intake data; genuine exposure difference; feature proxy; staffing geography.

**Discriminating check:** compare eligibility, missingness, score distribution, action rate, mature outcomes, and uncertainty by permitted slice. Expected signal: a disparity caused by an upstream missing field often co-occurs with missingness and channel shift.

**Bounded fix:** pause the affected policy segment, remove or constrain the feature, improve intake data, and obtain governance review. **Risk:** overall queue yield can decline. **Verification:** repeat pre-specified slice analysis on a later mature cohort; do not declare fairness from one aggregate rate.

### F-05 — Duplicate or non-idempotent referral
**Symptom:** the same claim appears twice in SIU or score and queue records disagree.

**Competing causes:** client retry; intake amendment; lost queue acknowledgement; key collision.

**Discriminating check:** match immutable claim-decision key, snapshot hash, policy version, and queue receipt. Expected signal: a retry has same key/hash, whereas an amendment has a documented new version.

**Bounded fix:** durable upsert and explicit supersession protocol. **Risk:** over-deduplication can hide a legitimate amended claim. **Verification:** injected retry and amendment traces produce exactly one active queue record with an audit chain.

**Takeaway:** every failure diagnosis needs a competing explanation, a signal that separates it, a limited repair, and a verification step.

## 13. Transferable Lesson
Risk-ranking systems are allocation policies, not truth machines. The transferable sequence is: define a decision-time unit and action; construct data available then; state who is observed and who is not; constrain a policy to real operational capacity; evaluate outcome and harm on the assigned population; preserve a human, auditable path for exceptions; and keep a safe existing-process fallback.

The same design transfers to payment review, moderation, document quality assurance, clinical coding review, and maintenance triage. It does not transfer unchanged where referrals themselves impose materially different harms, where labels have different legal meanings, or where a human review cannot make the action safe.

**Takeaway:** changing the intervention, label process, or capacity changes the problem even when the model family is the same.

## 14. Connections & Cross-References
Related methods: [Fraud evaluation, calibration and capacity](../../../02_data_science_method_playbooks/02_fraud_modelling_playbook/Cheat_Sheets/01_Fraud_Evaluation_Calibration_and_Capacity.md) provides reusable rank-metric and capacity reasoning. The local payment-card case contrasts synchronous authorization with asynchronous investigation. The [claims reserving methods draft](../../../10_insurance_actuarial_and_claims/01_pricing_underwriting_and_reserving/Cheat_Sheets/05_Claims_Reserving_Methods.md) provides adjacent claims context, not fraud-method evidence.

Open questions: which outcomes are permitted and auditable; whether randomized audit review is feasible; how much of a recovery is attributable to referral; and which jurisdiction-specific safeguards apply. None is resolved by this teaching case.

### 14a. Interview pack
#### Q01 — Why not use accuracy?
**Answer:** fraud/issue prevalence may be low and capacity is capped, so a high-accuracy classifier can refer no useful claims. Evaluate precision@K, recall@K, mature value, and claimant-service guardrails. **Reasoning:** accuracy weights all correct nonreferrals heavily. **Follow-up:** if K doubles, replay both policy and staffing; do not assume value doubles. **Acceptable alternative:** cost-weighted utility if inputs are defensible. **Red flag:** calling 99% accuracy deployment-ready. **Scoring:** 0 wrong; 1 names imbalance; 2 connects metric to capacity; 3 adds label-selection boundary. **Support:** Sections 3, 6, 9.

#### Q02 — Compute the illustrative precision and recall.
**Answer:** 60/200=30% precision@200; 60/100=60% recall@200. **Reasoning:** the first denominator is referrals, the second all mature positives. **Follow-up:** false positives are 140 and false negatives 40. **Acceptable alternative:** report both counts before percentages. **Red flag:** dividing 60 by 10,000 for either measure. **Scoring:** 0 incorrect; 1 one metric; 2 both metrics and denominators; 3 names maturity assumption. **Support:** C-01/C-02.

#### Q03 — What is wrong with training on final payment amount?
**Answer:** it is generally unavailable at intake and leaks future handling. **Reasoning:** the score would exploit an outcome of the process it is supposed to influence. **Follow-up:** retain only a versioned estimate present at cutoff if approved. **Acceptable alternative:** lagged historical amount features with availability proof. **Red flag:** “the warehouse has it now.” **Scoring:** 0 denies issue; 1 says leakage; 2 explains timing; 3 proposes replay test. **Support:** Section 5d.

#### Q04 — How do selected labels bias the result?
**Answer:** SIU referrals receive more investigation, so their outcome is more observable than nonreferrals. **Reasoning:** a score can learn prior referral policy rather than underlying value. **Follow-up:** use permitted random audits and maturity strata. **Acceptable alternative:** carefully bounded propensity analysis, with limits. **Red flag:** treating nonreferred claims as legitimate. **Scoring:** 0 unsafe; 1 names bias; 2 mechanism and mitigation; 3 names residual uncertainty. **Support:** Sections 5a, 9, F-02.

#### Q05 — Show the weekly capacity calculation.
**Answer:** 40 starts/day * 5 days/week = 200 starts/week. **Reasoning:** starts are not completions. **Follow-up:** if only 30 starts/day, set weekly K to 150 or explicitly manage backlog. **Acceptable alternative:** simulation using arrival/service distributions. **Red flag:** multiplying investigator count without measuring work rate. **Scoring:** 0 wrong; 1 arithmetic only; 2 unit-aware; 3 states queue boundary. **Support:** C-03.

#### Q06 — Is the $65,000 a business case result?
**Answer:** no; it is a conditional illustrative value ledger. **Reasoning:** attributable value, investigation effect, and baseline counterfactual are unobserved. **Follow-up:** at $1,000 per positive the scenario becomes $5,000. **Acceptable alternative:** defer value until controlled evidence exists. **Red flag:** presenting it as measured savings. **Scoring:** 0 claims proof; 1 caveat only; 2 lists missing evidence; 3 recomputes sensitivity. **Support:** C-04.

#### Q07 — Why retain the rules baseline?
**Answer:** it is the operational comparator and fallback; removing it prevents a like-for-like policy evaluation. **Reasoning:** rules may encode valid mandatory controls. **Follow-up:** version policy and run both on same replay cohort. **Acceptable alternative:** an approved manual queue baseline. **Red flag:** assuming ML supersedes controls. **Scoring:** 0 unsafe; 1 calls it baseline; 2 explains comparison; 3 includes fallback. **Support:** Sections 2, 5c, 8.

#### Q08 — A referral queue is delayed; diagnose.
**Answer:** distinguish surge, reduced starts, long handling time, duplicate messages, and stale priority using arrival/start/completion/key reconciliation. **Reasoning:** queue age alone has multiple causes. **Follow-up:** first expose overflow and stop silent drops. **Acceptable alternative:** queue simulation after data validation. **Red flag:** increase K without capacity. **Scoring:** 0 single-cause claim; 1 names backlog; 2 gives discriminating signals; 3 bounded remediation. **Support:** F-03.

#### Q09 — What does calibration add?
**Answer:** it maps score ranges to observed outcome rates in a defined cohort; it can support policy reasoning but does not prove referral causality. **Reasoning:** ranking quality and probability calibration differ. **Follow-up:** assess calibration by time/channel after maturity. **Acceptable alternative:** use rank policy without probability interpretation. **Red flag:** “a 0.8 score means 80% fraud everywhere.” **Scoring:** 0 false certainty; 1 definition; 2 boundary; 3 validation plan. **Support:** Sections 5c and 9.

#### Q10 — A new channel has missing provider IDs. Deploy unchanged?
**Answer:** no; it changes feature availability and may change the score distribution. Route to approved baseline or missing-aware policy only after temporal/channel validation. **Reasoning:** missingness can be operational failure or domain shift. **Follow-up:** compare missingness and outcomes on a mature pilot cohort. **Acceptable alternative:** pause channel expansion. **Red flag:** impute the modal provider silently. **Scoring:** 0 deploys blindly; 1 notes drift; 2 gives safe path; 3 names verification. **Support:** Sections 5d, F-04.

#### Q11 — How do retries avoid duplicate referrals?
**Answer:** store and return a decision keyed by immutable claim-decision key and snapshot hash; use explicit supersession for amended intake. **Reasoning:** network retry is not a new claim. **Follow-up:** test identical retry and valid amendment traces. **Acceptable alternative:** transactional outbox with equivalent durable semantics. **Red flag:** broker exactly-once alone proves end-to-end uniqueness. **Scoring:** 0 unsafe; 1 mentions dedupe; 2 distinguishes amendment; 3 specifies verification. **Support:** Section 8, F-05.

#### Q12 — What changed constraint would stop the pilot?
**Answer:** a policy change that prohibits a needed feature, capacity below the planned audit minimum, or a guardrail breach requires pause and reassessment. **Reasoning:** a different data/action contract is a different intervention. **Follow-up:** continue existing routing and preserve decision logs. **Acceptable alternative:** restrict to a validated segment. **Red flag:** retune threshold to hide a guardrail breach. **Scoring:** 0 ignores constraint; 1 pauses; 2 explains fallback; 3 identifies revalidation scope. **Support:** Sections 4, 9, 10.

### 14b. Decision-defense drills
#### Drill D-01 — Define the policy unit before discussing features
The policy unit is an eligible claim-intake record at a fixed cutoff, not a claimant, policy, repair provider, or eventual claim file.
The claim key identifies the decision record.
The intake-version key distinguishes a valid amendment from a retry.
The cutoff is the earliest time the ranking policy can act.
The action is referral priority, not a coverage determination.
The owner is the claims operations leader.
The reviewer is an SIU investigator.
The outcome is an explicitly defined mature review result.
The consequence is allocation of scarce investigation time.
The baseline is versioned rules plus adjuster referral.
The counterfactual is the baseline action for the same eligible claim.
The guardrail population includes legitimate claimants affected by delay.
The audit population includes referrals and permitted random-audit claims.
The excluded population is documented rather than silently dropped.
The lesson is that changing any of these elements changes the estimand.

#### Drill D-02 — Trace one illustrative record without future information
At 09:00 UTC a claim intake event arrives with claim key C-17 and snapshot version 1.
The policy stores the receipt time before feature construction.
The reported amount is present with stated currency and source version.
The policy-tenure band is retrieved only from a version whose availability time is before 09:00.
The prior-claim count is computed over claims available before 09:00.
The provider identifier is absent, so the missingness indicator is true.
The absence is not converted to a risky provider category.
No adjuster note is yet available.
No payment, settlement, disposition, or recovery field is permitted.
Schema validation passes because required intake fields exist.
The feature contract records its version.
The score artifact records its version.
The capacity policy records that 40 daily starts remain.
The policy selects or does not select C-17 according to its frozen ranking rule.
The durable decision log stores key, snapshot hash, score bucket, action, reason, and policy version.
If the client retries C-17 with the same hash, the stored action is returned.
If intake correction produces version 2, the policy records a superseding decision rather than overwriting version 1.
An investigator sees only the permitted workbench view.
Any later disposition is written as a later outcome event, never backfilled as an original feature.
This trace is an invariant to test in replay, not evidence of a real system.

#### Drill D-03 — Reconcile the illustrative confusion matrix
The eligible cohort contains 10,000 constructed claims.
The mature-positive count is 100 constructed claims.
The referral capacity is 200 constructed claims.
The constructed top-K set contains 60 true positives.
The constructed top-K set therefore contains 140 claims without the defined mature outcome.
The nonreferred set contains 9,800 claims.
The nonreferred set contains 40 mature positives.
The nonreferred set therefore contains 9,760 claims without the defined mature outcome.
The referral-row total is 60 plus 140, or 200.
The nonreferral-row total is 40 plus 9,760, or 9,800.
The positive-column total is 60 plus 40, or 100.
The negative-column total is 140 plus 9,760, or 9,900.
The grand total is 10,000.
Precision is 60 divided by 200.
Recall is 60 divided by 100.
The referred fraction is 200 divided by 10,000, or 2%.
The false-positive count is not proof that 140 referrals were harmful.
The false-negative count is not proof that 40 claims were mishandled.
Both labels are shorthand for the stipulated historical mature outcome.
Selective investigation can invalidate interpretation of the nonreferred column.
The matrix is a consistency check, not a causal impact estimate.

#### Drill D-04 — Defend the capacity boundary
Capacity is stated in starts because a referral can be opened before it is resolved.
Completion capacity may be lower or higher than start capacity in a given week.
Investigator headcount is not capacity without availability and case-time data.
Training time, quality review, leave, escalations, and rework reduce usable starts.
Urgent referrals can preempt a planned ranked slot.
Rules-mandated routing can consume capacity before discretionary ranking.
The top-K policy must reserve mandatory work explicitly.
The policy must expose overflow rather than moving it to an invisible queue.
Queue age is a guardrail because a stale referral can lose practical value.
The policy owner must choose whether unreviewed referrals expire, rerank, or retain position.
Each choice changes the action taken and must be logged.
If arrivals exceed starts continuously, a larger buffer does not solve the imbalance.
If a review takes twice as long as assumed, K is not safe merely because the score remains accurate.
If staffing rises, a new K still requires a policy evaluation because lower-ranked cases have a different score distribution.
Capacity should be measured by channel and claim complexity when those change service time.
No constructed workload value is a service-level commitment.

#### Drill D-05 — Test a temporal split
Development uses an earlier closed period.
Validation uses a later period that the developer cannot use to fit transformations.
The forward test period remains untouched until model and policy are frozen.
All claims for the same linked entity need an explicit boundary rule.
An entity appearing in two periods can be valid if features use only prior information.
An entity-level random split can conceal time drift in intake process.
A row-random split can put later process labels into the apparent past.
The feature pipeline fits imputation on development data only.
The encoder learns categories from development data only.
The calibration procedure fits on an allowed calibration partition only.
Threshold selection uses validation capacity and guardrails only.
The forward test is evaluated once after the choice is frozen.
Any policy revision creates a new evaluation version.
Missingness distribution is compared across periods.
Label maturity is compared across periods.
Eligibility rules are versioned across periods.
Changes in a referral program are treated as interventions, not noise.
The test report names records excluded for maturity and why.
The test report reports unresolved counts rather than recoding them.
Temporal evaluation cannot alone identify referral causality.

#### Drill D-06 — Distinguish four different values
Reported claim amount is an intake statement in currency units.
Estimated covered loss is a later claim-handling estimate, not necessarily available at intake.
Established recovery is an outcome that may occur after investigation.
Avoided payment is a counterfactual estimate requiring a defensible method.
Investigation cost is a resource cost, usually staff time and vendor cost.
Claimant-service cost is a pre-agreed representation of delay or burden, not a moral valuation.
The score can use none of these unless their availability and permission are established.
The value ledger must not double count recovery and avoided payment.
The value ledger must identify whether costs are incremental or already sunk.
The value ledger must separate nominal and inflation-adjusted currency if time periods differ.
The value ledger must name the currency and window.
The value ledger must use the same cohort for baseline and candidate.
The value ledger must disclose unresolved outcomes.
The value ledger must not imply causal attribution from a referral disposition.
A sensitivity range is more honest than a single invented point estimate.
If value attribution is unavailable, rank metrics and operational metrics remain distinct.
The decision can still be “collect evidence” rather than “deploy.”

#### Drill D-07 — Write a model-card claim ledger
Claim CLM-01: a top-K policy can be replayed from a versioned intake snapshot.
Evidence type: proposed mechanism.
Support: durable snapshot, feature contract, and replay configuration must be obtained.
It does not establish predictive performance.
Claim CLM-02: precision@200 is 30% under the constructed counts.
Evidence type: derived and hand-checked arithmetic.
Support: C-01 and the confusion-matrix reconciliation.
It does not establish a production precision rate.
Claim CLM-03: workload is 250 hours under 200 reviews at 75 minutes.
Evidence type: derived illustrative arithmetic.
Support: M-03-equivalent calculation in Section 6.
It does not establish actual staffing need.
Claim CLM-04: a current-value join can leak future information.
Evidence type: method principle.
Support: availability-time audit design and cited preprocessing guidance.
It does not establish that leakage happened in an insurer.
Claim CLM-05: randomized audit review can reduce selective-label uncertainty.
Evidence type: proposed evaluation design.
Support: approved policy and audit protocol are required.
It does not establish feasibility or causal results.
This ledger prevents a reader from promoting a design premise into a result.

#### Drill D-08 — Specify reason codes
REFERRAL_RANK means the record was selected by the discretionary rank policy.
REFERRAL_RULE means a mandatory approved rule routed the record.
REFERRAL_MANUAL means an authorized adjuster or supervisor used an override.
NO_REFERRAL_CAPACITY means the candidate was eligible but not selected within K.
NO_REFERRAL_INELIGIBLE means the record did not meet the declared population definition.
FALLBACK_FEATURE_STALE means an approved feature-freshness condition failed.
FALLBACK_POLICY_UNAVAILABLE means ranking was unavailable and baseline routing was used.
SUPERSEDED_INTAKE means a later intake version replaced the prior decision under policy.
DUPLICATE_RETRY means the prior durable decision was returned.
Every reason code has a policy version and timestamp.
No reason code is an accusation or claim outcome.
Reason-code distributions are monitoring inputs.
Unexpected growth in fallback codes triggers operational investigation.
Manual overrides need an authorized actor and bounded free-text policy.
The code taxonomy itself requires change control.
Reason codes make later evaluation interpretable but do not prove correctness.

#### Drill D-09 — Evaluate calibration honestly
Calibration compares predicted risk with observed outcomes in a specified cohort.
It requires a defined outcome and maturity rule.
It is evaluated after the feature and policy population are fixed.
A reliability plot is meaningless if unresolved records are silently negative.
Calibration can differ by intake channel and time period.
A well-calibrated risk model can still have poor top-K operational value.
A well-ranked model can be poorly calibrated as a probability model.
Calibration does not estimate the incremental effect of investigation.
Recalibration changes the score meaning and must be versioned.
An isotonic or logistic calibrator must be fit without using the forward test for tuning.
Small score bins need uncertainty intervals or pooling.
Sparse approved slices need a privacy and minimum-count policy.
The result should say “calibrated in this evaluation cohort,” not “calibrated universally.”
If the policy only needs ordering, avoid false precision in probability wording.
If a score bucket changes action, audit that boundary separately.

#### Drill D-10 — Define rollout gates
Gate one: permitted fields and availability-time lineage are approved.
Gate two: baseline eligibility and capacity are measured.
Gate three: replay reproduces baseline decisions within agreed reconciliation tolerance.
Gate four: duplicate/retry and amended-intake traces meet the decision invariant.
Gate five: outcome taxonomy and maturity curve are reviewed by owners.
Gate six: candidate evaluation uses a locked temporal split.
Gate seven: claimant-service and approved-slice guardrails are specified.
Gate eight: audit-sampling feasibility and governance are decided.
Gate nine: fallback routing and monitoring ownership are tested.
Gate ten: an authorized owner chooses pilot, pause, or rejection.
Passing a technical gate does not pass governance.
Passing an offline gate does not pass a prospective policy gate.
Failing a guardrail pauses the affected scope.
The rollback is the preserved existing routing policy.
The post-pilot review records all fallback and override traffic.
No gate may be marked passed without its referenced artifact.

#### Drill D-11 — Additional interview questions
#### Q13 — What denominator belongs in a referral-rate guardrail?
**Answer:** eligible claims assigned to the policy, stratified by approved channel or cohort when relevant.
**Reasoning:** referred-only denominators hide differential eligibility and exclusion.
**Follow-up:** report exclusions and their reasons beside the rate.
**Acceptable alternative:** an intent-to-treat denominator for a randomized pilot.
**Red flag:** divide by reviewed claims when discussing claimant exposure.
**Scoring:** 0 wrong denominator; 1 names eligible claims; 2 explains selection; 3 adds exclusion audit.
**Support:** Sections 3 and 9.

#### Q14 — How would you test an amended claim record?
**Answer:** send an identical retry and then a documented later intake version with a changed snapshot hash.
**Reasoning:** the retry must return one durable action; the amendment must create an auditable supersession.
**Follow-up:** reconcile active queue count after both events.
**Acceptable alternative:** transactional outbox with the same invariant.
**Red flag:** overwrite the original record and lose its action.
**Scoring:** 0 unsafe; 1 says dedupe; 2 distinguishes amendment; 3 gives an oracle.
**Support:** Section 8 and F-05.

#### Q15 — Why is a high provider-risk association insufficient?
**Answer:** association may reflect provider mix, reporting practices, entity-resolution error, or a genuine signal.
**Reasoning:** a referral policy needs timing, evidence, permitted use, and outcome validation.
**Follow-up:** compare provider-key quality and join cardinality before feature use.
**Acceptable alternative:** exclude unverified provider features.
**Red flag:** infer wrongdoing from an aggregate.
**Scoring:** 0 accusation; 1 correlation caveat; 2 competing causes; 3 validation plan.
**Support:** Sections 5a, 5d, and F-04.

## Appendix A — Evidence-gated evaluation protocol
This appendix is an executable reading of the proposed validation plan. It documents what evidence must exist before a future claim; it does not assert that the evidence exists.
### E-01 — cohort freezing
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `claim-key and intake-version snapshot`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** the eligible list cannot change after policy assignment.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** a late correction becomes a separately linked superseding decision.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-02 — cutoff timing
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `availability timestamp for every field`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** each retained feature precedes the queue timestamp.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** unavailable fields are absent rather than imputed from the future.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-03 — identity linkage
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `claim-to-policy cardinality`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** one-to-many links retain both keys and effective dates.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** a current policy dimension cannot rewrite historic ownership.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-04 — missingness semantics
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `source absent versus feed failure`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** missingness reason is a separate categorical state.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** an upstream outage triggers fallback rather than learned imputation.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-05 — outcome taxonomy
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `disposition, recovery, reversal, and unresolved state`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** no disposition is silently mapped to a final negative.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** revised outcomes retain previous values and revision time.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-06 — capacity measurement
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `assigned, started, closed, reopened and aged files`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** K uses usable starts for the decision window.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** headcount is never substituted for queue capacity.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-07 — baseline replay
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `rules version and adjuster route`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** candidate and baseline see identical eligibility and capacity.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** baseline exceptions remain visible in comparison.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-08 — feature transformation
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `fit scope and transform timestamp`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** learned transform is fit only on training fold.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** forward data is never used to fill a training statistic.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-09 — calibration
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `score bins, maturity, and selection route`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** reported bins use named cohort and final outcomes.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** selected-only curve is labeled observation-conditioned.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-10 — reason codes
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `policy reason and operational reason`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** every referral and nonreferral has a reproducible reason.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** human prose does not replace structured audit fields.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-11 — assignment
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `policy arm and assignment timestamp`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** all eligible assigned cases remain in intent-to-treat analysis.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** fallback traffic is not removed after randomization.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-12 — service guardrail
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `intake-to-action and claimant-contact timing`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** time window matches policy cohort.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** median alone cannot hide tail delay.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-13 — slice review
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `approved slicing definition and count`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** sufficient-count uncertainty is shown.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** small cells are not overinterpreted or exposed.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-14 — override analysis
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `recommendation, execution, authority and reason`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** execution is distinguishable from recommendation.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** override is neither assumed error nor discarded.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-15 — retry safety
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `decision key and snapshot hash`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** same key/hash yields one active queue item.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** changed snapshot creates supersession rather than duplication.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-16 — queue reconciliation
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `decision, enqueue, start, closure, and cancellation events`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** every accepted referral has one terminal or active state.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** unacknowledged handoff is an incident, not a nonreferral.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-17 — value attribution
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `gross amount, recovery, cost, and counterfactual definition`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** currency components share cohort and window.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** score rank is not treated as recovered value.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-18 — maturity curve
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `cohort age and final-label rate`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** unresolved count is retained in every report.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** recent cohort is not compared as fully mature.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-19 — release compatibility
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `model, feature, policy, and fallback versions`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** all version pairs are declared compatible.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** rollback preserves the original decision evidence.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-20 — fallback test
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `dependency loss and routing outcome`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** loss of scoring does not silently drop eligible claims.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** fallback volume and reason reconcile to outage window.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-21 — retention boundary
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `fields, owners, permissions and deletion rule`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** replay access follows approved retention basis.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** a useful feature is excluded if its use is not approved.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-22 — appeal and correction
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `appeal event and revised decision`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** appeal does not overwrite original recommendation.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** analysis distinguishes initial and final states.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-23 — prospective stop rule
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `predeclared harm and operational threshold`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** owner can pause affected route promptly.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** post-hoc threshold changes are versioned separately.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
### E-24 — decision record
- **Owner:** claims operations owns workflow state; analytics owns measurement; model-risk/privacy owners approve their respective gates.
- **Required input:** `claim ID, time, versions, action, and evidence state`.
- **Method:** materialize the input as a versioned artifact before the relevant replay, policy run, or analysis.
- **Invariant:** a reviewer can reconstruct why an action occurred.
- **Failure interpretation:** a failed check can mean a data defect, process change, policy bug, or genuinely changed population; do not default to model drift.
- **Immediate response:** mark the affected metric or recommendation unavailable, use the preserved baseline route where authorized, and retain the failed artifact.
- **Competing-cause check:** compare source timestamps, join keys, policy version, and queue state before retraining or changing K.
- **Correction boundary:** repair the identified contract; do not retroactively overwrite decision-time values to improve the replay.
- **Verification oracle:** missing record narrows the conclusion to unknown.
- **Evidence boundary:** passing this control validates a data/process property, not model quality, causal benefit, fairness, or regulatory compliance.
- **Review record:** store checker version, evaluated cohort, failure count, exception rationale, owner decision, and recheck trigger.
- **Risk of mitigation:** fallback or pause can reduce queue yield, delay investigation, or alter claimant experience; measure it separately.
- **Status:** proposed; not executed in this illustrative case.
## Appendix B — Sensitivity interpretation traces
The calculations reuse the explicitly illustrative arithmetic in Section 6. They are not insurer outcomes or a substitute for a controlled evaluation.
### S-01 — maturity
- **Base condition:** 60 positives out of 200 referrals.
- **Changed condition:** 45 positives after strict maturity exclusion.
- **What remains fixed:** eligibility definition, target semantics, and decision-time data must be frozen before this comparison.
- **Interpretation:** do not call the difference model degradation until cohort age and revisions are reconciled.
- **Wrong inference to reject:** a changed reported metric proves that the ranker became better or worse without examining observation, policy, and workflow changes.
- **Required report fields:** assigned count, referred count, started count, mature count, unresolved count, cost components, and service guardrails.
- **Decision consequence:** if a material conclusion reverses under a plausible condition, retain uncertainty and defer expansion.
- **Audit trail:** preserve both configurations, versions, computation inputs, and the owner who selected the scenario.
- **Evidence:** DERIVED/ILLUSTRATIVE only.
### S-02 — capacity
- **Base condition:** K=200 and 60 constructed positives.
- **Changed condition:** K=120 with top 36 constructed positives.
- **What remains fixed:** eligibility definition, target semantics, and decision-time data must be frozen before this comparison.
- **Interpretation:** value, recall, and service workload must be recomputed together.
- **Wrong inference to reject:** a changed reported metric proves that the ranker became better or worse without examining observation, policy, and workflow changes.
- **Required report fields:** assigned count, referred count, started count, mature count, unresolved count, cost components, and service guardrails.
- **Decision consequence:** if a material conclusion reverses under a plausible condition, retain uncertainty and defer expansion.
- **Audit trail:** preserve both configurations, versions, computation inputs, and the owner who selected the scenario.
- **Evidence:** DERIVED/ILLUSTRATIVE only.
### S-03 — service cost
- **Base condition:** 200 referrals at $25 constructed cost.
- **Changed condition:** 200 referrals at $100 constructed cost.
- **What remains fixed:** eligibility definition, target semantics, and decision-time data must be frozen before this comparison.
- **Interpretation:** net-value scenario falls by $15,000; no observed result follows.
- **Wrong inference to reject:** a changed reported metric proves that the ranker became better or worse without examining observation, policy, and workflow changes.
- **Required report fields:** assigned count, referred count, started count, mature count, unresolved count, cost components, and service guardrails.
- **Decision consequence:** if a material conclusion reverses under a plausible condition, retain uncertainty and defer expansion.
- **Audit trail:** preserve both configurations, versions, computation inputs, and the owner who selected the scenario.
- **Evidence:** DERIVED/ILLUSTRATIVE only.
### S-04 — recoverable value
- **Base condition:** 60 referrals at $2,000 constructed value.
- **Changed condition:** 60 referrals at $1,000 constructed value.
- **What remains fixed:** eligibility definition, target semantics, and decision-time data must be frozen before this comparison.
- **Interpretation:** the scenario falls from $65,000 to $5,000 using the written formula.
- **Wrong inference to reject:** a changed reported metric proves that the ranker became better or worse without examining observation, policy, and workflow changes.
- **Required report fields:** assigned count, referred count, started count, mature count, unresolved count, cost components, and service guardrails.
- **Decision consequence:** if a material conclusion reverses under a plausible condition, retain uncertainty and defer expansion.
- **Audit trail:** preserve both configurations, versions, computation inputs, and the owner who selected the scenario.
- **Evidence:** DERIVED/ILLUSTRATIVE only.
### S-05 — feature delay
- **Base condition:** all fields available by cutoff.
- **Changed condition:** one high-impact field arrives after cutoff.
- **What remains fixed:** eligibility definition, target semantics, and decision-time data must be frozen before this comparison.
- **Interpretation:** remove it from replay rather than backdating availability.
- **Wrong inference to reject:** a changed reported metric proves that the ranker became better or worse without examining observation, policy, and workflow changes.
- **Required report fields:** assigned count, referred count, started count, mature count, unresolved count, cost components, and service guardrails.
- **Decision consequence:** if a material conclusion reverses under a plausible condition, retain uncertainty and defer expansion.
- **Audit trail:** preserve both configurations, versions, computation inputs, and the owner who selected the scenario.
- **Evidence:** DERIVED/ILLUSTRATIVE only.
### S-06 — policy override
- **Base condition:** all routed recommendations executed.
- **Changed condition:** 15% documented overrides.
- **What remains fixed:** eligibility definition, target semantics, and decision-time data must be frozen before this comparison.
- **Interpretation:** evaluate assignment, recommendation, execution, and outcome separately.
- **Wrong inference to reject:** a changed reported metric proves that the ranker became better or worse without examining observation, policy, and workflow changes.
- **Required report fields:** assigned count, referred count, started count, mature count, unresolved count, cost components, and service guardrails.
- **Decision consequence:** if a material conclusion reverses under a plausible condition, retain uncertainty and defer expansion.
- **Audit trail:** preserve both configurations, versions, computation inputs, and the owner who selected the scenario.
- **Evidence:** DERIVED/ILLUSTRATIVE only.

## 15. Quick Reference Card
| Field | Value |
|---|---|
| Decision | Rank eligible intake claims for human SIU referral; never automate coverage denial. |
| Baseline | Versioned rules plus adjuster referral; measure on same cohort and capacity. |
| Core quantities | precision@K=TP_K/K; recall@K=TP_K/P; capacity W=starts/day*days. |
| Critical condition | Features must exist at intake cutoff; labels need explicit maturity and observation state. |
| Primary failure | Selected or immature outcomes can make rank metrics look better than decision value. |
| Safe fallback | Approved existing routing with fallback reason logged and reconciliation retained. |
| Evidence/stage | Illustrative, not measured; proposed design; draft and unreviewed. |
| Review trigger | Intake, outcome, capacity, policy, or approved-data change. |
