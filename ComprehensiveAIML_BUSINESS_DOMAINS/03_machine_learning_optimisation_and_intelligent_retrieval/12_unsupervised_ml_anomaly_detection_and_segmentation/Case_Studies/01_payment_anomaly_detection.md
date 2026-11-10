---
case_id: "CASE-2026-PAY001"
case_title: "Payment anomaly detection: choosing a review-queue ranker under fixed investigator capacity"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "payments — card-not-present fraud risk operations"
primary_pattern: "anomaly-triage"
secondary_patterns: ["probability-calibration", "capacity-constrained-ranking"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Head of Payment Risk (decision owner); Fraud Operations Manager (desk owner)"
implementation_stage: "proposed"
source_index: "INDEX.md"
review_trigger: "Change in case-label maturity window, review capacity, average fraud loss, preventable fraction, score model version, or the scheme chargeback rule set that defines the label."
---

# Payment Anomaly Detection

## 1. The One-Line Framing

> Given **one business day of card-not-present authorisation cases, each scored by an unsupervised anomaly model and mapped to a calibrated probability**, decide **which 400 cases the 18-investigator desk should open first** for the payments risk owner, to maximise **prevented fraud loss net of investigation cost**, subject to **fixed investigator headcount, a 24-hour review SLA, and a 120-day label-maturity window**, compared with **the incumbent 34-rule weighted-count queue at the same 400-case capacity**.

Every number in this file is a constructed teaching value. The setting is a hypothetical acquirer; it is not a report about a named company, a deployed system, or a measured gain. The arithmetic is derived and hand-checked; the inputs are illustrative.

The decision is narrow on purpose. It is not "should we use machine learning for fraud", and it is not "which anomaly algorithm is best". It is a single capital-allocation choice inside a fixed operating budget: replace the ranking function that fills a queue of known size, or keep the incumbent ranking and spend the quarter on something else. Everything else in the case — feature engineering, calibration design, queueing, monitoring — exists only because it changes that choice.

**Task category and cadence.** Ranking and allocation, executed once per business day (queue build at 04:00 local), with continuous scoring of authorisation events feeding it. The decision recurs 365 times a year and the policy under evaluation would run for one quarter before a review gate.

**Primary success condition in plain language.** For the same 400 investigations per day, the desk confirms materially more genuine fraud cases, and the added confirmed cases are worth more than the investigation time they consume, without breaking the review SLA or weakening the hard blocks that never reach the queue.

**Patterns.** Primary canonical pattern `anomaly-triage` (an unlabelled unusualness score converted into a capacity-bounded review policy). Secondary `probability-calibration` (turning a rank into a probability so that a break-even rule can be applied) and `capacity-constrained-ranking` (the operating point is set by headcount, not by a statistical threshold).

**What is surprising here, and why the obvious framing misses it.** The intuitive version of this decision is "does the model rank fraud better than the rules?" and the intuitive way to check it is to look at the top of the queue, because that is where every demo plots its results. The measured answer is that the two rankers are almost identical at the top and diverge sharply at the bottom. Over a matured 90-business-day cohort, cumulative precision at the first 100 cases is 0.844 for the anomaly ranker against 0.820 for the rules — a 2.9% relative difference, which is inside day-to-day noise and would justify cancelling the project. At the actual operating point of 400 cases the comparison is 0.630 against 0.540, a 16.7% relative difference in confirmed cases. Over half of the entire daily gain, 18.6 of 36 additional confirmed cases, is produced in the last quarter of the queue, ranks 301–400, where the rules fall to 0.140 and the anomaly score still yields 0.326. A team that evaluates at `k = 100` will conclude that the rebuild is not worth doing, and it will be wrong about the only cut it actually operates.

The second surprise is about capacity. Because the marginal case at rank 400 is still worth `0.326 × 144 = 46.94` currency of expected prevented loss against `13.50` currency of review cost, the desk is materially under-sized — and that is true under *both* rankers, just far more true under the anomaly ranker. The marginal rule-queue case at rank 400 returns `0.140 × 144 − 13.50 = 6.66`, which is positive but four times thinner. So the correct sequencing is: fix the ranking at frozen headcount this quarter, then build a capacity case next cycle on better margins. Adding investigators first would spend the same money to review a worse list.

**Why the framing is not "improve the model".** Three constraints bind before model quality does. First, labels mature over 120 days, so the most recent quarter of behaviour is unlabelled at decision time and the newest fraud patterns are invisible to any supervised fit. Second, only cases that were reviewed ever receive an investigator-confirmed outcome, so the label set is selected by the incumbent policy that the new policy is supposed to replace. Third, the queue is exactly as long as the desk can clear, so any scoring change that produces "more alerts" without a capacity rule simply produces backlog. The case is therefore about a *policy*, not a *score*.

**Interview cue.** A reader should be able to state, in thirty seconds and without naming a library: the unit of analysis, the capacity, the comparator, the constraint, and the quantity that decides.

## 2. Context & Background

### 2.1 The product and the money flow

The setting is a mid-sized acquirer that processes card-not-present transactions for 41,000 merchant accounts in one region and one settlement currency. A cardholder places an order on a merchant's checkout page; the merchant's payment service provider submits an authorisation request through the acquirer; the issuer approves or declines; funds settle later; and if the cardholder later disputes the charge as fraudulent, a chargeback is raised through the scheme and the acquirer's merchant bears the loss under the merchant agreement, subject to the acquirer's own liability rules.

The acquirer's exposure is therefore indirect but real. It carries chargeback processing cost, scheme monitoring exposure, merchant attrition when dispute volume is high, and — in the segment where the acquirer has taken on fraud liability — direct loss. Risk operations exists to convert a large stream of authorisations into a small number of investigated cases, and to act before the loss is realised.

Two decisions live inside that workflow and they are not the same decision:

1. **Authorisation-time blocking.** Rules or a model decline or challenge a transaction while the customer waits. Errors are immediate and customer-visible.
2. **Post-authorisation review.** Cases that were approved are examined afterwards; the desk can block the credential, cancel unfulfilled orders, contact the merchant, or file a recovery claim. Errors are cheaper per case but slower.

This case concerns only the second decision. That boundary matters because the two decisions share features and compete for the same engineering attention, and because a common failure is to smuggle an authorisation-time decline into a review-queue change without the latency budget, the customer-friction budget, or the challenge flow that a decline requires. Section 8 states explicitly that v1 of this design does not decline anything.

### 2.2 The workflow the intervention changes

```text
cardholder checkout
      |
      v
[authorisation request] --(acquirer rules: hard blocks)--> decline / challenge
      |
      | approved
      v
[authorisation record + 3DS result + device + merchant metadata]  (event time t0)
      |
      v
[nightly / hourly feature assembly]  ---> per-case behavioural aggregates
      |
      v
[ranking policy]  ---> top-K cases, K = investigator capacity
      |
      v
[case store + investigator UI] ---> block / recover / close / escalate
      |
      v
[outcome: chargeback, issuer fraud report, no fraud]  (matures at t0 + up to 120 d)
      |
      v
[label store] ---> calibration and evaluation (120-day embargo)
```

Before the intervention the ranking policy is a weighted count of 34 rules. After the intervention the ranking policy is a calibrated anomaly score with a capacity cut. Everything else in the diagram is unchanged, which is what makes the comparison clean: same population, same desk, same action set, same label definition.

### 2.3 Stakeholders and who bears the consequence

| Role | Relationship to the decision | Consequence of being wrong |
|---|---|---|
| Head of Payment Risk | Owns the decision; can approve, defer, or reject | Accountable for fraud loss and for scheme monitoring position |
| Fraud Operations Manager | Runs the 18-investigator desk; owns the SLA | Absorbs backlog, overtime, and investigator attrition if volume is mis-set |
| Data science lead | Builds the scorer and the calibration | Owns the model, not the loss; the split is a known governance weakness |
| Platform engineering | Owns the event stream, feature store, and case store | Called at 03:00 when the queue build fails |
| Merchant relationship managers | Represent merchants whose accounts get flagged | Escalation path when a large merchant is repeatedly flagged |
| Payments compliance | Owns scheme rule interpretation and dispute evidence | Determines what the label actually means |

The decision owner is the Head of Payment Risk, not the data science lead. That is deliberate: the choice trades investigation capacity against loss, and only the risk owner holds both sides of that trade.

### 2.4 Baseline and its documented weaknesses

The incumbent policy is a weighted rule count. Thirty-four rules contribute points; the points are summed; cases are sorted descending; the top 400 are reviewed. Rules include velocity counts per card and per device, amount deviation from the card's own trailing median, mismatch flags (AVS, CVV, country), retry bursts after declines, and a set of merchant-category heuristics.

Recorded weaknesses of the baseline, stated as properties of the construction rather than as measured incident history:

- **Fixed weights.** Rule weights were set by analyst judgement and have not been re-derived against loss since they were introduced. A rule that fires on 40% of traffic contributes the same points as a rule that fires on 0.2%.
- **No interaction structure.** Two individually unremarkable signals — a new device on an established card, at an unusual hour, for a merchant category the card has never used — score the same as each alone plus an unrelated third.
- **Threshold brittleness.** Because points are integers, the top-400 boundary is a large tie group. Small rule-weight edits move hundreds of cases across the cut, which makes quarter-on-quarter comparison noisy even when nothing about fraud changed.
- **No probability.** Points are not a probability and cannot be compared to a cost. The desk cannot answer "is this case worth the eighteen minutes?" except by intuition.
- **Rule accumulation.** Each new fraud pattern adds a rule; the count has grown from 19 to 34 in three years; nobody has retired one.

These weaknesses are the reason a change is being considered at all. They are not evidence that the anomaly approach fixes them — Section 6 tests that — but they define what "better" has to mean.

### 2.5 Scale, timing, and boundaries

| Dimension | Value (illustrative) | Note |
|---|---|---|
| Eligible authorisations per business day | 1,200,000 | One region, one currency, card-not-present only |
| Cases per day after dedupe | 900,000 | Case key: `(card_token, merchant_account_id, business_date)` |
| Genuine fraud cases per day on a matured-label basis | 1,560 | Case-level prevalence 0.1733% |
| Cases reviewed per day | 400 | 18 investigators, 400 productive minutes each, 18 minutes per case |
| Investigators | 18 | 8-hour shift, 83% productive time |
| Mean service time per case | 18 min | Includes write-up; excludes escalation rework |
| Average gross loss per confirmed fraud case | 200 currency | Includes downstream authorisations on the same credential |
| Preventable fraction | 0.72 | Share of loss avoidable if the case is actioned within 24 h |
| Net value per prevented case | 144.00 currency | 200 × 0.72; the quantity that enters the break-even rule |
| Review cost per case | 13.50 currency | 0.3 h at 45 currency per staff-hour |
| Label maturity | 120 days | Chargeback plus issuer fraud report window assumed |
| Peak season | November–December | Fraud case volume assumed 40% above the annual mean |

The 120-day maturity figure and the scheme's reason-code semantics are inputs that the payments compliance team would have to confirm against the scheme documentation it holds. This case does not assert a specific scheme rule set, a specific chargeback timeline for a specific brand, or any regulatory conclusion; it uses 120 days as a stated scenario parameter and shows in Section 6 how the conclusion moves if the true window is 90 or 180 days.

### 2.6 Prior work and why a change is on the table

Two prior attempts are worth naming because they shaped the constraint set. A supervised gradient-boosted classifier was prototyped against investigator-confirmed outcomes and rejected before production for a reason that is structural rather than technical: the labels were produced by investigators who were themselves selecting what to investigate using the incumbent rules, so the classifier learned to reproduce the incumbent ranking and could not, in principle, find fraud the rules never surfaced. Separately, a vendor risk score was trialled; its rank correlation with the incumbent queue was high in the head and its calibration was not exposed, so the desk could not set an operating point from it and the trial ended without a decision.

Both attempts are recorded here as scenario context, not as documented organisational history. Their analytical content is what matters: a supervised model on selection-biased labels is mostly a relabelling of the incumbent policy, and a score without a calibrated probability cannot be turned into a capacity decision.

### 2.7 What the context forces

Three design consequences follow directly from the workflow and are carried into the rest of the case:

1. Any label used for training or evaluation must be **outcome-based** (chargeback, issuer fraud report) rather than **decision-based** (investigator closed as fraud). Otherwise the new policy is graded on the old policy's taste.
2. Any score used to set a cut must be **converted to a probability** on a matured cohort, because the cut is chosen by comparing value against cost, which are both in currency.
3. Any queue must be **sized by capacity first**, with the score deciding membership. A score-driven threshold that ignores capacity produces a queue the desk cannot clear, and an unclearable queue loses value through aging even when every individual decision is correct.

## 3. The Business Question

### 3.1 The question in the stakeholder's terms

The Head of Payment Risk states the question as a paraphrase of an internal planning discussion, not as a quotation from a real meeting:

> "We have eighteen investigators and we are not going to have twenty-eight next quarter. If I keep the same four hundred reviews a day but change what we look at, does the desk catch enough extra fraud to justify the rebuild, the calibration work, and the risk of a bad first quarter? And if it does, where does that leave the capacity conversation?"

### 3.2 Decision contract

| Field | Content |
|---|---|
| Question | At a fixed capacity of 400 investigations per day, should the review queue be filled by the calibrated anomaly ranking instead of the 34-rule weighted count for the next quarter? |
| Decision owner | Head of Payment Risk, with the Fraud Operations Manager as required signatory for SLA feasibility |
| Trigger and deadline | Quarterly planning cycle; the decision must be recorded before the capacity freeze date so that investigator rostering can be set |
| Available actions | (a) switch the ranker at constant capacity; (b) keep the rules and spend the quarter on rule maintenance; (c) switch and simultaneously add investigators; (d) defer pending a four-week shadow run |
| Value and downside | Upside: additional confirmed fraud cases and prevented loss per day. Downside: rebuild cost, investigator retraining, a possibly worse first month, and the organisational cost of a reversal |
| Objective | Maximise expected prevented fraud loss minus investigation cost per day, aggregated over a matured evaluation window, subject to guardrails |
| Hard constraints | 400 investigations per day sustained; 24-hour review SLA for 90% of cases; no change to authorisation-time decline logic; no use of fields unavailable at case-scoring time |
| Scope | One region, card-not-present, 900,000 cases per business day, one settlement currency, one quarter |
| Out of scope | Authorisation-time declines; 3-D Secure challenge policy; merchant onboarding; chargeback representment strategy; any change to scheme monitoring thresholds |

### 3.3 The proxy-versus-objective conflict

The business objective is prevented loss net of review cost. The quantity that can be measured cheaply and quickly is *confirmed cases per reviewed case* (yield). These can disagree in three specific ways, and each disagreement is a guardrail in Section 9:

- **Yield can rise while prevented loss falls.** If the ranking shifts toward high-count, low-value fraud patterns, yield rises and value falls. Guardrail: report value per confirmed case, not only counts.
- **Yield can rise while customer harm rises.** Aggressive case-level actions (blocking a credential, cancelling orders) create friction for legitimate customers. Guardrail: track the false-action rate among reviewed-and-cleared cases and the customer-contact volume.
- **Both can rise while the scheme position worsens.** Chargeback rate is monitored by the scheme against transaction volume; recovering a case does not always remove the dispute. Guardrail: report the dispute rate per 10,000 authorisations alongside the queue metrics.

The evaluation in Section 9 therefore reports yield as the primary metric but requires the value-per-case and friction guardrails to move in the right direction before the switch is approved. A yield gain that is bought entirely by deprioritising high-value cases would not pass.

### 3.4 What a good answer looks like

A defensible answer has four parts and no more:

1. A point estimate of the daily confirmed-case difference at 400 reviews, with an interval that accounts for finite review samples and label maturity.
2. A net value calculation in currency per day, with the review cost treated as constant across the two policies (because capacity is constant).
3. A statement about what the offline comparison cannot establish — specifically, that the prevented-loss fraction was assumed, not measured, and that no live action was taken.
4. A capacity recommendation with the correct sign: whether moving the cut deeper is value-positive or value-negative under each ranker, since that determines whether money should follow the switch.

### 3.5 What the decision is not

It is not a model-selection contest. The choice between isolation forest, robust Mahalanobis distance, and a local-density method is made in Section 5c on mechanism grounds and confirmed by a small sensitivity check; no outcome of the business decision depends on the winner. It is not a claim that unsupervised methods are sufficient for fraud detection in general; it is a claim that under a 120-day label embargo and a selection-biased reviewed cohort, an unlabelled unusualness score plus an externally-anchored calibration is the only construction that can be evaluated honestly in the available window.

## 4. Assumptions & Constraints

### 4.1 Assumption register

Each assumption carries a verification plan and a stated consequence if it fails. The three that can reverse the recommendation are marked **critical**.

| ID | Assumption | Type | Verification | Status | If wrong | Fallback |
|---|---|---|---|---|---|---|
| A-01 | Case-level fraud prevalence is stable within ±25% over the evaluation window | Data | Weekly matured-prevalence series over the previous four quarters | unverified | Break-even depth shifts; direction of the yield comparison may hold but magnitude is overstated | Re-derive the cut per month; report a prevalence-adjusted comparison |
| A-02 | **Critical** — Chargeback plus issuer-report labels capture at least 85% of genuine fraud cases within 120 days | Data | Reconcile a 2,000-case random audit against merchant-reported loss | unverified | Recall is understated for both policies roughly equally; the *ranking* comparison survives, the absolute value does not | Report recall only as a ratio between policies; label absolute recall `not measured` |
| A-03 | **Critical** — Preventable fraction is 0.72 for cases actioned within 24 hours | Business | Pilot with held-out merchants; compare blocked-value to confirmed-value | unverified | All currency figures scale linearly; the sign of the capacity conclusion is unaffected because break-even moves with the same factor | Present results as value-neutral: confirmed cases and marginal precision only |
| A-04 | Review cost is constant across ranker and equal to 13.50 currency per case | Business | Time-and-motion sample of 200 investigations per arm | unverified | If anomaly cases take longer to investigate, the net difference shrinks by the extra minutes | Recompute with measured handle time; the pass rule includes a handle-time guardrail |
| A-05 | Case dedupe key `(card_token, merchant_account_id, business_date)` is stable across the window | Data | Entity-resolution audit on 10,000 cases; measure token rotation rate | unverified | Capacity is consumed by duplicate cases; both policies affected, but the anomaly score is more sensitive because it clusters behaviour | Add a 72-hour cooldown key; re-measure duplicate rate |
| A-06 | All features are available at case-scoring time (04:00 the following day) | Data | Feature timestamp audit against the authorisation decision time | verified in design, not in data | Any post-decision field is leakage and inflates offline results | Remove the field; re-run the whole evaluation |
| A-07 | Isolation-forest score is stable across seeds within ±2% rank correlation | Model | Five seeds, Spearman correlation of the top-1,000 | unverified | Queue membership jitters at the cut; quarter-on-quarter comparison is noisy | Average five seeds or fix the production seed and freeze it |
| A-08 | Reviewers act on cases in score order and do not reorder by intuition | Human | Sample audit of case-open order versus queue order | unverified | Measured yield reflects reviewer skill, not ranker quality | Measure both; report the order-adherence rate as a covariate |
| A-09 | Merchant account metadata (category, tenure, prior dispute ratio) is current within 24 hours | Data | Freshness check on the merchant dimension table | unverified | Merchant features lag; new merchants look high-risk for a day | Use a point-in-time merchant snapshot with an explicit staleness flag |
| A-10 | The 24-hour SLA is measured from case creation, not from investigator assignment | Process | Read the case-store event log | unverified | SLA appears met while customers wait | Recompute the SLA from creation timestamp; restate the baseline |
| A-11 | No major merchant onboarding or offboarding during the quarter | Business | Merchant base census at start and end | unverified | Population shift contaminates the comparison | Stratify by merchant tenure; report both strata |
| A-12 | The scheme's chargeback reason codes distinguish fraud from non-fraud disputes | Data | Reason-code mapping reviewed by payments compliance | unverified | Fraud labels include service disputes, inflating apparent prevalence | Restrict to fraud reason codes; report the excluded share |

### 4.2 Constraint register

| ID | Constraint | Type | Source | If violated | Response |
|---|---|---|---|---|---|
| C-01 | 400 investigations per day sustained | Capacity | Investigator headcount and shift pattern | Queue ages; SLA breaches; value decays | Cut the queue at capacity, never at a fixed score |
| C-02 | 90% of cases reviewed within 24 hours | SLA | Operations charter | Merchant and cardholder harm; recovery window closes | Alert on queue age, not only queue length |
| C-03 | No change to authorisation-time decline logic in this quarter | Scope | Change-control boundary | Customer-visible regressions with no staged rollout | Score is advisory only in v1 |
| C-04 | No feature derived from any field written after the authorisation decision | Leakage | Feature contract | Offline results inflate and do not reproduce live | Automated timestamp audit in the feature pipeline |
| C-05 | No use of full card numbers or raw personal data outside the existing tokenised scope | Security | PCI-DSS scoping as currently operated | Scope expansion, audit cost | Use existing card tokens and merchant identifiers only |
| C-06 | Total additional spend this quarter ≤ 120,000 currency (engineering plus tooling) | Budget | Planning envelope | Project cannot be staffed | Reduce scope to calibration and queue change; defer refit automation |
| C-07 | Model must emit a human-readable reason set for every queued case | Governance | Dispute evidence and investigator workflow | Investigators cannot justify actions; appeal handling fails | Use per-feature deviation attribution from the reference population |
| C-08 | Rollback to the rule queue must be possible within one queue build (04:00 to 04:00) | Resilience | Operational risk policy | A bad ranking persists for a day or more | Version the queue as an immutable partition; switch by pointer |

### 4.3 The three assumptions that can reverse the decision

**A-03, the preventable fraction.** Every currency figure in this case is proportional to `ρ`. If the true preventable fraction is 0.45 rather than 0.72, the break-even probability rises from 0.09375 to 0.15, which moves the value-optimal depth for the anomaly ranker from 3,000 cases down to roughly 1,200. The *ranking* conclusion (anomaly beats rules at 400) survives because the comparison at fixed capacity is a comparison of confirmed counts multiplied by the same constant. The *capacity* conclusion does not survive unchanged, and the case must say so.

**A-02, label completeness.** If chargebacks capture only 60% of genuine fraud, both policies' precision estimates are biased downward by a similar factor *if* ascertainment is independent of the ranker. It is not obviously independent: a case that the new ranker surfaces may be investigated and refunded before a chargeback is ever raised, which removes the label. That mechanism would bias the new policy's measured precision downward relative to its true performance, and it is the main reason the evaluation uses outcomes only and forbids pre-emptive refunds during the measurement window.

**A-05, dedupe stability.** Card-token rotation is the quiet capacity leak. If 8% of cases are duplicates that should have been merged, the desk spends 32 investigations per day on repeat work, and a score that clusters behaviour will create more duplicates than a rule count does, because the whole point of the anomaly score is to react to a pattern rather than an event.

### 4.4 Assumptions the case explicitly refuses to make

- It does not assume the reviewed cohort is a random sample. It is not.
- It does not assume the anomaly score is a probability. It is a rank, and Section 5c converts it.
- It does not assume that offline yield predicts online prevented loss. The two are separated in Sections 9 and 11.
- It does not assume that a lower alert volume is better. The queue is capacity-bounded; alert volume above capacity is a policy defect, not a model victory.
- It does not assume that the 34 rules are useless. Section 6 quantifies exactly what they contribute that the anomaly score misses, and the design reserves capacity for it.

## 5. Data, Tools & Method
### 5a. Data Specification
| Field | Contract |
|---|---|
| Unit | one authorization case after deterministic deduplication |
| Inputs | pre-authorization amount, merchant, tokenized card history, device and velocity aggregates available before queue build |
| Label | confirmed fraud or chargeback matured after 120 days; unknown is not negative |
| Split | forward days grouped by token and merchant; no later dispute state in features |
### 5b. Tools & Stack
Proposed components are immutable daily input partitions, a feature-availability validator, an isolation-style anomaly ranker, a calibration table, deterministic queue builder, and investigation feedback store. They are design elements, not evidence of deployment.
### 5c. Method Summary
Generate a score from pre-cut behavioral deviations; calibrate only on a separate mature validation window; reserve rule-trigger capacity; rank the remaining eligible cases by expected preventable loss; export 400 stable ranked records.
### 5d. Provenance, quality, exclusions and leakage controls
A source column includes event time, ingestion time, token version and deletion status. If ingestion time is missing, exclude that field. Deduplicate before ranking, not after selecting the top 400, because post-ranking dedupe silently changes capacity.
### 5e. Reproducibility and method-selection ledger
| Artifact | Purpose | Limitation |
|---|---|---|
| ILL-PAY-01 | illustrative daily counts and loss assumptions | not a payment dataset |
| ILL-PAY-02 | hand calculation below | arithmetic only |
| OPEN-PAY-01 | source and policy evidence | unavailable; no scheme or company claim |
**Takeaway:** anomaly detection is a capacity-allocation workflow whose data contract is as important as the score.

## 6. Analysis Walkthrough
Validity: the following planning calculation assumes 400 nonduplicate reviewed cases, a 0.22 confirmed-fraud rate, average preventable loss of 750 currency units, and preventable fraction 0.72. All values are illustrative.
```text
FORMULA [PAY-01]: L = K * p * v * rho
WHERE: K reviewed cases [cases]; p confirmed fraud/case; v loss [currency/fraud]; rho preventable fraction [dimensionless].
CONSTRAINT: p, v and rho must describe the same decision population and action window.
NUMERICAL EXAMPLE: L=400*0.22*750*0.72=47,520 currency units/day.
INTERPRETATION: this is expected preventable-loss capacity, not observed savings.
```
Independent arithmetic: 400*0.22=88 expected confirmed fraud cases; 88*750=66,000 gross loss; 66,000*0.72=47,520. At rho=0, prevention is zero; at rho=1, the upper bound is 66,000.
| Stress | p | rho | Expected preventable loss | Consequence |
|---|---:|---:|---:|---|
| low yield | 0.10 | 0.45 | 13,500 | keep rule baseline and investigate calibration |
| base illustrative | 0.22 | 0.72 | 47,520 | measure queue policy |
| label missingness | unknown | 0.72 | unknown | do not convert observed confirmations into a rate |
**Takeaway:** capacity, label maturity, and actionability jointly determine whether a rank has decision value.

## 7. The Finding
The proposed anomaly queue is eligible only for a controlled comparison against the rule queue. The illustrative calculation sizes a potential benefit but does not establish prevented loss. The decision is to run a capacity-matched measurement with immutable daily snapshots and no authorization-time automation.
**Takeaway:** preserve the evidence boundary and choose the smallest next action that can falsify the proposed benefit.

## 8. System/Solution Design
```text
authorization events -> frozen pre-cut features -> availability/deduplication gate -> anomaly score
 -> calibration and rule reserve -> stable top-400 queue -> investigator decision -> matured outcome ledger
                              failure -> prior approved rule queue with date disclosure
```
The risk owner owns policy, fraud operations owns disposition quality, and engineering owns replayability. A score never declines a transaction in this design; an investigator retains operational authority.
**Takeaway:** preserve the evidence boundary and choose the smallest next action that can falsify the proposed benefit.

## 9. Evaluation & Validation
Evaluate precision@400 on matured labels, queue age, duplicate rate, investigator acceptance, and a capacity-matched rule comparator. The policy test randomizes or alternates comparable days only after checking merchant mix and seasonality; a simple before/after comparison is confounded. Do not interpret unresolved cases as legitimate negatives.
**Takeaway:** preserve the evidence boundary and choose the smallest next action that can falsify the proposed benefit.

## 10. Action Taken
No live action was taken in this illustrative case. The proposed action is a time-bounded shadow queue, an immutable replay, investigator training on reason sets, and a rollback rehearsal. Any change to decline behavior is outside scope and requires separate authority.
**Takeaway:** preserve the evidence boundary and choose the smallest next action that can falsify the proposed benefit.

## 11. Outcome & Results
No deployment, fraud reduction, chargeback improvement, or benchmark result is claimed. Verified outputs are only the shown arithmetic and deterministic properties of the proposed queue contract. The planned result is a measurement protocol, not a completed intervention.
**Takeaway:** preserve the evidence boundary and choose the smallest next action that can falsify the proposed benefit.

## 12. Failure Modes & What Went Wrong
| Symptom | Competing causes | Diagnostic | Recovery |
|---|---|---|---|
| Yield falls | drift, calibration error, label delay | frozen replay by day and label age | hold policy and recalibrate only after maturity |
| Queue duplicates | token rotation, retry events, broken key | count duplicates before and after rank | deterministic merge before selection |
| SLA misses | capacity loss, high investigation time | queue-age and staffing trace | reduce queue depth or add approved capacity |
| Score looks excellent | post-decision leakage, selective labels | ingestion-time audit and blind holdout | remove field and rebuild |
A mitigation that changes labels, refunds, or investigator behavior also changes evaluation; record it before comparing periods.
**Takeaway:** preserve the evidence boundary and choose the smallest next action that can falsify the proposed benefit.

## 13. Transferable Lesson
Unsupervised scores identify unusual patterns, not fraud probabilities or automated authority. Their utility emerges only after a decision-time feature contract, calibrated or otherwise measurable queue policy, capacity-aware baseline, and outcome process that does not select away the labels needed for review.
**Takeaway:** preserve the evidence boundary and choose the smallest next action that can falsify the proposed benefit.

## 14. Connections & Cross-References
### 14.1 Interview pack
**Q1. Why is an anomaly score not automatically a fraud probability?**
**Answer:** Restate the selection population, decision time, and capacity, then test the named assumption with a frozen replay or matched outcome window. **Reasoning:** a score, label, and action are distinct objects. **Follow-up:** specify the fallback queue. **Red flag:** calling an illustrative value an observed fraud result.
**Q2. Why deduplicate before capacity selection?**
**Answer:** Restate the selection population, decision time, and capacity, then test the named assumption with a frozen replay or matched outcome window. **Reasoning:** a score, label, and action are distinct objects. **Follow-up:** specify the fallback queue. **Red flag:** calling an illustrative value an observed fraud result.
**Q3. Why is a 120-day maturity window material?**
**Answer:** Restate the selection population, decision time, and capacity, then test the named assumption with a frozen replay or matched outcome window. **Reasoning:** a score, label, and action are distinct objects. **Follow-up:** specify the fallback queue. **Red flag:** calling an illustrative value an observed fraud result.
**Q4. What invalidates a before/after comparison?**
**Answer:** Restate the selection population, decision time, and capacity, then test the named assumption with a frozen replay or matched outcome window. **Reasoning:** a score, label, and action are distinct objects. **Follow-up:** specify the fallback queue. **Red flag:** calling an illustrative value an observed fraud result.
**Q5. When can the rule queue remain preferable?**
**Answer:** Restate the selection population, decision time, and capacity, then test the named assumption with a frozen replay or matched outcome window. **Reasoning:** a score, label, and action are distinct objects. **Follow-up:** specify the fallback queue. **Red flag:** calling an illustrative value an observed fraud result.
**Q6. What does fixed capacity change about thresholding?**
**Answer:** Restate the selection population, decision time, and capacity, then test the named assumption with a frozen replay or matched outcome window. **Reasoning:** a score, label, and action are distinct objects. **Follow-up:** specify the fallback queue. **Red flag:** calling an illustrative value an observed fraud result.
**Q7. How would token rotation bias the queue?**
**Answer:** Restate the selection population, decision time, and capacity, then test the named assumption with a frozen replay or matched outcome window. **Reasoning:** a score, label, and action are distinct objects. **Follow-up:** specify the fallback queue. **Red flag:** calling an illustrative value an observed fraud result.
**Q8. What is a plausible wrong fix for label delay?**
**Answer:** Restate the selection population, decision time, and capacity, then test the named assumption with a frozen replay or matched outcome window. **Reasoning:** a score, label, and action are distinct objects. **Follow-up:** specify the fallback queue. **Red flag:** calling an illustrative value an observed fraud result.
**Q9. Why is investigator feedback not ground truth by itself?**
**Answer:** Restate the selection population, decision time, and capacity, then test the named assumption with a frozen replay or matched outcome window. **Reasoning:** a score, label, and action are distinct objects. **Follow-up:** specify the fallback queue. **Red flag:** calling an illustrative value an observed fraud result.
**Q10. What changes if average loss varies by merchant?**
**Answer:** Restate the selection population, decision time, and capacity, then test the named assumption with a frozen replay or matched outcome window. **Reasoning:** a score, label, and action are distinct objects. **Follow-up:** specify the fallback queue. **Red flag:** calling an illustrative value an observed fraud result.
**Q11. Who may change a decline rule?**
**Answer:** Restate the selection population, decision time, and capacity, then test the named assumption with a frozen replay or matched outcome window. **Reasoning:** a score, label, and action are distinct objects. **Follow-up:** specify the fallback queue. **Red flag:** calling an illustrative value an observed fraud result.
**Q12. How is rollback verified?**
**Answer:** Restate the selection population, decision time, and capacity, then test the named assumption with a frozen replay or matched outcome window. **Reasoning:** a score, label, and action are distinct objects. **Follow-up:** specify the fallback queue. **Red flag:** calling an illustrative value an observed fraud result.
**Takeaway:** defend the queue as a decision policy with an explicit comparator, not a model contest.


### 14.2 Payment queue audit drills

These hypothetical controls make the queue contract inspectable. They are not operational logs or evidence of a live payment program.

#### Audit 01 — authorization event variant 1
**Contract:** use only fields recorded before the 04:00 queue cut.
**Record:** persist `event ingestion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** replay the event partition by ingestion time; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** late records are held for the next queue. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 02 — tokenized card history variant 1
**Contract:** do not join a future token rotation to a past event.
**Record:** persist `token version time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** diff identity maps by snapshot; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain the historical token mapping. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 03 — merchant profile variant 1
**Contract:** do not use a profile correction published after scoring.
**Record:** persist `profile publication time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare profile revision to cut; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** pin the prior revision. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 04 — device fingerprint variant 1
**Contract:** missing consent or delayed collection means no feature.
**Record:** persist `collection availability`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count unavailable device fields by day; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude rather than synthesize a value. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 05 — velocity aggregate variant 1
**Contract:** the aggregate must close before scoring.
**Record:** persist `window close time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** recompute from raw events; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rebuild the window on a frozen partition. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 06 — rule trigger variant 1
**Contract:** baseline and candidate need the same eligibility policy.
**Record:** persist `rule version`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** log rule version beside each case; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rerun both policies under one version. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 07 — deduplication key variant 1
**Contract:** a merge must precede ranking.
**Record:** persist `merge timestamp`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count top-400 collisions before export; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** dedupe then rank deterministically. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 08 — investigator disposition variant 1
**Contract:** review feedback is not a mature fraud label.
**Record:** persist `disposition completion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** separate disposition from chargeback outcome; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain both outcomes. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 09 — chargeback label variant 1
**Contract:** immature cases must remain unknown.
**Record:** persist `label maturity time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** plot rate by age bucket; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude immature denominator. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 10 — queue capacity variant 1
**Contract:** capacity changes require a new threshold.
**Record:** persist `roster approval time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare export size to approved roster; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** publish a new ranked cut. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 11 — authorization event variant 2
**Contract:** use only fields recorded before the 04:00 queue cut.
**Record:** persist `event ingestion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** replay the event partition by ingestion time; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** late records are held for the next queue. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 12 — tokenized card history variant 2
**Contract:** do not join a future token rotation to a past event.
**Record:** persist `token version time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** diff identity maps by snapshot; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain the historical token mapping. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 13 — merchant profile variant 2
**Contract:** do not use a profile correction published after scoring.
**Record:** persist `profile publication time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare profile revision to cut; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** pin the prior revision. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 14 — device fingerprint variant 2
**Contract:** missing consent or delayed collection means no feature.
**Record:** persist `collection availability`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count unavailable device fields by day; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude rather than synthesize a value. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 15 — velocity aggregate variant 2
**Contract:** the aggregate must close before scoring.
**Record:** persist `window close time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** recompute from raw events; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rebuild the window on a frozen partition. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 16 — rule trigger variant 2
**Contract:** baseline and candidate need the same eligibility policy.
**Record:** persist `rule version`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** log rule version beside each case; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rerun both policies under one version. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 17 — deduplication key variant 2
**Contract:** a merge must precede ranking.
**Record:** persist `merge timestamp`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count top-400 collisions before export; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** dedupe then rank deterministically. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 18 — investigator disposition variant 2
**Contract:** review feedback is not a mature fraud label.
**Record:** persist `disposition completion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** separate disposition from chargeback outcome; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain both outcomes. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 19 — chargeback label variant 2
**Contract:** immature cases must remain unknown.
**Record:** persist `label maturity time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** plot rate by age bucket; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude immature denominator. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 20 — queue capacity variant 2
**Contract:** capacity changes require a new threshold.
**Record:** persist `roster approval time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare export size to approved roster; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** publish a new ranked cut. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 21 — authorization event variant 3
**Contract:** use only fields recorded before the 04:00 queue cut.
**Record:** persist `event ingestion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** replay the event partition by ingestion time; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** late records are held for the next queue. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 22 — tokenized card history variant 3
**Contract:** do not join a future token rotation to a past event.
**Record:** persist `token version time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** diff identity maps by snapshot; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain the historical token mapping. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 23 — merchant profile variant 3
**Contract:** do not use a profile correction published after scoring.
**Record:** persist `profile publication time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare profile revision to cut; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** pin the prior revision. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 24 — device fingerprint variant 3
**Contract:** missing consent or delayed collection means no feature.
**Record:** persist `collection availability`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count unavailable device fields by day; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude rather than synthesize a value. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 25 — velocity aggregate variant 3
**Contract:** the aggregate must close before scoring.
**Record:** persist `window close time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** recompute from raw events; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rebuild the window on a frozen partition. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 26 — rule trigger variant 3
**Contract:** baseline and candidate need the same eligibility policy.
**Record:** persist `rule version`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** log rule version beside each case; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rerun both policies under one version. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 27 — deduplication key variant 3
**Contract:** a merge must precede ranking.
**Record:** persist `merge timestamp`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count top-400 collisions before export; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** dedupe then rank deterministically. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 28 — investigator disposition variant 3
**Contract:** review feedback is not a mature fraud label.
**Record:** persist `disposition completion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** separate disposition from chargeback outcome; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain both outcomes. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 29 — chargeback label variant 3
**Contract:** immature cases must remain unknown.
**Record:** persist `label maturity time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** plot rate by age bucket; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude immature denominator. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 30 — queue capacity variant 3
**Contract:** capacity changes require a new threshold.
**Record:** persist `roster approval time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare export size to approved roster; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** publish a new ranked cut. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 31 — authorization event variant 4
**Contract:** use only fields recorded before the 04:00 queue cut.
**Record:** persist `event ingestion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** replay the event partition by ingestion time; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** late records are held for the next queue. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 32 — tokenized card history variant 4
**Contract:** do not join a future token rotation to a past event.
**Record:** persist `token version time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** diff identity maps by snapshot; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain the historical token mapping. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 33 — merchant profile variant 4
**Contract:** do not use a profile correction published after scoring.
**Record:** persist `profile publication time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare profile revision to cut; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** pin the prior revision. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 34 — device fingerprint variant 4
**Contract:** missing consent or delayed collection means no feature.
**Record:** persist `collection availability`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count unavailable device fields by day; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude rather than synthesize a value. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 35 — velocity aggregate variant 4
**Contract:** the aggregate must close before scoring.
**Record:** persist `window close time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** recompute from raw events; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rebuild the window on a frozen partition. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 36 — rule trigger variant 4
**Contract:** baseline and candidate need the same eligibility policy.
**Record:** persist `rule version`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** log rule version beside each case; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rerun both policies under one version. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 37 — deduplication key variant 4
**Contract:** a merge must precede ranking.
**Record:** persist `merge timestamp`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count top-400 collisions before export; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** dedupe then rank deterministically. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 38 — investigator disposition variant 4
**Contract:** review feedback is not a mature fraud label.
**Record:** persist `disposition completion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** separate disposition from chargeback outcome; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain both outcomes. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 39 — chargeback label variant 4
**Contract:** immature cases must remain unknown.
**Record:** persist `label maturity time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** plot rate by age bucket; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude immature denominator. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 40 — queue capacity variant 4
**Contract:** capacity changes require a new threshold.
**Record:** persist `roster approval time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare export size to approved roster; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** publish a new ranked cut. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 41 — authorization event variant 5
**Contract:** use only fields recorded before the 04:00 queue cut.
**Record:** persist `event ingestion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** replay the event partition by ingestion time; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** late records are held for the next queue. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 42 — tokenized card history variant 5
**Contract:** do not join a future token rotation to a past event.
**Record:** persist `token version time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** diff identity maps by snapshot; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain the historical token mapping. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 43 — merchant profile variant 5
**Contract:** do not use a profile correction published after scoring.
**Record:** persist `profile publication time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare profile revision to cut; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** pin the prior revision. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 44 — device fingerprint variant 5
**Contract:** missing consent or delayed collection means no feature.
**Record:** persist `collection availability`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count unavailable device fields by day; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude rather than synthesize a value. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 45 — velocity aggregate variant 5
**Contract:** the aggregate must close before scoring.
**Record:** persist `window close time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** recompute from raw events; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rebuild the window on a frozen partition. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 46 — rule trigger variant 5
**Contract:** baseline and candidate need the same eligibility policy.
**Record:** persist `rule version`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** log rule version beside each case; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rerun both policies under one version. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 47 — deduplication key variant 5
**Contract:** a merge must precede ranking.
**Record:** persist `merge timestamp`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count top-400 collisions before export; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** dedupe then rank deterministically. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 48 — investigator disposition variant 5
**Contract:** review feedback is not a mature fraud label.
**Record:** persist `disposition completion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** separate disposition from chargeback outcome; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain both outcomes. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 49 — chargeback label variant 5
**Contract:** immature cases must remain unknown.
**Record:** persist `label maturity time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** plot rate by age bucket; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude immature denominator. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 50 — queue capacity variant 5
**Contract:** capacity changes require a new threshold.
**Record:** persist `roster approval time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare export size to approved roster; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** publish a new ranked cut. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 51 — authorization event variant 6
**Contract:** use only fields recorded before the 04:00 queue cut.
**Record:** persist `event ingestion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** replay the event partition by ingestion time; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** late records are held for the next queue. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 52 — tokenized card history variant 6
**Contract:** do not join a future token rotation to a past event.
**Record:** persist `token version time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** diff identity maps by snapshot; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain the historical token mapping. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 53 — merchant profile variant 6
**Contract:** do not use a profile correction published after scoring.
**Record:** persist `profile publication time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare profile revision to cut; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** pin the prior revision. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 54 — device fingerprint variant 6
**Contract:** missing consent or delayed collection means no feature.
**Record:** persist `collection availability`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count unavailable device fields by day; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude rather than synthesize a value. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 55 — velocity aggregate variant 6
**Contract:** the aggregate must close before scoring.
**Record:** persist `window close time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** recompute from raw events; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rebuild the window on a frozen partition. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 56 — rule trigger variant 6
**Contract:** baseline and candidate need the same eligibility policy.
**Record:** persist `rule version`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** log rule version beside each case; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rerun both policies under one version. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 57 — deduplication key variant 6
**Contract:** a merge must precede ranking.
**Record:** persist `merge timestamp`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count top-400 collisions before export; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** dedupe then rank deterministically. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 58 — investigator disposition variant 6
**Contract:** review feedback is not a mature fraud label.
**Record:** persist `disposition completion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** separate disposition from chargeback outcome; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain both outcomes. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 59 — chargeback label variant 6
**Contract:** immature cases must remain unknown.
**Record:** persist `label maturity time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** plot rate by age bucket; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude immature denominator. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 60 — queue capacity variant 6
**Contract:** capacity changes require a new threshold.
**Record:** persist `roster approval time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare export size to approved roster; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** publish a new ranked cut. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 61 — authorization event variant 7
**Contract:** use only fields recorded before the 04:00 queue cut.
**Record:** persist `event ingestion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** replay the event partition by ingestion time; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** late records are held for the next queue. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 62 — tokenized card history variant 7
**Contract:** do not join a future token rotation to a past event.
**Record:** persist `token version time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** diff identity maps by snapshot; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain the historical token mapping. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 63 — merchant profile variant 7
**Contract:** do not use a profile correction published after scoring.
**Record:** persist `profile publication time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare profile revision to cut; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** pin the prior revision. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 64 — device fingerprint variant 7
**Contract:** missing consent or delayed collection means no feature.
**Record:** persist `collection availability`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count unavailable device fields by day; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude rather than synthesize a value. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 65 — velocity aggregate variant 7
**Contract:** the aggregate must close before scoring.
**Record:** persist `window close time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** recompute from raw events; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rebuild the window on a frozen partition. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 66 — rule trigger variant 7
**Contract:** baseline and candidate need the same eligibility policy.
**Record:** persist `rule version`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** log rule version beside each case; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rerun both policies under one version. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 67 — deduplication key variant 7
**Contract:** a merge must precede ranking.
**Record:** persist `merge timestamp`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count top-400 collisions before export; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** dedupe then rank deterministically. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 68 — investigator disposition variant 7
**Contract:** review feedback is not a mature fraud label.
**Record:** persist `disposition completion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** separate disposition from chargeback outcome; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain both outcomes. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 69 — chargeback label variant 7
**Contract:** immature cases must remain unknown.
**Record:** persist `label maturity time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** plot rate by age bucket; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude immature denominator. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 70 — queue capacity variant 7
**Contract:** capacity changes require a new threshold.
**Record:** persist `roster approval time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare export size to approved roster; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** publish a new ranked cut. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 71 — authorization event variant 8
**Contract:** use only fields recorded before the 04:00 queue cut.
**Record:** persist `event ingestion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** replay the event partition by ingestion time; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** late records are held for the next queue. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 72 — tokenized card history variant 8
**Contract:** do not join a future token rotation to a past event.
**Record:** persist `token version time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** diff identity maps by snapshot; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain the historical token mapping. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 73 — merchant profile variant 8
**Contract:** do not use a profile correction published after scoring.
**Record:** persist `profile publication time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare profile revision to cut; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** pin the prior revision. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 74 — device fingerprint variant 8
**Contract:** missing consent or delayed collection means no feature.
**Record:** persist `collection availability`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count unavailable device fields by day; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude rather than synthesize a value. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 75 — velocity aggregate variant 8
**Contract:** the aggregate must close before scoring.
**Record:** persist `window close time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** recompute from raw events; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rebuild the window on a frozen partition. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 76 — rule trigger variant 8
**Contract:** baseline and candidate need the same eligibility policy.
**Record:** persist `rule version`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** log rule version beside each case; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** rerun both policies under one version. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 77 — deduplication key variant 8
**Contract:** a merge must precede ranking.
**Record:** persist `merge timestamp`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** count top-400 collisions before export; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** dedupe then rank deterministically. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 78 — investigator disposition variant 8
**Contract:** review feedback is not a mature fraud label.
**Record:** persist `disposition completion time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** separate disposition from chargeback outcome; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** retain both outcomes. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 79 — chargeback label variant 8
**Contract:** immature cases must remain unknown.
**Record:** persist `label maturity time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** plot rate by age bucket; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** exclude immature denominator. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
#### Audit 80 — queue capacity variant 8
**Contract:** capacity changes require a new threshold.
**Record:** persist `roster approval time`, input-partition ID, model version, calibration version, queue-build time, and stable case ID.
**Check:** compare export size to approved roster; report failed rows and their potential rank effect, not only a pass/fail aggregate.
**Failure interpretation:** a timestamp mismatch can create leakage, stale scoring, or a missing-value shift; distinguish these before changing the ranker.
**Recovery:** publish a new ranked cut. Do not silently remove cases after the top-400 cut, because that changes the capacity policy.
**Verification:** replay the same partition twice and require identical deduplicated IDs, ranks, reason sets, and selected count.
**Limit:** this verifies a data and state transition; it does not establish fraud prevalence, prevented loss, compliance, or investigator efficacy.
## 15. Quick Reference Card
| Field | Rule |
|---|---|
| Decision | rank at most 400 deduplicated cases for review |
| Baseline | 34-rule queue at the same capacity |
| Formula | `400*0.22*750*0.72=47,520` illustrative expected preventable loss/day |
| Gate | all features available before queue build; matured labels only |
| Failure recovery | halt export and use prior approved rule queue |
| Evidence / stage | illustrative / proposed / draft; no company outcome claimed |
