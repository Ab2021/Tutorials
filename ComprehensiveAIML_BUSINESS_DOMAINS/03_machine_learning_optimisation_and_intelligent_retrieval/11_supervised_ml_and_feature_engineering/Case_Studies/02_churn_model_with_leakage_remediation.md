---
case_id: "CASE-M11-02"
case_title: "Churn leakage remediation: withdrawing a retention model that could not be scored on the day it decided"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "draft"
reviewed: null
content_status: "draft"
source_status: "OPEN"
case_type: "predictive-ml"
domain: "B2B SaaS retention and customer success"
primary_pattern: "temporal-validation"
secondary_patterns: ["leakage-control", "decision-threshold-under-capacity"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Retention analytics lead (data science) with customer-success operations as decision owner"
source_index: "INDEX.md"
review_trigger: "Billing-system snapshot semantics, CRM activity taxonomy, contact capacity, offer economics, contract-length mix, or churn definition change"
---

# Case study: Churn leakage remediation

## 1. The One-Line Framing

Given a monthly snapshot of mid-market subscription accounts and the signals that were genuinely recorded before that snapshot, decide which 900 accounts enter the human outreach queue for the retention team, to protect contribution-margin revenue that would otherwise be lost to voluntary cancellation, subject to a hard customer-success contact capacity of 900 accounts per nine-week cycle and a requirement that every feature be reproducible from the warehouse as of the decision instant, compared with the currently deployed rules-based health-score queue.

This is a **predictive-ML remediation case**, and it is **ILLUSTRATIVE**. No named company, account population, CRM system, model artifact, or business outcome is represented anywhere in this file. Every count, rate, and currency figure is a constructed teaching value chosen so the arithmetic is checkable by hand; the arithmetic is verified, the setting is not. Where a number would normally come from a vendor benchmark or an industry survey, that number is absent and the gap is stated instead.

The decision recurs on a nine-week cycle, twelve times a year, at the point when customer success plans its outreach calendar. The primary success condition in plain language is: the accounts that the queue selects should churn less often than the accounts the health score would have selected under the same contact capacity, and the claim must survive being recomputed on a forward holdout with only decision-time-available inputs.

The reusable pattern is `temporal-validation`; the two secondary patterns are `leakage-control` and `decision-threshold-under-capacity`. The decisive tradeoff, stated before any evidence, is this: the model that the team had built was the best-performing model anyone in the function had ever seen offline, and that was precisely the symptom. A 52% precision at the contact capacity, against a 6% base rate, is an 8.7x lift on a problem where the published range for voluntary B2B churn models is far lower. The obvious simpler framing -- "the model is great, ship it, and investigate the feature importances later" -- misses the part of the decision that actually matters, because the model cannot be scored for an account on the morning the queue is built. The features that made it great are, in the main, records of what happened after the decision it claims to make.

**Takeaway:** a performance number is evidence about a protocol, not about a system. When the number is implausibly good, the first hypothesis to test is that the protocol is wrong, not that the team is lucky.

## 2. Context & Background

### 2.1 The product and the commercial shape

The teaching setting is a mid-market business-to-business software subscription. Accounts are companies; each account holds one or more named seats; each account pays a monthly recurring charge under an annual or multi-year contract. The commercial team reports on recurring revenue, not on user counts, so every metric in this case is denominated in accounts or in monthly recurring revenue, never in logins.

The relevant behavioural unit is the **account-month**: an account observed at one monthly billing boundary. Accounts are not independent across months -- the same account appears up to twelve times in a year of history, with correlated outcomes -- and that dependence becomes a first-class problem in Sections 6 and 9. It is the single most common reason a retention model's uncertainty is overstated.

### 2.2 The workflow that the decision interrupts

Six weeks before a renewal date, the renewal operations team generates a renewal forecast. Four weeks before, customer success plans outreach. Two weeks before, the account executive receives an expansion or defence brief. If the customer intends to cancel, they submit notice through the account portal, and the notice date -- not the contract end date -- is the event that matters for measurement.

The retention queue is built at the *planning* boundary, roughly ten weeks before the average renewal date, because outreach has to be scheduled into human calendars and offers have to be approved by finance. That timing is not a technical detail. It means the model must produce a score at a moment when the current month's usage is only partially complete, when the renewal conversation has not yet happened, and when no cancellation signal exists for most accounts. Any feature that records the renewal conversation itself is, by construction, unavailable.

### 2.3 Stakeholders and their conflicting objectives

| Stakeholder | What they own | What they are measured on | What they would optimize if unopposed |
|---|---|---|---|
| Customer-success operations lead | The outreach calendar and the 900-account capacity | Contact coverage and post-contact retention | Number of accounts touched, because coverage is the metric they can control |
| Retention analytics lead | The queue's ordering and the evidence behind it | Whether queued accounts churn less than a comparable unqueued group | Offline discrimination, because it is the metric that moves fastest |
| Finance business partner | The discount and credit budget for save offers | Cost per retained dollar of recurring revenue | Approving no offers until the effect is proven |
| Renewal operations | The renewal forecast | Forecast accuracy against booked renewals | Excluding uncertain accounts from the forecast |
| Data platform owner | Snapshot semantics and feature availability | Freshness and correctness of the warehouse | Serving whatever is cheapest to materialise |
| Privacy and legal reviewer | Permitted use of support, sentiment, and usage data | Documented purpose and retention limits | Removing more fields than necessary |

The conflict is structural rather than personal. The analytics lead is rewarded for a discrimination number; the operations lead is rewarded for coverage; finance is rewarded for not spending. A queue that maximizes discrimination at fixed capacity can easily be a queue that consumes the entire discount budget on accounts that would have stayed anyway.

### 2.4 The baseline and its documented weakness

The deployed process is a **rules health score**. It is a weighted sum of five signals: login frequency in the trailing 30 days, open support-ticket count, days since the last executive sponsor meeting, contract days remaining, and whether the account has an active expansion opportunity. Each signal is banded, banded scores are summed, and the top 900 accounts by score enter the queue.

The health score has three documented weaknesses, all of which are properties of the design rather than of its parameters:

1. **It is not calibrated.** The sum has no probability interpretation, so it cannot be compared with the cost of a contact, and it cannot be used to decide whether 900 contacts is the right number rather than merely the available number.
2. **Its weights were set by workshop, not estimation.** The bands and weights were agreed in a half-day session with customer success and have not been revisited in three years, during which the product's pricing model changed twice.
3. **It double-counts correlated signals.** Open support tickets and days since sponsor meeting are strongly associated, so the effective weight on "account is in difficulty" is larger than any single band suggests. This is not fatal, but it makes the score's ordering sensitive to small changes in the ticket threshold.

The health score's measured performance is **not measured** in this case. No baseline precision is asserted. This is deliberate: the whole argument of the case is that an unmeasured baseline plus an inflated model number produces a confident wrong decision, and it would be inconsistent to fill that gap with a fabricated baseline.

### 2.5 Scale, timing, and boundaries

| Context field | Teaching value | Status |
|---|---|---|
| Account population | 60,000 mid-market accounts in scope | illustrative |
| Monthly recurring revenue, median in-scope account | 1,400 USD per month | illustrative |
| Contribution margin on recurring revenue | 78% | illustrative |
| Expected remaining tenure at snapshot | 11 months | illustrative |
| Churn base rate over the 90-day label window | 6.00% | illustrative |
| Contact capacity per nine-week cycle | 900 accounts | illustrative |
| Analyst time per contacted account | 34 minutes | illustrative |
| Geography | Single region, single currency, single billing system | illustrative |
| History available for modelling | 24 monthly snapshots | illustrative |
| Decision cadence | Every nine weeks, 5.2 cycles per year | derived: 52/9 |

The boundaries matter as much as the numbers. Accounts below a revenue floor are excluded because outreach is not economically justified; accounts already in formal notice are excluded because the outcome is determined; accounts in their first 30 days are excluded because they have no usage baseline. Those exclusions are applied *before* any modelling and are therefore not leakage -- but they change the population that the precision number refers to, and a precision computed on a different population is not comparable.

### 2.6 Prior attempts

Two prior attempts are relevant, and both are labelled as narrative rather than as measurement because no artifact survives in the teaching setting.

The first attempt added a gradient-boosted model on top of the existing health score as a re-ranker. It was abandoned after two cycles because the re-ranked queue was perceived by customer success as "the same accounts in a slightly different order." That perception was probably correct: a re-ranker trained on the same leaky features inherits the leak, and a leaky model's ordering is dominated by whichever accounts are closest to the extraction date rather than by which accounts are at risk.

The second attempt built a churn model on the previous twelve months of snapshots with a random train/test split. It reported an area under the receiver operating characteristic curve of 0.94 and a top-decile lift of eleven. Customer success ran one cycle on it, observed that the top of the queue was full of accounts that had already renewed, and stopped using it. The stop reason was recorded as "model does not reflect reality." That diagnosis was directionally right and mechanistically wrong, and Section 12 returns to it.

**Takeaway:** the failure mode that repeatedly kills retention models is not weak signal; it is that the model is answering "which accounts were at risk in the past, as seen from today" rather than "which accounts are at risk today, as seen from today."

## 3. The Business Question

### 3.1 The question as the decision owner put it

> "I have 900 outreach slots and twelve cycles a year. Your model says the top 900 accounts are twice as likely to leave as the health score's top 900. Before I rebuild my calendar around it, tell me whether those 900 accounts are the ones I can actually identify ten weeks before renewal -- and if they are not, tell me what the honest number is, because I would rather plan against a real 30% than a fictional 52%."

This is an **illustrative stakeholder question**, reconstructed for teaching. It is not a quotation from a real person, and the numbers in it are the teaching values used throughout.

### 3.2 The decision contract

| Field | Content |
|---|---|
| Question | Should the retention queue be ordered by the gradient-boosted churn model, by the existing rules health score, or by the remediated model; and at what contact capacity? |
| Decision owner | Customer-success operations lead, with the analytics lead as recommender and finance as budget approver |
| Trigger | The offline evaluation reported 52% precision at 900; the operations lead was asked to commit the next four cycles to the new queue |
| Deadline | Two weeks before the next outreach planning cycle, because calendar blocks are published then |
| Available actions | (a) keep the health score; (b) adopt the as-built model; (c) adopt a remediated model with a forward holdout; (d) adopt the remediated model with a randomized holdout group; (e) defer and run a measurement cycle first |
| Value at stake | Approximately 41 to 65 incremental saves per cycle, worth 0.50 to 0.78 million USD of contribution margin per cycle under the stated assumptions |
| Downside | A wrong queue consumes 510 analyst hours and 108,900 USD of contact and incentive cost per cycle, and displaces whatever else customer success would have done with that capacity |
| Objective | Maximize incremental retained contribution margin per cycle at fixed capacity, subject to a calibration and reproducibility gate |
| Hard constraints | Features must be reproducible from the warehouse as of the planning instant; the queue must be explainable to the account executive; contact capacity cannot exceed 900 per cycle |
| Scope | 60,000 mid-market accounts, one region, one billing system, twelve cycles per year |
| Out of scope | Pricing changes, product changes, involuntary churn from payment failure, expansion targeting, and individual-level treatment assignment |

### 3.3 Why the objective and the proxy can disagree

The objective is retained contribution margin. The proxy that the analytics team reported is precision at the contact capacity. These agree only under three conditions: the probability of saving a contacted account is the same for every true churner; the cost of a contact is the same for every account; and the accounts displaced from the queue have no value.

None of the three holds exactly. High-value accounts receive better save offers, so the save probability rises with revenue. Contacts cost roughly the same regardless of account size, so cost per contact is flat while value per contact is not. And the displaced accounts -- the 901st through, say, 1,400th -- have real but lower risk, so displacing them is not free.

The guardrail that keeps the proxy honest is therefore a **value-weighted** comparison: the queue's expected retained margin, not its raw precision, is the quantity that must improve. Precision remains the diagnostic because it is the quantity the model can be tuned on, but no threshold is justified by precision alone.

### 3.4 What an expensive false decision looks like

A false decision here has two distinct shapes, and they cost different things.

A **false positive** puts a healthy account in the queue. The cost is a contact: 34 analyst minutes plus the offer budget if an offer is made. If the account was never going to leave, the contact is pure cost, though it may produce a mildly positive side effect on relationship quality that this case does not attempt to measure.

A **false negative** leaves an at-risk account out of the queue. The cost is the lost contribution margin over the remaining life, minus whatever retention would have happened anyway. In this setting a false negative is roughly 12,012 USD of contribution margin against a 121 USD contact, a ratio near 99:1. That asymmetry is why recall matters even though capacity is the binding constraint, and it is why the queue must be evaluated at the operating point rather than at an arbitrary threshold.

**Takeaway:** the business question is not "how accurate is the model"; it is "at 900 contacts per cycle, does this ordering retain more contribution margin than the alternative ordering, and can that claim be reproduced on data the model has never influenced?"

## 4. Assumptions & Constraints

### 4.1 The assumption register

| ID | Assumption or constraint | Type | Verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Every feature can be reconstructed from the warehouse as of the planning instant, using only events recorded before it | Data | Point-in-time rebuild of the full feature table; compare to the as-built table row by row | **violated** in the as-built model; verified after remediation | The model cannot be scored at all, or is scored on different inputs than it was trained on | Drop or rebuild the offending feature | Any new feature; data platform owner |
| A-02 | The 90-day churn label is mature for every row used in evaluation | Data | Vintage-by-vintage maturity audit against the contract-end and notice-date records | **violated** for the two most recent vintages | Churn is understated by up to 28% and the model learns from falsely negative rows | Restrict to matured vintages; report the maturity curve | Monthly; analytics lead |
| A-03 | The churn base rate is stable enough that a 90-day historical rate transfers to the next cycle | Model | Compare month-over-month rates and a trailing-twelve-month band | unverified | Threshold and value estimates are miscalibrated | Recalibrate per cycle; widen the decision band | Base rate moves more than 1 pp; analytics lead |
| A-04 | The account is the right unit, and account identity is stable across the window | Data | Identity audit on the billing key; count of accounts with mid-window mergers or splits | unverified | Repeated rows and entity leakage; inflated evaluation | Aggregate to the billing account and split by entity | Any CRM migration; data platform owner |
| A-05 | Contact capacity is genuinely 900 and does not flex with queue quality | Business | Confirm against the published calendar and staffing plan | confirmed for the cycle, by assertion | Threshold is wrong in either direction | Recompute at the actual capacity | Staffing change; operations lead |
| A-06 | The save rate for contacted true churners is 13%, and 1% for contacted non-churners | Business | Randomized holdout within the queue | **not measured**; treated as a prior | Value estimate is wrong by a factor, not by a percentage | Run the 9:1 holdout in Section 9 | Holdout result; analytics lead |
| A-07 | A contacted account's outcome is unaffected by other contacted accounts | Model | Interference check: compare treated accounts sharing a customer-success pod | unverified | Individual-level effects are overstated | Cluster-randomize or aggregate to the pod | Pod-level design change; analytics lead |
| A-08 | The offer budget is 100 USD per contacted account on average | Business | Finance approval record per cycle | confirmed for the cycle | Cost per contact changes, moving the break-even save count | Recompute break-even | Finance review; finance partner |
| C-01 | No feature may be derived from the cancellation event, the notice record, or the post-decision account state | Constraint | Feature lineage review against the decision clock | enforced by design after remediation | Direct target leakage; the headline failure of this case | Remove the field | Every feature addition; analytics lead |
| C-02 | Every learned transformation must be fitted inside the evaluation fold, never on the pooled dataset | Constraint | Pipeline audit: refit per fold and diff the transformed values | enforced by design after remediation | Preprocessing leakage, visible as a collapse when refitted | Wrap transforms in a fold-scoped pipeline | Any pipeline refactor; analytics lead |
| C-03 | The evaluation split must be forward in time and grouped by account | Constraint | Split audit: maximum training date less than minimum test date; zero account overlap | enforced by design after remediation | Entity leakage and optimistic intervals | Rebuild the split | Any resampling change; analytics lead |
| C-04 | No protected or special-category attribute may enter the model, directly or as a proxy | Constraint | Feature review plus proxy test on the retained set | review needed | Legal and ethical exposure; subgroup harm | Drop the field and its proxies | Legal review; privacy reviewer |
| C-05 | The queue must be explainable: each account carries at most five reason codes | Constraint | Reason-code coverage check on the queue | review needed | Account executives ignore the queue, destroying adoption | Reduce to three codes | Adoption falls; operations lead |
| C-06 | Inference must complete within the planning window, which is one business day | Infrastructure | Timed rebuild on the production snapshot | not yet tested | Calendar misses the planning date | Precompute features nightly | Volume growth; data platform owner |

### 4.2 The three assumptions that can reverse the design

Of the fourteen entries above, three can reverse the recommendation rather than merely shift a number.

**A-01 is the first.** If a material share of the feature set cannot be reconstructed as of the planning instant, then the as-built model has no deployable form at all -- it is not a model with a leaky feature, it is a model whose input vector does not exist at decision time. The remediation is not parameter tuning; it is deleting the inputs and accepting the performance loss.

**A-06 is the second.** The entire value case rests on a 13% save rate among contacted true churners. If the true save rate is 4%, the remediated model retains 12.7 accounts per cycle instead of 41.4, and the net return collapses from 3.57 dollars per dollar spent to roughly 0.40 -- a loss. Section 6 shows the exact crossover, and Section 9 shows the experiment that would settle it.

**A-05 is the third.** If capacity is not actually 900 but flexes with queue quality, then the threshold is mis-specified in both directions: the model is being tuned to a constraint that does not bind, and the marginal-value calculation that justifies the 900th contact is invalid. Capacity is a business fact that has to be confirmed with the operations lead, not read off a spreadsheet.

### 4.3 What is a constraint and what is a choice

Two of the entries above are sometimes confused in practice. C-01 through C-03 are **not choices**. They are properties that any evaluation must have in order to be an evaluation of the deployed system. A team cannot decide to relax them for speed, in the way it can decide to accept a coarser feature or a simpler model, because relaxing them changes what the number means rather than how precise it is.

By contrast, the choice of model family, the choice between a point estimate and a calibrated probability, and the choice of whether to run a randomized holdout are genuine choices with cost and benefit. The rest of this file keeps that distinction explicit, because the most common failure in leakage remediation is treating a validity constraint as a tuning parameter.

```text
INVARIANT [CH-I1]: every row's feature vector is a function of events timestamped strictly
                   before the row's decision instant, and of nothing else.
WHERE: a row is one (account, planning cycle) pair; the decision instant is the planning
       boundary at which the queue is built.
CONSTRAINT: the invariant is a property of the data pipeline, not of the model; it must hold
            for training rows and scoring rows identically.
CHECK: for each feature, the maximum contributing event timestamp in the lineage record must
       be earlier than the row's decision instant; a single violation invalidates the field.
INTERPRETATION: a model that satisfies this on training data but not on scoring data is
                not merely degraded, it is undefined at serving time.
```

**Takeaway:** separate the constraints that define validity from the choices that trade off cost, and never let a scheduling pressure migrate an item from the first list to the second.


## 5. Data, Tools & Method
### 5a. Data Specification
| Field | Contract |
|---|---|
| Unit | one billing account at one nine-week planning instant; the billing-account key is stable within a snapshot |
| Window | 24 historical monthly snapshots; rows are eligible only after the full 90-day label window matures |
| Inputs | trailing usage aggregates, support counts, contract schedule, prior invoice state, and pre-decision CRM activity timestamps |
| Label | voluntary cancellation notice within 90 calendar days after the planning instant; payment failure is excluded |
| Exclusions | formal notices, first-30-day accounts, and accounts below the approved revenue floor, applied before scoring |
| Split | grouped account-level, forward-time train/validation/holdout; no row may move backward across its decision time |
The timestamp rule is operational: `feature_available_at <= decision_at`, while `label_observed_at > decision_at`. A source that stores an event after its business occurrence is still unavailable until its recorded availability time. This prevents an apparently harmless backfill from becoming look-ahead.
### 5b. Tools & Stack
The proposed stack is deliberately ordinary: versioned warehouse snapshots, SQL transformations with an availability-time column, a fold-scoped Python training pipeline, a calibrated gradient-boosted candidate, and a static queue export. No production deployment or run artifact is claimed. The stack is a design proposal, not evidence that any named organization uses it.
### 5c. Method Summary
1. Freeze an as-of snapshot for each planning instant and reject fields without an availability ledger.
2. Derive trailing-window features only from events whose recorded time precedes that instant.
3. Train baseline regularized logistic regression and a constrained boosted-tree candidate inside forward folds.
4. Calibrate probabilities on a validation vintage, rank eligible accounts, and select exactly 900 when capacity is fixed.
5. Attach reason codes, a feature-freshness flag, and a holdout assignment before exporting the queue.
| Candidate | Status | Why it is considered | Disqualifier |
|---|---|---|---|
| Rules health score | baseline, not measured | familiar and available | no probability or freshness audit |
| Logistic regression | proposed comparator | stable and inspectable | may miss interactions |
| Boosted trees plus calibration | proposed candidate | nonlinear interactions | only valid after point-in-time reconstruction |
| Unremediated model | rejected | high reported offline precision | cancellation and renewal fields are unavailable at scoring |
### 5d. Provenance, quality, exclusions and leakage controls
Every feature receives a lineage record with source table, event-time field, availability-time field, transformation version, maximum contributing timestamp, and owner. Missing availability time is a rejection condition, not an invitation to infer one. Transformations learn bins, imputations, and encodings from each training fold only.
### 5e. Reproducibility and method-selection ledger
| Artifact ID | Purpose | Accessibility | Limitation |
|---|---|---|---|
| ILL-CH-01 | constructed account-month table specification | included in prose only | no real rows exist |
| ILL-CH-02 | hand calculation in Section 6 | reproducible by arithmetic | not a model run |
| OPEN-CH-01 | source body and real workflow evidence | unavailable | no deployment or outcome may be inferred |
**Takeaway:** a reproducible prediction begins with a reproducible decision-time row, not with a reproducible estimator call.

## 6. Analysis Walkthrough
### 6.1 Leakage audit trace
| Step | Input | Operation | Output | Interpretation |
|---:|---|---|---|---|
| 1 | 60,000 eligible accounts | capacity ratio | 900 / 60,000 = 1.5% selected | this is the operating point, not a generic threshold |
| 2 | reported top-900 precision 52% | inspect feature availability | renewal-conversation and post-notice fields found | reported score is invalid for deployment |
| 3 | remediated forward holdout | apply availability ledger | hypothetical precision 18% | illustrative diagnostic number, not measured performance |
| 4 | 900 contacts | expected true churners | 900 × 0.18 = 162 accounts | capacity converts probability quality into workload |
| 5 | 13% illustrative save rate | expected incremental saves | 162 × 0.13 = 21.06 accounts | causal effect remains unmeasured without holdout |
Validity before use: the next calculation assumes the queue has 18% precision, the save rate applies only to true churners contacted, and contact effects are additive. Those are illustrative planning assumptions, not an efficacy claim.
```text
FORMULA [CH-01]: N_churn = K * p
WHERE: K=900 contacts [accounts]; p=0.18 precision [churners/contact].
CONSTRAINT: precision is for the same eligible population and decision time.
NUMERICAL EXAMPLE: N_churn=900*0.18=162 true churners.
INTERPRETATION: the queue creates a review workload hypothesis, not retained customers.
```
```text
FORMULA [CH-02]: N_save = K * p * s
WHERE: s=0.13 assumed incremental save probability [saves/true churner contacted].
CONSTRAINT: assumes no interference and a causal effect that has not been measured.
NUMERICAL EXAMPLE: N_save=900*0.18*0.13=21.06 expected saves.
INTERPRETATION: use this only to size a validation experiment.
```
The arithmetic check is independent: 162 × 0.13 = 21.06, and 21.06 / 900 = 0.0234 expected saves per contact. At p=0, expected saves are zero; at p=1, the model-implied upper bound is 117 saves under the same assumed effect. Neither boundary establishes a real intervention effect.
### 6.2 Sensitivity
| Stress | p | s | Expected saves | Decision consequence |
|---|---:|---:|---:|---|
| Low discrimination | 0.12 | 0.13 | 14.04 | require a smaller pilot or improved queue |
| Base illustrative | 0.18 | 0.13 | 21.06 | run randomized holdout before scale |
| Weak treatment | 0.18 | 0.04 | 6.48 | do not infer value from ranking alone |
| Capacity 450 | 0.18 | 0.13 | 10.53 | recalculate marginal queue cut-off |
The recommendation does not reverse on the precision number alone: it remains “validate with a holdout.” It reverses only if someone tries to describe these constructed results as an observed retention effect.
**Takeaway:** leakage remediation may lower an offline metric while improving the truthfulness of the decision that metric is permitted to support.

## 7. The Finding
The as-built model must not be adopted because some decisive inputs are unavailable at the planning instant. A point-in-time remediated pipeline is the only candidate eligible for a forward evaluation, but its illustrative 18% precision and calculated 21.06 expected saves are not outcomes. The owner should authorize a bounded randomized measurement cycle, not a broad replacement of the health-score queue. This conclusion is conditional on the timestamp ledger, the label-maturity audit, and the capacity remaining at 900 contacts.
**Takeaway:** reject an undeployable score even when it appears highly accurate; the honest next action is measurement, not confidence.

## 8. System/Solution Design
```text
[warehouse event] -> [availability-time validator] -> [as-of snapshot]
 -> [fold-scoped feature builder] -> [calibrated ranker] -> [capacity selector]
 -> [reason-code queue + holdout assignment] -> [customer-success workflow]
                                      |
                                      v
                            [lineage and freshness audit]
```
The data platform owner owns snapshot creation; analytics owns the model and calibration record; operations owns queue acceptance; finance approves offers; no component can silently substitute a late feature. A failed freshness check produces a queue hold and a baseline fallback, not a score with missing provenance.
At export, the selector sorts by stable `(risk_score descending, account_id ascending)` order. Deterministic ties prevent a capacity boundary from changing merely because a database returns equal values in a different order. The export carries snapshot ID, model version, threshold score, rank, reason codes, and holdout arm.
The recovery path is bounded: if the snapshot is late, use the last validated baseline queue only after operations accepts its date; if calibration is missing, do not substitute raw scores; if more than a defined fraction of rows fails lineage checks, stop and investigate rather than silently dropping accounts.
**Takeaway:** the serving contract is a timestamped queue with an auditable fallback, not an abstract model endpoint.

## 9. Evaluation & Validation
The validation protocol has two layers. First, a forward holdout verifies that every feature and transformation is available at scoring time and that ranking is stable. Second, a randomized queue holdout estimates the effect of contact versus no contact among accounts selected by the same policy. Neither AUC nor precision alone estimates the effect of outreach.
| Metric | Numerator | Denominator | Window | Guardrail | Status |
|---|---|---|---|---|---|
| Precision@900 | matured churn notices in selected queue | 900 selected accounts | 90 days | compare same eligibility rule | proposed |
| Calibration | observed churn rate by score bin | accounts in bin | 90 days | enough matured rows per bin | proposed |
| Incremental retention | treated retained minus control retained | randomized selected accounts | contract-defined horizon | pod interference check | not measured |
| Queue usability | accepted reason-code records | exported records | cycle day | rejection reasons logged | proposed |
The 9:1 randomization means 810 selected accounts receive outreach and 90 form the holdout, subject to ethics and operations approval. This is a design, not an executed experiment. The control result must be compared with a random arm from the same ranked population; comparing it with non-selected accounts confounds risk with treatment.
Boundary: if cluster interference within customer-success pods is material, randomize by pod or do not interpret individual-level differences causally. A larger sample cannot repair a unit-of-randomization error.
**Takeaway:** validation must test both scoring validity and intervention effect, because a correct ranker can still support an ineffective action.

## 10. Action Taken
In this illustrative design exercise, no production action was taken. The proposed decision is to withdraw the as-built model, keep the existing rules queue as a clearly labeled baseline, and authorize only the following preparatory actions: complete the availability ledger, rebuild matured historical snapshots, pre-register the holdout rule, and rehearse the stop/fallback path.
The decision owner retains authority to reject the queue. A model score does not authorize a discount, a contact, or a customer treatment. Finance and operations may impose a stricter capacity or offer rule, which requires the value calculation and ranking threshold to be recomputed.
**Takeaway:** “proposed” means a documented next step with an owner and stop condition, not an implied deployment.

## 11. Outcome & Results
No retained revenue, churn reduction, model lift, adoption rate, or production latency was measured for this teaching case. The only verified results are the arithmetic and the logical consequence that unavailable decision-time features invalidate a deployed-score claim. The projected counts in Section 6 are illustrative planning calculations and must not be promoted to outcomes.
| Claim ID | Claim | Evidence | Does not establish |
|---|---|---|---|
| CLM-CH-01 | 900 × 0.18 = 162 | verified arithmetic on illustrative inputs | real precision |
| CLM-CH-02 | 162 × 0.13 = 21.06 | verified arithmetic on illustrative inputs | causal save effect |
| CLM-CH-03 | late feature violates the serving contract | derived from timestamp definition | prevalence in any company |
**Takeaway:** a transparent not-measured result is more useful than a fabricated business-impact number.

## 12. Failure Modes & What Went Wrong
| Failure | Competing causes | Diagnostic | Recovery | Verification |
|---|---|---|---|---|
| Implausibly high precision | genuine signal; direct leakage; random split leakage | feature availability and split audit | remove unavailable fields; use forward grouped split | score can be built at each decision instant |
| Precision collapses forward | drift; label immaturity; lost leakage | vintage maturity curve and schema diff | mature labels; retrain only after root cause | stable metric definition across vintages |
| Queue cannot be acted on | no reason codes; stale CRM identity | export acceptance audit | stable ID mapping and capped explanations | operations accepts sampled records |
| Value estimate is inflated | noncausal save assumption; interference | randomized holdout and pod analysis | change randomization unit or narrow claim | pre-specified comparison passes |
| Capacity boundary thrashes | nondeterministic ties; duplicate accounts | repeat-sort and identity audit | stable secondary key and deduplication | identical snapshot gives identical queue |
An important recovery constraint is that deleting a feature is not proof that its proxy disappeared. The next audit checks proxy candidates and asks whether their availability timestamp is equally late.
**Takeaway:** diagnose the data-time contract before retuning the estimator; most apparent model failures are evaluation-contract failures.

## 13. Transferable Lesson
A prediction workflow is defined by the information available when a decision is made, the action the decision changes, and the counterfactual needed to evaluate that action. The same rule applies to fraud queues, clinical risk flags, ranking interventions, and staffing forecasts: a late label-like signal may make a retrospective model look excellent while making a prospective decision impossible. Calibration, stable capacity selection, and a human-owned fallback make the boundary testable rather than aspirational.
**Takeaway:** model quality is conditional on a temporal contract; without it, validation describes hindsight rather than decision support.

## 14. Connections & Cross-References
### 14.1 Interview pack
**Q1. Why is a random row split unsafe?**
**Answer:** Repeated accounts and future-derived features can appear on both sides, making the test easier than future scoring. Use a forward, entity-grouped split.
**Reasoning:** The answer follows from the decision-time and causal-evaluation contracts above.
**Follow-up / scoring:** Strong answers name a check and a boundary; red flag: treating an offline score as an intervention result.
**Q2. Why is 52% precision suspicious rather than impossible?**
**Answer:** It is not mathematically impossible, but a large gap from a 6% base rate demands a feature-availability and split audit before interpreting it as signal.
**Reasoning:** The answer follows from the decision-time and causal-evaluation contracts above.
**Follow-up / scoring:** Strong answers name a check and a boundary; red flag: treating an offline score as an intervention result.
**Q3. What does calibration add to ranking?**
**Answer:** A rank can prioritize work, while calibrated probabilities support expected-value and capacity decisions; calibration still requires matched population and time.
**Reasoning:** The answer follows from the decision-time and causal-evaluation contracts above.
**Follow-up / scoring:** Strong answers name a check and a boundary; red flag: treating an offline score as an intervention result.
**Q4. Why not select every account above 0.18?**
**Answer:** Capacity is fixed at 900. The operational rule is rank-and-cut, then measure the marginal value near the cut.
**Reasoning:** The answer follows from the decision-time and causal-evaluation contracts above.
**Follow-up / scoring:** Strong answers name a check and a boundary; red flag: treating an offline score as an intervention result.
**Q5. What is the most dangerous leakage source here?**
**Answer:** A renewal conversation or cancellation notice recorded after the planning instant directly exposes the future outcome and cannot exist at serving time.
**Reasoning:** The answer follows from the decision-time and causal-evaluation contracts above.
**Follow-up / scoring:** Strong answers name a check and a boundary; red flag: treating an offline score as an intervention result.
**Q6. How do label delays distort training?**
**Answer:** Immature 90-day windows mislabel future churners as negatives. Restrict evaluation to matured vintages or model censoring explicitly.
**Reasoning:** The answer follows from the decision-time and causal-evaluation contracts above.
**Follow-up / scoring:** Strong answers name a check and a boundary; red flag: treating an offline score as an intervention result.
**Q7. What invalidates the holdout comparison?**
**Answer:** Different eligibility rules, post-selection exclusions, or interference between arms destroy comparability; randomize before outreach and preserve assignments.
**Reasoning:** The answer follows from the decision-time and causal-evaluation contracts above.
**Follow-up / scoring:** Strong answers name a check and a boundary; red flag: treating an offline score as an intervention result.
**Q8. When would logistic regression win?**
**Answer:** When its forward performance is adequate and its simpler coefficients, stability, and explanation are more valuable than nonlinear lift.
**Reasoning:** The answer follows from the decision-time and causal-evaluation contracts above.
**Follow-up / scoring:** Strong answers name a check and a boundary; red flag: treating an offline score as an intervention result.
**Q9. What is a plausible wrong fix?**
**Answer:** Keeping the leaky field but masking it at serving time. That changes the feature distribution and preserves an invalid training/evaluation protocol.
**Reasoning:** The answer follows from the decision-time and causal-evaluation contracts above.
**Follow-up / scoring:** Strong answers name a check and a boundary; red flag: treating an offline score as an intervention result.
**Q10. What changes if capacity doubles?**
**Answer:** Recompute precision and expected value at 1,800 contacts; lower-ranked accounts may have materially lower risk, so linear extrapolation is not justified.
**Reasoning:** The answer follows from the decision-time and causal-evaluation contracts above.
**Follow-up / scoring:** Strong answers name a check and a boundary; red flag: treating an offline score as an intervention result.
**Q11. Who may approve a customer offer?**
**Answer:** The designated business and finance owners, not the model. A score prioritizes review but has no operational authority.
**Reasoning:** The answer follows from the decision-time and causal-evaluation contracts above.
**Follow-up / scoring:** Strong answers name a check and a boundary; red flag: treating an offline score as an intervention result.
**Q12. How is reproducibility checked?**
**Answer:** Rebuild a named as-of snapshot, rerun transformations with recorded versions, verify maximum source timestamps, and compare deterministic queue ranks.
**Reasoning:** The answer follows from the decision-time and causal-evaluation contracts above.
**Follow-up / scoring:** Strong answers name a check and a boundary; red flag: treating an offline score as an intervention result.
**Takeaway:** a defensible answer names the population, decision time, comparator, and evidence limit before proposing a model.


### 14.2 Audit drills for the decision-time contract

The following drills are design controls, not records of a live system. Each one translates the abstract temporal rule into an observable check so that an auditor can reject a persuasive but non-reproducible queue.

#### Drill 01 — usage events control variant 1
**Decision:** determine whether `usage events` may enter the score for a fixed planning instant.
**Invariant:** an event arriving after the planning cut must be excluded even if its business date looks earlier.
**Evidence needed:** persist the source snapshot identifier, `event timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare maximum contributing availability time with the cut; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rebuild the trailing window from frozen events. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a later backfill must not alter the historical score. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `usage events` is predictive, fair, permitted, or causally actionable.
#### Drill 02 — support tickets control variant 1
**Decision:** determine whether `support tickets` may enter the score for a fixed planning instant.
**Invariant:** a ticket closed after the cut cannot make an earlier snapshot look resolved.
**Evidence needed:** persist the source snapshot identifier, `ticket status timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit both created and state-change availability; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use only states materialized before scoring. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** status history must be append-safe. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `support tickets` is predictive, fair, permitted, or causally actionable.
#### Drill 03 — billing records control variant 1
**Decision:** determine whether `billing records` may enter the score for a fixed planning instant.
**Invariant:** a payment posted after the cut is not known at planning time.
**Evidence needed:** persist the source snapshot identifier, `invoice posting time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** diff as-of invoice snapshots; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** retain the latest pre-cut state only. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** late settlement corrections require a new score run. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `billing records` is predictive, fair, permitted, or causally actionable.
#### Drill 04 — CRM activities control variant 1
**Decision:** determine whether `CRM activities` may enter the score for a fixed planning instant.
**Invariant:** a logged call can be entered days after it occurred.
**Evidence needed:** persist the source snapshot identifier, `write timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare activity occurrence and write timestamps; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use recorded availability or exclude the field. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** manual backfill is treated as unavailable. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `CRM activities` is predictive, fair, permitted, or causally actionable.
#### Drill 05 — contract schedule control variant 1
**Decision:** determine whether `contract schedule` may enter the score for a fixed planning instant.
**Invariant:** future amendments cannot revise the old decision row.
**Evidence needed:** persist the source snapshot identifier, `effective-date version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** version contract schedules by extraction time; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** select the latest version visible at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** retroactive amendments are a lineage alert. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `contract schedule` is predictive, fair, permitted, or causally actionable.
#### Drill 06 — account identity control variant 1
**Decision:** determine whether `account identity` may enter the score for a fixed planning instant.
**Invariant:** a post-cut account merge must not consolidate historical entities.
**Evidence needed:** persist the source snapshot identifier, `merge timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit key mapping version; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** keep original billing key for the old snapshot. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** identity reconciliation is separately versioned. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `account identity` is predictive, fair, permitted, or causally actionable.
#### Drill 07 — reason codes control variant 1
**Decision:** determine whether `reason codes` may enter the score for a fixed planning instant.
**Invariant:** explanation text must correspond to the same feature snapshot as the rank.
**Evidence needed:** persist the source snapshot identifier, `generation version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** join rank, model and reason-code versions; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** regenerate all three together. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a missing reason code blocks export. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `reason codes` is predictive, fair, permitted, or causally actionable.
#### Drill 08 — label maturity control variant 1
**Decision:** determine whether `label maturity` may enter the score for a fixed planning instant.
**Invariant:** a cancellation notice not yet ingested cannot be a negative label.
**Evidence needed:** persist the source snapshot identifier, `notice ingestion time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** plot observed rate by label age; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** exclude immature vintages. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** maturity policy is recorded with each evaluation. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `label maturity` is predictive, fair, permitted, or causally actionable.
#### Drill 09 — capacity roster control variant 1
**Decision:** determine whether `capacity roster` may enter the score for a fixed planning instant.
**Invariant:** capacity changes after ranking do not retroactively validate the threshold.
**Evidence needed:** persist the source snapshot identifier, `calendar publication time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare queue size to approved roster; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rerank only under a newly versioned capacity. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** operators sign the final capacity value. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `capacity roster` is predictive, fair, permitted, or causally actionable.
#### Drill 10 — offer eligibility control variant 1
**Decision:** determine whether `offer eligibility` may enter the score for a fixed planning instant.
**Invariant:** later policy changes cannot be encoded as historical feature values.
**Evidence needed:** persist the source snapshot identifier, `finance approval time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** retain policy version in the action record; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** evaluate under the policy active at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** policy changes trigger a fresh decision review. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `offer eligibility` is predictive, fair, permitted, or causally actionable.
#### Drill 11 — usage events control variant 2
**Decision:** determine whether `usage events` may enter the score for a fixed planning instant.
**Invariant:** an event arriving after the planning cut must be excluded even if its business date looks earlier.
**Evidence needed:** persist the source snapshot identifier, `event timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare maximum contributing availability time with the cut; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rebuild the trailing window from frozen events. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a later backfill must not alter the historical score. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `usage events` is predictive, fair, permitted, or causally actionable.
#### Drill 12 — support tickets control variant 2
**Decision:** determine whether `support tickets` may enter the score for a fixed planning instant.
**Invariant:** a ticket closed after the cut cannot make an earlier snapshot look resolved.
**Evidence needed:** persist the source snapshot identifier, `ticket status timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit both created and state-change availability; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use only states materialized before scoring. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** status history must be append-safe. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `support tickets` is predictive, fair, permitted, or causally actionable.
#### Drill 13 — billing records control variant 2
**Decision:** determine whether `billing records` may enter the score for a fixed planning instant.
**Invariant:** a payment posted after the cut is not known at planning time.
**Evidence needed:** persist the source snapshot identifier, `invoice posting time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** diff as-of invoice snapshots; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** retain the latest pre-cut state only. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** late settlement corrections require a new score run. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `billing records` is predictive, fair, permitted, or causally actionable.
#### Drill 14 — CRM activities control variant 2
**Decision:** determine whether `CRM activities` may enter the score for a fixed planning instant.
**Invariant:** a logged call can be entered days after it occurred.
**Evidence needed:** persist the source snapshot identifier, `write timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare activity occurrence and write timestamps; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use recorded availability or exclude the field. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** manual backfill is treated as unavailable. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `CRM activities` is predictive, fair, permitted, or causally actionable.
#### Drill 15 — contract schedule control variant 2
**Decision:** determine whether `contract schedule` may enter the score for a fixed planning instant.
**Invariant:** future amendments cannot revise the old decision row.
**Evidence needed:** persist the source snapshot identifier, `effective-date version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** version contract schedules by extraction time; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** select the latest version visible at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** retroactive amendments are a lineage alert. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `contract schedule` is predictive, fair, permitted, or causally actionable.
#### Drill 16 — account identity control variant 2
**Decision:** determine whether `account identity` may enter the score for a fixed planning instant.
**Invariant:** a post-cut account merge must not consolidate historical entities.
**Evidence needed:** persist the source snapshot identifier, `merge timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit key mapping version; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** keep original billing key for the old snapshot. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** identity reconciliation is separately versioned. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `account identity` is predictive, fair, permitted, or causally actionable.
#### Drill 17 — reason codes control variant 2
**Decision:** determine whether `reason codes` may enter the score for a fixed planning instant.
**Invariant:** explanation text must correspond to the same feature snapshot as the rank.
**Evidence needed:** persist the source snapshot identifier, `generation version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** join rank, model and reason-code versions; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** regenerate all three together. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a missing reason code blocks export. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `reason codes` is predictive, fair, permitted, or causally actionable.
#### Drill 18 — label maturity control variant 2
**Decision:** determine whether `label maturity` may enter the score for a fixed planning instant.
**Invariant:** a cancellation notice not yet ingested cannot be a negative label.
**Evidence needed:** persist the source snapshot identifier, `notice ingestion time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** plot observed rate by label age; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** exclude immature vintages. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** maturity policy is recorded with each evaluation. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `label maturity` is predictive, fair, permitted, or causally actionable.
#### Drill 19 — capacity roster control variant 2
**Decision:** determine whether `capacity roster` may enter the score for a fixed planning instant.
**Invariant:** capacity changes after ranking do not retroactively validate the threshold.
**Evidence needed:** persist the source snapshot identifier, `calendar publication time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare queue size to approved roster; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rerank only under a newly versioned capacity. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** operators sign the final capacity value. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `capacity roster` is predictive, fair, permitted, or causally actionable.
#### Drill 20 — offer eligibility control variant 2
**Decision:** determine whether `offer eligibility` may enter the score for a fixed planning instant.
**Invariant:** later policy changes cannot be encoded as historical feature values.
**Evidence needed:** persist the source snapshot identifier, `finance approval time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** retain policy version in the action record; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** evaluate under the policy active at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** policy changes trigger a fresh decision review. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `offer eligibility` is predictive, fair, permitted, or causally actionable.
#### Drill 21 — usage events control variant 3
**Decision:** determine whether `usage events` may enter the score for a fixed planning instant.
**Invariant:** an event arriving after the planning cut must be excluded even if its business date looks earlier.
**Evidence needed:** persist the source snapshot identifier, `event timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare maximum contributing availability time with the cut; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rebuild the trailing window from frozen events. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a later backfill must not alter the historical score. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `usage events` is predictive, fair, permitted, or causally actionable.
#### Drill 22 — support tickets control variant 3
**Decision:** determine whether `support tickets` may enter the score for a fixed planning instant.
**Invariant:** a ticket closed after the cut cannot make an earlier snapshot look resolved.
**Evidence needed:** persist the source snapshot identifier, `ticket status timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit both created and state-change availability; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use only states materialized before scoring. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** status history must be append-safe. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `support tickets` is predictive, fair, permitted, or causally actionable.
#### Drill 23 — billing records control variant 3
**Decision:** determine whether `billing records` may enter the score for a fixed planning instant.
**Invariant:** a payment posted after the cut is not known at planning time.
**Evidence needed:** persist the source snapshot identifier, `invoice posting time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** diff as-of invoice snapshots; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** retain the latest pre-cut state only. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** late settlement corrections require a new score run. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `billing records` is predictive, fair, permitted, or causally actionable.
#### Drill 24 — CRM activities control variant 3
**Decision:** determine whether `CRM activities` may enter the score for a fixed planning instant.
**Invariant:** a logged call can be entered days after it occurred.
**Evidence needed:** persist the source snapshot identifier, `write timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare activity occurrence and write timestamps; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use recorded availability or exclude the field. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** manual backfill is treated as unavailable. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `CRM activities` is predictive, fair, permitted, or causally actionable.
#### Drill 25 — contract schedule control variant 3
**Decision:** determine whether `contract schedule` may enter the score for a fixed planning instant.
**Invariant:** future amendments cannot revise the old decision row.
**Evidence needed:** persist the source snapshot identifier, `effective-date version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** version contract schedules by extraction time; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** select the latest version visible at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** retroactive amendments are a lineage alert. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `contract schedule` is predictive, fair, permitted, or causally actionable.
#### Drill 26 — account identity control variant 3
**Decision:** determine whether `account identity` may enter the score for a fixed planning instant.
**Invariant:** a post-cut account merge must not consolidate historical entities.
**Evidence needed:** persist the source snapshot identifier, `merge timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit key mapping version; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** keep original billing key for the old snapshot. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** identity reconciliation is separately versioned. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `account identity` is predictive, fair, permitted, or causally actionable.
#### Drill 27 — reason codes control variant 3
**Decision:** determine whether `reason codes` may enter the score for a fixed planning instant.
**Invariant:** explanation text must correspond to the same feature snapshot as the rank.
**Evidence needed:** persist the source snapshot identifier, `generation version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** join rank, model and reason-code versions; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** regenerate all three together. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a missing reason code blocks export. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `reason codes` is predictive, fair, permitted, or causally actionable.
#### Drill 28 — label maturity control variant 3
**Decision:** determine whether `label maturity` may enter the score for a fixed planning instant.
**Invariant:** a cancellation notice not yet ingested cannot be a negative label.
**Evidence needed:** persist the source snapshot identifier, `notice ingestion time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** plot observed rate by label age; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** exclude immature vintages. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** maturity policy is recorded with each evaluation. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `label maturity` is predictive, fair, permitted, or causally actionable.
#### Drill 29 — capacity roster control variant 3
**Decision:** determine whether `capacity roster` may enter the score for a fixed planning instant.
**Invariant:** capacity changes after ranking do not retroactively validate the threshold.
**Evidence needed:** persist the source snapshot identifier, `calendar publication time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare queue size to approved roster; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rerank only under a newly versioned capacity. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** operators sign the final capacity value. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `capacity roster` is predictive, fair, permitted, or causally actionable.
#### Drill 30 — offer eligibility control variant 3
**Decision:** determine whether `offer eligibility` may enter the score for a fixed planning instant.
**Invariant:** later policy changes cannot be encoded as historical feature values.
**Evidence needed:** persist the source snapshot identifier, `finance approval time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** retain policy version in the action record; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** evaluate under the policy active at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** policy changes trigger a fresh decision review. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `offer eligibility` is predictive, fair, permitted, or causally actionable.
#### Drill 31 — usage events control variant 4
**Decision:** determine whether `usage events` may enter the score for a fixed planning instant.
**Invariant:** an event arriving after the planning cut must be excluded even if its business date looks earlier.
**Evidence needed:** persist the source snapshot identifier, `event timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare maximum contributing availability time with the cut; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rebuild the trailing window from frozen events. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a later backfill must not alter the historical score. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `usage events` is predictive, fair, permitted, or causally actionable.
#### Drill 32 — support tickets control variant 4
**Decision:** determine whether `support tickets` may enter the score for a fixed planning instant.
**Invariant:** a ticket closed after the cut cannot make an earlier snapshot look resolved.
**Evidence needed:** persist the source snapshot identifier, `ticket status timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit both created and state-change availability; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use only states materialized before scoring. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** status history must be append-safe. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `support tickets` is predictive, fair, permitted, or causally actionable.
#### Drill 33 — billing records control variant 4
**Decision:** determine whether `billing records` may enter the score for a fixed planning instant.
**Invariant:** a payment posted after the cut is not known at planning time.
**Evidence needed:** persist the source snapshot identifier, `invoice posting time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** diff as-of invoice snapshots; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** retain the latest pre-cut state only. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** late settlement corrections require a new score run. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `billing records` is predictive, fair, permitted, or causally actionable.
#### Drill 34 — CRM activities control variant 4
**Decision:** determine whether `CRM activities` may enter the score for a fixed planning instant.
**Invariant:** a logged call can be entered days after it occurred.
**Evidence needed:** persist the source snapshot identifier, `write timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare activity occurrence and write timestamps; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use recorded availability or exclude the field. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** manual backfill is treated as unavailable. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `CRM activities` is predictive, fair, permitted, or causally actionable.
#### Drill 35 — contract schedule control variant 4
**Decision:** determine whether `contract schedule` may enter the score for a fixed planning instant.
**Invariant:** future amendments cannot revise the old decision row.
**Evidence needed:** persist the source snapshot identifier, `effective-date version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** version contract schedules by extraction time; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** select the latest version visible at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** retroactive amendments are a lineage alert. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `contract schedule` is predictive, fair, permitted, or causally actionable.
#### Drill 36 — account identity control variant 4
**Decision:** determine whether `account identity` may enter the score for a fixed planning instant.
**Invariant:** a post-cut account merge must not consolidate historical entities.
**Evidence needed:** persist the source snapshot identifier, `merge timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit key mapping version; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** keep original billing key for the old snapshot. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** identity reconciliation is separately versioned. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `account identity` is predictive, fair, permitted, or causally actionable.
#### Drill 37 — reason codes control variant 4
**Decision:** determine whether `reason codes` may enter the score for a fixed planning instant.
**Invariant:** explanation text must correspond to the same feature snapshot as the rank.
**Evidence needed:** persist the source snapshot identifier, `generation version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** join rank, model and reason-code versions; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** regenerate all three together. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a missing reason code blocks export. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `reason codes` is predictive, fair, permitted, or causally actionable.
#### Drill 38 — label maturity control variant 4
**Decision:** determine whether `label maturity` may enter the score for a fixed planning instant.
**Invariant:** a cancellation notice not yet ingested cannot be a negative label.
**Evidence needed:** persist the source snapshot identifier, `notice ingestion time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** plot observed rate by label age; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** exclude immature vintages. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** maturity policy is recorded with each evaluation. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `label maturity` is predictive, fair, permitted, or causally actionable.
#### Drill 39 — capacity roster control variant 4
**Decision:** determine whether `capacity roster` may enter the score for a fixed planning instant.
**Invariant:** capacity changes after ranking do not retroactively validate the threshold.
**Evidence needed:** persist the source snapshot identifier, `calendar publication time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare queue size to approved roster; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rerank only under a newly versioned capacity. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** operators sign the final capacity value. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `capacity roster` is predictive, fair, permitted, or causally actionable.
#### Drill 40 — offer eligibility control variant 4
**Decision:** determine whether `offer eligibility` may enter the score for a fixed planning instant.
**Invariant:** later policy changes cannot be encoded as historical feature values.
**Evidence needed:** persist the source snapshot identifier, `finance approval time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** retain policy version in the action record; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** evaluate under the policy active at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** policy changes trigger a fresh decision review. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `offer eligibility` is predictive, fair, permitted, or causally actionable.
#### Drill 41 — usage events control variant 5
**Decision:** determine whether `usage events` may enter the score for a fixed planning instant.
**Invariant:** an event arriving after the planning cut must be excluded even if its business date looks earlier.
**Evidence needed:** persist the source snapshot identifier, `event timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare maximum contributing availability time with the cut; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rebuild the trailing window from frozen events. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a later backfill must not alter the historical score. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `usage events` is predictive, fair, permitted, or causally actionable.
#### Drill 42 — support tickets control variant 5
**Decision:** determine whether `support tickets` may enter the score for a fixed planning instant.
**Invariant:** a ticket closed after the cut cannot make an earlier snapshot look resolved.
**Evidence needed:** persist the source snapshot identifier, `ticket status timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit both created and state-change availability; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use only states materialized before scoring. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** status history must be append-safe. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `support tickets` is predictive, fair, permitted, or causally actionable.
#### Drill 43 — billing records control variant 5
**Decision:** determine whether `billing records` may enter the score for a fixed planning instant.
**Invariant:** a payment posted after the cut is not known at planning time.
**Evidence needed:** persist the source snapshot identifier, `invoice posting time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** diff as-of invoice snapshots; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** retain the latest pre-cut state only. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** late settlement corrections require a new score run. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `billing records` is predictive, fair, permitted, or causally actionable.
#### Drill 44 — CRM activities control variant 5
**Decision:** determine whether `CRM activities` may enter the score for a fixed planning instant.
**Invariant:** a logged call can be entered days after it occurred.
**Evidence needed:** persist the source snapshot identifier, `write timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare activity occurrence and write timestamps; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use recorded availability or exclude the field. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** manual backfill is treated as unavailable. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `CRM activities` is predictive, fair, permitted, or causally actionable.
#### Drill 45 — contract schedule control variant 5
**Decision:** determine whether `contract schedule` may enter the score for a fixed planning instant.
**Invariant:** future amendments cannot revise the old decision row.
**Evidence needed:** persist the source snapshot identifier, `effective-date version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** version contract schedules by extraction time; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** select the latest version visible at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** retroactive amendments are a lineage alert. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `contract schedule` is predictive, fair, permitted, or causally actionable.
#### Drill 46 — account identity control variant 5
**Decision:** determine whether `account identity` may enter the score for a fixed planning instant.
**Invariant:** a post-cut account merge must not consolidate historical entities.
**Evidence needed:** persist the source snapshot identifier, `merge timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit key mapping version; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** keep original billing key for the old snapshot. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** identity reconciliation is separately versioned. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `account identity` is predictive, fair, permitted, or causally actionable.
#### Drill 47 — reason codes control variant 5
**Decision:** determine whether `reason codes` may enter the score for a fixed planning instant.
**Invariant:** explanation text must correspond to the same feature snapshot as the rank.
**Evidence needed:** persist the source snapshot identifier, `generation version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** join rank, model and reason-code versions; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** regenerate all three together. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a missing reason code blocks export. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `reason codes` is predictive, fair, permitted, or causally actionable.
#### Drill 48 — label maturity control variant 5
**Decision:** determine whether `label maturity` may enter the score for a fixed planning instant.
**Invariant:** a cancellation notice not yet ingested cannot be a negative label.
**Evidence needed:** persist the source snapshot identifier, `notice ingestion time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** plot observed rate by label age; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** exclude immature vintages. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** maturity policy is recorded with each evaluation. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `label maturity` is predictive, fair, permitted, or causally actionable.
#### Drill 49 — capacity roster control variant 5
**Decision:** determine whether `capacity roster` may enter the score for a fixed planning instant.
**Invariant:** capacity changes after ranking do not retroactively validate the threshold.
**Evidence needed:** persist the source snapshot identifier, `calendar publication time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare queue size to approved roster; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rerank only under a newly versioned capacity. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** operators sign the final capacity value. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `capacity roster` is predictive, fair, permitted, or causally actionable.
#### Drill 50 — offer eligibility control variant 5
**Decision:** determine whether `offer eligibility` may enter the score for a fixed planning instant.
**Invariant:** later policy changes cannot be encoded as historical feature values.
**Evidence needed:** persist the source snapshot identifier, `finance approval time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** retain policy version in the action record; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** evaluate under the policy active at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** policy changes trigger a fresh decision review. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `offer eligibility` is predictive, fair, permitted, or causally actionable.
#### Drill 51 — usage events control variant 6
**Decision:** determine whether `usage events` may enter the score for a fixed planning instant.
**Invariant:** an event arriving after the planning cut must be excluded even if its business date looks earlier.
**Evidence needed:** persist the source snapshot identifier, `event timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare maximum contributing availability time with the cut; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rebuild the trailing window from frozen events. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a later backfill must not alter the historical score. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `usage events` is predictive, fair, permitted, or causally actionable.
#### Drill 52 — support tickets control variant 6
**Decision:** determine whether `support tickets` may enter the score for a fixed planning instant.
**Invariant:** a ticket closed after the cut cannot make an earlier snapshot look resolved.
**Evidence needed:** persist the source snapshot identifier, `ticket status timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit both created and state-change availability; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use only states materialized before scoring. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** status history must be append-safe. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `support tickets` is predictive, fair, permitted, or causally actionable.
#### Drill 53 — billing records control variant 6
**Decision:** determine whether `billing records` may enter the score for a fixed planning instant.
**Invariant:** a payment posted after the cut is not known at planning time.
**Evidence needed:** persist the source snapshot identifier, `invoice posting time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** diff as-of invoice snapshots; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** retain the latest pre-cut state only. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** late settlement corrections require a new score run. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `billing records` is predictive, fair, permitted, or causally actionable.
#### Drill 54 — CRM activities control variant 6
**Decision:** determine whether `CRM activities` may enter the score for a fixed planning instant.
**Invariant:** a logged call can be entered days after it occurred.
**Evidence needed:** persist the source snapshot identifier, `write timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare activity occurrence and write timestamps; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use recorded availability or exclude the field. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** manual backfill is treated as unavailable. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `CRM activities` is predictive, fair, permitted, or causally actionable.
#### Drill 55 — contract schedule control variant 6
**Decision:** determine whether `contract schedule` may enter the score for a fixed planning instant.
**Invariant:** future amendments cannot revise the old decision row.
**Evidence needed:** persist the source snapshot identifier, `effective-date version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** version contract schedules by extraction time; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** select the latest version visible at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** retroactive amendments are a lineage alert. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `contract schedule` is predictive, fair, permitted, or causally actionable.
#### Drill 56 — account identity control variant 6
**Decision:** determine whether `account identity` may enter the score for a fixed planning instant.
**Invariant:** a post-cut account merge must not consolidate historical entities.
**Evidence needed:** persist the source snapshot identifier, `merge timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit key mapping version; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** keep original billing key for the old snapshot. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** identity reconciliation is separately versioned. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `account identity` is predictive, fair, permitted, or causally actionable.
#### Drill 57 — reason codes control variant 6
**Decision:** determine whether `reason codes` may enter the score for a fixed planning instant.
**Invariant:** explanation text must correspond to the same feature snapshot as the rank.
**Evidence needed:** persist the source snapshot identifier, `generation version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** join rank, model and reason-code versions; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** regenerate all three together. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a missing reason code blocks export. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `reason codes` is predictive, fair, permitted, or causally actionable.
#### Drill 58 — label maturity control variant 6
**Decision:** determine whether `label maturity` may enter the score for a fixed planning instant.
**Invariant:** a cancellation notice not yet ingested cannot be a negative label.
**Evidence needed:** persist the source snapshot identifier, `notice ingestion time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** plot observed rate by label age; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** exclude immature vintages. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** maturity policy is recorded with each evaluation. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `label maturity` is predictive, fair, permitted, or causally actionable.
#### Drill 59 — capacity roster control variant 6
**Decision:** determine whether `capacity roster` may enter the score for a fixed planning instant.
**Invariant:** capacity changes after ranking do not retroactively validate the threshold.
**Evidence needed:** persist the source snapshot identifier, `calendar publication time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare queue size to approved roster; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rerank only under a newly versioned capacity. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** operators sign the final capacity value. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `capacity roster` is predictive, fair, permitted, or causally actionable.
#### Drill 60 — offer eligibility control variant 6
**Decision:** determine whether `offer eligibility` may enter the score for a fixed planning instant.
**Invariant:** later policy changes cannot be encoded as historical feature values.
**Evidence needed:** persist the source snapshot identifier, `finance approval time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** retain policy version in the action record; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** evaluate under the policy active at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** policy changes trigger a fresh decision review. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `offer eligibility` is predictive, fair, permitted, or causally actionable.
#### Drill 61 — usage events control variant 7
**Decision:** determine whether `usage events` may enter the score for a fixed planning instant.
**Invariant:** an event arriving after the planning cut must be excluded even if its business date looks earlier.
**Evidence needed:** persist the source snapshot identifier, `event timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare maximum contributing availability time with the cut; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rebuild the trailing window from frozen events. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a later backfill must not alter the historical score. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `usage events` is predictive, fair, permitted, or causally actionable.
#### Drill 62 — support tickets control variant 7
**Decision:** determine whether `support tickets` may enter the score for a fixed planning instant.
**Invariant:** a ticket closed after the cut cannot make an earlier snapshot look resolved.
**Evidence needed:** persist the source snapshot identifier, `ticket status timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit both created and state-change availability; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use only states materialized before scoring. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** status history must be append-safe. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `support tickets` is predictive, fair, permitted, or causally actionable.
#### Drill 63 — billing records control variant 7
**Decision:** determine whether `billing records` may enter the score for a fixed planning instant.
**Invariant:** a payment posted after the cut is not known at planning time.
**Evidence needed:** persist the source snapshot identifier, `invoice posting time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** diff as-of invoice snapshots; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** retain the latest pre-cut state only. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** late settlement corrections require a new score run. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `billing records` is predictive, fair, permitted, or causally actionable.
#### Drill 64 — CRM activities control variant 7
**Decision:** determine whether `CRM activities` may enter the score for a fixed planning instant.
**Invariant:** a logged call can be entered days after it occurred.
**Evidence needed:** persist the source snapshot identifier, `write timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare activity occurrence and write timestamps; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use recorded availability or exclude the field. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** manual backfill is treated as unavailable. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `CRM activities` is predictive, fair, permitted, or causally actionable.
#### Drill 65 — contract schedule control variant 7
**Decision:** determine whether `contract schedule` may enter the score for a fixed planning instant.
**Invariant:** future amendments cannot revise the old decision row.
**Evidence needed:** persist the source snapshot identifier, `effective-date version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** version contract schedules by extraction time; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** select the latest version visible at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** retroactive amendments are a lineage alert. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `contract schedule` is predictive, fair, permitted, or causally actionable.
#### Drill 66 — account identity control variant 7
**Decision:** determine whether `account identity` may enter the score for a fixed planning instant.
**Invariant:** a post-cut account merge must not consolidate historical entities.
**Evidence needed:** persist the source snapshot identifier, `merge timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit key mapping version; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** keep original billing key for the old snapshot. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** identity reconciliation is separately versioned. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `account identity` is predictive, fair, permitted, or causally actionable.
#### Drill 67 — reason codes control variant 7
**Decision:** determine whether `reason codes` may enter the score for a fixed planning instant.
**Invariant:** explanation text must correspond to the same feature snapshot as the rank.
**Evidence needed:** persist the source snapshot identifier, `generation version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** join rank, model and reason-code versions; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** regenerate all three together. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a missing reason code blocks export. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `reason codes` is predictive, fair, permitted, or causally actionable.
#### Drill 68 — label maturity control variant 7
**Decision:** determine whether `label maturity` may enter the score for a fixed planning instant.
**Invariant:** a cancellation notice not yet ingested cannot be a negative label.
**Evidence needed:** persist the source snapshot identifier, `notice ingestion time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** plot observed rate by label age; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** exclude immature vintages. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** maturity policy is recorded with each evaluation. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `label maturity` is predictive, fair, permitted, or causally actionable.
#### Drill 69 — capacity roster control variant 7
**Decision:** determine whether `capacity roster` may enter the score for a fixed planning instant.
**Invariant:** capacity changes after ranking do not retroactively validate the threshold.
**Evidence needed:** persist the source snapshot identifier, `calendar publication time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare queue size to approved roster; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rerank only under a newly versioned capacity. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** operators sign the final capacity value. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `capacity roster` is predictive, fair, permitted, or causally actionable.
#### Drill 70 — offer eligibility control variant 7
**Decision:** determine whether `offer eligibility` may enter the score for a fixed planning instant.
**Invariant:** later policy changes cannot be encoded as historical feature values.
**Evidence needed:** persist the source snapshot identifier, `finance approval time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** retain policy version in the action record; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** evaluate under the policy active at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** policy changes trigger a fresh decision review. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `offer eligibility` is predictive, fair, permitted, or causally actionable.
#### Drill 71 — usage events control variant 8
**Decision:** determine whether `usage events` may enter the score for a fixed planning instant.
**Invariant:** an event arriving after the planning cut must be excluded even if its business date looks earlier.
**Evidence needed:** persist the source snapshot identifier, `event timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare maximum contributing availability time with the cut; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rebuild the trailing window from frozen events. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a later backfill must not alter the historical score. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `usage events` is predictive, fair, permitted, or causally actionable.
#### Drill 72 — support tickets control variant 8
**Decision:** determine whether `support tickets` may enter the score for a fixed planning instant.
**Invariant:** a ticket closed after the cut cannot make an earlier snapshot look resolved.
**Evidence needed:** persist the source snapshot identifier, `ticket status timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit both created and state-change availability; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use only states materialized before scoring. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** status history must be append-safe. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `support tickets` is predictive, fair, permitted, or causally actionable.
#### Drill 73 — billing records control variant 8
**Decision:** determine whether `billing records` may enter the score for a fixed planning instant.
**Invariant:** a payment posted after the cut is not known at planning time.
**Evidence needed:** persist the source snapshot identifier, `invoice posting time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** diff as-of invoice snapshots; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** retain the latest pre-cut state only. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** late settlement corrections require a new score run. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `billing records` is predictive, fair, permitted, or causally actionable.
#### Drill 74 — CRM activities control variant 8
**Decision:** determine whether `CRM activities` may enter the score for a fixed planning instant.
**Invariant:** a logged call can be entered days after it occurred.
**Evidence needed:** persist the source snapshot identifier, `write timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare activity occurrence and write timestamps; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** use recorded availability or exclude the field. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** manual backfill is treated as unavailable. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `CRM activities` is predictive, fair, permitted, or causally actionable.
#### Drill 75 — contract schedule control variant 8
**Decision:** determine whether `contract schedule` may enter the score for a fixed planning instant.
**Invariant:** future amendments cannot revise the old decision row.
**Evidence needed:** persist the source snapshot identifier, `effective-date version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** version contract schedules by extraction time; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** select the latest version visible at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** retroactive amendments are a lineage alert. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `contract schedule` is predictive, fair, permitted, or causally actionable.
#### Drill 76 — account identity control variant 8
**Decision:** determine whether `account identity` may enter the score for a fixed planning instant.
**Invariant:** a post-cut account merge must not consolidate historical entities.
**Evidence needed:** persist the source snapshot identifier, `merge timestamp`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** audit key mapping version; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** keep original billing key for the old snapshot. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** identity reconciliation is separately versioned. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `account identity` is predictive, fair, permitted, or causally actionable.
#### Drill 77 — reason codes control variant 8
**Decision:** determine whether `reason codes` may enter the score for a fixed planning instant.
**Invariant:** explanation text must correspond to the same feature snapshot as the rank.
**Evidence needed:** persist the source snapshot identifier, `generation version`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** join rank, model and reason-code versions; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** regenerate all three together. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** a missing reason code blocks export. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `reason codes` is predictive, fair, permitted, or causally actionable.
#### Drill 78 — label maturity control variant 8
**Decision:** determine whether `label maturity` may enter the score for a fixed planning instant.
**Invariant:** a cancellation notice not yet ingested cannot be a negative label.
**Evidence needed:** persist the source snapshot identifier, `notice ingestion time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** plot observed rate by label age; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** exclude immature vintages. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** maturity policy is recorded with each evaluation. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `label maturity` is predictive, fair, permitted, or causally actionable.
#### Drill 79 — capacity roster control variant 8
**Decision:** determine whether `capacity roster` may enter the score for a fixed planning instant.
**Invariant:** capacity changes after ranking do not retroactively validate the threshold.
**Evidence needed:** persist the source snapshot identifier, `calendar publication time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** compare queue size to approved roster; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** rerank only under a newly versioned capacity. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** operators sign the final capacity value. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `capacity roster` is predictive, fair, permitted, or causally actionable.
#### Drill 80 — offer eligibility control variant 8
**Decision:** determine whether `offer eligibility` may enter the score for a fixed planning instant.
**Invariant:** later policy changes cannot be encoded as historical feature values.
**Evidence needed:** persist the source snapshot identifier, `finance approval time`, transformation revision, and the decision instant for every scored account.
**Diagnostic:** retain policy version in the action record; report the count of rows that violate strict pre-cut availability rather than averaging the timestamp error away.
**Recovery:** evaluate under the policy active at the cut. Do not impute a timestamp or retain the feature merely because it improves retrospective discrimination.
**Verification:** policy changes trigger a fresh decision review. A rerun on identical frozen inputs must produce the same eligibility flag, score order, and reason-code payload.
**Boundary:** this control establishes chronology, not that `offer eligibility` is predictive, fair, permitted, or causally actionable.
## 15. Quick Reference Card
| Field | Decision-ready statement |
|---|---|
| Decision | select a capacity-bounded outreach queue only from decision-time-available features |
| Owner | customer-success operations; analytics recommends; finance governs offers |
| Baseline | rules health score, outcome not measured |
| Validity gate | `feature_available_at <= decision_at`; forward grouped evaluation |
| Worked calculation | 900 × 0.18 × 0.13 = 21.06 illustrative expected saves |
| Main risk | leakage or immature labels makes an offline metric describe hindsight |
| Recovery | stop export, fall back with date disclosure, rebuild lineage and snapshot |
| Evidence / stage | illustrative / proposed / draft; no company outcome or deployment asserted |
| Recheck | source timestamps, label definition, capacity, save-effect evidence, and policy change |
