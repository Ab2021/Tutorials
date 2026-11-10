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

<!--CONTINUE-->
