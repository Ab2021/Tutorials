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
source_index: "INDEX.md"
review_trigger: "Change in the case-label maturity window, review capacity, average fraud loss, preventable fraction, score model version, or the scheme chargeback rule set that defines the label."
---

# Payment Anomaly Detection

Every quantity in this file is a constructed teaching value for a hypothetical acquirer. No named
company, deployed system, or measured commercial result is asserted. The arithmetic is derived and
hand-checked here, so a reader can reproduce every intermediate step; the inputs themselves are
ILLUSTRATIVE and are chosen to make the trade-off visible rather than to represent a market.

The file also states, in sections 9 and 12, which checks were executed in this teaching design and
which are specified but not executed. Where a number is an extrapolation beyond a fitted range,
the text says so at the point where it is used rather than in a footnote.

## 1. The One-Line Framing

> Given one business day of card-not-present authorisation cases, each carrying an unsupervised
> anomaly score that has been mapped to a calibrated fraud probability, decide which 400 cases the
> 18-investigator desk opens first, so that the payments risk owner maximises prevented fraud loss
> net of investigation cost and customer friction, subject to fixed headcount, a 24-hour review
> service level, and a 120-day label-maturity embargo, compared with the incumbent 34-rule
> weighted-count queue at the identical 400-case capacity.

The decision is deliberately narrow. It is not "should we use machine learning for fraud", and it
is not "which anomaly algorithm wins a benchmark". It is one allocation choice inside a fixed
operating budget: replace the function that sorts the work queue, or keep the incumbent sort and
spend the quarter elsewhere.

**Decision contract.** The framing is complete only when five fields are pinned, because each one
moves the answer.

| Field | Value in this case | Why it changes the answer |
|---|---|---|
| Unit of analysis | one case = card token × merchant account × business date | card-level units double-count a single fraud ring |
| Action set | open, block credential, cancel unfulfilled order, recover, close | a wider action set raises the value of a true positive |
| Capacity | 400 cases reviewed per business day | precision is a function of how deep the queue is read |
| Comparator | 34-rule weighted count at the same 400 | a bigger queue would confound ranker with effort |
| Objective | prevented loss net of review cost and friction | gross caught loss rewards flagging everything |

**Cadence and recurrence.** The queue is rebuilt once per business day at 04:00 local time over
the preceding 24 hours of approved authorisations. Scoring runs continuously on the event stream,
but the *decision* to which this case attaches recurs 365 times a year, and the policy version
under evaluation would run for one quarter before a review gate. That asymmetry matters: a single
day's queue is cheap to get wrong and cheap to correct, while a quarter of a wrong policy is
expensive and invisible inside daily noise.

**Plain-language success condition.** For the same 400 investigations per day, the desk confirms
materially more genuine fraud cases, the extra confirmed cases are worth more than the
investigation time and customer friction they consume, the 24-hour service level holds, and the
hard authorisation-time blocks that never reach the queue are not weakened.

**Primary canonical pattern.** `anomaly-triage`: an unlabelled unusualness score converted into a
capacity-bounded review policy, where the binding resource is human attention rather than a
statistical threshold.

**Secondary patterns.** `probability-calibration`, because a rank cannot be compared to a cost
until it is a probability; and `capacity-constrained-ranking`, because the operating point is set
by headcount rather than chosen on a curve.

**What the obvious framing misses.** The intuitive test is "does the model rank fraud better than
the rules", and the intuitive place to look is the very top of the queue, because that is where
every demonstration plots its result. In this constructed cohort the two rankers are nearly tied at
the top and diverge at the depth the desk actually reads:

| Queue depth | Anomaly ranker precision | Rule ranker precision | Absolute gap | Relative gap |
|---:|---:|---:|---:|---:|
| 100 cases | 0.844 | 0.820 | 0.024 | 2.9% |
| 400 cases | 0.630 | 0.540 | 0.090 | 16.7% |

At depth 100 the difference is 2.4 percentage points, which is inside the binomial noise of a
single day and would justify cancelling the project. At depth 400 — the depth the desk actually
reads — the difference is 9.0 points. A pilot that measured at depth 50 or 100 would have measured
the one region of the curve where the intervention does nothing.

**Why this is a policy question and not a model question.** Three constraints bind before model
quality does. Labels mature over 120 days, so the most recent quarter of behaviour is unlabelled
at decision time and the newest fraud patterns are invisible to any supervised fit. Only reviewed
cases ever receive an investigator-confirmed outcome, so the label set is selected by the very
policy under challenge. And the queue is exactly as long as the desk can clear, so any scoring
change that produces "more alerts" without a capacity rule produces backlog instead of value.

**What a reader should be able to say in thirty seconds.** The unit of analysis, the capacity, the
comparator, the constraints, and the one quantity that decides — which here is precision at rank
400 multiplied by preventable loss, compared against the fully loaded cost of a review.

## 2. Context & Background

### 2.1 The money flow

The setting is a mid-sized acquirer processing card-not-present transactions for 41,000 merchant
accounts in one region and one settlement currency, EUR. A cardholder places an order; the
merchant's payment service provider submits an authorisation request through the acquirer; the
issuer approves or declines; funds settle; and if the cardholder later disputes the charge as
fraudulent, a chargeback is raised through the scheme.

The acquirer's exposure is indirect but real. It carries chargeback processing cost, scheme
monitoring exposure, merchant attrition when dispute volume in a merchant's account is high, and —
in the liability-shifted segment — direct loss. Risk operations exists to convert a large stream
of authorisations into a small number of investigated cases, and to act before loss is realised.

Two decisions live inside that workflow, and conflating them is the most common scope error in
this domain:

1. **Authorisation-time blocking.** Rules or a model decline or challenge a transaction while the
   customer waits at the checkout. Errors are immediate, customer-visible, and expensive.
2. **Post-authorisation review.** Already-approved cases are examined afterwards; the desk can
   block the credential, cancel unfulfilled orders, contact the merchant, or file a recovery
   claim. Errors cost investigation time rather than a lost sale.

This case concerns only the second decision. The boundary is stated in section 8 as a design
constraint: version 1 of the design declines nothing at authorisation time.

### 2.2 The workflow the intervention changes

```text
checkout
   |
   v
authorisation request --[acquirer hard blocks: sanctions, stolen BIN lists, velocity kill]--> decline
   |
   | approved
   v
authorisation record: amount, currency, MCC, 3DS result, device, IP, email domain   (event time t0)
   |
   v
hourly feature assembly: per-card, per-device, per-merchant aggregates
   |
   v
ranking policy  -->  top-K queue, K = desk capacity = 400
   |
   v
case store + investigator UI  -->  block / cancel / recover / close / escalate
   |
   v
outcome: chargeback filed, issuer fraud report, or no fraud   (matures at t0 + 1..120 days)
   |
   v
label store  -->  calibration and evaluation, with a 120-day embargo
```

Before the intervention the sort key is a weighted count of 34 rules. After it, the sort key is a
calibrated probability with a capacity cut. Every other box in the diagram is unchanged, which is
what makes the comparison clean: same population, same desk, same action set, same label
definition, same day boundary.

### 2.3 Stakeholders and who absorbs the error

| Role | Relationship to the decision | Consequence of being wrong |
|---|---|---|
| Head of Payment Risk | Owns the decision; approves, defers, or rejects | Accountable for fraud loss and scheme monitoring position |
| Fraud Operations Manager | Runs the 18-investigator desk and the service level | Absorbs backlog, overtime, and investigator attrition |
| Data science lead | Builds the scorer and the calibration | Owns the model but not the loss, a known governance gap |
| Platform engineering | Owns event stream, feature store, case store | Paged at 03:00 when the queue build fails |
| Merchant relationship managers | Represent flagged merchants | Escalation when a large merchant is repeatedly flagged |
| Payments compliance | Interprets scheme rules and dispute evidence | Defines what the label means in the first place |

The decision owner is the Head of Payment Risk, not the data science lead. That is deliberate: the
choice trades investigator capacity against loss, and only the risk owner holds both sides of that
trade in one budget.

### 2.4 The baseline and its documented weaknesses

The incumbent policy is a weighted rule count. Thirty-four rules contribute integer points; points
are summed; cases are sorted descending; the first 400 are reviewed. Rules include velocity counts
per card and per device, amount deviation from the card's own trailing median, address and country
mismatch flags, retry bursts after declines, and merchant-category heuristics.

Its weaknesses are properties of the construction, not claims about incidents:

- **Fixed weights.** Weights were set by analyst judgement and never re-derived against loss. A
  rule firing on 40% of traffic contributes the same points as a rule firing on 0.2%.
- **No interaction terms.** A new device on an established card, at an unusual hour, in a category
  the card has never used, scores the same as one flag plus an unrelated third.
- **Integer ties at the cut.** The 400th and 430th cases often share a score, so small weight edits
  move hundreds of cases across the boundary and make quarter-on-quarter comparison noisy.
- **No probability.** Points cannot be compared to a cost, so the desk cannot answer "is this case
  worth eighteen minutes" except by intuition.
- **Rule accumulation.** Each new fraud pattern adds a rule; the count grew from 19 to 34 in three
  years and none has been retired.

These weaknesses motivate a change but do not prove the anomaly approach repairs them. Section 6
tests the repair claim explicitly, and section 12 lists the ways the repair itself fails.

### 2.5 Scale, timing, and boundaries

| Dimension | Illustrative value | Note |
|---|---|---|
| Approved authorisations per business day | 1,150,000 | one region, one currency, card-not-present |
| Distinct cases after dedupe | 880,000 | case key card token × merchant account × business date |
| Fraud cases present in a day | 2,200 | 0.25% of cases; this is the ceiling on any queue |
| Desk headcount | 18 investigators | 26 productive reviews each per day |
| Theoretical daily slots | 468 | 18 × 26 |
| Queue size | 400 | 14.5% slack below theoretical capacity |
| Average loss per confirmed fraud case | EUR 180 | prior 12-month mean, ILLUSTRATIVE |
| Preventable fraction given a 24-hour SLA | 0.80 | orders not yet shipped can be cancelled |
| Fully loaded cost per review | EUR 13.50 | labour plus tooling, derived in section 6 |
| Customer friction per false positive | EUR 9.00 | measured as contact cost plus churn expectation |
| Label maturity embargo | 120 days | scheme chargeback windows plus issuer reporting lag |
| Peak season | November–December | fraud case volume assumed 40% above the annual mean |

Two of those rows need a warning because they are routinely mis-stated in practice. The 0.25%
prevalence is a *ceiling*: no queue of any length can contain more genuine fraud than the day
contains, which is 2,200 cases. And the 0.80 preventable fraction is a property of timing, not of
the model — the same alert filed at hour 30 instead of hour 6 prevents far less, so any claim about
model value is implicitly a claim about latency.

### 2.6 Why the anomaly approach was considered at all

No labelled history exists for the newest fraud patterns, because a pattern must be disputed before
it can be labelled, and disputes arrive up to 120 days late. A supervised classifier trained on
matured labels therefore has a structural blind spot for the thing that matters most: an attack
that started three weeks ago. An unsupervised anomaly score, fitted on the recent feature
distribution, can flag behaviour that is unusual relative to the current population without waiting
for a label.

That argument is real but weaker than it sounds, for two reasons the case keeps returning to.
Unusualness is not fraud: most unusual card behaviour is a new customer, a holiday, or a merchant
category shift. And "fitted on the recent distribution" is itself a modelling choice with a
failure mode — an attacker who adapts slowly becomes the new normal, and the score stops firing.

### 2.7 The organisational trap

The tempting framing inside the organisation is a platform programme: build a scoring service, a
feature store, a case store, and a model registry, then measure lift. That framing consumes a
quarter and answers none of the questions in section 3, because lift at an unspecified depth is not
a decision input. The framing used here is the reverse: fix the decision, derive the arithmetic,
and let the arithmetic tell you how much platform is justified.
## 3. The Business Question

### 3.1 The question as the risk owner would sign it

> At the 04:00 queue build on any business day, with the case population, the label definition and
> the 400-slot capacity all fixed, does replacing the 34-rule weighted count with a calibrated
> anomaly score increase prevented fraud loss net of review cost and customer friction, by an
> amount that survives the uncertainty in the label set, the day-to-day variance of the queue, and
> the possibility that the review desk behaves differently when the queue is reordered?

The question is a comparison, not a level. "Is the model good?" has no answer that a budget can
use, because any precision figure can be made to look strong by reading a shallow queue or weak by
reading a deep one. The comparison fixes depth, population, cost, and action set, so the only
remaining variable is the sort order.

### 3.2 The three sub-questions that must be answered first

| Sub-question | Why it is prior to the main question | What answers it |
|---|---|---|
| Is the label available at the depth we read? | Precision at rank 400 is undefined if labels exist only for cases the old policy opened | label-coverage audit by rank band |
| Is the marginal case at rank 400 worth more than it costs? | If the marginal case is negative, the right answer is a smaller queue, not a better sort | break-even precision arithmetic |
| Does reordering change what the desk can act on? | A confirmed fraud already shipped cannot be prevented; rank order interacts with latency | preventable-fraction-by-age table |

The second sub-question is the one that most often reverses the decision. A team that improves
precision at fixed capacity still loses money if the capacity itself is past the point where the
marginal case pays for itself, because the improvement buys more of a losing activity. Section 6
computes this explicitly and finds that in this constructed setting capacity is *undersized* under
both rankers, which changes what the recommendation should be for next quarter.

### 3.3 What "no" would look like

A decision memo is only usable if it states in advance the observation that would kill it. Here the
kill conditions are concrete:

- Precision at rank 400 is not better than the rule queue by more than the day-to-day standard
  error, over a matured 90-business-day cohort, at the read depth.
- The extra confirmed cases are concentrated in cohorts the desk cannot act on before shipment,
  so prevented loss does not rise even though confirmed cases do.
- The reordering increases the number of blocked credentials per day and the false-positive
  friction cost rises by more than the prevented loss.
- Label coverage at ranks 200–400 is too sparse to estimate precision there without an assumption
  that itself decides the answer.

Each of those is a different failure and would call for a different response: a modelling problem,
a latency problem, a policy problem, and a measurement problem respectively. Collapsing them into
"the pilot was inconclusive" is the usual way a real finding is lost.

### 3.4 The candidate set that was considered and rejected

The decision was framed against four alternatives, not one:

1. **Keep the rule queue, retune weights.** Cheap, respects the label problem, but the tie-group
   brittleness means retuning is hard to evaluate; treated as the status quo comparator.
2. **Add a supervised fraud model.** Strongest where labels are dense, blind to patterns younger
   than the embargo; deferred to the next cycle when a matured anomaly-derived candidate set could
   seed it.
3. **Increase headcount instead of changing the ranker.** Attractive because the marginal case is
   profitable, but it spends on reviewing a worse list and cannot be justified without first
   fixing the sort; sequenced afterwards.
4. **Decline more at authorisation time.** Highest leverage per case and highest customer cost;
   blocked by the latency and challenge-flow prerequisites in section 8.

Only alternative 4 is genuinely higher-leverage than the chosen intervention, and it is not
available this quarter. That ordering is the substance of the recommendation, and it is why the
case spends more words on sequencing than on model selection.

### 3.5 Why the queue length is not a decision variable this quarter

An earlier draft of this analysis let the queue length float and maximise value over K. That
version is mathematically tidier and operationally useless: K is set by headcount, hiring is a
six-week lead-time process, and the desk cannot partially staff a day. Treating K as fixed and
asking a different question — "what is the value of the marginal slot?" — produces a number the
risk owner can hold next to a hiring requisition, which is the actionable form.

## 4. Assumptions & Constraints

Each assumption below is stated with the observation that would falsify it and the check that
detects falsification. Assumptions without a check are wishes, and this case has none.

| # | Assumption | If violated | Check |
|---:|---|---|---|
| A1 | A case has one binary outcome: fraud or not, resolved within 120 days | label noise dilutes every precision comparison toward zero | outcome audit on a 2,000-case sample |
| A2 | Average loss per confirmed case is EUR 180 and does not depend on rank | value ranking diverges from count ranking | loss by score decile, 12-month window |
| A3 | Preventable fraction is 0.80 at a 24-hour review latency | the model's gain does not convert to money | order-shipment timestamp join |
| A4 | Fully loaded review cost is EUR 13.50 and is constant in rank | deep cases may be cheaper or dearer | time-and-motion sample by rank band |
| A5 | Precision declines monotonically with rank | a non-monotone curve breaks the top-K premise | cumulative precision by band |
| A6 | The case population composition is stable day to day | precision shifts with mix, not with model | composition PSI by merchant category |
| A7 | Fraud prevalence 0.25% | prevalence sets the ceiling and the base rate for lift | daily case-label counts |
| A8 | Labels are missing at random given score | selection bias inflates the incumbent's measured precision | label coverage by rank band |
| A9 | No feedback from blocks into future features | the model learns its own interventions | feature provenance audit |
| A10 | The 24-hour service level holds at 400 cases | backlog converts preventable loss into confirmed loss | queue-age distribution at close |

### 4.1 A8 is the assumption that actually fails

Assumption A8 deserves separate treatment because it is the one that is usually false and the one
that quietly decides the comparison. In the incumbent regime, only the top 400 cases per day are
ever reviewed, so only those cases can carry an investigator-confirmed outcome. A case at rank
4,000 may have been fraud, but nobody looked, so it appears in the label store as "not fraud" or
does not appear at all.

Every precision figure computed on that label store is therefore conditional on the old policy's
selection. The bias has a known direction when rankers disagree: cases the old rules ranked low and
the new model ranks high are systematically under-labelled, so the new model's precision at those
ranks is *understated*, not overstated. That sounds reassuring until you notice the mirror image:
cases the old rules ranked high and the new model ranks low are over-labelled, which flatters the
incumbent's measured precision at the same depth. The comparison is biased in both directions and
the net sign is an empirical question, not a comfort.

The mitigation used here is a *sampled review cell*: 60 randomly chosen cases per day from ranks
401–3,000 are reviewed regardless of score, for the duration of the pilot. At 60 reviews per day
out of 400 slots that is 15% of capacity, which is expensive; section 9 prices it and section 10
explains why the quarter's plan budgets for it anyway.

### 4.2 Hard constraints

- **Headcount.** 18 investigators, 26 productive reviews each per day. Hiring is out of scope.
- **Service level.** Every queued case must be opened within 24 hours of the queue build.
- **Latency budget for the queue build.** The 04:00 build must finish before 06:00, which bounds
  feature materialisation to a two-hour batch window.
- **Regulatory.** Cardholder data stays in the region; the feature store cannot hold raw PAN; any
  score that leads to a credential block must be explainable to the scheme in dispute evidence.
- **Change freeze.** No policy change during the last two weeks of November and December, because
  peak-season volume is 40% above mean and any comparison made then confounds season with policy.
- **Audit trail.** Every queue must be reproducible from a frozen population snapshot plus the
  policy version, or the comparison cannot be defended after the fact.

### 4.3 Soft constraints and the cost of honouring them

The explainability requirement is soft in the sense that no rule mandates a specific form, but it
has a real cost: an anomaly score from an isolation forest is a path-length aggregate and is not
naturally decomposable into a per-case reason. Two options exist, and they are not equivalent.

| Option | Mechanism | Cost | Weakness |
|---|---|---|---|
| Local surrogate attribution | fit a shallow tree to the score near each case | cheap, per-case | explains the surrogate, not the model |
| Contribution decomposition | per-feature deviation contributions from the fitted transform | exact per feature | assumes additivity the score does not have |
| Nearest-neighbour exemplars | show the case's 5 nearest historical cases and their outcomes | cheap, human-legible | leaks the outcome of similar cases into the UI |

The design uses exemplars plus a top-5 feature-deviation list, accepts that the explanation is
approximate, and states in the dispute-evidence template that the score is a prioritisation signal
rather than a determination. That last sentence is not boilerplate: writing it down is what allows
the desk to act on the queue without turning a model rank into an accusation.

### 4.4 What is explicitly out of scope

Authorisation-time declines, merchant onboarding risk, chargeback representment strategy, and
collections. Each of those interacts with this decision, which is why section 14 links them, but
none is changed by it. The scope statement is repeated in the quick reference card because the most
expensive scope error in risk programmes is a queue change that silently acquires the authority to
decline.

## 5. Data, Tools & Method

### 5a Data Specification

**Grain.** One row per *case*, where a case is the tuple (card token, merchant account, business
date). Multiple authorisations on the same card at the same merchant on the same day collapse into
one case, because the desk investigates the relationship, not the individual attempt. The chosen
grain is a deliberate loss of resolution: it hides a burst of 40 declines behind one row, which is
why burst features are attached as aggregates rather than as separate rows.

**Cut-off timestamp.** The queue for business date D is built at 04:00 local on D+1, covering
authorisations with event time in [D 00:00, D 24:00). All features must be computable from data
whose processing time is earlier than D+1 04:00. Any source that habitually arrives later is
either excluded or lagged by one day with the lag recorded in the feature registry.

**Population.** Approved card-not-present authorisations for the 41,000 merchant accounts, one
region, one currency, excluding: transactions declined at authorisation by hard blocks (they never
become cases), test merchant accounts, and cases whose card token was blocked in the preceding 30
days (already actioned, no further value).

**Keys and units.**

| Field | Type | Unit | Note |
|---|---|---|---|
| case_id | string | — | hash of the three-part key, stable across days |
| card_token | string | — | surrogate; PAN never leaves the tokeniser |
| merchant_account_id | string | — | 41,000 distinct values |
| business_date | date | day | event-time date, not processing date |
| amount_eur | decimal | EUR, 2 dp | authorised amount |
| decline_count_24h | integer | count | attempts before approval, same card |
| device_novelty_days | integer | days | days since first sight of device on this card |
| merchant_category | enum | ISO 18245 MCC | 240 observed values |
| txn_hour_local | integer | 0–23 | local hour of authorisation |
| chargeback_flag | boolean | — | label; arrives up to 120 days after t0 |
| issuer_fraud_report | boolean | — | label; scheme-level, 30–90 day lag |

**What is not available at decision time.** The label itself; the merchant's fulfilment status
beyond the order record's own promised ship date; the outcome of any investigation opened today;
the eventual chargeback reason code; whether the cardholder will dispute at all. A model that uses
any of these leaks the future, and the leakage audit in section 5d is built specifically to catch
that class of mistake.

**Feature families.** Five families, each with a stated window because the window is part of the
definition:

1. **Velocity** — counts of attempts, declines, and distinct merchants over 1 h, 24 h, 7 d.
2. **Deviation** — amount versus the card's trailing 90-day median and 90th percentile.
3. **Novelty** — age of device, IP subnet, email domain, and shipping address on this card.
4. **Consistency** — mismatch flags between billing country, IP country, and merchant country.
5. **Graph** — number of distinct cards sharing the device or IP in the trailing 7 days.

The graph family is the one that carries most of the fraud-ring signal and also the one that
creates the sharpest privacy question, since it aggregates behaviour across unrelated cardholders.
Section 8 addresses that with aggregation-only storage and a minimum support count.

### 5b Tools & Stack

| Layer | Choice | Why this and not the alternative |
|---|---|---|
| Event capture | append-only authorisation log | immutable source of truth; replayable |
| Feature assembly | hourly batch over a 90-day window | two-hour budget rules out per-case graph expansion |
| Anomaly scorer | isolation forest, 300 trees, depth limit 8 | linear in cases, no distance matrix, handles mixed scale |
| Calibration | isotonic regression on matured labels | monotone, no distributional assumption; needs volume |
| Queue builder | deterministic top-K with tie-break on amount | reproducibility; ties must not depend on row order |
| Case store | relational tables with policy version stamp | auditability of what the desk saw and when |
| Evaluation harness | frozen population replay | comparison must be reconstructable months later |
| Monitoring | score distribution, queue overlap, label coverage | each detects a different failure in section 12 |

The scorer choice carries an argument worth stating. A density method such as LOF needs a
neighbourhood computation per case and its cost grows with the square of the local density; on
880,000 cases per day inside a two-hour window that is not affordable without sampling. An
autoencoder would fit the interaction structure better but its score is not monotone in any
interpretable quantity and its retraining cadence interacts badly with the change freeze. The
isolation forest is chosen for cost and for the property that its score has a clean null
distribution when features are independent, which makes the threshold interpretable.

The isolation forest's weakness is equally clear and is carried into the design: it is
axis-aligned, so a fraud pattern defined by a *correlation* between amount and hour is invisible to
it, and it is fitted on the recent population, so a pattern that becomes the majority stops being
anomalous.

### 5c Method Summary

1. Freeze the population and the label snapshot at a stated date, with the embargo applied.
2. Fit the anomaly scorer on the trailing 90 days of features, unlabelled, excluding the embargo
   window so that the fit does not see any outcome.
3. Score the cohort; convert raw scores to probabilities by isotonic regression on matured labels
   from a *held-out earlier* window, never the evaluation window.
4. Sort descending; apply the capacity cut at 400; tie-break deterministically.
5. Compute cumulative precision at every depth from 50 to 600, with binomial intervals.
6. Run the same pipeline on the rule score as the comparator, on the identical population.
7. Compute net value under the cost model, and the marginal value at the cut.
8. Run the sensitivity grid over preventable fraction, loss, cost, and friction.
9. Execute the label-coverage audit and the leakage audit before reporting any number.
10. State, in the memo, which of the nine checks above were actually executed.

Step 10 is not ceremony. In this teaching design steps 1–8 are executed arithmetic, step 9 is
specified with its sampling plan but not executed (there is no real label store), and step 3's
isotonic fit is described rather than run. Section 9 records that distinction per metric.

### 5d Provenance, Quality, Exclusions, Leakage

**Provenance.** Four sources, each with an owner and a freshness contract:

| Source | Owner | Freshness contract | Failure if stale |
|---|---|---|---|
| authorisation log | platform engineering | < 15 min | queue misses the tail of the day |
| device and IP enrichment | vendor | < 60 min | novelty features default to "new", inflating scores |
| merchant master | merchant ops | daily | MCC drift misroutes the graph features |
| label store | risk analytics | daily, 120-day lag | calibration drifts without any signal |

**Quality checks that must pass before a queue is published.** Row count within 3% of the trailing
28-day median; null rate on device novelty below 2%; distinct merchant count within 1% of master;
label coverage at ranks 1–400 above 92%; score distribution median within 5% of trailing 14 days.
A failed check does not stop the queue — it attaches a flag to the run and pages the on-call
analyst, because a queue of a slightly wrong shape is better than no queue at all.

**Exclusions and their justification.** Cards blocked in the last 30 days are excluded because the
action has already been taken and re-queuing them spends a slot on a resolved case. Merchant
accounts in onboarding for fewer than 30 days are excluded because their baseline is undefined and
their velocity features are dominated by ramp-up. Test accounts and internal merchant IDs are
excluded by an explicit list, not by a name pattern, because name patterns silently re-include
accounts.

**Leakage vectors and their checks.**

- *Post-outcome fields.* Any field written after a dispute exists (chargeback reason code, dispute
  status, recovery amount) leaks the label. Check: for every feature, the provenance record must
  show a write time strictly before D+1 04:00 for at least 99.5% of cases.
- *Intervention fields.* "Credential blocked" and "order cancelled" are consequences of the queue.
  Including them makes the model learn yesterday's decisions. Check: feature registry forbids any
  field whose producer is the case store.
- *Temporal leakage in calibration.* Fitting isotonic regression on the same window that is used to
  report precision overstates precision. Check: calibration window and evaluation window must be
  disjoint, with the split recorded in the run manifest.
- *Population leakage.* Fitting the scorer on the whole cohort including evaluation days lets the
  model see the attack it is supposed to detect. Check: fit window ends at least one day before the
  first evaluation day; for a 90-day cohort the fit uses days −90 to −1.
- *Survivorship in the label store.* Only reviewed cases are labelled, so a naive join produces a
  population that is the old policy's selected set. Check: report label coverage by rank band and
  refuse to publish precision for a band below 60% coverage.

The last check is the one that most often fails in real work, and it fails silently: the numbers
look plausible, the population looks large, and the fact that it was pre-filtered is invisible
unless someone explicitly measures coverage by band.

### 5e Reproducibility & Method-Selection Ledger

**Reproducibility contract.** A queue is reproducible if (a) the population snapshot is content
addressed, (b) the feature definitions are versioned, (c) the scorer artefact hash is recorded,
(d) the calibration curve version is recorded, (e) the tie-break rule is deterministic, and (f) the
random seed for tree construction is fixed. All six are stamped into every queue record. Replaying
a queue must reproduce the same 400 case IDs in the same order; section 12 lists the ways this
check fails even when the code is unchanged.

**Method-selection ledger.**

| Candidate | Rejected because | Revisit trigger |
|---|---|---|
| LOF on full population | cost per case quadratic in local density; two-hour budget | if sampling to 100k cases proves faithful |
| One-class SVM | fit time scales badly beyond 200k rows; kernel choice unstable | never for this volume |
| Autoencoder reconstruction error | needs a retraining cadence incompatible with the freeze | if freeze moves to quarterly |
| Supervised gradient boosting | blind to patterns younger than the 120-day embargo | when matured labels exceed 18 months |
| Rule queue with retuned weights | integer ties make evaluation noisy at the cut | if a tie-free rule score is introduced |
| Isolation forest | **selected** | re-evaluate if correlation-defined fraud dominates |

**Seed and version discipline.** Tree count, subsample size, and seed are recorded per run. A
change in any of the three changes the score distribution and therefore the queue; section 12
treats an unrecorded seed change as a defect with the same severity as a code change.
## 6. Analysis Walkthrough

Four derivations are shown in full. Each follows the same anatomy: the formula, the quantities it
is defined on, the constraint that makes it valid, a numerical example with intermediate values,
and the interpretation that a decision-maker would actually use. Every number is recomputed here
rather than quoted from a model output.

### 6.1 Derivation 1 — break-even precision, and why capacity is a money question

```text
FORMULA
  Reviewing one more case is worth doing when its expected net value is positive:
      V(p) = p · L_p  −  c  −  (1 − p) · f
  where p is the probability the case is genuine fraud.
  Setting V(p*) = 0 and solving:
      p* · L_p  −  c  −  f  +  p* · f  =  0
      p* · (L_p + f)                     =  c + f
      p*                                 =  (c + f) / (L_p + f)

WHERE
  L_p  = preventable loss per confirmed case, EUR      = φ · L = 0.80 × 180 = 144
  φ    = preventable fraction at 24-hour latency, 0..1 = 0.80
  L    = average loss per confirmed case, EUR          = 180
  c    = fully loaded cost per review, EUR             = 13.50
  f    = customer friction per non-fraud review, EUR    = 9.00
  p    = P(genuine fraud | case), 0..1

CONSTRAINT
  L_p > 0 and c + f > 0, otherwise the ratio is undefined or negative and the answer is trivial.
  Requires that friction is incurred once per non-fraud review and that a true fraud review incurs
  no friction. If a confirmed fraud also generates a friction cost (a blocked legitimate-looking
  credential that the customer must re-enrol), the denominator changes and p* rises.

NUMERICAL EXAMPLE
  numerator      c + f        = 13.50 + 9.00        = 22.50  EUR per reviewed case
  denominator    L_p + f      = 144.00 + 9.00       = 153.00 EUR per reviewed case
  p*             = 22.50 / 153.00                    = 0.14706
  check at p*:   0.14706 × 144 = 21.177 ; 21.177 − 13.50 − 0.85294×9 = 21.177 − 13.50 − 7.676
                 = 0.001 ≈ 0, rounding residual from the 5-decimal p*
  check at p=0.63: 0.63 × 144 = 90.72 ; 90.72 − 13.50 − 0.37 × 9 = 90.72 − 13.50 − 3.33 = 73.89 EUR

INTERPRETATION
  Any case whose fraud probability clears 14.7% is worth an investigation slot at these costs.
  Because precision at rank 400 is 0.630 — more than four times the break-even level — the desk is
  not at its economic limit; it is limited by headcount. The correct quarter-one action is to fix
  the ranking, not to shrink the queue. The 0.147 figure also sets the price of a wrong threshold:
  a queue cut at precision 0.30 would still be profitable — 0.30 × 144 − 13.50 − 0.70 × 9 =
  23.40 EUR per slot — but it would capture only 23.40/73.89 = 32% of the value available at
  precision 0.63.

BOUNDARY / COUNTEREXAMPLE
  If the review latency slips so that the preventable fraction falls to φ = 0.30, then
  L_p = 54, p* = 22.50 / 63.00 = 0.357. At that point precision at rank 400 (0.630) is still above
  break-even, but only 1.76× rather than 4.3×, and the value of a slot collapses from 73.89 to
  0.63×54 − 13.50 − 3.33 = 34.02 − 16.83 = 17.19 EUR. A five-times-slower desk destroys 77% of the
  per-slot value without changing the model, the data, or the precision. Latency is a first-class
  term in the value model, not an operational detail.
```

The friction term is worth a second look because it is the one practitioners leave out. Its effect
on the break-even point is not additive-and-small: raising friction from EUR 9 to EUR 30 moves
p* from 0.147 to (13.50+30)/(144+30) = 43.50/174 = 0.250. A queue cut at 0.25 rather than 0.147 is
a materially shorter queue, and the difference is entirely attributable to how the organisation
prices a wrongly contacted customer.

### 6.2 Derivation 2 — precision at depth, and the fitted ranking curve

A single number is not enough to answer the question, because the answer depends on capacity. What
is needed is the whole function of precision against rank, estimated from a matured cohort.

```text
FORMULA
  Cumulative precision at depth k:
      P(k) = (1/k) · Σ_{i=1..k} y_i
  with y_i ∈ {0,1} the matured fraud outcome of the case at rank i.

  Empirically P(k) is smooth and convex-decreasing, so fit the parsimonious two-parameter law
      P(k) = A · k^(−γ),   A > 0,  0 < γ < 1
  The two parameters are identified by any two points (k1,P1), (k2,P2):
      γ = ln(P1/P2) / ln(k2/k1)
      A = P1 · k1^γ
  The marginal precision at depth k, i.e. the probability that the k-th case is fraud, follows by
  differentiating the cumulative count C(k) = k·P(k) = A·k^(1−γ):
      P_marg(k) = dC/dk = A·(1−γ)·k^(−γ) = (1−γ) · P(k)

WHERE
  k   = queue depth in cases, integer ≥ 1
  y_i = matured fraud indicator, 1 if chargeback or issuer report confirmed
  A   = scale, cases^(γ); units chosen so P is dimensionless
  γ   = decay exponent, dimensionless

CONSTRAINT
  Valid only where 0 < P(k) ≤ 1, i.e. k ≥ A^(1/γ) is required for P ≤ 1 — here A^(1/γ) ≈ 44.8, so
  the law must not be used at depths below about 45 cases. It also assumes the ranking is a single
  ordered list with no ties; ties break the "k-th case" interpretation and require fractional
  allocation of the marginal case.

NUMERICAL EXAMPLE
  Observed: P(100) = 0.844, P(400) = 0.630.
  Step 1, the exponent:
    P1/P2      = 0.844 / 0.630            = 1.339683
    ln(1.339683)                          = 0.292438
    ln(k2/k1)  = ln(400/100) = ln(4)      = 1.386294
    γ          = 0.292438 / 1.386294      = 0.210946
  Step 2, the scale, taken from the depth-100 point:
    A = P1 · k1^γ = 0.844 × 100^0.210946
      = 0.844 × exp(0.210946 × 4.605170)
      = 0.844 × exp(0.971447)
      = 0.844 × 2.641451
      = 2.229634
  Step 3, resubstitute both fitting points:
    P(100) = 2.229634 × exp(−0.210946 × 4.605170) = 2.229634 × 0.378548 = 0.844000 ✓
    P(400) = 2.229634 × exp(−0.210946 × 5.991465) = 2.229634 × 0.282534 = 0.629952 ✓
    (the 0.00005 residual at k=400 is rounding in γ at the sixth decimal)
  Step 4, the marginal:
    P_marg(400) = (1 − 0.210946) × 0.629952 = 0.789054 × 0.629952 = 0.497077
    C(400)      = 400 × 0.629952 = 251.98 cases caught, i.e. 252 to the nearest case
    C(401) = A × 401^0.789054 = 2.229634 × exp(0.789054 × 5.993961)
           = 2.229634 × exp(4.729370) = 2.229634 × 113.2232 = 252.4963
    C(401) − C(400) = 252.4963 − 251.9994 = 0.4969 additional fraud cases
    Compare the derivative: (1−γ)·C(400)/400 = 0.789054 × 251.9994 / 400 = 0.4971
    The discrete step and the derivative differ by 0.0002, which is the curvature of C over one case.

INTERPRETATION
  The 400th case is fraud with probability 0.497, less than half — which is still 3.4× the
  break-even level of 0.147, so the marginal slot earns 0.497×153 − 22.50 = 76.06 − 22.50 = 53.56
  EUR. Wait: 0.49705 × 144 − 13.50 − 0.50295 × 9 = 71.58 − 13.50 − 4.527 = 53.55 EUR. That is the
  honest marginal figure, and it is 20.3 EUR below the *average* slot value of 73.89. The gap is
  the whole story of diminishing returns: the average flatters the cut, the marginal decides it.
  Note also that the marginal value is computed from the differentiable law, which slightly
  understates the discrete step: C(401) − C(400) = 0.49697 fraud cases, consistent to 3 decimals.

BOUNDARY / COUNTEREXAMPLE
  Push k until the law predicts more fraud cases than exist. The cohort contains 2,200 fraud cases
  per day, so k · P(k) = 2,200 at k = (2200/A)^(1/(1−γ)) = 6,232. Beyond that depth the fitted law implies a precision
  that cannot be delivered because the supply has run out — the true curve must collapse toward
  zero far faster than a power law. Any capacity plan derived from the law above roughly 6,000
  slots is therefore invalid: the binding constraint becomes supply of fraud, not precision.
```

Two properties of this fit are worth carrying forward. It has no interpretable parameter — A and γ
are curve shape, not quantities with units a business can reason about — so it is a measurement aid
rather than a model of fraud. And it must be validated by resubstituting *both* fitting points:
a curve that reproduces one and misses the other was mis-identified, which is the most common
arithmetic failure in this kind of back-of-envelope ranking work. Substituting a third point (say
rank 200, where the observed cumulative precision is 0.729) is the real test: the fitted law gives
2.23068 × exp(−0.211025 × 5.29832) = 2.23068 × 0.32664 = 0.72863 against an observed 0.7292 — an
error of 0.0006, or 0.06 percentage points. That confirms the power law is adequate across
100–400 and says nothing at all about ranges outside it.

### 6.3 Derivation 3 — the value of the change at fixed capacity

```text
FORMULA
  Net daily value of a policy at capacity K and precision p:
      N(p, K) = K · p · L_p  −  K · c  −  K · (1 − p) · f
  Advantage of the anomaly ranker over the rule ranker at identical K:
      ΔN = N(p_a, K) − N(p_r, K)
         = K · (p_a − p_r) · (L_p + f)
  Annualised advantage:
      ΔN_year = ΔN · B,  B = business days per year

WHERE
  p_a = precision of the anomaly ranker at depth K   = 0.630
  p_r = precision of the rule ranker at depth K      = 0.540
  K   = 400 cases per day
  B   = 250 business days (5-day week, 52 weeks, minus 10 holidays)

CONSTRAINT
  Requires the same population, the same label definition, the same K, and the same action set for
  both arms. If the two arms read different depths, ΔN conflates the ranking effect with an effort
  effect and the decomposition above is invalid. Requires L_p and f to be equal across arms, which
  holds by construction because the desk does not know which ranker produced a queue entry.

NUMERICAL EXAMPLE
  p_a − p_r =  0.630 − 0.540 = 0.090
  L_p + f   =  144 + 9       = 153.00 EUR
  ΔN        =  400 × 0.090 × 153 = 36 × 153 = 5,508 EUR per business day
  Cross-check by components:
      extra fraud caught       400 × 0.090          =  36.0 cases/day
      extra loss prevented     36.0 × 144           = 5,184 EUR/day
      friction avoided         400 × 0.090 × 9     =   324 EUR/day (fewer false positives)
      total                                        = 5,508 EUR/day ✓
  ΔN_year   =  5,508 × 250 = 1,377,000 EUR per year
  N(p_a,400) = 400 × 0.630 × 144 − 400 × 13.50 − 400 × 0.370 × 9
             = 36,288 − 5,400 − 1,332 = 29,556 EUR/day
  N(p_r,400) = 400 × 0.540 × 144 − 5,400 − 400 × 0.460 × 9
             = 31,104 − 5,400 − 1,656 = 24,048 EUR/day
  ΔN       = 29,556 − 24,048 = 5,508 EUR/day ✓ and 5,508×250 = 1,377,000 ✓

INTERPRETATION
  The change is worth about EUR 1.4 million a year at constant headcount, against a build cost
  estimated at EUR 180,000 (section 8). Payback is 180,000 / 5,508 = 32.7 business days. That
  payback period is short enough that the decision does not depend on the precise build cost, which
  is what makes the recommendation robust: even a build three times over budget pays back inside a
  quarter.

BOUNDARY / COUNTEREXAMPLE
  Let the two rankers be read at different depths — a common accident when the anomaly queue is
  longer because it produces more alerts. If the anomaly arm reads 500 and the rule arm 400, then
  N(0.601, 500) = 500×0.601×144 − 6,750 − 500×0.399×9 = 43,272 − 6,750 − 1,796 = 34,726 and the
  comparison against 24,048 reports a 10,678 gain, of which 5,508 is ranking and 5,170 is 100 extra
  reviews the desk does not have. Always normalise to one K before reporting ΔN.
```

### 6.4 Derivation 4 — how much evidence is needed to see a 9-point gap

```text
FORMULA
  Standard error of a precision estimate at depth k with p̂ realised precision:
      SE(p̂) = sqrt( p̂(1 − p̂) / k )
  Standard error of the gap between two independent arms at the same depth:
      SE(Δ) = sqrt( SE(p̂_a)² + SE(p̂_r)² )
  Days of cohort needed for a two-sided test at significance α and power 1−β:
      D = ( (z_{1−α/2} + z_{1−β}) · SE_day / Δ )²
  where SE_day is the empirical day-to-day standard deviation of the gap, which is larger than the
  binomial SE because cases within a day are not independent.

WHERE
  α     = 0.05, so z_{0.975} = 1.960
  1 − β = 0.80, so z_{0.80}   = 0.842
  Δ     = 0.090 (observed gap at depth 400)
  SE_day= 0.060 (empirical, ILLUSTRATIVE, over 90 business days)

CONSTRAINT
  The binomial form assumes independent cases. Cases are correlated within a day (a fraud ring
  generates many cases at once), so the binomial SE understates the true variance. The design
  effect measured as (SE_day / SE_binomial)² must be reported; if it exceeds 2, the binomial
  formula must not be used for sample sizing.

NUMERICAL EXAMPLE
  Binomial: SE(p_a) = sqrt(0.630 × 0.370 / 400) = sqrt(0.23310/400) = sqrt(0.00058275) = 0.024140
            SE(p_r) = sqrt(0.540 × 0.460 / 400) = sqrt(0.24840/400) = sqrt(0.00062100) = 0.024920
            SE(Δ)   = sqrt(0.024140² + 0.024920²) = sqrt(0.00058275 + 0.00062100)
                    = sqrt(0.00120375) = 0.034695
  Design effect: (0.060 / 0.034695)² = (1.7294)² = 2.99 — cases are visibly correlated, as expected.
  Days needed:  D = ((1.960 + 0.842) × 0.060 / 0.090)² = (2.802 × 0.6667)² = (1.8680)² = 3.49 days
  With the naive binomial SE the same formula returns ((2.802 × 0.034695)/0.090)² = (1.0801)² =
  1.17 days, which would have justified a two-day pilot and produced a coin flip.

INTERPRETATION
  Roughly four independent business days of matured, unbiased measurement are needed to see the
  gap, and maturity takes 120 days, so the earliest honest read is about four months after the
  pilot starts. The 90-day cohort gives an effective standard error of 0.060/sqrt(90) = 0.00632,
  making the observed gap 14.2 standard errors — comfortably decisive. The binding delay is
  therefore label maturity, not sample size; more days help only after the embargo clears.

BOUNDARY / COUNTEREXAMPLE
  Now do the same arithmetic at depth 100, where the gap is 0.024. Binomial SE at k=100:
  SE(0.844) = sqrt(0.844×0.156/100) = 0.036286; SE(0.820) = sqrt(0.820×0.180/100) = 0.038419;
  SE(Δ) = sqrt(0.001317 + 0.001476) = 0.052846. Days needed at design effect 2.99, i.e.
  SE_day = 0.052846 × 1.7294 = 0.09139: ((2.802 × 0.09139)/0.024)² = (10.670)² = 113.9 days.
  A pilot that measured at depth 100 would need 114 days of matured data to distinguish the two
  rankers, and would have concluded "no difference" long before then. Read depth is a measurement
  design choice with the same status as sample size.
```

### 6.5 What the four derivations jointly establish

| Derivation | Question answered | Answer | What it rules out |
|---|---|---|---|
| break-even precision | is a slot worth buying? | p* = 0.147 | shrinking the queue as a first move |
| precision at depth | what does rank 400 deliver? | 0.630, marginal 0.497 | judging by top-of-queue plots |
| value of change | what is the ranking worth? | EUR 5,508/day, 1.377 M/yr | platform-first sequencing |
| evidence sizing | how long to measure? | 3.5 days of matured data, 114 if read shallow | a two-week pilot verdict |

Three of the four answers point the same way: the change is worth doing, it is worth doing at the
existing capacity, and it must be measured deep in the queue over a matured cohort. The fourth
says the measurement itself is the expensive part, which is why section 9 prices the audit cell
explicitly instead of listing it as good practice.

One number deserves emphasis because it is counter-intuitive: the marginal value at the cut is
53.55 EUR while the average is 73.89 EUR. Organisations that justify capacity expansion using
average value systematically over-buy investigators. Section 10 uses the marginal figure, not the
average, which is why it recommends four hires rather than a doubling of the desk.

## 7. The Finding

**The finding, in one sentence.** At the depth the desk actually reads, the calibrated anomaly
ranker confirms 36 more fraud cases per 400 reviews than the incumbent rule score, worth EUR 5,508
per business day after review cost and customer friction, and this gain is invisible at the top of
the queue where such changes are normally evaluated.

The supporting statements, each traceable to a derivation above:

1. **The gain is depth-dependent, not top-dependent.** At depth 100 the gap is 2.4 points; at depth
   400 it is 9.0 points. The rule score concentrates its information in a few strong signals that
   dominate the top of the list, then degrades quickly as it runs out of strong signals and starts
   relying on weak ones that fire on large populations. The anomaly score is flatter at the top and
   degrades more slowly.
2. **The mechanism is interaction coverage, not magic.** The rule score is additive over 34
   binaries, so it cannot express "new device AND unusual hour AND first-time category" as a joint
   event with its own weight. The anomaly score is fitted on the joint distribution, so a rare
   combination lands in a sparse region of the space and scores high without anyone writing a rule.
3. **Capacity is underneath the economic limit under both rankers.** Break-even precision is 0.147;
   both rankers clear it by a wide margin at 400. The marginal slot at 400 returns 53.55 EUR under
   the anomaly ranker and 0.540×153 − 22.50 = 60.12 EUR under the rule ranker — still strongly
   positive. The queue is short for both, just less short for the rules.
4. **The measured difference is statistically decisive but only after maturity.** The 90-day
   cohort gives 14.2 standard errors on a gap that a two-day pilot could not have seen. The
   finding is not fragile; the *measurement schedule* is what was fragile.
5. **The finding is conditional on latency.** At a 48-hour review latency the preventable fraction
   falls toward 0.55 and the per-slot value at the cut falls from 53.55 to about
   0.497×99 − 22.50 = 26.70 EUR. The ranking advantage survives; the absolute case for the
   programme shrinks by roughly half.

### 7.1 What the finding is not

It is not evidence that the anomaly score is a good fraud model in general. The comparison is
against a rule set with acknowledged construction weaknesses, at one capacity, in one region, on
one label definition. Beating a weak baseline is a low bar, and section 13 states the trap
precisely: the temptation is to attribute the gain to the algorithm when most of it is attributable
to replacing integer weights with a calibrated continuous score.

A cheap test separates those two explanations. Replace the rule score with a *logistic regression
on the same 34 rule indicators* — no new features, no anomaly model, just calibrated weights. If
that variant captures most of the 9-point gap, the gain belongs to calibration rather than to
anomaly detection, and the correct next investment is feature work, not a different algorithm.

### 7.2 The confound that cannot be removed

The label set is produced by the incumbent policy. In the pilot, both rankers are evaluated on the
same sampled-review cell, so the comparison is fair *within the cell*; outside the cell, the
incumbent's measured precision benefits from being the policy that chose what got labelled. The
finding therefore holds for "precision at depth as measured under a common review protocol" and
does not extend to "true precision in the full population", which is unmeasurable without
reviewing everything. Stating that limit is more useful than a wider confidence interval, because
it tells the reader exactly which decision the number supports.

### 7.3 The finding in the form the memo uses

> Replacing the queue sort increases confirmed fraud per 400 reviews from 216 to 252 and net daily
> value from EUR 24,048 to EUR 29,556. The break-even precision implied by our cost structure is
> 0.147, so the queue could profitably be longer; the marginal slot at rank 400 still returns
> EUR 53.55. Recommend: adopt the ranker, hold headcount at 18 for one quarter, then open a hiring
> case at the marginal value, and do not attempt to evaluate the change at a queue depth below 300.

Every clause is a decision or a prohibition. Nothing in it says "the model is better", because
that sentence cannot be acted on.
## 8. System/Solution Design

### 8.1 Design stance

The design is a *queue builder*, not a fraud engine. Its only output is an ordered list of 400 case
identifiers per business day plus, for each, a calibrated probability and a short evidence block.
It cannot decline, cannot block, and cannot contact a cardholder. Every consequential action remains
an investigator decision recorded in the case store. That narrowness is the single most important
design property, because it keeps the failure mode of a wrong score at "wasted eighteen minutes"
rather than "declined legitimate purchase".

The second stance is that the probability is the deliverable and the rank is a derived view. Ranks
are unstable at the boundary; probabilities are comparable across days, across capacity changes, and
against the break-even threshold. A system that ships only a rank cannot answer the question in
section 3 the next time capacity changes.

### 8.2 Component view

```text
                     ┌──────────────────────────────────────────┐
  authorisation log →│ ingest + case assembly (dedupe on 3 keys) │→ case table (880k/day)
                     └──────────────────────────────────────────┘
                                        │
                     ┌──────────────────┴───────────────────┐
                     │ feature assembly, 5 families, 90d    │
                     │  windows; graph counts min-support 5 │
                     └──────────────────┬───────────────────┘
                                        │ feature matrix (sparse, 41 cols)
                     ┌──────────────────┴───────────────────┐
                     │ isolation forest fit (daily, 90d)    │
                     │ 300 trees, depth 8, seed recorded    │
                     └──────────────────┬───────────────────┘
                                        │ raw score s ∈ [0,1]
                     ┌──────────────────┴───────────────────┐
                     │ isotonic calibration → p ∈ [0,1]     │
                     │ curve version + window stamped       │
                     └──────────────────┬───────────────────┘
                                        │
                     ┌──────────────────┴───────────────────┐
                     │ queue builder: sort desc p, tie-break │
                     │  on amount desc then case_id asc      │
                     │  cut at K=400; stamp policy version   │
                     └──────────────────┬───────────────────┘
                                        │
              ┌─────────────────────────┴──────────────────────────┐
              │ case store (immutable snapshot) + investigator UI  │
              │  evidence block: p, top-5 deviations, 5 neighbours │
              └─────────────────────────┬──────────────────────────┘
                                        │
                     ┌──────────────────┴───────────────────┐
                     │ outcome capture → label store        │
                     │ 120-day embargo before any evaluation│
                     └──────────────────────────────────────┘
```

Four properties of this pipeline are load-bearing and each has a named owner:

- **Determinism.** Sort with an explicit tie-break, not with a library default. With 880,000 cases
  and a continuous score, exact ties are rare but calibrated scores are quantised by the isotonic
  steps, so ties at the cut are common: dozens of cases can share one probability value. The
  tie-break on amount descending then case_id ascending makes the queue reproducible and, more
  importantly, makes it *stable* — a case near the cut does not jump in and out because of row
  ordering in the source table.
- **Immutability.** The queue snapshot is written once and never rewritten. Late-arriving
  authorisations are handled by a documented correction run that produces a new snapshot with a new
  version stamp, and the desk sees the correction explicitly. Silent mutation of yesterday's queue
  destroys the audit trail and makes any later dispute about "what the desk saw" unanswerable.
- **Versioning.** Population snapshot hash, feature definition version, forest artefact hash,
  calibration curve version, and policy version are all stamped per run. Section 12 shows what
  happens when one of the five is missing.
- **Latency.** The build must finish by 06:00 so the desk starts at 07:00 with a full queue. The
  two-hour budget is the reason feature assembly is batch rather than per-case, and the reason the
  graph family uses trailing-window counts rather than a live traversal.

### 8.3 Calibration design and why it is separate from scoring

The isolation forest produces a score with no probabilistic meaning: a value of 0.62 means "unusual
relative to the fitted sample" and nothing about fraud frequency. Converting it into a probability
is a separate, supervised step that uses matured labels. Three design choices matter.

| Choice | Reason | Failure it prevents |
|---|---|---|
| Isotonic, not logistic | no monotone-linear assumption; the score-to-odds relation is empirically curved | systematic overconfidence in the tail |
| Fit window disjoint from evaluation window | prevents leakage of the evaluated days into the curve | inflated precision at every depth |
| Curve refitted monthly, not daily | label arrivals are bursty; a daily refit chases chargeback-batch noise | queue instability with no change in behaviour |

The monthly refit cadence has a cost that must be accepted explicitly: between refits, the
probability scale can drift if prevalence moves. The guardrail is a prevalence monitor that compares
the realised fraud rate among reviewed cases to the predicted mean probability; a gap beyond 15%
relative triggers an out-of-cycle refit. Without that monitor, a drift in prevalence quietly changes
what "rank 400" means while every component reports healthy.

### 8.4 The evidence block shown to an investigator

An investigator opens a case and must decide within minutes. The UI therefore shows, for each case,
exactly five things and nothing else on the first screen:

1. Calibrated probability, with the phrase "prioritisation signal, not a determination".
2. The five features with the largest normalised deviation from the card's own history, with values.
3. The five nearest historical cases in feature space and whether each was confirmed fraud — with a
   one-line warning that neighbours are drawn from reviewed cases only.
4. The action set as explicit buttons: block credential, cancel order, request merchant contact,
   close, escalate.
5. The reason-code picklist, pre-sorted by the top deviations, so that scheme reporting is a
   by-product of the decision rather than an afterthought.

Showing the neighbours introduces a measurable bias: an investigator who sees three confirmed-fraud
neighbours is more likely to confirm. That is why the pilot's audit cell — cases selected at random
rather than by score — is reviewed by investigators who do *not* see the evidence block. The
protocol difference is stated in the evaluation section, because a comparison between scored and
random cases that also differs in UI information is not a clean comparison of one thing.

### 8.5 Action policy

The probability does not map directly to an action. Two thresholds sit between score and action, and
both are policy parameters owned by the risk owner rather than by the model.

| Band | Probability range | Default action | Justification |
|---|---|---|---|
| Hard | p ≥ 0.85 | block credential + cancel unfulfilled orders, no review | high precision; the residual error is cheaper than the loss |
| Review | 0.147 ≤ p < 0.85 | investigator decides | above break-even, below automatic action |
| Hold | p < 0.147 | not queued; logged for the audit cell | below break-even, so investigation does not pay |

The 0.85 automatic band deserves scrutiny. At p = 0.85 with L_p = 144 and c = 0, the expected loss
avoided is 122.4 EUR per case against friction of 0.15 × 9 = 1.35 EUR, so the band is clearly
profitable. The real objection is not arithmetic but error asymmetry: a blocked credential costs a
legitimate customer a re-enrolment and possibly the relationship, and the customer sees no
explanation. That is why the band is capped at a volume ceiling — no more than 25 automatic blocks
per day — even when the score would justify more. The ceiling is a governance control that
deliberately overrides the arithmetic, and the memo says so rather than pretending the arithmetic
produced it.

### 8.6 Rollout plan

| Phase | Duration | Capacity to new ranker | Gate to proceed |
|---|---|---|---|
| Shadow | 3 weeks | 0% (parallel build only) | queue overlap within expected range; no build failures |
| Audit cell | 90 days, overlapping shadow | 340 of 400 slots | label coverage ≥ 92% by rank band |
| Full switch | quarter 2 | 400 of 400 slots | matured read shows gap > 4 points at depth 400 |
| Capacity case | quarter 3 | 400, then 504 on approval | four hires funded against marginal value |

The shadow phase exists to catch plumbing defects, not to measure value: with a 120-day embargo,
nothing measured in the first three weeks has labels. Its only output is operational — does the
build finish on time, does the case store accept the snapshot, does the UI render the evidence
block, does the desk tolerate the ordering. Those questions are answered in days and save a quarter
of embarrassment.

The audit cell costs 60 slots a day for 90 days, which is 5,400 reviews and 5,400 × 13.50 =
72,900 EUR of investigation capacity diverted. The value forgone is larger than that: at the
anomaly ranker's marginal value, those 60 slots per day would have returned
60 × 53.55 = 3,213 EUR per day, so the forgone value over 90 days is 289,170 EUR, of which
72,900 EUR is paid out as review cost and the remainder is un-prevented fraud. That is a real price
for unbiased measurement, and it is worth paying exactly once — which is why the audit cell must
produce a reusable estimate rather than a one-off comparison.

## 9. Evaluation & Validation

Every metric below is stated with its numerator, denominator, population, window, comparator,
guardrail, uncertainty treatment and execution status. A metric without a denominator is a claim
dressed as a number.

### 9.1 Primary metric — net daily value at fixed capacity

| Field | Specification |
|---|---|
| Numerator | Σ over reviewed cases of (y_i · L_p − c − (1−y_i) · f), EUR |
| Denominator | one business day at K = 400 reviews |
| Population | approved CNP cases for the business date, excluding hard-blocked and 30-day-blocked cards |
| Window | matured cohort of 90 business days, each day read at its own rank 400 |
| Comparator | same population, same K, rule score ordering, same label definition |
| Guardrail | service level: percentage of queued cases opened within 24 h must stay ≥ 99% |
| Uncertainty | day-level bootstrap over business days with the day as the resampling unit |
| Status | arithmetic executed on constructed inputs; no production measurement |

The bootstrap over days rather than cases is the correct choice and is worth naming: cases within a
day share a fraud ring, a merchant, and a season, so resampling cases would understate variance by
roughly the design effect measured in section 6.4, which is about 3. Using cases as the unit is the
single most common way to manufacture false confidence in this kind of evaluation.

### 9.2 Secondary metrics

| Metric | Definition | Target | Why it is secondary |
|---|---|---|---|
| Cumulative precision at depth | fraud confirmed / k reviewed, at k = 100, 200, 300, 400 | ≥ rule + 4 points at 400 | it is an input to net value, not an end |
| Cases confirmed per day | count, not value-weighted | +25 to +40 | ignores loss heterogeneity by construction |
| Queue overlap with incumbent | size of the intersection of the two top-400 sets, divided by 400 | 0.20–0.55 | diagnostic of divergence, not of quality |
| Block precision | confirmed fraud among auto-blocked | ≥ 0.80 | small volume, high consequence |
| False-positive friction cost | non-fraud reviews × f | ≤ 1,500 EUR/day | a guardrail on the primary |
| Investigator minutes per case | median time to disposition | ≤ 20 min | cost input; drifts with UI changes |
| Label coverage by rank band | labelled / reviewed, per 100-band | ≥ 0.92 for 1–400 | validity precondition for all of the above |

Queue overlap is unusual enough to justify its slot. Under the constructed distributions, the two
rankers share only about 46 of 400 cases if their top sets were independent draws with those
precisions and the day's 2,200 fraud cases were the only overlap source — far lower than a naive
expectation of full agreement. If a candidate implementation instead shows overlap near 380, the
suspicion is not that the model is good but that the pipeline is accidentally sorting by the
incumbent score, for example because a legacy field still named `risk_points` is being used as a
tie-break. Overlap is a plumbing check disguised as a quality metric.

### 9.3 Guardrails and kill switches

| Guardrail | Threshold | Action on breach |
|---|---|---|
| Service level breach | < 99% opened in 24 h for 2 consecutive days | revert to rule queue, page desk lead |
| False-positive block rate | > 200 blocks/day | suspend the hard band, review thresholds |
| Auto-block precision | < 0.75 over 14 days | raise the hard-band threshold to 0.90 |
| Feature null rate | > 2% on any novelty feature | disable that feature, rebuild, log incident |
| Score distribution shift | median score moves > 5% vs trailing 14d | freeze calibration refit, investigate mix |
| Label coverage | < 0.85 at any band 1–400 | halt evaluation claims; extend audit cell |
| Queue build duration | > 100 min | reduce forest trees to 200 before the window is missed |

The revert path is deliberately the incumbent policy and not "the previous model version". A rollback
to a version of the same family does not restore a known-good operating state, because the failure
that triggered the rollback may be present in the previous version too.

### 9.4 Uncertainty

Three sources are quantified and one is merely named.

- **Sampling uncertainty**, from the finite cohort: day-level bootstrap, reported as a 90% interval
  on net daily value. Constructed value: EUR 5,508/day. With a day-level standard deviation of 0.06 on the
  precision gap, the day-level standard deviation of net value is 400 × 0.06 × 153 = 3,672 EUR, so
  the 90-day cohort mean has a standard error of 3,672/sqrt(90) = 387 EUR and a 90% interval of
  ±1.645 × 387 = ±637 EUR. A single day's interval is ±6,040 EUR, which is why a one-day read
  cannot settle anything.
- **Label uncertainty**, from outcomes that may resolve after the embargo: a 3% late-resolution rate
  moves precision by at most 0.019 in either arm, which is small relative to the 0.09 gap but not
  negligible against a 0.04 decision threshold.
- **Model uncertainty**, from the choice of forest seed and hyperparameters: a 10-seed rerun of the
  same cohort typically moves precision at depth 400 by 1–2 points, so the 9-point gap is not a seed
  artefact, but a 4-point gap would be.
- **Structural uncertainty**, named but not quantified: the counterfactual "what would the rule queue
  have found if it had read deeper" is not observable, because deeper cases were never reviewed
  under the old policy except in the audit cell.

### 9.5 Executed versus specified

| Check | Status in this manuscript |
|---|---|
| Break-even arithmetic | executed, hand-checked |
| Fitted precision law and marginal value | executed, three points resubstituted |
| Net value and annualisation | executed, component cross-check |
| Sample-size and design-effect arithmetic | executed with an ILLUSTRATIVE day variance |
| Composition PSI monitor definition | specified, not executed (no production stream) |
| Label-coverage audit by rank band | specified with sampling plan, not executed |
| 10-seed stability rerun | specified, not executed |
| Day-level bootstrap interval | specified method; the ±1,600 figure is ILLUSTRATIVE |
| Leakage audit on feature provenance | specified; the 99.5% threshold is a design target |

The distinction is the point of the table. A reader who cannot tell an executed check from a
proposed one will over-trust the manuscript, and the failure is not in the arithmetic but in the
reader's inference about it.

### 9.6 The comparison protocol that makes the numbers valid

Two rankers cannot be compared unless four things are held equal. The protocol fixes them:

1. **Same population snapshot.** Both queues are built from one frozen case table with one hash.
2. **Same depth.** Both read 400. Any depth difference is reported separately, per section 6.3.
3. **Same review protocol.** Investigators do not know which ranker produced the case; the case
   store hides the policy field from the UI, and the audit cell is reviewed blind to both.
4. **Same label cut-off.** Both arms are scored against the label snapshot taken at the same instant
   with the same 120-day embargo.

If any of the four is broken, the comparison must be labelled "indicative" and cannot be used as the
gate in section 8.6. This is stated as a rule rather than a preference because the gate is the only
place where the organisation commits money.
## 10. Action Taken

### 10.1 The decision

The risk owner approved the following, recorded as four decisions rather than one so that each can
be reversed independently:

| # | Decision | Scope | Reversal condition |
|---:|---|---|---|
| D1 | Adopt the calibrated anomaly ranker as the queue sort | 400 slots/day from quarter 2 | matured gap < 0.04 at depth 400 |
| D2 | Keep headcount at 18 through quarter 2 | no hiring this cycle | none; fixed by budget |
| D3 | Fund a 60-slot/day biased-free audit cell for 90 days | 15% of capacity | label coverage ≥ 0.92 reached earlier |
| D4 | Defer the four-hire case to quarter 3, priced on marginal value | +104 slots/day | none |

D3 is the decision that most organisations skip, and skipping it is what turns D1 into an
unfalsifiable claim. The audit cell is the only mechanism that can produce a precision estimate at
ranks 401–3,000, and without it every statement about the new ranker is conditioned on the old
ranker's selection.

### 10.2 The sequencing argument, stated as arithmetic

The temptation is to fund capacity and ranking together. The arithmetic says otherwise:

- Adding 104 slots at the *rule* ranker's depth-504 precision (0.504) yields
  104 × 0.504 × 144 − 104 × 13.50 − 104 × 0.496 × 9 = 7,548 − 1,404 − 464 = 5,680 EUR/day.
- Adding the same 104 slots at the *anomaly* ranker's depth-504 precision (0.600) yields
  104 × 0.600 × 144 − 1,404 − 104 × 0.400 × 9 = 8,986 − 1,404 − 374 = 7,208 EUR/day.
- Ranking first, capacity second, captures 7,208 rather than 5,680 — a 27% difference on the
  expansion, for zero additional cost.

That is the whole sequencing argument in three lines. It is not that capacity is unimportant; it is
that capacity bought before the sort is fixed reviews a worse list, and the difference is 1,528 EUR
per day, or 382,000 EUR a year at 250 business days.

### 10.3 What was explicitly refused

Three requests were declined, and the refusals are part of the decision record:

1. **Add authorisation-time decline for p ≥ 0.95.** The latency budget is 120 ms at the checkout and
   the scoring path takes 8–15 s in batch. Serving per-transaction would require a different feature
   set (no 24-hour aggregates) and a challenge flow that does not exist. Declined as out of scope,
   with a note that it is the highest-value next programme.
2. **Retire the 34 rules immediately.** The rules remain as hard blocks and as a fallback sort. The
   fallback is exercised in the guardrail table, and an untested fallback is not a fallback.
3. **Extend the queue to 700 to "capture more fraud".** At depth 700 the marginal value is still
   positive — the fitted law gives P(700) = 2.23068 × 700^(−0.211025) = 0.5598, marginal precision
   0.788975 × 0.5598 = 0.4417, net 0.4417 × 153 − 22.50 = 45.08 EUR — but the desk cannot staff 700
   reviews. A queue longer than capacity converts value into backlog, so the request was answered
   with a hiring case rather than a queue change.

The third refusal contains the correction of a mistake that appears in many drafts of this kind of
memo: quoting the *average* value per slot when defending a queue extension. The average at depth
400 is 73.89 EUR; the marginal at 700 is 46.7 EUR. The decision is always about the marginal.

### 10.4 Implementation sequence with owners

| Week | Milestone | Owner | Done when |
|---:|---|---|---|
| 1–2 | Feature registry v1 with provenance timestamps | data platform | every field has a write-time contract |
| 2–3 | Shadow queue build, side by side with rules | data science | 10 consecutive builds match rerun byte-for-byte |
| 3–4 | Case-store schema with policy version and blind field | platform | UI hides the ranker identity for test users |
| 4–6 | Calibration on the most recent matured 180 days | data science | curve reproduces held-out precision within 3 points |
| 6–7 | Desk training: what the probability means, what it does not | ops manager | 18/18 investigators complete the 45-minute module |
| 8–20 | Audit cell live at 60 slots/day | ops + analytics | coverage by band recorded daily |
| 21 | First matured read at 90-day cohort | analytics | gap estimate with day-bootstrap interval |
| 22 | Gate review | risk owner | D1 confirmed or reversed on the 0.04 rule |

The week-1 milestone is the one that is always compressed and never should be. A feature without a
provenance write-time contract cannot be audited for leakage, and a leakage audit that happens after
the model is built has one honest outcome: rebuild.

### 10.5 The monitoring that was actually instrumented

Five dashboards, each answering a question that a different failure would raise:

- **Queue health**: build duration, row counts, null rates, tie-group size at the cut.
- **Score behaviour**: score histogram versus trailing 14 days, median shift, share above 0.85.
- **Desk behaviour**: cases opened per hour, median minutes per case, disposition mix, reopen rate.
- **Outcome flow**: labels arriving per day by age bucket, coverage by rank band, late-resolution
  rate.
- **Value**: net daily value with day-bootstrap band, block precision, friction cost.

The desk-behaviour dashboard exists because a queue change changes investigator behaviour, and that
change is a confound in every later comparison. If median minutes per case falls from 19 to 14 in the
first month, part of any precision gain is attributable to shallower investigation rather than better
ranking, and the comparison protocol in section 9.6 must be amended to control for it.

## 11. Outcome & Results

### 11.1 What the constructed evaluation shows

These figures are the output of the arithmetic in section 6 applied to constructed inputs. They are
not measurements of any real deployment, and the labelling is deliberate: `ILLUSTRATIVE — derived`.

| Quantity | Rule queue | Anomaly queue | Difference |
|---|---:|---:|---:|
| Reviews per business day | 400 | 400 | 0 |
| Precision at depth 400 | 0.540 | 0.630 | +0.090 (+16.7% rel.) |
| Confirmed fraud cases per day | 216.0 | 252.0 | +36.0 |
| Gross prevented loss per day, EUR | 31,104 | 36,288 | +5,184 |
| Review cost per day, EUR | 5,400 | 5,400 | 0 |
| Friction cost per day, EUR | 1,656 | 1,332 | −324 |
| Net value per day, EUR | 24,048 | 29,556 | +5,508 |
| Annualised net value, EUR (250 days) | 6,012,000 | 7,389,000 | +1,377,000 |
| Marginal slot value at rank 400, EUR | 60.12 | 53.55 | −6.57 |

The last row is the one an inexperienced reader will find contradictory: the anomaly queue has a
*lower* marginal slot value than the rule queue, yet is clearly better. Both statements are true and
they are not in conflict. The rule queue's marginal case at 400 is 0.540 precision against a
break-even of 0.147, so it is still profitable; the anomaly queue reaches lower precision by rank 400
because it has already captured more of the easy cases earlier. A better ranker exhausts the
high-precision region sooner, so its marginal case at a fixed depth is worth less. Anyone who uses
marginal value as a *quality* metric rather than a *capacity* metric will conclude the opposite of
the truth.

### 11.2 The quarter-one result under the audit-cell protocol

The audit cell changes the arithmetic while it runs, and the honest report must show that:

| Phase | Slots to anomaly ranker | Slots to audit cell | Net value/day, EUR | Notes |
|---|---:|---:|---:|---|
| Status quo | 0 | 0 | 24,048 | rule queue, no measurement |
| Quarter 1 (audit) | 340 | 60 | 26,266 | unbiased labels accumulate |
| Quarter 2 (full) | 400 | 0 | 29,556 | gate cleared on matured read |
| Quarter 3 (capacity) | 504 | 0 | 34,928 | four hires; marginal 49.93/slot |

Quarter 1 gives up 29,556 − 26,266 = 3,290 EUR/day versus the unmeasured full switch, which is
296,093 EUR over the 90-day cell. That is the price of the answer to "is the gap real", and the
alternative — switching blind — risks locking in a change that cannot be evaluated for four months
while its claimed benefit is asserted rather than measured.

Quarter 3 is included because it shows the arithmetic the hiring case rests on. The increment from
recomputing total value at K = 504 rather than adding per-slot estimates is
34,928 − 29,556 = 5,372 EUR/day, or 1,343,000 EUR a year against 300,000 EUR of fully loaded
investigator cost (four staff at 75,000) — a 4.5× return on the incremental spend. Multiplying 104 by
the depth-504 marginal of 49.93 gives 5,192 EUR/day, which understates the increment by 180 EUR/day
because the marginal declines across the added range; the full recomputation (average marginal
51.65 EUR per added slot) is the figure the hiring case must use.

### 11.3 Confirmed-case counts and what they hide

36 additional confirmed cases per day is 9,000 a year. That number will be quoted in the memo and it
should be, with one caveat attached: confirmed cases are not equal in value. The average loss is
EUR 180, but the distribution is heavy-tailed — a fraud ring hitting a single merchant can produce
losses an order of magnitude above the mean. If the anomaly ranker disproportionately catches
small-value card-testing bursts (which are numerous and individually cheap) rather than large-value
account takeover (which is rare and expensive), the count gain could coexist with a smaller value
gain, or even a value loss.

The check is a loss-weighted precision curve, computed alongside the count-based one. It is
specified in section 9.2 as "secondary" and it should arguably be primary: the whole point of the
exercise is money, and case counts are a proxy that can move in the opposite direction. In this
constructed setting the assumption A2 (loss independent of rank) is doing real work, and section 4
already flags the check that would falsify it — loss by score decile over a 12-month window.

### 11.4 What did not change

Three things were expected to move and did not, which is as informative as the things that did:

- **Investigator throughput.** Median minutes per case stayed at 19 (constructed), because the
  evidence block replaced time previously spent searching card history. Net desk capacity was
  unchanged, which is why the capacity arithmetic could hold K fixed.
- **Block volume.** Automatic blocks stayed under the 25/day ceiling, so the hard band did not
  expand even though the calibration made it available.
- **Merchant escalations.** No measurable change in relationship-manager escalations, because the
  flagged merchant mix shifted toward small accounts rather than large ones.

The third is a warning rather than a success. A queue change that redistributes attention across
merchants has political consequences even when it improves the aggregate, and a case that reports
only the aggregate will be surprised by the escalation three months later.

### 11.5 The result a sceptic should demand

The single most useful artefact from this programme is not the lift figure. It is the pair of
precision curves, by depth, with intervals, on one frozen population, under one review protocol — a
chart that can be re-derived from a manifest six months later. If that artefact does not exist, the
lift figure is a story; if it exists, the next three decisions in this area become cheap.

## 12. Failure Modes & What Went Wrong

Eight mechanisms, each with the observation that distinguishes it from its nearest competitor. The
distinguishing checks matter more than the mechanisms: two failures that look identical from a
dashboard and require opposite fixes are the normal condition of production triage systems.

### 12.1 Calibration drift after a prevalence shift

**Mechanism.** The isotonic curve is fitted monthly on matured labels. If the underlying fraud rate
rises between refits — a new attack, a seasonal shift, a change in the issuer's dispute behaviour —
the mapping from score to probability becomes stale, and the break-even rule is applied to a
probability that no longer means what it says. The rank order is unaffected, so the queue looks
normal and precision at depth 400 looks normal; only the *thresholded actions* are wrong.

**Competing causes.** (a) A genuine rise in fraud prevalence; (b) a change in the label definition
by the scheme, which also raises measured prevalence; (c) a change in the review protocol that
changes which cases get confirmed.

**Discriminating check.** Compare the realised fraud rate among reviewed cases to the mean predicted
probability for those same cases. If the realised rate exceeds prediction by more than 15% relative
while the label definition and review protocol are unchanged (verified by their own version stamps),
it is drift (a). If the label definition version changed, it is (b) and the fix is a re-fit, not a
model change. If the audit-cell sampling rate changed, it is (c).

**Correction.** Out-of-cycle refit on the most recent matured 60 days; if fewer than 400 matured
labels are available in that window, hold the curve and widen the review band instead.

### 12.2 The feedback loop that makes the model self-confirming

**Mechanism.** Blocked credentials generate no further transactions. If any feature is computed from
"transactions on this credential", a blocked card drops out of the feature population and its
aggregates freeze at the pre-block values. Over months, the model's training population consists
disproportionately of credentials the model did not block, and its notion of normal narrows toward
the behaviour it already accepts.

**Competing causes.** (a) Feature computation truncating history at block time; (b) a shift in the
merchant mix toward categories with lower transaction counts; (c) a genuine reduction in fraud due to
the blocks, which is the intended effect.

**Discriminating check.** Hold out blocked credentials from the fit and rerun: if the score
distribution on never-blocked credentials shifts materially, the loop is present. A second check
compares the distribution of feature values for credentials blocked 30+ days ago against matched
never-blocked credentials at the same tenure.

**Correction.** Compute features from a credential's *full* history including post-block inactivity
as an explicit state, and exclude blocked spells from the fitting population with an indicator
rather than by omission.

### 12.3 Timestamp and processing-lag contamination

**Mechanism.** The case population is defined on event time (authorisation date), but enrichment
fields arrive at processing time. A device-intelligence lookup that lands 6 hours late is missing for
the last 6 hours of the business date, so late-day cases get a default novelty value and score
either too high (treated as new device) or too low (treated as known). The queue then has a
systematic hour-of-day bias that looks like a behavioural pattern.

**Competing causes.** (a) Late enrichment; (b) a genuine diurnal fraud pattern; (c) a time-zone error
in the business-date boundary, shifting which cases belong to which day.

**Discriminating check.** Plot the null rate of every novelty feature by hour of business date. A
null rate that climbs after hour 18 and a score distribution that shifts with it identifies (a).
A genuine diurnal pattern shows no null-rate gradient. A boundary error shows the same gradient
inverted on the following day's early hours.

**Correction.** Lag the affected feature by one business day and accept the loss of freshness, or
impute with the card's own prior value plus an explicit missing indicator — never with a global
default, which is what makes the bias directional.

### 12.4 Label coverage collapse at the depth that matters

**Mechanism.** Only reviewed cases carry confirmed outcomes. Under the incumbent policy the reviewed
set is the incumbent's top 400, so the label store has near-zero coverage at ranks 401–3,000. When the
new ranker promotes a case from rank 2,000 to rank 150, that case has no label; when the precision
calculation treats unlabelled as non-fraud, the new ranker's measured precision falls.

**Competing causes.** (a) Genuine lower precision of the new ranker; (b) coverage-driven
under-measurement as described; (c) a join defect losing labels that do exist.

**Discriminating check.** The coverage table by rank band is the discriminator. If coverage is 0.95
for ranks 1–200 and 0.30 for ranks 200–400, the measured precision gap is partly a measurement
artefact. To separate (a) from (b), restrict both arms to the covered subset and recompute: if the
gap survives on covered cases only, (a); if it vanishes, (b).

**Correction.** The audit cell is the structural fix. The tactical fix is to report precision only for
bands with coverage ≥ 0.60 and to mark the rest as unknown rather than zero.

### 12.5 Queue instability at the cut from score quantisation

**Mechanism.** Isotonic calibration maps many distinct raw scores to the same probability. At the
400-case cut, the tie group can contain 15–40 cases. Any change in the tie-break, or in row ordering,
or in a feature's null handling, reshuffles the group and changes which cases the desk sees — without
any change in model quality. The desk perceives randomness, loses trust, and the quarter-on-quarter
comparison gains a variance component that is pure plumbing.

**Competing causes.** (a) Quantisation ties; (b) a genuinely unstable model (seed or hyperparameter
sensitivity); (c) population change moving cases across the boundary legitimately.

**Discriminating check.** Rerun the build on the identical frozen snapshot: a byte-identical queue
rules out (c). Rerun with 10 different seeds: if the top-400 set changes by more than 5% across
seeds, it is (b). If seeds barely matter but the tie group is large, it is (a).

**Correction.** Break ties on a stable secondary key that correlates with value (amount, then
case_id). Better: report the queue as "all cases with p ≥ p_cut" plus an explicit overflow list,
so the boundary is defined by a threshold rather than by a rank.

### 12.6 Capacity arithmetic applied to the wrong quantity

**Mechanism.** The hiring case uses average value per slot instead of marginal value at the expanded
depth. At K = 400 the average anomaly-slot value is 73.89 EUR; at K = 504 the marginal slot is worth
49.93 EUR. Using 73.89 for 104 new slots overstates the daily return by
104 × (73.89 − 49.93) = 2,492 EUR, or 623,000 EUR a year — larger than the entire cost of the four
hires.

**Competing causes.** (a) Using average instead of marginal value; (b) using precision at depth 400
for decisions at depth 504; (c) using gross prevented loss instead of net.

**Discriminating check.** Recompute the expansion value three ways — average-slot, marginal-slot, and
full recomputation at the new K — and compare. If the three disagree by more than 10%, at least one
of (a)–(c) is present, and the full recomputation is the only defensible figure.

**Correction.** Always recompute total net value at the proposed K rather than adding a per-slot
increment to the current total. The increment method is only valid if the marginal value is treated
as constant, which it demonstrably is not.

### 12.7 The evaluation that reads the wrong depth

**Mechanism.** A pilot evaluated at depth 50–100 shows a 2.4-point gap that is inside noise and is
declared inconclusive, cancelling a programme worth 1.4 million a year. The mechanism is that the
two rankers are near-identical where the rule set has its strongest signals, and diverge where the
rule set runs out of them.

**Competing causes.** (a) Reading a depth that is not the operating depth; (b) a genuinely null
effect; (c) insufficient cohort for the depth used.

**Discriminating check.** Compute the gap at every 50-case increment from 50 to 600 with intervals.
A gap that grows monotonically with depth identifies (a). A gap that stays within ±0.02 everywhere
identifies (b). A gap that grows but with intervals straddling zero everywhere identifies (c), and
the fix is a longer cohort, not a deeper read.

**Correction.** Fix the read depth to the operating capacity before the pilot starts, and state it in
the pre-registration. Changing depth after seeing results is how a null becomes a win and a win
becomes a null, depending on who is arguing.

### 12.8 Silent model swap behind an unchanged queue interface

**Mechanism.** The queue builder consumes a score field whose producer is a separate service. That
service is redeployed with a new forest artefact and the same interface. Nothing in the queue record
changes except the artefact hash, which nobody reads. Precision moves by 4 points and the
investigation goes to the desk, the features, and the label store — everything except the one thing
that changed.

**Competing causes.** (a) Unrecorded model change; (b) feature drift; (c) label-store lag creating an
apparent change in recent data.

**Discriminating check.** Diff the run manifests for the two dates. A differing artefact hash with
identical feature versions and identical population hash is conclusive for (a). If the manifests are
identical, compare feature distributions (b) and label arrival curves (c).

**Correction.** Make the artefact hash a first-class field in the queue record and a gate in the
publish step: a queue whose artefact hash is not on the approved list cannot be published. This is a
five-line control that prevents a class of investigation that routinely consumes two weeks.

### 12.9 The cross-cutting lesson from the eight

Six of the eight mechanisms produce a queue that is *plausible* — right length, right shape, no
errors. The failure is not visible in the artefact; it is visible only in a comparison against a
frozen reference, or in a monitor that measures the thing the artefact is supposed to mean rather
than the thing it is. Monitoring the model's output distribution catches none of them. Monitoring
calibration drift, coverage, manifest diffs, and depth-specific gaps catches most.

## 13. Transferable Lesson

### 13.1 The lesson in one paragraph

In a capacity-constrained triage system, the value of a ranking change is a property of the *depth
you read*, not of the model. Before evaluating any ranker, fix the operating depth to the resource
you actually have, compute the break-even probability implied by your costs, and check whether that
depth is above or below it. Then measure at that depth, on a population that was not selected by the
policy you are replacing, over a cohort long enough to mature its labels. A team that does these
three things will usually discover that the ranking is easier than the measurement, and that the
measurement is where the quarter goes.

### 13.2 The four transferable propositions

| # | Proposition | Evidence in this case | Where it transfers |
|---:|---|---|---|
| T1 | Read depth is a measurement design parameter | 2.4-point gap at 100, 9.0 at 400; 114 days vs 3.5 days of data | any top-K triage: alerts, claims, churn, moderation |
| T2 | Marginal value, not average, sets capacity | 73.89 average vs 53.55 marginal at 400 | hiring, queue length, SLA negotiation |
| T3 | Selection bias runs both ways and nets unpredictably | label coverage 0.95 top vs 0.30 mid-band | any evaluation on data the incumbent policy produced |
| T4 | Plausible queues hide wrong arithmetic | six of eight failure modes leave a normal-looking artefact | monitoring design, release gates |

### 13.3 The counter-lesson: where this reasoning does not transfer

The whole apparatus assumes a *stable, well-defined binary label with a maturity lag*. Where the
label is immediate (a click, a cart abandonment, a server error), the measurement problem collapses
and the modelling problem dominates; the sequencing advice in section 10 would be wrong there, and a
team that spent a quarter building an audit cell for click-through would have wasted it.

The second boundary is capacity. If capacity is effectively unlimited — a human-in-the-loop review
where the reviewer is a cheap classifier or a script — then the break-even probability falls toward
zero and the only metric is total net value, which favours recall over precision. Most of the
argument here exists because attention is scarce; remove that and the ranking question changes shape.

The third boundary is action reversibility. This case leans on the fact that a false positive costs
EUR 9 of friction, not a lost customer. In a domain where a false positive is a denied loan or a
closed account, the friction term dominates the loss term and the break-even probability can exceed
0.9, at which point no triage system is worth building without an appeal process.

### 13.4 Three habits worth carrying

1. **Write the kill condition before the pilot.** "Precision at depth 400 is not better by more than
   the day-level standard error" is a sentence that can end a programme honestly. Without it, the
   programme ends on the basis of who is in the room.
2. **Price the measurement.** The audit cell cost 296,093 EUR of forgone value. Naming that number
   converts "we should validate" from a virtue into a line item that someone can approve or reject.
3. **Separate the artefact from the claim.** The precision curve on a frozen population is an
   artefact; "the model is better" is a claim. Only artefacts survive a change of personnel, which
   is the actual time horizon of a risk programme.

### 13.5 What I would do differently

Two things, stated plainly because they are the most likely reader questions.

First, the calibration would be fitted on *loss-weighted* outcomes rather than binary fraud. The
decision is about money, and a calibration that predicts P(fraud) rather than E[loss] forces the
organisation to assume loss is constant across ranks — assumption A2, which section 4 already flags
as the most likely to be false. A two-part model, P(fraud) times E[loss | fraud], would remove the
assumption at the cost of a noisier fit; the second part has far fewer observations and would need
pooling across merchant categories.

Second, the audit cell would sample *stratified* by rank band rather than uniformly across ranks
401–3,000. Uniform sampling spends most of its 60 daily slots in the dense middle of the range where
coverage is already adequate, and too few at ranks 350–450 where the decision boundary lies. A
stratified design with 20 slots in 401–600 and 40 spread over 601–3,000 would give a tighter interval
on the quantity that matters for the same cost.

## 14. Connections & Cross-References

### 14.1 Inside this unit

| Artefact | Connection | What this case takes or gives |
|---|---|---|
| `Cheat_Sheets/01_Anomaly_Detection_Triage.md` | theory of the triage decision | takes the break-even and maturity arithmetic; gives a worked capacity case |
| `Cheat_Sheets/02_Clustering_and_Embeddings.md` | grouping structure in behaviour | gives the graph-feature caution; takes the note that clusters are not labels |
| `Cheat_Sheets/03_Change_Point_Detection.md` | detecting regime shifts | gives the prevalence-shift trigger in 12.1; takes the drift test |
| `Cheat_Sheets/04_Segmentation_Evaluation.md` | evaluating unsupervised structure | shares the coverage-by-band discipline |
| `Cheat_Sheets/05_Human_Review_Loops.md` | reviewer capacity and behaviour | gives the desk-behaviour confound in 10.5 |
| `Cheat_Sheets/06_25_Common_Issues.md` | failure catalogue | four of its issues are instances of 12.1, 12.4, 12.5, 12.8 |
| `Case_Studies/02_customer_segmentation_for_campaigns.md` | sibling case, same unit | contrast: segments are chosen, queues are filled |

### 14.2 Outside this unit

- **Calibration and probability** — `probability-calibration` pattern; isotonic regression, proper
  scoring rules, reliability diagrams. The case uses calibration as a component and does not derive
  it.
- **Queueing and service levels** — the 24-hour SLA and the backlog arithmetic belong to operations
  research, not to machine learning. The 400/468 ratio (0.855 utilisation) is a queueing parameter.
- **Graph features and privacy** — the 7-day card-sharing-device count is a cross-customer aggregate.
  Minimum support counts and aggregation-only storage are privacy engineering, treated in the data
  governance track.
- **Scheme rules and dispute evidence** — the label definition is a legal artefact before it is a
  data artefact. A change in scheme rules changes the label, which changes every metric in section 9.
- **Model risk governance** — the artefact-hash gate in 12.8, the kill conditions in 3.3, and the
  executed/specified table in 9.5 are governance controls, not engineering ones.

### 14.3 Contrast with the sibling case

The sibling case in this unit selects customer segments for campaigns. The structural difference is
worth stating because it explains why the same vocabulary produces different arithmetic:

| Dimension | Payment triage (this case) | Campaign segmentation (sibling) |
|---|---|---|
| Who is chosen | cases, then investigated | segments, then targeted |
| Scarcity | investigator slots, hard-capped | contact budget and attention |
| Label | binary fraud, 1–120 day lag | incremental response, needs a holdout |
| Core risk | missing a fraud, label selection bias | segment is descriptive, not causal |
| Core number | precision at depth, marginal value | incremental contribution per contact |
| Failure that looks like success | high precision on a shallow read | high response rate with no counterfactual |

The last row is the shared lesson in two costumes: both cases are about a number that improves when
you measure the wrong thing, and in both the fix is a fixed comparison protocol rather than a better
model.

### Interview pack

**Q1. A colleague reports that the anomaly model beats the rules by 2.4 points at the top of the
queue. What is the first question you ask, and what would you do with the answer?**
Answer: I ask at what depth, and whether that depth is the operating capacity. At depth 100 the
constructed gap is 0.024 with a standard error of 0.053, so it is indistinguishable from zero; at
depth 400 the same pair of rankers differ by 0.090. If the answer is "top 100" and the desk reads
400, I would re-measure at 400 on a matured cohort before spending anything.
Reasoning: rankers are usually near-identical where the strongest signals dominate and diverge as the
signal thins; the top of the queue is the least informative place to compare two ranking functions.
Follow-up: what cohort length would you need at each depth? Answer: about 3.5 business days of
matured data at depth 400 given a day-level standard deviation of 0.06, and about 114 days at depth
100 — which is why the shallow read is not just uninformative but unaffordable.
Red flag: answering "check the AUC" — a global ranking metric does not answer a capacity-constrained
question and can be identical while the queue is different.
Scoring: full credit requires naming depth as the flaw and proposing a depth-matched re-measurement.

**Q2. Compute the break-even fraud probability for a review slot given L = 180 EUR, φ = 0.80,
c = 13.50 EUR and f = 9.00 EUR. Interpret the result against a queue cut at 0.63.**
Answer: p* = (c + f)/(φL + f) = 22.50/153 = 0.14706. A slot is worth buying above 14.7% fraud
probability. At 0.63 the marginal reviewed case clears break-even by 4.3×, so the queue is
economically under-sized rather than over-sized.
Reasoning: the numerator is the cost of a review plus the friction of a false positive; the
denominator is the loss avoided plus the friction you do not pay when the case is genuine.
Follow-up: recompute with φ = 0.30 (a 48-hour latency). Answer: p* = 22.50/(54+9) = 0.357, so a
slower desk raises the bar to 35.7% and destroys roughly three-quarters of the per-slot value at
fixed precision.
Red flag: computing c/(φL) = 0.094 and ignoring friction — a 56% understatement of the threshold in
this cost structure.
Scoring: correct formula and value = pass; naming the latency sensitivity without prompting = strong.

**Q3. Your queue builder produces 400 cases, but 38 of them share the same calibrated probability at
the cut. What is the defect and how do you fix it in one release?**
Answer: isotonic calibration quantises the score, so the cut falls inside a tie group and the
selection within the group is decided by row ordering. Fix: define the boundary by a probability
threshold rather than a rank, publish "all cases with p ≥ p_cut" plus a deterministic overflow list
ordered by amount then case_id.
Reasoning: a rank-based cut on a quantised score is only stable if the tie-break is a total order on
a business-relevant key.
Follow-up: how would you report precision for such a queue? Answer: precision over the thresholded
set, with the overflow list reported separately — mixing them makes the metric depend on how many
ties happened to exist that day.
Red flag: "increase the number of isotonic bins" — that changes the calibration's variance properties
and does not address the undefined selection inside a tie.
Scoring: identify quantisation as the cause and propose threshold-based publication for full credit.

**Q4. The 34 rule score is additive over binary indicators. Explain mechanically why an anomaly
score can beat it at depth 400 but not at depth 50.**
Answer: an additive score can only express main effects; a rare *joint* configuration — new device
plus unusual hour plus first-time merchant category — has no term for its interaction, so it scores
as a sum of individually weak signals. At the top of the queue, a handful of very strong single
signals dominate and both rankers find the same cases. Deeper, the additive score runs out of strong
signals and starts ranking on individually weak ones, while the fitted joint distribution still
separates rare combinations.
Reasoning: it is a model-class argument about interaction coverage, not about the specific algorithm.
Follow-up: what cheap experiment would test that explanation? Answer: logistic regression on the same
34 indicators. If calibrated weights recover most of the gap, the gain is from calibration and
interactions, not from anomaly detection, and the next investment should be feature work.
Red flag: attributing the gap to "the model learning non-linear patterns" without an experiment to
separate calibration from interaction coverage.
Scoring: full credit for the interaction-coverage mechanism plus the logistic-regression test.

**Q5. You are told only reviewed cases have labels. Explain the direction of the bias in each arm of
the comparison, and why the net sign is not obvious.**
Answer: cases the old rules ranked low and the new model ranks high are under-labelled, so the new
model's measured precision at those ranks is understated. Cases the old rules ranked high and the new
model ranks low are over-labelled, so the incumbent's measured precision is overstated. Both biases
inflate the apparent advantage of the newcomer; but the second effect also inflates the measured
precision of the newcomer for the cases it retains, so the net sign depends on how the two sets
overlap. It must be measured, not assumed.
Reasoning: precision is a ratio of labelled positives to reviewed cases; changing which cases are
reviewed changes both.
Follow-up: how would you fix the measurement? Answer: a randomly reviewed audit cell at ranks beyond
the incumbent cut, blind to ranker identity — 60 slots a day for 90 days in this design, at a cost of
about 296,000 EUR of forgone value.
Red flag: "the bias favours the incumbent" or "the bias favours the challenger" stated as a fact
without deriving which cells are mislabelled.
Scoring: naming both directions and refusing to assert the net sign is the full-credit answer.

**Q6. Capacity is expanded from 400 to 504 slots. Compute the daily value of the expansion if the
marginal slot at depth 504 is worth 49.93 EUR, and explain why the average slot value would give a
wrong answer.**
Answer: 104 × 49.93 = 5,192 EUR/day, or 1,298,000 EUR a year at 250 days, against 300,000 EUR of
fully loaded cost for four investigators — a 4.3× return. Full recomputation of total net value at
504 slots gives 34,928 − 29,556 = 5,372 EUR/day, so the per-slot route understates by 180 EUR/day;
the difference is the decline of the marginal across the 104 added slots, whose average is 51.65. Using the average value at depth 400
(73.89 EUR) would give 104 × 73.89 = 7,684 EUR/day and overstate the case by 2,492 EUR/day, which is
more than the annual cost of the hires.
Reasoning: the marginal value declines with depth, so an increment must be priced at the value of the
last slot, not the mean of the existing ones.
Follow-up: where does the marginal value cross zero in this cost structure? Answer: at precision
0.147, which the fitted law reaches far beyond any staffing level — so capacity is limited by labour
supply and the change freeze, not by economics.
Red flag: adding 104 × average to the current total — it double-counts value already captured.
Scoring: correct per-slot multiplication plus recognition of the average/marginal gap.

**Q7. A dashboard shows the score histogram is stable, build times are normal, and the queue length is
400. Name two failures that this dashboard cannot detect.**
Answer: calibration drift (the score distribution can be unchanged while the score-to-probability
mapping is stale) and label-coverage collapse at the depth being read (the queue is the right length
and the labels are simply absent where the decision is made). A third: a silent artefact swap behind
an unchanged interface.
Reasoning: the dashboard measures the output's shape, not its meaning or its evidence base.
Follow-up: what single monitor would you add first? Answer: realised fraud rate among reviewed cases
versus mean predicted probability, because it detects drift and, with a coverage breakdown, also
detects the measurement failure.
Red flag: "add alerting on the number of alerts" — that measures volume, which was already stable.
Scoring: two genuinely distinct mechanisms with a reason each is invisible.

**Q8. You need a two-sided test at α = 0.05, power 0.80, to detect a 0.09 precision gap. The
day-level standard deviation of the gap is 0.06. How many business days?**
Answer: D = ((1.960 + 0.842) × 0.06 / 0.09)² = (2.802 × 0.6667)² = (1.868)² = 3.49, so 4 days.
Reasoning: the numerator is the sum of the standard normal quantiles for the two error rates; the
ratio to the effect size converts to a standard-error count, and squaring converts to a variance
ratio against the per-day variance.
Follow-up: the binomial standard error at depth 400 is 0.0347, which would give 1.2 days. Why is the
empirical figure larger? Answer: cases within a day are correlated through fraud rings and shared
merchants; the design effect is (0.06/0.0347)² = 2.99, so the effective sample is a third of the
nominal count.
Red flag: using the binomial standard error without checking the design effect — a three-fold
underestimate of the required cohort.
Scoring: correct arithmetic and the design-effect explanation.

**Q9. The desk's median minutes per case falls from 19 to 14 in the first month after the switch.
What does that do to your evaluation, and what do you check?**
Answer: it confounds the comparison, because shallower investigation changes the confirmation rate
in both arms but not symmetrically — the arm with richer evidence blocks is investigated less
thoroughly relative to its information content. I would check the disposition mix and the reopen rate
by arm; a rising reopen rate indicates premature closing, which suppresses measured precision.
Reasoning: the label is produced by human investigation, so investigator behaviour is part of the
measurement instrument.
Follow-up: how would you control for it? Answer: fix a per-case time budget for the audit cell and
report precision at fixed time rather than at fixed disposition; or randomise evidence-block richness
within each arm.
Red flag: treating investigator time as a pure efficiency win with no effect on the label.
Scoring: names the instrumentation effect and proposes a control.

**Q10. Your isolation forest is axis-aligned. Give a fraud pattern that it will systematically miss,
and one that it will systematically over-flag.**
Answer: it will miss a pattern defined by a *correlation* — e.g. amounts that are normal for the
merchant category but paired with an hour that is normal for the cardholder, where the joint
combination is impossible for a genuine cardholder. Axis-aligned splits cannot isolate a diagonal
region efficiently. It will over-flag legitimate multivariate outliers such as a first purchase in a
new category at an unusual hour by a genuine customer, because "rare in the joint distribution" is
not "fraudulent".
Reasoning: the score is a function of how easily a point is isolated by axis-parallel cuts, which is a
statement about marginal sparsity, not about the joint structure that defines the fraud.
Follow-up: what would you add? Answer: a small set of explicit ratio or residual features (amount
relative to the card's category-specific median, hour relative to the card's own hour distribution)
to linearise the diagonal, or a density model if the cost budget allows.
Red flag: "use more trees" — tree count reduces variance, not the axis-alignment bias.
Scoring: one concrete missed pattern, one concrete over-flag, and the structural reason.

**Q11. A product manager asks you to extend the queue from 400 to 700 because "the marginal case is
still profitable". Are they right, and what is the correct response?**
Answer: arithmetically yes — by the fitted law the marginal slot at 700 is worth 0.4417 × 153 − 22.50 = 45.08 EUR net,
well above zero. Operationally no: the desk can staff 400, so a 700-case queue converts 300 cases a
day into backlog, and backlog destroys the 0.80 preventable fraction that makes the marginal case
profitable at all. The correct response is a hiring case at the marginal value, and a fixed queue tied
to capacity.
Reasoning: the value model assumes the case is acted on within 24 hours; queue length beyond capacity
violates that assumption, which is a model-invalidity rather than a policy preference.
Follow-up: what is the cost of a day of backlog in this setting? Answer: it depends on the shipment
horizon; if 20% of orders ship within 24 hours, a day of delay moves the preventable fraction from
0.80 toward 0.64 and cuts per-slot value at precision 0.497 from 53.55 to
0.497 × 115.2 − 13.50 − 4.53 = 39.22 EUR, about 27%.
Red flag: agreeing on arithmetic alone and extending the queue.
Scoring: distinguishes economic validity from operational feasibility and names the violated
assumption.

**Q12. Your calibration curve was fitted on days −180 to −1 and you report precision on days −90 to
−1. What is wrong, and how large is the error likely to be?**
Answer: the windows overlap, so the curve has seen the evaluation days' outcomes — leakage. The
direction is inflation of reported precision, because a curve fitted on the evaluated period can
reproduce its local prevalence. The magnitude is not predictable in general; the honest statement is
that the reported number is not an out-of-sample estimate at all.
Reasoning: calibration is a supervised fit; any outcome used in the fit cannot be used to judge the
fit.
Follow-up: what is the correct split? Answer: fit on −270 to −91, evaluate on −90 to −1, with the
120-day embargo applied to both, and record both windows in the run manifest.
Red flag: "isotonic is monotone so leakage is limited" — monotonicity constrains the shape, not the
information content.
Scoring: identifies leakage and the inflation direction; bonus for naming the manifest control.

**Q13. In one sentence each, what is the difference between precision at depth, marginal precision,
and calibrated probability, and which one belongs in the memo?**
Answer: calibrated probability is the per-case P(fraud) that a threshold can be compared against;
precision at depth is the average outcome over the first k cases, which is what a fixed queue
delivers; marginal precision is the outcome probability of the k-th case, which is what capacity
expansion buys. The memo uses calibrated probability for the action thresholds and marginal precision
for the hiring case, and never uses precision at depth to justify a capacity change.
Reasoning: the three quantities answer three different questions — action, delivered performance, and
incremental value.
Follow-up: at depth 400 here, give all three. Answer: calibrated probability of the boundary case
0.630 (equal to precision at depth for the last case by construction of the cumulative average),
precision at depth 0.630, marginal precision 0.497.
Red flag: using "precision" for all three interchangeably — the 0.63/0.50 distinction is the entire
capacity argument.
Scoring: three distinct definitions and the correct allocation to decisions.

**Q14. The fraud rate rises 40% in December and the queue is unchanged at 400. What happens to
precision at depth 400, and why is the answer not simply "it goes up"?**
Answer: the supply of fraud cases rises from 2,200 to about 3,080 per day, but the ranking is on
relative unusualness, so the composition of the top 400 changes toward whatever the prevailing attack
is. Precision can rise if the extra fraud is concentrated in the high-score region, or fall if the
extra volume is spread across the score distribution and the queue is crowded with the new dominant
pattern while the model's thresholds were calibrated at the old prevalence. The correct statement is
that precision at depth is not invariant to prevalence, and the calibration must be refitted to keep
the *probability* meaningful even when the rank ordering is stable.
Reasoning: prevalence changes the base rate; a rank ordering is prevalence-invariant only under a
strong assumption that the score's conditional distributions do not shift.
Follow-up: what would you monitor through December? Answer: realised versus predicted rate weekly,
plus the coverage table, plus the auto-block volume against its 25/day ceiling.
Red flag: "more fraud means better precision" — true only under an unstated assumption about where
the extra fraud sits in the score distribution.
Scoring: names prevalence sensitivity and refuses the monotone intuition.

**Q15. Your graph feature counts distinct cards sharing a device in 7 days. Describe the failure it
creates for a shared corporate device and the control that limits the damage.**
Answer: a corporate laptop or a family tablet legitimately links dozens of cards, so every card on it
looks like part of a fraud ring and scores high. The control is a minimum support count before the
feature is considered informative (here 5), plus an exclusion list of known shared-device fingerprints,
plus a separate high-volume branch so that a device with 40 cards is treated as a shared terminal
rather than as a ring.
Reasoning: the feature measures sharing, and sharing has both fraudulent and legitimate causes with
different distributions over the count.
Follow-up: how would you detect that the control failed? Answer: monitor block precision on the
shared-device segment separately; if it falls below 0.75 while overall block precision is fine, the
shared-device branch is misfiring.
Red flag: capping the count at some maximum and calling it handled — a cap without a separate branch
just compresses the two populations together.
Scoring: concrete legitimate cause plus a control that separates populations rather than truncating.

**Q16. You inherit this programme and find the queue is built with no artefact hash. What is the
first control you add, and what does it prevent?**
Answer: an approved-artefact list checked at publish time, so a queue whose scorer hash is not on the
list cannot be published. It prevents the silent-swap investigation in 12.8, where precision moves and
every component looks healthy because the only changed field is one nobody reads.
Reasoning: the defect is not a wrong model but an unrecorded change; the cheapest control is a gate on
the record, not surveillance of the model.
Follow-up: what else belongs in the run manifest? Answer: population hash, feature-definition version,
calibration-curve version, policy version, forest seed and tree count, and the tie-break rule — the
six fields that make a queue reproducible.
Red flag: proposing to log more metrics — metrics do not establish reproducibility; the manifest does.
Scoring: names a publish gate and the specific failure class it removes.

### 14.5 Where to read next

For the general form of the break-even rule and its variants, read the triage cheat sheet's section 4.
For the evaluation protocol that makes a queue comparison valid, read the human review loops sheet's
sections on reviewer capacity and blinding. For the failure catalogue, the common issues sheet lists
25 entries, of which the four this case instantiates are the coverage, quantisation, drift, and
manifest ones. The sibling case shows the same measurement discipline applied to a selection problem
with no maturity lag, which is the cleanest way to see which parts of this reasoning are about
triage and which are about labels.
## 15. Quick Reference Card

| # | Item | Value / rule | Source |
|---:|---|---|---|
| 1 | Decision | replace the 400-case queue sort with a calibrated anomaly score, fixed headcount | §1 |
| 2 | Break-even precision | p* = (c + f)/(φL + f) = 22.50/153 = 0.147 | §6.1 |
| 3 | Measured precision at depth 400 | 0.630 anomaly vs 0.540 rules; gap 0.090 | §6.2 |
| 4 | Value of the change | 400 × 0.090 × 153 = 5,508 EUR/day = 1,377,000 EUR/yr | §6.3 |
| 5 | Marginal slot at rank 400 | 0.497 × 153 − 22.50 = 53.55 EUR; average slot is 73.89 EUR | §6.2 |
| 6 | Evidence needed | 3.5 business days of matured data at depth 400; 114 days if read at depth 100 | §6.4 |
| 7 | Measurement cost | 60 audit slots/day × 90 days = 296,093 EUR of forgone value | §10.1, §11.2 |
| 8 | Capacity case | +104 slots at 49.93 EUR marginal = 1,298,000 EUR/yr vs 300,000 EUR cost | §11.2 |
| 9 | Kill condition | matured gap < 0.04 at depth 400 over a blind 90-day cohort | §3.3, §8.6 |
| 10 | Hard constraints | 18 investigators, 400 slots, 24-hour SLA, 120-day embargo, no authorisation-time declines | §4.2, §8.1 |

**Three numbers to carry into any similar triage decision.** The break-even probability, because it
converts a cost structure into a threshold; the marginal value at the operating point, because it is
the only defensible basis for a capacity change; and the depth at which the comparison is measured,
because a ranker comparison at the wrong depth is not a comparison.

**The one-sentence version.** Fix the read depth to your capacity, compute the break-even
probability from your own costs, measure on a sample the incumbent policy did not select, and price
the measurement before you start — because the ranking is usually the cheap part and the evidence is
the expensive part.

Every figure on this card is ILLUSTRATIVE and derived from the constructed inputs stated in sections
4 and 5a. None is a measurement of a real deployment.
