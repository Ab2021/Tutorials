---
title: "Fraud Evaluation, Calibration, and Capacity-Constrained Decisions"
topic: "Fraud evaluation, calibration, and capacity-constrained decisions"
collection: "fraud_modelling_playbook"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Fraud Evaluation, Calibration, and Capacity-Constrained Decisions

## 1. File Header

This sheet is the reference card for the hardest question in a fraud programme: what should the
model be allowed to *do*, once it has produced a number. It sits at the junction of three
disciplines that are usually taught separately — evaluation (does the score rank and probability
correctly), calibration (does the score mean what it claims), and decision theory (given a review
budget, which alerts are worth an investigator's next twelve minutes). The formulas below are
DERIVED from their stated assumptions. Every numeric input is ILLUSTRATIVE: constructed to be
internally consistent and re-computed by hand, not measured at any company. Where a formula holds
only under conditions, the condition is stated before the number, because a cutoff copied from
another programme without its cost and capacity inputs is the single most common cause of a
"working" fraud model that destroys value.

**Audience.** Fraud analytics leads, risk data scientists, and the operations managers who own the
review queue. The reader is assumed to know binary classification and conditional probability. No
assumption is made about knowledge of decision theory, proper scoring rules, or queueing.

**What this sheet is for.** Three concrete jobs: (a) convert a model's ranking into a currency
decision rule with an explicit cutoff and an explicit capacity constraint; (b) explain why a
high-AUC model can still lose money and how to detect that in one afternoon; (c) size and defend a
review capacity number, including the cost of being one investigator short.

**What it deliberately excludes.** Feature engineering for fraud, graph and device-linkage methods,
adversarial drift modelling, and regulatory dispute handling. Those live in the companion sheets on
imbalanced learning and on probability scoring. This sheet uses their outputs as inputs.

**Notation.** Currency amounts are written with an explicit unit, `cur`, meaning one unit of the
programme's reporting currency. Rates are dimensionless fractions in [0,1] unless stated as
per-1,000. Time is in days within a monthly decision cycle unless stated. Event time (when the
transaction happened) and processing time (when the label became known) are always distinguished:
mixing them is a silent leak and is treated as failure mode F-02 in section 8.

**The running example, fixed once and used everywhere.** A card issuer screens 4,000,000 eligible
transactions in a 30-day cycle. Confirmed fraud prevalence is 0.0012, so 4,800 of those
transactions are fraudulent and 3,995,200 are not. Mean net exposure on a confirmed fraud is 340
cur. Reviewing an alert prevents 55 per cent of that exposure on average, so preventable value per
caught fraud is V = 340 x 0.55 = 187.00 cur. A review consumes 0.200 investigator-hours (12
minutes) and costs 8.20 cur in loaded labour plus 1.30 cur in customer friction, so c = 9.50 cur
per review. The review team is 30 investigators at 160 productive hours each per cycle, giving
4,800 hours and therefore 24,000 reviews at most. Those eight numbers — N, pi, L, r, c, K, AHT,
hours — generate every arithmetic result in this sheet. Change one and the recalculations in
sections 5 and 9 show what moves.

**Evidence status of each block.** Sections 4, 5, and 6 are DERIVED: each step follows from the
stated assumptions and each number is re-computed independently in the text. Section 7 is a
selection guide built from those derivations, so its recommendations are only as good as the cost
inputs. Sections 8 and 9 mix DERIVED structure with HEURISTIC operating thresholds, marked as
such where they appear. No claim in this sheet is a measurement of a live programme's performance.

**Reading paths.** A reader who needs a cutoff today should read sections 3, 4.1, 4.2, and 12. A
reader defending a capacity request to finance should read sections 5, 7, and 9. A reader
diagnosing a model that "looks fine but loses money" should read sections 8 and 11 in that order.
A reader building the evaluation harness should read 5, 6, and 10.

## 2. Concept Map / Mental Model

The mental model is a funnel with three separately measurable stages and one shared currency
denominator. Stage one is the *score*: a monotone function of evidence, producing a real number
per event. Stage two is the *probability claim*: the assertion that this number equals the chance
the event is fraud, which is what makes expected-value arithmetic legitimate. Stage three is the
*action*: a threshold applied to either the score or the probability, subject to a capacity the
programme cannot exceed. Only stage three produces money. Stages one and two are instrumental.

```text
evidence ---- score s_i ---- (calibration map) ---- p_i ---- (decision rule, capacity K)
                     |                                    |
             ranking quality                    expected value per alert
             (AUC, recall@K)                    (precision x V - c)
                     |                                    |
                     +--------- both needed, neither sufficient ----------+
```

The diagram encodes the central claim: ranking quality and calibration are *orthogonal*
properties, and a decision needs the second plus an action model, not the first alone. A model can
have AUC 0.91 and be useless because its scores are compressed into 0.30–0.40, where no
probability-derived cutoff can be expressed. A model can be perfectly calibrated and useless
because its likelihood ratio at every achievable operating point is too weak to clear the cost
line. Neither defect is visible from the other's diagnostic.

**Why the funnel and not a single metric.** Any single scalar collapses the three stages, and each
collapse has a characteristic failure. Collapsing to AUC hides the operating region: AUC averages
over all thresholds, including the ones the programme will never use because they are above or
below capacity. Collapsing to precision hides the denominator: precision at capacity 24,000 and
precision at capacity 2,000 are not comparable numbers about the same model. Collapsing to
"fraud saved" hides the counterfactual: money recovered is not money saved unless the review
actually changed the outcome, which is the counterfactual-outcome problem in definition D-20.

**The three budgets.** A fraud programme runs on three budgets simultaneously and they trade
against each other. The *review budget* is investigator hours. The *friction budget* is the number
of genuine customers who must be contacted, delayed, or declined per cycle, which is a commercial
and reputational constraint rather than a cost line. The *loss budget* is the chargeback and
write-off exposure the programme is chartered to hold under a ceiling. A cutoff is a point in the
space where all three are satisfied; specifying only one of them (the usual case: hours) leaves the
other two to be discovered as incidents.

**The capacity feedback loop.** Capacity is not exogenous in the medium run. Alerts that
investigators cannot reach age in the queue; aged alerts have lower preventable value because the
exposure window has often closed; lower value per alert lowers the justified capacity; a lower
capacity raises the cutoff; a higher cutoff changes which labelled examples the next training set
contains, because only reviewed events get reliable labels. That loop is the mechanism behind
failure mode F-06 (label-selection drift) and it means a capacity decision is also a training-data
decision.

**A second loop, on the adversary side.** A published cutoff is an instruction. Fraudsters learn
which amounts, merchants, and hours are reviewed and migrate to the unreviewed region, which
raises prevalence exactly where the model is weakest and leaves measured precision at the reviewed
cutoff looking stable. This is why evaluation windows must be short and why section 5.5 insists on
measuring performance *below* the cutoff via audit rather than only at the reviewed front.

**Mental model in one sentence.** A score is an ordering, a calibrated score is a probability, and
only a probability multiplied by a value and compared against a cost, under a hard capacity, tells
you what to do.

## 3. Foundational Definitions

Each definition states the meaning, the units and valid range, and the specific misuse that
follows from confusing it with a neighbouring concept. Definitions are numbered D-01 onward and
are referenced from the formulas in section 4.

### D-01 Decision cohort

The set of events eligible for screening in one decision cycle, after exclusions that are known
before scoring: already-blocked accounts, test traffic, internal transactions, and events outside
the programme's charter. Units: events per cycle. Valid range: any finite positive count. Misuse:
computing prevalence on the *scored* population after removing events the model already rejected,
which truncates the denominator and inflates measured prevalence. In the running example the
cohort is 4,000,000 events per 30-day cycle and is fixed before any model runs.

### D-02 Outcome and label

Y_i in {0,1} is the eventual ground truth for event i: 1 if the event is confirmed fraudulent
under the programme's written definition, 0 if it is confirmed legitimate. Units: indicator.
Valid range: binary, but with a third state in practice — unknown, for events whose dispute window
has not closed, which must be excluded from training and from any metric rather than coded 0.
Misuse: coding pending disputes as negatives, which teaches the model that recent fraud is
legitimate and creates a downward bias in recent-period precision.

### D-03 Maturity horizon

The lag, in days, after which a label is treated as final. Two dates matter: the event date and the
date the label becomes final. In the running example the maturity horizon is 90 days. Valid range:
positive; must be at least as long as the longest chargeback cycle in the portfolio. Misuse:
evaluating on events younger than the horizon and reporting the resulting undercount as a decline.

### D-04 Selection indicator

a_i in {0,1} records whether the programme took an action (review, block, hold) on event i. This
is a *decision*, not an outcome, and it is the variable a policy is allowed to set. Units:
indicator per event. Misuse: treating a_i as a feature. Anything downstream of the decision — the
reviewer's note, the customer's call, the manual override — is contaminated by the decision and
must never enter the score.

### D-05 Risk score

s_i is a real-valued output of the scoring function, ordered so that larger means more suspicious.
Units: arbitrary but fixed; only the ordering is preserved under any strictly increasing
transform. Valid range: the model's output range, often unbounded logits. Misuse: comparing scores
across models or across retrains as if the scale were common; two retrains can have identical AUC
and score ranges differing by 10x.

### D-06 Calibration

A score is calibrated on population P if, among events with score s, the fraction that are
fraudulent equals the probability the score claims. Formally, for a probability score p,
E[Y | p] = p on P. Units: dimensionless equality. Valid range: a property of a (score, population,
period) triple, not of a model. Misuse: assuming calibration transfers. A score calibrated on a
0.5-prevalence modelling sample is not calibrated on a 0.0012-prevalence live stream; section 4.3
gives the correction and its precondition.

### D-07 True positive

TP = count of events with Y=1 and a=1. Units: events per cycle. Valid range: non-negative integer.
Misuse: counting *recovered currency* as a TP count, which mixes a count with an amount and makes
precision uninterpretable. In the running example TP at capacity is 2,510 events.

### D-08 False positive

FP = count of events with Y=0 and a=1. Units: events per cycle. Valid range: non-negative. Misuse:
netting off "customer-verified" false positives before computing precision, which silently changes
the definition of a positive and makes the number incomparable with the cutoff that generated it.

### D-09 False negative

FN = count of events with Y=1 and a=0. Units: events per cycle. Valid range: non-negative. Misuse:
estimating FN from reviewed data alone. Unreviewed fraudulent events are by construction
unlabelled, so FN is not observed and must be estimated by audit, by issuer-reported chargebacks,
or by an explicit assumption — never by subtraction from a known total unless that total is
independently measured.

### D-10 True negative

TN = count of events with Y=0 and a=0. Units: events per cycle. Valid range: non-negative. Misuse:
treating TN as a performance figure. TN is dominated by cohort size; a model that reviews nothing
has 3,995,200 true negatives and no value.

### D-11 Prevalence

pi = P(Y=1) on the decision cohort before any action. Units: dimensionless fraction; often quoted
per 1,000 for readability (0.0012 = 1.2 per 1,000). Valid range: (0,1); the endpoints make
predictive-value ratios degenerate. Misuse: using the modelling sample's prevalence. In the running
example pi = 0.0012 on 4,000,000 events, which is 4,800 frauds.

### D-12 Recall

Recall = TP / (TP + FN) = P(a=1 | Y=1). Units: dimensionless. Valid range: [0,1]. Misuse:
reporting recall from a sample in which only alerted events received final labels, which forces
recall upward because unalerted frauds are unobserved. Section 5.5 shows the audit estimator that
fixes this.

### D-13 Precision

Precision = TP / (TP + FP) = P(Y=1 | a=1). Units: dimensionless. Valid range: [0,1], undefined when
no event is selected. Misuse: quoting precision without the operating capacity, since precision is
a decreasing function of the number of alerts taken.

### D-14 False-positive rate

FPR = FP / (FP + TN) = P(a=1 | Y=0). Units: dimensionless. Valid range: [0,1]. Misuse: substituting
1 - precision for FPR. They share a denominator only in the special case where the selected count
equals the negative count, which essentially never holds.

### D-15 Capacity

K = the maximum number of actions the programme can execute in one cycle. Units: actions per
cycle. Valid range: non-negative integer; the interesting case is 0 < K < N. Misuse: treating K as
a target to fill. K is an upper bound; a policy that reviews below K because marginal alerts have
negative expected value is correct behaviour, not underperformance.

### D-16 Action benefit

Delta_i = p_i x L_i x r_i - c_i - h_i, the expected net currency from taking the action on event i
instead of not taking it. Units: cur per event. Valid range: unbounded real; negative allowed.
Misuse: defining the benefit using the *observed* outcome of the action (money recovered on caught
fraud) rather than the counterfactual (money that would have been lost without review), which
double counts losses that were unavoidable.

### D-17 Calibration sample

The labelled set on which a reliability curve is estimated, with its own period, population, and
prevalence. Units: events. Valid range: large enough that each populated score bin contains at
least roughly 20 positives; with pi = 0.0012 that means about 17,000 events per bin. Misuse:
building a ten-bin reliability curve on a 5,000-event sample at 0.12 per cent prevalence, where
most bins contain zero or one positive and the curve is pure noise.

### D-18 Temporal holdout

An evaluation split whose events are strictly later in event time than all training events, with a
gap of at least the maturity horizon between the last training label and the first evaluation
event. Units: events. Valid range: any size, but the split must not require labels that mature
after the evaluation date. Misuse: a random split on transaction data, which leaks both seasonality
and the fraud ring structure that makes neighbouring transactions correlated.

### D-19 Counterfactual outcome

The outcome that would have occurred under the action not taken. Units: currency or indicator.
Valid range: never observed for the same event; only estimated. Misuse: assuming a blocked
transaction would certainly have been fraudulent, or that a reviewed-and-cleared transaction would
certainly have been fine, which turns a policy effect into an identity.

### D-20 Tie policy

The deterministic rule that orders events with identical scores when the cutoff falls inside a tie
group. Units: rule, not number. Valid range: must be total, deterministic, and independent of the
outcome. Misuse: breaking ties by an outcome-correlated field such as merchant category fraud rate
computed on the same period, which leaks the label into allocation and inflates measured precision.

### D-21 Uncertainty interval

A set of values for a quantity consistent with the sampling design at a stated coverage. Units:
same as the quantity. Valid range: coverage in (0,1), typically 0.95. Misuse: quoting a binomial
interval when the sampling unit is a fraud ring rather than a transaction; section 5.4 quantifies
the design-effect inflation.

### D-22 Support and positivity

Support is the region of feature or score space where the training data has mass; positivity is
the requirement that every event in the decision cohort has a strictly positive chance of being
observed. Units: properties of the design. Valid range: must hold for any estimator that reweights
by observation probability. Misuse: running an inverse-probability audit estimate with a stratum
whose inclusion probability is zero, which leaves that region unidentified while the point estimate
looks finite.

### D-23 Preventable share

r_i = the fraction of conditional loss that the action prevents, in [0,1]. Units: dimensionless.
Valid range: [0,1]; typically 0.3–0.7 for card-not-present review. Misuse: setting r = 1 because
"we blocked it", ignoring that some blocked transactions migrate to another channel and complete
the fraud elsewhere.

### D-24 Review cost

c_i = direct labour plus friction cost of one action, in cur per event. Valid range: positive.
Misuse: excluding friction. At 0.200 hours and 41 cur/hour the labour component is 8.20 cur, while
a customer-verification contact that costs 1.30 cur in call handling and goodwill raises the true
cost to 9.50 cur, a 16 per cent increase that moves the cutoff by the same 16 per cent.

### D-25 Score-to-decision gap

The difference between the model's best achievable operating point and the operating point the
programme can staff. Units: events per cycle or cur per cycle. Valid range: non-negative by
definition of best. Misuse: reporting the gap as a model defect. In the running example the
unconstrained economic optimum is 28,000 reviews and capacity is 24,000, so the gap is a capacity
problem worth 4,262 cur per cycle, not a modelling problem.
