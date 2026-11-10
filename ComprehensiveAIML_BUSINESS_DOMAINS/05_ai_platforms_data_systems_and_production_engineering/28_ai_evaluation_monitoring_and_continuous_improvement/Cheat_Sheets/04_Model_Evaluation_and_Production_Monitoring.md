---
title: "Model Evaluation and Production Monitoring"
topic: "Paired offline and online evaluation of predictive and agentic services"
collection: "ai-evaluation-monitoring"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Model Evaluation and Production Monitoring

> **Evidence declaration.** Every formula below is derived from first principles and was recomputed
> at the precision printed. Every dataset in the worked example is a constructed teaching scenario:
> twenty thousand decisions, one thousand adjudicated positives, and a review team of stated capacity.
> No organisation, deployment, or measured production result is described. Figures labelled
> ILLUSTRATIVE exist so that a reader can check a diagnostic against an expected magnitude; none may be
> cited as an observed benchmark, a vendor claim, or an industry norm.

## 1. File Header

### 1.1 Purpose

This sheet teaches how to decide whether a scoring or agentic service is still doing what it was
built to do, using evidence that is available at the moment the decision has to be made. The subject
is not model fitting. It is the measurement contract that sits between a fitted artefact and an
operational decision: which population was scored, which outcome definition applies, how long the
outcome took to become known, what the score was compared against, and what uncertainty remains when
the comparison is finished.

The sheet is written for the reviewer who has to answer one of four questions on a given morning:
should this version be promoted, should this version be rolled back, is this alert a real change or
routine variation, and is the number on the dashboard answering the question the business asked.

### 1.2 Audience and assumed background

The reader is assumed to know binary classification, conditional probability, the difference between
a rate and a count, and what an on-call rota does at three in the morning. Familiarity with logistic
regression or gradient-boosted trees helps but is not required: nothing below depends on how a score
is produced, only on what the score means once it exists.

A reader who has never operated a service will still be able to follow the arithmetic, but the
selection judgements in section 7 will read as arbitrary. Those judgements derive from constraints —
review capacity, label latency, harm asymmetry — that only appear once a service has users.

### 1.3 Learning outcomes

After working this sheet a reader should be able to do six things without reference to anything else.

1. State an evaluation protocol as a tuple of population, cut-off, outcome definition, maturity rule,
   comparator, and decision rule, and detect a protocol that omits one of the six.
2. Derive precision, recall, and their interval estimates from a confusion matrix, and explain why a
   point estimate without a denominator definition is not a measurement.
3. Derive the cost-optimal decision threshold from the two error costs and the prevalence, and explain
   why the threshold that maximises F1 is usually not that threshold.
4. Distinguish a monitoring signal, an evaluation estimate, and a causal claim, and refuse to let an
   observation from one category be promoted into another without new evidence.
5. Compute the review workload implied by an alert rate and show when that workload exceeds capacity
   in a way that silently converts missed reviews into apparent precision.
6. Design a verification step that replays both a triggering record and a boundary record.

### 1.4 Scope and exclusions

In scope: binary and multiclass classifier monitoring, ranking quality monitoring where an action
follows a score, agent and LLM run success measurement where a rubric substitutes for a label,
calibration, drift statistics used as investigation cues, threshold policy, label maturity, review
capacity, regression gating, and the arithmetic of the associated uncertainty.

Out of scope: the training algorithm, hyperparameter search, feature engineering method, formal
fairness auditing under a legal standard, causal identification of a product effect, vendor API
particulars, and cost accounting beyond the stylised currency units used in the examples. Fairness is
touched only where it changes a measurement — for example when a slice is too small for its interval
to support a claim. A reader who needs a policy decision on disparate impact must take it elsewhere;
this sheet can only say what the data can and cannot support.

### 1.5 Notation conventions

One display convention is used throughout: formulas appear in fenced `text` blocks, and each symbol is
defined at first use with meaning, units, and valid range. Proportions are dimensionless fractions in
`[0, 1]` and are printed to five decimals when they come from the worked example. Counts are integers.
Currency is a stylised unit written `cu` and never converted to a real currency. Durations are in days
unless a suffix says otherwise. Percentages appear only in prose commentary, never inside a formula,
because mixing the two is one of the most common sources of a factor-of-one-hundred error in a review
document.

Event time and processing time are kept distinct in every statement. Event time is when the thing
happened to the user. Processing time is when the record of it arrived in the warehouse. A window over
event time and a window over processing time select different rows, and the difference is not a
rounding detail: it is the difference between "we scored these cases" and "we have learned about these
cases", and conflating them is how an evaluation silently becomes a measurement of pipeline lag.

### 1.6 Evidence status vocabulary

Five statuses appear in this sheet and they are not interchangeable.

- **measured** — observed on a stated population under a stated protocol, with the observation stored.
- **derived** — computed by arithmetic from stated inputs; reproducible by any reader with the inputs.
- **illustrative** — a constructed input chosen to make an arithmetic check possible.
- **heuristic** — a rule of thumb with no derivation; useful for triage, inadmissible as a threshold.
- **not-measured** — the quantity is named but no observation exists.

Every number in this sheet is `derived` from `illustrative` inputs. That combination is legitimate and
common in design work: it answers "what magnitude should I expect" without pretending to answer "what
did this service achieve". The pair must travel together. A derived number quoted without its
illustrative input label becomes a fabricated measurement in the hands of the next reader.

### 1.7 Source boundary

Topic provenance for this sheet is the unit topic map entry "production monitoring" and the sibling
submodules 9.1 through 9.6 in the same unit, which cover training metrics, perplexity, benchmark
design, judge and human evaluation, training-failure debugging, and infrastructure profiling. Those
files exist in this tree and their headings were read to avoid duplication; their specific numeric
claims are not reproduced here.

No external document was available for inspection while writing. Statements about statistical
procedures — Wilson intervals, the Dvoretzky–Kiefer–Wolfowitz inequality, CUSUM average run length,
Little's law — are stated in their standard form and their assumptions are given explicitly so that a
reader can check them against any textbook. Where a threshold such as "PSI above 0.2 means trouble"
appears in industry writing, this sheet deliberately does not adopt it, because the statistic depends
on binning and sample size and a universal cutoff is not derivable.

### 1.8 How to use this sheet

Read section 3 once for vocabulary, then section 4 for the derivations that matter, then work section
5 with a calculator. Section 5 is the only place where the numbers interlock, and a reader who has
reproduced the threshold ladder and the coverage correction will find the rest of the sheet a
restatement of those two calculations in different clothes.

For live debugging, jump to section 8. Each entry there is written as a discriminator: a check whose
result differs under the competing explanations, runnable before any fix is attempted. For an
interview or an exam, section 11 contains twelve questions of which seven require computation.

**Takeaway:** an evaluation is a protocol with six parts; a monitoring signal is not an evaluation;
and a derived number must travel with the label that says its inputs were constructed.

## 2. Concept Map / Mental Model

### 2.1 Two loops, one decision point

Production measurement has two loops that run at different speeds and answer different questions.

The fast loop runs on data that exists at request time: request volume, error codes, latency
percentiles, score distribution, feature freshness, action rates. It can be computed within seconds
and it never needs the future. It can tell you that something changed. It cannot tell you whether the
change harmed anyone, because harm is defined by an outcome that has not happened yet.

The slow loop runs on data that requires the future to arrive: adjudicated outcomes, human rubric
scores, downstream conversions, chargebacks, confirmed incidents. It can tell you whether the service
was right. It arrives late, incompletely, and selectively.

```text
request --> features --> score --> action --> user experience --> outcome
   |            |          |         |              |              |
   |            |          |         |              |              +--> slow loop
   |            |          |         |              +-----------------> (days to weeks)
   |            |          |         +--------------------------------> action log
   |            |          +------------------------------------------> score log
   |            +-----------------------------------------------------> feature snapshot
   +------------------------------------------------------------------> request log

Fast loop reads the four left-most columns. Slow loop needs the right-most column.
Neither loop alone can answer "is this version safe to keep".
```

The decision point sits where the two loops meet. A promotion decision needs the slow loop, because
promotion is a claim about correctness. A rollback decision can be triggered by the fast loop alone,
because rollback is a claim about risk, and risk can be inferred from a mechanism that is visibly
broken even when the harm has not yet been counted.

### 2.2 Five layers that fail independently

Measurement failures cluster into five layers. The layers are useful because a defect in one does not
disturb the others, so cheap checks can eliminate whole layers before any statistics are needed.

```text
[1] RECORD     was the prediction logged, with version and timestamp?
[2] JOIN       was the outcome attached as of the decision, not as of today?
[3] SCORE      is the arithmetic right, including denominators and boundaries?
[4] INTERPRET  does the number answer the question that was asked?
[5] ACT        is the person or policy allowed to act on it, and is the action recorded?
```

A layer-1 defect shows up as rows that are missing or impossible. A layer-2 defect shows up as rows
that are individually correct and jointly wrong — each value is right, the pair is from different
moments. A layer-3 defect is right at the centre of the data and wrong at an edge. A layer-4 defect is
arithmetically flawless and answers a neighbouring question. A layer-5 defect produces a defensible
recommendation with no accountable effect.

The ordering matters for triage cost. Counting rows is a single query. Reconstructing one record by
hand is ten minutes of attention. Recomputing a stratified estimate is a query per stratum. Only the
last two layers require judgement, so a triage that starts with judgement is a triage that will spend
its budget on the wrong layer.

### 2.3 The maturity axis

Every evaluation of a deployed model is an evaluation of a cohort at a particular age. Two runs of the
same query a week apart can produce different precision with no change to the model, because the set
of decisions whose outcomes have arrived has grown.

```text
day 0        day 15              day 45                     day 75
 |            |                   |                          |
 decisions    half the labels    reporting cutoff           mature
 scored       have arrived        for this quarter           for most horizons

An estimate computed at day 15 and an estimate computed at day 75 answer:
  day 15 -> "what is quality among cases that resolved quickly?"
  day 75 -> "what is quality among cases that have resolved at all?"
Neither answers "what is quality on the full cohort" unless coverage is handled.
```

The practical consequence is that coverage is not a data-quality footnote. It is a parameter of the
estimator. Section 4.3 derives the inverse-probability correction; section 5.6 shows what it does to a
precision figure that looked respectable.

### 2.4 Why score drift is not quality drift

A score distribution can move for at least five distinct reasons, and only one of them is a model
problem.

1. The input population changed and the model is behaving correctly on the new population.
2. A feature's units or encoding changed, so the model is behaving correctly on corrupted inputs.
3. The prevalence of the outcome changed, so the same discrimination produces a different score mix.
4. The threshold or the downstream consumer changed, so the same scores are used differently.
5. The relationship between inputs and outcome changed, so the model is now wrong.

Reasons one through four are operational or population facts. Reason five is concept drift in the
strict sense. A monitoring system that reports "score drift detected" for all five and stops there has
produced a symptom, not a diagnosis, and the operator's next action — retrain — is correct for exactly
one of the five. Retraining on corrupted inputs, which is reason two, makes the situation worse in a
way that is invisible until the corruption is repaired.

### 2.5 The claim ladder

Four kinds of statement get confused in review documents, and they sit on a ladder where each rung
requires strictly more evidence than the one below.

| Rung | Claim | Minimum evidence | What it cannot support |
|---|---|---|---|
| 1 | The service is up and responding | request-level success indicator | anything about correctness |
| 2 | The service produced the outputs we intended | versioned log plus contract checks | anything about outcomes |
| 3 | The service's outputs are associated with the intended outcome | mature cohort plus defined metric | that the service caused the outcome |
| 4 | Changing the service changed the outcome | randomised or credible quasi-experiment | generalisation beyond the studied population |

Most production disagreements are two people arguing at different rungs. The operator who says "error
rate is fine" is at rung 1. The analyst who says "precision fell" is at rung 3. Both are right, and
the sentence "the service is fine" is the sentence that is wrong, because it does not say which rung
it is on.

**Takeaway:** separate fast from slow signals, triage the cheap layers first, treat cohort maturity as
part of the estimator, and never let a rung-2 observation be quoted as a rung-3 claim.

## 3. Foundational Definitions

### 3.1 The core vocabulary

Each definition names the quantity, its unit or type, and the confusion it most often attracts. The
confusions are the load-bearing part: a definition nobody misuses does not need to be written down.

| Term | Definition and scope | Type, unit | Characteristic confusion |
|---|---|---|---|
| Decision | One scored case at one recorded moment, identified by a stable key | event | treated as a user or an account |
| Eligible decision | A decision inside the declared population, window, and exclusion set | event subset | silently redefined between comparisons |
| Score | The model's numeric output before any policy is applied | real in `[0,1]` or unbounded | treated as a probability when only a rank |
| Threshold | The cut that converts a score into an action | policy parameter | treated as a property of the model |
| Action | What the system did after the score: alert, block, review, none | event | treated as equivalent to the label |
| Outcome | The adjudicated truth for a decision over a stated horizon | binary or graded | confused with the action or the score |
| Horizon | Length of the future window over which the outcome is defined | duration | shortened mid-comparison without notice |
| Maturity | Whether enough time has passed for the outcome to be substantially knowable | boolean per cohort | treated as "all labels arrived" |
| Cut-off | The event-time boundary that freezes the cohort for evaluation | timestamp | confused with query run time |
| Coverage | Eligible decisions with a usable outcome divided by eligible decisions | fraction | reported as accuracy |
| Eligibility rule | The predicate that decides whether a decision belongs to the population | predicate | left implicit, then varied |
| Prevalence | Outcome-positive share of the labelled population | fraction | assumed stable across windows |
| Calibration | Agreement between predicted probability and observed frequency in groups | conditional property | conflated with discrimination |
| Discrimination | Ability to rank positives above negatives | ordinal property | conflated with calibration |
| Concept drift | Change in the outcome distribution conditional on the inputs | distributional change | inferred from input change alone |
| Covariate shift | Change in the input distribution with the conditional outcome unchanged | distributional change | assumed to imply harm |
| Label shift | Change in outcome prevalence with the conditional input distribution unchanged | distributional change | mistaken for concept drift |
| Selection bias | The labelled subset differs systematically from the eligible population | sampling property | assumed away by larger samples |
| Error budget | Permitted bad events over a window at a stated target | count | read as a percentage rather than a count |
| Burn rate | Bad events consumed divided by budget for the same window | ratio | computed against the wrong target |
| Regression gate | A pre-declared rule that blocks promotion when a suite fails | policy | rewritten after seeing the candidate |
| Judge | A model or human applying a rubric to an output | instrument | treated as ground truth rather than measurement |
| Trace | Linked record of the steps in one run | event graph | treated as proof of correctness |

### 3.2 Population comes before metric

Almost every evaluation disagreement that is not about arithmetic is about population. The same model
can show precision 0.48 and precision 0.37 on the same day without either number being wrong, if one
is computed over alerted cases that were reviewed and the other over all eligible decisions weighted
back to the full population. Section 5.6 works this exact contrast.

The discipline that prevents the argument is to write the population as a predicate before computing
anything. "Decisions scored by model version 7 in the card-not-present channel, with a complete feature
vector, excluding internal test accounts, in the calendar month of March, with outcomes adjudicated by
45 days after the decision." That sentence is the population. Anything shorter is a sketch, and a
sketch will be filled in differently by the next person.

### 3.3 Action is not outcome

The most expensive confusion in applied monitoring is treating the system's action as the label. It is
attractive because the action is recorded at decision time and requires no waiting.

An alert is not a fraud. A block is not a bad user. A flagged ticket is not a defect. Each is a decision
the system made, informed by a model that may or may not be right. When an organisation measures "false
positive rate" as the share of alerts that an analyst later closed as uninteresting, it has measured
the analyst's triage policy against the alerting policy, not the model's correctness against the world.
Both are worth measuring. The second one requires an adjudication that is independent of the decision
being evaluated.

The operational test is simple: if the label cannot disagree with the action for reasons unrelated to
model quality, it is not a label. An analyst who is told to close anything without a documented reason
will close things, and that closure rate will look like model performance.

### 3.4 Latency is a property of the label, not of the model

Labels arrive with a distribution of delays. Some outcomes resolve in minutes, some in weeks, and the
fast ones are rarely a random sample of all outcomes. Two structurally different patterns dominate:

- **Fast positives.** Fraud confirms quickly when the cardholder complains immediately, and slowly
  when nobody notices. Early precision overweights the fast positives and reads too high.
- **Fast negatives.** Support tickets resolved within the hour are mostly easy negatives; hard ones
  linger. Early precision overweights easy negatives and also reads too high.

Both patterns push the early estimate up, which is why "precision improved right after launch" is
weak evidence. A rising precision series in the first days of a deployment is the default behaviour of
a partially-labelled cohort, not an achievement.

### 3.5 Capacity is part of the metric

A metric that implies more human work than exists is not a metric, it is a queue. When an alert rate
implies 95 reviews per day against a team that can handle 90, the backlog grows monotonically, and the
backlog itself suppresses the measured false-positive rate, because unreviewed alerts do not get a
label.

This is the mechanism behind the most misleading pattern in monitoring: a service that looks
increasingly precise precisely because it is drowning. Section 9.3 gives the queueing arithmetic.

**Takeaway:** define the population as a predicate, never let an action stand in for an outcome, treat
label latency as a sampling mechanism, and check capacity before believing a precision figure.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

The base case is a service indicator: what fraction of eligible requests satisfied a good-event
definition. Its derivation is the definition of a proportion, and its only interesting content is the
error budget, which converts a target into a count.

```text
FORMULA (indicator):  SLI = G / N
WHERE: G = good eligible events [count]; N = all eligible events [count]; both integers, N > 0.
CONSTRAINT: 0 <= G <= N. The good-event predicate must be fixed before the window opens, otherwise
            the indicator can be redefined after seeing the data.
NUMERICAL EXAMPLE: G = 4,313,000 and N = 4,320,000 give SLI = 4,313,000 / 4,320,000 = 0.9983796.
INTERPRETATION: 7,000 requests failed the stated predicate; the rest satisfied it.
```

```text
FORMULA (error budget):  B = N * (1 - S)
WHERE: B = permitted bad events in the window [count]; S = target [fraction of eligible events];
       N = eligible events in the window [count].
CONSTRAINT: N and the window must be identical to those used for the SLI, and the target must be
            expressed over the same window length. A daily target applied to a monthly count is a
            different commitment.
NUMERICAL EXAMPLE: at S = 0.995 and N = 4,320,000, B = 4,320,000 * 0.005 = 21,600.
                  at S = 0.999 and the same N, B = 4,320,000 * 0.001 = 4,320.
INTERPRETATION: the same 7,000 failures consume 32.4% of the first budget and 162% of the second.
```

The burn rate is the ratio that makes the budget useful, because it is dimensionless and therefore
comparable across services of different volume.

```text
FORMULA (burn):  R = F / B
WHERE: F = failures observed in the window [count]; B = budget for that window at the agreed target.
CONSTRAINT: F and B must cover the same eligible-event set and the same window length.
NUMERICAL EXAMPLE: F = 7,000, B = 21,600 gives R = 0.324; with B = 4,320, R = 1.621.
INTERPRETATION: identical traffic is comfortably inside one commitment and 62% outside another; the
                only difference is the target that was agreed.
```

Validity condition for section 4.1: the good-event predicate, the eligible-event predicate, the
measurement point, and the window are all fixed in advance and identical between numerator and
denominator. Break any one of the four and the ratio is still computable, still printable, and no
longer a service indicator.

### 4.2 General case

The general case is a decision table with delayed, incomplete outcomes. The derivation has three
steps: partition the mature cohort by prediction and outcome, define the ratios on the partition, and
quantify each ratio with an interval whose assumptions are stated.

```text
FORMULA (partition):  N_mature = TP + FP + FN + TN
WHERE: TP, FP, FN, TN are integer counts of mature eligible decisions;
       TP = predicted positive and outcome positive; FP = predicted positive, outcome negative;
       FN = predicted negative, outcome positive; TN = predicted negative, outcome negative.
CONSTRAINT: every decision appears in exactly one cell; decisions without a usable outcome are not
            in any cell and must be counted separately as pending.
NUMERICAL EXAMPLE: 750 + 1,283 + 250 + 16,197 = 18,480 mature decisions.
INTERPRETATION: 1,520 of 20,000 eligible decisions are outside the table entirely.
```

```text
FORMULA (ratios):  precision = TP/(TP+FP);  recall = TP/(TP+FN);  FPR = FP/(FP+TN)
WHERE: each denominator is a row or column total of the partition, all counts, all non-negative.
CONSTRAINT: a denominator of zero makes the ratio undefined, not zero and not one.
NUMERICAL EXAMPLE: precision = 750/2,033 = 0.36891; recall = 750/1,000 = 0.75000;
                   FPR = 1,283/17,480 = 0.07340; specificity = 16,197/17,480 = 0.92660.
INTERPRETATION: 63.1% of alerts are wrong by the adjudicated label, and three quarters of positives
                are captured at this threshold.
```

The interval is not optional. A proportion from a finite sample has a sampling distribution, and the
simple normal approximation fails exactly where monitoring lives — near zero and near one, and at
small counts.

```text
FORMULA (Wilson interval):  centre = (p + z^2/(2n)) / (1 + z^2/n)
                            half   = z * sqrt( p(1-p)/n + z^2/(4n^2) ) / (1 + z^2/n)
WHERE: p = k/n is the observed proportion [dimensionless]; n = denominator [count];
       z = standard normal quantile, 1.959964 for a two-sided 95% interval.
CONSTRAINT: observations approximately independent and identically distributed within the cohort.
NUMERICAL EXAMPLE: k = 750, n = 2,033 gives p = 0.36891, centre 0.36916, half 0.02096,
                   interval [0.34820, 0.39012].
INTERPRETATION: the plausible range for precision spans 4.2 percentage points, which is wider than
                most threshold changes worth arguing about.
```

Derivation of the Wilson form, since it is the interval most often used without justification: the
score statistic `(p_hat - p)/sqrt(p(1-p)/n)` is approximately standard normal; solving
`|p_hat - p| <= z*sqrt(p(1-p)/n)` for `p` yields a quadratic whose two roots are the printed bounds.
The inversion is what makes the interval behave at boundaries, where the uncorrected Wald form
collapses to zero width. Section 4.4 shows that collapse.

### 4.3 Variants

Three variants do most of the practical work: the cost-optimal threshold, the coverage correction, and
the probability-score decomposition.

#### 4.3.1 Cost-weighted threshold

An operating threshold is an economic decision, not a statistical one. Deriving it takes one line of
expected-cost algebra once the costs are named.

```text
FORMULA (expected cost per decision):
    C(t) = C_FP * P(score >= t | negative) * (1 - pi) + C_FN * P(score < t | positive) * pi
WHERE: t = threshold [score units]; C_FP = cost of a false positive [cu];
       C_FN = cost of a false negative [cu]; pi = prevalence [fraction].
CONSTRAINT: costs are per decision and constant across the population; if either cost depends on the
             score or the segment, the derivation must be reworked per segment.
NUMERICAL EXAMPLE: pi = 0.05, C_FP = 8, C_FN = 42, at t = 0.25 with FPR 0.07340 and recall 0.75:
    C = 8 * 0.07340 * 0.95 + 42 * 0.25 * 0.05 = 0.55784 + 0.52500 = 1.08284.
INTERPRETATION: about 1.08 cu of avoidable loss per decision at this operating point.
```

```text
FORMULA (optimal threshold for a calibrated score):  p* = C_FP / (C_FP + C_FN)
WHERE: p* is the probability at which the expected cost of alerting equals the expected cost of not
       alerting; the score must be calibrated for p* to be a score threshold.
CONSTRAINT: calibration on the deployment population; an uncalibrated rank cannot use this rule.
NUMERICAL EXAMPLE: C_FP = 8 and C_FN = 42 give p* = 8/50 = 0.16.
INTERPRETATION: alert when the predicted probability exceeds 0.16, not when it exceeds 0.5.
```

The derivation is a comparison of two conditional expectations at a single case. Alerting costs
`C_FP * (1 - p)` in expectation because a false positive is the only way an alert is wrong; not
alerting costs `C_FN * p`. Setting them equal gives `C_FP(1-p) = C_FN p`, hence
`p = C_FP/(C_FP + C_FN)`. The result is independent of prevalence, which surprises people: prevalence
determines how many cases sit above the threshold, not where the threshold belongs. That independence
is the reason the cost rule is more stable than a rule based on observed precision.

#### 4.3.2 Coverage correction

When labels arrive selectively, a rate over labelled cases estimates a rate over a different population
than the one being managed. Inverse-probability weighting repairs this if the labelling probability is
known or estimable per stratum.

```text
FORMULA (weighted precision):  P_w = sum_s ( n_s * c_s * yhat_s ) / sum_s ( n_s * c_s )
WHERE: s indexes strata; n_s = eligible decisions in stratum s [count];
       c_s = labelling probability in stratum s [fraction, 0 < c_s <= 1];
       yhat_s = share of labelled stratum-s alerts that were correct [fraction].
CONSTRAINT: c_s estimated from an audited sample, not from the labelling system's own claim; the
             strata must capture the mechanism that drives selection.
NUMERICAL EXAMPLE: see 5.6, where the labelled-only rate is 0.48067 and the weighted rate is 0.36891.
INTERPRETATION: the labelled-only figure is optimistic by 11.2 percentage points because correct
                alerts are more likely to be labelled than incorrect ones.
```

#### 4.3.3 Probability-score decomposition

For a service that emits probabilities rather than ranks, the Brier score decomposes into three terms
that diagnose different diseases.

```text
FORMULA:  BS = (1/n) * sum_i (p_i - y_i)^2 = REL - RES + UNC
WHERE: REL = sum_b (n_b/n) (o_b - pbar_b)^2  [reliability, lower is better]
       RES = sum_b (n_b/n) (o_b - obar)^2    [resolution, higher is better]
       UNC = obar (1 - obar)                 [irreducible, property of the population]
       pbar_b = mean predicted probability in bin b; o_b = observed frequency in bin b;
       obar = overall observed frequency [all dimensionless proportions].
CONSTRAINT: the identity holds for any binning; the individual terms do not.
NUMERICAL EXAMPLE: with six bins, REL = 0.00256, RES = 0.07705, UNC = 0.20855, so
                   BS = 0.00256 - 0.07705 + 0.20855 = 0.13406.
INTERPRETATION: the model is nearly perfectly reliable and moderately informative; almost all of the
                score is the population's own irreducible uncertainty.
```

The reason to prefer the decomposition over a single number: reliability can be fixed by recalibration
without changing the ranking, and resolution cannot be fixed without new signal. A model with REL near
zero and RES near zero is a well-calibrated model that knows nothing, which is exactly what a
constant predictor achieves.

### 4.4 Limits and counterexamples

Each limit below is a place where the formula is still computable and no longer means what the reader
assumes.

**Limit 1: Wald intervals at a boundary.** The Wald interval `p ± z*sqrt(p(1-p)/n)` collapses to zero
width when `p = 0` and is badly asymmetric when `p` is small, because the estimated variance is zero at
the boundary. With 0 successes in 300 trials the Wald interval is `[0, 0]`, which asserts certainty
from an observation that is entirely compatible with a 1% rate. The Wilson interval on the same data
gives an upper bound near 0.0128, and the exact one-sided bound is `1 - 0.05^(1/300) = 0.00994`. A
zero-count result is strong evidence of a low rate; it is never evidence of a zero rate.

**Limit 2: binned statistics depend on the bins.** The population stability index and expected
calibration error both average over bins, so both change when the bins change with identical data.

```text
FORMULA (PSI):  PSI = sum_i (q_i - p_i) * ln(q_i / p_i)
WHERE: p_i = reference share in bin i [fraction]; q_i = current share in bin i [fraction];
       every share positive; shares each sum to one.
CONSTRAINT: identical bin edges and identical feature semantics between the two periods; a zero share
            makes the term infinite, and any smoothing changes the estimator.
NUMERICAL EXAMPLE: p = (0.40, 0.35, 0.15, 0.09, 0.01), q = (0.30, 0.33, 0.18, 0.13, 0.06) gives
    contributions (0.02877, 0.00118, 0.00547, 0.01471, 0.08959), total PSI = 0.13971.
INTERPRETATION: the movement is concentrated in the sparsest bin, which is also the least stable one.
```

Adding `0.001` to every share and renormalising changes the same comparison to 0.13484, a 3.5% change
from a smoothing choice rather than from data. Worse, the sparsest bin carries 64% of the total, so a
handful of records in the tail can move the statistic more than a broad, real shift elsewhere. The
diagnostic is to print contributions per bin, not just the total.

**Limit 3: discrimination metrics are invariant where decision metrics are not.** The area under the
ROC curve does not change when prevalence changes, because it is computed from the true- and
false-positive rates. Average precision does change, because it is computed from precision, which
depends on prevalence. With the same five score bands, scaling the negatives by five leaves AUC at
0.90758 and moves average precision from 0.43095 to 0.15453. A reader who compares an AP figure across
two populations with different prevalence is comparing two different questions.

**Limit 4: the cost-optimal threshold assumes calibrated probabilities.** `p* = C_FP/(C_FP+C_FN)`
is a statement about probability, not about score. If the score is a rank squeezed through a sigmoid by
a different process, the numeric value 0.16 may correspond to a very different operating point than
the probability 0.16. The check is to compute the observed positive rate among cases scoring near 0.16;
if it is not near 0.16, the score is not the probability the rule requires.

**Limit 5: maturity models extrapolate.** Fitting `1 - exp(-d/tau)` to observed coverage and reading
off a future date assumes a stationary arrival process. An adjudication backlog, a holiday, or a
processor change breaks stationarity and the extrapolation with it. Section 5.5 fits such a model to
this sheet's illustrative data and shows what a single-parameter fit cannot see.

**Takeaway:** every formula here has a region where it is computable and meaningless; name that region
before the number reaches a decision document.

## 5. Worked Numerical Example

All inputs in this section are ILLUSTRATIVE. The arithmetic is exact at the precision printed and can
be reproduced from the inputs alone.

### 5.1 The cohort

A scoring service evaluates card-not-present payment attempts. In the reference month it produced
20,000 eligible decisions: unique attempts, complete feature vectors, excluding internal test accounts,
identified by attempt key, scored by model version 7 and feature-set version 12.

The outcome definition is adjudicated fraud within a 30-day horizon from the decision. The reporting
cut-off is 45 days after the last decision in the window. The 15-day buffer beyond the horizon exists
because adjudication itself takes time; it is an allowance, not a guarantee.

At the cut-off, 18,480 decisions have a usable adjudication. 1,520 do not: some are still pending with
the issuer, some failed the join on a case identifier, and a small number were adjudicated as
indeterminate. Those 1,520 remain `unknown`. They are not negatives, and coding them as negatives is
the single most common way to make an evaluation look better than it is.

Among the 18,480 with labels there are 1,000 positives and 17,480 negatives. Coverage is
18,480/20,000 = 0.924, checked in section 5.7.

### 5.2 The operating point

The service alerts when the score exceeds 0.25. The resulting partition is the one used in section 4.2.

```text
                          outcome positive   outcome negative   total
predicted positive                750              1,283         2,033
predicted negative                250             16,197        16,447
total                            1,000             17,480        18,480
```

Derived quantities, each recomputed by hand:

- precision `750/2,033 = 0.36891`, Wilson `[0.34820, 0.39012]`
- recall `750/1,000 = 0.75000`, Wilson `[0.72224, 0.77585]`
- false-positive rate `1,283/17,480 = 0.07340`, Wilson `[0.06963, 0.07736]`
- specificity `16,197/17,480 = 0.92660`
- negative predictive value `16,197/(16,197 + 250) = 16,197/16,447 = 0.98480`
- accuracy `(750 + 16,197)/18,480 = 16,947/18,480 = 0.91705`
- balanced accuracy `(0.75 + 0.92660)/2 = 0.83830`
- lift over prevalence `0.36891/0.05411 = 6.8175`
- F1 `2 * 0.36891 * 0.75/(0.36891 + 0.75) = 0.49456`

Two of these deserve comment because they are routinely misread. Accuracy 0.91705 sounds strong until
it is compared with the accuracy of never alerting, which is `17,480/18,480 = 0.94589` — the do-nothing
policy is more accurate than the deployed model. And F1 0.49456 is a compromise between precision and
recall that encodes no information about the relative cost of the two errors; it is a summary of the
operating point, not a justification for it.

### 5.3 The threshold ladder

Five thresholds were evaluated on the same mature cohort. Each row is a complete partition; the bands
are nested, so TP and FP both increase as the threshold falls.

| Threshold | TP | FP | Precision | Recall | FPR | Alerts | Cost per decision |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 210 | 90 | 0.70000 | 0.21000 | 0.00515 | 300 | 1.83442 |
| 0.45 | 450 | 383 | 0.54022 | 0.45000 | 0.02191 | 833 | 1.41580 |
| 0.25 | 750 | 1,283 | 0.36891 | 0.75000 | 0.07340 | 2,033 | 1.12359 |
| 0.10 | 900 | 2,633 | 0.25474 | 0.90000 | 0.15063 | 3,533 | 1.36710 |
| 0.00 | 1,000 | 17,480 | 0.05411 | 1.00000 | 1.00000 | 18,480 | 7.56710 |

Cost per decision uses `C_FP = 8` and `C_FN = 42` over the 18,480-decision denominator:

```text
t = 0.25: (8 * 1,283 + 42 * 250) / 18,480 = (10,264 + 10,500) / 18,480 = 20,764 / 18,480 = 1.12359
t = 0.10: (8 * 2,633 + 42 * 100) / 18,480 = (21,064 + 4,200) / 18,480 = 25,264 / 18,480 = 1.36710
t = 0.45: (8 * 383 + 42 * 550) / 18,480 = (3,064 + 23,100) / 18,480 = 26,164 / 18,480 = 1.41580
t = 0.70: (8 * 90 + 42 * 790) / 18,480 = (720 + 33,180) / 18,480 = 33,900 / 18,480 = 1.83442
never alert: (0 + 42 * 1,000) / 18,480 = 42,000 / 18,480 = 2.27273
```

The minimum over the ladder is at 0.25. Note that the ladder is coarse: the theoretical optimum is
`p* = 8/50 = 0.16`, which lies between 0.10 and 0.25, and the true cost curve between those two
thresholds is not sampled here. The correct statement is "0.25 is the best of the five evaluated
points", not "0.25 is optimal". Section 5.8 quantifies the sensitivity of this choice.

### 5.4 What the ladder does not show

Three quantities are missing from the table and each has changed a promotion decision somewhere.

**Review workload.** 2,033 alerts in the month is 67.8 per day at 30 days of traffic. The team's stated
capacity is 90 reviews per day. Utilisation is `67.8/90 = 0.753`; the queue is stable but not idle, and
the daily peak matters more than the mean (section 9.3). At threshold 0.10 the mean rises to 117.8 per
day, which exceeds capacity, so that row is not implementable regardless of its cost figure.

**Segment composition.** The aggregate hides the slices. A slice of 900 decisions produced 120 alerts
against an expected `900 * 0.10165 = 91.5` at the overall alert rate; the standardised deviation is
`(0.13333 - 0.10165)/sqrt(0.10165 * 0.89835/900) = 3.145`, which is outside a 3-sigma band. With 40
slices examined routinely the Bonferroni threshold at family-wise 5% is `z = 2.807`, so this one
survives; most single-slice excursions of similar size do not (section 5.9).

**Downstream consequences.** Cost per decision prices a false positive at 8 cu, which assumes the alert
consumes review time and nothing else. If an alert also blocks the transaction, the false-positive cost
includes the customer's abandonment, which is usually larger than the review cost and is not measured
here. A cost table built only from measurable costs systematically prefers aggressive alerting.

### 5.5 Maturity, fitted and stress-tested

Coverage of 0.924 at 45 days implies an arrival curve. Fitting the single-parameter exponential
`coverage(d) = 1 - exp(-d/tau)` to the observed value gives

```text
solve 1 - exp(-45/tau) = 0.924  =>  exp(-45/tau) = 0.076  =>  tau = -45 / ln(0.076) = 17.462 days
implied half-life = tau * ln(2) = 12.10 days
```

Three consequences follow, and the third is the one that changes behaviour.

First, the model implies 0.8206 coverage at 30 days and 0.9678 at 60. If the cut-off were moved to 60
days, roughly 876 additional labels would appear for the same cohort — an 4.7% increase in denominator
without any change in the model.

Second, the model implies that 8% of outcomes arrive after the reporting cut-off for this horizon. A
quarterly report that never revisits the number is therefore not a snapshot of the truth; it is a
snapshot of what was knowable 45 days after the fact.

Third, the single-parameter fit cannot distinguish a slow but steady adjudication process from a
process with a backlog. Both produce the same coverage at one date. The discriminating observation is
the coverage curve at two dates: a stationary process moves forward with the cohort, whereas a backlog
freezes coverage in place while pending counts accumulate at the intake rate of 1,520/45 = 33.8 per
day. Only the second is an incident.

### 5.6 Coverage bias, quantified

Suppose the labelling process is not random: 95% of confirmed-fraud alerts receive adjudication
because the issuer pursues them, while only 60% of cleared alerts do, because nobody chases a cleared
alert. The labelled alert set is then 1,482.3 records of the 2,033 alerts, and the number of those that
are genuinely fraudulent is `750 * 0.95 = 712.5`.

```text
labelled-only precision = 712.5 / 1,482.3 = 0.48067
true precision          = 750 / 2,033    = 0.36891
optimism                = 11.18 percentage points
unlabelled alerts       = 2,033 - 1,482.3 = 550.7
```

The direction is predictable from the selection mechanism: precision is computed as correct alerts over
labelled alerts, and the numerator is over-represented among the labelled. Without knowing which
unlabelled alerts are correct, the honest bounds are `[0.36891, 0.63979]`, where the upper bound
allocates every unlabelled alert to the correct class. A report that prints 0.48 without the bounds is
not wrong about its own subset; it is answering a different question from the one the roadmap asked.

The correction in section 4.3.2 recovers 0.36891 exactly when the labelling probabilities are per-class
constants, because the weights cancel. It fails when the probability depends on the score within class
— for example when high-scoring cleared alerts receive extra scrutiny because an analyst was
suspicious. That mechanism reorders cases within class, and no per-class weight corrects it; only an
audited random sample does.

### 5.7 Independent arithmetic re-check

Three checks that do not reuse the formulas above, run because a mistake in one path is invisible from
inside that path.

```text
CHECK 1 (row and column sums):
  rows:    750 + 1,283 = 2,033 ;  250 + 16,197 = 16,447 ;  2,033 + 16,447 = 18,480
  columns: 750 + 250 = 1,000   ;  1,283 + 16,197 = 17,480 ;  1,000 + 17,480 = 18,480
  both margins agree with the stated totals, so the partition is internally consistent.

CHECK 2 (coverage from components):
  18,480 labelled / 20,000 eligible = 0.924 ; 1,520 pending / 20,000 = 0.076 ; sum = 1.000
  1,520 / 45 days = 33.78 pending decisions per day of cohort age.

CHECK 3 (cost identity):
  cost per decision at 0.25 = 20,764 / 18,480 = 1.12359
  cost of the do-nothing policy  = 42,000 / 18,480 = 2.27273
  saving of the deployed policy  = (42,000 - 20,764)/18,480 = 21,236/18,480 = 1.14914
  and 2.27273 - 1.12359 = 1.14914, so the two routes agree.
```

The third check is the useful one operationally: it states the value of the model as 1.15 cu per
decision relative to doing nothing, which is a sentence a finance reviewer can act on and does not
require them to trust a ratio.

### 5.8 Sensitivity of the threshold choice

The threshold recommendation rests on two inputs that are estimates, not facts. Both should be moved
before the recommendation is published.

```text
SENSITIVITY A (cost of a miss):
  C_FN = 21  => p* = 8/(8+21) = 0.2759   (alert less; the current 0.25 is slightly aggressive)
  C_FN = 42  => p* = 8/(8+42) = 0.1600   (the base case)
  C_FN = 84  => p* = 8/(8+84) = 0.0870   (alert more; the current 0.25 is far too conservative)
  At C_FN = 84 the cost at 0.25 becomes (8*1,283 + 84*250)/18,480 = 1.56323, while the cost at 0.10
  becomes (8*2,633 + 84*100)/18,480 = 1.47316, so the ranking reverses.

SENSITIVITY B (prevalence):
  precision at fixed recall 0.75 and FPR 0.07340, for prevalence pi:
    pi = 0.005 -> ppv 0.0488 ;  pi = 0.02 -> 0.1726 ;  pi = 0.05 -> 0.3497 ;  pi = 0.10 -> 0.5317
  A population shift from 5% to 2% prevalence reduces precision from 0.35 to 0.17 with no model change.
```

Sensitivity A is the reason a threshold decision must record the cost inputs and their provenance. A
review that says "we moved the threshold to 0.25 because it maximised expected value" and does not say
which cost of a miss was assumed has recorded an output and lost the input, which makes the decision
unauditable when the cost assumption is later revised.

Sensitivity B is the reason a stability monitor on precision alone generates false alarms during
seasonal prevalence changes. The threshold on precision should be conditioned on prevalence, or the
monitor should track recall and false-positive rate separately and reconstruct precision from them.

### 5.9 Alert-rate monitoring with the right denominator

The fast-loop signal for this service is the alert rate: alerts divided by eligible decisions. Its
reference value is `2,033/20,000 = 0.10165` for the month, or about 67.8 alerts per day at 667
decisions per day.

```text
FORMULA (control band):  band = p0 ± 3 * sqrt(p0(1-p0)/n)
WHERE: p0 = reference rate [fraction]; n = decisions in the monitored window [count].
CONSTRAINT: independent decisions within the window; one decision per key.
NUMERICAL EXAMPLE (daily, n = 667): sqrt(0.10165*0.89835/667) = 0.011701,
    band = [0.06655, 0.13675], i.e. roughly 44 to 91 alerts per day.
INTERPRETATION: a day with 100 alerts is outside the band; a day with 70 is not evidence of anything.
```

Because the band is wide at daily volume, a genuine increase in alert rate takes time to separate from
noise. Using the standard two-proportion power formula at 3-sigma with 80% power:

```text
FORMULA:  n = (z_a * sqrt(2*pbar*(1-pbar)) + z_b * sqrt(p0(1-p0) + p1(1-p1)))^2 / (p1-p0)^2
WHERE: p0 = reference rate [fraction]; p1 = shifted rate [fraction]; pbar = (p0+p1)/2;
       z_a = 3.0 for a two-sided 0.27% level; z_b = 0.8416 for 80% power;
       n = decisions required in the monitored window [count].
CONSTRAINT: independent decisions, one decision per key, and a single decision rule per window;
            a cumulative-sum chart does not use this formula.
NUMERICAL EXAMPLE: p0 = 0.10165, p1 = 0.13215 (a 30% relative rise), pbar = 0.11690
    sqrt(2*0.11690*0.88310) = sqrt(0.206469) = 0.45439
    sqrt(0.10165*0.89835 + 0.13215*0.86785) = sqrt(0.091317 + 0.114686) = sqrt(0.206004) = 0.45388
    numerator = (3.0*0.45439 + 0.8416*0.45388)^2 = (1.36317 + 0.38198)^2 = 1.74515^2 = 3.04554
    n = 3.04554 / (0.03050)^2 = 3.04554 / 0.00093025 = 3,274 decisions, about 4.9 days at 667/day.
INTERPRETATION: a 30% rise needs about five days of data at this volume, not one day.
```

A 10% relative rise, from 0.10165 to 0.11182, needs about 27,233 decisions, or 40.8 days. An 8% rise,
to 0.10978, needs about 42,195 decisions, or 63.3 days. That is the latency no threshold can shorten;
only a cumulative-sum design can, because it accumulates evidence rather than requiring each window to
stand alone.

A CUSUM on the alert rate with reference value `k = 0.5` standard deviations and decision interval
`h = 5` has an approximate in-control average run length near 295 windows and an out-of-control average
run length of about 10.3 windows for a one-sigma shift, 5.7 for 1.5 sigma, and 3.9 for two sigma.
Those numbers are what justifies preferring CUSUM for slow drifts: the fixed-band test above would take
weeks to react to a one-sigma shift that CUSUM flags in about ten windows.

### 5.10 The agentic variant, measured the same way

The same protocol applies to a run-based service, with the run replacing the decision and a rubric
replacing the adjudication.

ILLUSTRATIVE: 600 agent runs in a day. 540 complete without an infrastructure error. Of those, 402
satisfy an independently applied rubric. First-attempt success, ignoring retries, is 300 runs.

```text
completion rate        = 540/600 = 0.90000  (Wilson [0.87339, 0.92152])
rubric success per run = 402/600 = 0.67000  (Wilson [0.63140, 0.70644])
success among completed= 402/540 = 0.74444
first-attempt success  = 300/600 = 0.50000
retry contribution     = (402-300)/300 = 0.34000 of first-attempt successes, i.e. +34%
```

The headline 0.67 is 34% higher than the 0.50 a user experiences on their first attempt, and both are
true statements about different quantities. The retry-inflated figure answers "does the system
eventually produce an acceptable answer"; the first-attempt figure answers "how often is the first
answer acceptable". Product decisions usually need the second, engineering throughput usually needs
the first, and a document that prints one while implying the other is a defect.

Cost per successful run, with total run cost 336 cu, is `336/402 = 0.83582`; cost per attempt is
`336/600 = 0.56000`. The ratio is exactly the success rate's reciprocal effect, which is why cost per
success improves superlinearly as success rises: at 0.50 success the same 0.56 per attempt yields
1.12 per success.

**Takeaway:** the worked example interlocks at the partition, the cost ladder, and the coverage
correction; a reader who has reproduced those three can rederive every other figure in the sheet.

## 6. Visual / Geometric Intuition

### 6.1 The partition as an area

```text
|<----------------------------- 20,000 eligible decisions ----------------------------->|
|<------------------------ 18,480 labelled (0.924) --------------------->|<- 1,520 ->|
|                                                                        |  pending  |
|   TP 750        |   FP 1,283          |   FN 250    |     TN 16,197    |  unknown  |
|<--- alerts 2,033 --->|<--------------- not alerted 16,447 ------------->|
|<--------- positives 1,000 --------->|<--------- negatives 17,480 -------->|
```

Read across: the first brace is coverage, the second is the alert set, the third is the outcome. Each
brace slices the same rectangle a different way, and any metric that uses two different braces without
saying so is comparing areas that do not align.

### 6.2 The ROC curve as a staircase

With five score bands the ROC curve is a staircase, not a smooth arc. Each step is a threshold where
the model's decision changes for an entire band.

```text
TPR
1.00 |                                              *  (threshold 0.00)
     |                                        *
0.75 |                              *
0.45 |              *
0.21 |     *
0.00 +-----|--------|--------|--------|--------|------> FPR
        0.005    0.022    0.073    0.151    1.000

Area under this staircase, summed as trapezoids: 0.90758
Area under the precision-recall staircase:       0.43095
```

Two geometric facts follow. First, the area is an average over all thresholds including ones nobody
would use, so a model can have high AUC and a poor operating point. Second, the staircase's leftmost
steps are short in FPR but long in TPR only when the model is well separated at the top; a curve that
reaches 0.21 TPR at 0.005 FPR is separating the extreme positives well and the middle poorly, which is
the normal shape for fraud.

### 6.3 The cost curve and its flat bottom

```text
cost/decision
2.27 |*  never alert
     |
1.83 |     *
1.42 |           *
1.37 |                    *
1.12 |              *   <- 0.25, best of five sampled points
1.00 +---|------|----|----|----|------> threshold
       0.00  0.10 0.16 0.25 0.45 0.70
                  ^
                  theoretical optimum p* = C_FP/(C_FP+C_FN) = 0.16
```

The curve is asymmetric: moving from 0.25 to 0.10 costs 0.24 cu per decision because false positives
grow faster than false negatives shrink, while moving from 0.25 to 0.45 costs 0.29 cu because missed
positives become expensive. The flat bottom is roughly half an order of magnitude wide, which is why
the threshold can be chosen for operational reasons — review capacity, customer friction — without much
economic penalty, provided the choice stays inside that bottom.

### 6.4 The time axis: event time versus processing time

```text
event time   |------ 30-day outcome horizon ------|
                                                  \
processing   |---------------- 45-day cut-off -----------------|
time                                                          \
                                        labels still arriving  |
                                                               v
                                                        as-of snapshot
```

A query that filters on processing time gets a different set of rows each day for the same event-time
window. A query that filters on event time and then joins whatever labels exist at query time gets a
stable cohort with a moving label set. Only the second is a reproducible evaluation; the first is a
report whose denominator changes while nobody edits it.

### 6.5 Geometry of the coverage correction

```text
labelled-only view:   [================ 1,482.3 labelled alerts ================]
                      correct 712.5  ->  0.48067

full eligible view:   [==== 1,482.3 labelled ====][=== 550.7 unlabelled ===]
                      correct 750.0            ->  0.36891

the extra 37.5 correct alerts are the ones the labelled view cannot see,
because the labelling mechanism favoured the class where they sit
```

The diagram is the argument: the labelled view is not a small version of the full view, it is a
different shape, and the missing piece is not missing at random.

**Takeaway:** draw the braces before computing the ratio; a metric is only as well-defined as the
boundaries of the set it is computed over.

## 7. Comparison / Selection Table

### 7.1 Choosing the measurement instrument

| Instrument | Use when | Strength | Disqualifier | Follow-on evidence |
|---|---|---|---|---|
| Request SLI | Operational detection must be immediate | Exists before any label | Correct responses can be wrong | Trace sample plus quality spot check |
| Offline holdout | Candidate versions must be compared cheaply | Reproducible, paired, fast | Stale or contaminated fixture | Fresh production cohort |
| Prospective labelled cohort | A quality claim will be published | Measures the real population | Waits for maturity | Coverage and selection audit |
| Feature-shift statistic | Labels are far in the future | Early investigation cue | Shift need not cause harm | Pipeline and outcome diagnostics |
| Score-shift statistic | Action volume matters operationally | Detects policy effects | Stable scores hide concept drift | Slice-level quality |
| Calibration audit | Probabilities drive a decision | Separates reliability from resolution | Needs enough positives per bin | Recalibration plus regression suite |
| Human rubric | Outcomes are nuanced or safety-relevant | Captures qualitative failure | Rater variance and cost | Double scoring plus adjudication |
| Model judge | Volume exceeds human capacity | Cheap, fast, versionable | Correlated with the model it grades | Calibration against humans plus drift checks |
| Randomised experiment | The decision is whether the change caused the effect | Causal under design assumptions | May be unsafe or infeasible | Guardrails, adherence, power analysis |
| Shadow run | Candidate needs live inputs without effects | Exercises the serving path | No outcome effect to measure | Paired comparison where valid |
| Canary | Rollout risk must be bounded | Limits exposure | Small population misses rare harm | Prespecified gates and rollback |

### 7.2 Choosing the summary number

| Question asked | Correct number | Wrong but common substitute | Why the substitute fails |
|---|---|---|---|
| How many alerts are correct? | precision with its denominator | accuracy | accuracy is dominated by the negative class |
| How many positives are caught? | recall on the eligible population | recall among labelled | labelled positives are selectively easy |
| Is the ranking good? | AUC or AP on a stated population | accuracy at one threshold | a threshold discards the ranking |
| Is the probability honest? | calibration curve plus Brier | AUC | ranking says nothing about level |
| Did the change help? | cost per decision or an experiment | F1 | F1 encodes no cost ratio |
| Is the service healthy? | SLI plus burn rate | mean latency | means hide tail failure |
| Did quality drift? | mature-cohort trend with intervals | score distribution shift | score shift is not quality |
| Is the agent reliable? | first-attempt success plus completion | success after retries | retries inflate the perceived rate |

### 7.3 Choosing the response to a signal

| Diagnosed mechanism | Response | Reversible | Must not do |
|---|---|---|---|
| Broken feature pipeline | Repair and replay | yes, replay from source | retrain on corrupted rows |
| Intentional population change | Re-evaluate per segment | yes | raise the global threshold to hide it |
| Prevalence change only | Recompute expected metrics | yes | retrain for a label-shift effect |
| Genuine concept drift with evidence | Candidate retrain plus prospective test | yes, staged | promote on offline gain alone |
| Review capacity exceeded | Reduce alert volume or add capacity | yes | let the backlog label the model |
| Judge or rubric drift | Recalibrate the judge, re-score a fixed set | yes | compare new scores to old scores |
| Threshold policy change | Re-run the cost ladder | yes | keep the old metric targets |

**Takeaway:** select the instrument from the claim being made, the number from the question being
asked, and the response from the diagnosed mechanism.

## 8. Failure Modes & Edge Cases

Each entry below is a diagnostic pattern. The competing causes are both live until a check separates
them, and the check is chosen so the two causes produce different observations.

### 8.1 Precision rises in the first week after launch

**Symptom.** The daily precision chart jumps from 0.31 to 0.48 within five days of a deployment and
stays there for a week before settling near 0.37.

**Competing causes.** (a) Labelling is selective by outcome latency, so the earliest labels are
disproportionately the easy, high-confidence cases. (b) The new version genuinely improved at the top of
the score range and the early traffic was concentrated there. (c) The deployment changed the alert
volume, shrinking the denominator and shifting the mix.

**Discriminating check.** Compute precision separately by label-arrival lag bucket on the *old* version's
history. If the old version also shows a 0.31-to-0.48 pattern at lag 0–5 days, cause (a) explains the
new observation and no model improvement is demonstrated. If the lag pattern is flat for the old
version, examine (b) and (c).

**Correction.** Hold the promotion decision until the cohort reaches the coverage used in the baseline,
or compare both versions at identical lag profiles.

**Prevention.** Publish every precision figure with its coverage and its lag window, and refuse to
compare figures whose lag profiles differ by more than one day.

**Verification.** Replay one deployment day at the 45-day cut-off and confirm the settled value
reproduces the baseline's coverage; then replay the same day at 5 days and confirm it reproduces the
inflated value, proving the inflation is a maturity artefact.

### 8.2 Precision falls but nothing in the model changed

**Symptom.** Model and feature versions are unchanged; precision drops from 0.37 to 0.25 over three
weeks while recall holds steady.

**Competing causes.** (a) Prevalence fell, so the same discrimination yields lower precision. (b) The
labelling process changed, admitting different alerts. (c) A feature drifted in a way that degraded
ranking.

**Discriminating check.** Decompose precision into its two inputs. If FPR and recall are unchanged and
only prevalence moved, precision falls exactly as predicted by the PPV identity — a prevalence change
from 5.4% to 2% takes precision from 0.35 to 0.17 with no model change. If FPR rose at constant
prevalence, the ranking degraded and cause (c) is live. If FPR and recall are stable but the labelled
share of alert types changed, cause (b) is live.

**Correction.** For (a), restate targets in terms of recall and FPR, which are prevalence-invariant, and
recompute the expected precision. For (b), audit the labelling pipeline. For (c), investigate the
feature.

**Prevention.** Monitor recall and FPR as the primary quality signals, with precision derived and
annotated with the prevalence it assumed.

**Verification.** Recompute the historical series from the raw partition and confirm the reconstructed
precision matches the dashboard for a period with no known change.

### 8.3 The alert queue grows while the alert rate is unchanged

**Symptom.** Alert volume per day is flat at about 68 against a capacity of 90, yet the open queue
climbs by dozens per week.

**Competing causes.** (a) Arrivals exceed capacity in bursts even though the mean is below it. (b) Mean
service time per alert increased, lowering effective capacity. (c) Alerts are being closed without
adjudication, so they re-enter or are counted as resolved without a label.

**Discriminating check.** Plot arrivals and completions per day as two series, not one net figure, and
check utilisation at the peak hour. At `lambda = 94.9` and `mu = 90` the utilisation is 1.054, above one,
and the queue is unstable even though the daily mean looks comfortable. If peak utilisation is below
one but the queue still grows, measure service time per alert; a rise from 8 to 11 minutes cuts
capacity by 27% with no change in staffing.

**Correction.** For (a), smooth arrivals or add peak capacity. For (b), investigate what slowed review —
usually a new evidence source or a longer form. For (c), make closure without a reason a schema
violation.

**Prevention.** Alert on utilisation and on queue age, not only on arrival rate.

**Verification.** Replay one week with arrival and completion timestamps and confirm the reconstructed
queue depth matches the observed one at the end of each day.

### 8.4 A judge-based regression suite reports a large improvement

**Symptom.** The candidate version scores 0.82 against 0.74 for the incumbent on the judge-based suite,
with 1,200 items.

**Competing causes.** (a) The generator genuinely improved. (b) The judge changed — a prompt edit, a
model upgrade, or a temperature change — and is now more lenient. (c) The candidate's outputs shifted
in style toward what the judge prefers without becoming more correct.

**Discriminating check.** Re-score a frozen set of 300 historical outputs with the current judge and
compare against the stored scores from the original run. Any change in stored-versus-rescored agreement
identifies (b) before the candidate is considered. Then have humans score 100 stratified items blind
to version; if human agreement with the judge on the candidate's items is lower than on the incumbent's,
cause (c) is live.

**Correction.** For (b), pin the judge version and re-run both arms under one judge. For (c), discard
the judge for that rubric dimension and use human scores.

**Prevention.** Treat the judge as a versioned instrument: log its model identifier, prompt hash, and
decoding parameters with every score, and re-score a fixed audit set at each judge change.

**Verification.** Re-run the suite with the incumbent scored twice under identical judge conditions and
confirm the scores match to within the judge's own test-retest variation, measured on the audit set.

### 8.5 Offline gain does not reproduce in the canary

**Symptom.** The candidate improves offline cost per decision by 0.06 cu; at 10% traffic the production
cost is worse than the incumbent's.

**Competing causes.** (a) The offline cohort differs from live traffic in prevalence or segment mix.
(b) The candidate's latency changed the timeout behaviour, so some live decisions never receive a score
and fall back. (c) The canary's scoring path uses a different feature snapshot than the offline replay.

**Discriminating check.** Compare the score distribution of canary traffic against the offline cohort
on the same day. A difference in the low-score mass where fallbacks occur identifies (b); a segment-mix
difference identifies (a). Then compare feature values for the same request key between the online
snapshot and the offline replay — any disagreement identifies (c).

**Correction.** For (a), re-weight the offline evaluation to the live mix before deciding. For (b),
separate fallback decisions from scored decisions in the live metrics. For (c), fix the snapshot
contract.

**Prevention.** Require the offline harness to consume the same feature snapshot bytes as the serving
path, and to report the fallback rate at the candidate's latency.

**Verification.** Replay the canary week offline using live feature snapshots and confirm the
reconstructed cost matches the observed canary cost within the sampling interval.

### 8.6 A drift statistic fires on an unchanged model

**Symptom.** PSI on a numeric feature crosses the team's informal 0.2 warning line overnight; the model
and its inputs are unchanged.

**Competing causes.** (a) A genuine population shift in that feature. (b) A sparse-bin effect where a
handful of records in the tail dominates the statistic. (c) A reference-period or binning change
introduced with a dashboard release.

**Discriminating check.** Print per-bin contributions. In the section 4.4 example, the sparsest bin with
a 0.05 share contributes 0.08959 of a 0.13971 total, so a tail change alone can cross a threshold. Then
check the reference period's row count and the bin edges' commit history; a change in either identifies
(c) without looking at the data.

**Correction.** For (b), use quantile bins with a minimum count per bin, and report the share of the
statistic contributed by bins below that minimum. For (c), pin bin edges and reference data as
versioned artefacts.

**Prevention.** Store bin edges, reference window, and sample sizes with every PSI value, and never
compare values computed under different binning.

**Verification.** Recompute the statistic from raw data for both the night of the alert and a quiet night,
and confirm the quiet night reproduces its stored value.

**Takeaway:** each failure above is separated from its competitors by an observation, not by a stronger
opinion; write the observation down before changing anything.

## 9. Complexity & Cost Analysis

### 9.1 Cost of the labels

Labels are the expensive input, and their cost scales with how many are needed for a decision-grade
interval. Three quantities drive it.

```text
FORMULA (sample size for a proportion):  n = z^2 * p(1-p) / e^2
WHERE: p = expected proportion [fraction]; e = half-width of the interval [fraction];
       z = 1.959964 for a two-sided 95% interval.
CONSTRAINT: simple random sampling from the eligible population; clustering inflates the required n.
NUMERICAL EXAMPLE: p = 0.36891, e = 0.03 gives n = 3.8415 * 0.23281 / 0.0009 = 993.7.
                  the same p with e = 0.02 gives n = 2,235.9; with e = 0.05, n = 357.7.
INTERPRETATION: halving the interval width costs about 2.25 times the labels and buys a claim that
                distinguishes a 3-point regression from noise.
```

At 6 cu per adjudication, a 500-record audit costs 3,000 cu, which is 0.30 cu per decision across
20,000 decisions — comparable to a tenth of the total error cost of the model. That comparison is the
right way to price measurement: a label set that costs 0.30 cu per decision to detect a 0.24 cu per
decision improvement is worth buying; one that costs 3 cu per decision to detect the same improvement
is not.

### 9.2 Cost of the compute and storage

Monitoring data grows with cardinality, not with traffic alone. A decision record with 40 numeric
features at 8 bytes each is 320 bytes, so 20,000 decisions per day is 6.4 MB per day of feature
snapshots, or 2.34 GB per year. Adding 200 high-cardinality labels per record changes the order of
magnitude: at 60 bytes per label pair the same traffic produces 240 MB per day.

Traces are larger still. Five thousand agent runs per day at 30 KB of trace payload is 0.15 GB per day,
and the payload grows with retries, tool outputs, and prompt history rather than with user count. The
operational consequence is that retention is a design parameter: keeping full traces for 30 days and
aggregates for a year is usually affordable, while keeping full traces for a year is usually not.

Model-judge scoring is the cheapest label substitute in this sheet's arithmetic: 5,000 items at 0.0008
cu each is 4 cu. The cost is not the constraint. The constraint is validity, which is why section 8.4
treats the judge as an instrument requiring calibration.

### 9.3 Queueing: when the metric becomes the outage

Review capacity turns an alert rate into a queue. For a single-server model with Poisson arrivals,

```text
FORMULA:  W = 1 / (mu - lambda)   for lambda < mu
WHERE: lambda = arrival rate [alerts/day]; mu = service rate [alerts/day]; W = mean time in system
       [days]; the model assumes exponential service and a single server.
CONSTRAINT: strictly lambda < mu; at lambda >= mu the queue has no stationary mean.
NUMERICAL EXAMPLE: lambda = 67.77, mu = 90 gives W = 1/22.23 = 0.0450 days = 1.08 hours.
                  at a 1.4x peak, lambda = 94.9 and lambda > mu, so W is undefined and growing.
INTERPRETATION: a service that looks comfortable on the daily mean is unstable in its peak hour.
```

The failure this predicts is specific: an alert rate chosen for average capacity fails during the peak,
the backlog persists into the next day, and the reviews that are skipped are the ones the analysts
judged least interesting — which is precisely the set where false positives live. Precision then rises
as the queue grows, for reasons that have nothing to do with the model. The measured behaviour is the
opposite of the intuitive expectation and is one of the most reliable signatures of this failure.

### 9.4 Cost of a delayed detection

A monitoring design trades label cost against detection latency. For the alert-rate example, detecting a
30% relative increase at 3-sigma with 80% power takes about 3,329 decisions, or 5.0 days at 667 per day.
Detecting an 8% relative increase takes roughly 42,195 decisions, or 63.3 days — over two months during
which the degraded model keeps running. A CUSUM with `k = 0.5` and `h = 5` reaches a one-sigma shift in
about 10.3 windows, so daily windows give detection in ten days for a shift the fixed-band chart would
need weeks to separate from noise.

The cost of that latency is the integral of harm over the delay. If a degraded model misses 350
positives per day at 42 cu each, one day of delay costs 14,700 cu, which dwarfs the 3,000 cu cost of the
audit that would have caught it a week earlier. This asymmetry — cheap measurement, expensive delay —
is the economic argument for monitoring investment, and it is also why monitoring budgets collapse
after a quiet quarter.

### 9.5 Complexity of the evaluation pipeline itself

An evaluation that joins four tables, applies two strata, and computes five metrics has a cost that
grows with the product of the join keys. In practice the pipeline cost is dominated by the label join,
which is a large shuffle, and by the per-segment recomputation, which is linear in the number of
segments times the row count. Forty segments over 20,000 rows is nothing; forty segments over forty
million rows is a scheduled job with its own failure modes.

The design consequence is to materialise the partition once per cohort and compute every ratio from it.
A pipeline that re-scans the raw log per metric will disagree with itself the first time the log's
retention or late-arrival behaviour changes.

**Takeaway:** price labels per decision, price storage per cardinality, and price monitoring delay as
the harm it fails to prevent.

## 10. Key Theorems / Results Reference

### 10.1 Positive predictive value identity

`PPV = se*pi / (se*pi + (1-sp)(1-pi))` where `se` is sensitivity, `sp` specificity, and `pi`
prevalence. Its force is that PPV is not a property of the model: two deployments of the same model
have different precision if their prevalences differ. Corollary: the same model in a 0.5% prevalence
population has PPV near 0.05 — barely above the base rate — while at 10% it exceeds 0.5.

### 10.2 Rank invariance of AUC and dependence of AP

AUC equals the probability that a randomly chosen positive outranks a randomly chosen negative, so it
is unchanged by any monotone transformation of scores and by resampling within class. Average precision
is a sum of precisions, each of which contains the negative count in its denominator, so it changes when
class balance changes. This is the formal reason a cross-population AP comparison is invalid while a
cross-population AUC comparison is at least well-defined.

### 10.3 Dvoretzky–Kiefer–Wolfowitz

For `n` independent draws from a distribution, the empirical CDF is within `epsilon` of the true CDF
with probability at least `1 - 2*exp(-2*n*epsilon^2)`, uniformly over all thresholds. Solving for
`epsilon` at 95% gives `sqrt(ln(40)/(2n))`. For `n = 900` that is 0.0453, for `n = 2,033` it is 0.0301,
and for `n = 20,000` it is 0.0096. The practical reading: with fewer than about a thousand labelled
cases, no calibration curve can be trusted to better than four percentage points at any point, so a
"perfectly calibrated" claim from a small sample is a claim about sampling noise.

### 10.4 Wilson versus Wald

The Wilson interval inverts the score test and therefore has non-zero width at `p = 0` and asymmetric
bounds near the edges. The Wald interval uses the estimated variance and degenerates at the boundary. For
any monitoring use where zero-count or near-zero-count cells are expected — rare harms, new segments,
safety incidents — Wald is the wrong default and its failure is silent.

### 10.5 Little's law

`L = lambda * W` for a stable system, with `L` the mean number in the system, `lambda` the arrival rate,
and `W` the mean time in system. It requires only that the averages exist, which is why it can be applied
to a review queue without assuming anything about the service distribution. Its diagnostic use is
negative: if observed queue depth, arrival rate, and dwell time violate the identity, records are being
duplicated, dropped, or counted at the wrong boundary.

### 10.6 CUSUM average run length

For a CUSUM with reference value `k` and decision interval `h`, the in-control average run length grows
approximately exponentially in `h`, while the out-of-control average run length falls as the shift
grows. With `k = 0.5` standard deviations and `h = 5`, the in-control run length is near 295 windows and
the out-of-control run length is about 10.3 windows at a one-sigma shift, 5.7 at 1.5 sigma, and 3.9 at
two sigma. The design consequence is that CUSUM is worth its complexity only when the shift of interest
is small relative to noise; for large shifts a fixed band reacts in one window.

### 10.7 Multiplicity under continuous monitoring

If a chart is inspected `m` times with an independent 3-sigma rule, the probability of at least one
false alarm is `1 - (1-0.0027)^m`. Twenty-four hourly looks give 0.0628; 365 daily looks give 0.6272. The
implication is not to stop monitoring but to predeclare the number of looks and the alpha allocation, or
to use a statistic whose false-alarm rate is defined per look by construction.

**Takeaway:** these seven results are the ones that decide whether a printed interval, a drift flag, or a
ranking comparison is admissible; cite the assumption with the number.

## 11. Interview / Exam Questions

### Q1. A cohort has 750 true positives, 1,283 false positives, and 1,520 unlabelled decisions of 20,000. State precision and its 95% interval, and say what the unlabelled decisions do to the claim.

**Answer.** Precision is `750/2,033 = 0.36891`. The Wilson interval with `n = 2,033` is
`[0.34820, 0.39012]`. The unlabelled 1,520 are outside the denominator entirely; precision is a
statement about labelled alerts, and its extension to all alerts requires an assumption about the
missing 550.7 alerts. Bounds are `[0.36891, 0.63979]`, the top end allocating every unlabelled alert to
the correct class.

**Reasoning.** The denominator of precision is the alert set, not the cohort, so coverage enters through
the alert set rather than through the population. A candidate who divides by 20,000 has computed an
unrelated quantity.

**Follow-up.** What single audit design would narrow those bounds most cheaply?

**Red flag.** Reporting `0.36891 ± 0.044` from a Wald formula, which gives a symmetric interval that
understates the upper tail here.

**Scoring.** Full credit needs the point estimate, the Wilson bounds, and the explicit statement that the
missing alerts are a separate assumption.

### Q2. Costs are 8 cu per false positive and 42 cu per false negative. What threshold is optimal, and does prevalence affect it?

**Answer.** `p* = 8/(8+42) = 0.16`. Prevalence does not enter the threshold rule: it determines how many
cases clear the threshold, not where the threshold sits. Prevalence does enter the resulting cost per
decision, which is why the same threshold produces different total loss in different populations.

**Reasoning.** The rule equates `C_FP(1-p)` with `C_FN p` at a single case. Both sides scale with the
case's own probability, and the population's prevalence never appears.

**Follow-up.** If the score is not calibrated, what does 0.16 mean?

**Red flag.** Answering 0.16 by computing the observed precision and calling it the threshold.

**Scoring.** Full credit requires the formula, the number, the prevalence-independence argument, and the
calibration caveat.

### Q3. Same service: at 0.25 the cost per decision is 1.12359. At 0.10 it is 1.36710. Which threshold should operations run, and what else must be checked before deciding?

**Answer.** 0.25 is cheaper by `0.24351` cu per decision, worth about `0.24351 * 18,480 = 4,500` cu per
month at this volume. But 0.10 produces 3,533 alerts over 30 days, or 117.8 per day, against a capacity
of 90, so it is not implementable regardless of cost. Capacity, not cost, is the binding constraint on
the aggressive option.

**Reasoning.** The cost comparison is valid only inside the feasible set. Utilisation above one makes
the queue unstable, and an unstable queue corrupts the labels that would measure the outcome.

**Follow-up.** What is the cheapest way to make the 0.10 option feasible, and what would it cost?

**Red flag.** Choosing 0.25 on cost alone and never computing the review load.

**Scoring.** Full credit gives the cost difference, the workload in alerts per day, and the capacity
comparison.

### Q4. Coverage is 0.924 at a 45-day cut-off. What does a single-parameter exponential fit imply, and what does it fail to detect?

**Answer.** Solving `1 - exp(-45/tau) = 0.924` gives `tau = 17.462` days and a half-life of 12.10 days.
The fit implies 0.8206 coverage at 30 days and 0.9678 at 60. It cannot distinguish steady slow
adjudication from a growing backlog: both produce one number at one date. The discriminator is coverage
measured at two dates plus the pending intake rate, here 33.8 decisions per day.

**Reasoning.** One observation cannot identify a two-parameter process, and a backlog is a
non-stationarity that changes the shape, not just the level.

**Follow-up.** What would moving the cut-off to 60 days buy, in labels?

**Red flag.** Treating the fitted `tau` as a measured adjudication latency.

**Scoring.** Full credit needs the solved `tau`, the extrapolations, and the explicit identification
failure.

### Q5. A judge suite shows 0.82 for a candidate and 0.74 for the incumbent over 1,200 items. What do you do first?

**Answer.** Re-score 300 frozen historical outputs with the current judge and compare with their stored
scores. If the stored-versus-rescored agreement changed, the judge drifted and the comparison is invalid
until both arms are scored under one judge version. Only after that check do human scores on a stratified
100-item sample adjudicate the residual difference.

**Reasoning.** The candidate and incumbent were scored at different times, so the instrument is a
confound. Fixing the instrument is cheaper than adding human labels and must precede them.

**Follow-up.** If the judge is confirmed stable, what sample size separates a true three-point gain?

**Red flag.** Recommending more human labels before checking whether the judge changed.

**Scoring.** Full credit requires the instrument-check-first sequencing and a concrete re-scoring design.

### Q6. Labelled alerts give precision 0.48067; the weighted estimate is 0.36891. Explain the gap and state when the weighting fails.

**Answer.** The labelling probabilities are 0.95 for confirmed fraud and 0.60 for cleared alerts, so
correct alerts are over-represented among labelled ones: 712.5 of 1,482.3 labelled alerts are correct,
while the true count is 750 of 2,033. The gap is 11.18 percentage points. The weighting fails when
labelling probability depends on the score within class, which reorders cases inside a class and cannot
be corrected by a per-class weight.

**Reasoning.** Inverse-probability weighting corrects class-level selection, not within-class ranking.

**Follow-up.** What audit design estimates the within-class dependence?

**Red flag.** Describing the gap as random noise that a larger sample would remove.

**Scoring.** Full credit requires both numbers, the mechanism, and the specific failure condition.

### Q7. Daily traffic is 667 decisions with a reference alert rate of 0.10165. Compute the 3-sigma band and say whether 100 alerts on a day is a signal.

**Answer.** `sqrt(0.10165*0.89835/667) = 0.011701`, so the band is `[0.06655, 0.13675]`, or roughly 44
to 91 alerts. A day with 100 alerts is above the upper bound and worth investigating; a day with 70 is
inside the band and is not evidence of a change.

**Reasoning.** The band is the sampling distribution of a rate at that volume, so it widens as volume
falls. Reporting a rate without its band invites reaction to noise.

**Follow-up.** How many days of data would a 30% increase need before it is detectable?

**Red flag.** Comparing 100 to the mean without converting either to a rate or a band.

**Scoring.** Full credit gives the standard error, both bounds in rate and count form, and the verdict.

### Q8. Why can a service's precision improve while its reviews are being skipped?

**Answer.** Skipped reviews do not produce labels, and the skipped alerts are the ones analysts judged
least likely to be real. The measured precision is computed over the reviewed subset, from which the
least promising alerts have been removed by the backlog. Precision therefore rises as capacity is
exceeded.

**Reasoning.** The mechanism is selection on the same variable being measured, induced by a queue. It is
visible as a simultaneous rise in queue age and fall in coverage.

**Follow-up.** Which single chart detects it?

**Red flag.** Explaining the pattern as "the model improved under pressure".

**Scoring.** Full credit names the selection mechanism, the queue as its cause, and the coverage or
queue-age signal that reveals it.

### Q9. A model's ECE is 0.0448 with six bins and 0.0404 with four. Is the model calibrated?

**Answer.** Not on this evidence. The ECE changes by 10% with the binning, on identical predictions, so
the number is a property of the binning as much as of the model. The usable statement is
"ECE = 0.045 under six equal-count bins, with 5,000 cases", plus the reliability curve and the Brier
decomposition. Here reliability is 0.00256 and resolution 0.07705, which is a nearly perfectly reliable
and moderately informative model.

**Reasoning.** Binned calibration metrics average over bins; a change of bins changes the weights and
the within-bin dispersion. The decomposition is binning-dependent too, but it separates the two
questions so a reader can see which is broken.

**Follow-up.** Which bins would you report, and why?

**Red flag.** Answering "yes, 0.045 is below 0.05" without noting bin sensitivity.

**Scoring.** Full credit requires the two ECE values, the reason, and a statement that survives rebinning.

### Q10. An agent shows 0.67 rubric success and 0.50 first-attempt success. Which number goes in the product review, and what does the other one measure?

**Answer.** The product review needs 0.50, because that is what a user experiences on their first
request. The 0.67 measures eventual success after retries, which is the right number for throughput and
cost-per-completed-task analysis. Reporting 0.67 in a product context overstates the user experience by
34% relative.

**Reasoning.** The two figures differ by the retry mechanism, which is a system property rather than a
capability. Mixing them makes the retry budget look like a quality improvement.

**Follow-up.** What is the cost per successful run at each definition?

**Red flag.** Averaging the two numbers, or reporting only the higher one.

**Scoring.** Full credit assigns each number to its decision, quantifies the gap, and explains the retry
mechanism.

### Q11. AUC is 0.90758 in the reference population. Average precision is 0.43095. In a population with one fifth the prevalence, what happens to each?

**Answer.** AUC is unchanged at 0.90758, because it is built from true- and false-positive rates and does
not involve prevalence. Average precision falls to about 0.15453, because each precision term's
denominator fills with negatives. Any cross-population AP comparison is therefore invalid without
restating prevalence.

**Reasoning.** AUC is a rank statistic averaged over class pairs; AP is a precision statistic summed
along the ranking. Only the second contains the negative base.

**Follow-up.** Which metric would you use to compare two models across two populations, and what would
you add?

**Red flag.** Quoting AUC and AP as if both were prevalence-free "quality" numbers.

**Scoring.** Full credit gives both behaviours, the mechanism, and a defensible cross-population choice.

### Q12. Design the verification for a fix to a late-arriving label join.

**Answer.** Replay one decision whose label arrived well before the cut-off, and one whose label arrived
within one day of the cut-off. The first confirms the fix does not break the normal path; the second
confirms the as-of logic actually respects the boundary. Also replay a decision whose label never
arrived, which must remain unknown rather than becoming a negative.

**Reasoning.** Boundary records discriminate; interior records pass under both the correct and the
incorrect implementation, so a fix tested only on an interior record has been narrowed to fit it.

**Follow-up.** What would you assert in the regression suite so this cannot silently regress?

**Red flag.** Verifying with the aggregate metric, which can look correct while individual joins are
wrong in offsetting ways.

**Scoring.** Full credit requires all three records, the reason each is chosen, and a durable assertion.

**Takeaway:** the questions interlock — coverage, cost, capacity, and instrument drift are the four axes
on which production evaluation claims fail.

## 12. Quick Reference Card

| Concern | Rule | Number from this sheet |
|---|---|---|
| Population | Write it as a predicate before computing any ratio | 20,000 eligible, 18,480 labelled, coverage 0.924 |
| Denomination | Precision, recall, and FPR have different denominators; state which | 750/2,033, 750/1,000, 1,283/17,480 |
| Interval | Wilson, not Wald, near zero counts | precision [0.34820, 0.39012] |
| Threshold | Cost rule, not F1: `p* = C_FP/(C_FP+C_FN)` | 8/50 = 0.16, best sampled point 0.25 |
| Capacity | Convert the alert rate to reviews per day and compare with the peak | 67.8 mean vs 90 capacity, peak utilisation 1.054 |
| Coverage | Weight by labelling probability or publish bounds | 0.48067 labelled-only vs 0.36891 weighted |
| Maturity | Fit and stress-test; one date cannot separate backlog from lag | `tau = 17.46` days, half-life 12.1 |
| Drift | Print per-bin contributions before believing a total | PSI 0.13971 with 0.08959 in one bin |
| Calibration | Report binning with the number; use the Brier decomposition | ECE 0.0448 (6 bins) vs 0.0404 (4 bins) |
| Verification | Replay the trigger and a boundary record | interior, boundary, and never-labelled cases |
