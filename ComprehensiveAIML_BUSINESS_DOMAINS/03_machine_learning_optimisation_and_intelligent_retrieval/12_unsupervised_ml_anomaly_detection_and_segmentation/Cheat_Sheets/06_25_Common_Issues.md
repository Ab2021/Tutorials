---
title: "Common Issues: Anomaly Detection and Segmentation"
topic: "Common issues in unsupervised detection and segmentation work: misdiagnosis catalogue"
collection: "unsupervised-ml-anomaly-detection-and-segmentation"
level: "advanced"
evidence: "ILLUSTRATIVE"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Common Issues: Anomaly Detection and Segmentation

> **Evidence:** every population, alert count, cost, threshold and distribution below is a constructed
> teaching value (`ILLUSTRATIVE`); every formula and arithmetic step is `DERIVED` and is printed so you
> can recompute it. No deployment, vendor benchmark or organisational result is claimed. Section 10
> cites results by author and by statement, not by measured effect size.

## 1. File Header

**Purpose.** This is the troubleshooting companion to the five teaching cards in this collection. A
detection or segmentation system rarely fails by emitting nothing. It fails by emitting output that is
*plausible* while the reason for the output is wrong: an alert that fires for a calendar reason, a
segment that exists because of a duplicated join, a threshold that was correct for last quarter's
volume. This sheet catalogues twenty-eight such mechanisms, gives each a stable identifier, separates
mechanisms that look alike, and supplies one computation per mechanism whose two outcomes lead to
different actions.

**Audience.** Two readers share this file. The first owns a live detector and has a dashboard that
disagrees with reality at 02:00. The second must defend a segmentation in a review and is being asked
how they know the groups are real. Both need the same artefact: a mechanism plus a number. Neither
needs a list of virtues.

**Prerequisites.** You should be able to compute a mean and a sample standard deviation by hand, invert
a normal tail probability approximately, read a two-by-two contingency table, and state the difference
between event time and processing time. The geometry of clustering is assumed from
`02_Clustering_and_Embeddings.md`, the capacity argument from `01_Anomaly_Detection_Triage.md`, the
index definitions from `04_Segmentation_Evaluation.md`, and the queue mechanics from
`05_Human_Review_Loops.md`. This sheet uses those results and does not re-derive them.

**How to use it.** Find the symptom in Section 7's selection table, jump to the matching identifier in
Section 8, and run the discriminating check *before* touching a parameter. Re-tuning a threshold is the
most common wrong first move, because it treats every failure as calibration. Section 8 contains
twenty-eight issues and only three of them are calibration failures.

**Identifiers.** `CI-001` through `CI-028` are stable inside this file and are cited from Sections 7, 8
and 11. A defect report should quote the identifier and the check that was run. "The model is bad" is
not a defect report; "CI-002, contamination 1.4 percent in the fit window, sigma inflated by 1.41" is.

**Category vocabulary.** Four categories exhaust the space this sheet covers, and the category decides
who fixes the problem before anyone debates the fix.

| Category | What it means | Usual owner | Wrong first move it invites |
|---|---|---|---|
| `VIOLATED_ASSUMPTION` | the method's premises are false for this data | method owner with data owner | tune the threshold |
| `NUMERICAL_EDGE` | arithmetic is undefined, saturating, or quantised | engineer | retrain the model |
| `SCALE_FAILURE` | mechanism correct, volume breaks the operation | platform owner | buy compute |
| `CONCEPT_CONFUSION` | the team answers a different question than the one asked | the reviewer | rename the metric |

**Severity rubric.** Severity is not how bad the model is. It is how long the error survives without
being noticed, because detection lag multiplies cost. A quiet error that lives a quarter outranks a
loud one a human catches in an hour.

| Severity | Definition | Response window |
|---|---|---|
| S1 | confidently wrong decisions across a whole population | same day |
| S2 | moves a reported number by more than its confidence interval | this sprint |
| S3 | degrades a diagnostic but not the decision | next review |
| S4 | cosmetic or documentation-only | when the file is next touched |

**Scope exclusions.** Supervised drift detectors, failures of the scoring service itself, and
identifiability theory for mixtures are out of scope. The first belongs to change-point work, the
second to platform on-call, the third to a statistics course. What remains is the case this sheet
exists for: the system is up, the numbers are produced, and the inference behind them is wrong.

**On the count of twenty-eight.** The number is not a claim of completeness. It is the residue after a
merge pass in which any two entries whose discriminating check was identical were collapsed, and any
entry whose check did not change behaviour was deleted. A longer catalogue would have been easier to
write and less useful to carry.

**Maintenance.** Every issue names a prevention that runs without a human. When a prevention lands as
an assertion or a CI test, the issue stays in this file as the explanation of *why* the assertion
exists. Deleting the explanation is how a team re-learns the failure two years later.

## 2. Concept Map / Mental Model

### 2.1 Six layers, and why failures do not travel upward

Every detection or segmentation pipeline is six layers. A defect at layer *i* cannot be repaired at
layer *j > i*, and this single rule settles most arguments, because an argument is usually two people
describing defects at different layers.

```text
LAYER 0  INSTRUMENT       LAYER 1  POPULATION      LAYER 2  FEATURES       LAYER 3  MODEL
sensor, log, event        eligibility, dedup,      units, scaling,       score, k, threshold,
clock, timezone           cohort, censoring        lags, encoding        initialisation
      |                        |                        |                     |
      v                        v                        v                     v
                         LAYER 4  DECISION                          LAYER 5  ACTION
                         capacity, queue, review,                  who does what, feedback,
                         cost per alert, SLA                       closure codes, retraining
```

Layers are ordered by how silently they corrupt everything downstream. Clock skew at layer 0 emerges as
an anomaly at layer 3 and as a queue spike at layer 4, and no layer-3 parameter removes it. A duplicated
join at layer 1 emerges as a new segment at layer 3 and as a campaign that underperforms at layer 5.

### 2.2 Five attribution questions

When an alert is wrong, exactly one of five statements is true, and they become mutually exclusive only
after measurement:

- The event is not anomalous and the score is high for a legitimate reason the model of "normal"
  excludes, which is a layer-2 or layer-3 defect.
- The event is anomalous but nobody would act, so the alert is a detection success and a definition
  failure living at layer 4.
- The event did not happen and the pipeline manufactured it, which is layer 0 or layer 1.
- The event happened but arrived after the decision timestamp, so the detector is judged on information
  it could not have had; the defect is at layer 1 with a processing-time cause.
- The score did not move and the threshold did, because the calibration population changed; that is
  layer 3 with a different mechanism from a model change.

The five map onto the four categories plus one cross-cutting item, which is why category is the first
field of every catalogue row.

### 2.3 Symptom families

Symptoms cluster, and the cluster narrows the search before any computation is run.

```text
too many alerts -----------> capacity, prevalence, multiplicity, threshold recomputation
no alerts at all ----------> saturation, index collapse, upstream silence read as health
segments move overnight ---> label swap, encoder change, k change, real migration
segment sizes look wrong --> grain, dedup, population filter, missing cohort
metrics look excellent ----> leakage, label maturity, population mismatch
numbers differ between runs -> seed, tie-breaking, float precision, window boundary
```

"No alerts" is the symptom least often investigated and the only one that can mean total failure. A
detector that stopped scoring and a detector that correctly found nothing produce identical dashboards
until something emits a heartbeat carrying row counts and a score distribution.

### 2.4 Triage order

Order the steps by cost, cheapest first, and stop as soon as a step explains the symptom.

1. Count rows in and rows out per hour for seven days. A moved ratio ends the investigation at layer 0
   or layer 1, and it takes one query.
2. Print the score distribution's median and 99th percentile per day. A moved median is a population
   change; a moved tail with a fixed median is a model or threshold change.
3. Recompute today's alerts under yesterday's threshold, then yesterday's alerts under today's. One
   comparison separates threshold movement from score movement.
4. Only now open the model, the feature code, or the fit window.

Steps one to three explain the majority of the entries in Section 8, and together they cost minutes.
Practitioners skip them because the model is the interesting object, which is exactly why a broken
threshold survives a week.

### 2.5 The one invariant worth memorising

For a fixed false-positive rate `fpr` and recall `r`, precision of an alert stream is bounded by the
base rate. Writing `pi` for prevalence among scored units:

```text
PPV = r*pi / (r*pi + fpr*(1-pi))
```

With `pi = 1.667e-4` (400 incidents among 2.4e6 units), `r = 0.885`, `fpr = 1.35e-3`, the expression
gives `0.0985`: fewer than one alert in ten is real. No ranking improvement rescues a stream whose
false-positive rate is not small *relative to* the base rate. The ratio `fpr/pi`, not the AUC, is the
number a reviewer should request, and here it is `8.1`.

### 2.6 Two standing rules that come from the questions

The first rule: whenever the words "customers" and "customer-days" appear in one sentence, ask for the
`GROUP BY`. The unit of scoring is a design decision, and a per-day model gives an active customer as
many chances to alert as the number of days they appear, which inflates every per-customer rate in
proportion to activity.

The second rule: whenever a precision figure shares a dashboard with a quality filter, ask for the
denominator. The filter changes `pi`, and `pi` moves precision through the formula in 2.5 with no change
in the model. In the constructed example of CI-027, the same alert set shows precision 0.522 on filtered
units and 0.082 on all units, a factor of 6.4 produced entirely by the denominator.

### 2.7 Masking: why the first plausible cause is usually not the cause

Entries interact systematically rather than incidentally. Three patterns account for most misdiagnosis:

| Masking pattern | What it looks like | What is actually happening |
|---|---|---|
| estimator masking | one outlier hides another | the inflated standard deviation pushes the second point inside the cut |
| duplication masking | a seasonal pattern appears | a backfill duplicated a subset of hours, so one day looks busy |
| staleness masking | a measurement change appears | a cached reference window was not refreshed, so scores drift |
| filter masking | precision improved | the population filter removed the hard negatives |

The masking map matters because the visible symptom and the mechanism are separated by one layer. A
team that fixes the symptom adds a second defect on top of the first, and the composite is harder to
diagnose than either alone.

### 2.8 How to read a catalogue entry

Every row in Section 8 carries the same fields in the same order, because the order is the reasoning.

| Field | Required content | Sign the entry is weak |
|---|---|---|
| Category | exactly one vocabulary term | two terms means it is two issues |
| Symptom | visible without model internals | stated as a metric, so the metric becomes the finding |
| Competing causes | at least two mechanisms | one cause makes it a note, not a diagnosis |
| Discriminating check | a computation with two regions | "inspect the data" is not a check |
| Correction | changes the mechanism | "adjust the threshold" treats the symptom |
| Prevention | runs without a human | "be careful" is not prevention |
| Verification | measurable post-condition | identical to the correction means it is circular |

The discriminating check has a hard test: it returns a number, and the two regions of that number lead
to different actions. A check that returns a plot is a check only if the plot has a written decision
rule, otherwise it is a second opinion with no adjudication procedure.

## 3. Foundational Definitions

Each definition is written so that the corresponding failure in Section 8 becomes a statement about a
named symbol. Where a definition has an operational consequence, the consequence carries a number.

**Unit of scoring.** The object receiving exactly one score per period, for example one account-day or
one device-hour. Unit confusion is the most common cause of "too many alerts", because a model scoring
account-days and a reviewer thinking in accounts differ by the account's active-day count.

**Population at risk.** All units eligible to be scored at a time, before any filter. Population and
scored set diverge whenever a filter drops rows silently, and the divergence is the first quantity to
measure because every rate downstream is divided by one of them.

**Event time.** The UTC instant at which the modelled phenomenon occurred. Only event time may order
outcomes.

**Processing time.** The UTC instant at which a record became available, including ingestion latency,
backfill and correction. Mixing the two timestamps credits a detector with foreknowledge, and the error
shows up only as a live-versus-replay gap.

**Score.** A real number where larger means more anomalous, by convention in this file. Only ranks are
comparable across models; raw values are not, because each model has its own scale.

**Threshold.** The cut on score producing an alert. It is a decision parameter, not a model parameter,
so it belongs to whoever owns review capacity. When the two owners differ, the threshold becomes
political and drifts.

**Capacity.** Alerts the review process absorbs per unit time, computed from staff, minutes per case
and working hours. In the running example, 60 alerts per day at 6 minutes each is 6.0 reviewer-hours
per day, or 0.75 of a full-time equivalent at 8 productive hours.

**Alert.** A scored unit above threshold. An alert is a request for review, not evidence of an incident.

**Incident.** A scored unit a reviewer would confirm as requiring action, adjudicated against a written
definition. Without that definition precision and recall are undefined rather than unknown, and every
argument about the model becomes an argument about vocabulary.

**Base rate `pi`.** Incidents divided by scored units. Values between `1e-5` and `1e-2` are typical in
operational monitoring; the running example uses `400/2.4e6 = 1.667e-4`.

**False-positive rate `fpr`.** False alerts divided by all non-incident scored units, including
non-alerted ones. It is not `1 - precision`; that quantity is the false discovery rate.

**Precision (PPV).** True alerts over all alerts. It is a property of the pair (threshold, population),
so quoting it without both is uninterpretable.

**Recall.** True alerts over all incidents, and the denominator requires labels for incidents that never
alerted. Unlabelled silent incidents make the denominator unknowable, which is why recall claims decay
as label maturity changes.

**Label maturity.** The fraction of the evaluation window whose adjudication is complete. Labels mature
over days, so a recall computed on the most recent week measures the labelling process.

**Dwell time.** Two different quantities share the name: incident-onset to alert, and alert to closure.
Mixing them produces the familiar claim of detecting in forty minutes when the forty minutes were spent
in the review queue.

**Drift.** Change in the distribution of inputs, scores or outcomes, with three distinct mechanisms:
population drift (the world changed), pipeline drift (the measurement changed), and representation drift
(feature or encoder code changed). Each requires a different response, so a drift alert must name which.

**PSI.** A bucketed symmetric divergence between reference and current distributions. Dimensionless and
unbounded above, with no null distribution, so the familiar 0.1 and 0.25 cut-offs are conventions rather
than tests. In the worked example of 4.3 the statistic is 0.1767, and under a no-change null with 2000
rows in eight bins the same statistic has expectation about 0.0035.

**Effective sample size.** For a stationary series with autocorrelation `rho`, the number of independent
observations carrying the same information, approximately `n(1-rho)/(1+rho)`. It governs every interval
and every significance claim computed on a time series, and ignoring it is CI-005.

**Quantile floor.** With `n` units the threshold is an order statistic, so the achievable alert rate is
quantised in steps of `1/n`. At `n = 500` the smallest distinguishable tail rate is `0.002`, which is
twelve times the `1.667e-4` base rate you are trying to resolve.

**Contamination.** Share of the fitting window that is itself anomalous. Unsupervised detectors estimate
"normal" from the window, so contamination biases the estimate toward the anomaly, and the bias scales
the estimated spread roughly by `sqrt(1 + c*(delta^2 - 1))` for contamination share `c` and anomaly
separation `delta`.

**Leakage, unsupervised variant.** Fitting any transformation, scale, threshold or cluster on data that
includes the evaluation window. The absence of labels removes the easiest way to detect leakage; it does
not remove the leakage.

**Segment.** A subset produced by a partitional or density method and intended to receive a
differentiated action. A segment without an action is a group.

**Partition stability.** Agreement between partitions under perturbation of seed, sample or window,
reported as a distribution over perturbations. A single pairwise number is not a stability estimate.

**Label swap.** A change in the ordinal identity of a segment when membership barely moves. Ordinal ids
are arbitrary, and treating them as persistent is the most common segmentation reporting bug.

**Purity.** The share of units in the majority true class, summed over clusters. It rises mechanically
with the number of clusters, so it must be compared against its independence baseline, computed in 4.3
as 0.3333 for the constructed contingency table.

**Adjusted Rand Index.** Agreement between two partitions corrected for chance. Chance correction is why
ARI near zero means "no better than random", whereas purity near zero is impossible.

**Silhouette.** Per-point contrast between mean within-cluster distance and mean nearest-cluster
distance, normalised by the larger. It is invariant to a common rescaling of all coordinates and to
nothing else, which is CI-008's neighbour in spirit.

**CUSUM.** A sequential statistic accumulating signed deviations from a target, reset at zero. Its
in-control average run length is set by the decision boundary `h` and slack `k`, and the pair (h, k) is a
design choice with a computable false-alarm budget.

**Run length (ARL).** Expected number of observations until a signal. `ARL0` is the in-control value and
sets the false-alarm rate per monitored entity; `ARL1` sets detection delay at a stated shift.

**Saturation.** The condition in which a score or similarity stops discriminating because all values sit
at a bound, for example all cosine similarities near one in a high-dimensional embedding.

**Heartbeat.** A periodic record of row counts, score quantiles and null rates emitted by the pipeline
regardless of alert volume. Its absence is what makes CI-024 possible.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

The base case is a single threshold on a score whose distribution is known under the null, and it
produces the four quantities every later formula modifies.

**FORMULA.** With null density `f0`, alternative density `f1`, prevalence `pi`, threshold `t`, the
alert rate is `alpha(t) = P(S > t | null)`, recall is `r(t) = P(S > t | alternative)`, and precision is
`PPV(t) = pi*r(t) / (pi*r(t) + (1-pi)*alpha(t))`.

**WHERE.** A score with a stable null, one unit per row, one decision per unit, and a labelled or
assumed alternative. No temporal aggregation, so the base case has no autocorrelation to correct.

**CONSTRAINT.** `0 < pi < 1`, `alpha` and `r` monotone decreasing and increasing in `t` respectively,
and the null must be estimated from data that excludes the evaluation window.

**NUMERICAL EXAMPLE.** Take `pi = 400/2.4e6 = 1.6667e-4`, `t` at four null standard deviations, so
`alpha = 3.1686e-5`, and an alternative centred `delta = 4.2` standard deviations above the null, so
`r = Phi(4.2 - 4) = Phi(0.2) = 0.5793`. Then `pi*r = 9.657e-5` and `(1-pi)*alpha = 3.1685e-5`, giving
`PPV = 9.657e-5 / 1.2826e-4 = 0.753`.

**INTERPRETATION.** Three quarters of alerts are real at four sigma, and the arithmetic is driven by the
product `pi*r`, not by `r` alone. Doubling recall to 1.0 would move `PPV` only to 0.835, while halving
`alpha` moves it to 0.868, so at this operating point threshold placement beats model improvement.

**VALIDITY CONDITION.** The formula assumes the null estimate is unbiased and the alternative is a
single shifted mode. It breaks when the null is contaminated (CI-002), when the null is heavy-tailed
(CI-001), or when more than one incident type contributes to `pi` with different `delta` (Section 4.4).

### 4.2 General case

The general case replaces "pick a threshold" with "pick a threshold subject to a capacity constraint",
because capacity, not statistics, binds in practice.

**FORMULA.** Given daily alert capacity `C` and `N` scored units, the operating false-positive rate is
`alpha* = C/N` and the operating threshold is `z* = Phi^-1(1 - alpha*)`, provided the score is
standardised so the null is `N(0,1)`. Recall at that point is `r* = Phi(delta - z*)`, and the number of
true incidents caught per day is `pi*N*r*`.

**WHERE.** Any stream where a human reviews each alert and the queue cannot grow, which is most
operational detection. It applies to segmentation only through the analogous constraint on how many
segments an audience can act on.

**CONSTRAINT.** `C < N`, `alpha*` inside the resolvable range given `N`, and the standardisation must be
recomputed on the current population; otherwise `z*` is expressed in a stale unit.

**NUMERICAL EXAMPLE.** `N = 2.4e6`, `C = 60`, so `alpha* = 60/2.4e6 = 2.5e-5` and `z* = 4.0557`. With
`delta = 4.2`, `r* = Phi(0.1443) = 0.5574`, so expected true alerts are `400*0.5574 = 222.96` and expected
false alerts are `2.4e6*2.5e-5 = 60`.

**INTERPRETATION.** The capacity constraint is *not* the same as "60 alerts per day", because 60 true
alerts also arrive: total volume is `222.96 + 60 = 282.96`, which overshoots capacity by a factor of
4.7. Solving for the threshold that yields exactly 60 total alerts requires the full equation of 5.1,
which returns `z* = 5.2385` and recall 0.1495. Any capacity plan that ignores the true-alert term
under-provisions the queue by that factor.

**VALIDITY CONDITION.** Requires a stable null scale. It breaks when `N` varies by day (CI-020), when
the null is estimated from a different mix of entities (CI-006), or when alerts arrive in bursts so the
daily average hides an hourly peak.

### 4.3 Variants

**Robust location and scale.** Replacing mean and standard deviation with median and MAD gives
`sigma_hat = 1.4826 * MAD` and `z_robust = (x - median) / sigma_hat`, where the constant makes the
estimator consistent for a normal. In the eight-week example the standard deviation moves from 13.904
to 51.829 when one low week is added, while `sigma_hat` from the MAD stays at 17.050; the corresponding
scores for the low week are `-2.397` and `-8.241`, so only the robust version detects it.

**PSI.** With reference shares `p_i` and current shares `q_i` over `B` bins,
`PSI = sum_i (q_i - p_i) * ln(q_i / p_i)`. Its eight terms in the worked distribution are 0.024812,
0.017261, 0.002503, 0.000800, 0.009554, 0.023511, 0.033892 and 0.064378, summing to 0.176712. The last
bin contributes 36.4 percent of the total, which is the diagnostic content: the change is concentrated
in the sparse upper tail, not spread across the bulk.

**CUSUM.** For standardised observations with slack `k`, `S_t = max(0, S_{t-1} + x_t - k)` and a signal
when `S_t > h`. Siegmund's approximation for the in-control average run length is
`ARL0 ≈ (exp(2kb) + 2kb - 1) / (2k^2)` with `b = h + 1.166`. For `h = 5`, `k = 0.5` this gives 962.9,
close to the 935.6 obtained by simulation of 20,000 runs; for `h = 8` it gives 19,149 against a simulated
19,084. The approximation is good enough to size a boundary, and its error is downward at small `h`.

**Effective sample size.** For autocorrelation `rho` decaying geometrically with `m` lags,
`Neff = n / (1 + 2*sum_{j=1..m} rho^j)`. With `n = 240000`, `rho = 0.15`, `m = 10`, the sum is 0.176471
and `Neff = 177391`. Confidence width scales as `sqrt(n/Neff) = 1.1632`, so a nominal four-sigma cut has
effective height 3.4868 and the realised tail rate is `2.444e-4`, producing 587 alerts where 60 were
budgeted.

**Silhouette.** For point `i` in cluster `A`, `a(i)` is the mean distance to other members of `A` and
`b(i)` the mean distance to the nearest other cluster, with `s(i) = (b - a)/max(a, b)`. On the eight-point
example the two natural groups give a mean of 0.9320; compressing the second coordinate by ten raises it
to 0.9478, and multiplying it by a hundred leaves 0.930. A common rescaling of one axis is therefore
almost invisible, while the k=3 split of the same points falls to 0.550, which is the comparison that
carries information.

**ARI.** With `n_ij` the contingency counts, `a_i` row sums, `b_j` column sums,
`ARI = (sum_ij C(n_ij,2) - E) / (0.5*(sum_i C(a_i,2) + sum_j C(b_j,2)) - E)` and
`E = sum_i C(a_i,2) * sum_j C(b_j,2) / C(n,2)`. For the 3-by-4 table of 180 units, the observed index sum
is 2710, row sum 5310, column sum 4410, `E = 5310*4410/16110 = 1453.575`, the maximum is 4860, and
`ARI = 1256.425/3406.425 = 0.3688`.

**Poisson threshold for count features.** For an entity with rate `lambda` per period, the exact alert
cut is the smallest `c` with `P(X >= c) <= alpha_entity`. At `lambda = 40` and a per-entity annual budget
of one false alert, `alpha = 1/(40*365) = 6.849e-5`, so `z = 3.8136` and the normal cut is
`40 + 3.8136*6.3246 = 64.12`. The exact Poisson tail gives `P(X >= 65) = 1.726e-4` and
`P(X >= 68) = 3.466e-5`, so the normal approximation is wrong by a factor of 2.83 at the cut that
matters, which is CI-018.

### 4.4 Limits and counterexamples

**Limit 1: heavy tails invert the ranking of cuts.** A four-sigma normal cut has tail `3.169e-5`. Under
a Student-t with five degrees of freedom the same cut has tail `4.885e-3`, so 2.4e6 units produce 11,725
alerts instead of 60, a factor of 195. The matching t quantile at `2.25e-4` is 8.157 rather than 4.056,
a ratio of 2.01. Any calibration that assumes a normal tail on a heavy-tailed score is off by a factor
of two in the threshold and two orders of magnitude in the volume.

**Limit 2: distribution-free bounds are useless at these rates.** Chebyshev gives `P(|X| >= 3) <= 1/9`,
which is 82 times the normal tail of `1.350e-3` and larger than the base rate by a factor of 600. A
guarantee of that looseness cannot size an alert budget; it can only prove that a budget of zero is
unattainable without distributional assumptions.

**Limit 3: the quantile floor.** When the threshold is an empirical order statistic rather than a fitted
quantile, the resolvable rate is `1/n`. With `n = 500` the floor is `0.002`, 12 times the `1.667e-4` base
rate, so the empirical threshold cannot express the intended operating point at all. At `n = 4445` one
false alert is the finest distinction available, and any target below that is a statement about the
extrapolation, not about the data.

**Limit 4: the constraint can be infeasible, and then the honest answer is to say so.** At capacity 60
the recall ceiling is `60/400 = 0.15`: even with a perfect classifier that only raises true positives,
capacity caps recall at fifteen percent. A team that reports recall from a model evaluation without this
ceiling is reporting a quantity the operation cannot deliver. The correct framing is a two-part
statement — "recall 0.15 at capacity 60" — and the decision is then whether to raise capacity or accept
the ceiling, not whether to improve the model.

## 5. Worked Numerical Example

The example is the capacity-constrained operating point of 4.2, solved properly, re-checked by an
independent route, and then stressed. Every input is constructed and marked `ILLUSTRATIVE`.

### 5.1 Setup

```text
N        scored units per day            2,400,000
incidents per day                            400     so pi = 1.6667e-4
delta    alternative separation            4.2 sigma
C        review capacity per day              60 alerts
c        cost per reviewed alert          $6.50     (6 minutes at $65/hour)
v        value of a true incident handled    $420
```

The model is a standardised score with a normal null and a single shifted alternative; that assumption is
itself the subject of 4.4 and is relaxed in 5.4.

### 5.2 Solving for the threshold at capacity

Total alerts are false plus true, so the equation to solve is

```text
N*Phi(-z) + 400*Phi(4.2 - z) = 60
```

Bisection on `z` in `[0, 9]` converges to `z* = 5.2385`. Evaluating the two terms at that point:
`2.4e6 * 2.8711e-7 = 0.689` would be the false-alert count at five sigma, and at 5.2385 the tail is
`8.109e-8`, giving 0.195 false alerts; true alerts are `400 * Phi(-1.0385) = 400 * 0.14951 = 59.805`.
The sum is 60.000, confirming the root.

### 5.3 Economic reading and an independent re-check

Recall at the operating point is `59.805/400 = 0.1495`, precision is `59.805/60 = 0.9968`, and the review
cost per true incident is `60*6.50/59.805 = 6.52` dollars. Expected net value per day is
`59.805*420 - 0.195*6.50 = 25,118` dollars.

Re-check by the odds form, which uses a different algebra and therefore catches sign and denominator
errors. The posterior odds of an alert being real are
`(0.1495 / 0.8505) * (0.99983 / 1.6667e-4) = 0.17577 * 5999.2 = 1054.5` to one, giving
`PPV = 1054.5/1055.5 = 0.99905`. The direct ratio gives 0.9968. The two differ because the odds form
above used the incident rate as if incidents were the only positives; including the fact that a true
alert can also be a non-incident high score reconciles them. The discrepancy of 0.002 is the size of the
approximation, and a reviewer should expect the two routes to agree to about three digits, not more.

### 5.4 Sensitivity

**To capacity.** Solving the same equation for other values of `C` gives the whole operating curve.

| Capacity C | Threshold z* | Recall | Precision | Alerts/day | Review hours | FTE at 8h |
|---|---|---|---|---|---|---|
| 30 | 5.6399 | 0.0749 | 0.9983 | 30 | 3.0 | 0.38 |
| 60 | 5.2385 | 0.1495 | 0.9968 | 60 | 6.0 | 0.75 |
| 120 | 4.7427 | 0.2937 | 0.9945 | 120 | 12.0 | 1.50 |
| 240 | 4.1743 | 0.5103 | 0.9854 | 240 | 24.0 | 3.00 |
| 3,594 | 3.0000 | 0.8850 | 0.0985 | 3,594 | 359.4 | 44.92 |

The fifth row is the "three sigma" default, included to show that a conventional cut is not a free
choice: it costs 45 reviewer full-time equivalents and spends 90.15 percent of review time on false
alerts, because `3239.9*6.50 / (3593.9*6.50) = 0.9015`.

**To separation `delta`.** Holding capacity at 60, a weaker model with `delta = 3.2` needs `z* = 4.3930`
and delivers recall 0.1164; a stronger one with `delta = 5.2` needs `z* = 6.2364` and delivers 0.1500.
The stronger model buys 0.0336 of recall at fixed capacity, which is a 22 percent relative gain — real,
but far smaller than the 5.2-versus-3.2 gap in the statistic suggests, because the threshold absorbs most
of the improvement. This is why capacity-constrained teams get little from ranking work and a lot from
population definition.

**To prevalence.** Halving incidents to 200 per day changes the threshold only through the true-alert
term, and precision at a fixed three-sigma cut falls from 0.0985 to 0.0518. Prevalence moves precision,
not the threshold, which is the practical content of 2.5.

### 5.5 The economics disagree with the capacity constraint, and that is the finding

Maximising net value `400*Phi(4.2-z)*420 - 2.4e6*Phi(-z)*6.50` over a grid of `z` gives an optimum at
`z = 3.18` with 2,106 alerts, precision 0.1607 and net value 130,662 dollars per day. At the capacity
point `z* = 5.2385` the net value is 25,118 dollars. The economically optimal policy is worth five times
more and requires thirty-five times the review capacity.

| Rule | z | Alerts | Precision | Net value/day | Review FTE |
|---|---|---|---|---|---|
| Capacity-feasible at 60 | 5.2385 | 60 | 0.9968 | 25,118 | 0.75 |
| Value-maximising | 3.1800 | 2,106 | 0.1607 | 130,662 | 26.3 |
| Break-even precision | 2.3157 | 25,077 | 0.0155 | 0 | 313.5 |
| Conventional three sigma | 3.0000 | 3,594 | 0.0985 | 127,609 | 44.92 |

The break-even row uses `c/v = 6.50/420 = 0.015476`, the precision below which reviewing an alert costs
more than it returns. Any threshold with precision under that value destroys value regardless of how
many true incidents it finds, and the row shows that the loss region begins at a threshold of about 2.32
sigma, which is well below the three sigma default.

### 5.6 What the example does not prove

The arithmetic is exact for the stated model and proves nothing about a real detector. Its transferable
content is the shape: capacity sets recall through a ceiling that no model improvement removes; the
value-optimal threshold sits far below the capacity-feasible one whenever a true incident is worth much
more than a review; and the ratio `c/v` locates break-even precision without any model at all.

## 6. Visual / Geometric Intuition

### 6.1 The alert-volume curve is a cliff, not a slope

```text
alerts/day (log scale), N = 2.4e6, normal null
     1e5 |*
         | *
     1e4 |   *
         |     *
     1e3 |        *
         |           *
   1e2   |              *  *
         |                    *  *
   1e1   |                            *  *  *
         +----|----|----|----|----|----|----|----
             2.5   3.0  3.5  4.0  4.5  5.0  5.5   z
values:     15285 3594  862  308  161   85   39
```

Between `z = 3` and `z = 4` the volume falls by a factor of 11.7, and between 4 and 5 by another 4.4.
The practical consequence: a threshold set to two decimal places is a decision about workload, not a
detail. Moving a nominal cut from 4.00 to 4.05 removes 16 alerts per day, which is a quarter of the
capacity in 5.2.

### 6.2 Distance concentration kills nearest-neighbour intuition

Simulated pairwise distances among 300 standard normal points in `d` dimensions:

```text
d      mean distance   min     max     max/min
2        1.751        0.0027   6.251   2340
20       6.278        2.351   11.182     4.76
200     19.860       16.032   24.048     1.50
2000    63.290       58.908   67.707     1.15
```

The mean grows as `sqrt(2d)`: 20.0 predicted at `d = 200`, 19.86 observed. The spread does not keep pace,
so the *contrast* `(max - min)/mean` collapses from 3.57 at `d = 2` to 0.40 at `d = 200` and 0.14 at
`d = 2000`. In high dimensions every point is roughly equidistant from every other, and any method whose
decision is "is the nearest neighbour closer than usual" loses its signal. The minimum of 0.0027 at
`d = 2` is a near-duplicate pair, which is itself the interesting artefact: at low dimension duplicates
dominate the tail, and at high dimension nothing does.

### 6.3 Contamination is not visible in the fit window

```text
clean window (n=200,000)          contaminated by 1% at 10 sigma
mean 410                          mean 419.9 (moved 0.71 sigma)
sd   13.904                       sd   19.615
3-sigma cut 451.7                 3-sigma cut 468.8
alpha above 451.7 = 1.35e-3       alpha above 451.7 = 1.675e-2
```

The contaminated fit raises its own cut by 17.1 units and, at the original cut, admits twelve times as
many false positives. Nothing in the fit output announces the contamination; the only visible symptom is
a threshold that has drifted upward while the alert count also rose.

### 6.4 Segmentation: what a silhouette number hides

```text
two natural blobs                          same points, k = 3
  s(i) = 0.935 0.932 0.916 0.944            0.551 mean
         0.930 0.932 0.920 0.946
  mean 0.9320

second coordinate divided by 10:  mean 0.9478   (+0.0158)
second coordinate times 100:      mean 0.9300   (-0.0020)
```

Scaling one axis changes the score in the third decimal, so silhouette is not a defence against a
feature-scaling bug. It responds to *group structure*, not to units, which is why CI-007 must be caught by
inspection of the distance budget rather than by any cluster-quality index.

### 6.5 The elbow that is not one

```text
k        inertia      relative drop
1        100.0            —
2         82.0         18.0%
3         74.0          9.8%
4         70.0          5.4%
5         68.0          2.9%
6         67.0          1.5%
7         66.3          1.0%
```

The drop halves at every step from `k = 2` onward, which is the signature of a method reducing within-cluster
variance on data with no cluster structure: each added centroid splits the largest variance direction and
always reduces inertia. A curve of this shape is evidence *against* structure, and CI-014's correction is
to compare against the inertia of clustered uniform noise of the same shape rather than to pick the corner.

## 7. Comparison / Selection Table

### 7.1 Symptom to first check

| Symptom as reported | First check | Separates |
|---|---|---|
| "alerts tripled overnight" | rows-in equals rows-out per hour | pipeline duplication (CI-021) from population growth (CI-020) |
| "alerts stopped" | score distribution median and count of scored rows | saturation (CI-011) from upstream silence (CI-024) |
| "precision collapsed in QA" | recompute precision on the unfiltered denominator | filter change (CI-027) from model change |
| "the same account alerts daily" | count alerts per entity, compare to entity rate | unit confusion (CI-006) from genuine persistence |
| "segments swapped names" | Jaccard on co-clustered pairs across refits | label swap (CI-026) from real migration |
| "cluster quality is 0.93" | distance-budget share by feature family | scaling defect (CI-007) from real separation |
| "numbers differ between runs" | fix seed and re-run twice | seeding (CI-016) from float precision (CI-011) |
| "model degrades every Monday" | weekday-stratified residual | seasonality in the window (CI-004) from staffing effects |

### 7.2 Method to assumption to characteristic failure

| Method | Load-bearing assumption | Failure when violated | Issue |
|---|---|---|---|
| z-score on mean/sd | symmetric, light tails, no outliers | masking and swamping | CI-002, CI-010 |
| robust z on median/MAD | less than half the window is anomalous | zero MAD under ties | CI-010 |
| isolation forest | anomalies are few and separable | contamination above the sample rate | CI-002 |
| LOF | local density is meaningful | variable density across the population | CI-009 |
| k-means | convex, similar-size, isotropic clusters | ring and crescent structure | CI-008 |
| hierarchical, single linkage | a distance chain marks a cluster | chaining merges distinct groups | CI-007 |
| DBSCAN | one density scale for all clusters | mixed-density populations | CI-009 |
| autoencoder residual | the training set represents normal | contamination and over-capacity | CI-002, CI-019 |
| CUSUM | independent increments, known in-control mean | autocorrelation | CI-005 |

### 7.3 Threshold rule selection

| Rule | Data needed | Guarantees | Fails when |
|---|---|---|---|
| fixed sigma cut | a scale estimate | interpretable, cheap | tails are not normal (CI-001) |
| empirical quantile | at least `1/alpha` rows | exact in-sample rate | small `n` (CI-015), no extrapolation |
| capacity-constrained | `N`, `C`, `delta` | workload is feasible | `delta` unknown or unstable |
| value-optimal | `c`, `v`, prevalence | maximises net benefit | value estimates politically contested |
| sequential (CUSUM) | in-control mean, `k`, `h` | fast detection at stated delay | autocorrelated input (CI-005) |
| per-entity Poisson | entity rate, budget | rate-appropriate for each entity | sparse entities (CI-018) |

### 7.4 Choosing an index for a partition

| Index | Chance-corrected | Needs truth labels | Comparable across `k` | Use for |
|---|---|---|---|---|
| purity | no | yes | no, rises with `k` | never alone |
| ARI | yes | yes | roughly | comparing two partitions |
| NMI | partially | yes | yes | many small clusters |
| silhouette | no | no | poorly | geometry sanity only |
| Jaccard on pairs | no | no | yes | stability across refits |
| within/between ratio | no | no | no | never, it is monotone in `k` |

## 8. Failure Modes & Edge Cases

The catalogue is twenty-eight entries. Read the table for the row that matches, then run the check in
that row before applying the correction. Numbers quoted in the check column are the constructed values
from Sections 4 and 5; recompute them for your own population.

### 8.1 Violated assumptions

| ID | Category | Symptom | Competing causes | Discriminating check | Correction | Prevention | Verification |
|---|---|---|---|---|---|---|---|
| CI-001 | VIOLATED_ASSUMPTION | alert volume far above the budget at a nominal four-sigma cut | tail is heavy; or scale is under-estimated | compare empirical `P(S>4)` to `3.17e-5`: t5 gives `4.89e-3` (195x) | fit a tail model or use an empirical quantile | emit the observed tail rate at two cuts nightly | realised alert count within 15% of budget for 10 days |
| CI-002 | VIOLATED_ASSUMPTION | threshold drifts upward while false alerts rise | contamination; or population shift | refit excluding the top 2% and compare sd: 13.904 vs 19.615 at 1% contamination | robust scale or trimmed fit window | assert contaminated share below 0.5% before fit | sd change under 5% when top 2% removed |
| CI-003 | VIOLATED_ASSUMPTION | alert burst that never recurs | level shift; or one-day data error | split-window means: 410 before, 470 after, pooled sd 13.9 gives 4.3 sigma | re-baseline after a confirmed step | change-point alarm on the score median | baseline date recorded with each threshold |
| CI-004 | VIOLATED_ASSUMPTION | weekday alerts suppressed, weekend alerts inflated | seasonality inside the window; or staffing | stratify residuals by weekday and compare medians | per-weekday baseline or weekday covariate | fit windows preserve weekday composition | weekday median residual within 0.2 sigma |
| CI-005 | VIOLATED_ASSUMPTION | significance far overstated on time-series scores | autocorrelation; or duplicated rows | `Neff = n/(1+2*sum rho)`: 177,391 vs 240,000 gives width 1.163 | inflate variance by `n/Neff` | compute lag-1 autocorrelation in the metric job | effective width used in every interval |
| CI-006 | VIOLATED_ASSUMPTION | high-activity entities alert constantly | rate heterogeneity; or genuine incidents | alerts per entity vs entity event count: 40/day entity needs cut 64.1, not a global z | per-entity rate model | store entity rate with each score | top-decile entities under 2% of alerts |
| CI-007 | VIOLATED_ASSUMPTION | clusters track a categorical field | one-hot dominance; or real association | share of squared distance from one-hot: 76.9% at 40 binary columns, 98.7% with scaled numerics | weight blocks or use Gower distance | assert per-block distance share before fit | no block above 50% of mean squared distance |
| CI-008 | VIOLATED_ASSUMPTION | ring or crescent data split across clusters | non-convex truth; or bad `k` | compare k-means inertia to spectral on the same data; k-means on a ring splits it | density or spectral method | plot two principal components before choosing | cluster assignment matches generator on synthetic control |
| CI-009 | VIOLATED_ASSUMPTION | LOF flags the sparse half of a two-density population | mixed density; or contamination | LOF distribution by known stratum: bimodal at fixed `k` | normalise by stratum or use a density-ratio model | monitor LOF by stratum after deployment | false-flag rate per stratum within 2x |

### 8.2 Numerical edges

| ID | Category | Symptom seen | Causes in play | Check that separates | Correction | Prevention | Verification |
|---|---|---|---|---|---|---|---|
| CI-010 | NUMERICAL_EDGE | every score infinite or null | MAD is zero under ties; or zero-variance feature | count distinct values: if median absolute deviation is 0, the divisor is 0 | drop the feature or use a floor from a prior period | assert `scale > 0` before scoring | no infinite or null scores in a 7-day sample |
| CI-011 | NUMERICAL_EDGE | two runs differ in the last digits | float32 accumulation; or tie order | recompute in float64: difference above `1e-7` relative means precision | accumulate in float64, compare with tolerance | pin dtype in the scoring schema | max relative drift under `1e-9` across runs |
| CI-012 | NUMERICAL_EDGE | one cluster never receives points | constant feature; or duplicate centroids | count distinct rows in the feature block | drop constant features, re-seed | schema test rejects zero-variance columns | every centroid has at least one member |
| CI-013 | NUMERICAL_EDGE | crash on an empty cluster | singleton cluster; or `k` above support | print cluster sizes before the update step | reseed empty centroids to the farthest point | assert `min(size) > 0` per iteration | no empty cluster across 10 seeds |
| CI-014 | NUMERICAL_EDGE | PSI infinite or NaN | a bin with zero current share; or empty bin | inspect per-bin terms: a 0.002 to 0.0001 move contributes 0.0057, not infinity | add a small epsilon to both shares | reject bins with reference share under 1% | no non-finite PSI in the drift job |
| CI-015 | NUMERICAL_EDGE | threshold cannot express the target rate | `n` below `1/alpha`; or quantile interpolation | `1/n` vs target: at `n=500` the floor is 0.002, 12x the target | fit a parametric tail or accumulate more units | assert `n > 5/alpha` before using a quantile | target rate reproduced within one order statistic |
| CI-016 | NUMERICAL_EDGE | alert count jumps between identical runs | tie at the cut; or unordered input | count scores exactly equal to the threshold | deterministic tie-break on a stable key | sort by (score, id) before cutting | identical alert set hash across two runs |
| CI-017 | NUMERICAL_EDGE | rate reported 100x too small | percent versus proportion; or basis points | recompute: 0.02% of 120 is 0.024, not 2.4 | state units in the field name | schema names carry the unit suffix | unit test on a known conversion |
| CI-018 | NUMERICAL_EDGE | count threshold too low for sparse entities | normal approximation to Poisson; or wrong rate | exact tail at the cut: 2.87e-4 normal versus 2.87e-4 exact at 64, but 1.01e-4 normal versus 2.87e-4 exact at 63.5 | use the exact Poisson quantile | unit test the tail function against a table | realised per-entity false-alarm rate within 20% of budget |

### 8.3 Scale failures

| ID | Category | Symptom | Causes to distinguish | Separating computation | Correction | Prevention | Verification |
|---|---|---|---|---|---|---|---|
| CI-019 | SCALE_FAILURE | out-of-memory in the neighbour step | full pairwise matrix; or dense embedding | `n^2*8` bytes: 200,000 points is 320 GB, half precision 160 GB | approximate neighbours or block the computation | size assertion against the memory budget | peak memory under the pod limit |
| CI-020 | SCALE_FAILURE | alert count doubles with no model change | population doubled; or rate doubled | `N*alpha`: 2.4e6 gives 1,200, 4.8e6 gives 2,400 at `5e-4` | express the budget as a rate, not a count | threshold from a quantile of current `N` | alerts per 100,000 units stable within 10% |
| CI-021 | SCALE_FAILURE | a day looks 8% busy and full of anomalies | backfill duplicates; or real surge | duplicate key count: 1.8% of 240,000 is 4,320 rows | dedupe on (entity, event time, source) | uniqueness constraint on the fact table | row count equals distinct key count |
| CI-022 | SCALE_FAILURE | queue never clears | arrival rate above service rate | `3593.9/40 = 89.8` hours of backlog | raise the threshold or raise capacity | queue-depth alarm at 4 hours of work | p95 queue age under the SLA |
| CI-023 | SCALE_FAILURE | sample and full-population thresholds disagree | sampling bias; or variance at small `n` | refit on a 10% sample and compare `z*`; a shift above 0.2 sigma is variance | size the sample by `5/alpha` | sample only for development, never for calibration | calibration threshold reproducible from a sample |
| CI-024 | SCALE_FAILURE | no alerts for days and nobody notices | genuine quiet; or scoring stopped | scored-row count and score median per hour: a missing heartbeat is the signal | add a heartbeat with row counts | dead-man switch on the heartbeat | alert when rows scored drop 50% for 2 hours |

### 8.4 Concept confusion

| ID | Category | Symptom | Candidate causes | Discriminating check | Correction | Prevention | Verification |
|---|---|---|---|---|---|---|---|
| CI-025 | CONCEPT_CONFUSION | purity 0.69 presented as accuracy | class imbalance; or real structure | independence baseline: `0.3333` against observed `0.6944`, gain 0.361 | report gain over baseline and ARI 0.3688 | require a null baseline in the template | every partition metric printed with its baseline |
| CI-026 | CONCEPT_CONFUSION | segment names reshuffle monthly | label swap; or real migration | Jaccard on co-clustered pairs: `820/1180 = 0.695` means stable membership | match segments by centroid, not by id | stability-matched ids written to a mapping table | same-id membership Jaccard above 0.8 |
| CI-027 | CONCEPT_CONFUSION | precision improved after a filter change | population filter; or model change | recompute on all units: 0.522 filtered versus 0.082 unfiltered | report both denominators | precision field carries a population key | both values present in the report |
| CI-028 | CONCEPT_CONFUSION | recall claimed at 0.85 while capacity limits it to 0.15 | censored denominator; or optimistic labels | `C/incidents = 60/400 = 0.15` is the ceiling | state recall and capacity together | template requires the ceiling row | claimed recall never exceeds the ceiling |

### 8.5 Interaction map

Three interactions account for most composite incidents, and each is a one-line check.

The first is estimator masking: CI-002 produces the visible symptom of CI-001, because an inflated scale
hides the very points that would reveal a heavy tail. Check by trimming then re-testing the tail.

The second is duplication masking: CI-021 produces the visible symptom of CI-004, because a duplicated
hour looks like a busy weekday. Check row counts per hour before any seasonality analysis.

The third is staleness masking: CI-023's sampling error and CI-020's volume change both present as "the
threshold stopped fitting", and the separating question is whether `N` or only the sample changed.

### 8.6 Worked diagnoses

**Diagnosis A: alerts doubled, model unchanged.** The dashboard shows 2,400 alerts against a 1,200
baseline. Step one, row counts, shows scored units rose from 2.4e6 to 4.8e6 with an identical rate of
`5e-4`. That is CI-020, a scale failure, not a model failure; the rate is correct and the budget was
expressed as a count. The correction is to set the threshold from a quantile of the current population,
and the verification is alerts per 100,000 units, which stays at 50 before and after.

**Diagnosis B: an account alerts every day.** One entity contributes 240 alerts in a month. Its event
rate is 40 per day, so a global four-sigma cut at `2.5e-5` is tested 40 times daily and produces one
false alert per thousand days — yet it alerts on 240 of 30 days, which is impossible under a correct
null. The per-entity cut at `alpha = 6.85e-5` is 64.1 events, and the check is the count of days above
64: it is zero, while the count above the global cut is 30. That is CI-006, and the correction is a
per-entity rate model, not a higher global threshold.

**Diagnosis C: precision fell by a factor of six.** The model review reports 0.522; an audit recomputes
0.082. Both are correct on their denominators: 348 alerts contained 288 true positives among units
passing a quality filter, while the unfiltered denominator adds 3,240 false alerts from units the filter
removed. The check is to print `pi` under both denominators, and the correction is CI-027's: precision is
a property of a population, and the report must name it.

**Diagnosis D: one bad account hides another.** Eight weeks of one metric read 389, 398, 405, 412, 415,
421, 430 and 268. On the seven-week baseline the mean is 410 and the standard deviation 13.904, so the
low week scores `-10.21`. Including it, the mean falls to 392.25 and the standard deviation rises to
51.829, so the same observation scores `-2.40` and is missed. The MAD route gives median 408.5 and
`sigma_hat = 17.050`, scoring it `-8.24`. That is CI-002 with CI-010 as its neighbour: masking by a
single contaminating observation, cured by a robust scale rather than by a wider cut.

**Diagnosis E: an apparent seasonality that is a join.** Tuesday shows 900 alerts against a 500 baseline,
all on one product line. Row counts per line-hour show the denominator rose from 41,667 to 75,000 because
a backfill duplicated 33,333 hours. The duplicate keys carry yesterday's event time and today's processing
time, so a processing-time scatter shows the duplicates as a vertical band. That is CI-021 with CI-004
as the tempting alternative; the discriminating check is the duplicate key count, not a weekday profile.

**Diagnosis F: segments renamed with stable membership.** Six segments are refit monthly and each month
the names move. A Jaccard computation on co-clustered pairs between refits gives `820/(1000+1000-820) =
0.6949`, and centroid matching shows six of six centroids within 0.1 sigma of their previous position.
Membership is stable and the ids are unstable, which is CI-026; the correction is a mapping table and the
verification is that same-id Jaccard exceeds 0.8 after the mapping lands.

## 9. Complexity & Cost Analysis

### 9.1 Computational cost of the methods named in this sheet

For `n` units, `d` features, `k` clusters and `T` iterations:

```text
k-means            O(n*k*d*T)     200,000*6*40*50 = 2.4e9 distance-component operations
hierarchical       O(n^2 log n)   memory n^2/2 distances: 20e9 entries at n=200,000
LOF                O(n^2*d)       200,000^2*40 = 1.6e12 operations, not feasible in one pass
isolation forest   O(n*log n)     subsample 256 per tree, so cost is per tree not per n
autoencoder        O(n*d*h*T)     dominated by matrix products, GPU-bound
CUSUM              O(n) per stream  one add, one compare per observation
PSI                O(n + B)       one pass for shares, `B` terms for the sum
```

The LOF row is the important one: at `n = 200,000` the exact computation needs `1.6e12` distance-component
operations, and the pairwise matrix alone needs `320` GB in float64. Either the population is reduced, an
approximate neighbour index is used, or the method is abandoned. There is no tuning that closes that gap.

### 9.2 Human cost dominates, and it is linear in alerts

At six minutes per alert and $65 per hour, an alert costs $6.50 of reviewer time. The daily totals for
the operating points of Section 5:

| Policy | Alerts/day | Review hours | FTE | Cost/day | Cost/month (22 days) |
|---|---|---|---|---|---|
| Capacity 60 | 60 | 6.0 | 0.75 | $390 | $8,580 |
| Four sigma | 308 | 30.8 | 3.85 | $2,002 | $44,044 |
| Three sigma | 3,594 | 359.4 | 44.92 | $23,360 | $513,928 |
| Value-optimal | 2,106 | 210.6 | 26.3 | $13,689 | $301,158 |

The three-sigma row is the one to memorise: a conventional default costs half a million dollars a month
in review time at this volume, spends 90 percent of it on false alerts, and requires 45 people. Any
proposal to "start with three sigma" must carry that staffing number.

### 9.3 Cost per true incident is the ratio that survives scrutiny

Dividing review cost by true alerts gives the price of finding one incident:

```text
z = 3.0    3593.9*6.50/354.0  = $65.99 per true alert
z = 4.0     307.8*6.50/231.7  = $8.63
z = 4.5     161.0*6.50/152.8  = $6.85
z = 5.2385   60.0*6.50/59.805 = $6.52
```

The floor is the per-alert cost, $6.50, reached when precision approaches one. The ratio is therefore a
direct measure of wasted review, and it degrades gracefully at first and catastrophically later: from 4.0
to 3.0 the cost per true alert rises by 7.6 times while true alerts rise by only 1.53 times.

### 9.4 Storage and latency budget

Scores at 2.4e6 rows per day with 40 columns of float64 need `2.4e6*40*8 = 768` MB per day, or 23 GB for
a month of history, before any index. The neighbour index for approximate search typically costs 1.5 to
2 times the vector store, so the planning number is 60 GB per month of embeddings, which is the figure
that decides whether a 40-dimensional embedding is affordable at daily granularity.

Latency has two separate budgets. Scoring is usually milliseconds and rarely the constraint. Event time
to availability is the constraint, because a detector that sees an event twelve hours late cannot be
evaluated on the same day's decisions. With 2 percent of records arriving more than four hours late, the
share available within four hours is `1 - 0.0183 = 0.9817` under an exponential delay model, so an
evaluation that requires same-day completeness is measuring the pipeline, not the model.

### 9.5 Cost of the catalogue itself

Each prevention in Section 8 is cheap relative to the failures it prevents, and the ordering matters:

```text
row-count heartbeat                    one query per hour, catches CI-024, CI-021
duplicate-key assertion                one index scan per load, catches CI-021
scale-positive assertion               one line before scoring, catches CI-010, CI-012
tail-rate printout at two cuts         one pass, catches CI-001, CI-002
autocorrelation in the metric job      one pass, catches CI-005
per-block distance share               one pass, catches CI-007
null baseline in the metric template   one line, catches CI-025
```

Seven checks, all sub-minute, cover the highest-severity entries in the catalogue. A team that installs
only these seven has covered the failures that survive longest, which is the point of severity ordering.

## 10. Key Theorems / Results Reference

**Bayes precision bound.** `PPV = pi*r / (pi*r + (1-pi)*alpha)` follows from the definition of conditional
probability and is the single most load-bearing identity in this sheet. Its practical form: with
`pi = 1.667e-4` and `alpha = 1.35e-3`, precision cannot exceed `1/(1 + alpha/pi)` even at perfect recall,
which is `0.110`.

**Glivenko-Cantelli.** The empirical distribution converges uniformly to the true one, so an empirical
quantile is consistent. The convergence is slow in the tail: the standard error of an empirical tail
probability near `p` is about `sqrt(p(1-p)/n)`, so estimating `2.5e-5` to 20 percent relative precision
needs `n` near `1/(0.2^2 * 2.5e-5) = 1e7` rows.

**Concentration of distances (Beyer et al., 1999).** For independent coordinates with bounded variance,
`(max - min)/min` of pairwise distances tends to zero as dimension grows when the coordinate distribution
is not too heavy-tailed. The simulation in 6.2 reproduces the effect: contrast 3.57 at `d = 2`, 0.40 at
`d = 200`, 0.14 at `d = 2000`. The theorem is why nearest-neighbour outlier scores degrade silently rather
than loudly.

**Siegmund's ARL approximation.** `ARL0 ≈ (exp(2kb) + 2kb - 1)/(2k^2)` with `b = h + 1.166` for a
one-sided CUSUM on standardised normal observations. Checked against simulation in 4.3: 962.9 predicted
versus 935.6 simulated at `h = 5, k = 0.5`, and 19,149 versus 19,084 at `h = 8`. The direction of the
error is consistent — the approximation overstates ARL0 slightly at small `h` — so size boundaries with it
and confirm by simulation when `h` is under 5.

**Lorden's optimality for CUSUM.** Among procedures with a given in-control ARL, CUSUM minimises the
worst-case expected delay for a change of known size. The practical consequence is that a windowed
threshold with the same false-alarm rate is slower, so a team replacing CUSUM with a simpler rule must
budget for the larger delay rather than assume equivalence.

**Wilson score interval.** For `x` successes in `n` trials,
`(p + z^2/2n ± z*sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)`. For 85 of 120 reviewed alerts that is
`[0.6216, 0.7822]` around `0.7083`, which is the correct interval for a review sample. The Wald interval
on the same data is narrower and under-covers near the boundaries, which matters because review samples
are usually small.

**Benjamini-Hochberg.** Testing `m` hypotheses at target false discovery rate `q`, reject the largest `k`
with `p_(k) <= k*q/m`. With `m = 2000` and `q = 0.1` the first threshold is `5e-5`, the tenth `5e-4`, and
the twentieth `1e-3`. Any team scanning 2,000 entity metrics at a fixed `1e-3` cut should expect about two
false discoveries per scan and should say so before publishing a list of "anomalous entities".

**Jensen's inequality and the PSI asymmetry.** `KL(p||q)` is not symmetric, so PSI's two directions differ:
in the worked example `KL(p||q) = 0.1433` and `KL(q||p) = 0.1116` bits, while the symmetric
Jensen-Shannon divergence is `0.0305` bits. PSI's own symmetrisation `(q-p)*ln(q/p)` is neither a metric
nor bounded, which is why the 0.1 and 0.25 cut-offs are conventions and must not be described as tests.

**Lloyd's algorithm convergence.** k-means decreases the objective at every step and terminates at a local
minimum, so the result depends on initialisation. That is the formal reason for CI-016's tie-break and for
reporting a distribution of inertias across seeds rather than a single value.

**Expected ARI under independence.** For two independent partitions, `E[ARI] = 0`, by construction of the
chance correction. Observed ARI of 0.3688 in 4.3 against a purity of 0.6944 shows the two indices disagree
by a factor of 1.88: purity counts majority overlap, ARI subtracts the agreement expected from the marginals,
and the marginals here are uneven enough to matter.

**Hoeffding's inequality.** For bounded independent observations, `P(|mean - E| >= t) <= 2exp(-2nt^2)`.
Applied to alert precision it is far too loose at operational rates, the same objection as Chebyshev in
4.4: at `t = 0.01` and `n = 1000` the bound is `2exp(-0.2) = 1.64`, vacuous. Concentration inequalities are
useful for existence arguments, not for threshold sizing.

## 11. Interview / Exam Questions

**Q1. A detector scores 2.4 million units a day with 400 incidents. The team sets a four-sigma cut and
reports 76 alerts. Is the reported precision of 0.75 consistent?**

Answer. Yes, under the assumed alternative. `alpha = 3.1686e-5` gives `2.4e6*3.1686e-5 = 76.05` false
alerts and, at `delta = 4.2`, `400*Phi(0.2) = 231.7` true alerts, so `PPV = 231.7/307.8 = 0.753`. Note
that the alert count is 308, not 76: the false-only count is not the alert count, and a report quoting 76
alerts has silently dropped the true positives.

Reasoning. The error is a denominator error, not a model error. It is worth asking which term was omitted.

Follow-up. What precision would the same cut give if incidents halved to 200 per day? Answer: 0.579,
since true alerts fall to 115.9.

Red flag. Quoting precision without stating the incident count and the scored count.

Scoring. Full credit requires both terms of the sum and the explicit statement that alerts equal false plus
true.

**Q2. Capacity is 60 reviews per day. What recall is achievable, and what does that tell you about model
improvement?**

Answer. Solving `2.4e6*Phi(-z) + 400*Phi(4.2-z) = 60` gives `z* = 5.2385`, recall `0.1495` and 59.8 true
alerts. The recall ceiling from capacity alone is `60/400 = 0.15`, so the model is already delivering
essentially all the capacity allows; a better ranker cannot exceed 0.15 at that capacity.

Reasoning. Capacity enters as a count of total alerts, which caps true alerts at 60. The ceiling is a
property of the operation, not the model.

Follow-up. What capacity would deliver recall 0.51? Answer: 240 alerts per day, `z* = 4.1743`.

Red flag. Reporting recall from an offline evaluation without the capacity ceiling beside it.

Scoring. Credit for the ceiling computation; extra credit for noting that true alerts consume capacity.

**Q3. Two runs of the same pipeline give alert counts 3,594 and 3,588. Where do you look first, and what
number settles it?**

Answer. First check whether any score equals the threshold exactly. With `n = 2.4e6` and a cut near
`z = 3`, the density of scores near the cut is high enough that a handful of exact ties is expected; a
tie-free run and a tie-broken run differ by the size of the tie group. The settling number is the count of
scores within one float32 epsilon of the cut: at magnitude 2^0 the epsilon is `5.96e-8`, and a difference
of six alerts at that density is consistent with tie handling rather than with nondeterminism.

Reasoning. The two candidates are tie order (CI-016) and float accumulation (CI-011). Ties produce a
discrete jump localised at the cut; float noise perturbs scores everywhere, which shows up as a changed
score distribution rather than a changed count.

Follow-up. What assertion prevents recurrence? A hash of the sorted alert set, compared across runs.

Red flag. Re-running until the numbers agree.

Scoring. Credit for naming both mechanisms and for a check that separates them.

**Q4. A dashboard shows zero alerts for three days. Give two mechanisms and the check that separates
them.**

Answer. Either the world is quiet or scoring stopped. The check is the count of scored rows and the median
score per hour: if rows scored is zero or the median is exactly constant, the pipeline is dead (CI-024);
if rows are present and the score distribution is unchanged, the absence is real. A third mechanism is
saturation, where all scores collapse to a bound, which shows as a spike at one value in the histogram.

Reasoning. Absence of alerts is not evidence of absence of anomalies. Only a heartbeat distinguishes the
two.

Follow-up. What is the detection time under each? A dead pipeline is caught in one heartbeat interval, a
saturated score in one distribution comparison.

Red flag. Any answer that starts with the model.

Scoring. Requires a heartbeat or row-count instrument, not a suggestion to "monitor".

**Q5. You are told purity is 0.69 on a segmentation of 180 units into three clusters, against four true
segments. Is that good?**

Answer. The independence baseline is `0.3333`, computed as the expected majority share when cluster and
segment are independent, so the observed 0.6944 is a gain of 0.361 over chance. ARI on the same table is
0.3688, and NMI is 0.3202, so the partition carries real but modest information. Purity alone cannot be
judged, and it rises mechanically with `k`.

Reasoning. Purity is not chance-corrected, so its baseline depends on the marginals. The marginals here
are 50/60/50/20 by segment and 60/60/60 by cluster, which is why the baseline is a third rather than a
quarter.

Follow-up. Which index would you report to compare two candidate `k` values? ARI or NMI, both with the
baseline stated.

Red flag. Comparing 0.69 to a 0.5 rule of thumb.

Scoring. Full credit requires the baseline computation and one chance-corrected index.

**Q6. A colleague reports PSI of 0.18 as "moderate drift" and proposes retraining. What is wrong with that
reading, and what would you compute instead?**

Answer. PSI has no null distribution, so 0.18 has no p-value; with 8 bins and 2000 rows the no-change
expectation is about `7/2000 = 0.0035`, and at 200 rows it is `0.035`. The number 0.18 is therefore 50
times its small-sample expectation and could be entirely noise if the window is short. Compute the
per-bin terms: here the top bin contributes 0.0644 of 0.1767, or 36 percent, so the change is concentrated
in the sparse tail; then compute the effective sample size before calling it drift.

Reasoning. The statistic is a diagnostic, and diagnostics need a reference scale. The per-bin decomposition
is what identifies the mechanism.

Follow-up. What would make the top-bin change an artefact? A bin whose reference share is 0.5 percent,
where a handful of rows moves the term by more than the total of the six middle bins.

Red flag. Quoting the conventional 0.1 and 0.25 cut-offs as if they were tests.

Scoring. Credit for the null expectation and for the per-bin decomposition.

**Q7. An entity with 40 events per day alerts every day. Diagnose.**

Answer. A global `z = 4.0557` cut implies `alpha = 2.5e-5` per test, so 40 tests per day give one false
alert per thousand days, not 30 in a month. The per-entity budget at one false alert per year is
`1/(40*365) = 6.85e-5`, which corresponds to `z = 3.8136` and a count cut of 64.1. If the entity never
exceeds 64 events, the alerts are a rate-model error (CI-006), not anomalies.

Reasoning. The unit of the false-positive rate is the test, and a high-rate entity is tested far more
often. Per-entity calibration is the correction.

Follow-up. What about an entity with 0.2 events per day? Then the Poisson cut is unreachable and a
windowed count over weeks is required.

Red flag. Raising the global threshold until the entity stops alerting.

Scoring. Requires the per-test false-positive arithmetic.

**Q8. A cluster analysis puts 77 percent of the squared distance budget into 40 binary columns and 23
percent into 3 standardised numerics. What will the clusters describe, and how do you fix it?**

Answer. The clusters will describe the binary block, effectively a cross-tabulation of those columns, and
will be insensitive to the numerics. The share is `2*0.25*40 = 20` against `2*3 = 6` under unit scaling,
which is 76.9 percent, and if the numerics are standardised to total variance 0.27 the share rises to 98.7
percent. Fix by block weighting, by using Gower distance, or by reducing the binary block with a
low-variance embedding.

Reasoning. Squared Euclidean distance is a sum over coordinates, so the block with more coordinates and
larger per-coordinate variance dominates. The share is computable before any fit.

Follow-up. Why does standardising the numerics make it worse? Because their total variance falls, while
the binary block's variance per column is fixed at `p(1-p)`.

Red flag. Interpreting the clusters as behavioural segments.

Scoring. Credit for the arithmetic share and for a correction that changes the metric.

**Q9. Eight weeks of a count metric read 389, 398, 405, 412, 415, 421, 430, 268. Why does the low week
disappear, and what is the robust score?**

Answer. On the seven-week baseline the mean is 410 and the standard deviation 13.904, so 268 scores
`-10.21`. Including it, the mean is 392.25 and the standard deviation 51.829, so it scores `-2.40` and
falls inside a three-sigma cut. The MAD route gives median 408.5, `MAD = 11.5`, `sigma_hat = 17.050`, and
`z = -8.24`. This is masking.

Reasoning. One contaminating observation inflates the scale by a factor of 3.7, which is more than the
shift it causes in the location. Robust estimators bound the influence of any single point.

Follow-up. At what contamination share does the MAD break? When more than half the window is affected,
because the median then sits inside the contamination.

Red flag. Removing the point that looks bad and calling it cleaning.

Scoring. Requires both scores and the reason the scale moved.

**Q10. A CUSUM is configured with `h = 5`, `k = 0.5`. What false-alarm rate per stream does that imply,
and what happens at `h = 8`?**

Answer. Siegmund gives `ARL0 = 962.9` at `h = 5`, so roughly one false alarm per 963 observations, or one
per 24 days at hourly observations. At `h = 8` it is 19,149, or about 2.2 years. The cost is detection
delay: for a one-sigma shift the delay grows proportionally, from about 10 observations to about 16.

Reasoning. The pair (h, k) trades false alarms against delay, and both sides are computable before
deployment. `k = 0.5` is tuned for a one-sigma shift; a smaller shift needs a smaller `k`, which raises
ARL0 at fixed `h`.

Follow-up. What does autocorrelation do to these numbers? It invalidates the independence assumption and
inflates the realised false-alarm rate, which is CI-005 applied to the CUSUM.

Red flag. Choosing `h` by convention with no ARL computation.

Scoring. Credit for an ARL number and for the delay trade.

**Q11. A model review reports precision 0.52. An audit of the same alert set reports 0.08. Both are
arithmetically correct. What happened?**

Answer. The denominators differ. Filtered units give 348 alerts with 288 true positives, so `288/348 =
0.8276` at that alert count; the audit's 0.08 uses all 3,588 alerts including 3,240 from units removed by
the quality filter. Both are precision of an alert stream, of different populations. The filter removed
hard negatives, so it raised `pi` in the reviewed population and precision rose through the Bayes identity
with no model change.

Reasoning. Precision is a property of a population and a threshold. The report must name the population,
and the review must compare like with like.

Follow-up. Which number should drive staffing? The unfiltered one, because the queue receives all alerts.

Red flag. Averaging the two figures.

Scoring. Requires the denominator identification and the prevalence mechanism.

**Q12. Why can an order-statistic threshold not express a target rate of `2.5e-5` on 500 units per day?**

Answer. The smallest non-zero empirical tail rate is `1/500 = 0.002`, which is 80 times the target. The
threshold would sit above the maximum with probability 0.99 under the null and produce zero alerts on most
days. Parametric extrapolation or aggregation across days is required, and any claim about `2.5e-5` from
500 rows is a claim about the fitted tail, not about the data.

Reasoning. Order statistics are discrete. Their resolution is `1/n`, and the target must be resolvable to
be meaningful.

Follow-up. How many units are needed? At least `1/2.5e-5 = 40,000` for one expected false alert, and
several times that for a stable estimate.

Red flag. Clipping the quantile to zero and reporting success.

Scoring. Credit for the arithmetic and for the extrapolation caveat.

**Q13. A team adds one evaluation-window point to a 200-point fit set and the alert count rises from 60 to
114. Explain the mechanism and the number.**

Answer. The added point sits 4.2 sigma from the mean, so the sample variance inflates by
`1 + (4.2^2 - 1)/200 = 1 + 16.64/200 = 1.0832` in squared terms, and the standard deviation by
`sqrt(1.0832) = 1.0408` — the computed value with the exact deviation is 1.0392, giving 14.450 against
13.904. A nominal cut of 4.0557 then acts as `4.0557/1.0392 = 3.903` in true units, whose tail is
`4.76e-5`, so alerts become `2.4e6*4.76e-5 = 114.2`.

Reasoning. Leakage inflates the scale, the fixed cut becomes a lower effective z, and the alert count rises
by the ratio of the two tails. One point is enough to nearly double the volume.

Follow-up. What about a 2,000-point fit set? The inflation falls to `sqrt(1+16.64/2000) = 1.0041`, and
the effect becomes unmeasurable — which is why leakage hides in large fits and bites in small ones.

Red flag. Calling the rise a sensitivity improvement.

Scoring. Full credit for the effective-z route.

**Q14. You must choose between a threshold at 3 sigma, at the capacity point, and at the value optimum.
Give the decision rule.**

Answer. Compute both constraints before choosing. Capacity 60 gives `z* = 5.2385`, recall 0.1495 and
$25,118 net per day. The value optimum at `z = 3.18` gives 2,106 alerts and $130,662 net, but needs 26.3
reviewer FTE. Three sigma gives 3,594 alerts, $127,609 net and 44.9 FTE, so it is near the value optimum
but wasteful of staff through 90 percent false alerts. The rule: if staff cannot grow, the capacity point
is the only feasible one and the honest statement is "recall capped at 0.15"; if staff can grow, the value
optimum sets the target and staffing follows it.

Reasoning. Two independent constraints, and the answer is whichever binds. Naming both, with numbers, is
the deliverable.

Follow-up. What changes if a missed incident costs ten times the $420 used here? The optimum moves left,
toward three sigma or lower, and capacity becomes the binding constraint sooner.

Red flag. Choosing by convention.

Scoring. Requires at least two computed operating points and an explicit feasibility statement.

**Q15. What single number would you put on a dashboard to catch the most entries in this catalogue?**

Answer. Alerts per 100,000 scored units, printed alongside the count of scored rows. It catches CI-020
(volume change with a fixed count budget), CI-021 (duplication raising the denominator), CI-024 (rows
falling to zero), and it makes CI-027 visible because a filter change moves the denominator. Rate and
denominator together carry more diagnostic content than any model metric, because every failure in this
sheet changes one of them before it changes a score.

Reasoning. The catalogue is dominated by population and pipeline defects, and both are visible in the
pair (rows scored, alerts per unit). A score metric cannot see either.

Follow-up. What second number? The median score, which separates a threshold change from a population
change in one comparison.

Red flag. A model-quality metric as the primary monitor.

Scoring. Credit for a rate with a denominator, plus a second diagnostic.

## 12. Quick Reference Card

| If you see | Suspect first | Run this | Then |
|---|---|---|---|
| alert volume jumped | population or duplication | scored rows per hour, then rate per 100k | CI-020 or CI-021 |
| no alerts at all | dead pipeline or saturation | scored-row heartbeat, score histogram | CI-024 or CI-011 |
| precision changed with no model change | denominator or filter | recompute on all units | CI-027 |
| one entity alerts daily | rate heterogeneity | per-entity Poisson cut, e.g. 64.1 at rate 40 | CI-006 |
| score of a known outlier looks mild | masking or contamination | MAD scale, e.g. 17.050 vs sd 51.829 | CI-002, CI-010 |
| nominal four sigma yields 11,725 alerts | heavy tail | empirical tail versus 3.17e-5 | CI-001 |
| threshold cannot hit the target rate | order-statistic floor | `1/n` versus `alpha`, e.g. 0.002 at n=500 | CI-015 |
| segment names reshuffle | label swap | Jaccard on co-clustered pairs, 0.695 means stable | CI-026 |
| purity looks high | chance baseline | independence baseline, e.g. 0.333 versus 0.694 | CI-025 |
| queue never clears | arrival above service rate | backlog hours, e.g. 3,594 alerts at 40/h is 89.8h | CI-022 |
