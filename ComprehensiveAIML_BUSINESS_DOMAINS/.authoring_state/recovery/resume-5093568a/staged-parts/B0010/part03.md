### 6.3 Derivation CC-03 — Span precision, recall, and the inflation factor

The metric that actually governs publication is not word error but span-level correctness: did the
system mark the right interval as an objection, and did it mark any interval that is not one?

```text
FORMULA [CC-03]: P = TP / (TP + FP);  R = TP / (TP + FN);  F1 = 2TP / (2TP + FP + FN)
WHERE: TP spans match a gold span by the declared match rule; FP spans have no gold
       counterpart; FN gold spans have no model counterpart. Match rule: temporal
       intersection-over-union >= 0.5 AND the same rubric category.
CONSTRAINT: one-to-one matching (a model span may satisfy at most one gold span);
       spans overlapping a routed call are excluded from both numerator and denominator
       so that routing does not flatter precision.
NUMERICAL EXAMPLE: over 600 calls, gold holds 412 objection spans; the pipeline proposes
       468. Matching gives TP = 287, FP = 181, FN = 125. Check: TP+FP = 468 (proposed),
       TP+FN = 412 (gold). P = 287/468 = 0.6132; R = 287/412 = 0.6966;
       F1 = 2(287)/(2(287)+181+125) = 574/880 = 0.6523.
INTERPRETATION: A proposed-span count of 468 overstates gold by 468/412 − 1 = 13.59%,
       while at the same time missing three gold spans in ten. The error is not a
       scaling problem; it is a two-sided problem.
```

The count is 13.59% too high and simultaneously 30.34% incomplete on the gold side — two biases in
opposite directions that a total-count comparison partially cancels. This is the most dangerous
configuration in the whole case: a dashboard that plots "objections per day" from the raw proposed
count will look *approximately right* while being composed of the wrong spans. The cancellation is
coincidental, and it evaporates the moment the rubric changes the phrase list.

**Per-100-calls view.** Gold density is 412/600 = 0.687 spans per call; proposed density is
468/600 = 0.780 spans per call. Over a 900-call day that is 618.3 gold versus 702.0 proposed — a
gap of 83.7 spans per day, composed of roughly 27 missing spans and 111 spurious ones if the daily
mix matches the sample. The two errors do not net out in the category breakdown, which is what the
report is actually for.

**Where the errors concentrate.**

| Error class | Count | Share of FP or FN | Characteristic |
|---|---|---|---|
| Agent script phrase matched as customer objection | 64 | 35.4% of FP | Script line reads "I understand your concern about the charge" |
| Negation missed | 41 | 22.7% of FP | "I don't have a problem with that" matched on the phrase "problem with" |
| Fragment split into two spans | 38 | 21.0% of FP | One objection interrupted by an agent backchannel becomes two |
| Weak or partial speech | 29 | 16.0% of FP | Boundary placed inside the objection, IoU falls below 0.5 |
| Boundary truncation | 9 | 4.9% of FP | Span starts after the objection's first clause |
| Objection inside a routed call | 96 | 76.8% of FN | Counted as routed, so excluded from the TP set by the matching rule |
| Objection phrased outside the rubric | 22 | 17.6% of FN | Real objection, no rubric phrase, no candidate generated |
| Objection split across a transfer | 7 | 5.6% of FN | Conversation continues on a new leg with a new call leg id |

This table is the design input: 35.4% of false positives come from ignoring speaker role, which is
a gate problem, not a model problem. Fixing role attribution removes the largest single FP class
for the cost of a metadata join.

**Boundary where CC-03 breaks.** The IoU ≥ 0.5 rule is a convention, not a property. It fails when
gold spans are much longer than model spans: a 6-second gold objection matched by a correct 2-second
core has IoU = 1/3 and scores as both a FP and a FN, double-penalising one honest detection. It
also fails for nested or overlapping objections — a customer who raises a charge objection and then
a service objection in one breath produces overlapping gold spans that a one-to-one rule cannot
represent. The check is to report the span-length distribution alongside P and R; if model spans
are systematically shorter than gold, IoU is measuring boundary policy rather than detection, and
boundary-tolerant matching (token-level overlap on the objection's head phrase) must be used
instead, with both numbers reported.

### 6.4 Derivation CC-04 — Calibration, and why an untested confidence cut is a guess

A confidence score is only usable for routing if it is calibrated: among spans the model scores at
0.94, about 94% should actually be correct. Uncalibrated confidence produces a threshold that looks
principled and behaves arbitrarily.

```text
FORMULA [CC-04]: ECE = Σ_b (n_b / n) · | mean(conf_b) − acc_b |
WHERE: b indexes confidence bins; n_b spans in bin b; n = Σ n_b;
       mean(conf_b) mean predicted probability in bin b; acc_b observed accuracy in bin b.
CONSTRAINT: bins declared before inspection; each bin needs enough spans for acc_b to be
       stable; accuracy measured on held-out data, not the data used to fit the calibrator.
NUMERICAL EXAMPLE: 1,200 candidate spans in ten bins. Predicted minus observed by bin:
       +0.031, +0.018, +0.014, +0.002, +0.003, −0.001, +0.075, +0.065, +0.040, +0.003.
       Weighted by bin share the absolute gaps give ECE = 0.0267 = 2.67 percentage points.
INTERPRETATION: Small overall, but the error is concentrated: the three highest-confidence
       bins over-predict by 4.0 to 7.5 points, exactly where a high threshold is applied.
```

**The bin table, closed and recomputed (ILLUSTRATIVE).**

| Bin | Spans `n_b` | Mean predicted | Observed accuracy | Gap (pred − obs) |
|---|---|---|---|---|
| 0.52–0.56 | 36 | 0.556 | 0.525 | +0.031 |
| 0.56–0.60 | 54 | 0.593 | 0.575 | +0.018 |
| 0.60–0.64 | 72 | 0.639 | 0.625 | +0.014 |
| 0.64–0.68 | 96 | 0.677 | 0.675 | +0.002 |
| 0.68–0.74 | 114 | 0.728 | 0.725 | +0.003 |
| 0.74–0.80 | 84 | 0.774 | 0.775 | −0.001 |
| 0.80–0.90 | 138 | 0.900 | 0.825 | +0.075 |
| 0.90–0.94 | 156 | 0.940 | 0.875 | +0.065 |
| 0.94–0.97 | 174 | 0.965 | 0.925 | +0.040 |
| 0.97–1.00 | 276 | 0.978 | 0.975 | +0.003 |
| **Total** | **1,200** | **0.8498** | **0.8233** | **+0.0265** |

Bin counts sum to 1,200 as required. Overall predicted is 1,019.8/1,200 = 0.8498 and overall
observed is 987.9/1,200 = 0.8233, so the aggregate over-prediction is 2.65 points, consistent with
the ECE of 2.67 computed from absolute gaps. The apparent inconsistency in the middle of the table —
a bin at 0.90 predicted scoring only 0.825 — is the whole point: the model's confidence saturates
after about 0.90 and stops carrying information.

**Threshold selection, computed rather than asserted.** Three candidate cuts:

| Cut on `p` | Spans accepted | Coverage | Empirical precision | Spans routed |
|---|---|---|---|---|
| ≥ 0.85 | 744 | 62.0% | 680.5/744 = 91.45% | 456 |
| ≥ 0.94 | 606 | 50.5% | 566.6/606 = 93.49% | 594 |
| ≥ 0.96 | 450 | 37.5% | 430.1/450 = 95.57% | 750 |

Moving the cut from 0.85 to 0.96 buys 4.1 points of precision and costs 24.5 points of coverage —
172 spans per 1,200, which at 702 candidates per day is 100.6 spans, or 180.3 staffed minutes of
extra review per day. That trade is a staffing decision disguised as a modelling decision, which is
why §6.6 computes it in minutes rather than in accuracy points.

**The asymmetry that decides the cut.** A false positive enters a published count as a fact and is
never revisited. A false negative is a span that a reviewer never sees, and its absence is invisible
in the data. Both are bad, but only the false positive is *published*; the negative merely
undercounts. That asymmetry argues for the higher cut, and the capacity model decides how high it
can go.

**Boundary where CC-04 breaks.** ECE is a marginal, bin-averaged quantity and says nothing about
conditional calibration. A model can have ECE of 2.7 points overall while being badly
over-confident on one channel — say legacy headsets, where the acoustic evidence is weakest — and
that is the stratum where the threshold is applied most often. The check is to compute ECE per
channel, per language, and per queue. A second boundary: the observed accuracy in each bin is
itself estimated from a finite number of spans, so a bin with 36 spans has a standard error of
about `sqrt(0.525×0.475/36) = 0.083` — 8.3 points — larger than the gap it is used to measure. Bins
below roughly 50 spans cannot support a calibration claim, and the two lowest bins in this table are
therefore reported to show the shape, not to justify a decision.

### 6.5 Derivation CC-05 — Staffing: the daily average says feasible, the peak hour says no

The naive capacity calculation multiplies arrivals by service time and compares with staffed
minutes. It is correct arithmetic applied to the wrong moment.

```text
FORMULA [CC-05, daily average]: W = A · r · s ;  U = W / (k · h · 60)
WHERE: A eligible candidate spans per day (spans, not calls — only candidates are routed);
       r routed share of candidates (dimensionless); s effective staffed minutes per
       routed span (minutes); k reviewers; h productive staffed hours per reviewer per day.
CONSTRAINT: stationary arrivals and service times; no rework, breaks, or escalations;
       excludes queueing delay entirely — this is a load test, not a service test.
NUMERICAL EXAMPLE: A = 900 calls × 0.78 candidates/call = 702 candidates/day. At
       r = 0.38, 266.8 spans route. Nominal handle time 1.6 min, rework factor 1.12
       (12% of dispositions reopened), so s = 1.6 × 1.12 = 1.792 min.
       W = 702 × 0.38 × 1.792 = 478.0 staffed minutes. With k = 2 reviewers at
       h = 7.0 productive hours, capacity = 840 min, so U = 478.0/840 = 56.91%.
INTERPRETATION: 57% average utilisation reads as comfortable headroom. It is not a
       service guarantee, because a queue's delay is governed by the peak hour.
```

The rework factor deserves its own justification because it is the most commonly omitted term. A
reviewer who opens a span, cannot resolve it, and reopens it after consulting the rubric spends
the time twice. If 12% of dispositions are reopened at full cost, effective service time is 12%
above nominal. Omit it and the plan understates load by 10.7% (`1.792/1.6 = 1.12`), which is
invisible on a 57% utilisation and fatal on a 90% one.

**Now the same day at the peak hour.** Arrivals are not uniform. In this illustration 900 legs
over an 8-hour service window is a mean of 112.5 legs/hour, and the busiest hour runs 1.32 times
the mean — 148.5 legs/hour — a modest peaking factor that any contact centre will recognise as
conservative.

```text
FORMULA [CC-05b]: Erlang-C delay probability for c reviewers and offered load a,
       a = λ · s / 60  (λ routed spans per hour, s minutes per span, so a is dimensionless)
       C(c, a) = [ a^c / (c!(1 − a/c)) ] / [ Σ_{k=0}^{c−1} a^k/k! + a^c/(c!(1 − a/c)) ]
       W_q = C(c, a) / (c·μ − λ),  μ = 1/s (spans per minute per reviewer)
       P(wait > t) = C(c, a) · exp(−(cμ − λ) t)
CONSTRAINT: Poisson arrivals, exponential service, no abandonment, no priority classes,
       no batching. All four are violated in reality to some degree; results are
       directional, not contractual.
NUMERICAL EXAMPLE: at the peak, candidates = 148.5 × 0.78 = 115.83 spans/hour. At
       r = 0.38, λ = 44.02 spans/hour = 0.7336/min. With s = 1.792 min, μ = 0.5580/min,
       a = 0.7336/0.5580 = 1.3146, so a/c = 0.6573 with c = 2 and C = 0.5214.
       W_q = 0.5214 / (2(0.5580) − 0.7336) = 0.5214 / 0.3824 = 1.363 min.
       P(wait > 10 min) = 0.5214 · exp(−0.3824 × 10) = 0.5214 × 0.02196 = 1.14%.
INTERPRETATION: The same policy that shows 57% daily utilisation puts 52% of peak-hour
       spans in a queue, with a 1.36-minute mean wait and about 1.1% waiting over ten
       minutes — roughly 0.5 spans in the peak hour.
```

**The sensitivity table that decides the policy.** Recomputed at each routing share, two reviewers:

| Routed share `r` | Offered load `a` | Utilisation `a/c` | Delay prob. `C` | Mean wait | P(wait > 10 min) |
|---|---|---|---|---|---|
| 0.20 | 0.692 | 0.346 | 0.178 | 0.24 min | 0.01% |
| 0.30 | 1.038 | 0.519 | 0.355 | 0.66 min | 0.17% |
| 0.35 | 1.211 | 0.605 | 0.457 | 1.04 min | 0.56% |
| 0.38 | 1.315 | 0.657 | 0.521 | 1.36 min | 1.14% |
| 0.40 | 1.384 | 0.692 | 0.566 | 1.65 min | 1.82% |
| 0.45 | 1.557 | 0.778 | 0.681 | 2.76 min | 5.74% |
| 0.50 | 1.730 | 0.865 | 0.802 | 5.32 min | 17.75% |
| 0.55 | 1.903 | 0.951 | 0.928 | 17.09 min | 53.90% |

The shape is the whole lesson: from `r = 0.30` to `r = 0.45` the routed volume rises by half, but
the ten-minute tail rises by a factor of 34 (0.17% to 5.74%). Beyond `r = 0.50` the queue is not
slow, it is unstable in practice, because the exponential-service assumption is the optimistic case
and real handle times are heavier-tailed. At `r = 0.60` the offered load reaches 2.076 with two
reviewers — strictly infeasible, since `a/c > 1` means arrivals exceed service capacity and the
queue grows without bound.

**Three reviewers change the picture, at a price.** At `r = 0.60` with `c = 3`: `a = 2.076`,
`a/c = 0.692`, `C = 0.4805`, `W_q = 0.932 min`, `P(wait > 10) = 0.28%`. So the third reviewer
supports a 60% routing share with a better tail than two reviewers deliver at 38%. The comparison
that matters is not "2 versus 3 reviewers" but "38% routing with 2" versus "60% routing with 3":
the third reviewer buys 22 points of additional coverage — 154 more spans per day at 702
candidates — for 420 additional staffed minutes per day.

**Where the daily-average model actually fails.** It fails by saying yes. At `r = 0.55` the
daily average gives `W = 702 × 0.55 × 1.792 = 691.9` minutes against 840 capacity, i.e. 82.37%
utilisation — apparently fine — while the peak-hour tail is 53.9%. A staffing plan built on the
daily figure would approve a policy under which half the peak-hour spans wait more than ten
minutes; since the review queue is not the customer-facing queue, nobody notices until the
backlog appears three days later.

**Boundary where CC-05 breaks.** Erlang-C assumes Poisson arrivals and exponential service. Three
violations matter here and each pushes the true delay higher than computed. First, arrivals are not
Poisson at the peak: calls cluster after an outage or a campaign drop, producing bursts the model
smooths away. Second, handle time is not exponential but right-skewed — a long, contested call can
consume twenty minutes — and the model's mean wait is optimistic when the tail is heavy. Third, no
abandonment: a real reviewer who finds a twenty-minute queue stops working the queue and does
something else, which removes capacity exactly when it is needed. The practical consequence is to
treat the computed tail as a *lower bound* and to add a hard cap on routed volume per hour rather
than relying on the model's probability.

### 6.6 Derivation CC-06 — Publishing an aggregate with an audit-estimated numerator

The aggregate has two components: spans a reviewer confirmed, and spans the model accepted that no
reviewer saw. The second component is estimated from a sample, so the published figure carries
sampling error that the first component does not.

```text
FORMULA [CC-06]: N_hat = N_confirmed + N_auto · p_hat
                 with p_hat from an audit of n accepted spans: p_hat = x/n,
                 Wilson interval [ (p̂ + z²/2n ± z√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n) ]
CONSTRAINT: the audit sample is drawn from the accepted frame with known inclusion
       probabilities; accepted spans are exchangeable with unaudited accepted spans;
       the confirmed count is treated as exact (it was reviewed, not sampled).
NUMERICAL EXAMPLE: 702 candidates/day; 266.8 routed and reviewed, of which 178 confirm
       (66.67% of routed); 435.2 accepted without review. An audit of n = 150 accepted
       spans finds x = 141 correct, p̂ = 0.94, Wilson 95% interval [0.8899, 0.9681].
       N_hat = 178 + 435.2 × 0.94 = 178 + 409.1 = 587.1 spans/day.
       Interval: 178 + 435.2×0.8899 = 565.3 to 178 + 435.2×0.9681 = 599.3.
       Per 900 calls: 587.1/900 = 65.23 per 100 calls, interval 62.81 to 66.59.
INTERPRETATION: The interval is ±1.9 per 100 calls, which is tight enough to compare
       months. The dominant error is not sampling; it is the 6% accepted-error rate
       applied to the 62% of candidates that no reviewer saw (435.2 of 702).
```

Recomputing the Wilson interval by hand, so the numbers are checkable:
`z = 1.96`, `z² = 3.8416`, `p̂ = 0.94`, `n = 150`. Centre term
`(0.94 + 3.8416/300)/(1 + 3.8416/150) = (0.94 + 0.012805)/(1.025611) = 0.952805/1.025611 = 0.92902`.
Half-width term `1.96 × sqrt(0.94×0.06/150 + 3.8416/90000) = 1.96 × sqrt(0.000376 + 0.00004268) =
1.96 × sqrt(0.00041868) = 1.96 × 0.020462 = 0.040105`; divided by `1.025611` gives `0.039103`. So
the interval is `0.92902 ± 0.03910 = [0.88992, 0.96812]`, matching the reported `[0.8899, 0.9681]`.

**Why not the normal approximation.** At `p̂ = 0.94` and `n = 150` the textbook interval
`0.94 ± 1.96·sqrt(0.94·0.06/150) = 0.94 ± 0.03803` gives `[0.9020, 0.9780]`, whose upper end
exceeds the plausible range and whose lower end is 1.2 points optimistic versus Wilson. The
asymmetry is real: with 9 failures in 150, the tail above 0.97 is thinner than the tail below 0.90,
and the Wilson form captures that. The practical effect on the published figure is ±1.2 per 100
calls, which is most of the month-over-month signal the report is meant to carry.

**How large should the audit be?** For a ±3-point margin at 95% confidence on `p ≈ 0.94`:
`n₀ = z²p(1−p)/e² = 3.8416 × 0.94 × 0.06 / 0.0009 = 0.216666/0.0009 = 240.7`. With a finite
population correction against an accepted frame of 435.2 spans/day,
`n = n₀ / (1 + (n₀−1)/N) = 240.7/(1 + 239.7/435.2) = 240.7/1.5508 = 155.2`. So 156 spans per day —
34.5% of the accepted frame — buys ±3 points, and the illustration's 150 is within one span of that
requirement, giving an actual margin of 3.08 points with the correction
(`1.96 × sqrt(0.94×0.06/150) × sqrt((435.2−150)/(435.2−1)) = 1.96 × 0.019391 × 0.8105 = 0.0308`).

**The denominator trap, quantified.** 837 of 900 calls are processed (93.0%), the rest excluded for
consent, capture, language, or integrity reasons. Reporting per *processed* call gives
`587.1/837 = 70.14` per 100 calls rather than `587.1/900 = 65.23` — a 7.53% inflation, produced
entirely by the choice of denominator. Neither number is wrong; publishing one while describing the
other is. The rule is that the denominator must be named in the same sentence as the rate, and the
exclusion count must be visible in the same table.

**Break-even on accepting a span unreviewed.** A reviewer costs `1.792 min × £46/hour / 60 = £1.3739`
per span reviewed. Accepting a span unreviewed saves that cost and admits a 6% error rate. The
break-even cost of one published error is therefore `£1.3739 / 0.06 = £22.90`. If a spurious
objection in the published count costs the organisation less than £22.90 in remediation — a
re-check, a correction, an apology for a miscategorised trend — then accepting at `p = 0.94` is
economically correct. If it costs more, the threshold must move. That is a finance question the
design should force into the open rather than decide implicitly through an arbitrary 0.90 cut.

**Boundary where CC-06 breaks.** The estimator is unbiased only if the audited accepted spans are
exchangeable with the unaudited ones. Three realistic violations: (a) the audit is drawn from the
same day it reports, so late-arriving accepted spans are never auditable — a stale-audit bias;
(b) the audit is drawn after a model update, so `p̂` describes a different release than the spans it
is applied to — a version-mixing bias; (c) reviewers of routed spans have already seen neighbouring
spans from the same call, so the confirmed count is not independent of the accepted count when
multiple spans come from one call — a clustering effect that shrinks the effective sample size. The
check for (c) is to compute the design effect from the intra-call correlation of correctness; if
spans cluster by call, `n = 150` spans may carry the information of far fewer independent
observations.
