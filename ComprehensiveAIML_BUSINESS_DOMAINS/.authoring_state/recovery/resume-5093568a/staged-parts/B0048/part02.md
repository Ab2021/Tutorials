## 4. Core Formulas with Full Derivation

Every result below is DERIVED from the assumptions in the block and from the running example's
eight inputs. Each block follows the same anatomy: FORMULA, WHERE (symbols with units), CONSTRAINT
(when the formula is valid), NUMERICAL EXAMPLE (recomputed by hand, ILLUSTRATIVE), INTERPRETATION
(what a practitioner does with it). The section ends with 4.4, which is where each formula is
pushed to the point where it breaks.

A note on why derivation rather than recall matters here. The cutoff that most programmes deploy
is inherited — it is the number in the previous vendor's documentation, or the 0.5 default of a
classifier, or the threshold that produced an acceptable alert volume in a pilot month. Each of
those is a number without its CONSTRAINT, and the constraint is where the money is. A cutoff of
0.5 is exactly correct when c = 0.5 x V; it is wrong by a factor of ten when V is large and c is
small, which is the normal situation in card fraud where a single caught fraud is worth tens of
reviews.

### 4.1 Base case

**B-1 Confusion-matrix rates.**

Derivation: the decision partitions the cohort twice, once by truth and once by action. Four
intersections result. Conditional on a selected event, the event is either fraudulent or not, so
the two counts TP and FP partition the selected set and their ratio sums to one. The same argument
applies to the truth-partition. No distributional assumption is needed; this is bookkeeping.

```text
FORMULA: precision = TP/(TP+FP);  recall = TP/(TP+FN);  FPR = FP/(FP+TN)
WHERE: TP,FP,FN,TN are event counts per cycle (events); no units cancel, all dimensionless
CONSTRAINT: one cohort, one action rule, one definition of Y, labels final for all four cells
NUMERICAL EXAMPLE: TP=2510, FP=21490 at K=24000, FN=2290 (of 4800 positives)
                   precision = 2510/24000 = 0.104583
                   recall    = 2510/4800  = 0.522917
                   FPR       = 21490/3995200 = 0.0053790
INTERPRETATION: about 10.5 per cent of reviews find fraud; just over half of frauds are reviewed
```

Two consistency checks that catch most reporting errors. First, precision x K must equal TP, and
recall x P must equal TP; if both identities hold, the four cells are mutually consistent and the
denominators were not swapped. Second, FP + TN must equal the negative count, 3,995,200 here. A
report where TN is quoted as the full cohort has almost certainly absorbed the positives.

**B-2 Prevalence conversion (the base-rate identity).**

Derivation: by the law of total probability applied to the selected set, the selected mass is
pi x TPR + (1-pi) x FPR. The fraudulent part of that mass is pi x TPR. Predictive value is the
ratio, because conditional probability is a ratio of joint to marginal mass. This is the same
identity as Bayes' rule written in rates rather than densities, which is why it needs no
distributional assumption, only that the class-conditional rates apply to the target population.

```text
FORMULA: PPV = (pi*TPR) / (pi*TPR + (1-pi)*FPR)
WHERE: pi prevalence (dimensionless), TPR and FPR are target-population conditional rates,
       PPV is the precision a programme will observe at that operating point
CONSTRAINT: TPR and FPR measured on the same population and period as the deployment
NUMERICAL EXAMPLE: pi=0.0012, TPR=0.522917, FPR=0.0053790
                   numerator   = 0.0012*0.522917 = 0.00062750
                   denominator = 0.00062750 + 0.9988*0.0053790 = 0.00062750+0.00537253
                               = 0.00600003
                   PPV = 0.00062750/0.00600003 = 0.104583
INTERPRETATION: reproduces B-1 exactly, so the rate form and the count form are consistent
```

The identity is the most useful single formula in the sheet because it lets a programme convert a
rate measured anywhere into the precision expected anywhere else, *provided* the rates transfer.
Sensitivity: holding rates fixed, PPV at pi = 0.0006 is 0.055145 and at pi = 0.0024 is 0.189546.
Roughly doubling prevalence slightly more than doubles precision here, because the negative term still
dominates at these small values; the relation becomes strongly non-linear as pi grows, reaching
0.495450 at pi = 0.01 and 0.894218 at pi = 0.08. A programme that halves fraud volume through
non-model means will see precision fall by nearly half and will be told the model degraded.

**B-3 Expected net value of one action.**

Derivation: condition on the two states. With no action, expected loss is p x L. With action, loss
is (1-r) x p x L plus the action's own cost and harm. The difference is p x L x r - c - h. Linearity
of expectation makes the per-event formula additive across events, which is what licenses B-5.

```text
FORMULA: Delta_i = p_i*L_i*r_i - c_i - h_i
WHERE: p_i dimensionless probability of fraud under no action (the counterfactual),
       L_i cur of loss conditional on fraud, r_i preventable share in [0,1],
       c_i cur direct action cost, h_i cur friction/harm, Delta_i cur per event
CONSTRAINT: additive separable harms; r independent of the action being taken; single action type
NUMERICAL EXAMPLE: p=0.104583, L=340, r=0.55, c=8.20, h=1.30
                   gross preventable = 0.104583*340*0.55 = 0.104583*187 = 19.557 cur
                   net = 19.557 - 8.20 - 1.30 = 10.057 cur per reviewed alert
INTERPRETATION: at the operating point each review is worth about 10 cur net; 24,000 of them
                is 241,370 cur, matching the tier arithmetic in section 5
```

The subtlety is p_i. If p_i is taken as the observed fraud rate among reviewed events, it is a
*post-action* quantity and the formula silently assumes that review had no effect on which events
were fraudulent. For block actions that is badly wrong: a blocked transaction cannot complete, so
the conditional loss given block is not the conditional loss given no block. Use the no-action
risk, estimated from pre-decision features, and keep r as the separate parameter that carries the
action's efficacy. That separation is what makes the next formula legitimate.

### 4.2 General case

**G-1 Unconstrained economic cutoff.**

Derivation: review when Delta > 0. Substitute B-3 and divide both sides by the positive constant
L x r. The inequality survives division, so the decision reduces to a comparison of a probability
against a cost-to-value ratio. Ties (Delta = 0) are indifferent, which matters operationally
because it means the cutoff can move inside a tie band without changing expected value.

```text
FORMULA: review iff p_i > (c_i + h_i) / (L_i * r_i)
WHERE: p_i dimensionless; numerator and denominator both cur per event, so the ratio is a
       probability; the ratio is the "cost of looking divided by the value of finding"
CONSTRAINT: L, r, c, h constant across events; no capacity limit; actions independent
NUMERICAL EXAMPLE: c+h = 8.20+1.30 = 9.50; L*r = 340*0.55 = 187.00
                   cutoff = 9.50/187.00 = 0.050802
INTERPRETATION: review anything with at least a 5.08 per cent chance of fraud
```

That 5.08 per cent is the number a programme should be able to state out loud, with its four
inputs. It is also the number that moves most when the programme's cost model is wrong. At
L = 250 the cutoff is 0.069091; at L = 480 it is 0.035985. At r = 0.45 it is 0.062092 and at
r = 0.65 it is 0.042986. A programme that has never estimated r for its own review process is
choosing a cutoff by accident.

**G-2 Capacity-constrained allocation.**

Derivation: with capacity K the problem becomes a 0-1 linear programme with one knapsack
constraint. Because every event consumes exactly one unit of capacity, the constraint matrix is a
single row of ones, which is totally unimodular; the LP relaxation has an integral optimum, and the
optimum is obtained by the greedy rule that sorts by Delta and takes the top K. The exchange
argument: if a selected event has lower Delta than an unselected event, swapping them cannot
reduce the objective and cannot violate the constraint. Hence greedy is optimal, not merely good.

```text
FORMULA: maximize sum_i Delta_i*x_i  s.t.  sum_i x_i <= K,  x_i in {0,1}
WHERE: Delta_i cur per event; K actions per cycle; x_i is the action indicator
CONSTRAINT: equal capacity consumption per action; additive benefits; Delta known, not estimated
NUMERICAL EXAMPLE: K=24000, Delta ordered by tier; the 24,000th alert sits in tier 6
                   (block 6 contributes 268 of its 4,000 positives)
                   TP at K = 620+505+430+372+315+268 = 2510
                   value = 2510*187.00 - 24000*9.50 = 469370 - 228000 = 241370 cur
INTERPRETATION: greedy by expected value equals greedy by risk only when Delta is monotone in p
```

The last line is the operative caveat. Delta is monotone in p only if L, r, c, and h are equal
across events. In real portfolios L varies by an order of magnitude between a 12-cur digital-goods
purchase and a 4,000-cur electronics order, and c varies with the review path. When L varies,
sorting by p is not optimal, and the loss from sorting by p can be computed: the programme gives up
the high-L events just below the probability cutoff in exchange for low-L events just above it.
Section 7 gives the selection rule, and 4.4 quantifies a case where the two orderings disagree.

**G-3 Review-duration-aware allocation.**

Derivation: if action i consumes d_i hours rather than one interchangeable slot, the constraint
becomes sum d_i x_i <= H. The constraint matrix is no longer a row of ones, total unimodularity is
lost, and greedy by value-density is a heuristic whose optimality gap is bounded by the largest
single item. The practical response is to keep the greedy rule but sort by Delta_i/d_i and accept a
bounded loss, or to solve the integer programme exactly when the candidate set is small.

```text
FORMULA: maximize sum Delta_i*x_i  s.t.  sum d_i*x_i <= H
WHERE: d_i investigator-hours per event; H total productive hours per cycle
CONSTRAINT: durations known before the decision; no preemption; no shared setup across cases
NUMERICAL EXAMPLE: two candidate alerts with equal probability
                   A: Delta=70 cur, d=0.6 h -> density 116.67 cur/h
                   B: Delta=100 cur, d=1.0 h -> density 100.00 cur/h
                   H=1.0 h: greedy by density takes A (70 cur); the optimum takes B (100 cur)
INTERPRETATION: density ordering loses 30 cur on a 100-cur opportunity, i.e. 30 per cent
```

The counterexample is deliberately small but it is the general failure: density heuristics are
exact for fractional knapsack and approximate for 0-1 knapsack. When H is large relative to the
items, the gap is negligible; when H is one shift and items are 45 minutes each, it is not. The
fix in practice is not to solve an integer programme but to bucket durations into two or three
classes (fraud-ring case, standard case, quick check) and run the greedy separately within each
class, which restores equal-consumption structure per class.

**G-4 Calibration-adjusted prior shift.**

Derivation: posterior odds equal the likelihood ratio times prior odds. Under the assumption that
the class-conditional feature distribution does not change, the likelihood ratio for a given score
is identical in source and target; only the prior odds differ. Dividing the source odds by source
prior odds and multiplying by target prior odds therefore yields the target odds. This is an
algebraic identity under the assumption, not an approximation, and it fails exactly when the
assumption fails.

```text
FORMULA: odds_t = odds_s * [pi_t/(1-pi_t)] / [pi_s/(1-pi_s)]
WHERE: odds = p/(1-p); pi_s source prevalence; pi_t target prevalence; both in (0,1)
CONSTRAINT: pure label-prior shift; no class-conditional drift in features or in the score
NUMERICAL EXAMPLE: model trained on an 8 per cent prevalence sample; for one event the
                   sample posterior is 0.62, so source odds = 0.62/0.38 = 1.6316
                   prior odds ratio = (0.0012/0.9988)/(0.08/0.92) = 0.012014/0.086957 = 0.13817
                   target odds = 1.6316*0.13817 = 0.22546; p_t = 0.22546/1.22546 = 0.18397
INTERPRETATION: a 0.62 sample posterior is a 0.18 deployment probability for the same evidence
```

The likelihood ratio implied by that arithmetic is 18.7632, which is a useful diagnostic in its own
right: it says the evidence in this event is worth a 19-fold shift in odds. A model whose maximum
achievable likelihood ratio across the whole score range is 15 cannot produce a deployment
probability above roughly 0.0177 at this prevalence, whatever its AUC. That is the quantitative
statement behind "the model needs more signal, not a better threshold".

**G-5 Inverse-probability estimate of unreviewed fraud.**

Derivation: each target event has an inclusion indicator R_i with expectation q_i under the audit
design. Because R_i is independent of Y_i given the stratum (that is what a well-run audit
guarantees), E[R_i Y_i / q_i] = E[R_i] E[Y_i]/q_i = Y_i. Summing over the cohort gives an unbiased
estimator of the total. Variance follows from the Bernoulli variance of R_i scaled by 1/q_i^2.

```text
FORMULA: T_hat = sum_i R_i*Y_i/q_i ;  Var(T_hat) = sum_i Y_i*(1-q_i)/q_i
WHERE: R_i audit inclusion indicator; q_i inclusion probability in (0,1]; Y_i binary outcome
CONSTRAINT: q_i known and strictly positive for every target event; labels correct when sampled
NUMERICAL EXAMPLE: a random audit draws 40,000 of 3,995,200 unreviewed events, q = 0.0100120
                   and finds 19 frauds; T_hat = 19/0.0100120 = 1897.7 events
                   Var = 19*(1-0.0100120)/0.0100120^2 = 187,590 -> se = 433.1
                   approximate 95 per cent interval: 1897.7 +/- 1.96*433.1 = 1049 to 2746
INTERPRETATION: the unreviewed tail holds roughly 1,900 frauds, not "zero because unreviewed"
```

The se here is 433.1 rather than the 435.4 quoted in section 3 because section 3 used the simpler
sqrt(19)/q approximation, 4.359/0.0100120 = 435.4. The difference between the two variance forms is
the (1-q) factor, negligible at q = 0.01, material at q near 0.5. Both are reported because the
simple form is what a spreadsheet will produce and the reader should recognise it.

Combining the audit with reviewed positives gives a recall estimate of 2510/(2510+1897.7) = 0.5695,
against a naive 2510/4800 = 0.5229 obtained by assuming the cohort prevalence counts are exact. The
two differ by 4.7 points, which is larger than the sampling error on the audit itself: the naive
figure is not an estimate of recall at all but a ratio of two quantities with different error
structures.

### 4.3 Variants

**V-1 Cost-per-caught reformulation.** Some programmes cannot state L in currency and prefer to work
in "cost per fraud caught". Algebraically, set the cutoff by requiring that the expected cost of the
reviews needed to catch one fraud be no greater than the value of catching it. With uniform random
selection at precision q, expected reviews per catch is 1/q, so the condition is c/q <= V, which
rearranges to q >= c/V — the same inequality as G-1 with precision replacing probability. It is the
same rule, but it is derived from a rate rather than a per-event expectation, so it holds for a
*population* of events and does not license event-level ranking decisions. Using it as a per-event
rule is the classic category error: it makes every event with the population's average probability
indifferent.

**V-2 Tiered action.** If the action set contains a cheap action (SMS challenge), a medium action
(specialist review) and an expensive action (account freeze), each with its own (c_j, r_j), the
optimal policy chooses the action with the highest Delta_j for each event, which is not the same as
choosing the highest-r action. At p = 0.104583, L = 340: freeze gives 0.104583 x 340 x 0.80 - 2.10 =
28.44658 - 2.10 = 26.34658 cur; review gives 10.06 cur; SMS challenge with r = 0.30 and c = 0.15 gives
10.6675 - 0.15 = 10.5175 cur. Freeze wins on this event despite the friction, but freeze on a
false positive costs the relationship, so h matters: with h = 40 cur for a wrongly frozen
legitimate customer, freeze drops to 0.104583 x 340 x 0.80 - 2.10 - 0.895417 x 40 =
28.44658 - 2.10 - 35.81668 = -9.47010 cur and review wins. The friction term, not the fraud term, decides.

**V-3 Rank-based capacity policy.** When probabilities are unreliable but ordering is good, use the
top-K by score and accept that the cutoff is a quantile of the score distribution rather than a
probability. The policy is still valid under G-2 provided Delta is monotone in the score, and it is
robust to calibration decay, but it cannot be budgeted: the number of reviews is fixed by K even if
the fraud rate collapses, so value per cycle falls with prevalence while cost stays flat. A
programme in this mode should re-derive K monthly, not annually.

**V-4 Two-stage triage.** Insert a 4-minute automated-plus-analyst triage before full review.
Triage costs 4/60 x 41 = 2.733 cur per alert and passes a fraction f to full review at 0.200 h and
9.50 cur. Effective cost per alerted event is 2.733 + f x 9.50 and effective hours are
0.0667 + f x 0.200. At f = 0.4 the per-alert cost is 6.5333 cur, the per-alert hours are 0.14667,
capacity rises to 4,800/0.14667 = 32,727 reviews, and value rises to roughly 338,000 cur against
241,370 for the single-stage policy. The gain comes from the capacity multiplier, not from better
ranking: the same model serves 36 per cent more decisions.

**V-5 Currency-recovery accounting.** Some finance functions report recovered currency rather than
prevented loss, because recovered currency is a cash inflow and prevented loss is not. Converting:
recovered = TP x L x rho where rho is the share of loss actually recoverable, which is bounded by
the chargeback rules and by whether the merchant is still solvent. With rho = 0.35, the 2,510
positives at capacity yield 2,510 x 340 x 0.35 = 298,690 cur of recovery against 469,370 cur of
prevented exposure. The two numbers are both true and must never be added: prevented exposure
includes exposure that would never have been recovered.

**V-6 Sampling-corrected precision.** If alerts were sampled for review rather than reviewed
exhaustively, observed precision must be corrected the same way as G-5, and additionally the
*sample* changes the meaning of capacity. With 60 per cent of alerts reviewed, capacity in alerts
is 24,000 but capacity in reviews is 14,400, and the un-reviewed 40 per cent must be assumed to
have the same precision only if sampling was random with respect to outcome — which triage
sampling never is.

### 4.4 Limits and counterexamples

**Limit 1: prevalence approaching zero.** With fixed positive FPR, PPV goes to zero linearly in pi.
At pi = 0.0001 the running example's rates give PPV = 0.009629, so fewer than one in a hundred
reviews finds fraud and the correct policy is to review nothing. The counterexample to "more data
will fix it": no amount of data changes the ratio, because the ratio is a property of the
population and the rates, not of the sample size.

**Limit 2: cutoff at or above one.** When c + h exceeds L x r, the cutoff exceeds one and no event
should be reviewed. With c = 9.50, r = 0.03 and L = 340 the cutoff is 0.93137; with L = 15 and
r = 0.5 it is 1.26667. A programme that computes a cutoff above one and then clamps it to 0.99 has
converted a finding ("this action is not worth taking at any probability") into a policy that
reviews almost everything.

**Limit 3: empty selection.** At K = 0, precision is 0/0 and undefined, not zero. Reports that
print 0.0 for precision at K = 0 create a spurious improvement when K rises. The correct display is
"undefined" with the selected count alongside.

**Limit 4: all-benefit-negative capacity.** If every Delta is negative, the optimal at-most-K policy
selects nothing. The constraint is an upper bound, not an equality; a queue manager rewarded for
filling capacity will fill it with negative-value reviews, which is how programmes end up with
"100 per cent capacity utilisation" and negative contribution.

**Limit 5: rank-preserving recalibration.** A monotone recalibration preserves top-K membership
when there are no newly created ties, so capacity policies are invariant to it. It changes every
probability-threshold policy and every expected-value calculation. This is the sharpest reason to
separate the two: an organisation using top-K can ignore calibration entirely, and one using a
cutoff cannot.

**Limit 6: heterogeneous loss.** Two events with identical p = 0.06 and different L (340 and 3,400)
have Delta of 11.22 - 9.50 = 1.72 cur and 112.20 - 9.50 = 102.70 cur. Ordering by p treats them as
equals; the optimal ordering puts the second far ahead. The counterexample to "rank by risk" is
simply a portfolio where amount varies, which is every portfolio.

**Limit 7: non-identifiability without audit.** If q = 0 for a region — say, small-value
transactions are never audited — then G-5 is undefined there and no reweighting can recover the
count. The formula's valid range excludes q = 0 by construction, and the practical implication is
that audit design must cover the whole cohort, including the boring middle.

**Limit 8: the two-stage break-even.** Triage helps only if the triage cost plus the screened-in
review cost is below the single-stage review cost per *decision*, accounting for the fact that
triage must be paid on every alert while review is paid only on f of them. Break-even is at
f = 2.733/(9.50 - 0.03 x 187) = 2.733/3.89 = 0.7026: above that pass rate triage costs more than it
saves in labour. It can still be worth doing for latency or consistency reasons, but not for money.

**Takeaway.** Each limit is a precondition of G-1 or G-2 restated as a failure. The three that
cause most real losses are heterogeneous loss (Limit 6), filling capacity with negative-value
reviews (Limit 4), and applying a cutoff computed on a different prevalence (Limit 1).

## 5. Worked Numerical Example
