## 9. Complexity & Cost Analysis

The cost section answers two questions that are frequently conflated: how much does it cost to
*compute* the decision, and how much does it cost to *make* it. The first is trivial in this domain
and the second is where the money is. Any proposal that spends effort on model infrastructure
without costing the review is optimising the wrong term.

### 9.1 Compute cost of scoring

Scoring 4,000,000 events at 0.35 ms each takes 4,000,000 x 0.35/1000 = 1,400 seconds = 23.3 minutes
of single-core time, or 0.39 core-hours. At 0.85 cur per core-hour that is 0.33 cur. Feature
materialisation is the real compute term: at 12 ms per event the batch takes 4,000,000 x 12/1000/3600
= 13.33 core-hours, or 11.33 cur. Storage at about 400 bytes per row for the decision record is
4,000,000 x 400 = 1.6 GB per cycle, 19.2 GB per year — a rounding error against the review budget.

```text
FORMULA: batch compute cost = N_scored * ms_per_event / 3,600,000 * price_per_core_hour
WHERE: N_scored events per cycle; ms_per_event inclusive of feature lookup; price cur/core-hour
CONSTRAINT: batch mode; linear scaling; no cross-event dependencies (fraud rings break this)
NUMERICAL EXAMPLE: 4,000,000 * 12 / 3,600,000 = 13.33 core-hours * 0.85 = 11.33 cur
                   scoring only: 4,000,000 * 0.35 / 3,600,000 = 0.389 * 0.85 = 0.33 cur
INTERPRETATION: total compute is 11.66 cur against 228,000 cur of review cost, a ratio of 1:19,557
```

That ratio is the strategic fact of this section. Compute is 0.00005 of the programme cost; a
30 per cent infrastructure saving is worth 3.5 cur per cycle, while a 1 per cent change in review
precision is worth about 2,400 cur. Any prioritisation that does not start from the review cost is
mis-aimed by four orders of magnitude.

The one place latency genuinely binds is streaming: 4,000,000 events per cycle over 30 days is 1.54
events per second average and about 4.63 at a 3x peak, so a p99 budget of 500 ms is 8,000 times the
inter-arrival time at peak. Latency becomes a constraint only if the programme scores synchronously
at authorisation on a per-event path with hard 100 ms limits, and even then the binding term is
feature lookup, not model evaluation.

### 9.2 Cost of the decision engine

Converting scores to a capacity-feasible selection is a sort. Sorting 48,000 alert candidates costs
about 48,000 x log2(48,000) = 48,000 x 15.55 = 746,000 comparisons — under a millisecond. Sorting
all 4,000,000 candidates to find the top 24,000 is 4,000,000 x 21.93 = 87.7 million comparisons,
still under a second in any reasonable implementation; a bounded heap of size 24,000 does
4,000,000 x log2(24,000) = 58.2 million. None of this matters. What matters is that a *tie* in the
boundary group makes the answer depend on the sort's stability, which is a correctness issue rather
than a cost one (failure mode F-05).

The integer programme in G-3 has no polynomial guarantee in general, but with equal durations the
LP relaxation is integral and the greedy sort solves it exactly; with two or three duration classes
the same argument applies within each class, giving an O(n log n) exact solution per class. The
practical ceiling is around 10^5 candidates before the sort itself needs to be pushed into the
scoring pipeline; beyond that, maintain a top-K heap per class during scoring rather than sorting
afterwards.

### 9.3 Cost of review — the dominant term

| Component | Unit cost | Quantity per cycle | Cycle cost | Share |
|---|---|---|---|---|
| Investigator labour | 8.20 cur/review | 24,000 reviews | 196,800 cur | 85.8 per cent |
| Customer friction | 1.30 cur/review | 24,000 reviews | 31,200 cur | 13.6 per cent |
| Feature compute | 11.33 cur/cycle | 1 | 11.33 cur | 0.005 per cent |
| Scoring compute | 0.33 cur/cycle | 1 | 0.33 cur | 0.0001 per cent |
| Calibration sample refresh | 3.20 cur/label | 250,000 labels per year | 800,000 cur/yr | see note |
| Audit stream | 0.60 min/event at 41 cur/h | 40,000 per cycle | 16,400 cur | 6.7 per cent |

The calibration row is deliberately annual and deliberately large: 250,000 fully adjudicated labels
at 3.20 cur each is 800,000 cur per year, which is 29 per cent of the annual programme cost of
2,752,800 cur. Most programmes do not pay this as a line item because they reuse review outcomes as
labels, which is precisely the selection drift of F-06. A programme that wants an unbiased
calibration curve must pay for labels outside the review queue, and the price is not small.

The audit row is optional until recall must be reported; once it is, 16,400 cur per cycle buys an
estimated total-fraud count with a standard error of about 4.3 per cent of the point estimate.

```text
FORMULA: total cycle cost = K*c + compute + audit
WHERE: K reviews; c = labour + friction; compute and audit in cur per cycle
CONSTRAINT: c excludes the cost of decisions that are wrong beyond friction (churn, complaints)
NUMERICAL EXAMPLE: 24,000*9.50 + 11.66 + 16,400 = 228,000 + 16,411.66 = 244,411.66 cur
                   value 241,370 cur -> net contribution -3,041.66 cur before labour overheads
INTERPRETATION: with a mandatory audit, the programme is roughly at break-even, not profitable
```

This is the most important honest number in the sheet. Without the audit the programme contributes
241,370 − 228,011.66 = 13,358.34 cur per cycle, a 5.9 per cent margin on cost. With a 40,000-event
audit at 16,400 cur the contribution turns slightly negative. The audit is still worth buying,
because without it the programme cannot know its recall and therefore cannot know whether the
241,370 cur is real — but the arithmetic shows that the audit must be sized deliberately. Halving
the audit to 20,000 events halves its cost to 8,200 cur and roughly multiplies the standard error
on unreviewed fraud by sqrt(2), from 431 to 610; the interval on recall widens from about 23 points
to 32. There is no free version of this measurement.

### 9.4 Cost per caught fraud and its shape

Cost per caught fraud is 24,000 x 9.50 / 2,510 = 90.84 cur. It is not a constant: it is the ratio of
two functions of K, and it rises steeply as precision falls. Per block, cost per caught is
9.50/0.155 = 61.29 cur in block 1, 90.84 cur at the capacity point, 168.14 cur in block 7, 250.00
cur in block 9 and 527.78 cur in block 12. The curve is hyperbolic in precision, so small changes
in the tail produce violent changes in the average: extending from 24,000 to 48,000 reviews raises
cost per caught from 90.84 to 135.55 cur (456,000/3,364) while lowering total value by 68,302 cur.

There is a second, subtler non-linearity. Because value is linear in TP but cost is linear in K,
value per cycle is maximised where marginal precision crosses c/V = 0.050802, and the function is
*flat* near that point: at 26,667 reviews (inside block 7) value is 2,660.7 x 187.00 − 26,667 x 9.50
= 497,551 − 253,337 = 244,214 cur, only 2,842 cur above the value at 24,000 (241,370). The flatness
is good news operationally — being 10 per cent off the optimal K costs about 1 per cent of value —
and bad news analytically, because a programme cannot distinguish a good cutoff from a marginally
worse one without very precise inputs.

### 9.5 Scale behaviour

Four scaling behaviours matter, and only two are favourable.

**Linear in N, with a constant ratio.** Doubling the cohort doubles compute, doubles audit cost,
and doubles review cost, and doubles value if prevalence is unchanged. Nothing improves; the
programme scales as a business, not as a technology.

**Sub-linear in K because of the heap.** Sorting is n log n but a bounded heap is n log K, and
log K grows very slowly. Ten times more capacity costs 1.4 times more selection work. Selection
never becomes the constraint.

**Super-linear in audit precision.** The standard error of the unreviewed-fraud estimate is
sqrt(T)/q, so halving the interval requires quadrupling the audit sample: 4x cost for 2x precision.
This is the only genuinely convex cost in the system and the reason recall is rarely reported.

**Discontinuous at the tie boundary.** Moving K by one event across a tie group of 1,400 can change
precision by 1,400/24,000 = 5.8 per cent of capacity and by more in value terms. Costs are smooth
almost everywhere and then jump.

### 9.6 Where the marginal dollar goes

Ranked by value per cur of additional spend over one cycle:

1. 1,000 extra reviews at block-7 precision: 1,000 x 1.0655 = 1,066 cur of value per 9,500 cur
   spent on an extra 200 investigator-hours — that is a *negative* return as priced at 41 cur/h
   (200 h x 41 = 8,200 cur cost for 1,066 cur value, net −7,134). Do not buy.
2. A triage stage at f = 0.4: 96,766 cur of additional value for 213,816 cur of restructured spend
   — the spend is not additional, it is substitution, which is why the return is large (it buys
   40 per cent more decisions for 6 per cent less money).
3. A 40,000-event audit: −16,400 cur of measurable contribution, but it prevents the programme from
   reporting a recall it cannot support. Buy it as measurement, not as value.
4. Better loss estimates (L from matured chargebacks): a 300-cur versus 340-cur error moves the
   optimum by 4,000 reviews and 4,262 cur per cycle. This is an analysis cost, near zero, with the
   highest return of anything on the list.

The ordering is the practical content of this section: the highest-return investment in fraud
decisioning is usually better cost and loss inputs, not more compute, more capacity, or a better
model.

## 10. Key Theorems / Results Reference

Each result is stated once, with its conditions, so that a reader can check whether their situation
satisfies them before using the formula downstream. Results are numbered R-1 onward and are the
authority for the derivations in section 4.

**R-1 Bayes' rule in rate form.** For any partition of a population by outcome and any selection
rule, precision = pi x TPR / (pi x TPR + (1-pi) x FPR). *Conditions:* the TPR and FPR are the rates
of the same selection rule on the target population. *Consequence:* precision is not transported
across populations by TPR and FPR alone unless those rates transport. *Reference:* this is the
standard form of the positive-predictive-value identity, derived in 4.1 B-2.

**R-2 Total unimodularity of the single-constraint 0-1 programme.** The programme
max c'x s.t. 1'x <= K, x in {0,1}^n has an integral LP optimum, and greedy selection by coefficient
is optimal. *Conditions:* one row of ones; coefficients known. *Consequence:* capacity allocation
with equal consumption is solved by sorting, with no solver and no optimality gap. *Failure:* unequal
consumption destroys the property (G-3).

**R-3 Optimal stopping/cutoff equivalence.** The per-event rule "act iff Delta > 0" and the
population rule "act iff precision at the margin >= c/V" coincide at the solution of the
unconstrained problem, and the second is the correct *aggregate* diagnostic while the first is the
correct *per-event* rule. *Conditions:* Delta additive and separable. *Consequence:* the cutoff can
be monitored as a rate on a dashboard without reference to individual events (G-1).

**R-4 Prior-shift invariance of the likelihood ratio.** Under pure label-prior shift, the likelihood
ratio associated with a score is unchanged; posterior odds scale exactly with prior odds.
*Conditions:* class-conditional feature distributions unchanged; calibrated source posterior.
*Consequence:* one multiplicative correction moves a model between prevalences (G-4).
*Counterexample:* a feature whose meaning changes with prevalence — a merchant category that is
fraudulent only at low volume — breaks the condition and the correction overshoots.

**R-5 Horvitz-Thompson unbiasedness.** For a design with known inclusion probabilities q_i > 0, the
estimator sum R_i Y_i / q_i is unbiased for the population total, and its variance is
sum Y_i^2 (1-q_i)/q_i. *Conditions:* correct q_i, correct labels, positivity. *Consequence:* the
unreviewed tail can be counted (G-5). *Failure:* a ratio estimator such as recall is only
approximately unbiased, with bias of order 1/n.

**R-6 Decomposition of proper scoring rules.** The Brier score decomposes into reliability,
resolution, and uncertainty. *Consequence:* a model can be improved by increasing resolution
without improving reliability, and vice versa, so a single Brier number cannot direct work. In the
example, a Brier of 0.008378 against an uncertainty term of pi(1-pi) = 0.00119856... check:
0.0012 x 0.9988 = 0.00119856 — so most of the Brier score is uncertainty, not error, and the
reliability-plus-resolution part is 0.008378 − 0.001199 = 0.007179. *Conditions:* binary outcome,
probability forecasts.

**R-7 Hanley-McNeil variance of the AUC.** For an AUC estimated from n1 positives and n2 negatives,
Var = [A(1−A) + (n1−1)(Q1−A^2) + (n2−1)(Q2−A^2)]/(n1 n2), with Q1 = A/(2−A) and Q2 = 2A^2/(1+A).
*Consequence:* with n1 = 4,800 and n2 = 3,995,200, Q1 = 0.9369, Q2 = 0.9514 and se = 0.001797, so
AUC 0.96742 has a 95 per cent interval 0.96390 to 0.97094. *Condition:* the negative class dominates
so heavily that the AUC is essentially a mean of per-positive ranks, which makes the interval
narrower than intuition suggests. The practical warning: a ±0.003 interval says nothing about
whether the *operating point* is good, because AUC is dominated by the easy middle of the score
range.

**R-8 Binormal ROC identity.** If scores are normal within class with equal variance, the ROC is
parametrised by a single separation a, and AUC = Phi(a/sqrt2). *Consequence:* a single operating
point pins the whole curve, which makes it possible to ask "what would precision be at a tenth of
the capacity" from one measurement. In the example a = 2.6079 and the implied AUC is 0.96742, which
matches the empirical block-based AUC closely enough to use the model for extrapolation.
*Condition:* equal-variance normality, which fraud scores rarely satisfy in the extreme tail.

**R-9 Flatness of value near the optimum.** Because value is TP(K) x V − K x c with TP concave in K,
the value function is flat to second order at its maximum: a displacement of delta in K costs
approximately (1/2) x |TP''| x delta^2. *Consequence:* operating 10 per cent off K* costs about 1
per cent of value, which is why cutoff debates are usually worth less than the meeting that
schedules them — and why a 20 per cent error in the *level* of L is worth much more than a 10 per
cent error in K.

**R-10 Non-identifiability of recall without audit.** Recall cannot be estimated from reviewed data
alone when selection depends on the score, because the events that would define the denominator are
unobserved. *Consequence:* any programme reporting recall from review outcomes is reporting a
score-conditional quantity as if it were unconditional. *Fix:* R-5 plus a design with q > 0 on the
unreviewed region.

**R-11 Decomposition of net contribution.** Net = (prevented exposure) − (labour) − (friction) −
(compute) − (measurement). *Consequence:* the four cost terms are of wildly different magnitudes
(85.8, 13.6, 0.005, 6.7 per cent in the example), so arguments that trade measurement against
labour are arguments about 6.7 per cent against 85.8 per cent and should be settled on that basis.

**R-12 Invariance of top-K under monotone transformation.** Any strictly increasing transform of the
score leaves the top-K set unchanged (up to newly created ties). *Consequence:* a capacity policy is
immune to calibration error, and a threshold policy is not. *Corollary:* the two policies need
different monitoring — the capacity policy needs prevalence monitoring, the threshold policy needs
calibration monitoring.

## 11. Interview / Exam Questions

Fifteen questions, mixing recall, calculation, and diagnosis. Answers are self-contained; each can
be checked against the section that derives it.

**Q1.** A model has TPR 0.52 and FPR 0.0054 at its operating point, and prevalence is 0.0012. What
precision should you expect, and why is it not 0.52?
*Answer.* PPV = 0.0012 x 0.52/(0.0012 x 0.52 + 0.9988 x 0.0054) = 0.000624/(0.000624 + 0.005394) =
0.000624/0.006018 = 0.10369, about 10.4 per cent. It is not 0.52 because TPR conditions on fraud
while precision conditions on selection; at low prevalence the negative mass in the denominator
dominates by a factor of about 8.6 here.
*Reasoning.* The identity is R-1. The mistake being tested is confusing P(select|fraud) with
P(fraud|select).
*Follow-up.* What prevalence would make precision equal 0.52 at these rates?
*Red flag.* Quoting recall as precision, or answering "it depends on the model" without the ratio.
*Scoring.* Full credit for the numeric answer and the direction of the conditioning; half credit for
the right formula with arithmetic slip.

**Q2.** Your cutoff is 0.5 because that is the classifier's default. Loss given fraud is 340 cur,
preventable share 0.55, review cost 9.50 cur. What is the correct cutoff and how much money does the
default lose?
*Answer.* 9.50/(340 x 0.55) = 9.50/187.00 = 0.050802. At a 0.5 cutoff the model selects only events
above 50 per cent, which in this example is no event at all (the top block's precision is 0.155), so
the policy reviews nothing and forgoes the whole 241,370 cur of contribution at capacity.
*Reasoning.* G-1 with the correct numerator and denominator; then checking the model's reachable
probability range, which is the step everyone skips.
*Follow-up.* If the model's maximum output is 0.20, is the cutoff still achievable, and what do you
do?
*Red flag.* Using 0.5 "because it is standard", or setting the cutoff at c/L without r.
*Scoring.* Correct number plus the observation that the default forecloses the decision entirely.

**Q3.** Precision at 24,000 reviews is 0.104583. What is the marginal value of the 24,001st review,
and would you fund a 25th investigator?
*Answer.* The 24,001st review sits at block-6 precision 0.06700, so its value is 0.067 x 187.00 −
9.50 = 12.529 − 9.50 = +3.029 cur. Yes, fund it: a further 4,000 reviews at block-6/7 precision
yield 268 x 187.00 − 38,000 = 12,116 cur for 800 extra hours, which is 15.15 cur per investigator-
hour against a loaded cost of 41 cur/h — wait, that comparison is wrong because the investigator
cost is already inside c. The correct statement: 4,000 extra reviews cost 38,000 cur and return
50,116 cur of prevented exposure, net 12,116 cur, for 800 hours; the loaded payroll for those
hours is 32,800 cur, which is *already counted* in c, so the net 12,116 cur is after paying for
them. Fund it.
*Reasoning.* Careful separation of the gross benefit (precision x V), the review cost (which
includes the labour), and the hiring cost (which is the labour). Double counting labour is the most
common error in this calculation.
*Follow-up.* What precision makes an additional review exactly break even, and where does that fall
on the block table?
*Red flag.* Comparing 12,116 cur against 32,800 cur and concluding the hire is unprofitable — that
counts labour twice.
*Scoring.* Full credit for +3.029 cur marginal and the correct no-double-count reasoning.

**Q4.** Why can a model with AUC 0.967 be unable to reach its own economic cutoff anywhere in the
score range?
*Answer.* Because AUC measures ordering over the whole curve, including the easy middle. What
matters at the cutoff is the maximum likelihood ratio the model can produce. In the example the
maximum likelihood ratio is 18.76, giving a maximum posterior of 0.022046 at prevalence 0.0012,
while the cutoff requires 0.050802 — a likelihood ratio of 44.55. The model literally cannot say
"more likely fraud than not" for any event, so no threshold reaches the line.
*Reasoning.* Convert the cutoff to required odds (q/(1−q) = 0.05354), divide by prior odds
(0.0012014) to get required likelihood ratio 44.55, and compare against the model's maximum.
*Follow-up.* What are the three remedies, and which is cheapest?
*Red flag.* "Lower the threshold" — the threshold is already at the bottom of the range.
*Scoring.* Full credit for the odds conversion and the comparison; bonus for naming a cheaper
action (higher r, lower c) as a remedy alongside a better model.

**Q5.** Your audit samples 40,000 unreviewed events and finds 19 frauds. Estimate total fraud and
recall, with an interval.
*Answer.* q = 40,000/3,976,000 = 0.0100604; unreviewed estimate 19/0.0100604 = 1,888.6; se =
sqrt(19 x 0.98994)/0.0100604 = 4.337/0.0100604 = 431.1; interval 1,043 to 2,734. Total fraud
2,510 + 1,889 = 4,399 and recall 0.5706, interval 0.479 to 0.706.
*Reasoning.* R-5 with the correct denominator (unreviewed, not the whole cohort) and a variance that
includes the (1−q) factor.
*Follow-up.* How much would the interval narrow if the audit were stratified by score decile?
*Red flag.* Dividing by the cohort size, or reporting recall as 2,510/4,800 without noting that the
4,800 is only known because the cohort prevalence was itself measured.
*Scoring.* Full credit for both point estimates and any correctly constructed interval.

**Q6.** Precision fell from 0.105 to 0.075 over three weeks. Diagnose in order, with the test that
separates each hypothesis.
*Answer.* First check maturity: 0.075/0.105 = 0.714, very close to a 72 per cent maturity fraction,
so recompute on matured cohorts only; if the series flattens, it was the artefact. Second check
capacity: if K rose, precision must fall by construction — recompute at fixed K. Third check
prevalence: recompute the rate-form prediction from current pi and the old TPR/FPR. Fourth check
the score distribution: a KS test between the current and previous score distributions detects a
retrain or pipeline change. Only after all four is adversarial drift the surviving explanation.
*Reasoning.* Ordered from cheapest and most likely to most expensive, with a discriminating test for
each.
*Follow-up.* Which of the four would also explain a *rise* in recall? (Answer: the maturity
artefact, and only that.)
*Red flag.* Jumping to "the fraudsters adapted" without checking the cohort clock.
*Scoring.* Full credit for the ordered tests, especially maturity first.

**Q7.** Two alerts both have p = 0.06. One is a 40-cur digital purchase, the other a 3,400-cur
electronics order. Which do you review first, and by how much do the values differ?
*Answer.* Delta = 0.06 x L x 0.55 − 9.50. For L = 340: 11.22 − 9.50 = 1.72 cur. For L = 3,400:
112.20 − 9.50 = 102.70 cur. The second is worth 59.7 times the first, so ranking by probability is
wrong by a factor of 60 on this pair.
*Reasoning.* G-1 generalised to heterogeneous L; the whole point is that Delta is not monotone in p
when L varies.
*Follow-up.* What is the value of sorting by expected value instead of probability across a whole
portfolio, and how would you measure it?
*Red flag.* "Same probability, so same priority."
*Scoring.* Full credit for both numbers and the ratio.

**Q8.** Show that greedy selection is optimal for the equal-duration capacity problem, and give a
case where it is not.
*Answer.* The programme max sum Delta_i x_i s.t. sum x_i <= K has a single constraint matrix of ones,
which is totally unimodular, so the LP optimum is integral; the LP optimum is obtained by taking the
K largest coefficients, so greedy is optimal. With unequal durations d_i the matrix is no longer TU:
two items with Delta 70/0.6 h = 116.67 cur/h and Delta 100/1.0 h = 100.00 cur/h, budget 1.0 h, greedy
takes the first for 70 cur while the optimum takes the second for 100 cur.
*Reasoning.* R-2 plus the standard fractional-knapsack counterexample.
*Follow-up.* What is the worst-case ratio of greedy to optimal for 0-1 knapsack, and does it matter
at n = 24,000?
*Red flag.* Claiming greedy is optimal in general.
*Scoring.* Full credit for TU and the counterexample; the worst-case ratio (1/2 asymptotically) is
bonus.

**Q9.** Your team is at 100 per cent utilisation and value is falling month over month. What is the
first thing you check?
*Answer.* Whether K exceeds K*. Compute value at K and at 0.75K using observed block precisions. In
the example, extending from 24,000 to 48,000 reviews takes value from 241,370 to 173,068 cur, a loss
of 68,302, while utilisation stays at 100 per cent. If value rises when K falls, the queue is
over-filling and the fix is a cutoff, not more staff.
*Reasoning.* Utilisation is a throughput metric, not a value metric; F-03.
*Follow-up.* What would you put on the dashboard instead of utilisation?
*Red flag.* Hiring more investigators to clear a queue that should not be worked.
*Scoring.* Full credit for the value-at-K test and the diagnosis.

**Q10.** Explain what happens to the cutoff if the preventable share r is overestimated, with
numbers.
*Answer.* Cutoff = c/(L r), so overestimating r lowers the cutoff and admits more marginal alerts.
At r = 0.55 the cutoff is 0.050802; at an assumed 0.70 it would be 9.50/238.00 = 0.039916, which
admits block 9 (precision 0.0380) and destroys value: extending to 36,000 reviews gives
3,076 x 187 − 36,000 x 9.50 = 575,212 − 342,000 = 233,212 cur, down 8,158 cur from capacity. The
error compounds because the higher r also inflates V, so the damage is both a lower cutoff and a
higher apparent value per catch.
*Reasoning.* Two-channel effect of one parameter; the second channel is what makes the error hard to
see.
*Follow-up.* How would you measure r without a randomised holdout?
*Red flag.* Treating r as a modelling choice rather than an operational measurement.
*Scoring.* Full credit for both channels and the value comparison.

**Q11.** Is it correct to say a model is "well calibrated" because ECE is 0.000035?
*Answer.* No. ECE is a weighted average over bins and hides local error; a model can pass it and be
40 per cent wrong in the one bin containing the cutoff. Check the operating-bin ratio, the
reliability slope (0.98474 here, fine), and the intercept (0.57212 per 1,000, slightly high), and
recompute precision at the cutoff from the bin, not from the average. Also check that the bins were
not fitted on the same data: a chi-square of 0.3372 on 7 df is suspiciously low.
*Reasoning.* Section 5.3 and Table 7.4.
*Follow-up.* Why is a chi-square that is too *low* a warning?
*Red flag.* Accepting a single aggregate calibration number as proof.
*Scoring.* Full credit for naming local/operating-bin error; bonus for the too-good chi-square.

**Q12.** You must choose between 4,000 extra reviews at precision 0.0565 and a triage stage with
f = 0.4. Which creates more value, and what does each cost?
*Answer.* Extra reviews: 4,000 x (0.0565 x 187 − 9.50) = 4,000 x 1.0655 = 4,262 cur, costing 38,000.
Triage at f = 0.4: capacity rises from 24,000 to 32,727, value 338,133 cur against 241,370, a gain
of 96,763 cur while total spend *falls* from 228,000 to 213,815 because triage replaces review time.
Triage is better by a factor of 22 on value and is cheaper.
*Reasoning.* V-4 versus the marginal block of G-2.
*Follow-up.* What pass rate makes triage pointless?
*Red flag.* Comparing the two on value without comparing their costs, or ignoring that triage
substitutes rather than adds spend.
*Scoring.* Full credit for both numbers and the substitution insight.

**Q13.** A colleague reports "we prevented 469,370 cur of fraud". What three questions do you ask?
*Answer.* (1) Is that prevented exposure or recovered cash? Recovered at rho = 0.35 is 298,690 cur,
and the two must never be summed. (2) Was it counterfactual — would the loss have occurred without
the action? Blocked transactions that would have been declined anyway contribute nothing. (3) What
is the net? Subtracting 228,000 cur of review cost gives 241,370 cur. Without the third number the
first is a gross figure that any programme can inflate by reviewing more.
*Reasoning.* D-16, V-5, F-07.
*Follow-up.* How do you measure the counterfactual portion cheaply?
*Red flag.* Accepting the gross number, or adding recovery to prevention.
*Scoring.* Full credit for all three questions.

**Q14.** What is the value of the audit stream in currency terms, given it makes the programme look
break-even?
*Answer.* The audit costs 16,400 cur per cycle and moves reported contribution from +13,358 to
−3,042 before overheads. Its value is not contribution but decision quality: without it recall is
unmeasurable (R-10), selection drift is undetectable (F-06), and the model cannot be retrained on
the unreviewed region. Quantitatively, it prevents the programme from claiming 2,510/4,800 = 0.523
recall when the truth is 0.571 — a 4.7-point overstatement that would misprice every capacity
expansion.
*Reasoning.* R-11 decomposition and the identifiability argument.
*Follow-up.* Could a smaller audit do the same job, and at what cost in interval width?
*Red flag.* Cutting the audit to improve the reported number.
*Scoring.* Full credit for separating measurement value from contribution.

**Q15.** Give the one-line rule for choosing between a probability cutoff, a top-K policy, and a
two-stage triage.
*Answer.* If the capacity constraint does not bind at the economic cutoff, use the probability
cutoff c/(L r); it maximises value and is self-correcting when prevalence moves. If capacity binds
below the economic cutoff, use top-K by expected value and spend the effort on expanding capacity.
If the team has no slack and capacity binds hard, insert triage and re-derive K from the effective
AHT — but only if the pass rate stays below 0.70.
*Reasoning.* Table 7.1, rows 1, 3, and 5, plus the break-even in 4.4 Limit 8.
*Follow-up.* How does the answer change if L varies by a factor of ten across the portfolio?
*Red flag.* Choosing by familiarity or by vendor default.
*Scoring.* Full credit for the three-way decision rule with its binding condition.

## 12. Quick Reference Card

| Item | Value | Condition | Section |
|---|---|---|---|
| Economic cutoff | (c+h)/(L*r) = 9.50/187.00 = 0.050802 | constant L, r; no capacity limit | 4.2 G-1 |
| Precision from rates | pi·TPR/(pi·TPR+(1-pi)·FPR) = 0.104583 | rates measured on the target population | 4.1 B-2 |
| Per-event value | p·L·r − c − h = 10.057 cur at the operating point | additive, separable harms | 4.1 B-3 |
| Capacity allocation | sort by expected value, take top K | equal consumption per action | 4.2 G-2 |
| Value at capacity | TP(K)·V − K·c = 241,370 cur at K = 24,000 | K below economic optimum | 5.1 |
| Prior-shift correction | odds_t = odds_s x [pi_t/(1-pi_t)]/[pi_s/(1-pi_s)] | pure label-prior shift only | 4.2 G-4 |
| Unreviewed total | audit positives / q = 1,888.6 with se 431.1 | q > 0 everywhere; labels final | 5.5 |
| Break-even investigator | 6,560/(V x precision at margin) ≈ 621 reviews | marginal block fully absorbed | 5.2 |
| Triage break-even | f < 0.7026 | triage 2.7333 cur, review 9.50 cur | 4.4 Limit 8 |
| Deferral rule | drop the lowest expected value, never the oldest | any surge | 7.5 |
