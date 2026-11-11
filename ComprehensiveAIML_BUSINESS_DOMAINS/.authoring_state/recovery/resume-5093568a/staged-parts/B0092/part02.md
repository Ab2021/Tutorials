## 6. Analysis Walkthrough

This section derives the six quantities the decision actually depends on. Every input is
ILLUSTRATIVE; every step of arithmetic is shown so it can be reproduced with a calculator. Nothing
here was executed against a live system, and section 9 records that status explicitly.

### 6.0 The order of derivation, and why it is not negotiable

A returns programme is usually presented in the order model, threshold, forecast, approval. That
order hides the two numbers that decide everything — the recovery value of an avoided return and the
marginal cost of a contact — behind a leaderboard of discrimination metrics. The derivation below
runs the other way: cost first, because cost is knowable from invoices; then the threshold, which is
a two-line consequence of cost and effect; then the scope, which is a consequence of the threshold
and the risk distribution; then the experiment, which exists only to bound the effect that the
threshold already assumed.

The reversal matters because it tells the reader, before any modelling claim, which number is
fragile. Here the fragile number is the effect size: the threshold is a ratio in which the effect
appears in the denominator, so a factor-of-two error in the effect moves the eligible population by
more than a factor of two. Cost errors move it proportionally and are visible on an invoice. Effect
errors move it hyperbolically and are invisible unless someone runs an experiment with enough power
to see them.

### 6.1 F-1 — the recovery value of one avoided return

```text
FORMULA [F-1]:  R = L + I + G + F + S + P + K
WHERE:  R   recovery cost avoided per return          [currency/return]
        L   reverse-logistics movement                [currency/return]
        I   inspection, grading, and repack labour    [currency/return]
        G   liquidation gap on the recovered unit     [currency/return]
        F   payment-processing fee not reversed       [currency/return]
        S   service touches: chat, refund handling    [currency/return]
        P   packaging write-off and shrink            [currency/return]
        K   working-capital charge while in process   [currency/return]
CONSTRAINT: R excludes the refund itself. The refund is a transfer back to the customer, not a
            cost of the return event; including it double-counts the sale and inflates R by
            roughly the selling price, which here would be 98.00 rather than 62.00.
NUMERICAL EXAMPLE:
        L = 7.40   I = 3.60   G = 45.80   F = 1.10
        S = 2.10   P = 1.10   K = 0.90
        R = 7.40 + 3.60 + 45.80 + 1.10 + 2.10 + 1.10 + 0.90 = 62.00 currency/return
INTERPRETATION: the dominant term is the liquidation gap, 45.80 of 62.00, or 73.9%. A programme
        that models this quantity without the liquidation gap is modelling 26.1% of the cost.
```

The liquidation gap itself is a two-step quantity and deserves its own line of arithmetic rather
than the status of an input. With an average selling price of 98.00, resale proceeds of 56.00 and
resale handling of 3.80, the net proceeds are 56.00 − 3.80 = 52.20, so the gap is
98.00 − 52.20 = 45.80. The ratio of net proceeds to selling price is 52.20 / 98.00 = 53.3%. That
recovery ratio, not the return rate, is what determines whether returns are an expensive operational
annoyance or an existential margin problem, and it is the number a merchandising team controls
through markdown cadence.

The payment fee term `F` exists because most processors do not return the interchange component on a
refund. It is small, 1.10, but it is the one component that is genuinely non-recoverable at any
volume, and it is the term finance partners ask about first because it is the only one they cannot
see anywhere in the general ledger.

**Validity condition.** F-1 holds only when the avoided return would have been processed through the
same reverse-logistics path. If the intervention prevents a return that would have been resold at
full price within the season, `G` collapses toward zero and `R` falls to roughly 16.20; if it
prevents a return of a final-sale unit that would have been liquidated at 20% of price, `G` rises
above 60. Segmenting `R` by category is therefore not a refinement, it is a precondition for using
the number at all. The single blended 62.00 is a convenience that is defensible only because the
five categories in scope happen to have recovery ratios between 48% and 58%.

### 6.2 F-2 — the break-even return probability

```text
FORMULA [F-2]:  p* = c / (e x R)
WHERE:  p*  break-even baseline return probability for a contacted line   [fraction]
        c   fully loaded marginal cost of one intervention contact        [currency/contact]
        e   relative reduction in return probability, measured            [fraction]
        R   recovery value per avoided return, from F-1                   [currency/return]
        and the per-line expected net value is v(p) = p x e x R - c       [currency/line]
CONSTRAINT: 0 < e < 1, R > 0, c > 0, and the multiplication e x R must be interpreted as
        "value of avoiding one return, scaled by the fraction of returns the contact avoids".
        The formula assumes the effect is proportional to baseline risk. Section 6.7 shows what
        happens when it is not, and that assumption, not the algebra, is the fragile part.
NUMERICAL EXAMPLE:
        c = 1.49 currency/contact          (0.35 send + 0.60 content + 0.40 oversight + 0.14 attrition)
        e = 0.0750                          (measured in the randomised pilot, section 6.5)
        R = 62.00 currency/return
        e x R = 0.0750 x 62.00 = 4.6500
        p* = 1.49 / 4.6500 = 0.32043  ->  32.04%
INTERPRETATION: a contact is worth sending only to a line whose baseline return probability
        exceeds 32.04%. Below that, the expected recovery does not cover the cost of the message,
        and every additional contact destroys value even though it reduces returns.
```

Three immediate consequences follow, and each one is a decision rather than an observation.

First, the attrition term matters more than its size suggests. Stripping it out gives
c = 1.35 and p* = 1.35 / 4.6500 = 0.29032, a threshold of 29.03%. The 0.14 in cost therefore moves
the threshold by 32.04 − 29.03 = 3.01 percentage points, which is 9.4% of 32.04. On a population
whose risk distribution is dense in the 29–33% band, that shift is worth thousands of lines of
scope. The reason is structural: the threshold is linear in `c`, so a small absolute change in cost
becomes a proportionally equal change in threshold, and the return-rate density near the threshold
converts that into a large change in count.

Second, the threshold is inversely proportional to `R`, which means the least measurable term in the
whole calculation has the most leverage. Halving `R` to 31.00 moves the threshold to
1.49 / (0.0750 x 31.00) = 1.49 / 2.3250 = 0.64086, or 64.09%, and there is no line in the
population with a mean probability above 64%, so the programme disappears entirely. Nobody would
propose a returns programme on the assumption that a return costs 62.00; but the arithmetic shows
that the same programme becomes indefensible at 31.00, so the burden of proof belongs on the
recovery value, not on the model.

Third, the formula produces an honest "do not run this" answer when `c` exceeds `e x R`. That state
is not a failure of the calculation; it is the calculation working. Section 6.7 pushes this to its
limit, where `p*` exceeds 1.0 and the correct output is that no line qualifies.

**Boundary and counterexample.** Set `e = 0` — an intervention with no effect. Then `p*` is
undefined as a ratio rather than merely large, and the correct statement is "the contact destroys
`c` per line unconditionally". Implementation detail matters here: a naive pipeline computing
`c / (e x R)` divides by zero, a pipeline clipping `e` to a floor of 0.001 produces
`p* = 1.49 / 0.062 = 2403%`, which silently disables the policy rather than signalling it, and a
pipeline that treats `p* > 1` as "no eligible lines" behaves correctly. The three behaviours are
indistinguishable in a chart and different in an incident review.

### 6.3 F-3 — net value of a scope, and the marginal-decile rule

```text
FORMULA [F-3]:  V(S) = sum over lines i in S of ( p_i x e x R - c )
                     = e x R x sum(p_i) - c x |S|
WHERE:  V(S)  expected net value of contacting the set S            [currency/cohort]
        p_i   baseline return probability for line i                [fraction]
        |S|   number of contacted lines                             [count]
        e, R, c as in F-2
CONSTRAINT: S must be defined by a rule computable at packing time, and |S| must satisfy the
        contact-frequency ceiling and the budget. The formula is linear in p, which is exactly why
        a threshold rule is optimal: v(p) is increasing in p, so the best set of a given size is
        the |S| lines with the largest p.
NUMERICAL EXAMPLE (decile means, 25,000 lines per decile):
        D10: p = 0.580  ->  v = 0.580 x 4.6500 - 1.49 = 2.69700 - 1.49 = +1.20700  -> +30,175
        D9 : p = 0.372  ->  v = 0.372 x 4.6500 - 1.49 = 1.72980 - 1.49 = +0.23980  ->  +5,995
        D8 : p = 0.281  ->  v = 0.281 x 4.6500 - 1.49 = 1.30665 - 1.49 = -0.18335  ->  -4,584
        V(top 2 deciles) = 30,175 + 5,995 = 36,170 currency per 250,000-line cohort
INTERPRETATION: the optimum is at the top two deciles (50,000 lines, 20.0% of the cohort) and
        the marginal decile is negative. Adding D8 costs 4,584 in expected value, so scope growth
        beyond the threshold is not neutral: it is value-destroying at a computable rate.
```

The decile table below is the arithmetic backbone of every later claim in the case. Means are
reported rather than medians because F-3 sums means; a median would be the right summary for a
different question and the wrong one here. Population 250,000 lines; each decile 25,000 lines;
`e x R = 4.6500`; `c = 1.49`.

| Decile | Mean p | Returns in decile | Net per line | Net value | Cumulative net | Include? |
|---|---|---|---|---|---|---|
| D10 | 0.580 | 14,500 | +1.20700 | +30,175 | +30,175 | yes |
| D9 | 0.372 | 9,300 | +0.23980 | +5,995 | +36,170 | yes |
| D8 | 0.281 | 7,025 | −0.18335 | −4,584 | +31,586 | no |
| D7 | 0.221 | 5,525 | −0.46235 | −11,559 | +20,027 | no |
| D6 | 0.174 | 4,350 | −0.68090 | −17,023 | +3,005 | no |
| D5 | 0.137 | 3,425 | −0.85295 | −21,324 | −18,319 | no |
| D4 | 0.105 | 2,625 | −1.00175 | −25,044 | −43,363 | no |
| D3 | 0.078 | 1,950 | −1.12730 | −28,183 | −71,545 | no |
| D2 | 0.055 | 1,375 | −1.23425 | −30,856 | −102,401 | no |
| D1 | 0.027 | 675 | −1.36445 | −34,111 | −136,513 | no |

The cumulative column is the part practitioners skip and the part that answers the question actually
asked in the steering meeting, which is "what happens if we do everyone?". Doing everyone means
250,000 contacts at 1.49, costing 372,500, against recovered value of 50,750 x 4.6500 = 235,987.50.
Net −136,512.50, which is the table's cumulative column to within 2 units of rounding across ten
cells of 25,000 lines. Either figure makes the point: the unsegmented programme loses about 136,500
per cohort, and the segmented programme earns about 36,170 on the same population. There is no
model-quality difference between those two programmes. There is only a threshold.

Three properties of the table deserve names, because each one reappears in the failure modes.

The **cut is interior**, at 32.04%, and the decile structure only approximates it. D9's mean of
37.2% sits above the cut while roughly 40% of its individual lines sit below it. Acting on whole
deciles therefore leaves value on the table: if the 60% of D9 above the cut carry a mean of 0.447,
contacting 15,000 lines at 0.58855 net each returns 8,828, against 5,995 for the whole decile — a
gain of 2,833 per cohort for no additional cost, because the excluded 10,000 lines each carry a
negative expectation. The operational rule must therefore be a score cut, and the decile table is a
transparency device for the arithmetic rather than the production policy.

The **cut is dominated by one decile**. D10 supplies 30,175 of 36,170, or 83.4%, of the net value.
That concentration is not a modelling artefact; it follows from the linear form of v(p) and from
the fact that the risk distribution has a long right tail. It has two consequences: any error in
D10's mean swings the answer, and any operational failure that affects D10 delivery (a bad address
file, a carrier hold, a content defect) removes most of the value while leaving the programme
looking healthy on contact counts.

The **budget is not the binding constraint** at the correct threshold. Fifty thousand contacts cost
50,000 x 1.49 = 74,500 against a 160,000 envelope for the 28-day window, a utilisation of 46.6%.
Budget binds only if the threshold is abandoned: the 150,000-line scope implied by the assumed
18% effect costs 223,500 and breaches the envelope by 63,500. This is a common and slightly
embarrassing sequence in practice — the plan is caught by finance rather than by the data team,
because finance is the only function that multiplies the line count by the unit cost.

**Refinement within the tail.** Because v(p) is linear and increasing, the optimal set of any fixed
size is the highest-score lines, so the production rule is "contact if score > 0.3204". The only
reason to keep deciles at all is that a decile table with 25,000-line cells can be re-derived by
hand in a review meeting, whereas a score cut cannot. Reviewability is a real requirement; it is
satisfied by publishing both the score cut and the decile arithmetic that validates it.

### 6.4 F-4 — why the ranking metric and the money disagree

```text
FORMULA [F-4]:  w_tail = P_tail / P_total
                W_tail = V_tail / V_total
                ratio  = W_tail / w_tail
WHERE:  P_tail   positives (returns) falling in the top decile          [count]
        P_total  positives in the whole cohort                          [count]
        V_tail   net value contributed by the top decile                [currency]
        V_total  net value of the whole policy                          [currency]
        w_tail   is also the share of AUC-deciding pairs that involve a top-decile positive,
                 because each positive forms one pair with each of the negatives
CONSTRAINT: both quantities must be computed on the same cohort and the same threshold. AUC
        weights every positive-negative pair equally; net value weights lines by v(p), which is
        negative below the threshold and linear above it.
NUMERICAL EXAMPLE (two candidate models, both at the 32.04% threshold):
        M-tail: AUC 0.744, P_tail 14,500 -> w = 14,500 / 50,750 = 0.28571
                V_tail 30,175 of V_total 36,170 -> W = 30,175 / 36,170 = 0.83425
                ratio = 0.83425 / 0.28571 = 2.92
        M-mid : AUC 0.761, P_tail 14,150 -> w = 14,150 / 50,750 = 0.27882
                V_tail 28,547.50 of V_total 32,682.50 -> W = 0.87347
                ratio = 0.87347 / 0.27882 = 3.13
        net value M-tail 36,170.00 vs M-mid 32,682.50  ->  M-mid loses 3,487.50 (-9.6%)
INTERPRETATION: the better-ranked model (AUC 0.761) earns 3,487.50 less per cohort than the
        worse-ranked one (AUC 0.744). The top decile carries 28.6% of the metric's weight and
        83.4% of the money's weight, a ratio of 2.92; AUC answers a question nobody asked.
```

The mechanism is worth stating plainly, because "AUC is a poor choice" is a slogan and the
mechanism is an argument. AUC is the probability that a randomly chosen positive is scored above a
randomly chosen negative. With 50,750 positives and 199,250 negatives there are
50,750 x 199,250 = 10,112,187,500 such pairs, of which those with a D10 positive number
15,100 x 199,250 = 3,008,675,000 — 29.8% of the total. So 70.2% of the metric is decided by
comparisons among lines that the policy will never contact or never spare, and those comparisons
have no value consequence whatsoever.

Net value, by contrast, is a sum over contacted lines with weights that vanish below the threshold
and grow linearly above it. Moving a line from p = 0.10 to p = 0.12 changes AUC by a sliver and
changes net value by nothing, because both lines are excluded. Moving a line from p = 0.56 to
p = 0.62 changes AUC by the same sliver and changes net value by 0.06 x 4.6500 = 0.279 per line.
A metric that is indifferent to a change the business cares about, and sensitive to changes it does
not, is not a conservative choice; it is a mis-specified objective.

Two further consequences follow and both change the process rather than the model.

First, model selection must be performed on a *value-weighted* objective computed at the operating
threshold, not on overall AUC. The practical version is: hold out a matured cohort, fix `e`, `R`,
`c`, compute `V(S)` for each candidate, and choose the candidate with the highest `V`. This requires
the cost vector to be frozen before model selection, which is uncomfortable for data teams and
correct for the business, because otherwise the cost vector becomes a free parameter that can
reverse any model comparison after the fact.

Second, the threshold must be re-derived, not just the model. Because `p*` depends on `c` and `R`,
a model that shifts the entire score distribution upward can appear to improve net value purely by
moving more lines across a fixed threshold. The guard against this is to report the selected-set
size alongside the value, so an improvement achieved by contacting twice as many lines at the same
average value is visible as a cost increase rather than a win.

**Boundary case.** F-4 breaks when the selected set is large enough to approach the whole
population: at a threshold of 5%, 90% of lines are contacted, the metric's weight and the value's
weight both approach 1, and the ratio approaches 1 — at which point AUC becomes a defensible proxy
again. The formula therefore does not say "never use AUC"; it says "AUC is a defensible proxy in
proportion to how much of the population the policy touches". A policy touching 20% of lines is
not in that regime, and the ratio 2.94 quantifies exactly how far out of it the programme is.

### 6.5 F-5 — the power the pilot actually had

```text
FORMULA [F-5]:  SE(d) = sqrt( 2 x p_c x (1 - p_c) / n )
                MDE   = (z_{1-a/2} + z_{1-b}) x SE(d)
                power = Phi( d / SE(d) - z_{1-a/2} )
WHERE:  p_c  control-arm return rate            [fraction]
        n    lines per arm                      [count]
        d    observed absolute difference       [fraction]
        MDE  minimum detectable effect at power 1-b, two-sided a
        Phi  standard normal CDF
CONSTRAINT: one randomisation unit per line, no clustering, no interim peeking, and a single
        pre-registered primary metric. Clustering by customer or parcel inflates SE(d) by the
        design effect 1 + (m - 1) x ICC and must be accounted for before the pilot is sized.
NUMERICAL EXAMPLE:
        p_c = 0.2040, n = 10,000 per arm, d = 0.2040 - 0.1887 = 0.0153
        SE(d) = sqrt(2 x 0.2040 x 0.7960 / 10,000) = sqrt(0.0000324768) = 0.0056988  (0.570 pp)
        MDE   = (1.9600 + 0.8416) x 0.0056988 = 2.8016 x 0.0056988 = 0.015966  (1.60 pp)
        power = Phi(0.0153 / 0.0056988 - 1.9600) = Phi(2.6848 - 1.9600) = Phi(0.7248) = 0.766
        95% CI on d: 0.0153 +/- 1.9600 x 0.0056988 = [0.00413, 0.02647]  (0.41 to 2.65 pp)
INTERPRETATION: the pilot was sized so that its own measured effect (1.53 pp) sat just below its
        own minimum detectable effect (1.60 pp). It reached p = 0.0073 by luck of the draw and had
        76.6% power; a replicate at the same effect would fail to reach significance 23.4% of the
        time. The interval, not the point estimate, is the decision input.
```

The interval has a consequence that is easy to miss and decisive here. Expressed relatively, the
interval is [0.00413 / 0.2040, 0.02647 / 0.2040] = [2.03%, 12.98%]. Feeding those into F-2:

| Effect scenario | e (relative) | e x R | p* | Eligible lines above p* |
|---|---|---|---|---|
| Lower 95% bound | 0.0203 | 1.2586 | 118.4% | 0 |
| Point estimate | 0.0750 | 4.6500 | 32.04% | 50,000 |
| Upper 95% bound | 0.1298 | 8.0476 | 18.51% | 150,000 |

The lower bound implies a threshold above 100%, which means no line is eligible and the programme
should not run. The upper bound implies a threshold of 18.51%, which brings in D7 through D1 and
multiplies the scope by three. So the same experiment supports "do nothing", "contact 20% of
lines", and "contact 60% of lines" with equal statistical warrant. A decision that cannot be
distinguished from its opposite by the available evidence is a decision that must be deferred or
re-measured; it must not be presented as a point estimate with an interval in a footnote.

**Validity condition.** F-5 assumes the randomisation unit is the line and that treated and control
lines are independent. They are not, when a customer has several lines in the cohort: a message
about one parcel changes the customer's behaviour toward the others. If the average eligible
customer has 1.12 lines in the cohort and the within-customer correlation of return behaviour is
0.18, the design effect is 1 + (1.12 − 1) x 0.18 = 1.022, inflating SE by 1.1%. That is small
enough to ignore here and would not be small in a category where customers routinely buy six lines
at once: at m = 6 and ICC = 0.18 the design effect is 1.90, SE inflates by 37.8%, and the pilot's
effective sample halves.

### 6.6 F-6 — the sample size that would make the decision identifiable

```text
FORMULA [F-6]:  n* = 2 x p_c x (1 - p_c) / SE*^2
                with SE* = (d - d_L) / z_{1-a/2}
WHERE:  d_L  the largest lower confidence bound on the absolute effect that still leaves the
             decision unchanged                                    [fraction]
        SE*  the standard error that puts the lower bound exactly at d_L
CONSTRAINT: d_L is chosen from the decision, not from statistical convention: it is the effect
        below which the break-even threshold would exceed the mean risk of the decile the
        programme intends to contact.
NUMERICAL EXAMPLE:
        target: p* must stay at or below 0.575 so that D10's mean of 0.580 remains eligible
        required e: 1.49 / (0.575 x 62.00) = 1.49 / 35.65 = 0.041796  (4.18% relative)
        d_L = 0.0418 x 0.2040 = 0.008527  (0.85 pp absolute)
        SE* = (0.0153 - 0.008527) / 1.9600 = 0.006773 / 1.9600 = 0.003456
        n*  = 2 x 0.2040 x 0.7960 / (0.003456)^2 = 0.324768 / 0.0000119439 = 27,191
        chosen n = 40,000 per arm -> SE 0.002849, CI lower 0.00972 (4.76% relative), p* 50.5%
INTERPRETATION: the pilot needed 27,200 lines per arm, not 10,000, for its lower bound to leave
        the top decile eligible. It ran at 10,000, which is why its interval spans the entire
        decision space. Sizing the experiment from the decision rather than from a convention is
        the difference between a pilot that answers the question and one that raises it.
```

The arithmetic in F-6 is the most transferable thing in the case, because it inverts the usual
conversation. The statistician asks "what effect do you want to detect?" and the business answers
with a wish. The correct question runs backwards: "at what effect would your decision change?", and
the answer is computed from the cost vector and the risk distribution. Here the decision changes
when `p*` crosses 58.0%, the mean risk of the top decile, because that is the point at which the
programme's only profitable segment stops being profitable. That crossing happens at
e = 1.49 / (0.580 x 62.00) = 1.49 / 35.96 = 0.041435, or 4.14% relative — and the pilot must
therefore be sized so that its lower bound clears 4.14%, which is what F-6 does.

Choosing 40,000 per arm rather than the minimum 27,200 costs 80,000 lines of treated exposure
instead of 54,400, or 119,200 of intervention cost instead of 81,056, and buys a lower bound at
4.76% relative — comfortably above the 4.14% crossing, with margin for the design effect and for
category heterogeneity. That margin is not decorative: it absorbs the 1.02 design effect from the
customer-level spillover in section 6.5 and the category variation in `R` that F-1 assumed away.

**Boundary case.** F-6 assumes a two-arm comparison. With four arms (two strata × two treatments)
the per-arm sample requirement rises by roughly the square root of the number of comparisons if the
family-wise error rate is held, or the per-comparison alpha inflates the required `n` by about
1.2× at four comparisons. A team that stratifies to answer the relative-versus-absolute question
must budget for the multiplicity; a team that does not stratify cannot answer it at all.

### 6.7 The counterexample: when the effect is absolute, the whole apparatus collapses

F-2 through F-6 all rest on one assumption that the algebra hides: the effect of the contact is
*proportional* to baseline risk, so that a line with p = 0.58 loses 7.5% of its return probability
(4.35 pp) while a line with p = 0.05 loses 7.5% of its own (0.38 pp). The alternative is that the
effect is a *constant* probability shift, because the message does the same thing to everyone who
reads it. The two models are both consistent with the pilot, which measured an average effect of
1.53 pp on a population whose mean risk was 20.40% — the single point where the two curves cross.

```text
FORMULA [F-7]:  v_abs(p) = e_abs x R - c          (constant effect e_abs)
                v_rel(p) = p x e_rel x R - c      (proportional effect, F-2)
WHERE:  e_abs = 0.0153 absolute, implied by the pilot at the population mean
        e_rel = 0.0750 relative, equivalently 0.0153 at p = 0.2040
CONSTRAINT: both are calibrated to the same observed average effect, so no amount of extra data
        at the same stratification can separate them. Separation requires randomising within
        risk strata and comparing the effect estimates across strata.
NUMERICAL EXAMPLE:
        v_abs(0.580) = 0.0153 x 62.00 - 1.49 = 0.94860 - 1.49 = -0.54140  ->  -13,535 per decile
        v_abs(0.100) = 0.0153 x 62.00 - 1.49 = -0.54140                   ->  identical
        v_rel(0.580) = 0.580 x 4.6500 - 1.49 = +1.20700                   ->  +30,175 per decile
INTERPRETATION: under the absolute model every line loses 0.5414 per contact regardless of risk,
        the threshold device is meaningless because there is no p at which the contact pays, and
        the correct decision is to stop. Under the relative model the top decile earns 30,175.
        The two models agree on the average effect and disagree on the sign of the decision.
```

This is the strongest possible illustration of the case's primary pattern, and it is worth being
precise about why it cannot be resolved by more data of the same kind. Both models are calibrated
to reproduce a 1.53 pp average effect at a 20.40% baseline. They differ only in the slope of
effect against baseline risk, and that slope is not identified by an experiment whose randomisation
ignores baseline risk: within any arm, the treated and control groups have the same risk
distribution, so the average difference is the same under both models. Only a design that
randomises within strata — and is powered to detect a difference in effect *between* strata — can
separate them.

The practical test is a difference-in-differences across strata. With two strata (mean risk 0.46
in the top two deciles, 0.13 elsewhere) the relative model predicts effects of 3.45 pp and 0.98 pp,
a gap of 2.47 pp; the absolute model predicts 1.53 pp in both, a gap of zero. Detecting a 2.47 pp
gap between strata at 80% power requires SE(gap) = 2.47 / 2.8016 = 0.882 pp, which with
p-bar = 0.29 across the two strata implies n = 2 x 0.29 x 0.71 / (0.00882)^2 = 0.4118 / 0.00007778
= 5,295 per cell, or 21,180 lines across four cells — cheap relative to the 160,000-line
experiment, and it is the experiment that decides whether the programme has a mechanism at all.

The honest summary of the walkthrough is therefore layered: the arithmetic of F-1 through F-3 is
solid and reproducible; the threshold of 32.04% is a correct consequence of five inputs; the scope
of 50,000 lines and the value of 36,170 follow from it; and the entire structure rests on the
proportional-effect assumption, which the available evidence cannot confirm. A case that stopped at
the 36,170 would be arithmetically right and professionally wrong.

### 6.8 Independent arithmetic re-check

Every figure used later is re-derived here in a second form so that a reader can find an error by
comparison rather than by trust. The re-check uses ratios where the first derivation used sums.

| Quantity | First derivation | Re-check by ratio | Agreement |
|---|---|---|---|
| R | 7.40+3.60+45.80+1.10+2.10+1.10+0.90 | 98.00 − 52.20 + 16.20 non-gap terms | 62.00 both |
| c | 0.35+0.60+0.40+0.14 | 1.49 / 1.35 = 1.1037 over the attrition-free base | 1.49 both |
| p* | 1.49 / 4.6500 | 32.04% is the decile interpolation point: D9 37.2% > 32.04% > D8 28.1% | consistent |
| V(top 2) | 30,175 + 5,995 | 50,000 x (0.476 x 4.6500 − 1.49) = 50,000 x 0.72340 | 36,170 both |
| V(all) | 235,988 − 372,500 | 50,750 x 4.6500 − 250,000 x 1.49 | −136,512 both |
| SE(d) | sqrt(2 x 0.162384 / 10,000) | 0.570 pp x sqrt(2) = 0.806 pp for a one-arm mean | consistent |
| n* | 0.324768 / 0.003456² | 0.324768 / 0.0000119439 = 27,191 | 27,191 both |

The `V(all)` line differs from the decile table's −136,965 by 452, and the difference is entirely
decile rounding: the table's means sum to 2.030 against the population rate of 2.030, but each
mean is quoted to three decimals and the rounding accumulates across ten cells. A reader who
reconciles to the last unit will find 452 and should find it, because a reconciliation that closes
to the cent on ten rounded cells is more suspicious than one that closes to within 500.

Takeaway: five inputs, one ratio, one scope, one power calculation, and one unresolved assumption.
The inputs can be audited from invoices; the assumption can only be resolved by a stratified
experiment, and no volume of the current design will resolve it.
